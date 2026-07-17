#include <diffusionworks/engines/greeks_monte_carlo.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "GreeksMonteCarloEngine";

}  // namespace

std::string_view to_string(GreekName greek) noexcept {
    switch (greek) {
        case GreekName::Delta:
            return "delta";
        case GreekName::Gamma:
            return "gamma";
        case GreekName::Vega:
            return "vega";
    }
    return "unknown";
}

std::optional<GreekName> parse_greek_name(std::string_view text) noexcept {
    if (text == "delta") {
        return GreekName::Delta;
    }
    if (text == "gamma") {
        return GreekName::Gamma;
    }
    if (text == "vega") {
        return GreekName::Vega;
    }
    return std::nullopt;
}

std::string_view to_string(GreekMethod method) noexcept {
    switch (method) {
        case GreekMethod::FiniteDifference:
            return "finite_difference";
        case GreekMethod::Pathwise:
            return "pathwise";
        case GreekMethod::LikelihoodRatio:
            return "likelihood_ratio";
    }
    return "unknown";
}

std::optional<GreekMethod> parse_greek_method(std::string_view text) noexcept {
    if (text == "finite_difference") {
        return GreekMethod::FiniteDifference;
    }
    if (text == "pathwise") {
        return GreekMethod::Pathwise;
    }
    if (text == "likelihood_ratio") {
        return GreekMethod::LikelihoodRatio;
    }
    return std::nullopt;
}

namespace {

/// The terminal spot on one path, given a standard normal draw.
///
/// The exact log-normal transition: no discretisation, so the only errors an
/// estimator built on this carries are its own bias (the bump) and sampling noise,
/// not a path error that would confound the comparison.
[[nodiscard]] double
terminal_spot(double spot, double drift_rate, double volatility, double maturity, double normal) {
    const double variance = volatility * volatility;
    return spot * std::exp((drift_rate - 0.5 * variance) * maturity +
                           volatility * std::sqrt(maturity) * normal);
}

/// The payoff's derivative with respect to the terminal spot: 1 above the strike for
/// a call, -1 below it for a put, and (conventionally) 0 at the kink. The pathwise
/// estimator's core. Its jump at the strike is why the *direct* second pathwise
/// derivative carries a distributional term there, so this engine does not offer
/// pathwise gamma -- a limitation of the naive method, not of the pathwise idea.
[[nodiscard]] double payoff_slope(OptionType type, double terminal, double strike) {
    if (type == OptionType::Call) {
        return terminal > strike ? 1.0 : 0.0;
    }
    return terminal < strike ? -1.0 : 0.0;
}

}  // namespace

Result<GreekEstimate> GreeksMonteCarloEngine::estimate(const MarketState& market,
                                                       const EuropeanOption& option,
                                                       const BlackScholesModel& model,
                                                       GreekName greek,
                                                       GreekMethod method,
                                                       const GreeksMonteCarloConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    // --- Reject the combinations this engine does not implement ------------
    if (method == GreekMethod::Pathwise && greek == GreekName::Gamma) {
        return Result<GreekEstimate>::failure(
            ErrorCode::UnsupportedCombination,
            "the direct second pathwise derivative is not supported for gamma: for a vanilla call "
            "it carries a distributional term at the strike (the payoff slope jumps there), which "
            "this implementation does not represent. This is a limitation of the naive pathwise "
            "method, not a statement that pathwise-style gamma is impossible -- a mixed "
            "pathwise-likelihood-ratio estimator, Malliavin weights, or payoff smoothing recover "
            "it, and none is implemented here. Use finite-difference or likelihood-ratio gamma.",
            kContext);
    }
    if (method == GreekMethod::LikelihoodRatio && greek != GreekName::Delta) {
        return Result<GreekEstimate>::failure(
            ErrorCode::NotImplemented,
            fmt::format("only the likelihood-ratio *delta* score is implemented; {} by the "
                        "likelihood-ratio method needs a second-order or volatility score this "
                        "engine does not yet carry, and returning a wrong number would be worse "
                        "than declining",
                        to_string(greek)),
            kContext);
    }

    if (config.paths < 2) {
        return Result<GreekEstimate>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a Monte Carlo Greek needs at least 2 paths to have a standard error, got "
                        "{}",
                        config.paths),
            kContext);
    }

    const double spot = market.spot();
    const double strike = option.strike();
    const double maturity = option.maturity();
    const double volatility = model.volatility();
    const double drift_rate = market.rate() - market.dividend_yield();
    const double discount = market.discount_factor(maturity);

    // --- Degenerate inputs, decided per method -----------------------------
    //
    // A zero volatility or maturity is fatal to some estimators and merely
    // degenerate for others, so the request is not rejected wholesale. The
    // likelihood-ratio score divides by sigma*sqrt(T), and the pathwise estimator's
    // derivatives assume a non-degenerate diffusion -- both are undefined here. The
    // finite-difference estimators are not: with no diffusion every path is the
    // forward, so the estimate is a deterministic difference of the price, which is
    // meaningful away from a payoff kink. It is returned with a warning rather than
    // refused, so that a caller asking for finite-difference delta at sigma = 0 is
    // not denied it because likelihood-ratio delta would have failed.
    const bool degenerate = maturity <= 0.0 || volatility <= 0.0;
    std::vector<std::string> warnings;
    if (degenerate) {
        if (method == GreekMethod::LikelihoodRatio) {
            return Result<GreekEstimate>::failure(
                ErrorCode::UnsupportedCombination,
                fmt::format("the likelihood-ratio score is undefined at this boundary (sigma = {}, "
                            "T = {}): it divides by sigma*sqrt(T). The analytic engine gives the "
                            "degenerate limit; the finite-difference method remains usable here.",
                            volatility,
                            maturity),
                kContext);
        }
        if (method == GreekMethod::Pathwise) {
            return Result<GreekEstimate>::failure(
                ErrorCode::UnsupportedCombination,
                fmt::format("the pathwise estimator assumes a non-degenerate diffusion, which is "
                            "absent at this boundary (sigma = {}, T = {}). The analytic engine "
                            "gives the degenerate limit; the finite-difference method remains "
                            "usable here.",
                            volatility,
                            maturity),
                kContext);
        }
        // Finite-difference: allowed, but say what it is.
        warnings.emplace_back(fmt::format(
            "at this boundary (sigma = {}, T = {}) there is no diffusion, so every path is the "
            "forward and this finite-difference estimate is deterministic (standard error zero) "
            "rather than sampled. It is a difference of the price and is meaningful away from the "
            "payoff kink; the analytic engine gives the exact degenerate limit.",
            volatility,
            maturity));
    }

    const double spot_bump = config.spot_bump_fraction * spot;
    const double volatility_bump = config.volatility_bump;
    if ((method == GreekMethod::FiniteDifference) &&
        (greek == GreekName::Delta || greek == GreekName::Gamma) && !(spot_bump > 0.0)) {
        return Result<GreekEstimate>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the spot bump fraction must be positive for finite-difference {}, got {}",
                        to_string(greek),
                        config.spot_bump_fraction),
            kContext);
    }
    if (method == GreekMethod::FiniteDifference && greek == GreekName::Vega) {
        if (!(volatility_bump > 0.0)) {
            return Result<GreekEstimate>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("the volatility bump must be positive for finite-difference vega, got "
                            "{}",
                            volatility_bump),
                kContext);
        }
        if (volatility - volatility_bump < 0.0) {
            // A central difference whose downward bump crosses into negative
            // volatility is not a small error -- it is wrong. S_T's law depends on
            // volatility only through sigma^2 and sigma*Z, and Z is symmetric, so a
            // negative sigma gives the *same* terminal distribution as its positive
            // mirror. The down-bumped price would then nearly equal the up-bumped
            // one and the estimated vega would collapse toward zero. Refused rather
            // than returned.
            return Result<GreekEstimate>::failure(
                ErrorCode::InvalidArgument,
                fmt::format(
                    "the volatility bump {} exceeds the volatility {}, so the central "
                    "difference's downward bump would cross into negative volatility, where "
                    "the terminal law mirrors the positive side and the estimate collapses. "
                    "Use a bump below the volatility, or a one-sided difference.",
                    volatility_bump,
                    volatility),
                kContext);
        }
    }

    const double sqrt_maturity = std::sqrt(maturity);

    // The per-path estimator contribution, given one standard normal draw. Every
    // method reads the *same* draw, which is what makes the finite-difference
    // estimators common-random-number estimators and what lets all three be compared
    // on one sample.
    const auto contribution = [&](double normal) -> double {
        switch (method) {
            case GreekMethod::FiniteDifference: {
                switch (greek) {
                    case GreekName::Delta: {
                        // Central difference in spot, sharing the draw so the two
                        // re-prices differ only through the bump.
                        const double growth =
                            terminal_spot(1.0, drift_rate, volatility, maturity, normal);
                        const double up = option.payoff((spot + spot_bump) * growth);
                        const double down = option.payoff((spot - spot_bump) * growth);
                        return discount * (up - down) / (2.0 * spot_bump);
                    }
                    case GreekName::Gamma: {
                        const double growth =
                            terminal_spot(1.0, drift_rate, volatility, maturity, normal);
                        const double up = option.payoff((spot + spot_bump) * growth);
                        const double mid = option.payoff(spot * growth);
                        const double down = option.payoff((spot - spot_bump) * growth);
                        return discount * (up - 2.0 * mid + down) / (spot_bump * spot_bump);
                    }
                    case GreekName::Vega: {
                        // Bumping volatility moves both the drift correction and the
                        // diffusion, so the whole terminal is recomputed at each
                        // bumped sigma -- against the same draw.
                        const double up = option.payoff(terminal_spot(
                            spot, drift_rate, volatility + volatility_bump, maturity, normal));
                        const double down = option.payoff(terminal_spot(
                            spot, drift_rate, volatility - volatility_bump, maturity, normal));
                        return discount * (up - down) / (2.0 * volatility_bump);
                    }
                }
                return 0.0;
            }
            case GreekMethod::Pathwise: {
                const double terminal =
                    terminal_spot(spot, drift_rate, volatility, maturity, normal);
                const double slope = payoff_slope(option.type(), terminal, strike);
                switch (greek) {
                    case GreekName::Delta:
                        // dS_T/dS0 = S_T/S0.
                        return discount * slope * terminal / spot;
                    case GreekName::Vega:
                        // dS_T/dsigma = S_T (sqrt(T) Z - sigma T).
                        return discount * slope * terminal *
                               (sqrt_maturity * normal - volatility * maturity);
                    case GreekName::Gamma:
                        return 0.0;  // unreachable: rejected above.
                }
                return 0.0;
            }
            case GreekMethod::LikelihoodRatio: {
                // Delta only (guarded above). The score of the log-normal density in
                // S0 is Z / (S0 sigma sqrt(T)); the payoff is left untouched, which
                // is why the method survives a discontinuous payoff.
                const double terminal =
                    terminal_spot(spot, drift_rate, volatility, maturity, normal);
                const double score = normal / (spot * volatility * sqrt_maturity);
                return discount * option.payoff(terminal) * score;
            }
        }
        return 0.0;
    };

    OnlineMoments samples;
    for (std::int64_t i = 0; i < config.paths; ++i) {
        RandomStream stream(config.seed, StreamPurpose::AssetShock, static_cast<std::uint64_t>(i));
        samples.add(contribution(stream.next_normal()));
    }

    const auto standard_error = samples.standard_error();
    if (!standard_error.ok()) {
        return Result<GreekEstimate>::failure(standard_error.error());
    }

    GreekEstimate estimate;
    estimate.greek = greek;
    estimate.method = method;
    estimate.value = samples.mean();
    estimate.standard_error = standard_error.value();
    estimate.paths = config.paths;
    if (method == GreekMethod::FiniteDifference) {
        estimate.bump = greek == GreekName::Vega ? volatility_bump : spot_bump;
    }
    estimate.warnings = std::move(warnings);
    estimate.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<GreekEstimate>::success(std::move(estimate));
}

}  // namespace diffusionworks
