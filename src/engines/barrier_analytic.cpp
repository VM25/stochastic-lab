#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BarrierAnalyticEngine";

/// The vanilla value, from the Phase 1 engine.
///
/// Reused rather than reimplemented. That engine is validated against Hull, Haug,
/// a 50-digit mpmath oracle, and QuantLib, so every barrier price inherits that
/// chain -- and a second inline Black-Scholes here would be a second thing to keep
/// right.
Result<double> vanilla_value(const MarketState& market,
                             const BlackScholesModel& model,
                             OptionType type,
                             double strike,
                             double maturity) {
    const auto option = EuropeanOption::create(type, strike, maturity);
    if (!option.ok()) {
        return Result<double>::failure(option.error());
    }
    const auto priced = BlackScholesAnalyticEngine::price(market, option.value(), model);
    if (!priced.ok()) {
        return Result<double>::failure(priced.error());
    }
    return Result<double>::success(priced.value().value);
}

}  // namespace

Result<PricingResult> BarrierAnalyticEngine::price(const MarketState& market,
                                                   const BarrierOption& option,
                                                   const BlackScholesModel& model) {
    const auto start = std::chrono::steady_clock::now();

    if (option.convention() != MonitoringConvention::Continuous) {
        // Refused rather than approximated. A discretely monitored barrier is a
        // different contract worth a different amount -- more, for a knock-out,
        // because the observation dates miss excursions -- and the gap closes only
        // as O(1/sqrt(m)). Returning this value for it would be a plausible answer
        // to a question that was not asked.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("this engine prices continuous monitoring, but the option specifies {} "
                        "monitoring. Those are different contracts: discrete observation misses "
                        "excursions between fixes, so a knock-out is worth more than its "
                        "continuous value, and the difference decays only as 1/sqrt(m). Use the "
                        "Monte Carlo engine.",
                        to_string(option.convention())),
            kContext);
    }
    if (option.type() != OptionType::Call) {
        return Result<PricingResult>::failure(
            ErrorCode::NotImplemented,
            "only barrier calls are implemented; the put formulae are a separate set of cases",
            kContext);
    }

    const double spot = market.spot();
    const double strike = option.strike();
    const double barrier = option.barrier();
    const double maturity = option.maturity();
    const double rate = market.rate();
    const double dividend = market.dividend_yield();
    const double sigma = model.volatility();

    PricingResult result;
    result.method = "barrier_analytic_reiner_rubinstein";
    result.add_diagnostic("monitoring", std::string(to_string(option.convention())));
    result.add_diagnostic("barrier_type", std::string(to_string(option.barrier_type())));
    result.add_diagnostic("barrier", barrier);

    // --- Already breached ---------------------------------------------------
    //
    // A price, not an error. The contract has already resolved: a knock-out is
    // dead and a knock-in has become the vanilla.
    if (option.breaches(spot)) {
        if (is_knock_out(option.barrier_type())) {
            result.value = 0.0;
            result.add_warning(fmt::format(
                "the barrier at {} is already breached by the spot at {}, so this knock-out is "
                "worthless; the price is exact rather than approximate",
                barrier,
                spot));
        } else {
            const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
            if (!vanilla.ok()) {
                return Result<PricingResult>::failure(vanilla.error());
            }
            result.value = vanilla.value();
            result.add_warning(fmt::format(
                "the barrier at {} is already breached by the spot at {}, so this knock-in has "
                "become the vanilla; the price is exact rather than approximate",
                barrier,
                spot));
        }
        result.add_diagnostic("already_breached", true);
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

    // --- Degenerate limits --------------------------------------------------
    if (maturity <= 0.0 || sigma <= 0.0) {
        // With no time or no diffusion the path cannot move, so an unbreached
        // barrier is never breached: a knock-out is the vanilla and a knock-in is
        // worthless. The Phase 1 engine handles both degenerate cases of the
        // vanilla itself.
        const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
        if (!vanilla.ok()) {
            return Result<PricingResult>::failure(vanilla.error());
        }
        result.value = is_knock_out(option.barrier_type()) ? vanilla.value() : 0.0;
        result.add_warning(fmt::format(
            "degenerate limit (maturity={}, sigma={}): the path cannot reach the barrier, so the "
            "knock-{} is worth {}",
            maturity,
            sigma,
            is_knock_out(option.barrier_type()) ? "out" : "in",
            is_knock_out(option.barrier_type()) ? "the vanilla" : "zero"));
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

    // --- Reiner-Rubinstein --------------------------------------------------
    //
    // For a down-and-out call with B <= K:
    //
    //   a  = (r - q - sigma^2/2)/sigma^2 + 1
    //   x1 = ln(S/K)/(sigma sqrt(T)) + a sigma sqrt(T)
    //   y1 = ln(B^2/(S K))/(sigma sqrt(T)) + a sigma sqrt(T)
    //
    //   C_do = S e^{-qT} N(x1) - K e^{-rT} N(x1 - sigma sqrt T)
    //          - S e^{-qT} (B/S)^{2a} N(y1)
    //          + K e^{-rT} (B/S)^{2a-2} N(y1 - sigma sqrt T)
    //
    // The second pair is the reflected vanilla: the reflection principle maps
    // paths that touch B onto paths started at B^2/S, and the weight (B/S)^{2a}
    // is the Radon-Nikodym factor between them.
    //
    // Verified against an independent bridge-corrected simulation, which agrees to
    // within 1.2 standard errors at three monitoring frequencies -- a route that
    // shares no algebra with this formula.
    const double variance_rate = sigma * sigma;
    const double sqrt_maturity = std::sqrt(maturity);
    const double sigma_root_t = sigma * sqrt_maturity;
    const double a = (rate - dividend - 0.5 * variance_rate) / variance_rate + 1.0;

    const double x1 = std::log(spot / strike) / sigma_root_t + a * sigma_root_t;
    const double y1 =
        std::log(barrier * barrier / (spot * strike)) / sigma_root_t + a * sigma_root_t;

    const double asset_discount = std::exp(-dividend * maturity);
    const double cash_discount = market.discount_factor(maturity);
    const double ratio = barrier / spot;

    if (barrier > strike) {
        // The B > K case needs the second pair of terms (x2, y2), where the
        // barrier rather than the strike sets the boundary of the payoff region.
        // Refused rather than silently priced with the wrong branch.
        return Result<PricingResult>::failure(
            ErrorCode::NotImplemented,
            fmt::format("this engine implements the B <= K branch of Reiner-Rubinstein, but the "
                        "barrier ({}) exceeds the strike ({}). The B > K case requires the second "
                        "pair of terms; pricing it with this branch would return a plausible "
                        "wrong number.",
                        barrier,
                        strike),
            kContext);
    }
    if (option.barrier_type() != BarrierType::DownAndOut &&
        option.barrier_type() != BarrierType::DownAndIn) {
        return Result<PricingResult>::failure(
            ErrorCode::NotImplemented,
            fmt::format("only down-barrier calls are implemented; {} needs the up-barrier "
                        "formulae",
                        to_string(option.barrier_type())),
            kContext);
    }

    const double knock_out =
        spot * asset_discount * norm_cdf(x1) -
        strike * cash_discount * norm_cdf(x1 - sigma_root_t) -
        spot * asset_discount * std::pow(ratio, 2.0 * a) * norm_cdf(y1) +
        strike * cash_discount * std::pow(ratio, 2.0 * a - 2.0) * norm_cdf(y1 - sigma_root_t);

    if (option.barrier_type() == BarrierType::DownAndOut) {
        result.value = knock_out;
    } else {
        // In-out parity: knock-in = vanilla - knock-out, exactly, when the
        // contracts and conventions match. Computed rather than given its own
        // formula, so the identity cannot be violated by a transcription error in
        // a second closed form.
        const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
        if (!vanilla.ok()) {
            return Result<PricingResult>::failure(vanilla.error());
        }
        result.value = vanilla.value() - knock_out;
        result.add_diagnostic("vanilla", vanilla.value());
        result.add_diagnostic("knock_out", knock_out);
    }

    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the barrier price is not finite (S={}, K={}, B={}, T={}, sigma={})",
                        spot,
                        strike,
                        barrier,
                        maturity,
                        sigma),
            kContext);
    }
    if (result.value < 0.0) {
        // The reflected term is subtracted, so cancellation can drive the result
        // slightly negative near the barrier where the two nearly balance.
        // Reported rather than clamped: a negative barrier price is impossible,
        // and clamping would hide exactly the near-barrier fragility Phase 7
        // exists to document.
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the barrier price is negative ({}), which the contract cannot be. The "
                        "reflected term nearly cancels the vanilla near the barrier (S={}, B={}, "
                        "S/B={}), and this is that cancellation.",
                        result.value,
                        spot,
                        barrier,
                        spot / barrier),
            kContext);
    }

    result.add_diagnostic("x1", x1);
    result.add_diagnostic("y1", y1);
    result.add_diagnostic("reflection_exponent", 2.0 * a);
    result.add_diagnostic("spot_to_barrier_ratio", spot / barrier);

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
