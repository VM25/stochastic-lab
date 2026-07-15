#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BlackScholesAnalyticEngine";

/// Wall-clock timer for the valuation itself.
class ScopedTimer {
public:
    [[nodiscard]] double elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

}  // namespace

BlackScholesTerms BlackScholesAnalyticEngine::terms(const MarketState& market,
                                                    const EuropeanOption& option,
                                                    const BlackScholesModel& model) noexcept {
    BlackScholesTerms t;

    const double maturity = option.maturity();
    const double strike = option.strike();

    t.total_volatility = model.total_volatility(maturity);
    t.forward = market.forward(maturity);
    t.discount_factor = market.discount_factor(maturity);
    t.dividend_discount_factor = market.dividend_discount_factor(maturity);

    // log(F/K) rather than log(F) - log(K): the difference of two logs loses
    // precision when F and K are close, which is exactly the near-the-money
    // region where the option has the most gamma.
    t.log_moneyness = std::log(t.forward / strike);

    t.degenerate = (t.total_volatility <= 0.0);

    if (t.degenerate) {
        // d1 and d2 diverge here. They are left at zero rather than filled with
        // an infinity that could leak into a formula or a JSON artifact; the
        // `degenerate` flag is the field callers must branch on.
        t.d1 = 0.0;
        t.d2 = 0.0;
        return t;
    }

    // Equivalent to the specification's
    //   d1 = [log(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),
    // rewritten around the forward. The two agree exactly in real arithmetic;
    // this arrangement evaluates log(F/K) once and avoids re-deriving the carry
    // term, and it is the form the degenerate branch also keys on.
    t.d1 = (t.log_moneyness + 0.5 * t.total_volatility * t.total_volatility) / t.total_volatility;
    t.d2 = t.d1 - t.total_volatility;

    return t;
}

Result<PricingResult> BlackScholesAnalyticEngine::price(const MarketState& market,
                                                        const EuropeanOption& option,
                                                        const BlackScholesModel& model) {
    const ScopedTimer timer;

    const BlackScholesTerms t = terms(market, option, model);

    const double spot = market.spot();
    const double strike = option.strike();
    const double discounted_spot = spot * t.dividend_discount_factor;
    const double discounted_strike = strike * t.discount_factor;

    PricingResult result;
    result.method = "black_scholes_analytic";

    if (t.degenerate) {
        // sigma*sqrt(T) = 0: the asset evolves deterministically to its forward,
        // so the option is worth the discounted intrinsic value of that forward.
        // This is the exact limit, not an approximation.
        switch (option.type()) {
            case OptionType::Call:
                result.value = std::max(discounted_spot - discounted_strike, 0.0);
                break;
            case OptionType::Put:
                result.value = std::max(discounted_strike - discounted_spot, 0.0);
                break;
        }

        result.add_warning(fmt::format(
            "degenerate limit: sigma*sqrt(T) = 0 (sigma={}, T={}); price is the discounted "
            "intrinsic value of the forward and carries no time value",
            model.volatility(),
            option.maturity()));
    } else {
        const double n_d1 = norm_cdf(t.d1);
        const double n_d2 = norm_cdf(t.d2);

        switch (option.type()) {
            case OptionType::Call:
                result.value = discounted_spot * n_d1 - discounted_strike * n_d2;
                break;
            case OptionType::Put:
                // N(-x) = 1 - N(x) exactly in real arithmetic, but norm_cdf(-d)
                // is computed directly through erfc and keeps relative accuracy
                // in the tail, where 1 - N(d) would cancel.
                result.value =
                    discounted_strike * norm_cdf(-t.d2) - discounted_spot * norm_cdf(-t.d1);
                break;
        }

        result.add_diagnostic("d1", t.d1);
        result.add_diagnostic("d2", t.d2);
    }

    // The closed form cannot return a negative value in exact arithmetic. A
    // negative here would mean catastrophic cancellation, so it is reported
    // rather than clamped to zero: clamping would hide the very defect the
    // check exists to find.
    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("price is not finite (spot={}, strike={}, maturity={}, volatility={})",
                        spot,
                        strike,
                        option.maturity(),
                        model.volatility()),
            kContext);
    }
    if (result.value < 0.0) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("price is negative ({}), which the closed form cannot produce; this "
                        "indicates catastrophic cancellation (spot={}, strike={}, maturity={}, "
                        "volatility={})",
                        result.value,
                        spot,
                        strike,
                        option.maturity(),
                        model.volatility()),
            kContext);
    }

    result.add_diagnostic("forward", t.forward);
    result.add_diagnostic("total_volatility", t.total_volatility);
    result.add_diagnostic("log_moneyness", t.log_moneyness);
    result.add_diagnostic("discount_factor", t.discount_factor);
    result.add_diagnostic("dividend_discount_factor", t.dividend_discount_factor);
    result.add_diagnostic("degenerate", t.degenerate);

    result.runtime_seconds = timer.elapsed_seconds();
    return Result<PricingResult>::success(std::move(result));
}

Result<Greeks> BlackScholesAnalyticEngine::greeks(const MarketState& market,
                                                  const EuropeanOption& option,
                                                  const BlackScholesModel& model) {
    const BlackScholesTerms t = terms(market, option, model);

    const double spot = market.spot();
    const double strike = option.strike();
    const double maturity = option.maturity();
    const double rate = market.rate();
    const double dividend_yield = market.dividend_yield();

    const double discounted_spot = spot * t.dividend_discount_factor;
    const double discounted_strike = strike * t.discount_factor;

    Greeks g;

    if (t.degenerate) {
        if (t.log_moneyness == 0.0) {
            // F = K with no diffusion: the payoff has a kink exactly at the
            // forward. Delta jumps between 0 and exp(-qT), and gamma is a Dirac
            // mass. No finite number is the answer, so none is returned.
            return Result<Greeks>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("Greeks are undefined at the payoff kink: sigma*sqrt(T) = 0 and the "
                            "forward equals the strike (F=K={}). Delta is discontinuous and gamma "
                            "is unbounded there.",
                            strike),
                kContext);
        }

        // Away from the kink the limits are the general formulas with
        // N(d1), N(d2) -> 0 or 1 and phi(d1) -> 0. They are written out rather
        // than obtained by substitution because gamma and theta's time-decay term
        // both evaluate to 0/0 in that substitution.
        const bool call_in_the_money = (t.log_moneyness > 0.0);

        g.gamma = 0.0;
        g.vega = 0.0;

        switch (option.type()) {
            case OptionType::Call:
                g.delta = call_in_the_money ? t.dividend_discount_factor : 0.0;
                g.theta = call_in_the_money
                              ? (dividend_yield * discounted_spot - rate * discounted_strike)
                              : 0.0;
                g.rho = call_in_the_money ? (strike * maturity * t.discount_factor) : 0.0;
                break;
            case OptionType::Put:
                g.delta = call_in_the_money ? 0.0 : -t.dividend_discount_factor;
                g.theta = call_in_the_money
                              ? 0.0
                              : (rate * discounted_strike - dividend_yield * discounted_spot);
                g.rho = call_in_the_money ? 0.0 : -(strike * maturity * t.discount_factor);
                break;
        }

        return Result<Greeks>::success(g);
    }

    const double pdf_d1 = norm_pdf(t.d1);

    // Gamma and vega do not depend on the option type: a call and a put with the
    // same terms differ by a forward, which is linear in spot and independent of
    // volatility.
    g.gamma = t.dividend_discount_factor * pdf_d1 / (spot * t.total_volatility);
    g.vega = discounted_spot * pdf_d1 * std::sqrt(maturity);

    // Shared time-decay term of theta.
    const double decay =
        -discounted_spot * pdf_d1 * model.volatility() / (2.0 * std::sqrt(maturity));

    switch (option.type()) {
        case OptionType::Call: {
            const double n_d1 = norm_cdf(t.d1);
            const double n_d2 = norm_cdf(t.d2);
            g.delta = t.dividend_discount_factor * n_d1;
            g.theta =
                decay - rate * discounted_strike * n_d2 + dividend_yield * discounted_spot * n_d1;
            g.rho = strike * maturity * t.discount_factor * n_d2;
            break;
        }
        case OptionType::Put: {
            const double n_minus_d1 = norm_cdf(-t.d1);
            const double n_minus_d2 = norm_cdf(-t.d2);
            // The specification writes put delta as e^{-qT}[N(d1) - 1]. That form
            // is evaluated here as -e^{-qT} N(-d1), which is algebraically
            // identical but numerically different where it matters: for a deep
            // out-of-the-money put d1 is large and positive, so N(d1) -> 1 and
            // N(d1) - 1 cancels to a value dominated by rounding, while N(-d1)
            // is computed straight from erfc and stays accurate.
            g.delta = -t.dividend_discount_factor * n_minus_d1;
            g.theta = decay + rate * discounted_strike * n_minus_d2 -
                      dividend_yield * discounted_spot * n_minus_d1;
            g.rho = -strike * maturity * t.discount_factor * n_minus_d2;
            break;
        }
    }

    if (!std::isfinite(g.delta) || !std::isfinite(g.gamma) || !std::isfinite(g.vega) ||
        !std::isfinite(g.theta) || !std::isfinite(g.rho)) {
        return Result<Greeks>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("a Greek is not finite (spot={}, strike={}, maturity={}, volatility={})",
                        spot,
                        strike,
                        maturity,
                        model.volatility()),
            kContext);
    }

    return Result<Greeks>::success(g);
}

}  // namespace diffusionworks
