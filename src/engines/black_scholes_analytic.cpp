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
            // The zero-diffusion payoff kink: sigma*sqrt(T) = 0 with F = K.
            //
            // Existence is decided per Greek, not all at once. The value
            // function V(S) = (S e^{-qT} - K e^{-rT})^+ has a corner in spot
            // here, but it is perfectly smooth in some of the other variables,
            // so refusing every Greek would discard real numbers while
            // returning a finite gamma would invent one.
            //
            //   delta  undefined. One-sided derivatives are 0 and e^{-qT}; they
            //          disagree, so no derivative exists. (The symmetric limit
            //          0.5 e^{-qT} is a subgradient midpoint, not a derivative,
            //          and is not reported as one.)
            //   gamma  undefined. The second derivative is a Dirac mass at the
            //          corner: unbounded, never finite.
            //   vega   defined. Along F = K the price is
            //              V(sigma) = D [2 N(sigma sqrt(T)/2) - 1],
            //          whose derivative at sigma = 0 is D sqrt(T) phi(0). Since
            //          sigma < 0 is outside the domain, that one-sided
            //          derivative is the derivative. It is also exactly the
            //          general vega formula evaluated at d1 = 0, and it
            //          correctly gives 0 when T = 0.
            //   theta  defined only when sigma = 0 and r = q. Then F = K forces
            //          S = K and V vanishes identically in T, so theta = 0.
            //          Otherwise undefined: with r != q the one-sided limits in
            //          T disagree, and with sigma > 0, T = 0 the at-the-money
            //          price behaves like sigma sqrt(T), whose T-derivative is
            //          unbounded at expiry.
            //   rho    defined only at T = 0, where V does not depend on r at
            //          all, so rho = 0. For T > 0 the one-sided limits are 0 and
            //          K T e^{-rT}, which differ.
            g.vega = discounted_spot * norm_pdf(0.0) * std::sqrt(maturity);

            g.mark_undefined(
                "delta",
                fmt::format("jump discontinuity at the zero-diffusion payoff kink (F = K = {}): "
                            "the one-sided derivatives are 0 and {}, which disagree",
                            strike,
                            t.dividend_discount_factor));
            g.mark_undefined(
                "gamma",
                fmt::format("unbounded at the zero-diffusion payoff kink (F = K = {}): the second "
                            "derivative is a Dirac mass, so no finite value exists",
                            strike));

            if (model.volatility() == 0.0 && rate == dividend_yield) {
                // F = K with r = q forces S = K, and the value is then
                // identically zero for every maturity.
                g.theta = 0.0;
            } else if (model.volatility() == 0.0) {
                g.mark_undefined(
                    "theta",
                    fmt::format("one-sided limits in maturity disagree at the zero-diffusion "
                                "payoff kink because r ({}) != q ({})",
                                rate,
                                dividend_yield));
            } else {
                g.mark_undefined("theta",
                                 "unbounded at expiry: the at-the-money price behaves like "
                                 "sigma*sqrt(T), whose maturity derivative diverges as T -> 0");
            }

            if (maturity == 0.0) {
                // At expiry the payoff does not involve the discount factor.
                g.rho = 0.0;
            } else {
                g.mark_undefined(
                    "rho",
                    fmt::format("jump discontinuity at the zero-diffusion payoff kink: the "
                                "one-sided derivatives in the rate are 0 and {}, which disagree",
                                strike * maturity * t.discount_factor));
            }

            return Result<Greeks>::success(std::move(g));
        }

        // Away from the kink the limits are the general formulas with
        // N(d1), N(d2) -> 0 or 1 and phi(d1) -> 0. They are written out rather
        // than obtained by substitution because gamma and theta's time-decay term
        // both evaluate to 0/0 in that substitution. All five exist: the value
        // function is locally affine in spot, and flat in volatility to
        // exponentially small order.
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

        return Result<Greeks>::success(std::move(g));
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

    // Away from the degenerate branch every Greek is defined, so a non-finite
    // value here is a numerical failure rather than a point where the derivative
    // does not exist. The two are kept apart deliberately: the degenerate branch
    // reports "no value exists", this reports "the computation broke".
    if (!std::isfinite(*g.delta) || !std::isfinite(*g.gamma) || !std::isfinite(*g.vega) ||
        !std::isfinite(*g.theta) || !std::isfinite(*g.rho)) {
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
