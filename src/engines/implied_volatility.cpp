#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/implied_volatility.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "ImpliedVolatility";

/// 2*pi. Written out rather than reached for through M_PI, which is a POSIX
/// extension and is not guaranteed under -std=c++20 with extensions off.
constexpr double kTwoPi = 6.283185307179586476925286766559;

/// The Black-Scholes price at a given volatility, using the project's reference
/// engine so the inversion matches exactly what the forward price would produce.
[[nodiscard]] double
price_at(const MarketState& market, const EuropeanOption& option, double volatility) {
    const auto model = BlackScholesModel::create(volatility);
    // The volatilities the solver evaluates are always strictly positive and finite,
    // so create() cannot fail here; if it ever did, a non-finite price would surface
    // the problem downstream rather than be masked.
    const auto priced = BlackScholesAnalyticEngine::price(market, option, model.value());
    return priced.value().value;
}

/// Vega, the derivative of the price in volatility, at a given volatility. Always
/// non-negative: the Black-Scholes price is increasing in volatility.
[[nodiscard]] double
vega_at(const MarketState& market, const EuropeanOption& option, double volatility) {
    const auto model = BlackScholesModel::create(volatility);
    const BlackScholesTerms t = BlackScholesAnalyticEngine::terms(market, option, model.value());
    const double sqrt_t = std::sqrt(option.maturity());
    // vega = S e^{-qT} phi(d1) sqrt(T) = forward * discount * phi(d1) * sqrt(T).
    return t.forward * t.discount_factor * norm_pdf(t.d1) * sqrt_t;
}

}  // namespace

Result<ImpliedVolatilityResult> ImpliedVolatility::solve(const MarketState& market,
                                                         const EuropeanOption& option,
                                                         double target_price,
                                                         const ImpliedVolatilityConfig& config) {
    if (option.maturity() <= 0.0) {
        return Result<ImpliedVolatilityResult>::failure(
            ErrorCode::UnsupportedCombination,
            "implied volatility is undefined at zero maturity: there is no time for volatility to "
            "act, so no volatility reproduces a price above intrinsic",
            kContext);
    }
    if (!std::isfinite(target_price)) {
        return Result<ImpliedVolatilityResult>::failure(
            ErrorCode::InvalidArgument, "the target price is not finite", kContext);
    }
    if (config.lower_volatility <= 0.0 || config.upper_volatility <= config.lower_volatility ||
        config.max_volatility < config.upper_volatility || config.max_iterations < 1) {
        return Result<ImpliedVolatilityResult>::failure(
            ErrorCode::InvalidArgument,
            "the implied-volatility bracket is ill-formed: need 0 < lower < upper <= max and at "
            "least one iteration",
            kContext);
    }

    // The no-arbitrage window. Below the lower bound or above the upper bound, no
    // volatility reproduces the price -- the quote itself is arbitrageable.
    const double maturity = option.maturity();
    const double discount = market.discount_factor(maturity);
    const double spot_pv = market.spot() * market.dividend_discount_factor(maturity);
    const double strike_pv = option.strike() * discount;

    double lower_bound = 0.0;
    double upper_bound = 0.0;
    if (option.type() == OptionType::Call) {
        lower_bound = std::max(spot_pv - strike_pv, 0.0);
        upper_bound = spot_pv;
    } else {
        lower_bound = std::max(strike_pv - spot_pv, 0.0);
        upper_bound = strike_pv;
    }

    // A small absolute slack so a target exactly on a bound is admitted rather than
    // rejected by a rounding bit.
    const double slack = config.price_tolerance + 1.0e-12 * std::max(1.0, upper_bound);
    if (target_price < lower_bound - slack || target_price > upper_bound + slack) {
        return Result<ImpliedVolatilityResult>::failure(
            ErrorCode::RootNotBracketed,
            fmt::format("target price {:.10g} is outside the no-arbitrage window [{:.10g}, "
                        "{:.10g}] for this {} option; no volatility reproduces it",
                        target_price,
                        lower_bound,
                        upper_bound,
                        to_string(option.type())),
            kContext);
    }

    ImpliedVolatilityResult result;
    result.target_price = target_price;
    result.lower_price_bound = lower_bound;
    result.upper_price_bound = upper_bound;

    // A target at or below the price of the volatility floor is an option quoted
    // essentially at intrinsic. The root is below the floor, so it is reported at the
    // floor rather than chased into a region where the price is flat to machine
    // precision and the inversion is meaningless.
    const double price_lower = price_at(market, option, config.lower_volatility);
    if (target_price <= price_lower + config.price_tolerance) {
        result.implied_volatility = config.lower_volatility;
        result.achieved_price = price_lower;
        result.at_lower_floor = true;
        return Result<ImpliedVolatilityResult>::success(result);
    }

    // Bracket the root from above, expanding the ceiling until the price at it clears
    // the target or the hard cap is hit.
    double high = config.upper_volatility;
    double price_high = price_at(market, option, high);
    while (price_high < target_price && high < config.max_volatility) {
        high = std::min(high * 2.0, config.max_volatility);
        price_high = price_at(market, option, high);
    }
    if (price_high < target_price) {
        return Result<ImpliedVolatilityResult>::failure(
            ErrorCode::RootNotBracketed,
            fmt::format(
                "target price {:.10g} exceeds the price {:.10g} at the volatility ceiling "
                "{:.4g}; the quote sits against the upper no-arbitrage bound and is refused "
                "rather than matched by an unbounded volatility",
                target_price,
                price_high,
                config.max_volatility),
            kContext);
    }

    // Safeguarded Newton inside the maintained bracket [low, high].
    double low = config.lower_volatility;
    // Brenner-Subrahmanyam starting estimate, clamped into the bracket. Only a
    // starting point; the safeguarded iteration is robust to it being rough.
    double x = std::clamp(
        std::sqrt(kTwoPi / maturity) * target_price / std::max(spot_pv, 1.0e-12), low, high);
    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        const double price = price_at(market, option, x);
        const double residual = price - target_price;
        if (!std::isfinite(price)) {
            return Result<ImpliedVolatilityResult>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("the Black-Scholes price at volatility {:.6g} was not finite", x),
                kContext);
        }

        if (std::abs(residual) <= config.price_tolerance) {
            result.implied_volatility = x;
            result.achieved_price = price;
            result.iterations = iteration;
            return Result<ImpliedVolatilityResult>::success(result);
        }

        // Tighten the bracket: the price is increasing in volatility.
        if (residual > 0.0) {
            high = x;
        } else {
            low = x;
        }
        if (high - low <= config.volatility_tolerance) {
            result.implied_volatility = x;
            result.achieved_price = price;
            result.iterations = iteration;
            return Result<ImpliedVolatilityResult>::success(result);
        }

        // Newton step, taken only when it stays inside the bracket and vega is usable;
        // otherwise bisection, which cannot leave the bracket and always makes progress.
        const double vega = vega_at(market, option, x);
        double next = 0.5 * (low + high);
        if (vega > 1.0e-12) {
            const double newton = x - residual / vega;
            if (newton > low && newton < high) {
                next = newton;
            }
        }
        x = next;
    }

    return Result<ImpliedVolatilityResult>::failure(
        ErrorCode::ConvergenceFailure,
        fmt::format("implied volatility did not converge within {} iterations for target price "
                    "{:.10g}; the last bracket was [{:.10g}, {:.10g}]",
                    config.max_iterations,
                    target_price,
                    low,
                    high),
        kContext);
}

}  // namespace diffusionworks
