#include <diffusionworks/engines/geometric_asian_analytic.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "GeometricAsianAnalyticEngine";

}  // namespace

GeometricAverageLaw GeometricAsianAnalyticEngine::law(const MarketState& market,
                                                      const AsianOption& option,
                                                      const BlackScholesModel& model) noexcept {
    const auto monitoring = static_cast<double>(option.monitoring_count());
    const double maturity = option.maturity();
    const double carry = market.rate() - market.dividend_yield();
    const double variance_rate = model.volatility() * model.volatility();

    GeometricAverageLaw result;

    // E[ln G] = ln S_0 + (r - q - sigma^2/2) * T(M+1)/(2M).
    //
    // The factor T(M+1)/(2M) is the average of the monitoring dates,
    // (1/M) sum_i t_i, which tends to T/2 as M grows: the average sees the drift
    // for roughly half the option's life, which is why an Asian is cheaper than
    // its European counterpart.
    const double average_time = maturity * (monitoring + 1.0) / (2.0 * monitoring);
    result.log_mean = std::log(market.spot()) + (carry - 0.5 * variance_rate) * average_time;

    // Var[ln G] = sigma^2 T (M+1)(2M+1) / (6 M^2), from
    // Var(sum_i W_{t_i}) = sum_{i,j} min(t_i, t_j) and
    // sum_{i,j} min(i,j) = M(M+1)(2M+1)/6.
    //
    // Tends to sigma^2 T/3 as M grows -- the classical continuous result -- and
    // equals sigma^2 T exactly at M = 1, where the average is just S_T.
    result.log_variance = variance_rate * maturity * (monitoring + 1.0) * (2.0 * monitoring + 1.0) /
                          (6.0 * monitoring * monitoring);

    result.forward = std::exp(result.log_mean + 0.5 * result.log_variance);
    return result;
}

Result<PricingResult> GeometricAsianAnalyticEngine::price(const MarketState& market,
                                                          const AsianOption& option,
                                                          const BlackScholesModel& model) {
    const auto start = std::chrono::steady_clock::now();

    if (option.averaging() != AveragingType::Geometric) {
        // Refused rather than priced. The arithmetic average of lognormals is not
        // lognormal, so this formula does not describe it, and returning a
        // number would answer a different question convincingly.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("this engine prices geometric averaging only, but the option uses {} "
                        "averaging. The arithmetic average of lognormals is not lognormal and "
                        "has no closed form.",
                        to_string(option.averaging())),
            kContext);
    }

    const GeometricAverageLaw average = law(market, option, model);
    const double discount = market.discount_factor(option.maturity());
    const double strike = option.strike();

    PricingResult result;
    result.method = "geometric_asian_analytic";

    // sigma = 0 leaves the average deterministic, so the option is worth the
    // discounted intrinsic value of a known number. Handled explicitly rather
    // than dividing by a zero standard deviation.
    const double log_standard_deviation = std::sqrt(average.log_variance);
    if (log_standard_deviation <= 0.0) {
        const double certain_average = std::exp(average.log_mean);
        result.value = discount * option.payoff(certain_average);
        result.add_warning(fmt::format(
            "degenerate limit: the geometric average has zero variance (sigma={}), so the price "
            "is the discounted intrinsic value of a known average ({})",
            model.volatility(),
            certain_average));
        result.add_diagnostic("log_mean", average.log_mean);
        result.add_diagnostic("log_variance", average.log_variance);
        result.add_diagnostic("average_forward", average.forward);
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

    // For lognormal X with ln X ~ N(m, s^2):
    //   E[(X-K)^+] = e^{m + s^2/2} N(d1) - K N(d2),  d1 = (m - ln K + s^2)/s,
    //                                                d2 = d1 - s.
    const double d1 =
        (average.log_mean - std::log(strike) + average.log_variance) / log_standard_deviation;
    const double d2 = d1 - log_standard_deviation;

    switch (option.type()) {
        case OptionType::Call:
            result.value = discount * (average.forward * norm_cdf(d1) - strike * norm_cdf(d2));
            break;
        case OptionType::Put:
            // N(-d) computed directly rather than as 1 - N(d), for the same
            // reason as in the Black-Scholes engine: the complement cancels in
            // the tail, which is exactly where a deep out-of-the-money price
            // lives.
            result.value = discount * (strike * norm_cdf(-d2) - average.forward * norm_cdf(-d1));
            break;
    }

    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("price is not finite (spot={}, strike={}, maturity={}, volatility={}, "
                        "monitoring={})",
                        market.spot(),
                        strike,
                        option.maturity(),
                        model.volatility(),
                        option.monitoring_count()),
            kContext);
    }
    if (result.value < 0.0) {
        // The closed form cannot produce a negative value, so one signals
        // catastrophic cancellation. Reported rather than clamped: clamping would
        // hide the defect this check exists to find.
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("price is negative ({}), which the closed form cannot produce; this "
                        "indicates catastrophic cancellation",
                        result.value),
            kContext);
    }

    result.add_diagnostic("d1", d1);
    result.add_diagnostic("d2", d2);
    result.add_diagnostic("log_mean", average.log_mean);
    result.add_diagnostic("log_variance", average.log_variance);
    result.add_diagnostic("average_forward", average.forward);
    result.add_diagnostic("monitoring_count", option.monitoring_count());

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
