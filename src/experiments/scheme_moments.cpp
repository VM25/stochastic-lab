#include <diffusionworks/experiments/scheme_moments.hpp>
#include <diffusionworks/simulation/time_grid.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "scheme_terminal_moments";

}  // namespace

Result<TerminalMoments> analytic_terminal_moments(const MarketState& market,
                                                  const BlackScholesModel& model,
                                                  double maturity) {
    if (!(maturity >= 0.0) || !std::isfinite(maturity)) {
        return Result<TerminalMoments>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be finite and non-negative, got {}", maturity),
            "analytic_terminal_moments");
    }

    const double carry = market.rate() - market.dividend_yield();
    const double variance_rate = model.volatility() * model.volatility();

    TerminalMoments moments;
    moments.first = market.spot() * std::exp(carry * maturity);
    moments.second =
        market.spot() * market.spot() * std::exp((2.0 * carry + variance_rate) * maturity);
    return Result<TerminalMoments>::success(moments);
}

Result<TerminalMoments> scheme_terminal_moments(const MarketState& market,
                                                const BlackScholesModel& model,
                                                DiscretizationScheme scheme,
                                                double maturity,
                                                std::int64_t steps) {
    const auto grid = TimeGrid::uniform(maturity, steps);
    if (!grid.ok()) {
        return Result<TerminalMoments>::failure(grid.error());
    }

    // The exact scheme samples the true terminal law, so its moments are the
    // analytic ones for any step count. Returned from the same function rather
    // than special-cased at the call site, so that a caller sweeping over schemes
    // gets a uniform answer.
    if (scheme == DiscretizationScheme::Exact) {
        return analytic_terminal_moments(market, model, maturity);
    }

    const double step = grid.value().step_size();
    const double carry = market.rate() - market.dividend_yield();
    const double variance_rate = model.volatility() * model.volatility();

    // Each step multiplies the state by a factor u_n that is independent of S_n,
    // because u_n depends only on Z_n while S_n depends only on Z_0..Z_{n-1}.
    // Independence is what lets the moments factor as E[S_M^k] = S_0^k (E[u^k])^M;
    // without it the powers would not separate.
    const double first_factor = 1.0 + carry * step;
    double second_factor = first_factor * first_factor + variance_rate * step;

    if (scheme == DiscretizationScheme::Milstein) {
        // The Levy area correction contributes (sigma^4/2) dt^2 to E[u^2] and
        // nothing to E[u]: see the header derivation. This term is what makes
        // Milstein's second moment differ from Euler's while its first does not.
        second_factor += 0.5 * variance_rate * variance_rate * step * step;
    }

    const auto exponent = static_cast<double>(steps);

    TerminalMoments moments;
    moments.first = market.spot() * std::pow(first_factor, exponent);
    moments.second = market.spot() * market.spot() * std::pow(second_factor, exponent);

    if (!std::isfinite(moments.first) || !std::isfinite(moments.second)) {
        return Result<TerminalMoments>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("scheme moments are not finite (scheme={}, steps={}, dt={}); the per-step "
                        "factor {} raised to {} overflowed",
                        to_string(scheme),
                        steps,
                        step,
                        second_factor,
                        steps),
            kContext);
    }

    return Result<TerminalMoments>::success(moments);
}

}  // namespace diffusionworks
