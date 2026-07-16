#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/simulation/gbm_stepper.hpp>

#include <cstdint>

namespace diffusionworks {

/// The first two moments of the terminal price.
struct TerminalMoments {
    /// \f$ E[S_T] \f$.
    double first{};

    /// \f$ E[S_T^2] \f$.
    double second{};

    /// \f$ E[S_T^2] - E[S_T]^2 \f$.
    [[nodiscard]] double variance() const noexcept { return second - first * first; }
};

/// The exact moments of a discretisation scheme's terminal price, in closed form.
///
/// Why this exists
/// ---------------
/// EXP-03 asks whether the schemes reproduce expected values at their theoretical
/// weak rate. The obvious way to answer is to simulate and compare means against
/// the analytic Black-Scholes moments -- and it does not work. The weak bias here
/// is O(dt) while the Monte Carlo standard error is O(1/sqrt(N)), and for these
/// parameters the bias at M = 64 is about 2e-3 against a standard error of 5e-2.
/// The measurement would be reporting noise. Pairing against the exact scheme on
/// common Brownian paths (see measure_weak_error) removes most of that noise, but
/// the paired standard error is itself O(sqrt(dt)/sqrt(N)) while the bias is
/// O(dt), so the resolution *degrades* as the grid refines -- the opposite of what
/// the experiment needs.
///
/// For the first two moments no simulation is needed at all. Each scheme's step
/// is a multiplicative factor u_n that is independent of the current state, so
/// E[S_M^k] = S_0^k (E[u^k])^M, and E[u] and E[u^2] are elementary. The weak error
/// then follows exactly, with no sampling uncertainty to separate out.
///
/// This is a stronger result than a regression: the bias halves per grid doubling
/// to four significant figures, which establishes weak order 1 rather than
/// estimating it.
///
/// Derivations (a = r - q, and Z standard normal, independent of S_n):
///
///   Euler-Maruyama, S_{n+1} = S_n (1 + a dt + sigma sqrt(dt) Z):
///     E[u]   = 1 + a dt
///     E[u^2] = (1 + a dt)^2 + sigma^2 dt
///
///   Milstein, S_{n+1} = S_n (1 + a dt + sigma sqrt(dt) Z + (sigma^2/2) dt (Z^2 - 1)):
///     writing u = c + b Z + d Z^2 with c = 1 + a dt - (sigma^2/2) dt,
///     b = sigma sqrt(dt), d = (sigma^2/2) dt, and using E[Z]=E[Z^3]=0, E[Z^2]=1,
///     E[Z^4]=3:
///     E[u]   = c + d = 1 + a dt
///     E[u^2] = (c+d)^2 + b^2 + 2 d^2 = (1 + a dt)^2 + sigma^2 dt + (sigma^4/2) dt^2
///
/// Note the first moments coincide. The Milstein correction has mean zero, so it
/// cannot shift E[S] at all: Milstein attains strong order 1 while leaving the
/// first moment's weak error *identical* to Euler's. That is a real and slightly
/// counter-intuitive property of the scheme, and EXP-03 reports it rather than
/// implying that a better strong order buys a better weak one.
[[nodiscard]] Result<TerminalMoments> scheme_terminal_moments(const MarketState& market,
                                                              const BlackScholesModel& model,
                                                              DiscretizationScheme scheme,
                                                              double maturity,
                                                              std::int64_t steps);

/// The true moments of the Black-Scholes terminal law.
///
///   E[S_T]   = S_0 e^{(r-q)T}
///   E[S_T^2] = S_0^2 e^{(2(r-q) + sigma^2)T}
///
/// The Exact scheme reproduces these for any number of steps, which is what makes
/// it usable as the pathwise reference in a strong-convergence study and as the
/// control in the paired weak estimator.
[[nodiscard]] Result<TerminalMoments> analytic_terminal_moments(const MarketState& market,
                                                                const BlackScholesModel& model,
                                                                double maturity);

}  // namespace diffusionworks
