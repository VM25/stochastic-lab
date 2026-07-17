#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace diffusionworks {

/// Which sensitivity is being estimated.
enum class GreekName : std::uint8_t { Delta, Gamma, Vega };

[[nodiscard]] std::string_view to_string(GreekName greek) noexcept;

/// How the sensitivity is estimated from Monte Carlo paths.
///
/// The three families make different trade-offs, and the point of Phase 8 is to
/// measure them rather than assert one is best:
///
///   FiniteDifference  bump a parameter and re-price. Biased by the bump size,
///                     which must be chosen against a bias-variance trade-off (the
///                     bias grows with h, the variance grows as 1/h or 1/h^2). Works
///                     for any payoff, since it only re-evaluates it.
///   Pathwise          differentiate the payoff along the path. Unbiased and
///                     low-variance, but needs a payoff that is differentiable
///                     almost everywhere with an integrable derivative -- so it
///                     handles delta and vega for a call, and *cannot* give gamma,
///                     whose second derivative of the kink is a Dirac mass.
///   LikelihoodRatio   differentiate the density, not the payoff. The payoff is
///                     untouched, so it works for discontinuous payoffs where the
///                     pathwise method fails -- at the cost of higher variance,
///                     because it multiplies the payoff by a score that grows
///                     without bound.
enum class GreekMethod : std::uint8_t { FiniteDifference, Pathwise, LikelihoodRatio };

[[nodiscard]] std::string_view to_string(GreekMethod method) noexcept;

/// Parameters for a Greek estimation run.
struct GreeksMonteCarloConfig {
    std::int64_t paths{200000};
    std::uint64_t seed{20260717};
    double confidence_level{0.95};

    /// The spot bump for finite-difference delta and gamma, as a *fraction of the
    /// spot*. A relative bump keeps the trade-off comparable across spot levels;
    /// EXP-08 sweeps it.
    double spot_bump_fraction{1e-2};

    /// The volatility bump for finite-difference vega, in absolute volatility units.
    double volatility_bump{1e-2};
};

/// One estimator's answer: a value, its sampling uncertainty, and what it cost.
///
/// Per ADR-009 no estimator returns a bare number. A Greek without a standard error
/// cannot be compared against another estimator or against the analytic value: the
/// whole question of Phase 8 -- which estimator is more accurate, which has lower
/// variance -- is unanswerable without the uncertainty, so it is not optional here.
struct GreekEstimate {
    GreekName greek{};
    GreekMethod method{};

    double value{};

    /// Standard error of the estimate across paths. This is the estimator's
    /// realised precision, and it is what distinguishes a low-variance estimator
    /// from a lucky one.
    double standard_error{};

    /// The finite-difference bump actually used, in the parameter's own units.
    /// Zero for the pathwise and likelihood-ratio methods, which take no bump.
    double bump{};

    std::int64_t paths{};
    double runtime_seconds{};

    /// Conditions that qualify the estimate without invalidating it -- a degenerate
    /// boundary where the estimator is deterministic rather than sampled, most
    /// commonly. A warning never stands in for a failure: an undefined estimator is
    /// an Error, not a warning.
    std::vector<std::string> warnings;
};

/// Monte Carlo Greek estimators for European options under geometric Brownian
/// motion, validated against the analytic sensitivities.
///
/// Common random numbers, by construction
/// --------------------------------------
/// Every estimator draws the *same* standard normals -- one per path, from the
/// asset-shock stream at the run's seed. The finite-difference estimators then
/// re-price the bumped and unbumped options against those shared draws, so the
/// difference is a difference of two strongly positively correlated numbers and its
/// variance collapses. Bumping with independent draws instead would swamp the
/// sensitivity in sampling noise: the variance of a difference of independent
/// estimates is O(1/h^2), which for a small bump is enormous. The pathwise and
/// likelihood-ratio estimators need no bump, but they read the same draws so a
/// caller comparing all three compares them on one sample.
class GreeksMonteCarloEngine {
public:
    /// Estimates one Greek by one method.
    ///
    /// Validity is decided per method, not globally: a combination one method cannot
    /// handle is refused for that method alone, never for the whole Greek request.
    ///
    /// Refused:
    ///   - pathwise gamma by the *direct* second pathwise derivative -- for a vanilla
    ///     call that derivative carries a distributional term at the strike, which
    ///     this implementation does not represent. Other techniques (a mixed
    ///     pathwise-likelihood-ratio estimator, Malliavin weights, payoff smoothing)
    ///     recover a pathwise-style gamma; none is implemented here, so the engine
    ///     declines rather than claiming gamma is impossible;
    ///   - likelihood-ratio gamma and vega -- only the delta score is implemented,
    ///     and a wrong number is worse than declining;
    ///   - likelihood-ratio and pathwise at zero volatility or zero maturity -- their
    ///     formulas divide by sigma*sqrt(T) or assume a non-degenerate diffusion. The
    ///     finite-difference estimators are *not* refused there: they remain a
    ///     meaningful (deterministic) difference of the price away from a payoff kink,
    ///     and carry a warning saying so.
    [[nodiscard]] static Result<GreekEstimate> estimate(const MarketState& market,
                                                        const EuropeanOption& option,
                                                        const BlackScholesModel& model,
                                                        GreekName greek,
                                                        GreekMethod method,
                                                        const GreeksMonteCarloConfig& config);
};

}  // namespace diffusionworks
