#pragma once

#include <diffusionworks/core/interval.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/simulation/gbm_stepper.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace diffusionworks {

/// How a level's error was obtained.
enum class ErrorSource : std::uint8_t {
    /// Estimated by simulation, and so carrying a sampling distribution.
    Simulated,

    /// Computed in closed form. Exact up to floating point, with no sampling
    /// uncertainty to report.
    Analytic,
};

[[nodiscard]] const char* to_string(ErrorSource s) noexcept;

/// One grid level of a convergence study.
struct ConvergenceLevel {
    std::int64_t steps{};
    double step_size{};

    ErrorSource source{ErrorSource::Simulated};

    /// The magnitude of the error at this level. This is what the power-law fit
    /// regresses, since an order is a statement about magnitude.
    double error{};

    /// The error with its sign, where the sign is meaningful.
    ///
    /// Present for weak studies, where a bias that is consistently negative says
    /// something an absolute value discards -- the explicit schemes under-state
    /// E[S_T] rather than scattering around it.
    std::optional<double> signed_error;

    /// Standard error of the error estimate. Absent when `source` is Analytic.
    std::optional<double> error_standard_error;

    /// The raw E[f(S_T)] estimate, for weak studies.
    std::optional<double> estimate;

    /// The reference E[f(S_T)] this level was measured against.
    std::optional<double> reference;

    /// Paths simulated. Zero for an analytic level.
    std::uint64_t paths{};

    /// Non-positive states seen. An explicit scheme on a coarse grid can step the
    /// price through zero; the count makes that visible rather than leaving a
    /// payoff's max(.,0) to absorb it silently.
    std::int64_t non_positive_states{0};

    /// How many standard errors the error stands above zero.
    ///
    /// The question "is this a measurement of bias, or of noise?" made numeric.
    /// Absent -- meaning "exactly resolved" -- for an analytic level.
    [[nodiscard]] std::optional<double> resolution() const noexcept {
        if (!error_standard_error.has_value() || *error_standard_error <= 0.0) {
            return std::nullopt;
        }
        return error / *error_standard_error;
    }
};

/// The order implied by two adjacent levels.
///
/// Reported because a single fitted slope hides the shape of the data. An error
/// that is C*dt^k plus higher-order terms has a *local* order that drifts with dt
/// and only reaches k in the limit, and seeing that drift is what distinguishes
/// "the scheme has the wrong order" from "the coarse end of this grid is not yet
/// asymptotic". The first is a defect; the second is arithmetic.
struct LocalOrder {
    std::int64_t coarse_steps{};
    std::int64_t fine_steps{};
    double order{};
};

/// The verdict of a convergence study against its theoretical order.
enum class ConvergenceVerdict : std::uint8_t {
    /// Both the full-range and the asymptotic-window intervals contain the
    /// theoretical order: the whole studied range behaves as the theory predicts.
    Consistent,

    /// The asymptotic-window interval contains the theoretical order but the
    /// full-range one does not.
    ///
    /// Not a failure. The error is C*dt^k only to leading order; at coarse dt the
    /// neglected terms bend the log-log plot and pull the fitted slope away from
    /// k. The evidence that the theory holds is that the local orders climb toward
    /// it as the grid refines, which is exactly what this verdict records.
    ConsistentAsymptotically,

    /// The asymptotic-window interval excludes the theoretical order. A real
    /// contradiction, and a blocker.
    Inconsistent,

    /// At least one fitted level's error does not stand clear of its own sampling
    /// noise, so any slope through them would describe the noise.
    ///
    /// Reported instead of a number. A fit through unresolved points still returns
    /// a slope, and that slope looks exactly like a measurement.
    NoiseDominated,
};

[[nodiscard]] const char* to_string(ConvergenceVerdict v) noexcept;

/// A fitted convergence study and its verdict.
struct ConvergenceStudy {
    std::string scheme;

    /// What was measured: "strong_error", or "weak_error:<test function>".
    std::string quantity;

    std::vector<ConvergenceLevel> levels;
    std::vector<LocalOrder> local_orders;

    /// The fit over every level.
    LinearFit full_fit;
    ConfidenceInterval full_slope_interval;

    /// The fit over the finest `asymptotic_level_count` levels.
    ///
    /// The window is chosen by the caller *before* the data is seen, and both fits
    /// are always reported. Selecting the window after the fact would be choosing
    /// the answer.
    std::size_t asymptotic_level_count{};
    LinearFit asymptotic_fit;
    ConfidenceInterval asymptotic_slope_interval;

    double theoretical_order{};
    ConvergenceVerdict verdict{};

    /// The smallest resolution across the levels, or nullopt if all are analytic.
    std::optional<double> worst_resolution;
};

/// How the weak-convergence test function maps a terminal price to a payoff.
enum class WeakTestFunction : std::uint8_t {
    /// f(S_T) = S_T. Reference: the forward, S_0 e^{(r-q)T}.
    Identity,

    /// f(S_T) = S_T^2. Reference: S_0^2 e^{(2(r-q)+sigma^2)T}.
    Square,

    /// f(S_T) = (S_T - K)^+. Reference: the undiscounted Black-Scholes call value.
    ///
    /// The only one of the three with no closed-form scheme moment, so the only
    /// one that must be simulated.
    CallPayoff,
};

[[nodiscard]] const char* to_string(WeakTestFunction f) noexcept;

/// Measures E|S_T^scheme - S_T^exact| at one grid level.
///
/// The two paths are driven by the *same* Brownian increments: same seed, same
/// path index, same stream coordinates, differing only in the stepping rule. That
/// coupling is what makes this a strong error rather than a difference of two
/// independent samples -- whose expected absolute difference is O(1) and does not
/// converge at all (FAILURE-MODES section 6).
[[nodiscard]] Result<ConvergenceLevel> measure_strong_error(const MarketState& market,
                                                            const BlackScholesModel& model,
                                                            DiscretizationScheme scheme,
                                                            double maturity,
                                                            std::int64_t steps,
                                                            std::uint64_t paths,
                                                            std::uint64_t seed);

/// The exact weak error at one grid level, in closed form.
///
/// Available for Identity and Square only, where the scheme's terminal moment has
/// an elementary expression (see scheme_moments.hpp). Exact, with no sampling
/// uncertainty: this is a computation rather than a measurement.
///
/// Preferred over the simulated form wherever it applies. The weak bias is O(dt)
/// while a paired standard error is O(sqrt(dt)/sqrt(N)), so simulation resolves
/// the bias *worse* as the grid refines -- the exact opposite of what a
/// convergence study needs.
[[nodiscard]] Result<ConvergenceLevel> weak_error_analytic(const MarketState& market,
                                                           const BlackScholesModel& model,
                                                           DiscretizationScheme scheme,
                                                           double maturity,
                                                           WeakTestFunction test_function,
                                                           std::int64_t steps);

/// Measures the weak error at one grid level by simulation, paired against the
/// exact scheme on common Brownian paths.
///
/// The estimator
/// -------------
/// The exact scheme's terminal value is distributed exactly as S_T, so
/// E[f(S_T^exact)] = E[f(S_T)] with no approximation. The weak error therefore
/// rewrites as a single expectation of a difference:
///
///     E[f(S_T^dt)] - E[f(S_T)] = E[ f(S_T^dt) - f(S_T^exact) ]
///
/// and it is estimated by averaging that difference over paths that share their
/// Brownian increments. Pairing does not change the estimand -- the identity above
/// holds however the two are coupled -- but it removes the common sampling noise,
/// which is what makes the bias visible at all. It is a control variate with
/// beta = 1 and an analytically known control mean, which is the Phase 4
/// machinery arriving at the same place from a different direction.
///
/// Measured here at 9x variance reduction over the unpaired difference. That is
/// still not always enough: because the bias is O(dt) and the paired standard
/// error is O(sqrt(dt)/sqrt(N)), the resolution behaves like sqrt(dt*N) and
/// degrades as the grid refines. The returned level carries its `resolution()` so
/// a caller can see whether the number means anything, and `fit_convergence`
/// refuses to fit unresolved levels.
///
/// `reference_deviation` is also filled in: the exact scheme's own sample mean
/// against the analytic value. Without it this estimator would only ever compare
/// the scheme to another scheme, and a bug in the exact stepper would report
/// itself as zero bias.
[[nodiscard]] Result<ConvergenceLevel> measure_weak_error(const MarketState& market,
                                                          const BlackScholesModel& model,
                                                          DiscretizationScheme scheme,
                                                          double maturity,
                                                          double strike,
                                                          WeakTestFunction test_function,
                                                          std::int64_t steps,
                                                          std::uint64_t paths,
                                                          std::uint64_t seed);

/// The exact E[f(S_T)] under the Black-Scholes law.
[[nodiscard]] Result<double> weak_reference(const MarketState& market,
                                            const BlackScholesModel& model,
                                            double maturity,
                                            double strike,
                                            WeakTestFunction test_function);

/// How many standard errors an error must stand above zero to be treated as
/// resolved rather than as noise.
///
/// Three is the conventional line for "distinguishable from zero", and the studies
/// here either clear it comfortably or fail it outright, so the exact value is not
/// load-bearing. What matters is that some line exists and that failing it produces
/// NoiseDominated rather than a slope.
inline constexpr double kResolutionThreshold = 3.0;

/// How far an *exact* study's fitted slope may sit from the theoretical order and
/// still be called consistent.
///
/// Needed because a regression interval is the wrong instrument for exact data.
/// The interval is built from the residual scatter, which measures sampling noise;
/// an analytic study has none, so the residual is ~1e-15, the interval collapses to
/// a point, and containment becomes a test no true order could ever pass. The
/// closed-form weak errors fit with slope 0.9997 against a true order of exactly 1
/// and were reported inconsistent -- the machinery was wrong, not the scheme.
///
/// The remaining gap is not uncertainty but the neglected higher-order term: the
/// error is C*dt^k*(1 + O(dt)), so the fitted slope is k + O(dt). Over the grids
/// used here that contribution is below 1e-3, which this tolerance clears by an
/// order of magnitude while still sitting 50x below the 0.5 gap between the
/// candidate orders -- so it cannot mistake order 1/2 for order 1.
inline constexpr double kAnalyticOrderTolerance = 0.01;

/// Fits log(error) against log(step_size) and judges the slope against theory.
///
/// `asymptotic_level_count` selects how many of the finest levels form the
/// asymptotic window. It must be at least 3 -- a slope needs an uncertainty -- and
/// must be chosen before the results are seen.
///
/// Requires at least three levels overall: EXP-02 names inferring an order from
/// too few grid levels as a failure condition, and two points can be joined by a
/// line of any slope.
[[nodiscard]] Result<ConvergenceStudy> fit_convergence(std::string scheme,
                                                       std::string quantity,
                                                       std::vector<ConvergenceLevel> levels,
                                                       double theoretical_order,
                                                       std::size_t asymptotic_level_count,
                                                       double confidence_level = 0.95);

}  // namespace diffusionworks
