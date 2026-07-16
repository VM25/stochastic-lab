#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/pde/asset_grid.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace diffusionworks {

/// The time-stepping scheme.
enum class PdeScheme : std::uint8_t {
    /// V^{n+1} = (I + dtau L) V^n.
    ///
    /// One matrix-vector product per step, no solve. Conditionally stable: the
    /// step must satisfy dtau <= 1/max|b_i|, which scales as dS^2, so halving the
    /// spatial spacing quarters the affordable step. Cheap per step and often
    /// expensive per unit of accuracy.
    Explicit,

    /// (I - dtau L) V^{n+1} = V^n.
    ///
    /// One tridiagonal solve per step. Unconditionally stable, and *that word does
    /// not mean accurate*. A large dtau produces a bounded, smooth, entirely wrong
    /// answer -- which is more dangerous than a divergent one, because nothing about
    /// it looks wrong. First-order accurate in tau.
    Implicit,

    /// (I - (dtau/2) L) V^{n+1} = (I + (dtau/2) L) V^n.
    ///
    /// Second-order accurate in tau, and unconditionally stable in the same limited
    /// sense as the implicit scheme. Its amplification factor approaches -1 for the
    /// highest-frequency modes rather than 0, so it does not damp them -- it
    /// oscillates them. With a non-smooth payoff that can show up as ringing near
    /// the strike over the first few steps. See RannacherSteps.
    CrankNicolson,
};

[[nodiscard]] const char* to_string(PdeScheme scheme) noexcept;

/// How many initial steps to take fully implicit before switching to
/// Crank-Nicolson.
///
/// Rannacher smoothing, and it is an explicit option rather than a silent
/// modification. Crank-Nicolson's amplification factor tends to -1 for the highest
/// frequencies, so the kink in a vanilla payoff is oscillated rather than damped.
/// A few fully implicit steps -- whose factor tends to 0 -- annihilate those modes
/// first, after which Crank-Nicolson proceeds on data that is effectively smooth.
///
/// Zero by default. A Crank-Nicolson run that quietly took implicit steps is not
/// Crank-Nicolson, and its convergence order is not Crank-Nicolson's, so turning
/// this on silently would misreport what was measured. When it is on, the record
/// says so and the scheme is described as what it is.
///
/// Two is the conventional choice: enough to kill the highest modes, few enough
/// that the first-order local error of those steps does not dominate the global
/// second-order behaviour.
struct RannacherSteps {
    std::int64_t count{0};
};

/// What the PDE solve observed about itself.
struct PdeDiagnostics {
    std::int64_t asset_nodes{};
    std::int64_t time_steps{};
    double s_max{};
    double asset_spacing{};
    double time_step{};

    /// dtau / (largest stable dtau) for the explicit scheme.
    ///
    /// Reported for every scheme, since the number is meaningful for all three: it
    /// is only a *limit* for the explicit one, but it says how far the implicit and
    /// Crank-Nicolson runs are operating beyond where an explicit scheme could go,
    /// which is exactly what their unconditional stability is buying.
    double explicit_stability_ratio{};

    /// Whether the explicit scheme's stability bound is satisfied.
    ///
    /// Only meaningful for Explicit. Absent otherwise rather than set true, which
    /// would suggest the other schemes had passed a test they never took.
    std::optional<bool> explicit_stable;

    /// Whether the spatial operator has the M-matrix sign structure everywhere.
    bool sign_structure_holds{};

    /// Interior nodes where the cell-Peclet condition fails.
    std::int64_t peclet_violating_nodes{};

    /// The largest S at which the sign structure fails, or 0 when it holds.
    ///
    /// The number that says whether to care. The violation occupies node indices
    /// below (r-q)/sigma^2, so in price terms it occupies S < dS*(r-q)/sigma^2 --
    /// a region that *shrinks as the grid refines*, because dS does. For ordinary
    /// parameters it is a sliver adjacent to S = 0 where a call is worthless
    /// anyway; for a put, where the value at S = 0 is the discounted strike, it
    /// deserves more attention.
    ///
    /// Reported in prices rather than node counts because "1 node violates" is
    /// unjudgeable, while "the structure fails for S <= 2.5, and the option is
    /// struck at 100" is immediate.
    double peclet_violating_max_s{};

    /// The worst |pivot|/row_scale over every tridiagonal solve in the run.
    ///
    /// A pivot-health diagnostic, not a condition-number estimate. Absent for the
    /// explicit scheme, which performs no solve.
    std::optional<double> worst_pivot_ratio;

    /// The largest ||Ax - b||_inf over every solve.
    std::optional<double> worst_residual;

    /// Rannacher steps actually taken.
    std::int64_t rannacher_steps{};

    /// The most negative value anywhere on the grid at any step.
    ///
    /// Zero when the solution stayed non-negative. A vanilla payoff is
    /// non-negative and its price must be too, so a negative value is a real
    /// defect -- oscillation, most likely -- rather than a rounding artifact.
    double most_negative_value{};

    /// A measure of spatial oscillation in the final solution: the largest
    /// second difference of sign, normalised.
    ///
    /// A Black-Scholes value function is convex in S for a vanilla option, so its
    /// second difference should be non-negative everywhere. Ringing shows up as
    /// alternating signs, which this counts.
    std::int64_t convexity_violations{};
};

struct PdeConfig {
    /// Nodes in the asset grid.
    std::int64_t asset_nodes{201};

    /// Steps in the backward-time grid.
    std::int64_t time_steps{100};

    /// The upper truncation of the asset domain.
    ///
    /// Absent means "choose from the model": a multiple of the strike wide enough
    /// that the asymptotic boundary is a good approximation. Stated explicitly
    /// because the choice matters and is measured (EXP-06).
    std::optional<double> s_max;

    PdeScheme scheme{PdeScheme::CrankNicolson};

    /// Place the strike exactly on a grid node.
    ///
    /// Removes one source of error -- the terminal condition is sampled at the kink
    /// rather than at two nodes straddling it. It does not by itself guarantee any
    /// particular convergence order.
    bool align_strike_to_node{true};

    RannacherSteps rannacher;
};

/// Prices a European option by solving the Black-Scholes PDE.
///
/// The value at the requested spot is interpolated from the grid, since the spot
/// generally does not fall on a node.
class FiniteDifferenceEngine {
public:
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const EuropeanOption& option,
                                                     const BlackScholesModel& model,
                                                     const PdeConfig& config);

    /// The full solution on the grid at tau = T, for tests and diagnostics.
    struct Solution {
        AssetGrid grid;
        std::vector<double> values;
        PdeDiagnostics diagnostics;
    };

    [[nodiscard]] static Result<Solution> solve(const MarketState& market,
                                                const EuropeanOption& option,
                                                const BlackScholesModel& model,
                                                const PdeConfig& config);
};

/// Linear interpolation of a grid function at `s`.
///
/// Linear rather than cubic, deliberately. A cubic spline is smoother and more
/// accurate on smooth data, and a vanilla value function is smooth *away from the
/// strike* -- but near it the second derivative is large and a cubic can overshoot,
/// introducing oscillation into an answer the scheme itself did not oscillate.
/// Linear interpolation cannot overshoot: its result is bounded by the two nodes it
/// spans. That matters more here than the extra order, because the whole point of
/// the exercise is to distinguish scheme error from artifact.
///
/// The cost is real: linear interpolation is second-order accurate in dS, so it can
/// contribute error of the same order as the scheme. On a node it is exact.
[[nodiscard]] Result<double>
interpolate_linear(const AssetGrid& grid, std::span<const double> values, double s);

}  // namespace diffusionworks
