#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace diffusionworks {

/// A uniform grid in the asset coordinate, from 0 to S_max.
///
/// Uniform in S rather than in log S. The choice is deliberate and it costs
/// something, so it is worth stating why.
///
/// A log-S grid turns the Black-Scholes operator into one with constant
/// coefficients, which is tidier and clusters points where the payoff varies. But
/// it cannot represent S = 0, where the lower boundary condition lives, and it
/// distributes points geometrically -- so a barrier or a strike lands between grid
/// points at an S-dependent distance, which matters for Phase 7.
///
/// A uniform S grid keeps S = 0 exact, makes the boundary conditions the ones the
/// specification writes down, and lets a strike be placed exactly on a node. It
/// pays for that with points wasted in the far tail, where the solution is nearly
/// linear and nothing happens. For vanilla European options on the grids this
/// project uses, that trade is worth making; MATHEMATICAL-SPEC section 12 writes
/// the boundaries in S, and matching the specification directly is worth more here
/// than a constant-coefficient operator.
class AssetGrid {
public:
    /// Builds a uniform grid with `nodes` points spanning [0, s_max].
    ///
    /// Requires s_max > 0 and finite, and at least 3 nodes: the scheme needs an
    /// interior point to step, and a grid of two points is only its boundaries.
    [[nodiscard]] static Result<AssetGrid> uniform(double s_max, std::int64_t nodes);

    /// Builds a uniform grid that places `strike` exactly on a node.
    ///
    /// What this does, and what it does not
    /// ------------------------------------
    /// The European payoff is not differentiable at S = K. That non-smoothness is
    /// a genuine difficulty for a finite-difference scheme, because the Taylor
    /// expansion behind any order estimate assumes derivatives that do not exist
    /// there.
    ///
    /// Aligning the strike to a node removes **one** source of error: the terminal
    /// condition is then sampled exactly at the kink rather than at two nodes that
    /// straddle it, and interpolation near the strike no longer spans it. It also
    /// stops the kink's offset from the nearest node drifting as the grid refines,
    /// which is a source of non-monotone error across a convergence sweep.
    ///
    /// It does **not** universally restore second-order convergence, and an
    /// off-node strike does **not** universally force first-order behaviour. The
    /// observed order depends on the scheme, the grid, and how the error is
    /// measured; the honest claim is that alignment removes a known and avoidable
    /// error term, and EXP-06 measures what is actually attained rather than
    /// assuming it.
    ///
    /// Crank-Nicolson in particular can still oscillate near the kink regardless of
    /// alignment: its amplification factor approaches -1 for the highest-frequency
    /// modes, so the discontinuity in the payoff's derivative is oscillated rather
    /// than damped over the first few steps. The standard remedy is Rannacher
    /// smoothing -- a small number of fully implicit steps before switching to
    /// Crank-Nicolson. If it proves necessary here it will be an explicit,
    /// documented, separately validated option, never a silent modification of the
    /// scheme, because a Crank-Nicolson that is quietly not Crank-Nicolson for its
    /// first steps is not the method its results would claim.
    ///
    /// The spacing is chosen so that S_max lands on a node too, which generally
    /// adjusts the requested node count. The grid reports the S_max it built.
    [[nodiscard]] static Result<AssetGrid>
    with_strike_on_node(double s_max, std::int64_t nodes, double strike);

    [[nodiscard]] double s_max() const noexcept { return s_max_; }

    [[nodiscard]] std::int64_t nodes() const noexcept {
        return static_cast<std::int64_t>(values_.size());
    }

    /// Spacing dS.
    [[nodiscard]] double spacing() const noexcept { return spacing_; }

    /// The asset value at node i, with node 0 at S = 0 and the last at S_max.
    [[nodiscard]] double at(std::int64_t index) const noexcept;

    [[nodiscard]] std::span<const double> values() const noexcept { return values_; }

    /// The index of the node nearest `s`, or nullopt when `s` lies outside the
    /// grid.
    ///
    /// Absent rather than clamped: a caller asking about a spot beyond S_max has a
    /// modelling problem, and silently answering about S_max instead would hide it.
    [[nodiscard]] std::optional<std::int64_t> nearest_index(double s) const noexcept;

private:
    AssetGrid(double s_max, double spacing, std::vector<double> values) noexcept;

    double s_max_{};
    double spacing_{};
    std::vector<double> values_;
};

}  // namespace diffusionworks
