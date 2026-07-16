#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/pde/asset_grid.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// The spatial discretisation of the Black-Scholes operator at one interior node.
///
/// Central differences on a uniform S grid. With S_i = i*dS the coefficients are
///
///     a_i = (1/2) sigma^2 i^2 - (1/2) (r-q) i        multiplies V_{i-1}
///     b_i = -sigma^2 i^2 - r                          multiplies V_i
///     c_i = (1/2) sigma^2 i^2 + (1/2) (r-q) i        multiplies V_{i+1}
///
/// so that (L V)_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1} approximates
///
///     (1/2) sigma^2 S^2 d2V/dS2 + (r-q) S dV/dS - r V.
///
/// The dS cancels: sigma^2 S_i^2 / dS^2 = sigma^2 i^2 exactly on a uniform grid
/// with S_0 = 0. Writing the coefficients in terms of the index rather than the
/// price is not a micro-optimisation -- it removes a division by dS^2 that would
/// otherwise dominate the rounding error at fine grids.
struct OperatorCoefficients {
    std::vector<double> a;  ///< sub-diagonal, indexed by node
    std::vector<double> b;  ///< main diagonal
    std::vector<double> c;  ///< super-diagonal
};

/// Whether the discretised operator has the sign structure a Black-Scholes
/// discretisation should have.
///
/// What an M-matrix buys, precisely
/// --------------------------------
/// The implicit system is (I - dtau L). When L has non-negative off-diagonals and
/// (I - dtau L) is diagonally dominant with positive diagonal and non-positive
/// off-diagonals, that matrix is an M-matrix: its inverse is entrywise
/// non-negative. That is a *structural* result and it has one concrete consequence
/// worth having -- a non-negative payoff produces a non-negative price, at every
/// step, for any dtau. Oscillation into negative values becomes impossible rather
/// than merely unlikely.
///
/// It is not a statement about accuracy. An M-matrix scheme can be monotone,
/// positivity-preserving, and still wrong by a wide margin on a coarse grid. The
/// property rules out one specific pathology; it does not certify the answer.
///
/// When it fails
/// -------------
/// a_i < 0 requires (1/2) sigma^2 i^2 < (1/2) (r-q) i, i.e. i < (r-q)/sigma^2.
/// So the low-index nodes lose the sign structure when the drift dominates
/// diffusion there -- which happens for small sigma, large r-q, or a grid whose
/// dS is coarse enough that the first few nodes sit in that regime. This is the
/// classic cell-Peclet condition, and the standard remedy is upwinding the
/// convection term at the offending nodes.
///
/// This engine does *not* upwind. It reports where the structure fails and lets
/// the caller see it, because upwinding silently would trade a visible sign
/// problem for an invisible first-order accuracy loss -- and this project's whole
/// posture is that the second is worse.
struct OperatorDiagnostics {
    /// Nodes where a_i < 0: the convection term overwhelms diffusion locally.
    std::vector<std::int64_t> negative_sub_diagonal_nodes;

    /// Nodes where c_i < 0. Requires (r-q) sufficiently negative; rarer than the
    /// sub-diagonal failure but the same phenomenon with the sign reversed.
    std::vector<std::int64_t> negative_super_diagonal_nodes;

    /// The largest index below which the cell-Peclet condition fails,
    /// floor((r-q)/sigma^2). Nodes at or above it have a_i >= 0.
    ///
    /// Reported even when no node actually fails, so a caller can see how much
    /// margin the grid has rather than only whether it crossed the line.
    double peclet_threshold_index{};

    /// Whether every interior node has the M-matrix sign structure.
    [[nodiscard]] bool sign_structure_holds() const noexcept {
        return negative_sub_diagonal_nodes.empty() && negative_super_diagonal_nodes.empty();
    }
};

/// Builds the spatial operator's coefficients for the interior nodes of `grid`.
///
/// Node 0 and node N-1 are boundaries and carry no operator row; their entries in
/// the returned vectors are zero, so that an index means the same thing here as it
/// does in the grid.
[[nodiscard]] Result<OperatorCoefficients> black_scholes_coefficients(
    const MarketState& market, const BlackScholesModel& model, const AssetGrid& grid);

/// Reports the sign structure of the operator, per OperatorDiagnostics.
[[nodiscard]] OperatorDiagnostics diagnose_operator(const MarketState& market,
                                                    const BlackScholesModel& model,
                                                    const AssetGrid& grid,
                                                    const OperatorCoefficients& coefficients);

/// The largest stable time step for the explicit scheme on this grid, from the
/// actual coefficients rather than a textbook formula.
///
/// The explicit step is V^{n+1} = (I + dtau L) V^n. Von Neumann analysis of that
/// iteration gives the familiar
///
///     dtau <= 1 / (sigma^2 N^2 + r)
///
/// where N is the top node index -- but that expression is derived under
/// assumptions (constant coefficients, periodic boundaries) the actual problem
/// does not satisfy. What genuinely governs the iteration is the amplification of
/// each row, so this reads the limit off the assembled diagonal:
///
///     dtau <= 1 / max_i |b_i|
///
/// which for the coefficients above is exactly 1/max_i(sigma^2 i^2 + r) and so
/// agrees with the textbook bound, while remaining correct if the discretisation
/// changes. Computing it from the coefficients rather than restating the formula
/// means the stability check and the operator can never disagree about what the
/// scheme is doing.
///
/// The bound is necessary, not sufficient: satisfying it does not make the answer
/// accurate, only non-divergent.
[[nodiscard]] Result<double> explicit_stability_limit(const OperatorCoefficients& coefficients);

}  // namespace diffusionworks
