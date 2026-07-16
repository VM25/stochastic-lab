#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace diffusionworks {

/// A tridiagonal system, stored as its three diagonals.
///
/// Row i reads
///
///     lower[i] * x[i-1] + diagonal[i] * x[i] + upper[i] * x[i+1] = rhs[i]
///
/// with lower[0] and upper[n-1] unused. They are required to be present anyway,
/// so that every diagonal has the same length and an index means the same thing
/// in all three: a solver whose arrays are offset relative to one another reads
/// the wrong coefficient without ever reading out of bounds, and the resulting
/// prices look ordinary.
struct TridiagonalSystem {
    std::vector<double> lower;
    std::vector<double> diagonal;
    std::vector<double> upper;
    std::vector<double> rhs;

    [[nodiscard]] std::size_t size() const noexcept { return diagonal.size(); }
};

/// What the elimination observed on its way through.
///
/// Reported rather than enforced. The distinction matters: `diagonally_dominant`
/// records whether the caller has a *theoretical guarantee*, while
/// `smallest_pivot_ratio` records what actually happened. A caller can have the
/// second without the first and be perfectly fine, which is why this solver
/// declines to turn the first into a gate.
struct TridiagonalDiagnostics {
    /// Whether the matrix is diagonally dominant.
    ///
    /// Diagonal dominance is a **sufficient** condition for the Thomas algorithm
    /// to be stable -- no pivot can vanish, and the growth factor is bounded. It
    /// is **not necessary**. Symmetric positive-definite systems, M-matrices, and
    /// a great many merely well-conditioned systems are solved accurately without
    /// it. Refusing every non-dominant system would therefore reject problems the
    /// algorithm handles correctly, and would state a falsehood about the
    /// mathematics while doing so.
    ///
    /// So it is a diagnostic. True means the result carries a proof; false means
    /// it does not, and the caller should read `smallest_pivot_ratio` and the
    /// residual to judge what they have.
    bool diagonally_dominant{};

    /// The smallest |pivot| / row_scale seen during elimination.
    ///
    /// This is the quantity that actually governs whether the solve is trustworthy,
    /// which is why it is measured rather than inferred from dominance. A ratio near
    /// 1 is healthy. A ratio near zero means the elimination nearly divided by
    /// nothing, and the answer -- though finite, and though its residual may look
    /// small -- can be arbitrarily far from the truth.
    double smallest_pivot_ratio{};
};

struct TridiagonalSolution {
    std::vector<double> values;
    TridiagonalDiagnostics diagnostics;
};

/// Solves a tridiagonal system by the Thomas algorithm.
///
/// The Thomas algorithm is Gaussian elimination specialised to three diagonals:
/// O(n) rather than O(n^3), which is what makes a fine PDE grid affordable at all.
/// It performs no pivoting, and that is the catch -- an unpivoted elimination can
/// meet a pivot driven to zero by cancellation, and dividing by it produces a
/// finite, enormous, entirely wrong answer that a plot renders without complaint.
///
/// The solver therefore checks **the failure mode itself** rather than a proxy for
/// it. A vanishing pivot is detected where it happens and reported with the row
/// that produced it. Diagonal dominance would rule that out in advance, but it is
/// sufficient rather than necessary (see TridiagonalDiagnostics), so it is
/// reported and not required: a solver that refused every non-dominant system
/// would reject problems it solves correctly, on a false premise.
///
/// Fails, rather than returning a plausible vector, when:
///   - the diagonals have inconsistent lengths, or the system is empty;
///   - any coefficient is non-finite;
///   - a pivot vanishes relative to the scale of the row that produced it;
///   - the computed solution contains a non-finite entry.
///
/// Callers who need the guarantee rather than the absence of an observed failure
/// should assert `diagnostics.diagonally_dominant` themselves. The PDE schemes do
/// exactly that on the operators they generate, where the property is genuinely
/// theirs to establish.
[[nodiscard]] Result<TridiagonalSolution> solve_tridiagonal(const TridiagonalSystem& system);

/// Whether the matrix is diagonally dominant: weakly in every row, strictly in at
/// least one.
///
/// A sufficient condition for the Thomas algorithm's stability, not a necessary
/// one. Exposed so a caller who generates its own matrices -- as the PDE schemes
/// do -- can assert the property about them before relying on it.
[[nodiscard]] bool is_diagonally_dominant(const TridiagonalSystem& system) noexcept;

/// The residual ||Ax - b||_inf of a claimed solution.
///
/// The independent check on the solve. Forming Ax from the original diagonals uses
/// none of the elimination's intermediate quantities, so it cannot agree with a
/// wrong answer by sharing its mistake.
///
/// A small residual is necessary but not sufficient for a correct answer: on an
/// ill-conditioned system the residual can be tiny while the error is large, since
/// ||x - x_exact|| is bounded by the residual times the condition number rather
/// than by the residual alone. Read it alongside `smallest_pivot_ratio`.
[[nodiscard]] Result<double> tridiagonal_residual(const TridiagonalSystem& system,
                                                  std::span<const double> solution);

}  // namespace diffusionworks
