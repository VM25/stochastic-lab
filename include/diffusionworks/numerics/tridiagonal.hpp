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
/// Both fields are reported, neither is enforced, and **neither is a statement
/// about the accuracy of the answer**. They describe the elimination, not the
/// error. Accuracy depends on the conditioning of the matrix, which nothing here
/// measures.
struct TridiagonalDiagnostics {
    /// Whether the matrix is diagonally dominant.
    ///
    /// A **sufficient structural certificate**, and only that. Strict dominance
    /// implies the matrix is non-singular (Levy-Desplanques), that no pivot can
    /// vanish during an unpivoted elimination, and that the growth factor is
    /// bounded -- so the algorithm is backward stable on it. Backward stability
    /// means the computed answer is the exact answer to a *nearby* system. It does
    /// **not** mean the answer is close to the true one: that additionally needs
    /// the system to be well conditioned, and a weakly dominant, nearly singular
    /// matrix is dominant and ill-conditioned at the same time.
    ///
    /// It is also **not necessary**. Symmetric positive-definite systems,
    /// M-matrices, and many merely well-behaved systems are solved exactly without
    /// it -- `SolvesANonDominantSystemThatIsNonethelessWellBehaved` is one.
    ///
    /// So: true means the elimination cannot break down, false means no such
    /// certificate exists and the caller should look at the other evidence. It
    /// never means "the answer is accurate", in either direction.
    bool diagonally_dominant{};

    /// The smallest |pivot| / row_scale seen during elimination.
    ///
    /// A **pivot-health diagnostic**, not a condition-number estimate and not a
    /// proxy for one. It answers exactly one question: did the elimination come
    /// close to dividing by nothing? A ratio near zero means it did, and the
    /// answer is worthless however small its residual. A healthy ratio means only
    /// that this particular failure did not occur -- an ill-conditioned system can
    /// have entirely healthy pivots and still return an answer whose error is
    /// large.
    ///
    /// Reading it as "the solve is fine" is the mistake this comment exists to
    /// prevent. It rules out one failure mode and reports on no other.
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
/// Callers who want the structural certificate rather than the absence of an
/// observed failure should assert `diagnostics.diagonally_dominant` themselves.
/// The PDE schemes do exactly that on the operators they generate, where the
/// property is theirs to establish. Note what that buys: no pivot breakdown and
/// bounded growth. It does not buy an accurate answer, which depends on
/// conditioning that none of these diagnostics measures.
[[nodiscard]] Result<TridiagonalSolution> solve_tridiagonal(const TridiagonalSystem& system);

/// Whether the matrix is diagonally dominant: weakly in every row, strictly in at
/// least one.
///
/// A sufficient structural certificate for the elimination -- no pivot breakdown,
/// bounded growth -- and not a necessary one, nor a statement about accuracy.
/// Exposed so a caller who generates its own matrices, as the PDE schemes do, can
/// assert the property about them before relying on it.
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
