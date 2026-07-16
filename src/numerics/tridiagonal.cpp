#include <diffusionworks/numerics/tridiagonal.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "solve_tridiagonal";

/// A pivot at or below this fraction of its row's scale is treated as vanished.
///
/// Relative, not absolute. The Black-Scholes operator's coefficients scale with
/// sigma^2 S^2 / dS^2 and span orders of magnitude across a grid, so a fixed
/// absolute floor would be meaningless at one end or the other.
///
/// The threshold does not decide correctness -- it decides where an answer stops
/// being worth returning. Below it the elimination has divided by something
/// indistinguishable from zero, and the result, though finite, carries no
/// information. `smallest_pivot_ratio` is reported regardless, so a caller can see
/// how close a solve came without having crossed the line.
constexpr double kPivotFloor = 1e-14;

Status validate(const TridiagonalSystem& system) {
    const std::size_t n = system.diagonal.size();
    if (n == 0) {
        return Status::failure(ErrorCode::InvalidArgument, "the system is empty", kContext);
    }
    if (system.lower.size() != n || system.upper.size() != n || system.rhs.size() != n) {
        // Refused rather than indexed defensively. Diagonals of different lengths
        // mean the caller's index does not mean the same thing in each, and a
        // solver that tolerates that reads the wrong coefficient while staying
        // safely in bounds -- producing a price rather than a crash.
        return Status::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the diagonals must have equal length, got lower={}, diagonal={}, "
                        "upper={}, rhs={}",
                        system.lower.size(),
                        n,
                        system.upper.size(),
                        system.rhs.size()),
            kContext);
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(system.lower[i]) || !std::isfinite(system.diagonal[i]) ||
            !std::isfinite(system.upper[i]) || !std::isfinite(system.rhs[i])) {
            return Status::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("row {} has a non-finite coefficient (lower={}, diagonal={}, upper={}, "
                            "rhs={})",
                            i,
                            system.lower[i],
                            system.diagonal[i],
                            system.upper[i],
                            system.rhs[i]),
                kContext);
        }
    }
    return Status::success();
}

}  // namespace

bool is_diagonally_dominant(const TridiagonalSystem& system) noexcept {
    const std::size_t n = system.diagonal.size();
    if (n == 0 || system.lower.size() != n || system.upper.size() != n) {
        return false;
    }

    bool strict_somewhere = false;
    for (std::size_t i = 0; i < n; ++i) {
        // The corner entries are not part of the matrix: row 0 has no sub-diagonal
        // and row n-1 has no super-diagonal. Including them would judge dominance
        // against coefficients that never multiply anything.
        const double off = (i == 0 ? 0.0 : std::abs(system.lower[i])) +
                           (i + 1 == n ? 0.0 : std::abs(system.upper[i]));
        const double diagonal = std::abs(system.diagonal[i]);

        if (diagonal < off) {
            return false;
        }
        if (diagonal > off) {
            strict_somewhere = true;
        }
    }
    return strict_somewhere;
}

Result<TridiagonalSolution> solve_tridiagonal(const TridiagonalSystem& system) {
    const Status valid = validate(system);
    if (!valid) {
        return Result<TridiagonalSolution>::failure(valid.error());
    }

    const std::size_t n = system.size();
    std::vector<double> modified_upper(n, 0.0);
    std::vector<double> modified_rhs(n, 0.0);

    // A pivot is "vanished" relative to the coefficients that produced it, not
    // relative to 1.
    const auto row_scale = [&](std::size_t i) {
        return std::max({std::abs(system.lower[i]),
                         std::abs(system.diagonal[i]),
                         std::abs(system.upper[i]),
                         1.0});
    };

    TridiagonalDiagnostics diagnostics;
    diagnostics.diagonally_dominant = is_diagonally_dominant(system);
    diagnostics.smallest_pivot_ratio = std::numeric_limits<double>::infinity();

    const auto check_pivot = [&](double pivot, std::size_t i) -> Status {
        const double ratio = std::abs(pivot) / row_scale(i);
        diagnostics.smallest_pivot_ratio = std::min(diagnostics.smallest_pivot_ratio, ratio);
        if (ratio <= kPivotFloor) {
            return Status::failure(
                ErrorCode::NonFiniteValue,
                fmt::format(
                    "the pivot at row {} vanished ({} against a row scale of {}, ratio {:.3e}). "
                    "The Thomas algorithm does not pivot, so the elimination cannot recover: "
                    "dividing by this would return a finite, arbitrarily wrong answer rather than "
                    "an obviously broken one. The matrix is {}diagonally dominant.",
                    i,
                    pivot,
                    row_scale(i),
                    ratio,
                    diagnostics.diagonally_dominant ? "" : "not "),
                kContext);
        }
        return Status::success();
    };

    // Forward elimination.
    double pivot = system.diagonal[0];
    if (const Status ok = check_pivot(pivot, 0); !ok) {
        return Result<TridiagonalSolution>::failure(ok.error());
    }
    modified_upper[0] = system.upper[0] / pivot;
    modified_rhs[0] = system.rhs[0] / pivot;

    for (std::size_t i = 1; i < n; ++i) {
        pivot = system.diagonal[i] - system.lower[i] * modified_upper[i - 1];
        if (const Status ok = check_pivot(pivot, i); !ok) {
            return Result<TridiagonalSolution>::failure(ok.error());
        }
        modified_upper[i] = system.upper[i] / pivot;
        modified_rhs[i] = (system.rhs[i] - system.lower[i] * modified_rhs[i - 1]) / pivot;
    }

    // Back substitution.
    std::vector<double> solution(n, 0.0);
    solution[n - 1] = modified_rhs[n - 1];
    for (std::size_t i = n - 1; i > 0; --i) {
        solution[i - 1] = modified_rhs[i - 1] - modified_upper[i - 1] * solution[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(solution[i])) {
            // Reachable with every pivot above the floor: the growth factor of an
            // unpivoted elimination is unbounded on a general matrix, so the
            // intermediate quantities can overflow even though no single division
            // was by zero.
            return Result<TridiagonalSolution>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("the solution is not finite at index {} ({}), so the elimination grew "
                            "without bound despite finite inputs and non-vanishing pivots "
                            "(smallest pivot ratio {:.3e}, matrix {}diagonally dominant)",
                            i,
                            solution[i],
                            diagnostics.smallest_pivot_ratio,
                            diagnostics.diagonally_dominant ? "" : "not "),
                kContext);
        }
    }

    return Result<TridiagonalSolution>::success(
        TridiagonalSolution{.values = std::move(solution), .diagnostics = diagnostics});
}

Result<double> tridiagonal_residual(const TridiagonalSystem& system,
                                    std::span<const double> solution) {
    const Status valid = validate(system);
    if (!valid) {
        return Result<double>::failure(valid.error());
    }
    if (solution.size() != system.size()) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the solution has {} entries but the system has {} rows",
                        solution.size(),
                        system.size()),
            "tridiagonal_residual");
    }

    const std::size_t n = system.size();
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        // Rebuilt from the original diagonals, deliberately. Reusing anything the
        // elimination computed would let a mistake cancel itself.
        double row = system.diagonal[i] * solution[i];
        if (i > 0) {
            row += system.lower[i] * solution[i - 1];
        }
        if (i + 1 < n) {
            row += system.upper[i] * solution[i + 1];
        }
        worst = std::max(worst, std::abs(row - system.rhs[i]));
    }
    return Result<double>::success(worst);
}

}  // namespace diffusionworks
