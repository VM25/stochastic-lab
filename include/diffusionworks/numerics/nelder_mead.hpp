#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace diffusionworks {

/// Why a Nelder-Mead run stopped.
enum class NelderMeadStatus : std::uint8_t {
    /// Both the spread of vertex values and the simplex size fell below tolerance.
    Converged,

    /// The iteration budget ran out first. The best point so far is returned, but the
    /// caller must treat it as unconverged -- this is the distinction the calibration
    /// gate turns on, so it is a status, not a footnote.
    MaxIterationsReached,
};

[[nodiscard]] const char* to_string(NelderMeadStatus status) noexcept;

/// Configuration for the simplex search. The coefficients are the classical
/// Nelder-Mead values; they are exposed so a study can vary them, not because they
/// need tuning for ordinary use.
struct NelderMeadConfig {
    int max_iterations{2000};

    /// Converged when the range of vertex objective values is below this.
    double function_tolerance{1.0e-10};

    /// And when the simplex has contracted below this in point space. Both are
    /// required: a flat objective with a large simplex has not converged, and a small
    /// simplex straddling a cliff has not either.
    double simplex_tolerance{1.0e-8};

    /// The initial simplex is the start point plus this step along each axis. A
    /// coordinate that is (near) zero is stepped by this absolute amount so the
    /// simplex is never degenerate.
    double initial_step{0.25};

    double reflection{1.0};
    double expansion{2.0};
    double contraction{0.5};
    double shrink{0.5};
};

/// The outcome of a simplex search.
struct NelderMeadResult {
    std::vector<double> point;
    double value{};
    int iterations{};

    /// Objective evaluations, the honest cost of the search and the quantity EXP-11
    /// reports rather than iteration count alone.
    int function_evaluations{};

    NelderMeadStatus status{NelderMeadStatus::MaxIterationsReached};
};

/// Minimises a real objective over R^n by the Nelder-Mead downhill simplex method.
///
/// Derivative-free and fully deterministic: the same objective and starting point
/// always produce the same path, so a calibration is reproducible from its
/// configuration. A non-finite objective value is treated as +infinity, so the
/// simplex retreats from an infeasible or non-priceable region rather than being
/// derailed by a NaN.
///
/// Fails only on a malformed request (an empty start, or non-positive tolerances). A
/// search that does not converge within the budget is not a failure -- it returns the
/// best point with status MaxIterationsReached, because "did not converge" is a result
/// the caller must be able to see and report, not an error to swallow.
[[nodiscard]] Result<NelderMeadResult>
nelder_mead(const std::function<double(std::span<const double>)>& objective,
            std::span<const double> initial,
            const NelderMeadConfig& config = {});

}  // namespace diffusionworks
