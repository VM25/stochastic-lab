#pragma once

#include <diffusionworks/core/interval.hpp>
#include <diffusionworks/core/result.hpp>

#include <cstdint>
#include <span>

namespace diffusionworks {

/// A fitted straight line, with the uncertainty of its coefficients.
///
/// The uncertainty is the point. FAILURE-MODES section 6 names a convergence
/// order judged by eye as a completion blocker, and a slope reported without an
/// interval cannot be compared against a theoretical order at all: "0.48" and
/// "0.48 +/- 0.15" support entirely different conclusions about whether a scheme
/// achieves order 0.5.
struct LinearFit {
    double slope{};
    double intercept{};

    /// Standard errors of the coefficients, from the residual variance.
    double slope_standard_error{};
    double intercept_standard_error{};

    /// Fraction of variance explained. Reported as a diagnostic, not as evidence:
    /// a high R^2 on four points says little, and a convergence plot with a
    /// curved trend can still fit a line well.
    double r_squared{};

    /// Residual standard deviation, in the units of y.
    double residual_standard_deviation{};

    std::uint64_t observations{};

    /// Degrees of freedom, n - 2.
    [[nodiscard]] double degrees_of_freedom() const noexcept {
        return static_cast<double>(observations) - 2.0;
    }

    /// Two-sided confidence interval for the slope at `level`.
    ///
    /// Student-t rather than normal: a convergence study has a handful of grid
    /// levels, which is exactly the regime where the normal approximation
    /// under-covers. At 4 points there are 2 degrees of freedom and the t
    /// critical value is 4.30 against the normal's 1.96 -- a factor of two in the
    /// width, and the difference between a slope that is consistent with 0.5 and
    /// one that appears to exclude it.
    [[nodiscard]] Result<ConfidenceInterval> slope_interval(double level = 0.95) const;
};

/// Fits y = intercept + slope*x by ordinary least squares.
///
/// Requires at least three points. Two points define a line exactly, leaving no
/// residual and no way to estimate uncertainty; reporting a slope from two points
/// with a zero standard error would claim certainty that the data cannot support.
/// EXP-02 requires several grid levels for the same reason.
///
/// Fails if x has no spread, where the slope is undefined rather than infinite.
[[nodiscard]] Result<LinearFit> ordinary_least_squares(std::span<const double> x,
                                                       std::span<const double> y);

/// Fits a power law y = C x^k by regressing log y on log x.
///
/// The natural form for a convergence study: an O(h^k) error is a straight line
/// of slope k on log-log axes, so the fitted slope *is* the empirical order.
///
/// Requires every x and y to be strictly positive. A non-positive value has no
/// logarithm, and silently dropping it would fit a different data set than the
/// one supplied -- most likely dropping exactly the grid level where the error
/// underflowed, which is information rather than noise.
[[nodiscard]] Result<LinearFit> fit_power_law(std::span<const double> x, std::span<const double> y);

}  // namespace diffusionworks
