#include <diffusionworks/statistics/distributions.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <fmt/format.h>

#include <cmath>
#include <limits>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "ordinary_least_squares";

/// The noise floor of the centred sum of squares, below which the x values carry
/// no spread that survived rounding.
///
/// Scaled by the *representation* error of the abscissae, not by an arbitrary
/// relative epsilon. Each deviation x[i] - x_mean carries absolute error about
/// eps*|x_mean|, so Sxx carries about n*(eps*|x_mean|)^2 of pure noise. Anything
/// at or below that is indistinguishable from zero spread.
///
/// The obvious-looking alternative -- a fixed relative tolerance times the mean
/// squared -- is wrong by many orders of magnitude: at x ~ 1e8 a 1e-12 relative
/// threshold evaluates to 1e4 and rejects perfectly well-determined data whose
/// true Sxx is 10.
[[nodiscard]] double spread_noise_floor(double n, double x_mean) noexcept {
    const double representation_error = std::numeric_limits<double>::epsilon() * std::abs(x_mean);
    return n * representation_error * representation_error;
}

}  // namespace

Result<ConfidenceInterval> LinearFit::slope_interval(double level) const {
    // DeMorgan does not hold for IEEE-754, so the negated form below is the guard
    // rather than a clumsy spelling of one. Every comparison against NaN is false,
    // which makes !(x > 0 && x < 1) *true* for NaN and rejects it, while the
    // "simplified" x <= 0 || x >= 1 is false and would let NaN through -- into a
    // Student-t quantile, and back out as a NaN interval wrapped around a real
    // slope. RegressionTest.RejectsInvalidConfidenceLevel fails if anyone applies
    // the transform.
    //
    // The directive sits immediately above the statement it suppresses: with the
    // explanation in between it would land on a comment and do nothing, which is
    // how this check came back after it was first suppressed.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(level > 0.0 && level < 1.0)) {
        return Result<ConfidenceInterval>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("confidence level must lie in (0, 1), got {}", level),
            "LinearFit::slope_interval");
    }
    if (observations < 3) {
        return Result<ConfidenceInterval>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a slope interval needs at least 3 observations, got {}", observations),
            "LinearFit::slope_interval");
    }

    // Student-t, not normal. See the header: with a handful of grid levels the
    // normal quantile produces an interval that is too narrow to be honest.
    const auto critical = student_t_quantile(0.5 * (1.0 + level), degrees_of_freedom());
    if (!critical.ok()) {
        return Result<ConfidenceInterval>::failure(critical.error());
    }

    const double half_width = critical.value() * slope_standard_error;
    return Result<ConfidenceInterval>::success(ConfidenceInterval{
        .lower = slope - half_width, .upper = slope + half_width, .level = level});
}

Result<LinearFit> ordinary_least_squares(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
        return Result<LinearFit>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("x and y must have equal length, got {} and {}", x.size(), y.size()),
            kContext);
    }
    if (x.size() < 3) {
        // Two points fit a line exactly: zero residual, and a standard error of
        // zero that would claim perfect certainty from no evidence. Refused
        // rather than returned, because the resulting slope would look like the
        // best-determined one in the study.
        return Result<LinearFit>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a fit with uncertainty needs at least 3 points, got {}. Two points "
                        "determine a line exactly and leave no residual from which to estimate "
                        "the standard error.",
                        x.size()),
            kContext);
    }

    const auto n = static_cast<double>(x.size());

    double x_mean = 0.0;
    double y_mean = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            return Result<LinearFit>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("observation {} is not finite (x={}, y={})", i, x[i], y[i]),
                kContext);
        }
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= n;
    y_mean /= n;

    // Centred sums. Computing Sxx as sum(x^2) - n*x_mean^2 is algebraically the
    // same and numerically much worse: for log-spaced abscissae the two terms
    // nearly cancel.
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    double sum_yy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - x_mean;
        const double dy = y[i] - y_mean;
        sum_xx += dx * dx;
        sum_xy += dx * dy;
        sum_yy += dy * dy;
    }

    if (sum_xx <= spread_noise_floor(n, x_mean)) {
        return Result<LinearFit>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the x values have no spread that survived rounding (all near {}, centred "
                        "sum of squares {} at or below the noise floor {}), so no slope is "
                        "determined",
                        x_mean,
                        sum_xx,
                        spread_noise_floor(n, x_mean)),
            kContext);
    }

    LinearFit fit;
    fit.observations = x.size();
    fit.slope = sum_xy / sum_xx;
    fit.intercept = y_mean - fit.slope * x_mean;

    // Residual sum of squares, accumulated directly rather than as
    // sum_yy - slope*sum_xy. The identity holds in exact arithmetic but cancels
    // badly on a near-perfect fit -- which a clean convergence study is -- and can
    // yield a small negative number whose square root is NaN.
    double residual_sum_squares = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double residual = y[i] - (fit.intercept + fit.slope * x[i]);
        residual_sum_squares += residual * residual;
    }

    const double residual_variance = residual_sum_squares / (n - 2.0);
    fit.residual_standard_deviation = std::sqrt(residual_variance);

    fit.slope_standard_error = std::sqrt(residual_variance / sum_xx);
    fit.intercept_standard_error =
        std::sqrt(residual_variance * (1.0 / n + x_mean * x_mean / sum_xx));

    // R^2 is undefined when y is constant: the ratio is 0/0, and "all variance
    // explained" and "no variance to explain" are not the same statement. Zero
    // spread in y with nonzero spread in x means a flat line, which explains
    // nothing, so 0 is the defensible value.
    fit.r_squared = sum_yy > 0.0 ? 1.0 - residual_sum_squares / sum_yy : 0.0;

    return Result<LinearFit>::success(fit);
}

Result<LinearFit> fit_power_law(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
        return Result<LinearFit>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("x and y must have equal length, got {} and {}", x.size(), y.size()),
            "fit_power_law");
    }

    std::vector<double> log_x;
    std::vector<double> log_y;
    log_x.reserve(x.size());
    log_y.reserve(y.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
        // Refused rather than filtered. A non-positive error value at one grid
        // level is a finding -- an underflow, or a scheme that happens to be
        // exact there -- and dropping it would quietly fit a different study than
        // the one that was run.
        if (!(x[i] > 0.0) || !(y[i] > 0.0)) {
            return Result<LinearFit>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("a power-law fit needs strictly positive data, but observation {} is "
                            "(x={}, y={}). A non-positive value has no logarithm; it is reported "
                            "rather than dropped, because at which grid level it occurred is "
                            "itself a result.",
                            i,
                            x[i],
                            y[i]),
                "fit_power_law");
        }
        log_x.push_back(std::log(x[i]));
        log_y.push_back(std::log(y[i]));
    }

    return ordinary_least_squares(log_x, log_y);
}

}  // namespace diffusionworks
