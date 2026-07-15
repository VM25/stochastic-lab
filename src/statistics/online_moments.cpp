#include <diffusionworks/statistics/distributions.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "statistics";

}  // namespace

Result<double> OnlineMoments::sample_variance() const {
    if (count_ < 2) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("sample variance is undefined for {} observation(s); at least 2 are "
                        "required",
                        count_),
            kContext);
    }

    const double variance = sum_squared_deviations_ / static_cast<double>(count_ - 1);

    // Welford's recurrence keeps the accumulator non-negative in exact
    // arithmetic, so a negative value here means rounding has overwhelmed the
    // computation. Reporting it is the point: clamping to zero would turn a
    // broken variance into a confident interval of zero width.
    if (variance < 0.0) {
        return Result<double>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("sample variance is negative ({}) after {} observations, which exact "
                        "arithmetic cannot produce; the accumulation has lost precision",
                        variance,
                        count_),
            kContext);
    }
    if (!std::isfinite(variance)) {
        return Result<double>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("sample variance is not finite after {} observations", count_),
            kContext);
    }

    return Result<double>::success(variance);
}

Result<double> OnlineMoments::standard_error() const {
    auto variance = sample_variance();
    if (!variance) {
        return variance;
    }
    return Result<double>::success(std::sqrt(variance.value() / static_cast<double>(count_)));
}

Result<ConfidenceInterval> OnlineMoments::confidence_interval(double level) const {
    // Negated conjunction rather than the DeMorgan form, so that a NaN level is
    // rejected: every IEEE comparison against NaN is false, which makes
    // `level <= 0.0 || level >= 1.0` false for NaN and would let it through.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(level > 0.0 && level < 1.0)) {
        return Result<ConfidenceInterval>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("confidence level must lie strictly inside (0, 1) but is {}", level),
            kContext);
    }

    auto error = standard_error();
    if (!error) {
        return Result<ConfidenceInterval>::failure(std::move(error).error());
    }

    const auto degrees_of_freedom = static_cast<double>(count_ - 1);
    const double upper_tail = 1.0 - 0.5 * (1.0 - level);

    auto critical = student_t_quantile(upper_tail, degrees_of_freedom);
    if (!critical) {
        return Result<ConfidenceInterval>::failure(std::move(critical).error());
    }

    const double half_width = critical.value() * error.value();
    return Result<ConfidenceInterval>::success(ConfidenceInterval{
        .lower = mean_ - half_width, .upper = mean_ + half_width, .level = level});
}

Result<double> OnlineCovariance::covariance() const {
    if (count_ < 2) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("sample covariance is undefined for {} observation(s); at least 2 are "
                        "required",
                        count_),
            kContext);
    }

    const double value = sum_co_deviations_ / static_cast<double>(count_ - 1);
    if (!std::isfinite(value)) {
        return Result<double>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("sample covariance is not finite after {} observations", count_),
            kContext);
    }
    return Result<double>::success(value);
}

Result<double> OnlineCovariance::correlation() const {
    auto cov = covariance();
    if (!cov) {
        return cov;
    }

    auto variance_x = moments_x_.sample_variance();
    if (!variance_x) {
        return variance_x;
    }
    auto variance_y = moments_y_.sample_variance();
    if (!variance_y) {
        return variance_y;
    }

    if (variance_x.value() == 0.0 || variance_y.value() == 0.0) {
        // A constant series has no direction, so there is nothing for the other
        // to align with. Returning 0 would assert independence that has not been
        // established.
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("correlation is undefined because a marginal variance is zero "
                        "(var_x={}, var_y={})",
                        variance_x.value(),
                        variance_y.value()),
            kContext);
    }

    const double denominator = std::sqrt(variance_x.value() * variance_y.value());
    const double value = cov.value() / denominator;

    // Rounding can push the ratio a few ulps outside [-1, 1]. Clamping is
    // legitimate here, unlike clamping a variance: the excursion is bounded by
    // rounding rather than symptomatic of a broken accumulation. Anything beyond
    // a small tolerance is a real defect and is reported.
    constexpr double kRoundingSlack = 1e-9;
    if (value > 1.0 + kRoundingSlack || value < -1.0 - kRoundingSlack) {
        return Result<double>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("correlation is {} which lies outside [-1, 1] by more than rounding can "
                        "explain",
                        value),
            kContext);
    }

    return Result<double>::success(std::clamp(value, -1.0, 1.0));
}

}  // namespace diffusionworks
