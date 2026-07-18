#include <diffusionworks/calibration/parameter_transform.hpp>

#include <array>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "HestonParameterTransform";

/// log((p - lo) / (hi - p)). Requires lo < p < hi.
[[nodiscard]] double logit(double value, double lower, double upper) {
    return std::log((value - lower) / (upper - value));
}

/// lo + (hi - lo) / (1 + e^{-x}). Total in x; the result is strictly inside (lo, hi).
[[nodiscard]] double inverse_logit(double x, double lower, double upper) {
    // Branch on the sign of x so the exponential never overflows: both forms are the
    // logistic function, evaluated on whichever side keeps the argument non-positive.
    const double s = x >= 0.0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
    return lower + (upper - lower) * s;
}

}  // namespace

Result<HestonModel> HestonParameters::to_model() const {
    return HestonModel::create(
        initial_variance, mean_reversion, long_run_variance, vol_of_variance, correlation);
}

HestonParameterBounds HestonParameterBounds::defaults() {
    return HestonParameterBounds{.lower = {.initial_variance = 1.0e-6,
                                           .mean_reversion = 1.0e-3,
                                           .long_run_variance = 1.0e-6,
                                           .vol_of_variance = 1.0e-3,
                                           .correlation = -0.999},
                                 .upper = {.initial_variance = 1.0,
                                           .mean_reversion = 20.0,
                                           .long_run_variance = 1.0,
                                           .vol_of_variance = 5.0,
                                           .correlation = 0.999}};
}

bool HestonParameterBounds::valid() const noexcept {
    return lower.initial_variance < upper.initial_variance &&
           lower.mean_reversion < upper.mean_reversion &&
           lower.long_run_variance < upper.long_run_variance &&
           lower.vol_of_variance < upper.vol_of_variance && lower.correlation < upper.correlation;
}

bool HestonParameterBounds::contains(const HestonParameters& p) const noexcept {
    return p.initial_variance > lower.initial_variance &&
           p.initial_variance < upper.initial_variance && p.mean_reversion > lower.mean_reversion &&
           p.mean_reversion < upper.mean_reversion &&
           p.long_run_variance > lower.long_run_variance &&
           p.long_run_variance < upper.long_run_variance &&
           p.vol_of_variance > lower.vol_of_variance && p.vol_of_variance < upper.vol_of_variance &&
           p.correlation > lower.correlation && p.correlation < upper.correlation;
}

Result<std::array<double, 5>> to_unconstrained(const HestonParameters& parameters,
                                               const HestonParameterBounds& bounds) {
    if (!bounds.valid()) {
        return Result<std::array<double, 5>>::failure(
            ErrorCode::InvalidArgument,
            "the parameter bounds are inside-out: every lower bound must be below its upper",
            kContext);
    }
    if (!bounds.contains(parameters)) {
        return Result<std::array<double, 5>>::failure(
            ErrorCode::InvalidArgument,
            "the parameters lie on or outside the bounds, so they have no finite unconstrained "
            "image; move the starting point strictly inside the box",
            kContext);
    }
    return Result<std::array<double, 5>>::success(std::array<double, 5>{
        logit(parameters.initial_variance,
              bounds.lower.initial_variance,
              bounds.upper.initial_variance),
        logit(parameters.mean_reversion, bounds.lower.mean_reversion, bounds.upper.mean_reversion),
        logit(parameters.long_run_variance,
              bounds.lower.long_run_variance,
              bounds.upper.long_run_variance),
        logit(
            parameters.vol_of_variance, bounds.lower.vol_of_variance, bounds.upper.vol_of_variance),
        logit(parameters.correlation, bounds.lower.correlation, bounds.upper.correlation)});
}

HestonParameters to_constrained(const std::array<double, 5>& point,
                                const HestonParameterBounds& bounds) {
    return HestonParameters{
        .initial_variance =
            inverse_logit(point[0], bounds.lower.initial_variance, bounds.upper.initial_variance),
        .mean_reversion =
            inverse_logit(point[1], bounds.lower.mean_reversion, bounds.upper.mean_reversion),
        .long_run_variance =
            inverse_logit(point[2], bounds.lower.long_run_variance, bounds.upper.long_run_variance),
        .vol_of_variance =
            inverse_logit(point[3], bounds.lower.vol_of_variance, bounds.upper.vol_of_variance),
        .correlation = inverse_logit(point[4], bounds.lower.correlation, bounds.upper.correlation)};
}

}  // namespace diffusionworks
