#include <diffusionworks/models/heston.hpp>

#include <fmt/format.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "HestonModel";

[[nodiscard]] Result<HestonModel> reject(const char* name, double value) {
    return Result<HestonModel>::failure(
        ErrorCode::InvalidArgument,
        fmt::format("{} must be non-negative and finite, got {}", name, value),
        kContext);
}

}  // namespace

Result<HestonModel> HestonModel::create(double initial_variance,
                                        double mean_reversion,
                                        double long_run_variance,
                                        double vol_of_variance,
                                        double correlation) {
    if (!(initial_variance >= 0.0) || !std::isfinite(initial_variance)) {
        return reject("initial_variance v0", initial_variance);
    }
    if (!(mean_reversion >= 0.0) || !std::isfinite(mean_reversion)) {
        return reject("mean_reversion kappa", mean_reversion);
    }
    if (!(long_run_variance >= 0.0) || !std::isfinite(long_run_variance)) {
        return reject("long_run_variance theta", long_run_variance);
    }
    if (!(vol_of_variance >= 0.0) || !std::isfinite(vol_of_variance)) {
        return reject("vol_of_variance xi", vol_of_variance);
    }
    if (!std::isfinite(correlation) || correlation < -1.0 || correlation > 1.0) {
        return Result<HestonModel>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("correlation rho must lie in [-1, 1] and be finite, got {}", correlation),
            kContext);
    }

    return Result<HestonModel>::success(HestonModel(
        initial_variance, mean_reversion, long_run_variance, vol_of_variance, correlation));
}

double HestonModel::feller_ratio() const noexcept {
    if (vol_of_variance_ == 0.0) {
        // Deterministic variance cannot reach zero by diffusion, so the condition is
        // satisfied with infinite margin. Returning infinity rather than dividing by
        // zero states that rather than producing a NaN.
        return std::numeric_limits<double>::infinity();
    }
    return 2.0 * mean_reversion_ * long_run_variance_ / (vol_of_variance_ * vol_of_variance_);
}

bool HestonModel::satisfies_feller() const noexcept {
    return 2.0 * mean_reversion_ * long_run_variance_ >= vol_of_variance_ * vol_of_variance_;
}

}  // namespace diffusionworks
