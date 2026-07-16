#include <diffusionworks/simulation/time_grid.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "TimeGrid";

}  // namespace

Result<TimeGrid> TimeGrid::uniform(double maturity, std::int64_t steps) {
    if (!std::isfinite(maturity)) {
        return Result<TimeGrid>::failure(ErrorCode::InvalidArgument,
                                         fmt::format("maturity must be finite but is {}", maturity),
                                         kContext);
    }
    if (maturity <= 0.0) {
        return Result<TimeGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be strictly positive but is {}; a grid over an instant "
                        "has no steps to take. The analytic engine handles T = 0 exactly.",
                        maturity),
            kContext);
    }
    if (steps < 1) {
        return Result<TimeGrid>::failure(ErrorCode::InvalidArgument,
                                         fmt::format("steps must be at least 1 but is {}", steps),
                                         kContext);
    }

    const double step_size = maturity / static_cast<double>(steps);

    // A step size that rounds to zero would make the loop advance nowhere while
    // appearing to run, which is worse than failing.
    if (step_size <= 0.0) {
        return Result<TimeGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("step size underflowed to {} for maturity {} over {} steps",
                        step_size,
                        maturity,
                        steps),
            kContext);
    }

    return Result<TimeGrid>::success(TimeGrid(maturity, steps, step_size, std::sqrt(step_size)));
}

double TimeGrid::time_at(std::int64_t index) const noexcept {
    // The last point returns maturity exactly rather than the rounding of
    // M*(T/M), so a payoff evaluated at expiry sees the contract's maturity and
    // not a value a few ulps away from it.
    if (index >= steps_) {
        return maturity_;
    }
    if (index <= 0) {
        return 0.0;
    }
    // i*T/M rather than i*dt: accumulating dt would drift over 10^6 steps, and
    // multiplying by the index does not.
    return static_cast<double>(index) * maturity_ / static_cast<double>(steps_);
}

}  // namespace diffusionworks
