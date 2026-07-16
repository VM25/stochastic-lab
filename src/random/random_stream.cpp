#include <diffusionworks/random/random_stream.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "CorrelationCoefficient";

}  // namespace

Result<CorrelationCoefficient> CorrelationCoefficient::create(double rho) {
    if (!std::isfinite(rho)) {
        return Result<CorrelationCoefficient>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("correlation must be finite but is {}", rho),
            kContext);
    }
    if (rho < -1.0 || rho > 1.0) {
        // Rejected rather than clamped. An unchecked correlate() given rho = 2
        // would evaluate sqrt(1 - 4), clamp the negative argument to zero, and
        // return a plausible pair carrying a correlation of 1 -- a wrong number
        // that looks entirely right, which is the failure mode this project
        // exists to distrust.
        return Result<CorrelationCoefficient>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("correlation must lie in [-1, 1] but is {}", rho),
            kContext);
    }

    // Clamped at zero only against rounding: 1 - rho*rho can come out marginally
    // negative at |rho| = 1, where the true complement is exactly zero. The
    // excursion is bounded by an ulp, unlike the rho = 2 case above, which is a
    // caller error rather than rounding -- hence one is clamped and the other
    // refused.
    const double complement = std::sqrt(std::max(0.0, 1.0 - rho * rho));

    return Result<CorrelationCoefficient>::success(CorrelationCoefficient(rho, complement));
}

}  // namespace diffusionworks
