#include <diffusionworks/models/black_scholes.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BlackScholesModel";

}  // namespace

Result<BlackScholesModel> BlackScholesModel::create(double volatility) {
    if (!std::isfinite(volatility)) {
        return Result<BlackScholesModel>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("volatility must be finite but is {}", volatility),
            kContext);
    }
    if (volatility < 0.0) {
        // Zero is admitted as the deterministic limit; negative volatility has
        // no meaning, since only sigma^2 enters the dynamics and the sign would
        // silently vanish.
        return Result<BlackScholesModel>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("volatility must be non-negative but is {}", volatility),
            kContext);
    }

    return Result<BlackScholesModel>::success(BlackScholesModel(volatility));
}

double BlackScholesModel::total_volatility(double maturity) const noexcept {
    return volatility_ * std::sqrt(maturity);
}

}  // namespace diffusionworks
