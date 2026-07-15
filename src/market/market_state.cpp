#include <diffusionworks/market/market_state.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "MarketState";

}  // namespace

Result<MarketState> MarketState::create(double spot, double rate, double dividend_yield) {
    if (!std::isfinite(spot)) {
        return Result<MarketState>::failure(ErrorCode::InvalidArgument,
                                            fmt::format("spot must be finite but is {}", spot),
                                            kContext);
    }
    if (spot <= 0.0) {
        // Geometric Brownian motion cannot reach or cross zero, and log(S) is
        // undefined there; a non-positive spot is outside the model's domain
        // rather than merely an unusual input.
        return Result<MarketState>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("spot must be strictly positive but is {}", spot),
            kContext);
    }
    if (!std::isfinite(rate)) {
        return Result<MarketState>::failure(ErrorCode::InvalidArgument,
                                            fmt::format("rate must be finite but is {}", rate),
                                            kContext);
    }
    if (!std::isfinite(dividend_yield)) {
        return Result<MarketState>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("dividend_yield must be finite but is {}", dividend_yield),
            kContext);
    }

    return Result<MarketState>::success(MarketState(spot, rate, dividend_yield));
}

double MarketState::discount_factor(double maturity) const noexcept {
    return std::exp(-rate_ * maturity);
}

double MarketState::dividend_discount_factor(double maturity) const noexcept {
    return std::exp(-dividend_yield_ * maturity);
}

double MarketState::forward(double maturity) const noexcept {
    return spot_ * std::exp((rate_ - dividend_yield_) * maturity);
}

}  // namespace diffusionworks
