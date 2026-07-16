#include <diffusionworks/instruments/asian_option.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "AsianOption";

}  // namespace

std::string_view to_string(AveragingType averaging) noexcept {
    switch (averaging) {
        case AveragingType::Arithmetic:
            return "arithmetic";
        case AveragingType::Geometric:
            return "geometric";
    }
    return "unknown";
}

std::optional<AveragingType> parse_averaging_type(std::string_view text) noexcept {
    if (text == "arithmetic") {
        return AveragingType::Arithmetic;
    }
    if (text == "geometric") {
        return AveragingType::Geometric;
    }
    return std::nullopt;
}

Result<AsianOption> AsianOption::create(OptionType type,
                                        AveragingType averaging,
                                        double strike,
                                        double maturity,
                                        std::int64_t monitoring_count) {
    if (!std::isfinite(strike) || strike <= 0.0) {
        return Result<AsianOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("strike must be finite and strictly positive but is {}", strike),
            kContext);
    }
    if (!std::isfinite(maturity) || maturity <= 0.0) {
        return Result<AsianOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be finite and strictly positive but is {}; an average "
                        "spanning zero time is not a limiting case, it is a contract that does "
                        "not exist",
                        maturity),
            kContext);
    }
    if (monitoring_count < 1) {
        return Result<AsianOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("monitoring_count must be at least 1 but is {}", monitoring_count),
            kContext);
    }

    return Result<AsianOption>::success(
        AsianOption(type, averaging, strike, maturity, monitoring_count));
}

double AsianOption::monitoring_time(std::int64_t index) const noexcept {
    if (index >= monitoring_count_) {
        // The last monitoring date is maturity exactly, not the rounding of
        // M*(T/M).
        return maturity_;
    }
    if (index <= 0) {
        // Index 0 is not a monitoring date under this convention; the first is
        // index 1. Returning t_1 keeps the function total rather than inventing a
        // date at zero, which the average deliberately excludes.
        return maturity_ / static_cast<double>(monitoring_count_);
    }
    return static_cast<double>(index) * maturity_ / static_cast<double>(monitoring_count_);
}

double AsianOption::payoff(double average) const noexcept {
    switch (type_) {
        case OptionType::Call:
            return std::max(average - strike_, 0.0);
        case OptionType::Put:
            return std::max(strike_ - average, 0.0);
    }
    return 0.0;
}

}  // namespace diffusionworks
