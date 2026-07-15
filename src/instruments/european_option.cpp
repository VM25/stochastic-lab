#include <diffusionworks/instruments/european_option.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "EuropeanOption";

}  // namespace

std::string_view to_string(OptionType type) noexcept {
    switch (type) {
        case OptionType::Call:
            return "call";
        case OptionType::Put:
            return "put";
    }
    return "unknown";
}

std::optional<OptionType> parse_option_type(std::string_view text) noexcept {
    if (text == "call") {
        return OptionType::Call;
    }
    if (text == "put") {
        return OptionType::Put;
    }
    return std::nullopt;
}

Result<EuropeanOption> EuropeanOption::create(OptionType type, double strike, double maturity) {
    if (!std::isfinite(strike)) {
        return Result<EuropeanOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("strike must be finite but is {}", strike),
            kContext);
    }
    if (strike <= 0.0) {
        return Result<EuropeanOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("strike must be strictly positive but is {}", strike),
            kContext);
    }
    if (!std::isfinite(maturity)) {
        return Result<EuropeanOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be finite but is {}", maturity),
            kContext);
    }
    if (maturity < 0.0) {
        // Zero is admitted (an expired option is worth its intrinsic value);
        // negative time is not a limiting case, it is nonsense.
        return Result<EuropeanOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be non-negative but is {}", maturity),
            kContext);
    }

    return Result<EuropeanOption>::success(EuropeanOption(type, strike, maturity));
}

double EuropeanOption::payoff(double terminal_spot) const noexcept {
    switch (type_) {
        case OptionType::Call:
            return std::max(terminal_spot - strike_, 0.0);
        case OptionType::Put:
            return std::max(strike_ - terminal_spot, 0.0);
    }
    return 0.0;
}

}  // namespace diffusionworks
