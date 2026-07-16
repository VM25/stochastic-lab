#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace diffusionworks {

/// Right conveyed by an option contract.
enum class OptionType : std::uint8_t {
    Call,
    Put,
};

[[nodiscard]] std::string_view to_string(OptionType type) noexcept;

[[nodiscard]] std::optional<OptionType> parse_option_type(std::string_view text) noexcept;

/// A European call or put.
///
/// Contract terms only. Per ADR-007 an instrument carries no valuation logic:
/// pricing lives in engines, so that adding a model or a method does not touch
/// the instrument.
class EuropeanOption {
public:
    /// Validates and constructs.
    ///
    /// Requires strike > 0 and finite, and maturity >= 0 and finite.
    ///
    /// Maturity of exactly zero is admitted rather than rejected: T -> 0 is a
    /// required limiting case (MATHEMATICAL-SPEC section 18), and an expired
    /// option has a well-defined value, namely its intrinsic payoff. Engines
    /// handle it explicitly.
    [[nodiscard]] static Result<EuropeanOption>
    create(OptionType type, double strike, double maturity);

    [[nodiscard]] OptionType type() const noexcept { return type_; }

    [[nodiscard]] double strike() const noexcept { return strike_; }

    [[nodiscard]] double maturity() const noexcept { return maturity_; }

    /// Terminal payoff \f$ (S_T-K)^+ \f$ for a call, \f$ (K-S_T)^+ \f$ for a put.
    [[nodiscard]] double payoff(double terminal_spot) const noexcept;

private:
    EuropeanOption(OptionType type, double strike, double maturity) noexcept
        : type_(type), strike_(strike), maturity_(maturity) {}

    OptionType type_;
    double strike_;
    double maturity_;
};

}  // namespace diffusionworks
