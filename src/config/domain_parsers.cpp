#include <diffusionworks/config/domain_parsers.hpp>

#include <fmt/format.h>

#include <string>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "config";

}  // namespace

Result<MarketState> parse_market_state(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"spot", "rate", "dividend_yield"});
    if (!unknown) {
        return Result<MarketState>::failure(unknown.error());
    }

    auto spot = node.positive_number("spot");
    if (!spot) {
        return Result<MarketState>::failure(std::move(spot).error());
    }

    // Rates are read with number(), not positive_number(): negative rates are
    // admissible, and the domain type enforces only finiteness.
    auto rate = node.number("rate");
    if (!rate) {
        return Result<MarketState>::failure(std::move(rate).error());
    }

    auto dividend_yield = node.number_or("dividend_yield", 0.0);
    if (!dividend_yield) {
        return Result<MarketState>::failure(std::move(dividend_yield).error());
    }

    return MarketState::create(spot.value(), rate.value(), dividend_yield.value());
}

Result<EuropeanOption> parse_european_option(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"type", "option_type", "strike", "maturity"});
    if (!unknown) {
        return Result<EuropeanOption>::failure(unknown.error());
    }

    auto instrument_type = node.string("type");
    if (!instrument_type) {
        return Result<EuropeanOption>::failure(std::move(instrument_type).error());
    }
    if (instrument_type.value() != "european") {
        return Result<EuropeanOption>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'{}.type' is '{}' but this parser builds European options; expected "
                        "'european'",
                        node.path(),
                        instrument_type.value()),
            kContext);
    }

    auto option_type_text = node.string("option_type");
    if (!option_type_text) {
        return Result<EuropeanOption>::failure(std::move(option_type_text).error());
    }
    const auto option_type = parse_option_type(option_type_text.value());
    if (!option_type.has_value()) {
        return Result<EuropeanOption>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("'{}.option_type' is '{}'; expected 'call' or 'put'",
                        node.path(),
                        option_type_text.value()),
            kContext);
    }

    auto strike = node.positive_number("strike");
    if (!strike) {
        return Result<EuropeanOption>::failure(std::move(strike).error());
    }

    // Read with number() rather than positive_number() so that T = 0 reaches the
    // domain type, which admits it as the expiry limit; number() still rejects a
    // non-finite value, and EuropeanOption rejects a negative one.
    auto maturity = node.number("maturity");
    if (!maturity) {
        return Result<EuropeanOption>::failure(std::move(maturity).error());
    }

    return EuropeanOption::create(*option_type, strike.value(), maturity.value());
}

Result<BlackScholesModel> parse_black_scholes_model(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"type", "volatility"});
    if (!unknown) {
        return Result<BlackScholesModel>::failure(unknown.error());
    }

    auto model_type = node.string("type");
    if (!model_type) {
        return Result<BlackScholesModel>::failure(std::move(model_type).error());
    }
    if (model_type.value() != "black_scholes") {
        return Result<BlackScholesModel>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'{}.type' is '{}' but this parser builds Black-Scholes models; expected "
                        "'black_scholes'",
                        node.path(),
                        model_type.value()),
            kContext);
    }

    // sigma = 0 is the deterministic limit and is admitted by the domain type.
    auto volatility = node.number("volatility");
    if (!volatility) {
        return Result<BlackScholesModel>::failure(std::move(volatility).error());
    }

    return BlackScholesModel::create(volatility.value());
}

}  // namespace diffusionworks
