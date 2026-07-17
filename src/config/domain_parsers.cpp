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

Result<AsianOption> parse_asian_option(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys(
        {"type", "option_type", "averaging", "strike", "maturity", "monitoring_count"});
    if (!unknown) {
        return Result<AsianOption>::failure(unknown.error());
    }

    auto instrument_type = node.string("type");
    if (!instrument_type) {
        return Result<AsianOption>::failure(std::move(instrument_type).error());
    }
    if (instrument_type.value() != "asian") {
        return Result<AsianOption>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'{}.type' is '{}' but this parser builds Asian options; expected 'asian'",
                        node.path(),
                        instrument_type.value()),
            kContext);
    }

    auto option_type_text = node.string("option_type");
    if (!option_type_text) {
        return Result<AsianOption>::failure(std::move(option_type_text).error());
    }
    const auto option_type = parse_option_type(option_type_text.value());
    if (!option_type.has_value()) {
        return Result<AsianOption>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("'{}.option_type' is '{}'; expected 'call' or 'put'",
                        node.path(),
                        option_type_text.value()),
            kContext);
    }

    // Required, not defaulted: an Asian price without its averaging convention is
    // ambiguous, and choosing one silently would price a different contract.
    auto averaging_text = node.string("averaging");
    if (!averaging_text) {
        return Result<AsianOption>::failure(std::move(averaging_text).error());
    }
    const auto averaging = parse_averaging_type(averaging_text.value());
    if (!averaging.has_value()) {
        return Result<AsianOption>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("'{}.averaging' is '{}'; expected 'arithmetic' or 'geometric'",
                        node.path(),
                        averaging_text.value()),
            kContext);
    }

    auto strike = node.positive_number("strike");
    if (!strike) {
        return Result<AsianOption>::failure(std::move(strike).error());
    }

    auto maturity = node.positive_number("maturity");
    if (!maturity) {
        return Result<AsianOption>::failure(std::move(maturity).error());
    }

    auto monitoring_count = node.positive_integer("monitoring_count");
    if (!monitoring_count) {
        return Result<AsianOption>::failure(std::move(monitoring_count).error());
    }

    return AsianOption::create(
        *option_type, *averaging, strike.value(), maturity.value(), monitoring_count.value());
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

Result<HestonModel> parse_heston_model(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"type",
                                                     "initial_variance",
                                                     "mean_reversion",
                                                     "long_run_variance",
                                                     "vol_of_variance",
                                                     "correlation"});
    if (!unknown) {
        return Result<HestonModel>::failure(unknown.error());
    }

    auto model_type = node.string("type");
    if (!model_type) {
        return Result<HestonModel>::failure(std::move(model_type).error());
    }
    if (model_type.value() != "heston") {
        return Result<HestonModel>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'{}.type' is '{}' but this parser builds Heston models; expected 'heston'",
                        node.path(),
                        model_type.value()),
            kContext);
    }

    auto v0 = node.number("initial_variance");
    if (!v0) {
        return Result<HestonModel>::failure(std::move(v0).error());
    }
    auto kappa = node.number("mean_reversion");
    if (!kappa) {
        return Result<HestonModel>::failure(std::move(kappa).error());
    }
    auto theta = node.number("long_run_variance");
    if (!theta) {
        return Result<HestonModel>::failure(std::move(theta).error());
    }
    auto xi = node.number("vol_of_variance");
    if (!xi) {
        return Result<HestonModel>::failure(std::move(xi).error());
    }
    auto rho = node.number("correlation");
    if (!rho) {
        return Result<HestonModel>::failure(std::move(rho).error());
    }

    return HestonModel::create(v0.value(), kappa.value(), theta.value(), xi.value(), rho.value());
}

Result<MonteCarloConfig> parse_monte_carlo_config(const ConfigNode& node,
                                                  std::optional<std::uint64_t> seed_override) {
    const Status unknown = node.reject_unknown_keys({"type",
                                                     "paths",
                                                     "steps",
                                                     "scheme",
                                                     "seed",
                                                     "confidence_level",
                                                     "antithetic",
                                                     "control_variate",
                                                     "control_variate_pilot_paths"});
    if (!unknown) {
        return Result<MonteCarloConfig>::failure(unknown.error());
    }

    MonteCarloConfig config;

    auto paths = node.positive_integer("paths");
    if (!paths) {
        return Result<MonteCarloConfig>::failure(std::move(paths).error());
    }
    config.paths = paths.value();

    auto steps = node.integer_or("steps", 1);
    if (!steps) {
        return Result<MonteCarloConfig>::failure(std::move(steps).error());
    }
    config.steps = steps.value();

    auto scheme_text = node.string_or("scheme", "exact");
    if (!scheme_text) {
        return Result<MonteCarloConfig>::failure(std::move(scheme_text).error());
    }
    const auto scheme = parse_discretization_scheme(scheme_text.value());
    if (!scheme.has_value()) {
        return Result<MonteCarloConfig>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("'{}.scheme' is '{}'; expected 'exact', 'euler_maruyama' or 'milstein'",
                        node.path(),
                        scheme_text.value()),
            kContext);
    }
    config.scheme = *scheme;

    auto confidence_level = node.number_or("confidence_level", 0.95);
    if (!confidence_level) {
        return Result<MonteCarloConfig>::failure(std::move(confidence_level).error());
    }
    config.confidence_level = confidence_level.value();

    // Variance reduction is off unless asked for: a technique that changes the
    // estimator must appear in the configuration, so a stored result says plainly
    // which estimator produced its number.
    auto antithetic = node.boolean_or("antithetic", false);
    if (!antithetic) {
        return Result<MonteCarloConfig>::failure(std::move(antithetic).error());
    }
    config.variance_reduction.antithetic = antithetic.value();

    auto control_variate = node.boolean_or("control_variate", false);
    if (!control_variate) {
        return Result<MonteCarloConfig>::failure(std::move(control_variate).error());
    }
    config.variance_reduction.control_variate = control_variate.value();

    auto pilot_paths = node.integer_or("control_variate_pilot_paths", 2000);
    if (!pilot_paths) {
        return Result<MonteCarloConfig>::failure(std::move(pilot_paths).error());
    }
    config.control_variate_pilot_paths = pilot_paths.value();

    // --seed beats the file, and the precedence lives here so that no caller can
    // reorder it. Without either, the run is rejected: a seed chosen implicitly
    // cannot be reproduced from the record the run leaves behind (ADR-010).
    if (seed_override.has_value()) {
        config.seed = *seed_override;
    } else {
        if (!node.contains("seed")) {
            return Result<MonteCarloConfig>::failure(
                ErrorCode::InvalidConfiguration,
                fmt::format("'{}.seed' is required for a stochastic method, and has no default: a "
                            "run whose seed was chosen for it cannot be reproduced from its own "
                            "record. Supply it in the configuration or with --seed.",
                            node.path()),
                kContext);
        }
        auto seed = node.integer("seed");
        if (!seed) {
            return Result<MonteCarloConfig>::failure(std::move(seed).error());
        }
        if (seed.value() < 0) {
            return Result<MonteCarloConfig>::failure(
                ErrorCode::InvalidConfiguration,
                fmt::format("'{}.seed' must be non-negative but is {}", node.path(), seed.value()),
                kContext);
        }
        config.seed = static_cast<std::uint64_t>(seed.value());
    }

    return Result<MonteCarloConfig>::success(config);
}

}  // namespace diffusionworks
