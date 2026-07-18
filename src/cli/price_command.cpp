#include "price_command.hpp"

#include <diffusionworks/config/domain_parsers.hpp>
#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>

#include "output.hpp"

#include <fmt/format.h>

#include <string>
#include <utility>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "price";

/// Serialises the market for the result document.
[[nodiscard]] nlohmann::json describe(const MarketState& market) {
    return nlohmann::json{
        {"spot", market.spot()},
        {"rate", market.rate()},
        {"dividend_yield", market.dividend_yield()},
    };
}

[[nodiscard]] nlohmann::json describe(const EuropeanOption& option) {
    return nlohmann::json{
        {"type", "european"},
        {"option_type", to_string(option.type())},
        {"strike", option.strike()},
        {"maturity", option.maturity()},
    };
}

[[nodiscard]] nlohmann::json describe(const AsianOption& option) {
    return nlohmann::json{
        {"type", "asian"},
        {"option_type", to_string(option.type())},
        // The averaging and monitoring travel with the instrument: an Asian price
        // quoted without them is ambiguous.
        {"averaging", to_string(option.averaging())},
        {"strike", option.strike()},
        {"maturity", option.maturity()},
        {"monitoring_count", option.monitoring_count()},
    };
}

[[nodiscard]] nlohmann::json describe(const BlackScholesModel& model) {
    return nlohmann::json{
        {"type", "black_scholes"},
        {"volatility", model.volatility()},
    };
}

[[nodiscard]] nlohmann::json describe(const HestonModel& model) {
    return nlohmann::json{
        {"type", "heston"},
        {"initial_variance", model.initial_variance()},
        {"mean_reversion", model.mean_reversion()},
        {"long_run_variance", model.long_run_variance()},
        {"vol_of_variance", model.vol_of_variance()},
        {"correlation", model.correlation()},
        // The Feller diagnostic travels with the model description: a reader of the
        // artifact sees whether the variance can reach zero without recomputing it.
        {"feller_ratio", model.feller_ratio()},
        {"satisfies_feller", model.satisfies_feller()},
    };
}

/// Prices a European option by whichever method the configuration names.
[[nodiscard]] Result<PricingResult> price_european(const MarketState& market,
                                                   const EuropeanOption& option,
                                                   const BlackScholesModel& model,
                                                   const ConfigNode& method,
                                                   const std::string& method_type,
                                                   const Options& options,
                                                   bool want_greeks) {
    if (method_type == "analytic") {
        const Status unknown = method.reject_unknown_keys({"type"});
        if (!unknown) {
            return Result<PricingResult>::failure(unknown.error());
        }

        auto priced = BlackScholesAnalyticEngine::price(market, option, model);
        if (!priced || !want_greeks) {
            return priced;
        }

        auto greeks = BlackScholesAnalyticEngine::greeks(market, option, model);
        if (!greeks) {
            return Result<PricingResult>::failure(std::move(greeks).error());
        }
        PricingResult result = std::move(priced).value();
        for (const UndefinedGreek& entry : greeks.value().undefined) {
            result.add_warning(fmt::format("{} is undefined here: {}", entry.name, entry.reason));
        }
        result.greeks = std::move(greeks).value();
        return Result<PricingResult>::success(std::move(result));
    }

    if (method_type == "monte_carlo") {
        auto config = parse_monte_carlo_config(method, options.seed);
        if (!config) {
            return Result<PricingResult>::failure(std::move(config).error());
        }
        if (options.threads.has_value()) {
            config.value().threads = static_cast<int>(options.threads.value());
        }
        return MonteCarloEngine::price(market, option, model, config.value());
    }

    return Result<PricingResult>::failure(
        ErrorCode::UnsupportedCombination,
        fmt::format("'method.type' is '{}'; European options support 'analytic' and "
                    "'monte_carlo'. Finite-difference methods are not yet implemented.",
                    method_type),
        kContext);
}

/// Prices an Asian option.
[[nodiscard]] Result<PricingResult> price_asian(const MarketState& market,
                                                const AsianOption& option,
                                                const BlackScholesModel& model,
                                                const ConfigNode& method,
                                                const std::string& method_type,
                                                const Options& options) {
    if (method_type != "monte_carlo") {
        // Naming 'analytic' for an arithmetic Asian must fail rather than fall
        // back: the sum of lognormals is not lognormal, so no closed form exists,
        // and quietly pricing something else would be worse than refusing.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'method.type' is '{}'; Asian options support 'monte_carlo' in this "
                        "build. The arithmetic average of lognormals is not lognormal and has no "
                        "closed form; the geometric analytic engine arrives in Phase 4.",
                        method_type),
            kContext);
    }

    auto config = parse_monte_carlo_config(method, options.seed);
    if (!config) {
        return Result<PricingResult>::failure(std::move(config).error());
    }
    if (options.threads.has_value()) {
        config.value().threads = static_cast<int>(options.threads.value());
    }
    return MonteCarloEngine::price(market, option, model, config.value());
}

}  // namespace

Result<nlohmann::json> run_price(const ConfigDocument& config, const Options& options) {
    const ConfigNode root = config.root();

    const Status unknown = root.reject_unknown_keys({"schema_version",
                                                     "command",
                                                     "market",
                                                     "instrument",
                                                     "model",
                                                     "method",
                                                     "greeks",
                                                     "output"});
    if (!unknown) {
        return Result<nlohmann::json>::failure(unknown.error());
    }

    // --- Market -------------------------------------------------------------

    auto market_node = root.object("market");
    if (!market_node) {
        return Result<nlohmann::json>::failure(std::move(market_node).error());
    }
    auto market = parse_market_state(market_node.value());
    if (!market) {
        return Result<nlohmann::json>::failure(std::move(market).error());
    }

    // --- Model --------------------------------------------------------------

    auto model_node = root.object("model");
    if (!model_node) {
        return Result<nlohmann::json>::failure(std::move(model_node).error());
    }
    auto model_type = model_node.value().string("type");
    if (!model_type) {
        return Result<nlohmann::json>::failure(std::move(model_type).error());
    }

    // The Heston model has its own engine and its own supported instruments, so it
    // takes a dedicated path here rather than flowing through the Black-Scholes
    // pricing dispatch. Handled and returned in full; the Black-Scholes path below is
    // left exactly as it was.
    if (model_type.value() == "heston") {
        auto heston = parse_heston_model(model_node.value());
        if (!heston) {
            return Result<nlohmann::json>::failure(std::move(heston).error());
        }

        auto instrument_node = root.object("instrument");
        if (!instrument_node) {
            return Result<nlohmann::json>::failure(std::move(instrument_node).error());
        }
        auto option = parse_european_option(instrument_node.value());
        if (!option) {
            return Result<nlohmann::json>::failure(
                ErrorCode::UnsupportedCombination,
                fmt::format("the Heston semi-analytic engine prices European options; the "
                            "instrument could not be read as one ({})",
                            option.error().message),
                kContext);
        }

        auto method_node = root.object("method");
        if (!method_node) {
            return Result<nlohmann::json>::failure(std::move(method_node).error());
        }
        auto heston_method = method_node.value().string("type");
        if (!heston_method) {
            return Result<nlohmann::json>::failure(std::move(heston_method).error());
        }

        Result<PricingResult> priced = Result<PricingResult>::failure(
            ErrorCode::NotImplemented, "no Heston method was dispatched", kContext);

        if (heston_method.value() == "heston_analytic") {
            const Status method_unknown = method_node.value().reject_unknown_keys(
                {"type", "quadrature_nodes", "convergence_tolerance"});
            if (!method_unknown) {
                return Result<nlohmann::json>::failure(method_unknown.error());
            }

            HestonAnalyticConfig heston_config;
            auto nodes =
                method_node.value().integer_or("quadrature_nodes", heston_config.quadrature_nodes);
            if (!nodes) {
                return Result<nlohmann::json>::failure(std::move(nodes).error());
            }
            heston_config.quadrature_nodes = nodes.value();
            auto tolerance = method_node.value().number_or("convergence_tolerance",
                                                           heston_config.convergence_tolerance);
            if (!tolerance) {
                return Result<nlohmann::json>::failure(std::move(tolerance).error());
            }
            heston_config.convergence_tolerance = tolerance.value();

            priced = HestonAnalyticEngine::price(
                market.value(), option.value(), heston.value(), heston_config);
        } else if (heston_method.value() == "heston_monte_carlo") {
            const Status method_unknown = method_node.value().reject_unknown_keys(
                {"type", "paths", "steps", "seed", "confidence_level"});
            if (!method_unknown) {
                return Result<nlohmann::json>::failure(method_unknown.error());
            }

            // The naive scheme is a diagnostic baseline studied in EXP-10, never a way
            // to price, so the CLI pricing path always uses full truncation.
            HestonMonteCarloConfig heston_config;
            auto paths = method_node.value().integer_or("paths", heston_config.paths);
            if (!paths) {
                return Result<nlohmann::json>::failure(std::move(paths).error());
            }
            heston_config.paths = paths.value();
            auto steps = method_node.value().integer_or("steps", heston_config.steps);
            if (!steps) {
                return Result<nlohmann::json>::failure(std::move(steps).error());
            }
            heston_config.steps = steps.value();
            auto confidence =
                method_node.value().number_or("confidence_level", heston_config.confidence_level);
            if (!confidence) {
                return Result<nlohmann::json>::failure(std::move(confidence).error());
            }
            heston_config.confidence_level = confidence.value();
            // The command-line seed overrides the file, matching every other engine.
            auto seed = method_node.value().integer_or(
                "seed", static_cast<std::int64_t>(heston_config.seed));
            if (!seed) {
                return Result<nlohmann::json>::failure(std::move(seed).error());
            }
            if (seed.value() < 0) {
                return Result<nlohmann::json>::failure(
                    ErrorCode::InvalidArgument, "method.seed must be non-negative", kContext);
            }
            heston_config.seed = options.seed.value_or(static_cast<std::uint64_t>(seed.value()));

            priced = HestonMonteCarloEngine::price(
                market.value(), option.value(), heston.value(), heston_config);
        } else {
            return Result<nlohmann::json>::failure(
                ErrorCode::UnsupportedCombination,
                fmt::format("the Heston model is priced by 'heston_analytic' or "
                            "'heston_monte_carlo'; 'method.type' is '{}'",
                            heston_method.value()),
                kContext);
        }
        if (!priced) {
            return Result<nlohmann::json>::failure(priced.error());
        }

        nlohmann::json document;
        document["status"] = priced.value().has_warnings() ? "warning" : "ok";
        document["command"] = "price";
        document["market"] = describe(market.value());
        document["instrument"] = describe(option.value());
        document["model"] = describe(heston.value());
        document["method"] = heston_method.value();
        document["result"] = to_json(priced.value());
        document["configuration"] = config.json();
        if (!config.source().empty()) {
            document["configuration_source"] = config.source().string();
        }
        document["build_metadata"] = to_json(collect_build_info());
        return Result<nlohmann::json>::success(std::move(document));
    }

    auto model = parse_black_scholes_model(model_node.value());
    if (!model) {
        return Result<nlohmann::json>::failure(std::move(model).error());
    }

    // --- Method -------------------------------------------------------------

    auto method_node = root.object("method");
    if (!method_node) {
        return Result<nlohmann::json>::failure(std::move(method_node).error());
    }
    auto method_type = method_node.value().string("type");
    if (!method_type) {
        return Result<nlohmann::json>::failure(std::move(method_type).error());
    }

    auto want_greeks = root.boolean_or("greeks", true);
    if (!want_greeks) {
        return Result<nlohmann::json>::failure(std::move(want_greeks).error());
    }

    // --- Instrument ---------------------------------------------------------

    auto instrument_node = root.object("instrument");
    if (!instrument_node) {
        return Result<nlohmann::json>::failure(std::move(instrument_node).error());
    }
    auto instrument_type = instrument_node.value().string("type");
    if (!instrument_type) {
        return Result<nlohmann::json>::failure(std::move(instrument_type).error());
    }

    nlohmann::json instrument_description;
    Result<PricingResult> priced = Result<PricingResult>::failure(
        ErrorCode::NotImplemented, "no instrument was dispatched", kContext);

    if (instrument_type.value() == "european") {
        auto option = parse_european_option(instrument_node.value());
        if (!option) {
            return Result<nlohmann::json>::failure(std::move(option).error());
        }
        instrument_description = describe(option.value());
        priced = price_european(market.value(),
                                option.value(),
                                model.value(),
                                method_node.value(),
                                method_type.value(),
                                options,
                                want_greeks.value());
    } else if (instrument_type.value() == "asian") {
        auto option = parse_asian_option(instrument_node.value());
        if (!option) {
            return Result<nlohmann::json>::failure(std::move(option).error());
        }
        instrument_description = describe(option.value());
        priced = price_asian(market.value(),
                             option.value(),
                             model.value(),
                             method_node.value(),
                             method_type.value(),
                             options);
    } else {
        return Result<nlohmann::json>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'instrument.type' is '{}'; this build supports 'european' and 'asian'",
                        instrument_type.value()),
            kContext);
    }

    if (!priced) {
        return Result<nlohmann::json>::failure(std::move(priced).error());
    }
    const PricingResult result = std::move(priced).value();

    // --- Result document ----------------------------------------------------

    nlohmann::json document;
    document["status"] = "ok";
    document["command"] = "price";
    document["market"] = describe(market.value());
    document["instrument"] = std::move(instrument_description);
    document["model"] = describe(model.value());
    document["method"] = method_type.value();
    document["result"] = to_json(result);

    // The exact configuration travels with the result, so a published number can
    // be re-run without hunting for the file that produced it.
    document["configuration"] = config.json();
    if (!config.source().empty()) {
        document["configuration_source"] = config.source().string();
    }

    // Monte Carlo pricing uses the requested threads (Phase 12). Analytic pricing is a
    // single closed-form evaluation with nothing to parallelise, so a thread count
    // supplied there had no effect and is reported as such rather than silently ignored.
    if (options.threads.has_value() && options.threads.value() != 1 &&
        method_type.value() != "monte_carlo") {
        document["warnings"] = nlohmann::json::array(
            {fmt::format("--threads {} was supplied but {} pricing is a single evaluation with "
                         "nothing to parallelise; the thread count had no effect",
                         options.threads.value(),
                         method_type.value())});
    }

    document["build_metadata"] = to_json(collect_build_info());

    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli
