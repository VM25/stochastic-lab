#include "price_command.hpp"

#include <diffusionworks/config/domain_parsers.hpp>
#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include "output.hpp"

#include <fmt/format.h>

#include <string>
#include <utility>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "price";

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

    // --- Domain objects -----------------------------------------------------

    auto market_node = root.object("market");
    if (!market_node) {
        return Result<nlohmann::json>::failure(std::move(market_node).error());
    }
    auto market = parse_market_state(market_node.value());
    if (!market) {
        return Result<nlohmann::json>::failure(std::move(market).error());
    }

    auto instrument_node = root.object("instrument");
    if (!instrument_node) {
        return Result<nlohmann::json>::failure(std::move(instrument_node).error());
    }
    auto option = parse_european_option(instrument_node.value());
    if (!option) {
        return Result<nlohmann::json>::failure(std::move(option).error());
    }

    auto model_node = root.object("model");
    if (!model_node) {
        return Result<nlohmann::json>::failure(std::move(model_node).error());
    }
    auto model = parse_black_scholes_model(model_node.value());
    if (!model) {
        return Result<nlohmann::json>::failure(std::move(model).error());
    }

    // --- Method selection ---------------------------------------------------

    auto method_node = root.object("method");
    if (!method_node) {
        return Result<nlohmann::json>::failure(std::move(method_node).error());
    }
    const Status method_unknown = method_node.value().reject_unknown_keys({"type"});
    if (!method_unknown) {
        return Result<nlohmann::json>::failure(method_unknown.error());
    }
    auto method_type = method_node.value().string("type");
    if (!method_type) {
        return Result<nlohmann::json>::failure(std::move(method_type).error());
    }
    if (method_type.value() != "analytic") {
        // Naming an unimplemented method must fail rather than silently fall
        // back to the analytic engine, which would answer a question the
        // configuration did not ask.
        return Result<nlohmann::json>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'method.type' is '{}'; this build supports 'analytic' for Black-Scholes "
                        "European options. Monte Carlo and finite-difference methods are not yet "
                        "implemented.",
                        method_type.value()),
            kContext);
    }

    auto want_greeks = root.boolean_or("greeks", true);
    if (!want_greeks) {
        return Result<nlohmann::json>::failure(std::move(want_greeks).error());
    }

    // --- Valuation ----------------------------------------------------------

    auto priced = BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
    if (!priced) {
        return Result<nlohmann::json>::failure(std::move(priced).error());
    }
    PricingResult result = std::move(priced).value();

    if (want_greeks.value()) {
        auto greeks =
            BlackScholesAnalyticEngine::greeks(market.value(), option.value(), model.value());
        if (!greeks) {
            // A failure here means the computation broke, not that a derivative
            // does not exist -- the engine reports non-existence per Greek. That
            // is serious enough to fail the run rather than warn.
            return Result<nlohmann::json>::failure(std::move(greeks).error());
        }

        // Individual Greeks can fail to exist where the price is still exact, at
        // the degenerate payoff kink. That does not invalidate the price, so the
        // run succeeds; each missing Greek is raised as a warning so its absence
        // is visible to a reader who only skims the top of the output.
        for (const UndefinedGreek& entry : greeks.value().undefined) {
            result.add_warning(fmt::format("{} is undefined here: {}", entry.name, entry.reason));
        }

        result.greeks = std::move(greeks).value();
    }

    // --- Result document ----------------------------------------------------

    nlohmann::json document;
    document["status"] = "ok";
    document["command"] = "price";

    document["market"] = nlohmann::json{
        {"spot", market.value().spot()},
        {"rate", market.value().rate()},
        {"dividend_yield", market.value().dividend_yield()},
    };

    document["instrument"] = nlohmann::json{
        {"type", "european"},
        {"option_type", to_string(option.value().type())},
        {"strike", option.value().strike()},
        {"maturity", option.value().maturity()},
    };

    document["model"] = nlohmann::json{
        {"type", "black_scholes"},
        {"volatility", model.value().volatility()},
    };

    document["method"] = method_type.value();
    document["result"] = to_json(result);

    // The exact configuration travels with the result, so a published number can
    // be re-run without hunting for the file that produced it.
    document["configuration"] = config.json();
    if (!config.source().empty()) {
        document["configuration_source"] = config.source().string();
    }

    // Seed and thread count are omitted rather than defaulted: an analytic price
    // consumes no randomness and runs on one thread by construction, and
    // reporting a seed would imply a stochastic method.
    if (options.seed.has_value()) {
        document["warnings"] = nlohmann::json::array(
            {"--seed was supplied but the analytic engine is deterministic and ignores it"});
    }

    document["build_metadata"] = to_json(collect_build_info());

    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli
