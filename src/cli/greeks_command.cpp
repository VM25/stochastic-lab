#include "greeks_command.hpp"

#include <diffusionworks/config/domain_parsers.hpp>
#include <diffusionworks/engines/greeks_monte_carlo.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>

#include "output.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "greeks";

/// Seeds spaced far enough apart that no two replications share a counter range,
/// matching the experiments' convention.
[[nodiscard]] std::vector<std::uint64_t> seeds_from(std::uint64_t master, std::uint64_t count) {
    std::vector<std::uint64_t> out;
    out.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(master + i * 1000003ULL);
    }
    return out;
}

}  // namespace

Result<nlohmann::json> run_greeks(const ConfigDocument& config, const Options& options) {
    const ConfigNode root = config.root();

    const Status unknown = root.reject_unknown_keys(
        {"schema_version", "command", "market", "instrument", "model", "greeks", "output"});
    if (!unknown) {
        return Result<nlohmann::json>::failure(unknown.error());
    }

    // --- Market, instrument, model -----------------------------------------
    auto market_node = root.object("market");
    if (!market_node) {
        return Result<nlohmann::json>::failure(std::move(market_node).error());
    }
    auto market = parse_market_state(market_node.value());
    if (!market) {
        return Result<nlohmann::json>::failure(std::move(market).error());
    }

    auto model_node = root.object("model");
    if (!model_node) {
        return Result<nlohmann::json>::failure(std::move(model_node).error());
    }
    auto model = parse_black_scholes_model(model_node.value());
    if (!model) {
        return Result<nlohmann::json>::failure(std::move(model).error());
    }

    auto instrument_node = root.object("instrument");
    if (!instrument_node) {
        return Result<nlohmann::json>::failure(std::move(instrument_node).error());
    }
    auto option = parse_european_option(instrument_node.value());
    if (!option) {
        // Greeks are estimated for European options here; the barrier and Asian
        // sensitivities are a separate set of estimators.
        return Result<nlohmann::json>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("the greeks command estimates European-option sensitivities; the "
                        "instrument could not be read as a European option ({})",
                        option.error().message),
            kContext);
    }

    // --- The greeks request -------------------------------------------------
    auto greeks_node = root.object("greeks");
    if (!greeks_node) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidConfiguration,
            "the greeks command needs a `greeks` block naming the greek and the method",
            kContext);
    }
    const Status greeks_unknown = greeks_node.value().reject_unknown_keys({"greek",
                                                                           "method",
                                                                           "spot_bump_fraction",
                                                                           "volatility_bump",
                                                                           "paths",
                                                                           "seeds",
                                                                           "seed_count",
                                                                           "master_seed"});
    if (!greeks_unknown) {
        return Result<nlohmann::json>::failure(greeks_unknown.error());
    }

    auto greek_text = greeks_node.value().string("greek");
    if (!greek_text) {
        return Result<nlohmann::json>::failure(std::move(greek_text).error());
    }
    const auto greek = parse_greek_name(greek_text.value());
    if (!greek.has_value()) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("unknown greek '{}'; expected delta, gamma, or vega", greek_text.value()),
            kContext);
    }

    auto method_text = greeks_node.value().string("method");
    if (!method_text) {
        return Result<nlohmann::json>::failure(std::move(method_text).error());
    }
    const auto method = parse_greek_method(method_text.value());
    if (!method.has_value()) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("unknown method '{}'; expected finite_difference, pathwise, or "
                        "likelihood_ratio",
                        method_text.value()),
            kContext);
    }

    GreeksMonteCarloConfig mc;
    {
        auto value = greeks_node.value().number_or("spot_bump_fraction", mc.spot_bump_fraction);
        if (!value) {
            return Result<nlohmann::json>::failure(std::move(value).error());
        }
        mc.spot_bump_fraction = value.value();
    }
    {
        auto value = greeks_node.value().number_or("volatility_bump", mc.volatility_bump);
        if (!value) {
            return Result<nlohmann::json>::failure(std::move(value).error());
        }
        mc.volatility_bump = value.value();
    }
    {
        auto value = greeks_node.value().integer_or("paths", mc.paths);
        if (!value) {
            return Result<nlohmann::json>::failure(std::move(value).error());
        }
        mc.paths = value.value();
    }
    if (options.threads.has_value()) {
        mc.threads = static_cast<int>(options.threads.value());
    }

    // --- The seed set -------------------------------------------------------
    //
    // Either an explicit list of seeds, or a count plus a master seed. The command
    // line's --seed overrides the master seed but not an explicit list, which is a
    // deliberately chosen set. Bias and uncertainty are measured across the set, so
    // it must have at least two members.
    std::vector<std::uint64_t> seeds;
    if (greeks_node.value().contains("seeds")) {
        auto array = greeks_node.value().array("seeds");
        if (!array) {
            return Result<nlohmann::json>::failure(std::move(array).error());
        }
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto value = array.value().number_at(i);
            if (!value) {
                return Result<nlohmann::json>::failure(std::move(value).error());
            }
            if (value.value() < 0.0) {
                return Result<nlohmann::json>::failure(
                    ErrorCode::InvalidConfiguration,
                    fmt::format("greeks.seeds[{}] must be non-negative, got {}", i, value.value()),
                    kContext);
            }
            seeds.push_back(static_cast<std::uint64_t>(value.value()));
        }
    } else {
        auto count = greeks_node.value().integer_or("seed_count", 16);
        if (!count) {
            return Result<nlohmann::json>::failure(std::move(count).error());
        }
        auto master = greeks_node.value().integer_or("master_seed", 20260717);
        if (!master) {
            return Result<nlohmann::json>::failure(std::move(master).error());
        }
        std::uint64_t master_seed =
            master.value() < 0 ? 0ULL : static_cast<std::uint64_t>(master.value());
        if (options.seed.has_value()) {
            master_seed = *options.seed;
        }
        if (count.value() < 2) {
            return Result<nlohmann::json>::failure(
                ErrorCode::InvalidConfiguration,
                fmt::format("greeks.seed_count must be at least 2 -- bias and uncertainty are "
                            "measured across the seed set -- got {}",
                            count.value()),
                kContext);
        }
        seeds = seeds_from(master_seed, static_cast<std::uint64_t>(count.value()));
    }
    if (seeds.size() < 2) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidConfiguration,
            "the greeks command needs at least 2 seeds: the estimate's uncertainty is measured "
            "across the seed set, and one seed cannot provide it",
            kContext);
    }

    // --- Run across the seed set -------------------------------------------
    std::vector<SeedResult> replications;
    replications.reserve(seeds.size());
    std::set<std::string> warnings;
    double bump = 0.0;
    double runtime = 0.0;
    double mean_per_run_standard_error = 0.0;

    for (const std::uint64_t seed : seeds) {
        GreeksMonteCarloConfig run = mc;
        run.seed = seed;
        const auto estimate = GreeksMonteCarloEngine::estimate(
            market.value(), option.value(), model.value(), greek.value(), method.value(), run);
        if (!estimate) {
            // A refusal (unsupported combination, degenerate input) is reported as
            // the failure it is, not silently skipped.
            return Result<nlohmann::json>::failure(estimate.error());
        }
        replications.push_back(SeedResult{.seed = seed, .estimate = estimate.value().value});
        bump = estimate.value().bump;
        runtime += estimate.value().runtime_seconds;
        mean_per_run_standard_error += estimate.value().standard_error;
        for (const std::string& warning : estimate.value().warnings) {
            warnings.insert(warning);
        }
    }
    mean_per_run_standard_error /= static_cast<double>(seeds.size());

    const auto summary = summarize_seeds(replications);
    if (!summary) {
        return Result<nlohmann::json>::failure(summary.error());
    }

    // --- Result document ----------------------------------------------------
    nlohmann::json document;
    document["command"] = "greeks";
    document["status"] = warnings.empty() ? "ok" : "warning";
    document["market"] = nlohmann::json{{"spot", market.value().spot()},
                                        {"rate", market.value().rate()},
                                        {"dividend_yield", market.value().dividend_yield()}};
    document["instrument"] = nlohmann::json{{"type", "european"},
                                            {"option_type", to_string(option.value().type())},
                                            {"strike", option.value().strike()},
                                            {"maturity", option.value().maturity()}};
    document["model"] =
        nlohmann::json{{"type", "black_scholes"}, {"volatility", model.value().volatility()}};

    document["result"] =
        nlohmann::json{{"greek", to_string(greek.value())},
                       {"estimator", to_string(method.value())},
                       // The bump is meaningful only for finite difference; the other methods take
                       // none, and report it as null rather than a misleading zero.
                       {"bump",
                        method.value() == GreekMethod::FiniteDifference ? nlohmann::json(bump)
                                                                        : nlohmann::json(nullptr)},
                       {"estimate", summary.value().mean},
                       // The uncertainty a caller should quote is the across-seed standard error:
                       // it is the estimator's realised dispersion, not one run's self-report.
                       {"standard_error", summary.value().standard_error},
                       {"across_seed_standard_deviation", summary.value().standard_deviation},
                       {"per_run_standard_error", mean_per_run_standard_error},
                       {"minimum", summary.value().minimum},
                       {"maximum", summary.value().maximum},
                       {"paths_per_seed", mc.paths},
                       {"seed_set", seeds},
                       {"seed_count", seeds.size()},
                       {"runtime_seconds", runtime}};

    if (!warnings.empty()) {
        document["warnings"] = std::vector<std::string>(warnings.begin(), warnings.end());
    }

    document["configuration"] = config.json();
    if (!config.source().empty()) {
        document["configuration_source"] = config.source().string();
    }

    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli
