#include "experiment_command.hpp"

#include <diffusionworks/experiments/convergence_experiments.hpp>

#include <fmt/format.h>

#include <optional>
#include <utility>
#include <vector>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "experiment";

/// Parses the `convergence` block of an experiment configuration.
///
/// Every field is optional and falls back to the documented default, but unknown
/// keys are rejected: a typo in `volatilty` must fail loudly rather than leave the
/// default in place and produce a plausible study of the wrong model.
Result<ConvergenceExperimentConfig> parse_convergence_config(const ConfigNode& root) {
    ConvergenceExperimentConfig config;

    if (!root.contains("convergence")) {
        return Result<ConvergenceExperimentConfig>::success(config);
    }

    auto node = root.object("convergence");
    if (!node) {
        return Result<ConvergenceExperimentConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "rate",
                                                             "dividend_yield",
                                                             "volatility",
                                                             "maturity",
                                                             "strike",
                                                             "master_seed",
                                                             "seed_count",
                                                             "step_counts",
                                                             "asymptotic_level_count",
                                                             "strong_paths",
                                                             "call_payoff_paths",
                                                             "path_counts"});
    if (!unknown) {
        return Result<ConvergenceExperimentConfig>::failure(unknown.error());
    }

    // Reads each field, keeping the first error rather than a chain of nested
    // conditionals. The first is the right one to report: it names the field the
    // author has to fix, and a later field's error is often a consequence of it.
    std::optional<Error> first_error;

    const auto read_number = [&](const char* key, double fallback) -> double {
        auto value = node.value().number_or(key, fallback);
        if (!value) {
            if (!first_error.has_value()) {
                first_error = value.error();
            }
            return fallback;
        }
        return value.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto value = node.value().integer_or(key, fallback);
        if (!value) {
            if (!first_error.has_value()) {
                first_error = value.error();
            }
            return fallback;
        }
        return value.value();
    };

    config.spot = read_number("spot", config.spot);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.volatility = read_number("volatility", config.volatility);
    config.maturity = read_number("maturity", config.maturity);
    config.strike = read_number("strike", config.strike);

    const std::int64_t seed =
        read_integer("master_seed", static_cast<std::int64_t>(config.master_seed));
    const std::int64_t seed_count =
        read_integer("seed_count", static_cast<std::int64_t>(config.seed_count));
    const std::int64_t strong_paths =
        read_integer("strong_paths", static_cast<std::int64_t>(config.strong_paths));
    const std::int64_t call_paths =
        read_integer("call_payoff_paths", static_cast<std::int64_t>(config.call_payoff_paths));
    const std::int64_t asymptotic = read_integer(
        "asymptotic_level_count", static_cast<std::int64_t>(config.asymptotic_level_count));

    if (first_error.has_value()) {
        return Result<ConvergenceExperimentConfig>::failure(*first_error);
    }

    if (seed < 0 || seed_count < 2 || strong_paths < 2 || call_paths < 2 || asymptotic < 3) {
        return Result<ConvergenceExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "master_seed must be non-negative; seed_count, strong_paths, and call_payoff_paths at "
            "least 2; and asymptotic_level_count at least 3 (a slope needs three points to carry "
            "an uncertainty)",
            kContext);
    }
    config.master_seed = static_cast<std::uint64_t>(seed);
    config.seed_count = static_cast<std::uint64_t>(seed_count);
    config.strong_paths = static_cast<std::uint64_t>(strong_paths);
    config.call_payoff_paths = static_cast<std::uint64_t>(call_paths);
    config.asymptotic_level_count = static_cast<std::size_t>(asymptotic);

    if (node.value().contains("step_counts")) {
        auto array = node.value().array("step_counts");
        if (!array) {
            return Result<ConvergenceExperimentConfig>::failure(std::move(array).error());
        }
        std::vector<std::int64_t> steps;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto value = array.value().number_at(i);
            if (!value) {
                return Result<ConvergenceExperimentConfig>::failure(std::move(value).error());
            }
            if (value.value() < 1.0) {
                return Result<ConvergenceExperimentConfig>::failure(
                    ErrorCode::InvalidArgument,
                    fmt::format(
                        "convergence.step_counts[{}] must be at least 1, got {}", i, value.value()),
                    kContext);
            }
            steps.push_back(static_cast<std::int64_t>(value.value()));
        }
        if (steps.size() < 3) {
            return Result<ConvergenceExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("convergence.step_counts needs at least 3 levels to determine an "
                            "order, got {}",
                            steps.size()),
                kContext);
        }
        config.step_counts = std::move(steps);
    }

    if (node.value().contains("path_counts")) {
        auto array = node.value().array("path_counts");
        if (!array) {
            return Result<ConvergenceExperimentConfig>::failure(std::move(array).error());
        }
        std::vector<std::uint64_t> paths;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto value = array.value().number_at(i);
            if (!value) {
                return Result<ConvergenceExperimentConfig>::failure(std::move(value).error());
            }
            if (value.value() < 2.0) {
                return Result<ConvergenceExperimentConfig>::failure(
                    ErrorCode::InvalidArgument,
                    fmt::format(
                        "convergence.path_counts[{}] must be at least 2, got {}", i, value.value()),
                    kContext);
            }
            paths.push_back(static_cast<std::uint64_t>(value.value()));
        }
        if (paths.size() < 3) {
            return Result<ConvergenceExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("convergence.path_counts needs at least 3 levels to determine a "
                            "rate, got {}",
                            paths.size()),
                kContext);
        }
        config.path_counts = std::move(paths);
    }

    if (config.asymptotic_level_count > config.step_counts.size()) {
        return Result<ConvergenceExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("convergence.asymptotic_level_count is {} but only {} step counts are "
                        "configured",
                        config.asymptotic_level_count,
                        config.step_counts.size()),
            kContext);
    }

    return Result<ConvergenceExperimentConfig>::success(std::move(config));
}

nlohmann::json table_to_json(const CsvTable& table) {
    return nlohmann::json{{"headers", table.headers}, {"rows", table.rows}};
}

}  // namespace

Result<nlohmann::json> run_experiment(const ConfigDocument& config, const Options& options) {
    if (!options.experiment_id.has_value()) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidArgument,
            "the experiment command requires --id, e.g. --id EXP-02. Running every experiment is "
            "not assumed from a bare invocation: each has its own cost and its own exit criteria.",
            kContext);
    }

    auto parsed = parse_convergence_config(config.root());
    if (!parsed) {
        return Result<nlohmann::json>::failure(std::move(parsed).error());
    }
    ConvergenceExperimentConfig experiment_config = std::move(parsed).value();

    // The command line overrides the file, so a sweep over seeds does not need one
    // configuration file per seed.
    if (options.seed.has_value()) {
        experiment_config.master_seed = *options.seed;
    }

    const std::string& id = *options.experiment_id;

    Result<ExperimentRecord> record = Result<ExperimentRecord>::failure(
        ErrorCode::NotImplemented,
        fmt::format("experiment '{}' is not implemented in this build. Implemented: EXP-01, "
                    "EXP-02, EXP-03, EXP-04.",
                    id),
        kContext);

    if (id == "EXP-01") {
        record = run_sampling_convergence(experiment_config);
    } else if (id == "EXP-02") {
        record = run_strong_convergence(experiment_config);
    } else if (id == "EXP-03") {
        record = run_weak_convergence(experiment_config);
    } else if (id == "EXP-04") {
        record = run_bias_variance_tradeoff(experiment_config);
    }

    if (!record) {
        return Result<nlohmann::json>::failure(std::move(record).error());
    }

    nlohmann::json document = record.value().to_json();
    document["table"] = table_to_json(record.value().table);
    // The configuration file is embedded verbatim alongside the parsed values, so
    // the artifact carries the exact inputs that produced it rather than a
    // re-serialisation that might have lost something.
    document["configuration_source"] = config.source().string();
    document["configuration_document"] = config.json();

    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli
