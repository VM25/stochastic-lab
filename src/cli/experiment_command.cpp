#include "experiment_command.hpp"

#include <diffusionworks/experiments/barrier_experiments.hpp>
#include <diffusionworks/experiments/calibration_experiments.hpp>
#include <diffusionworks/experiments/convergence_experiments.hpp>
#include <diffusionworks/experiments/greek_experiments.hpp>
#include <diffusionworks/experiments/heston_simulation_experiments.hpp>
#include <diffusionworks/experiments/pde_experiments.hpp>

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

/// Parses the `pde` block for EXP-06.
///
/// Absent means the documented defaults, which are the configuration EXP-06's
/// published record was produced from. Unknown keys are rejected for the same
/// reason as in the convergence block: a typo must fail rather than leave a
/// default in place and produce a plausible study of something else.
Result<PdeExperimentConfig> parse_pde_config(const ConfigNode& root) {
    PdeExperimentConfig config;

    if (!root.contains("pde")) {
        return Result<PdeExperimentConfig>::success(config);
    }

    auto node = root.object("pde");
    if (!node) {
        return Result<PdeExperimentConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "strike",
                                                             "rate",
                                                             "dividend_yield",
                                                             "volatility",
                                                             "maturity",
                                                             "space_nodes",
                                                             "space_sweep_time_steps",
                                                             "time_sweep_nodes",
                                                             "time_steps",
                                                             "time_sweep_reference_steps",
                                                             "s_max_multiples",
                                                             "s_max_sweep_spacing",
                                                             "s_max_sweep_time_steps",
                                                             "strike_offsets",
                                                             "alignment_nodes",
                                                             "alignment_time_steps",
                                                             "rannacher_counts",
                                                             "stability_ratios",
                                                             "stability_nodes",
                                                             "volatilities",
                                                             "maturities"});
    if (!unknown) {
        return Result<PdeExperimentConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };
    const auto read_integers = [&](const char* key, std::vector<std::int64_t> fallback) {
        const std::vector<double> raw = read_doubles(key, {});
        if (raw.empty()) {
            return fallback;
        }
        std::vector<std::int64_t> out;
        out.reserve(raw.size());
        for (const double v : raw) {
            out.push_back(static_cast<std::int64_t>(v));
        }
        return out;
    };

    config.spot = read_number("spot", config.spot);
    config.strike = read_number("strike", config.strike);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.volatility = read_number("volatility", config.volatility);
    config.maturity = read_number("maturity", config.maturity);
    config.s_max_sweep_spacing = read_number("s_max_sweep_spacing", config.s_max_sweep_spacing);

    config.space_sweep_time_steps =
        read_integer("space_sweep_time_steps", config.space_sweep_time_steps);
    config.time_sweep_nodes = read_integer("time_sweep_nodes", config.time_sweep_nodes);
    config.time_sweep_reference_steps =
        read_integer("time_sweep_reference_steps", config.time_sweep_reference_steps);
    config.s_max_sweep_time_steps =
        read_integer("s_max_sweep_time_steps", config.s_max_sweep_time_steps);
    config.alignment_nodes = read_integer("alignment_nodes", config.alignment_nodes);
    config.alignment_time_steps = read_integer("alignment_time_steps", config.alignment_time_steps);
    config.stability_nodes = read_integer("stability_nodes", config.stability_nodes);

    config.space_nodes = read_integers("space_nodes", config.space_nodes);
    config.time_steps = read_integers("time_steps", config.time_steps);
    config.rannacher_counts = read_integers("rannacher_counts", config.rannacher_counts);
    config.s_max_multiples = read_doubles("s_max_multiples", config.s_max_multiples);
    config.strike_offsets = read_doubles("strike_offsets", config.strike_offsets);
    config.stability_ratios = read_doubles("stability_ratios", config.stability_ratios);
    config.volatilities = read_doubles("volatilities", config.volatilities);
    config.maturities = read_doubles("maturities", config.maturities);

    if (first_error.has_value()) {
        return Result<PdeExperimentConfig>::failure(*first_error);
    }

    if (config.space_nodes.size() < 3 || config.time_steps.size() < 3) {
        return Result<PdeExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "space_nodes and time_steps each need at least 3 levels to determine an order",
            kContext);
    }

    return Result<PdeExperimentConfig>::success(std::move(config));
}

/// Parses the `barrier` block for EXP-07.
///
/// Same contract as the blocks above: absent means the documented defaults, which
/// are what the published record was produced from, and an unknown key is a
/// rejection rather than a silently ignored typo.
Result<BarrierExperimentConfig> parse_barrier_config(const ConfigNode& root) {
    BarrierExperimentConfig config;

    if (!root.contains("barrier")) {
        return Result<BarrierExperimentConfig>::success(config);
    }

    auto node = root.object("barrier");
    if (!node) {
        return Result<BarrierExperimentConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "strike",
                                                             "rate",
                                                             "dividend_yield",
                                                             "volatility",
                                                             "maturity",
                                                             "barriers",
                                                             "up_barriers",
                                                             "monitoring_counts",
                                                             "paths",
                                                             "seed_count",
                                                             "master_seed",
                                                             "volatilities",
                                                             "pde_resolutions",
                                                             "pde_rannacher_steps"});
    if (!unknown) {
        return Result<BarrierExperimentConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };

    config.spot = read_number("spot", config.spot);
    config.strike = read_number("strike", config.strike);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.volatility = read_number("volatility", config.volatility);
    config.maturity = read_number("maturity", config.maturity);

    const std::int64_t paths = read_integer("paths", config.paths);
    const std::int64_t seed_count =
        read_integer("seed_count", static_cast<std::int64_t>(config.seed_count));
    const std::int64_t master_seed =
        read_integer("master_seed", static_cast<std::int64_t>(config.master_seed));

    config.barriers = read_doubles("barriers", config.barriers);
    config.up_barriers = read_doubles("up_barriers", config.up_barriers);
    config.volatilities = read_doubles("volatilities", config.volatilities);

    const std::vector<double> monitoring = read_doubles("monitoring_counts", {});
    if (!monitoring.empty()) {
        std::vector<std::int64_t> counts;
        counts.reserve(monitoring.size());
        for (const double v : monitoring) {
            counts.push_back(static_cast<std::int64_t>(v));
        }
        config.monitoring_counts = std::move(counts);
    }

    const std::vector<double> pde_resolutions = read_doubles("pde_resolutions", {});
    if (!pde_resolutions.empty()) {
        std::vector<std::int64_t> resolutions;
        resolutions.reserve(pde_resolutions.size());
        for (const double v : pde_resolutions) {
            resolutions.push_back(static_cast<std::int64_t>(v));
        }
        config.pde_resolutions = std::move(resolutions);
    }
    config.pde_rannacher_steps = read_integer("pde_rannacher_steps", config.pde_rannacher_steps);

    if (first_error.has_value()) {
        return Result<BarrierExperimentConfig>::failure(*first_error);
    }

    if (paths < 2 || seed_count < 2 || master_seed < 0) {
        return Result<BarrierExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "paths must be at least 2; seed_count at least 2, since a bias measured from one seed "
            "cannot be told from a lucky draw, which is the question this experiment exists to "
            "settle; and master_seed non-negative",
            kContext);
    }
    config.paths = paths;
    config.seed_count = static_cast<std::uint64_t>(seed_count);
    config.master_seed = static_cast<std::uint64_t>(master_seed);

    if (config.barriers.empty() || config.up_barriers.empty() || config.volatilities.empty()) {
        return Result<BarrierExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "barriers, up_barriers, and volatilities must each be non-empty; the catalog asks for "
            "both barrier directions and they are not redundant",
            kContext);
    }
    if (config.monitoring_counts.size() < 3) {
        return Result<BarrierExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("barrier.monitoring_counts needs at least 3 frequencies to fit the bias "
                        "order, got {}",
                        config.monitoring_counts.size()),
            kContext);
    }
    for (const std::int64_t m : config.monitoring_counts) {
        if (m < 1) {
            return Result<BarrierExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("barrier.monitoring_counts entries must be at least 1, got {}", m),
                kContext);
        }
    }
    if (config.pde_resolutions.size() < 3) {
        return Result<BarrierExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("barrier.pde_resolutions needs at least 3 levels to fit a convergence "
                        "order, got {}",
                        config.pde_resolutions.size()),
            kContext);
    }
    for (const std::int64_t nodes : config.pde_resolutions) {
        if (nodes < 3) {
            return Result<BarrierExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("barrier.pde_resolutions entries must be at least 3 nodes, got {}",
                            nodes),
                kContext);
        }
    }

    return Result<BarrierExperimentConfig>::success(std::move(config));
}

/// Parses the `greeks` block for EXP-08.
///
/// Same contract as the other blocks: absent means the documented defaults that
/// produced the published record, and an unknown key is rejected rather than
/// silently ignored.
Result<GreekExperimentConfig> parse_greek_config(const ConfigNode& root) {
    GreekExperimentConfig config;

    if (!root.contains("greeks")) {
        return Result<GreekExperimentConfig>::success(config);
    }

    auto node = root.object("greeks");
    if (!node) {
        return Result<GreekExperimentConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"strike",
                                                             "rate",
                                                             "dividend_yield",
                                                             "spots",
                                                             "maturities",
                                                             "volatilities",
                                                             "spot_bump_fractions",
                                                             "volatility_bumps",
                                                             "paths",
                                                             "seed_count",
                                                             "master_seed"});
    if (!unknown) {
        return Result<GreekExperimentConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };

    config.strike = read_number("strike", config.strike);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.spots = read_doubles("spots", config.spots);
    config.maturities = read_doubles("maturities", config.maturities);
    config.volatilities = read_doubles("volatilities", config.volatilities);
    config.spot_bump_fractions = read_doubles("spot_bump_fractions", config.spot_bump_fractions);
    config.volatility_bumps = read_doubles("volatility_bumps", config.volatility_bumps);

    const std::int64_t paths = read_integer("paths", config.paths);
    const std::int64_t seed_count =
        read_integer("seed_count", static_cast<std::int64_t>(config.seed_count));
    const std::int64_t master_seed =
        read_integer("master_seed", static_cast<std::int64_t>(config.master_seed));

    if (first_error.has_value()) {
        return Result<GreekExperimentConfig>::failure(*first_error);
    }

    if (paths < 2 || seed_count < 2 || master_seed < 0) {
        return Result<GreekExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "paths must be at least 2; seed_count at least 2, since bias and variance are measured "
            "across seeds and one seed cannot separate them; and master_seed non-negative",
            kContext);
    }
    config.paths = paths;
    config.seed_count = static_cast<std::uint64_t>(seed_count);
    config.master_seed = static_cast<std::uint64_t>(master_seed);

    if (config.spots.empty() || config.maturities.empty() || config.volatilities.empty()) {
        return Result<GreekExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "spots, maturities, and volatilities must each be non-empty: the comparison exists to "
            "sweep regimes, and a single one would license the universal ranking the experiment "
            "refuses to make",
            kContext);
    }
    if (config.spot_bump_fractions.size() < 3 || config.volatility_bumps.size() < 3) {
        return Result<GreekExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "spot_bump_fractions and volatility_bumps each need at least 3 values to fit the "
            "variance-versus-bump scaling",
            kContext);
    }

    return Result<GreekExperimentConfig>::success(std::move(config));
}

/// Parses the `heston_simulation` block for EXP-10.
///
/// Same contract as the other blocks: absent means the documented defaults that
/// produced the published record, and an unknown key is rejected rather than
/// silently ignored.
Result<HestonSimulationExperimentConfig> parse_heston_simulation_config(const ConfigNode& root) {
    HestonSimulationExperimentConfig config;

    if (!root.contains("heston_simulation")) {
        return Result<HestonSimulationExperimentConfig>::success(config);
    }

    auto node = root.object("heston_simulation");
    if (!node) {
        return Result<HestonSimulationExperimentConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "strike",
                                                             "rate",
                                                             "dividend_yield",
                                                             "maturity",
                                                             "initial_variance",
                                                             "mean_reversion",
                                                             "long_run_variance",
                                                             "correlation",
                                                             "vol_of_variance",
                                                             "step_counts",
                                                             "paths",
                                                             "seed_count",
                                                             "master_seed",
                                                             "bias_resolution"});
    if (!unknown) {
        return Result<HestonSimulationExperimentConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };

    config.spot = read_number("spot", config.spot);
    config.strike = read_number("strike", config.strike);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.maturity = read_number("maturity", config.maturity);
    config.initial_variance = read_number("initial_variance", config.initial_variance);
    config.mean_reversion = read_number("mean_reversion", config.mean_reversion);
    config.long_run_variance = read_number("long_run_variance", config.long_run_variance);
    config.correlation = read_number("correlation", config.correlation);
    config.bias_resolution = read_number("bias_resolution", config.bias_resolution);

    config.vol_of_variance = read_doubles("vol_of_variance", config.vol_of_variance);

    const std::int64_t paths = read_integer("paths", config.paths);
    const std::int64_t seed_count =
        read_integer("seed_count", static_cast<std::int64_t>(config.seed_count));
    const std::int64_t master_seed =
        read_integer("master_seed", static_cast<std::int64_t>(config.master_seed));

    const std::vector<double> steps = read_doubles(
        "step_counts", std::vector<double>(config.step_counts.begin(), config.step_counts.end()));

    if (first_error.has_value()) {
        return Result<HestonSimulationExperimentConfig>::failure(*first_error);
    }

    if (paths < 2 || seed_count < 2 || master_seed < 0) {
        return Result<HestonSimulationExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "paths must be at least 2; seed_count at least 2, since the bias is measured across "
            "seeds and one seed cannot separate it from a lucky draw; and master_seed non-negative",
            kContext);
    }
    if (config.bias_resolution <= 0.0) {
        return Result<HestonSimulationExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "bias_resolution must be positive: it is the number of standard errors a bias must "
            "clear to be fitted rather than treated as noise",
            kContext);
    }
    config.paths = paths;
    config.seed_count = static_cast<std::uint64_t>(seed_count);
    config.master_seed = static_cast<std::uint64_t>(master_seed);

    if (config.vol_of_variance.empty()) {
        return Result<HestonSimulationExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            "vol_of_variance must be non-empty: each entry is a regime, and the experiment needs "
            "at least one -- the catalog asks for both a stable and a difficult one",
            kContext);
    }
    for (const double xi : config.vol_of_variance) {
        if (xi < 0.0) {
            return Result<HestonSimulationExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("vol_of_variance entries must be non-negative, got {}", xi),
                kContext);
        }
    }
    if (steps.size() < 3) {
        return Result<HestonSimulationExperimentConfig>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("heston_simulation.step_counts needs at least 3 levels to see the bias "
                        "decay, got {}",
                        steps.size()),
            kContext);
    }
    std::vector<std::int64_t> step_counts;
    step_counts.reserve(steps.size());
    for (const double s : steps) {
        if (s < 1.0) {
            return Result<HestonSimulationExperimentConfig>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("heston_simulation.step_counts entries must be at least 1, got {}", s),
                kContext);
        }
        step_counts.push_back(static_cast<std::int64_t>(s));
    }
    config.step_counts = std::move(step_counts);

    return Result<HestonSimulationExperimentConfig>::success(std::move(config));
}

/// Parses one Heston parameter vector (an object with the five named fields).
Result<HestonParameters> parse_heston_parameters(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"initial_variance",
                                                     "mean_reversion",
                                                     "long_run_variance",
                                                     "vol_of_variance",
                                                     "correlation"});
    if (!unknown) {
        return Result<HestonParameters>::failure(unknown.error());
    }
    const auto v0 = node.number("initial_variance");
    if (!v0) {
        return Result<HestonParameters>::failure(std::move(v0).error());
    }
    const auto kappa = node.number("mean_reversion");
    if (!kappa) {
        return Result<HestonParameters>::failure(std::move(kappa).error());
    }
    const auto theta = node.number("long_run_variance");
    if (!theta) {
        return Result<HestonParameters>::failure(std::move(theta).error());
    }
    const auto xi = node.number("vol_of_variance");
    if (!xi) {
        return Result<HestonParameters>::failure(std::move(xi).error());
    }
    const auto rho = node.number("correlation");
    if (!rho) {
        return Result<HestonParameters>::failure(std::move(rho).error());
    }
    return Result<HestonParameters>::success(HestonParameters{.initial_variance = v0.value(),
                                                              .mean_reversion = kappa.value(),
                                                              .long_run_variance = theta.value(),
                                                              .vol_of_variance = xi.value(),
                                                              .correlation = rho.value()});
}

/// Parses the `calibration_recovery` block for EXP-11.
Result<CalibrationRecoveryConfig> parse_calibration_recovery_config(const ConfigNode& root) {
    CalibrationRecoveryConfig config;

    if (!root.contains("calibration_recovery")) {
        return Result<CalibrationRecoveryConfig>::success(config);
    }

    auto node = root.object("calibration_recovery");
    if (!node) {
        return Result<CalibrationRecoveryConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "rate",
                                                             "dividend_yield",
                                                             "strikes",
                                                             "maturities",
                                                             "true_parameters",
                                                             "initial_guesses",
                                                             "objective",
                                                             "quadrature_nodes",
                                                             "max_iterations",
                                                             "recovery_tolerance",
                                                             "fit_tolerance_iv_rmse",
                                                             "truth_guess_epsilon"});
    if (!unknown) {
        return Result<CalibrationRecoveryConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };

    config.spot = read_number("spot", config.spot);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.recovery_tolerance = read_number("recovery_tolerance", config.recovery_tolerance);
    config.fit_tolerance_iv_rmse =
        read_number("fit_tolerance_iv_rmse", config.fit_tolerance_iv_rmse);
    config.truth_guess_epsilon = read_number("truth_guess_epsilon", config.truth_guess_epsilon);
    config.quadrature_nodes = read_integer("quadrature_nodes", config.quadrature_nodes);
    config.max_iterations = static_cast<int>(read_integer("max_iterations", config.max_iterations));
    config.strikes = read_doubles("strikes", config.strikes);
    config.maturities = read_doubles("maturities", config.maturities);

    if (first_error.has_value()) {
        return Result<CalibrationRecoveryConfig>::failure(*first_error);
    }

    auto objective = node.value().string_or("objective", std::string("implied_volatility"));
    if (!objective) {
        return Result<CalibrationRecoveryConfig>::failure(std::move(objective).error());
    }
    if (objective.value() == "implied_volatility") {
        config.objective = CalibrationObjectiveType::ImpliedVolatility;
    } else if (objective.value() == "price") {
        config.objective = CalibrationObjectiveType::Price;
    } else {
        return Result<CalibrationRecoveryConfig>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("calibration_recovery.objective must be 'implied_volatility' or 'price', "
                        "got '{}'",
                        objective.value()),
            kContext);
    }

    if (node.value().contains("true_parameters")) {
        auto params_node = node.value().object("true_parameters");
        if (!params_node) {
            return Result<CalibrationRecoveryConfig>::failure(std::move(params_node).error());
        }
        auto parsed = parse_heston_parameters(params_node.value());
        if (!parsed) {
            return Result<CalibrationRecoveryConfig>::failure(std::move(parsed).error());
        }
        config.true_parameters = parsed.value();
    }

    if (node.value().contains("initial_guesses")) {
        auto guesses = node.value().array("initial_guesses");
        if (!guesses) {
            return Result<CalibrationRecoveryConfig>::failure(std::move(guesses).error());
        }
        std::vector<HestonParameters> parsed_guesses;
        for (std::size_t i = 0; i < guesses.value().size(); ++i) {
            auto element = guesses.value().at(i);
            if (!element) {
                return Result<CalibrationRecoveryConfig>::failure(std::move(element).error());
            }
            auto parsed = parse_heston_parameters(element.value());
            if (!parsed) {
                return Result<CalibrationRecoveryConfig>::failure(std::move(parsed).error());
            }
            parsed_guesses.push_back(parsed.value());
        }
        config.initial_guesses = std::move(parsed_guesses);
    }

    if (config.strikes.empty() || config.maturities.empty()) {
        return Result<CalibrationRecoveryConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_recovery needs at least one strike and one maturity",
            kContext);
    }
    if (config.initial_guesses.size() < 2) {
        return Result<CalibrationRecoveryConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_recovery needs at least two initial guesses; the catalog names "
            "calibrating "
            "only from the true parameters as a failure condition",
            kContext);
    }
    if (config.quadrature_nodes < 2 || config.max_iterations < 1) {
        return Result<CalibrationRecoveryConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_recovery needs quadrature_nodes >= 2 and max_iterations >= 1",
            kContext);
    }

    return Result<CalibrationRecoveryConfig>::success(std::move(config));
}

/// Parses the `calibration_stability` block for EXP-12.
Result<MarketSurfaceStabilityConfig> parse_calibration_stability_config(const ConfigNode& root) {
    MarketSurfaceStabilityConfig config;

    if (!root.contains("calibration_stability")) {
        return Result<MarketSurfaceStabilityConfig>::success(config);
    }

    auto node = root.object("calibration_stability");
    if (!node) {
        return Result<MarketSurfaceStabilityConfig>::failure(std::move(node).error());
    }

    const Status unknown = node.value().reject_unknown_keys({"spot",
                                                             "rate",
                                                             "dividend_yield",
                                                             "strikes",
                                                             "maturities",
                                                             "surface_parameters",
                                                             "as_of",
                                                             "initial_guesses",
                                                             "objective",
                                                             "quadrature_nodes",
                                                             "max_iterations"});
    if (!unknown) {
        return Result<MarketSurfaceStabilityConfig>::failure(unknown.error());
    }

    std::optional<Error> first_error;
    const auto read_number = [&](const char* key, double fallback) -> double {
        auto v = node.value().number_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        auto v = node.value().integer_or(key, fallback);
        if (!v) {
            if (!first_error.has_value()) {
                first_error = v.error();
            }
            return fallback;
        }
        return v.value();
    };
    const auto read_doubles = [&](const char* key, std::vector<double> fallback) {
        if (!node.value().contains(key)) {
            return fallback;
        }
        auto array = node.value().array(key);
        if (!array) {
            if (!first_error.has_value()) {
                first_error = array.error();
            }
            return fallback;
        }
        std::vector<double> out;
        for (std::size_t i = 0; i < array.value().size(); ++i) {
            auto v = array.value().number_at(i);
            if (!v) {
                if (!first_error.has_value()) {
                    first_error = v.error();
                }
                return fallback;
            }
            out.push_back(v.value());
        }
        return out;
    };

    config.spot = read_number("spot", config.spot);
    config.rate = read_number("rate", config.rate);
    config.dividend_yield = read_number("dividend_yield", config.dividend_yield);
    config.quadrature_nodes = read_integer("quadrature_nodes", config.quadrature_nodes);
    config.max_iterations = static_cast<int>(read_integer("max_iterations", config.max_iterations));
    config.strikes = read_doubles("strikes", config.strikes);
    config.maturities = read_doubles("maturities", config.maturities);

    if (first_error.has_value()) {
        return Result<MarketSurfaceStabilityConfig>::failure(*first_error);
    }

    auto as_of = node.value().string_or("as_of", config.as_of);
    if (!as_of) {
        return Result<MarketSurfaceStabilityConfig>::failure(std::move(as_of).error());
    }
    config.as_of = as_of.value();

    auto objective = node.value().string_or("objective", std::string("implied_volatility"));
    if (!objective) {
        return Result<MarketSurfaceStabilityConfig>::failure(std::move(objective).error());
    }
    if (objective.value() == "implied_volatility") {
        config.objective = CalibrationObjectiveType::ImpliedVolatility;
    } else if (objective.value() == "price") {
        config.objective = CalibrationObjectiveType::Price;
    } else {
        return Result<MarketSurfaceStabilityConfig>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("calibration_stability.objective must be 'implied_volatility' or 'price', "
                        "got '{}'",
                        objective.value()),
            kContext);
    }

    if (node.value().contains("surface_parameters")) {
        auto params_node = node.value().object("surface_parameters");
        if (!params_node) {
            return Result<MarketSurfaceStabilityConfig>::failure(std::move(params_node).error());
        }
        auto parsed = parse_heston_parameters(params_node.value());
        if (!parsed) {
            return Result<MarketSurfaceStabilityConfig>::failure(std::move(parsed).error());
        }
        config.surface_parameters = parsed.value();
    }

    if (node.value().contains("initial_guesses")) {
        auto guesses = node.value().array("initial_guesses");
        if (!guesses) {
            return Result<MarketSurfaceStabilityConfig>::failure(std::move(guesses).error());
        }
        std::vector<HestonParameters> parsed_guesses;
        for (std::size_t i = 0; i < guesses.value().size(); ++i) {
            auto element = guesses.value().at(i);
            if (!element) {
                return Result<MarketSurfaceStabilityConfig>::failure(std::move(element).error());
            }
            auto parsed = parse_heston_parameters(element.value());
            if (!parsed) {
                return Result<MarketSurfaceStabilityConfig>::failure(std::move(parsed).error());
            }
            parsed_guesses.push_back(parsed.value());
        }
        config.initial_guesses = std::move(parsed_guesses);
    }

    if (config.strikes.empty() || config.maturities.empty()) {
        return Result<MarketSurfaceStabilityConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_stability needs at least one strike and one maturity",
            kContext);
    }
    if (config.initial_guesses.empty()) {
        return Result<MarketSurfaceStabilityConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_stability needs at least one initial guess",
            kContext);
    }
    if (config.quadrature_nodes < 2 || config.max_iterations < 1) {
        return Result<MarketSurfaceStabilityConfig>::failure(
            ErrorCode::InvalidArgument,
            "calibration_stability needs quadrature_nodes >= 2 and max_iterations >= 1",
            kContext);
    }

    return Result<MarketSurfaceStabilityConfig>::success(std::move(config));
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
                    "EXP-02, EXP-03, EXP-04, EXP-06, EXP-07, EXP-08, EXP-10, EXP-11, EXP-12.",
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
    } else if (id == "EXP-06") {
        // EXP-06 is deterministic: it takes no seed and its configuration block is
        // its own, so it does not share the convergence block above.
        auto pde = parse_pde_config(config.root());
        if (!pde) {
            return Result<nlohmann::json>::failure(std::move(pde).error());
        }
        record = run_pde_stability_and_convergence(pde.value());
    } else if (id == "EXP-07") {
        auto barrier = parse_barrier_config(config.root());
        if (!barrier) {
            return Result<nlohmann::json>::failure(std::move(barrier).error());
        }
        BarrierExperimentConfig barrier_config = std::move(barrier).value();
        if (options.seed.has_value()) {
            barrier_config.master_seed = *options.seed;
        }
        record = run_barrier_monitoring_bias(barrier_config);
    } else if (id == "EXP-08") {
        auto greeks = parse_greek_config(config.root());
        if (!greeks) {
            return Result<nlohmann::json>::failure(std::move(greeks).error());
        }
        GreekExperimentConfig greek_config = std::move(greeks).value();
        if (options.seed.has_value()) {
            greek_config.master_seed = *options.seed;
        }
        record = run_greek_estimator_comparison(greek_config);
    } else if (id == "EXP-10") {
        auto heston = parse_heston_simulation_config(config.root());
        if (!heston) {
            return Result<nlohmann::json>::failure(std::move(heston).error());
        }
        HestonSimulationExperimentConfig heston_config = std::move(heston).value();
        if (options.seed.has_value()) {
            heston_config.master_seed = *options.seed;
        }
        record = run_heston_variance_discretization(heston_config);
    } else if (id == "EXP-11") {
        auto calibration = parse_calibration_recovery_config(config.root());
        if (!calibration) {
            return Result<nlohmann::json>::failure(std::move(calibration).error());
        }
        record = run_heston_calibration_recovery(calibration.value());
    } else if (id == "EXP-12") {
        auto stability = parse_calibration_stability_config(config.root());
        if (!stability) {
            return Result<nlohmann::json>::failure(std::move(stability).error());
        }
        record = run_market_surface_stability(stability.value());
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
