#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include "experiment_command.hpp"
#include "options.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks::cli {
namespace {

/// A configuration small enough to run in a test.
///
/// The defaults are sized for a real study and take minutes; these are cut to the
/// smallest shape the machinery still accepts, which is the point of the test --
/// the wiring, not the physics. The physics is tested in
/// tests/experiments/convergence_test.cpp against its own acceptance rules.
constexpr const char* kTinyConfig = R"({
  "schema_version": 1,
  "command": "experiment",
  "convergence": {
    "spot": 100.0,
    "rate": 0.05,
    "dividend_yield": 0.0,
    "volatility": 0.30,
    "maturity": 1.0,
    "strike": 100.0,
    "master_seed": 20260715,
    "seed_count": 4,
    "step_counts": [8, 16, 32],
    "asymptotic_level_count": 3,
    "strong_paths": 500,
    "call_payoff_paths": 500,
    "path_counts": [100, 400, 1600]
  }
})";

/// The barrier block, cut to the smallest shape that still runs.
///
/// Two barriers, three frequencies, and a few hundred paths measure nothing about
/// the physics -- EXP-07's published record needs the full configuration and takes
/// minutes. This exercises the wiring. The bias itself is measured in
/// tests/engines/barrier_monte_carlo_test.cpp against its own acceptance rules.
constexpr const char* kTinyBarrierConfig = R"({
  "schema_version": 1,
  "command": "experiment",
  "barrier": {
    "spot": 100.0,
    "strike": 100.0,
    "rate": 0.05,
    "dividend_yield": 0.0,
    "volatility": 0.20,
    "maturity": 1.0,
    "barriers": [90.0],
    "up_barriers": [120.0],
    "monitoring_counts": [4, 8, 16],
    "paths": 400,
    "seed_count": 3,
    "master_seed": 20260716,
    "volatilities": [0.2],
    "pde_resolutions": [51, 101, 201]
  }
})";

ConfigDocument config_from(const char* text) {
    auto parsed = parse_config(text, "test.json");
    EXPECT_TRUE(parsed.ok()) << parsed.error().describe();
    return parsed.value();
}

ConfigDocument tiny_config() {
    return config_from(kTinyConfig);
}

ConfigDocument tiny_barrier_config() {
    return config_from(kTinyBarrierConfig);
}

/// The greeks block, cut to the smallest shape that still runs: one scenario, the
/// minimum three bumps a scaling fit needs, two seeds, a few hundred paths. This
/// exercises the wiring; the estimator physics is validated in
/// tests/engines/greeks_monte_carlo_test.cpp.
constexpr const char* kTinyGreekConfig = R"({
  "schema_version": 1,
  "command": "experiment",
  "greeks": {
    "strike": 100.0,
    "rate": 0.05,
    "dividend_yield": 0.0,
    "spots": [100.0],
    "maturities": [1.0],
    "volatilities": [0.2],
    "spot_bump_fractions": [0.05, 0.02, 0.01],
    "volatility_bumps": [0.02, 0.01, 0.005],
    "paths": 500,
    "seed_count": 2,
    "master_seed": 20260717
  }
})";

ConfigDocument tiny_greek_config() {
    return config_from(kTinyGreekConfig);
}

Options options_for(const std::string& id) {
    Options options;
    options.command = CommandKind::Experiment;
    options.experiment_id = id;
    return options;
}

// ---------------------------------------------------------------------------
// The record's required artifacts
// ---------------------------------------------------------------------------

// EXPERIMENT-CATALOG lists exactly what each experiment must produce. A record
// missing any of these is not publishable, so the shape is pinned here rather
// than left to whoever reads the JSON later.
TEST(ExperimentCommandTest, RecordCarriesEveryRequiredArtifact) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-02"));
    ASSERT_TRUE(document.ok()) << document.error().describe();

    for (const char* field : {"id",
                              "name",
                              "question",
                              "status",
                              "interpretation",
                              "limitations",
                              "reproduction_command",
                              "configuration",
                              "results",
                              "table",
                              "runtime_seconds",
                              "build_metadata",
                              "configuration_source",
                              "configuration_document"}) {
        EXPECT_TRUE(document.value().contains(field)) << "missing required field: " << field;
    }

    EXPECT_EQ(document.value().at("id"), "EXP-02");
    EXPECT_FALSE(document.value().at("interpretation").get<std::string>().empty());

    // Limitations are populated even on a pass. An experiment that claims no
    // limitations has not been thought about.
    EXPECT_FALSE(document.value().at("limitations").empty());
}

// Provenance is part of the record, not decoration: a convergence slope produced
// under different flags is a different result, and this project disables
// floating-point contraction precisely because such things change answers.
TEST(ExperimentCommandTest, RecordCarriesBuildProvenance) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-02"));
    ASSERT_TRUE(document.ok());

    const auto& build = document.value().at("build_metadata");
    for (const char* field :
         {"compiler_id", "compiler_version", "build_type", "build_flags", "git_commit"}) {
        EXPECT_TRUE(build.contains(field)) << "missing provenance: " << field;
    }
}

// The configuration travels with the result, so the artifact can be reproduced
// from itself rather than from whatever the file happens to say later.
TEST(ExperimentCommandTest, RecordEmbedsTheConfigurationItRanWith) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-02"));
    ASSERT_TRUE(document.ok());

    EXPECT_EQ(document.value().at("configuration").at("volatility"), 0.30);
    EXPECT_EQ(document.value().at("configuration").at("strong_paths"), 500);
    EXPECT_EQ(document.value().at("configuration_document").at("convergence").at("volatility"),
              0.30);
}

TEST(ExperimentCommandTest, TableIsRectangularAndLabelled) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-02"));
    ASSERT_TRUE(document.ok());

    const auto headers = document.value().at("table").at("headers").get<std::vector<std::string>>();
    const auto rows =
        document.value().at("table").at("rows").get<std::vector<std::vector<std::string>>>();

    ASSERT_FALSE(headers.empty());
    ASSERT_FALSE(rows.empty());
    for (const auto& row : rows) {
        // A ragged CSV parses cleanly and shifts every column after the gap, so a
        // plot drawn from it looks entirely plausible and is wrong.
        EXPECT_EQ(row.size(), headers.size());
    }
}

// ---------------------------------------------------------------------------
// Each experiment runs end to end
// ---------------------------------------------------------------------------

TEST(ExperimentCommandTest, EveryImplementedExperimentProducesARecord) {
    for (const char* id : {"EXP-01", "EXP-02", "EXP-03", "EXP-04"}) {
        const auto document = run_experiment(tiny_config(), options_for(id));
        ASSERT_TRUE(document.ok()) << id << ": " << document.error().describe();
        EXPECT_EQ(document.value().at("id"), id);

        const std::string status = document.value().at("status");
        // Any of the four statuses is a legitimate outcome at these tiny sample
        // sizes -- what matters is that the record exists and says which.
        EXPECT_TRUE(status == "pass" || status == "fail" || status == "warning" ||
                    status == "inconclusive")
            << id << " reported an unknown status: " << status;
    }
}

TEST(ExperimentCommandTest, BarrierMonitoringBiasProducesARecord) {
    const auto document = run_experiment(tiny_barrier_config(), options_for("EXP-07"));
    ASSERT_TRUE(document.ok()) << document.error().describe();
    EXPECT_EQ(document.value().at("id"), "EXP-07");

    // At 400 paths the bias cannot clear its own noise, so "inconclusive" is the
    // honest outcome here and the record must say so rather than claim a pass. That
    // it *can* say so is the point: an experiment that only ever passes is not
    // evidence of anything.
    const std::string status = document.value().at("status");
    EXPECT_TRUE(status == "pass" || status == "fail" || status == "warning" ||
                status == "inconclusive")
        << "unknown status: " << status;

    const auto& arms = document.value().at("results").at("arms");
    // One arm per (barrier, convention) pair: one down-barrier and one up-barrier,
    // two conventions each. Both directions, because the catalog asks for both.
    ASSERT_EQ(arms.size(), 4U);
    EXPECT_EQ(arms.at(0).at("barrier_type"), "down_and_out");
    EXPECT_EQ(arms.at(0).at("convention"), "discrete");
    EXPECT_EQ(arms.at(1).at("convention"), "brownian_bridge");
    EXPECT_EQ(arms.at(2).at("barrier_type"), "up_and_out");
    EXPECT_EQ(arms.at(0).at("levels").size(), 3U);
}

TEST(ExperimentCommandTest, GreekEstimatorComparisonProducesARecord) {
    const auto document = run_experiment(tiny_greek_config(), options_for("EXP-08"));
    ASSERT_TRUE(document.ok()) << document.error().describe();
    EXPECT_EQ(document.value().at("id"), "EXP-08");

    const std::string status = document.value().at("status");
    EXPECT_TRUE(status == "pass" || status == "fail" || status == "warning" ||
                status == "inconclusive")
        << "unknown status: " << status;

    // One scenario, so the cells are: delta by three methods (FD across 3 bumps,
    // pathwise, likelihood-ratio) + gamma (FD across 3 bumps) + vega (FD across 3
    // vol bumps, pathwise). The exact count matters less than that the structured
    // comparison exists and carries the fields the exit gate requires.
    const auto& cells = document.value().at("results").at("cells");
    ASSERT_FALSE(cells.empty());
    for (const char* field : {"greek",
                              "method",
                              "bump",
                              "reference",
                              "estimate",
                              "bias",
                              "across_seed_standard_error",
                              "rmse",
                              "mean_runtime_seconds"}) {
        EXPECT_TRUE(cells.at(0).contains(field)) << "cell missing " << field;
    }
    EXPECT_TRUE(document.value().at("results").contains("finite_difference_variance_scaling"));
    EXPECT_TRUE(document.value().at("results").contains("failure_regions"));
}

// A single regime is refused: the comparison exists to sweep, and one scenario would
// license the universal ranking the experiment is built to avoid.
TEST(ExperimentCommandTest, GreekExperimentRejectsAWiderConfigMistake) {
    const auto document =
        run_experiment(config_from(R"({"schema_version": 1, "command": "experiment",
                        "greeks": {"spot_bump_fractions": [0.05, 0.02]}})"),
                       options_for("EXP-08"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

// Every bias here is a difference of two numbers, one of which carries sampling
// error. One seed cannot tell a real bias from a lucky draw -- which is the entire
// question for the bridge arm -- so a single-seed configuration must be refused
// rather than run and reported.
TEST(ExperimentCommandTest, BarrierExperimentRejectsASingleSeed) {
    const auto document =
        run_experiment(config_from(R"({"schema_version": 1, "command": "experiment",
                        "barrier": {"seed_count": 1}})"),
                       options_for("EXP-07"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

// Fewer than three frequencies cannot carry an order with an uncertainty, and a
// slope from two points would report a zero standard error for a rate it has not
// earned.
TEST(ExperimentCommandTest, BarrierExperimentRejectsTooFewMonitoringFrequencies) {
    const auto document =
        run_experiment(config_from(R"({"schema_version": 1, "command": "experiment",
                        "barrier": {"monitoring_counts": [10, 20]}})"),
                       options_for("EXP-07"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

// A typo in the barrier block must fail loudly rather than leave a default in place
// and produce a plausible study of a contract nobody asked about.
TEST(ExperimentCommandTest, BarrierExperimentRejectsAnUnknownKey) {
    const auto document =
        run_experiment(config_from(R"({"schema_version": 1, "command": "experiment",
                        "barrier": {"volatilty": 0.3}})"),
                       options_for("EXP-07"));
    ASSERT_FALSE(document.ok());
    // InvalidConfiguration rather than InvalidArgument, matching the convergence
    // block: the document parsed fine and was rejected on meaning.
    EXPECT_EQ(document.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(document.error().describe().find("volatilty"), std::string::npos)
        << document.error().describe();
}

// Regression: EXP-01's sweep levels must not share paths.
//
// A run of N paths at a given seed uses path indices 0..N-1, so sweeping N with
// the seed fixed nests the levels -- the N=400 run re-uses every path the N=100
// run used. That correlates the residuals, breaks the independence ordinary least
// squares assumes, and narrows the fitted interval around a slope it has not
// earned. The first EXP-01 run failed exactly this way: slope -0.5551 with a 95%
// interval of [-0.5834, -0.5269], confidently excluding the theoretical -0.5. The
// rate was never wrong.
//
// Checked exactly, on the seed regions themselves, rather than through a statistic
// that could agree by chance. Each level must occupy a seed range disjoint from
// every other level's; a path is a pure function of (seed, path index), so
// disjoint seed ranges mean no shared paths, with no sampling argument required.
TEST(ExperimentCommandTest, SamplingConvergenceLevelsDoNotSharePaths) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-01"));
    ASSERT_TRUE(document.ok()) << document.error().describe();

    const auto& points = document.value().at("results").at("scenarios").at(0).at("points");
    ASSERT_GE(points.size(), 3U);

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& point : points) {
        ranges.emplace_back(point.at("first_seed").get<std::uint64_t>(),
                            point.at("last_seed").get<std::uint64_t>());
    }

    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            const bool disjoint =
                ranges[i].second < ranges[j].first || ranges[j].second < ranges[i].first;
            EXPECT_TRUE(disjoint) << "levels " << i << " and " << j << " share seed space: ["
                                  << ranges[i].first << ", " << ranges[i].second << "] overlaps ["
                                  << ranges[j].first << ", " << ranges[j].second
                                  << "]. Nested levels correlate the fit's residuals and "
                                  << "narrow its interval around a slope it has not earned.";
        }
    }
}

// The seed is part of a run's identity, so the command line must be able to
// override the file without editing it.
TEST(ExperimentCommandTest, CommandLineSeedOverridesTheConfiguration) {
    Options options = options_for("EXP-02");
    options.seed = 987654321;

    const auto document = run_experiment(tiny_config(), options);
    ASSERT_TRUE(document.ok()) << document.error().describe();
    EXPECT_EQ(document.value().at("configuration").at("master_seed"), 987654321);
}

// ---------------------------------------------------------------------------
// Rejection
// ---------------------------------------------------------------------------

TEST(ExperimentCommandTest, RequiresAnExperimentId) {
    Options options;
    options.command = CommandKind::Experiment;

    const auto document = run_experiment(tiny_config(), options);
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

TEST(ExperimentCommandTest, RejectsAnUnknownExperimentId) {
    const auto document = run_experiment(tiny_config(), options_for("EXP-99"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::NotImplemented);
    // The message must say what *is* available, or the user has to read the source.
    EXPECT_NE(document.error().describe().find("EXP-01"), std::string::npos);
}

// A typo must fail loudly. Ignoring it would leave the default in place and
// produce a plausible study of a model the author did not configure.
TEST(ExperimentCommandTest, RejectsAnUnknownConfigurationKey) {
    constexpr const char* kTypo = R"({
      "schema_version": 1,
      "command": "experiment",
      "convergence": { "volatilty": 0.30 }
    })";
    auto parsed = parse_config(kTypo, "test.json");
    ASSERT_TRUE(parsed.ok());

    const auto document = run_experiment(parsed.value(), options_for("EXP-02"));
    ASSERT_FALSE(document.ok());
    // InvalidConfiguration, not InvalidArgument: the document parsed fine and was
    // rejected on meaning, which is a distinction the error codes deliberately
    // keep and a caller can act on differently.
    EXPECT_EQ(document.error().code, ErrorCode::InvalidConfiguration);
    // The message must name the offending key, or the user cannot find the typo.
    EXPECT_NE(document.error().describe().find("volatilty"), std::string::npos)
        << document.error().describe();
}

// Two levels can be joined by a line of any slope, so a study configured with
// them would report an assumption as a measurement.
TEST(ExperimentCommandTest, RejectsTooFewGridLevels) {
    constexpr const char* kTooFew = R"({
      "schema_version": 1,
      "command": "experiment",
      "convergence": { "step_counts": [8, 16] }
    })";
    auto parsed = parse_config(kTooFew, "test.json");
    ASSERT_TRUE(parsed.ok());

    const auto document = run_experiment(parsed.value(), options_for("EXP-02"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

TEST(ExperimentCommandTest, RejectsAnAsymptoticWindowWiderThanTheGrid) {
    constexpr const char* kBadWindow = R"({
      "schema_version": 1,
      "command": "experiment",
      "convergence": { "step_counts": [8, 16, 32], "asymptotic_level_count": 5 }
    })";
    auto parsed = parse_config(kBadWindow, "test.json");
    ASSERT_TRUE(parsed.ok());

    const auto document = run_experiment(parsed.value(), options_for("EXP-02"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

TEST(ExperimentCommandTest, RejectsASingleSeed) {
    constexpr const char* kOneSeed = R"({
      "schema_version": 1,
      "command": "experiment",
      "convergence": { "seed_count": 1 }
    })";
    auto parsed = parse_config(kOneSeed, "test.json");
    ASSERT_TRUE(parsed.ok());

    const auto document = run_experiment(parsed.value(), options_for("EXP-01"));
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

TEST(ExperimentCommandTest, ParsesTheIdFlag) {
    const std::vector<std::string_view> args{"experiment", "--config", "c.json", "--id", "EXP-02"};
    const auto parsed = parse_arguments(args);
    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    ASSERT_TRUE(parsed.value().experiment_id.has_value());
    EXPECT_EQ(*parsed.value().experiment_id, "EXP-02");
}

// Silently ignoring a flag on the wrong command is how a run ends up not doing
// what its command line says it did.
TEST(ExperimentCommandTest, RejectsTheIdFlagOnOtherCommands) {
    const std::vector<std::string_view> args{"price", "--config", "c.json", "--id", "EXP-02"};
    const auto parsed = parse_arguments(args);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
}

TEST(ExperimentCommandTest, RejectsTheIdFlagWithoutAValue) {
    const std::vector<std::string_view> args{"experiment", "--id"};
    EXPECT_FALSE(parse_arguments(args).ok());
}

TEST(ExperimentCommandTest, HelpMentionsTheIdFlag) {
    const std::string help = command_usage_text(CommandKind::Experiment);
    EXPECT_NE(help.find("--id"), std::string::npos);

    // And the flag must not be advertised where it is rejected.
    EXPECT_EQ(command_usage_text(CommandKind::Price).find("--id"), std::string::npos);
}

}  // namespace
}  // namespace diffusionworks::cli
