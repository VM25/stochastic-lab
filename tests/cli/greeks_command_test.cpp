#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include "greeks_command.hpp"
#include "options.hpp"

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks::cli {
namespace {

/// A valid `greeks` configuration, mutable per test. S=K=100, r=5%, sigma=20%, T=1,
/// whose analytic call delta is 0.636831 -- so an end-to-end run is checked against
/// the same reference as the engine's unit tests.
nlohmann::json base_config() {
    return nlohmann::json{
        {"schema_version", 1},
        {"command", "greeks"},
        {"market", {{"spot", 100.0}, {"rate", 0.05}, {"dividend_yield", 0.0}}},
        {"instrument",
         {{"type", "european"}, {"option_type", "call"}, {"strike", 100.0}, {"maturity", 1.0}}},
        {"model", {{"type", "black_scholes"}, {"volatility", 0.20}}},
        {"greeks",
         {{"greek", "delta"},
          {"method", "pathwise"},
          {"paths", 20000},
          {"seed_count", 8},
          {"master_seed", 20260717}}},
    };
}

Result<nlohmann::json> run(const nlohmann::json& config, Options options = Options{}) {
    auto document = parse_config(config.dump());
    EXPECT_TRUE(document.ok()) << document.error().describe();
    options.command = CommandKind::Greeks;
    return run_greeks(document.value(), options);
}

// The structured output the exit gate requires: estimator, Greek, bump, estimate,
// uncertainty, the seed set, runtime, status, and (when present) warnings.
TEST(GreeksCommandTest, ProducesTheRequiredStructuredFields) {
    const auto document = run(base_config());
    ASSERT_TRUE(document.ok()) << document.error().describe();

    EXPECT_EQ(document.value().at("command"), "greeks");
    EXPECT_EQ(document.value().at("status"), "ok");

    const auto& result = document.value().at("result");
    for (const char* field : {"greek",
                              "estimator",
                              "bump",
                              "estimate",
                              "standard_error",
                              "across_seed_standard_deviation",
                              "seed_set",
                              "seed_count",
                              "paths_per_seed",
                              "runtime_seconds"}) {
        EXPECT_TRUE(result.contains(field)) << "missing field: " << field;
    }
    EXPECT_EQ(result.at("greek"), "delta");
    EXPECT_EQ(result.at("estimator"), "pathwise");
    EXPECT_EQ(result.at("seed_set").size(), 8U);
    // Pathwise takes no bump, so it is reported as null rather than a misleading zero.
    EXPECT_TRUE(result.at("bump").is_null());
}

// The estimate agrees with the analytic delta to within its reported across-seed
// uncertainty -- the same criterion the engine tests use, exercised through the CLI.
TEST(GreeksCommandTest, EstimateAgreesWithTheAnalyticReference) {
    const auto document = run(base_config());
    ASSERT_TRUE(document.ok());
    const auto& result = document.value().at("result");
    const double estimate = result.at("estimate").get<double>();
    const double se = result.at("standard_error").get<double>();
    ASSERT_GT(se, 0.0);
    EXPECT_LT(std::abs(estimate - 0.636831), 5.0 * se);
}

// The finite-difference method reports the bump it used; the other methods do not.
TEST(GreeksCommandTest, FiniteDifferenceReportsItsBump) {
    auto config = base_config();
    config["greeks"]["method"] = "finite_difference";
    config["greeks"]["spot_bump_fraction"] = 0.02;
    const auto document = run(config);
    ASSERT_TRUE(document.ok()) << document.error().describe();
    EXPECT_FALSE(document.value().at("result").at("bump").is_null());
    EXPECT_DOUBLE_EQ(document.value().at("result").at("bump").get<double>(), 2.0);  // 2% of 100
}

// A degenerate boundary is handled per method: finite difference succeeds with a
// warning and a deterministic estimate, and the document's status reflects it.
TEST(GreeksCommandTest, FiniteDifferenceAtZeroVolatilityWarns) {
    auto config = base_config();
    config["greeks"]["method"] = "finite_difference";
    config["market"]["spot"] = 120.0;  // deep in the money, away from the kink
    config["model"]["volatility"] = 0.0;
    const auto document = run(config);
    ASSERT_TRUE(document.ok()) << document.error().describe();
    EXPECT_EQ(document.value().at("status"), "warning");
    EXPECT_TRUE(document.value().contains("warnings"));
    EXPECT_NEAR(document.value().at("result").at("estimate").get<double>(), 1.0, 1e-9);
}

// An unsupported combination is surfaced as the error it is, not a silent zero.
TEST(GreeksCommandTest, RefusesPathwiseGamma) {
    auto config = base_config();
    config["greeks"]["greek"] = "gamma";
    config["greeks"]["method"] = "pathwise";
    const auto document = run(config);
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::UnsupportedCombination);
}

TEST(GreeksCommandTest, RejectsUnknownGreekOrMethod) {
    auto bad_greek = base_config();
    bad_greek["greeks"]["greek"] = "theta";
    EXPECT_FALSE(run(bad_greek).ok());

    auto bad_method = base_config();
    bad_method["greeks"]["method"] = "malliavin";
    EXPECT_FALSE(run(bad_method).ok());
}

// The uncertainty is measured across the seed set, so a single seed cannot provide
// it and is refused rather than reporting a fabricated zero.
TEST(GreeksCommandTest, RejectsASingleSeed) {
    auto config = base_config();
    config["greeks"]["seed_count"] = 1;
    const auto document = run(config);
    ASSERT_FALSE(document.ok());
    EXPECT_EQ(document.error().code, ErrorCode::InvalidConfiguration);
}

// An explicit seed set is honoured verbatim and echoed back, so the run reproduces
// from its own record.
TEST(GreeksCommandTest, HonoursAnExplicitSeedSet) {
    auto config = base_config();
    config["greeks"].erase("seed_count");
    config["greeks"].erase("master_seed");
    config["greeks"]["seeds"] = nlohmann::json::array({11, 22, 33});
    const auto document = run(config);
    ASSERT_TRUE(document.ok()) << document.error().describe();

    const auto seed_set =
        document.value().at("result").at("seed_set").get<std::vector<std::uint64_t>>();
    EXPECT_EQ(seed_set, (std::vector<std::uint64_t>{11, 22, 33}));
}

// The command line's --seed overrides the master seed, so a sweep needs no per-seed
// file. The generated set must then start from that override.
TEST(GreeksCommandTest, CommandLineSeedOverridesTheMasterSeed) {
    Options options;
    options.seed = 999;
    const auto document = run(base_config(), options);
    ASSERT_TRUE(document.ok());
    EXPECT_EQ(document.value().at("result").at("seed_set").at(0).get<std::uint64_t>(), 999U);
}

}  // namespace
}  // namespace diffusionworks::cli
