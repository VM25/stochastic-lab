#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include "options.hpp"
#include "simulate_command.hpp"

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace diffusionworks::cli {
namespace {

nlohmann::json base_config() {
    return nlohmann::json{
        {"schema_version", 1},
        {"command", "simulate"},
        {"market", {{"spot", 100.0}, {"rate", 0.05}, {"dividend_yield", 0.02}}},
        {"model", {{"type", "black_scholes"}, {"volatility", 0.30}}},
        {"maturity", 2.0},
        {"method",
         {{"type", "monte_carlo"},
          {"paths", 50000},
          {"steps", 8},
          {"scheme", "exact"},
          {"seed", 20260715}}},
    };
}

Result<nlohmann::json> run(const nlohmann::json& config, Options options = Options{}) {
    auto document = parse_config(config.dump());
    EXPECT_TRUE(document.ok()) << document.error().describe();
    options.command = CommandKind::Simulate;
    return run_simulate(document.value(), options);
}

// ---------------------------------------------------------------------------
// The comparison against theory, which is the reason simulate exists
// ---------------------------------------------------------------------------

TEST(SimulateCommandTest, ComparesTerminalMomentsAgainstTheClosedForm) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& terminal = result.value()["result"]["terminal"];
    ASSERT_TRUE(terminal.contains("theoretical_mean"));

    // E[S_T] = S_0 e^{(r-q)T} = 100 e^{0.06}
    EXPECT_NEAR(terminal["theoretical_mean"].get<double>(), 100.0 * std::exp(0.06), 1e-10);

    // Var[S_T] = S_0^2 e^{2(r-q)T}(e^{sigma^2 T} - 1)
    const double expected_variance = 10000.0 * std::exp(0.12) * std::expm1(0.09 * 2.0);
    EXPECT_NEAR(terminal["theoretical_variance"].get<double>(), expected_variance, 1e-6);

    // The deviation is reported in standard errors, because an absolute
    // difference cannot be judged without knowing the sampling noise.
    ASSERT_TRUE(terminal.contains("mean_deviation_in_standard_errors"));
    EXPECT_LT(std::abs(terminal["mean_deviation_in_standard_errors"].get<double>()), 4.0)
        << "the exact scheme should sit within sampling noise of theory";
}

TEST(SimulateCommandTest, ReportsLogReturnMomentsAgainstTheory) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    ASSERT_TRUE(result.value()["result"].contains("log_return"));
    const nlohmann::json& log_return = result.value()["result"]["log_return"];

    // log(S_T/S_0) ~ N((r - q - sigma^2/2)T, sigma^2 T)
    EXPECT_NEAR(log_return["theoretical_mean"].get<double>(), (0.03 - 0.045) * 2.0, 1e-12);
    EXPECT_NEAR(log_return["theoretical_variance"].get<double>(), 0.09 * 2.0, 1e-12);

    // The log domain isolates drift and diffusion without the lognormal's heavy
    // tail inflating the error, so it should agree closely.
    const double standard_error = std::sqrt(0.18 / 50000.0);
    EXPECT_NEAR(
        log_return["sample_mean"].get<double>(), (0.03 - 0.045) * 2.0, 4.0 * standard_error);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

TEST(SimulateCommandTest, ExactSchemeReportsNoCrossingsAndNoWarnings) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    EXPECT_EQ(result.value()["result"]["diagnostics"]["non_positive_states"].get<std::int64_t>(), 0)
        << "the exact scheme steps log-price and cannot cross zero";
    EXPECT_TRUE(result.value()["result"]["warnings"].empty());
}

// The explicit schemes can cross zero, and simulate exists to say so. A price
// would hide it: the payoff clamps and returns an ordinary number.
TEST(SimulateCommandTest, ReportsAndWarnsWhenPathsCrossZero) {
    nlohmann::json config = base_config();
    config["method"]["scheme"] = "euler_maruyama";
    config["method"]["steps"] = 2;
    config["model"]["volatility"] = 1.2;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& diagnostics = result.value()["result"]["diagnostics"];
    EXPECT_GT(diagnostics["non_positive_states"].get<std::int64_t>(), 0)
        << "Euler was expected to cross zero at this volatility and step size";
    EXPECT_GT(diagnostics["paths_with_non_positive_states"].get<std::int64_t>(), 0);

    EXPECT_FALSE(result.value()["result"]["warnings"].empty())
        << "crossings were counted but not announced";
}

// A truncated log-return sample must say so, or its moments would be read as
// describing every path.
TEST(SimulateCommandTest, DeclaresWhenLogMomentsCoverATruncatedSample) {
    nlohmann::json config = base_config();
    config["method"]["scheme"] = "euler_maruyama";
    config["method"]["steps"] = 2;
    config["model"]["volatility"] = 1.2;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const std::int64_t failures =
        result.value()["result"]["diagnostics"]["log_domain_failures"].get<std::int64_t>();
    if (failures == 0) {
        GTEST_SKIP() << "no path ended non-positive in this configuration";
    }

    const std::int64_t included =
        result.value()["result"]["log_return"]["paths_included"].get<std::int64_t>();
    EXPECT_EQ(included + failures, config["method"]["paths"].get<std::int64_t>())
        << "every path must be either included or counted as a failure, never quietly dropped";

    bool declared = false;
    for (const auto& warning : result.value()["result"]["warnings"]) {
        if (warning.get<std::string>().find("truncated") != std::string::npos) {
            declared = true;
        }
    }
    EXPECT_TRUE(declared) << "the log-return moments describe a truncated sample without saying so";
}

// ---------------------------------------------------------------------------
// Reproducibility and rejection
// ---------------------------------------------------------------------------

TEST(SimulateCommandTest, SameSeedReproducesExactly) {
    const auto first = run(base_config());
    const auto second = run(base_config());

    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value()["result"]["terminal"]["sample_mean"],
              second.value()["result"]["terminal"]["sample_mean"]);
}

// A seed chosen implicitly cannot be reproduced from the record the run leaves
// behind, so there is no default.
TEST(SimulateCommandTest, RequiresASeed) {
    nlohmann::json config = base_config();
    config["method"].erase("seed");

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(result.error().message.find("seed"), std::string::npos);
}

// --seed must beat the file, and the precedence must live in one place.
TEST(SimulateCommandTest, CommandLineSeedOverridesTheFile) {
    Options options;
    options.seed = 999;

    const auto overridden = run(base_config(), options);
    const auto from_file = run(base_config());

    ASSERT_TRUE(overridden.ok()) << overridden.error().describe();
    ASSERT_TRUE(from_file.ok());

    EXPECT_EQ(overridden.value()["method"]["seed"].get<std::uint64_t>(), 999U);
    EXPECT_NE(overridden.value()["result"]["terminal"]["sample_mean"],
              from_file.value()["result"]["terminal"]["sample_mean"]);
}

TEST(SimulateCommandTest, RejectsUnknownFieldsAndBadMethods) {
    nlohmann::json unknown = base_config();
    unknown["bogus"] = 1;
    EXPECT_FALSE(run(unknown).ok());

    nlohmann::json analytic = base_config();
    analytic["method"]["type"] = "analytic";
    const auto result = run(analytic);
    ASSERT_FALSE(result.ok()) << "simulate generates paths and cannot use an analytic method";
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);

    nlohmann::json bad_scheme = base_config();
    bad_scheme["method"]["scheme"] = "runge_kutta";
    EXPECT_FALSE(run(bad_scheme).ok());
}

TEST(SimulateCommandTest, RejectsAMaturityThatCannotBeGridded) {
    nlohmann::json config = base_config();
    config["maturity"] = 0.0;
    EXPECT_FALSE(run(config).ok());

    config["maturity"] = -1.0;
    EXPECT_FALSE(run(config).ok());
}

TEST(SimulateCommandTest, CarriesItsConfigurationAndProvenance) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok());

    EXPECT_EQ(result.value()["configuration"], base_config());
    ASSERT_TRUE(result.value().contains("build_metadata"));
    EXPECT_TRUE(result.value()["build_metadata"].contains("git_commit"));

    // The scheme, seed, and step count are what make the run reproducible.
    const nlohmann::json& method = result.value()["method"];
    EXPECT_EQ(method["scheme"], "exact");
    EXPECT_EQ(method["seed"].get<std::uint64_t>(), 20260715U);
    EXPECT_EQ(method["steps"].get<std::int64_t>(), 8);
}

}  // namespace
}  // namespace diffusionworks::cli
