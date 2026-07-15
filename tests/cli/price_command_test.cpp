#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include "options.hpp"
#include "output.hpp"
#include "price_command.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks::cli {
namespace {

/// A valid `price` configuration, mutable per test.
///
/// Uses Hull 9th ed. Example 15.6 (S=42, K=40, r=10%, sigma=20%, T=0.5), whose
/// published call value is 4.76, so an end-to-end run is checked against the
/// same published source as the engine's unit tests rather than against itself.
nlohmann::json base_config() {
    return nlohmann::json{
        {"schema_version", 1},
        {"command", "price"},
        {"market", {{"spot", 42.0}, {"rate", 0.10}, {"dividend_yield", 0.0}}},
        {"instrument",
         {{"type", "european"}, {"option_type", "call"}, {"strike", 40.0}, {"maturity", 0.5}}},
        {"model", {{"type", "black_scholes"}, {"volatility", 0.20}}},
        {"method", {{"type", "analytic"}}},
        {"greeks", true},
    };
}

Result<nlohmann::json> run(const nlohmann::json& config, Options options = Options{}) {
    auto document = parse_config(config.dump());
    EXPECT_TRUE(document.ok()) << document.error().describe();
    options.command = CommandKind::Price;
    return run_price(document.value(), options);
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

TEST(PriceCommandTest, PricesAgainstPublishedReference) {
    const auto result = run(base_config());

    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_EQ(result.value()["status"], "ok");
    EXPECT_NEAR(result.value()["result"]["value"].get<double>(), 4.76, 0.005);
}

// TECHNICAL-DESIGN section 19 fixes the output schema. These fields are what make
// a published number auditable, so their presence is pinned rather than assumed.
TEST(PriceCommandTest, DocumentCarriesRequiredMetadata) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();
    const nlohmann::json& document = result.value();

    for (const std::string field : {"status",
                                    "command",
                                    "market",
                                    "instrument",
                                    "model",
                                    "method",
                                    "result",
                                    "build_metadata",
                                    "configuration"}) {
        EXPECT_TRUE(document.contains(field)) << "missing required field: " << field;
    }

    const nlohmann::json& build = document["build_metadata"];
    for (const std::string field : {"compiler_id",
                                    "compiler_version",
                                    "build_type",
                                    "build_flags",
                                    "cxx_standard",
                                    "git_commit",
                                    "os_name",
                                    "cpu_brand",
                                    "logical_cores",
                                    "timestamp_utc"}) {
        EXPECT_TRUE(build.contains(field)) << "missing build metadata: " << field;
    }
}

// A result must carry the inputs that produced it, so that a published number can
// be re-run without hunting for the file behind it.
TEST(PriceCommandTest, DocumentEmbedsItsOwnConfiguration) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    EXPECT_EQ(result.value()["configuration"], base_config());
}

TEST(PriceCommandTest, ReportsGreeksWhenRequested) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& greeks = result.value()["result"]["greeks"];
    for (const std::string name : {"delta", "gamma", "vega", "theta", "rho"}) {
        EXPECT_TRUE(greeks.contains(name)) << "missing Greek: " << name;
        // Units travel with each number; a bare vega is ambiguous by a factor of
        // 100 between per-unit and per-point.
        EXPECT_TRUE(greeks["units"].contains(name)) << "missing units for: " << name;
    }

    // q = 0, so call delta is exactly N(d1); Hull quotes N(d1) = 0.7791.
    EXPECT_NEAR(greeks["delta"].get<double>(), 0.7791, 0.0001);
}

TEST(PriceCommandTest, OmitsGreeksWhenNotRequested) {
    nlohmann::json config = base_config();
    config["greeks"] = false;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_FALSE(result.value()["result"].contains("greeks"));
}

// An analytic price has no sampling error. Emitting "standard_error": 0.0 would
// claim certainty rather than inapplicability, so the field is absent.
TEST(PriceCommandTest, OmitsStochasticFieldsForAnAnalyticMethod) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& priced = result.value()["result"];
    EXPECT_FALSE(priced.contains("standard_error"));
    EXPECT_FALSE(priced.contains("confidence_interval"));
    EXPECT_FALSE(result.value().contains("seed"));
}

TEST(PriceCommandTest, ReportsDiagnostics) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& diagnostics = result.value()["result"]["diagnostics"];
    for (const std::string name : {"d1", "d2", "forward", "total_volatility", "degenerate"}) {
        EXPECT_TRUE(diagnostics.contains(name)) << "missing diagnostic: " << name;
    }
    EXPECT_NEAR(diagnostics["d1"].get<double>(), 0.7693, 0.0001);
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

// Naming an unimplemented method must fail rather than quietly fall back to the
// analytic engine, which would answer a question the configuration did not ask.
TEST(PriceCommandTest, RejectsUnimplementedMethod) {
    nlohmann::json config = base_config();
    config["method"]["type"] = "monte_carlo";

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
    EXPECT_NE(result.error().message.find("monte_carlo"), std::string::npos);
}

TEST(PriceCommandTest, RejectsUnsupportedInstrument) {
    nlohmann::json config = base_config();
    config["instrument"]["type"] = "asian";

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
}

TEST(PriceCommandTest, RejectsUnsupportedModel) {
    nlohmann::json config = base_config();
    config["model"]["type"] = "heston";

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
}

TEST(PriceCommandTest, RejectsInvalidDomainValues) {
    for (const auto& [section, field] :
         {std::pair{"market", "spot"}, std::pair{"instrument", "strike"}}) {
        nlohmann::json config = base_config();
        config[section][field] = -1.0;

        const auto result = run(config);
        ASSERT_FALSE(result.ok()) << section << "." << field;
        EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
    }
}

TEST(PriceCommandTest, RejectsNegativeMaturity) {
    nlohmann::json config = base_config();
    config["instrument"]["maturity"] = -1.0;

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(PriceCommandTest, RejectsUnknownFieldAtEveryLevel) {
    for (const std::string section : {"market", "instrument", "model", "method"}) {
        nlohmann::json config = base_config();
        config[section]["bogus"] = 1.0;

        const auto result = run(config);
        ASSERT_FALSE(result.ok()) << "unknown field accepted in section: " << section;
        EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
    }

    nlohmann::json config = base_config();
    config["bogus"] = 1.0;
    EXPECT_FALSE(run(config).ok()) << "unknown field accepted at root";
}

TEST(PriceCommandTest, RejectsMissingSection) {
    for (const std::string section : {"market", "instrument", "model", "method"}) {
        nlohmann::json config = base_config();
        config.erase(section);

        const auto result = run(config);
        ASSERT_FALSE(result.ok()) << "missing section accepted: " << section;
        EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
    }
}

// ---------------------------------------------------------------------------
// Degenerate cases
// ---------------------------------------------------------------------------

// The degenerate price is exact, so the run succeeds; the warning is what tells
// a reader the number carries no time value.
TEST(PriceCommandTest, ZeroVolatilitySucceedsWithWarning) {
    nlohmann::json config = base_config();
    config["model"]["volatility"] = 0.0;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_FALSE(result.value()["result"]["warnings"].empty());
}

// Greeks do not exist at the payoff kink, but the price does. The command must
// return the price and say why the Greeks are absent, rather than fail outright
// or drop the field silently.
TEST(PriceCommandTest, ReportsWhyGreeksAreUnavailableAtTheKink) {
    nlohmann::json config = base_config();
    config["market"]["spot"] = 100.0;
    config["market"]["rate"] = 0.05;
    config["market"]["dividend_yield"] = 0.05;  // r = q, so forward = spot
    config["instrument"]["strike"] = 100.0;
    config["model"]["volatility"] = 0.0;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_DOUBLE_EQ(result.value()["result"]["value"].get<double>(), 0.0);
    EXPECT_FALSE(result.value()["result"].contains("greeks"));

    bool explained = false;
    for (const auto& warning : result.value()["result"]["warnings"]) {
        if (warning.get<std::string>().find("Greeks") != std::string::npos) {
            explained = true;
        }
    }
    EXPECT_TRUE(explained) << "Greeks vanished without explanation";
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// The console and JSON renderings are views of one result object. If the console
// showed a rounded number that differed from the artifact, neither could be
// trusted.
TEST(PriceCommandTest, ConsoleRenderingAgreesWithJson) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const std::string console = render_console(result.value());
    const double value = result.value()["result"]["value"].get<double>();

    // 17 significant digits round-trips a double exactly.
    EXPECT_NE(console.find(fmt::format("{:.17g}", value)), std::string::npos)
        << "console output does not carry the exact value\n"
        << console;
}

TEST(PriceCommandTest, ConsoleRenderingShowsWarnings) {
    nlohmann::json config = base_config();
    config["model"]["volatility"] = 0.0;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const std::string console = render_console(result.value());
    EXPECT_NE(console.find("warnings"), std::string::npos) << console;
}

TEST(PriceCommandTest, ConsoleRenderingShowsProvenance) {
    const auto result = run(base_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const std::string console = render_console(result.value());
    EXPECT_NE(console.find("provenance"), std::string::npos);
    EXPECT_NE(console.find("commit"), std::string::npos);
}

}  // namespace
}  // namespace diffusionworks::cli
