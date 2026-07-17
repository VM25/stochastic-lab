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

/// A Heston pricing configuration. The parameters are the Fang-Oosterlee benchmark
/// (call 5.78515543437619), so the CLI end-to-end path is checked against the same
/// published value as the engine's unit tests.
nlohmann::json heston_config() {
    return nlohmann::json{
        {"schema_version", 1},
        {"command", "price"},
        {"market", {{"spot", 100.0}, {"rate", 0.0}, {"dividend_yield", 0.0}}},
        {"instrument",
         {{"type", "european"}, {"option_type", "call"}, {"strike", 100.0}, {"maturity", 1.0}}},
        {"model",
         {{"type", "heston"},
          {"initial_variance", 0.0175},
          {"mean_reversion", 1.5768},
          {"long_run_variance", 0.0398},
          {"vol_of_variance", 0.5751},
          {"correlation", -0.5711}}},
        {"method", {{"type", "heston_analytic"}}},
    };
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

// The Heston model reaches its engine through the same command, dispatched on
// model.type, and reproduces the published benchmark end to end.
TEST(PriceCommandTest, PricesHestonAgainstThePublishedBenchmark) {
    const auto result = run(heston_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_NEAR(result.value()["result"]["value"].get<double>(), 5.78515543437619, 1e-9);
    EXPECT_EQ(result.value()["model"]["type"], "heston");
    EXPECT_EQ(result.value()["method"], "heston_analytic");
}

// This benchmark violates the Feller condition, so the document carries the
// diagnostic and its status is "warning" -- the violation reaching the artifact
// rather than being swallowed by the engine.
TEST(PriceCommandTest, HestonSurfacesAFellerViolation) {
    const auto result = run(heston_config());
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value()["status"], "warning");
    EXPECT_FALSE(result.value()["model"]["satisfies_feller"].get<bool>());
    EXPECT_LT(result.value()["model"]["feller_ratio"].get<double>(), 1.0);
    EXPECT_TRUE(result.value().contains("result"));
    EXPECT_FALSE(result.value()["result"]["warnings"].empty());
}

// A Heston config that names the wrong method is refused rather than silently priced
// by the Black-Scholes path.
TEST(PriceCommandTest, HestonRejectsAMismatchedMethod) {
    auto config = heston_config();
    config["method"]["type"] = "analytic";
    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
}

// A typo in a Heston parameter name fails loudly rather than leaving a default in
// place and pricing a different model.
TEST(PriceCommandTest, HestonRejectsAnUnknownParameter) {
    auto config = heston_config();
    config["model"]["vol_of_vol"] = 0.5;  // misspelling of vol_of_variance
    const auto result = run(config);
    ASSERT_FALSE(result.ok());
}

// The Heston model is also priceable by simulation through the same command,
// dispatched on method.type. The simulated price reproduces the semi-analytic
// benchmark to within the sampling noise and carries its uncertainty and the
// variance diagnostics rather than a bare number.
TEST(PriceCommandTest, PricesHestonByMonteCarlo) {
    auto config = heston_config();
    config["method"] = {
        {"type", "heston_monte_carlo"}, {"paths", 40000}, {"steps", 120}, {"seed", 20260717}};
    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_EQ(result.value()["method"], "heston_monte_carlo");

    // The published benchmark is 5.78515543437619; at these settings the full-
    // truncation bias and the sampling error are both small, so the CLI path lands
    // close to it. The tight numerical validation lives in the engine unit tests;
    // this pins the wiring and that the number is right end to end.
    EXPECT_NEAR(result.value()["result"]["value"].get<double>(), 5.78515543437619, 0.15);

    // Simulation carries its uncertainty and its variance diagnostics.
    EXPECT_TRUE(result.value()["result"].contains("standard_error"));
    EXPECT_TRUE(result.value()["result"].contains("confidence_interval"));
    EXPECT_TRUE(result.value()["result"]["diagnostics"].contains("negative_variance_fraction"));
}

// The command-line seed overrides the method's seed, so a run can be reseeded
// without editing the file -- and a different seed gives a different draw.
TEST(PriceCommandTest, HestonMonteCarloSeedOverrideChangesTheDraw) {
    auto config = heston_config();
    config["method"] = {{"type", "heston_monte_carlo"}, {"paths", 8000}, {"steps", 20}};

    Options first;
    first.seed = 1;
    Options second;
    second.seed = 2;
    const auto a = run(config, first);
    const auto b = run(config, second);
    ASSERT_TRUE(a.ok()) << a.error().describe();
    ASSERT_TRUE(b.ok()) << b.error().describe();
    EXPECT_NE(a.value()["result"]["value"].get<double>(),
              b.value()["result"]["value"].get<double>());
}

// A typo in the Monte Carlo method block fails loudly rather than leaving a default.
TEST(PriceCommandTest, HestonMonteCarloRejectsAnUnknownMethodKey) {
    auto config = heston_config();
    config["method"] = {{"type", "heston_monte_carlo"}, {"path", 40000}};  // "path" not "paths"
    const auto result = run(config);
    ASSERT_FALSE(result.ok());
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
//
// Monte Carlo became supported in Phase 3, so the unimplemented case is now a
// finite-difference method, which arrives in Phase 6.
TEST(PriceCommandTest, RejectsUnimplementedMethod) {
    nlohmann::json config = base_config();
    config["method"]["type"] = "crank_nicolson";

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
    EXPECT_NE(result.error().message.find("crank_nicolson"), std::string::npos);
}

// Monte Carlo is now a supported method for a European option, and must produce
// a price with its uncertainty rather than a bare number.
TEST(PriceCommandTest, PricesEuropeanByMonteCarlo) {
    nlohmann::json config = base_config();
    config["method"] = {{"type", "monte_carlo"},
                        {"paths", 100000},
                        {"steps", 1},
                        {"scheme", "exact"},
                        {"seed", 20260715}};
    config["greeks"] = false;

    const auto result = run(config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& priced = result.value()["result"];
    EXPECT_EQ(priced["method"], "monte_carlo_exact");
    ASSERT_TRUE(priced.contains("standard_error"));
    ASSERT_TRUE(priced.contains("confidence_interval"));

    // Hull 9th ed. Example 15.6 gives 4.76 for these terms; the interval must
    // contain it.
    EXPECT_LE(priced["confidence_interval"]["lower"].get<double>(), 4.76);
    EXPECT_GE(priced["confidence_interval"]["upper"].get<double>(), 4.76);
}

// A stochastic method has no default seed: a run whose seed was chosen for it
// cannot be reproduced from its own record.
TEST(PriceCommandTest, MonteCarloRequiresASeed) {
    nlohmann::json config = base_config();
    config["method"] = {{"type", "monte_carlo"}, {"paths", 1000}, {"scheme", "exact"}};
    config["greeks"] = false;

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.error().message.find("seed"), std::string::npos);
}

// Asian options became supported in Phase 3, so the unsupported case is now a
// barrier, which arrives in Phase 7.
TEST(PriceCommandTest, RejectsUnsupportedInstrument) {
    nlohmann::json config = base_config();
    config["instrument"]["type"] = "down_and_out";

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
}

// An arithmetic Asian has no closed form: the sum of lognormals is not
// lognormal. Naming 'analytic' must fail rather than quietly price something
// else.
TEST(PriceCommandTest, RejectsAnalyticPricingOfAnArithmeticAsian) {
    nlohmann::json config = base_config();
    config["instrument"] = {{"type", "asian"},
                            {"option_type", "call"},
                            {"averaging", "arithmetic"},
                            {"strike", 100.0},
                            {"maturity", 1.0},
                            {"monitoring_count", 12}};

    const auto result = run(config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::UnsupportedCombination);
}

// An Asian price without its averaging convention and monitoring frequency is
// ambiguous, so both are required rather than defaulted.
TEST(PriceCommandTest, AsianRequiresItsAveragingConvention) {
    nlohmann::json config = base_config();
    config["instrument"] = {{"type", "asian"},
                            {"option_type", "call"},
                            {"strike", 100.0},
                            {"maturity", 1.0},
                            {"monitoring_count", 12}};
    config["method"] = {
        {"type", "monte_carlo"}, {"paths", 1000}, {"steps", 12}, {"scheme", "exact"}, {"seed", 1}};
    config["greeks"] = false;

    const auto result = run(config);
    ASSERT_FALSE(result.ok()) << "averaging must not default";
    EXPECT_NE(result.error().message.find("averaging"), std::string::npos);
}

TEST(PriceCommandTest, RejectsUnsupportedModel) {
    // black_scholes and heston are supported; a third type is not, and must be
    // refused rather than silently priced as one of them.
    nlohmann::json config = base_config();
    config["model"]["type"] = "sabr";

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

/// A configuration sitting exactly on the zero-diffusion payoff kink: r = q makes
/// the forward equal spot, so F = K = 100 with no diffusion.
nlohmann::json kink_config() {
    nlohmann::json config = base_config();
    config["market"]["spot"] = 100.0;
    config["market"]["rate"] = 0.05;
    config["market"]["dividend_yield"] = 0.05;
    config["instrument"]["strike"] = 100.0;
    config["model"]["volatility"] = 0.0;
    return config;
}

// At the kink the price is exact but the Greeks part company: delta jumps and
// gamma is a Dirac mass, while vega is a genuine one-sided derivative. The
// command must return the price, report the Greeks that exist, and omit the ones
// that do not.
TEST(PriceCommandTest, ReportsExactPriceAndSurvivingGreeksAtTheKink) {
    const auto result = run(kink_config());

    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_DOUBLE_EQ(result.value()["result"]["value"].get<double>(), 0.0);

    // The Greek block is present: refusing it wholesale would discard vega,
    // which genuinely exists here.
    ASSERT_TRUE(result.value()["result"].contains("greeks"));
    const nlohmann::json& greeks = result.value()["result"]["greeks"];
    EXPECT_TRUE(greeks.contains("vega"));
}

// The guarantee that matters most: no finite gamma may appear at the kink, in
// the artifact any more than in the engine.
TEST(PriceCommandTest, NeverEmitsAFiniteGammaAtTheKink) {
    const auto result = run(kink_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& greeks = result.value()["result"]["greeks"];
    EXPECT_FALSE(greeks.contains("gamma")) << "a finite gamma reached the artifact";
    EXPECT_FALSE(greeks.contains("delta"));

    // Omission alone is not enough: a consumer must be able to tell "undefined"
    // from "not requested".
    ASSERT_TRUE(greeks.contains("undefined"));
    EXPECT_TRUE(greeks["undefined"].contains("gamma"));
    EXPECT_TRUE(greeks["undefined"].contains("delta"));
    EXPECT_FALSE(greeks["undefined"]["gamma"].get<std::string>().empty());
}

// An absent Greek must also surface where a reader skimming the output will see
// it, not only in a nested field.
TEST(PriceCommandTest, WarnsAboutEveryUndefinedGreekAtTheKink) {
    const auto result = run(kink_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const nlohmann::json& warnings = result.value()["result"]["warnings"];
    ASSERT_FALSE(warnings.empty());

    const auto warned_about = [&](const std::string& name) {
        for (const auto& warning : warnings) {
            if (warning.get<std::string>().find(name) != std::string::npos) {
                return true;
            }
        }
        return false;
    };

    EXPECT_TRUE(warned_about("gamma")) << "gamma vanished without explanation";
    EXPECT_TRUE(warned_about("delta")) << "delta vanished without explanation";
}

// The console rendering must not let an absent Greek read as a missing line.
TEST(PriceCommandTest, ConsoleRenderingMarksUndefinedGreeks) {
    const auto result = run(kink_config());
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const std::string console = render_console(result.value());
    EXPECT_NE(console.find("undefined"), std::string::npos)
        << "console output hides the undefined Greeks\n"
        << console;
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
