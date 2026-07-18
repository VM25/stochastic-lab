#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include "calibrate_command.hpp"
#include "options.hpp"

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks::cli {
namespace {

/// A small implied-volatility surface generated from a known Heston model
/// (v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.6) at spot 100, rate 2%, with two
/// blind guesses and light pricing so the calibration runs in a few seconds.
constexpr const char* kConfig = R"({
  "schema_version": 1,
  "command": "calibrate",
  "market": {"spot": 100.0, "rate": 0.02, "dividend_yield": 0.0},
  "surface": {
    "source": "synthetic test surface",
    "as_of": "2026-07-18T00:00:00Z",
    "quotes": [
      {"strike": 90.0, "maturity": 0.5, "implied_volatility": 0.2225},
      {"strike": 100.0, "maturity": 0.5, "implied_volatility": 0.1981},
      {"strike": 110.0, "maturity": 0.5, "implied_volatility": 0.1780},
      {"strike": 90.0, "maturity": 1.0, "implied_volatility": 0.2189},
      {"strike": 100.0, "maturity": 1.0, "implied_volatility": 0.2001},
      {"strike": 110.0, "maturity": 1.0, "implied_volatility": 0.1844}
    ]
  },
  "calibration": {
    "objective": "implied_volatility",
    "quadrature_nodes": 128,
    "max_iterations": 1000,
    "initial_guesses": [
      {"initial_variance": 0.03, "mean_reversion": 1.0, "long_run_variance": 0.06, "vol_of_variance": 0.5, "correlation": -0.5},
      {"initial_variance": 0.02, "mean_reversion": 2.5, "long_run_variance": 0.03, "vol_of_variance": 0.7, "correlation": -0.3}
    ]
  }
})";

Result<nlohmann::json> run(const std::string& text) {
    auto document = parse_config(text, "test.json");
    EXPECT_TRUE(document.ok()) << document.error().describe();
    Options options;
    options.command = CommandKind::Calibrate;
    return run_calibrate(document.value(), options);
}

TEST(CalibrateCommandTest, CalibratesASurfaceEndToEnd) {
    const auto result = run(kConfig);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    const nlohmann::json& document = result.value();

    EXPECT_EQ(document["command"], "calibrate");
    const std::string status = document["status"];
    EXPECT_TRUE(status == "ok" || status == "warning") << status;

    // The surface generated from Heston is recovered well: the best fit converged and
    // matches the surface to within the four-decimal quoting of the implied vols.
    const auto& best = document["result"]["best"];
    EXPECT_EQ(best["status"], "converged");
    EXPECT_LT(best["implied_vol_rmse"].get<double>(), 1e-2);
    EXPECT_EQ(best["quotes_failed"].get<int>(), 0);

    // The record keeps the four things apart: per-start detail, the residual surface,
    // and the identifiability evidence.
    EXPECT_EQ(document["result"]["starts"].size(), 2U);
    EXPECT_EQ(document["result"]["residual_surface"].size(), 6U);
    EXPECT_TRUE(document["result"].contains("non_unique"));
    EXPECT_TRUE(document["result"].contains("parameter_dispersion"));
    EXPECT_EQ(document["surface"]["quote_count"], 6U);
    EXPECT_TRUE(document.contains("build_metadata"));
}

TEST(CalibrateCommandTest, ReportsTheResidualInBothUnits) {
    const auto result = run(kConfig);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    const auto& residual = result.value()["result"]["residual_surface"][0];
    for (const char* field : {"strike",
                              "maturity",
                              "market_price",
                              "model_price",
                              "market_implied_volatility",
                              "model_implied_volatility"}) {
        EXPECT_TRUE(residual.contains(field)) << "missing " << field;
    }
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

TEST(CalibrateCommandTest, RejectsAnEmptySurface) {
    const auto result = run(R"({
      "schema_version": 1, "command": "calibrate",
      "market": {"spot": 100.0, "rate": 0.02, "dividend_yield": 0.0},
      "surface": {"quotes": []},
      "calibration": {"initial_guesses": [
        {"initial_variance": 0.03, "mean_reversion": 1.0, "long_run_variance": 0.06, "vol_of_variance": 0.5, "correlation": -0.5}
      ]}
    })");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(CalibrateCommandTest, RejectsNoInitialGuesses) {
    const auto result = run(R"({
      "schema_version": 1, "command": "calibrate",
      "market": {"spot": 100.0, "rate": 0.02, "dividend_yield": 0.0},
      "surface": {"quotes": [{"strike": 100.0, "maturity": 1.0, "implied_volatility": 0.2}]},
      "calibration": {"initial_guesses": []}
    })");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(CalibrateCommandTest, RejectsAnUnknownQuoteKey) {
    const auto result = run(R"({
      "schema_version": 1, "command": "calibrate",
      "market": {"spot": 100.0, "rate": 0.02, "dividend_yield": 0.0},
      "surface": {"quotes": [{"strike": 100.0, "maturity": 1.0, "vol": 0.2}]},
      "calibration": {"initial_guesses": [
        {"initial_variance": 0.03, "mean_reversion": 1.0, "long_run_variance": 0.06, "vol_of_variance": 0.5, "correlation": -0.5}
      ]}
    })");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
}

TEST(CalibrateCommandTest, RejectsAnUnknownParameterKey) {
    const auto result = run(R"({
      "schema_version": 1, "command": "calibrate",
      "market": {"spot": 100.0, "rate": 0.02, "dividend_yield": 0.0},
      "surface": {"quotes": [{"strike": 100.0, "maturity": 1.0, "implied_volatility": 0.2}]},
      "calibration": {"initial_guesses": [
        {"initial_variance": 0.03, "mean_reversion": 1.0, "long_run_variance": 0.06, "vol_of_vol": 0.5, "correlation": -0.5}
      ]}
    })");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidConfiguration);
}

}  // namespace
}  // namespace diffusionworks::cli
