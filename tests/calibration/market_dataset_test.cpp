#include <diffusionworks/calibration/market_dataset.hpp>
#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

/// Builds a minimal dataset JSON with the given quotes array text spliced in, so a
/// test can vary one quote and keep the metadata fixed.
std::string dataset_with(const std::string& quotes) {
    return R"({
      "underlying": "TEST",
      "observation_timestamp": "2026-07-17T20:00:00Z",
      "source": "unit test",
      "licensing": "test data, no restrictions",
      "spot": 100.0, "rate": 0.03, "dividend_yield": 0.0,
      "quotes": )" +
           quotes + "}";
}

const char* kRealDataset = DW_TEST_DATA_DIR "/market/spy_options_2026-07-17.json";

// ---------------------------------------------------------------------------
// The committed real dataset
// ---------------------------------------------------------------------------

// The real SPY surface loads, carries its provenance, and validates cleanly: every
// quote survives at the default thresholds, the coverage is three maturities and six
// strikes, and each surviving quote has a sane implied volatility.
TEST(MarketDatasetTest, LoadsAndValidatesTheRealSpySurface) {
    const auto dataset = load_market_dataset(kRealDataset);
    ASSERT_TRUE(dataset.ok()) << dataset.error().describe();
    EXPECT_EQ(dataset.value().metadata.underlying, "SPY");
    EXPECT_FALSE(dataset.value().metadata.licensing.empty());
    EXPECT_FALSE(dataset.value().metadata.source.empty());
    EXPECT_EQ(dataset.value().quotes.size(), 18U);

    const auto validated = validate_dataset(dataset.value());
    ASSERT_TRUE(validated.ok()) << validated.error().describe();
    EXPECT_EQ(validated.value().total_quotes, 18);
    EXPECT_EQ(validated.value().included_quotes, 18);
    EXPECT_TRUE(validated.value().excluded.empty());
    EXPECT_EQ(validated.value().maturities.size(), 3U);
    EXPECT_EQ(validated.value().strikes.size(), 6U);
    for (const SurfaceQuote& q : validated.value().surface.quotes) {
        EXPECT_GT(q.implied_volatility, 0.05);
        EXPECT_LT(q.implied_volatility, 0.5);
    }
}

// A higher volume threshold excludes the thinly-traded quotes and says so by reason,
// demonstrating the staleness rule.
TEST(MarketDatasetTest, RaisingTheVolumeThresholdExcludesThinQuotes) {
    const auto dataset = load_market_dataset(kRealDataset);
    ASSERT_TRUE(dataset.ok());

    DatasetValidationConfig config;
    config.min_volume = 1000.0;  // several October quotes trade below this
    const auto validated = validate_dataset(dataset.value(), config);
    ASSERT_TRUE(validated.ok());
    EXPECT_GT(validated.value().low_volume, 0);
    EXPECT_LT(validated.value().included_quotes, 18);
    for (const ExcludedQuote& e : validated.value().excluded) {
        EXPECT_EQ(e.reason, ExclusionReason::LowVolume);
    }
}

// ---------------------------------------------------------------------------
// The validation rules, one at a time
// ---------------------------------------------------------------------------

TEST(MarketDatasetTest, ExcludesAZeroVolumeQuoteAsStale) {
    const auto dataset = parse_market_dataset(
        dataset_with(
            R"([{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":8.0,"volume":0}])"),
        "test");
    ASSERT_TRUE(dataset.ok()) << dataset.error().describe();
    const auto v = validate_dataset(dataset.value());
    ASSERT_TRUE(v.ok());
    EXPECT_EQ(v.value().included_quotes, 0);
    ASSERT_EQ(v.value().excluded.size(), 1U);
    EXPECT_EQ(v.value().excluded[0].reason, ExclusionReason::LowVolume);
}

TEST(MarketDatasetTest, ExcludesANonPositivePrice) {
    const auto dataset = parse_market_dataset(
        dataset_with(
            R"([{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":0.0,"volume":5000}])"),
        "test");
    ASSERT_TRUE(dataset.ok());
    const auto v = validate_dataset(dataset.value());
    ASSERT_TRUE(v.ok());
    ASSERT_EQ(v.value().excluded.size(), 1U);
    EXPECT_EQ(v.value().excluded[0].reason, ExclusionReason::NonPositivePrice);
}

// A call worth more than the discounted forward is arbitrageable and excluded as such.
TEST(MarketDatasetTest, ExcludesAnArbitrageViolation) {
    const auto dataset = parse_market_dataset(
        dataset_with(
            R"([{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":101.0,"volume":5000}])"),
        "test");
    ASSERT_TRUE(dataset.ok());
    const auto v = validate_dataset(dataset.value());
    ASSERT_TRUE(v.ok());
    ASSERT_EQ(v.value().excluded.size(), 1U);
    EXPECT_EQ(v.value().excluded[0].reason, ExclusionReason::ArbitrageViolation);
    EXPECT_EQ(v.value().arbitrage_violations, 1);
}

// A price giving an implausibly high implied volatility is excluded by the band.
TEST(MarketDatasetTest, ExcludesAnOutOfBandImpliedVolatility) {
    // A near-forward-priced ATM call implies an enormous volatility.
    const auto dataset = parse_market_dataset(
        dataset_with(
            R"([{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":95.0,"volume":5000}])"),
        "test");
    ASSERT_TRUE(dataset.ok());
    DatasetValidationConfig config;
    config.max_implied_volatility = 1.5;
    const auto v = validate_dataset(dataset.value(), config);
    ASSERT_TRUE(v.ok());
    ASSERT_EQ(v.value().excluded.size(), 1U);
    EXPECT_EQ(v.value().excluded[0].reason, ExclusionReason::ImpliedVolatilityOutOfBand);
}

// A clean quote survives and reaches the surface with its implied volatility.
TEST(MarketDatasetTest, KeepsACleanQuote) {
    const auto dataset = parse_market_dataset(
        dataset_with(
            R"([{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":8.0,"volume":5000}])"),
        "test");
    ASSERT_TRUE(dataset.ok());
    const auto v = validate_dataset(dataset.value());
    ASSERT_TRUE(v.ok());
    EXPECT_EQ(v.value().included_quotes, 1);
    ASSERT_EQ(v.value().surface.quotes.size(), 1U);
    EXPECT_GT(v.value().surface.quotes[0].implied_volatility, 0.0);
}

// ---------------------------------------------------------------------------
// Parsing refusals
// ---------------------------------------------------------------------------

TEST(MarketDatasetTest, RejectsMalformedJson) {
    const auto dataset = parse_market_dataset("{ not json", "test");
    ASSERT_FALSE(dataset.ok());
    EXPECT_EQ(dataset.error().code, ErrorCode::ParseFailure);
}

TEST(MarketDatasetTest, RejectsAMissingLicensingField) {
    const auto dataset = parse_market_dataset(R"({
      "underlying": "TEST", "observation_timestamp": "t", "source": "s",
      "spot": 100.0, "rate": 0.03, "dividend_yield": 0.0,
      "quotes": [{"option_type":"call","strike":100.0,"maturity_years":1.0,"close":8.0,"volume":5000}]
    })",
                                              "test");
    ASSERT_FALSE(dataset.ok());
    EXPECT_EQ(dataset.error().code, ErrorCode::ParseFailure);
    EXPECT_NE(dataset.error().message.find("licensing"), std::string::npos);
}

TEST(MarketDatasetTest, RejectsAnEmptyQuotesArray) {
    const auto dataset = parse_market_dataset(dataset_with("[]"), "test");
    ASSERT_FALSE(dataset.ok());
    EXPECT_EQ(dataset.error().code, ErrorCode::ParseFailure);
}

TEST(MarketDatasetTest, RejectsAMissingFile) {
    const auto dataset = load_market_dataset("/no/such/dataset.json");
    ASSERT_FALSE(dataset.ok());
    EXPECT_EQ(dataset.error().code, ErrorCode::IoFailure);
}

}  // namespace
}  // namespace diffusionworks
