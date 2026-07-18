#pragma once

#include <diffusionworks/calibration/volatility_surface.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace diffusionworks {

/// One raw quote as it appears in a market dataset: a price (a last-trade close or a
/// mid), not yet validated or inverted to a volatility.
struct RawOptionQuote {
    OptionType type{OptionType::Call};
    double strike{};
    double maturity_years{};
    std::string expiration_date;
    double price{};
    double volume{};
};

/// The provenance a real dataset must carry (PROJECT-SPEC, VALIDATION-PLAN): what the
/// underlying is, when it was observed, the market assumptions, and where it came from.
struct MarketDatasetMetadata {
    std::string underlying;
    std::string observation_timestamp;
    std::string source;
    std::string licensing;
    double spot{};
    double rate{};
    double dividend_yield{};
};

/// A parsed but unvalidated market dataset.
struct MarketDataset {
    MarketDatasetMetadata metadata;
    std::vector<RawOptionQuote> quotes;
};

/// Loads and parses a market dataset from a JSON file.
[[nodiscard]] Result<MarketDataset> load_market_dataset(const std::filesystem::path& path);

/// Parses a market dataset from JSON text. Exposed for tests, which supply the text
/// directly rather than through a file.
[[nodiscard]] Result<MarketDataset> parse_market_dataset(std::string_view json_text,
                                                         std::string_view name);

/// The thresholds the dataset is cleaned against, all documented in the dataset file.
struct DatasetValidationConfig {
    /// A quote whose day volume is below this is treated as stale -- its closing
    /// last-trade may be hours old -- and excluded.
    double min_volume{100.0};

    double min_implied_volatility{0.01};
    double max_implied_volatility{2.0};
};

/// Why a quote was excluded from the calibration surface.
enum class ExclusionReason : std::uint8_t {
    LowVolume,
    NonPositivePrice,
    ArbitrageViolation,
    NoImpliedVolatility,
    ImpliedVolatilityOutOfBand,
};

[[nodiscard]] const char* to_string(ExclusionReason reason) noexcept;

/// A quote that did not survive validation, kept with its reason rather than dropped.
struct ExcludedQuote {
    RawOptionQuote quote;
    ExclusionReason reason{};
    std::string detail;
};

/// The outcome of validating a dataset: the cleaned surface, every exclusion with its
/// reason, and coverage.
struct ValidatedDataset {
    /// The surface the calibration runs on -- the surviving quotes, each with the
    /// implied volatility the validation computed.
    VolatilitySurface surface;

    /// The quotes that did not survive, with reasons. Never silently dropped.
    std::vector<ExcludedQuote> excluded;

    int total_quotes{};
    int included_quotes{};

    int low_volume{};
    int non_positive_price{};
    int arbitrage_violations{};
    int inversion_failures{};
    int out_of_band{};

    /// Distinct maturities and strikes surviving, for coverage reporting.
    std::vector<double> maturities;
    std::vector<double> strikes;
};

/// Validates a dataset and builds the cleaned calibration surface.
///
/// Applies the documented rules -- non-positive price, stale (low volume), the
/// no-arbitrage price bounds, invertibility to a Black-Scholes implied volatility, and
/// a sane implied-volatility band -- and reports every exclusion with its reason
/// rather than dropping it. A quote that cannot be turned into a resolvable implied
/// volatility never reaches the calibration.
[[nodiscard]] Result<ValidatedDataset> validate_dataset(const MarketDataset& dataset,
                                                        const DatasetValidationConfig& config = {});

}  // namespace diffusionworks
