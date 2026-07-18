#include <diffusionworks/calibration/market_dataset.hpp>
#include <diffusionworks/engines/implied_volatility.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "MarketDataset";

/// Reads a required string field, or fails naming the field.
[[nodiscard]] Result<std::string> require_string(const nlohmann::json& node, const char* key) {
    if (!node.contains(key) || !node.at(key).is_string()) {
        return Result<std::string>::failure(ErrorCode::ParseFailure,
                                            fmt::format("missing or non-string field '{}'", key),
                                            kContext);
    }
    return Result<std::string>::success(node.at(key).get<std::string>());
}

/// Reads a required numeric field, or fails naming the field.
[[nodiscard]] Result<double> require_number(const nlohmann::json& node, const char* key) {
    if (!node.contains(key) || !node.at(key).is_number()) {
        return Result<double>::failure(ErrorCode::ParseFailure,
                                       fmt::format("missing or non-numeric field '{}'", key),
                                       kContext);
    }
    return Result<double>::success(node.at(key).get<double>());
}

}  // namespace

const char* to_string(ExclusionReason reason) noexcept {
    switch (reason) {
        case ExclusionReason::LowVolume:
            return "low_volume";
        case ExclusionReason::NonPositivePrice:
            return "non_positive_price";
        case ExclusionReason::ArbitrageViolation:
            return "arbitrage_violation";
        case ExclusionReason::NoImpliedVolatility:
            return "no_implied_volatility";
        case ExclusionReason::ImpliedVolatilityOutOfBand:
            return "implied_volatility_out_of_band";
    }
    return "unknown";
}

Result<MarketDataset> parse_market_dataset(std::string_view json_text, std::string_view name) {
    nlohmann::json root = nlohmann::json::parse(json_text, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded()) {
        return Result<MarketDataset>::failure(
            ErrorCode::ParseFailure, fmt::format("{} is not valid JSON", name), kContext);
    }
    if (!root.is_object()) {
        return Result<MarketDataset>::failure(
            ErrorCode::ParseFailure, fmt::format("{} is not a JSON object", name), kContext);
    }

    MarketDataset dataset;
    const auto underlying = require_string(root, "underlying");
    if (!underlying) {
        return Result<MarketDataset>::failure(underlying.error());
    }
    dataset.metadata.underlying = underlying.value();
    const auto observed = require_string(root, "observation_timestamp");
    if (!observed) {
        return Result<MarketDataset>::failure(observed.error());
    }
    dataset.metadata.observation_timestamp = observed.value();
    const auto source = require_string(root, "source");
    if (!source) {
        return Result<MarketDataset>::failure(source.error());
    }
    dataset.metadata.source = source.value();
    // Licensing is required for a real dataset: a number with no provenance is not
    // evidence, and provenance without licensing is not publishable.
    const auto licensing = require_string(root, "licensing");
    if (!licensing) {
        return Result<MarketDataset>::failure(licensing.error());
    }
    dataset.metadata.licensing = licensing.value();

    const auto spot = require_number(root, "spot");
    if (!spot) {
        return Result<MarketDataset>::failure(spot.error());
    }
    dataset.metadata.spot = spot.value();
    const auto rate = require_number(root, "rate");
    if (!rate) {
        return Result<MarketDataset>::failure(rate.error());
    }
    dataset.metadata.rate = rate.value();
    const auto dividend = require_number(root, "dividend_yield");
    if (!dividend) {
        return Result<MarketDataset>::failure(dividend.error());
    }
    dataset.metadata.dividend_yield = dividend.value();

    if (!root.contains("quotes") || !root.at("quotes").is_array()) {
        return Result<MarketDataset>::failure(
            ErrorCode::ParseFailure, "missing or non-array 'quotes'", kContext);
    }
    const nlohmann::json& quotes = root.at("quotes");
    if (quotes.empty()) {
        return Result<MarketDataset>::failure(
            ErrorCode::ParseFailure, "'quotes' is empty", kContext);
    }
    dataset.quotes.reserve(quotes.size());
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        const nlohmann::json& q = quotes.at(i);
        const auto type_text = require_string(q, "option_type");
        if (!type_text) {
            return Result<MarketDataset>::failure(type_text.error());
        }
        const auto type = parse_option_type(type_text.value());
        if (!type.has_value()) {
            return Result<MarketDataset>::failure(
                ErrorCode::ParseFailure,
                fmt::format("quote {} has option_type '{}'; expected 'call' or 'put'",
                            i,
                            type_text.value()),
                kContext);
        }
        const auto strike = require_number(q, "strike");
        if (!strike) {
            return Result<MarketDataset>::failure(strike.error());
        }
        const auto maturity = require_number(q, "maturity_years");
        if (!maturity) {
            return Result<MarketDataset>::failure(maturity.error());
        }
        const auto price = require_number(q, "close");
        if (!price) {
            return Result<MarketDataset>::failure(price.error());
        }
        const auto volume = require_number(q, "volume");
        if (!volume) {
            return Result<MarketDataset>::failure(volume.error());
        }
        RawOptionQuote quote;
        quote.type = type.value();
        quote.strike = strike.value();
        quote.maturity_years = maturity.value();
        quote.price = price.value();
        quote.volume = volume.value();
        if (q.contains("expiration_date") && q.at("expiration_date").is_string()) {
            quote.expiration_date = q.at("expiration_date").get<std::string>();
        }
        dataset.quotes.push_back(std::move(quote));
    }

    return Result<MarketDataset>::success(std::move(dataset));
}

Result<MarketDataset> load_market_dataset(const std::filesystem::path& path) {
    const std::ifstream stream(path);
    if (!stream) {
        return Result<MarketDataset>::failure(
            ErrorCode::IoFailure,
            fmt::format("could not open dataset file '{}'", path.string()),
            kContext);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return parse_market_dataset(buffer.str(), path.string());
}

Result<ValidatedDataset> validate_dataset(const MarketDataset& dataset,
                                          const DatasetValidationConfig& config) {
    const auto market = MarketState::create(
        dataset.metadata.spot, dataset.metadata.rate, dataset.metadata.dividend_yield);
    if (!market) {
        return Result<ValidatedDataset>::failure(market.error());
    }

    ValidatedDataset out;
    out.total_quotes = static_cast<int>(dataset.quotes.size());
    out.surface.spot = dataset.metadata.spot;
    out.surface.rate = dataset.metadata.rate;
    out.surface.dividend_yield = dataset.metadata.dividend_yield;
    out.surface.source = dataset.metadata.source;
    out.surface.as_of = dataset.metadata.observation_timestamp;

    const auto exclude = [&](const RawOptionQuote& q, ExclusionReason reason, std::string detail) {
        out.excluded.push_back(
            ExcludedQuote{.quote = q, .reason = reason, .detail = std::move(detail)});
    };

    for (const RawOptionQuote& q : dataset.quotes) {
        if (q.price <= 0.0) {
            ++out.non_positive_price;
            exclude(q, ExclusionReason::NonPositivePrice, "price is not positive");
            continue;
        }
        if (q.volume < config.min_volume) {
            ++out.low_volume;
            exclude(
                q,
                ExclusionReason::LowVolume,
                fmt::format("day volume {:g} below the minimum {:g}", q.volume, config.min_volume));
            continue;
        }

        const auto option = EuropeanOption::create(q.type, q.strike, q.maturity_years);
        if (!option) {
            return Result<ValidatedDataset>::failure(option.error());
        }
        const auto iv = ImpliedVolatility::solve(market.value(), option.value(), q.price);
        if (!iv) {
            // A price outside the no-arbitrage window is an arbitrage violation; any
            // other inversion failure is a price with no resolvable volatility.
            if (iv.error().code == ErrorCode::RootNotBracketed) {
                ++out.arbitrage_violations;
                exclude(q, ExclusionReason::ArbitrageViolation, iv.error().message);
            } else {
                ++out.inversion_failures;
                exclude(q, ExclusionReason::NoImpliedVolatility, iv.error().message);
            }
            continue;
        }
        if (iv.value().at_lower_floor) {
            ++out.inversion_failures;
            exclude(q,
                    ExclusionReason::NoImpliedVolatility,
                    "price sits at the volatility floor; the implied volatility is not resolvable");
            continue;
        }
        const double volatility = iv.value().implied_volatility;
        if (volatility < config.min_implied_volatility ||
            volatility > config.max_implied_volatility) {
            ++out.out_of_band;
            exclude(q,
                    ExclusionReason::ImpliedVolatilityOutOfBand,
                    fmt::format("implied volatility {:.4f} outside [{:g}, {:g}]",
                                volatility,
                                config.min_implied_volatility,
                                config.max_implied_volatility));
            continue;
        }

        out.surface.quotes.push_back(SurfaceQuote{.type = q.type,
                                                  .strike = q.strike,
                                                  .maturity = q.maturity_years,
                                                  .price = q.price,
                                                  .implied_volatility = volatility,
                                                  .weight = 1.0});
        out.maturities.push_back(q.maturity_years);
        out.strikes.push_back(q.strike);
    }

    out.included_quotes = static_cast<int>(out.surface.quotes.size());

    // Distinct, sorted maturities and strikes for coverage.
    const auto distinct = [](std::vector<double>& v) {
        std::ranges::sort(v);
        v.erase(std::ranges::unique(v).begin(), v.end());
    };
    distinct(out.maturities);
    distinct(out.strikes);

    return Result<ValidatedDataset>::success(std::move(out));
}

}  // namespace diffusionworks
