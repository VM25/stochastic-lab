#include <diffusionworks/calibration/volatility_surface.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/implied_volatility.hpp>
#include <diffusionworks/models/black_scholes.hpp>

#include <fmt/format.h>

#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "VolatilitySurface";

}  // namespace

Result<MarketState> VolatilitySurface::market() const {
    return MarketState::create(spot, rate, dividend_yield);
}

Result<VolatilitySurface>
build_surface_from_implied_vols(double spot,
                                double rate,
                                double dividend_yield,
                                const std::vector<ImpliedVolatilityQuote>& quotes,
                                std::string source,
                                std::string as_of) {
    if (quotes.empty()) {
        return Result<VolatilitySurface>::failure(
            ErrorCode::InvalidArgument, "a surface needs at least one quote", kContext);
    }
    const auto market = MarketState::create(spot, rate, dividend_yield);
    if (!market) {
        return Result<VolatilitySurface>::failure(market.error());
    }

    VolatilitySurface surface;
    surface.spot = spot;
    surface.rate = rate;
    surface.dividend_yield = dividend_yield;
    surface.source = std::move(source);
    surface.as_of = std::move(as_of);
    surface.quotes.reserve(quotes.size());

    for (const ImpliedVolatilityQuote& quote : quotes) {
        const auto option = EuropeanOption::create(quote.type, quote.strike, quote.maturity);
        if (!option) {
            return Result<VolatilitySurface>::failure(option.error());
        }
        const auto model = BlackScholesModel::create(quote.implied_volatility);
        if (!model) {
            return Result<VolatilitySurface>::failure(model.error());
        }
        const auto priced =
            BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
        if (!priced) {
            return Result<VolatilitySurface>::failure(priced.error());
        }
        surface.quotes.push_back(SurfaceQuote{.type = quote.type,
                                              .strike = quote.strike,
                                              .maturity = quote.maturity,
                                              .price = priced.value().value,
                                              .implied_volatility = quote.implied_volatility,
                                              .weight = quote.weight});
    }

    return Result<VolatilitySurface>::success(std::move(surface));
}

Result<VolatilitySurface> generate_heston_surface(const SyntheticSurfaceSpec& spec) {
    if (spec.strikes.empty() || spec.maturities.empty()) {
        return Result<VolatilitySurface>::failure(
            ErrorCode::InvalidArgument,
            "a synthetic surface needs at least one strike and one maturity",
            kContext);
    }

    const auto market = MarketState::create(spec.spot, spec.rate, spec.dividend_yield);
    if (!market) {
        return Result<VolatilitySurface>::failure(market.error());
    }
    const auto model = spec.parameters.to_model();
    if (!model) {
        return Result<VolatilitySurface>::failure(model.error());
    }

    VolatilitySurface surface;
    surface.spot = spec.spot;
    surface.rate = spec.rate;
    surface.dividend_yield = spec.dividend_yield;
    surface.source = fmt::format(
        "synthetic Heston surface: v0={:.6g}, kappa={:.6g}, theta={:.6g}, xi={:.6g}, rho={:.6g}",
        spec.parameters.initial_variance,
        spec.parameters.mean_reversion,
        spec.parameters.long_run_variance,
        spec.parameters.vol_of_variance,
        spec.parameters.correlation);
    surface.quotes.reserve(spec.strikes.size() * spec.maturities.size());

    for (const double maturity : spec.maturities) {
        for (const double strike : spec.strikes) {
            const auto option = EuropeanOption::create(OptionType::Call, strike, maturity);
            if (!option) {
                return Result<VolatilitySurface>::failure(option.error());
            }
            const auto priced =
                HestonAnalyticEngine::price(market.value(), option.value(), model.value());
            if (!priced) {
                return Result<VolatilitySurface>::failure(
                    ErrorCode::IntegrationFailure,
                    fmt::format("could not price the synthetic grid point K={:.6g}, T={:.6g}: {}",
                                strike,
                                maturity,
                                priced.error().message),
                    kContext);
            }
            const auto iv =
                ImpliedVolatility::solve(market.value(), option.value(), priced.value().value);
            if (!iv) {
                return Result<VolatilitySurface>::failure(
                    ErrorCode::RootNotBracketed,
                    fmt::format("the synthetic grid point K={:.6g}, T={:.6g} priced at {:.6g} has "
                                "no implied volatility ({}); choose a grid whose points carry a "
                                "resolvable smile rather than dropping this one silently",
                                strike,
                                maturity,
                                priced.value().value,
                                iv.error().message),
                    kContext);
            }

            surface.quotes.push_back(
                SurfaceQuote{.type = OptionType::Call,
                             .strike = strike,
                             .maturity = maturity,
                             .price = priced.value().value,
                             .implied_volatility = iv.value().implied_volatility,
                             .weight = 1.0});
        }
    }

    return Result<VolatilitySurface>::success(std::move(surface));
}

}  // namespace diffusionworks
