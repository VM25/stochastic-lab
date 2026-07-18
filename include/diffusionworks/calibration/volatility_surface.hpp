#pragma once

#include <diffusionworks/calibration/parameter_transform.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>

#include <string>
#include <vector>

namespace diffusionworks {

/// One option quote on a volatility surface: a contract, its price, the Black-Scholes
/// implied volatility of that price, and the weight it carries in a calibration.
struct SurfaceQuote {
    OptionType type{OptionType::Call};
    double strike{};
    double maturity{};

    /// The market (or synthetic) price this quote must be matched to.
    double price{};

    /// The Black-Scholes implied volatility of `price`. Carried alongside the price so
    /// a volatility-error objective does not have to re-invert on every read, and so a
    /// reader can see the smile the surface represents.
    double implied_volatility{};

    /// The calibration weight. A quote a calibration should trust less -- a wide
    /// spread, a thin market -- carries a smaller weight; the default trusts all quotes
    /// equally.
    double weight{1.0};
};

/// A set of quotes against one market, with the provenance a published calibration
/// must carry.
struct VolatilitySurface {
    double spot{};
    double rate{};
    double dividend_yield{};
    std::vector<SurfaceQuote> quotes;

    /// Where the surface came from. For a synthetic surface, the parameters that
    /// generated it; for a documented one, the source and any caveats. Never empty in
    /// a published surface -- a number with no provenance is not evidence.
    std::string source;

    /// The instant the surface is as of. A calibration is a snapshot, and a snapshot
    /// with no timestamp cannot be reproduced or compared.
    std::string as_of;

    /// Rebuilds the market state, or fails exactly as MarketState::create would.
    [[nodiscard]] Result<MarketState> market() const;
};

/// The grid and true parameters for a synthetic Heston surface.
struct SyntheticSurfaceSpec {
    double spot{100.0};
    double rate{0.02};
    double dividend_yield{0.0};

    /// Strikes and maturities of the grid. Every (strike, maturity) pair becomes a
    /// call quote.
    std::vector<double> strikes{80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> maturities{0.25, 0.5, 1.0, 2.0};

    /// The parameters the surface is generated from -- the ground truth a recovery
    /// experiment tries to get back.
    HestonParameters parameters{.initial_variance = 0.04,
                                .mean_reversion = 1.5,
                                .long_run_variance = 0.05,
                                .vol_of_variance = 0.4,
                                .correlation = -0.6};
};

/// Generates a synthetic volatility surface by pricing every grid point under Heston
/// and inverting each price to its Black-Scholes implied volatility.
///
/// The surface is the ground truth for EXP-11's recovery test: a calibration that
/// cannot get these parameters back from the surface they generated has not earned
/// trust on a real one. Generation fails, rather than quietly dropping a point, if a
/// grid point cannot be priced or its price has no implied volatility -- a bad grid is
/// a configuration error to surface, not a hole to paper over.
[[nodiscard]] Result<VolatilitySurface> generate_heston_surface(const SyntheticSurfaceSpec& spec);

}  // namespace diffusionworks
