#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

namespace diffusionworks {

/// Builds validated domain objects from configuration sections.
///
/// These sit above core (which knows nothing of options) and above the domain
/// types (which know nothing of JSON), keeping both free of the other's
/// concerns. The CLI and the experiment runner both need to turn a stored
/// configuration into the same objects, so the mapping lives here once rather
/// than being duplicated per caller -- a divergence between them would mean a
/// stored configuration reproduced differently depending on who ran it.
///
/// Each parser rejects unknown keys, so a typo fails loudly rather than leaving
/// a field at its default.

/// Parses a `market` section:
///
///     "market": { "spot": 100.0, "rate": 0.05, "dividend_yield": 0.02 }
///
/// `dividend_yield` defaults to 0.0; `spot` and `rate` are required. The default
/// is safe because zero yield is the plain equity convention and is reported in
/// the output metadata, so a reader always sees the value actually used.
[[nodiscard]] Result<MarketState> parse_market_state(const ConfigNode& node);

/// Parses an `instrument` section for a European option:
///
///     "instrument": {
///       "type": "european",
///       "option_type": "call",
///       "strike": 100.0,
///       "maturity": 1.0
///     }
[[nodiscard]] Result<EuropeanOption> parse_european_option(const ConfigNode& node);

/// Parses a `model` section for Black-Scholes:
///
///     "model": { "type": "black_scholes", "volatility": 0.2 }
[[nodiscard]] Result<BlackScholesModel> parse_black_scholes_model(const ConfigNode& node);

}  // namespace diffusionworks
