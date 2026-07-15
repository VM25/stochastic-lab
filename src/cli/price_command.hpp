#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <nlohmann/json.hpp>

namespace diffusionworks::cli {

/// Executes `diffusionworks price`.
///
/// Builds the market, instrument, and model from the configuration, selects an
/// engine, values the instrument, and assembles the result document defined by
/// TECHNICAL-DESIGN section 19.
///
/// Returns the document rather than printing it, so the same code path serves
/// console, JSON, and file output and the renderings cannot drift apart.
[[nodiscard]] Result<nlohmann::json> run_price(const ConfigDocument& config,
                                               const Options& options);

}  // namespace diffusionworks::cli
