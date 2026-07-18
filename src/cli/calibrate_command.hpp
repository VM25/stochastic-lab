#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <nlohmann/json.hpp>

namespace diffusionworks::cli {

/// Runs the `calibrate` command: fits Heston parameters to a volatility surface read
/// from the configuration, from several starting points, and returns the full
/// calibration record -- the best fit, every start, the residual surface, and the
/// evidence on whether the fit is unique.
[[nodiscard]] Result<nlohmann::json> run_calibrate(const ConfigDocument& config,
                                                   const Options& options);

}  // namespace diffusionworks::cli
