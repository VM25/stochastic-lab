#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <nlohmann/json.hpp>

namespace diffusionworks::cli {

/// Executes `diffusionworks simulate`.
///
/// Generates paths and reports what they did, rather than pricing anything. It
/// answers the question a price cannot: is the simulation itself behaving?
///
/// Under GBM the terminal law is known in closed form, so the sample moments can
/// be compared against theory directly. That comparison is the difference between
/// "the price looks about right" and "the paths have the distribution they are
/// supposed to have" -- and only the second is evidence (FAILURE-MODES section 5).
[[nodiscard]] Result<nlohmann::json> run_simulate(const ConfigDocument& config,
                                                  const Options& options);

}  // namespace diffusionworks::cli
