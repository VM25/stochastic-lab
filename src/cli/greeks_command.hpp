#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <nlohmann/json.hpp>

namespace diffusionworks::cli {

/// Executes `diffusionworks greeks`.
///
/// Estimates one Greek by one Monte Carlo estimator (finite difference, pathwise, or
/// likelihood ratio) across a seed set, and assembles the structured result: the
/// estimator, the Greek, the bump used, the estimate, its across-seed uncertainty,
/// the seed set, the runtime, a status, and any warnings.
///
/// Returns the document rather than printing it, so console, JSON, and file output
/// share one path and cannot drift apart.
[[nodiscard]] Result<nlohmann::json> run_greeks(const ConfigDocument& config,
                                                const Options& options);

}  // namespace diffusionworks::cli
