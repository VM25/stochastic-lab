#pragma once

#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <nlohmann/json.hpp>

namespace diffusionworks::cli {

/// Runs the catalog experiment named by `options.experiment_id`.
///
/// Returns the complete experiment record as a document: configuration, results,
/// summary table, status, interpretation, limitations, reproduction command, and
/// build provenance.
///
/// An experiment that reaches a Fail or Inconclusive status still returns
/// successfully. The record is the deliverable, and a failed experiment is a
/// result rather than an error -- FAILURE-MODES section 7 makes discarding one a
/// completion blocker. The caller decides what exit code that deserves.
[[nodiscard]] Result<nlohmann::json> run_experiment(const ConfigDocument& config,
                                                    const Options& options);

}  // namespace diffusionworks::cli
