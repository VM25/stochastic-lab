#pragma once

#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/core/result.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace diffusionworks {

/// The outcome of an experiment, per EXPERIMENT-CATALOG's required artifacts.
enum class ExperimentStatus : std::uint8_t {
    /// Every exit criterion was met.
    Pass,

    /// An exit criterion was not met. The record is still written: FAILURE-MODES
    /// section 7 makes deleting an inconvenient result a completion blocker, and
    /// an experiment that only ever passes is not evidence of anything.
    Fail,

    /// The criteria were met, but something about the run deserves to be read
    /// before the number is quoted.
    Warning,

    /// The experiment ran without error and did not answer its question -- most
    /// often because the effect it measures did not clear its own sampling noise.
    ///
    /// Distinct from Fail on purpose. "The scheme is wrong" and "this run cannot
    /// tell whether the scheme is wrong" are different findings, and collapsing
    /// them would either slander the method or excuse it.
    Inconclusive,
};

[[nodiscard]] const char* to_string(ExperimentStatus s) noexcept;

/// A tabular result, written alongside the JSON for plotting and inspection.
struct CsvTable {
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;

    [[nodiscard]] Result<std::string> render() const;
};

/// One experiment's complete record.
///
/// Carries everything EXPERIMENT-CATALOG's "Required Artifacts" demands except the
/// plots, which are produced from `table` by the Python layer: configuration,
/// machine-readable results, a summary table, a methodology note, a status, an
/// explanation of deviations, and an exact reproduction command.
///
/// The provenance fields are not decoration. A convergence slope is meaningless
/// without the seed, the compiler, and the flags that produced it -- this project
/// disables floating-point contraction precisely because those things change
/// answers -- so the record refuses to exist without them.
struct ExperimentRecord {
    /// The catalog identifier, e.g. "EXP-02".
    std::string id;
    std::string name;

    /// The question from the catalog, verbatim, so the record answers something
    /// specific rather than merely reporting numbers.
    std::string question;

    ExperimentStatus status{ExperimentStatus::Inconclusive};

    /// What the numbers mean, in prose. Written to be quotable.
    std::string interpretation;

    /// What this experiment does *not* establish. Populated even on a pass.
    std::vector<std::string> limitations;

    /// The exact command that reproduces this record.
    std::string reproduction_command;

    /// The parameters the experiment ran with.
    nlohmann::json configuration;

    /// The full machine-readable results.
    nlohmann::json results;

    CsvTable table;

    double runtime_seconds{};

    [[nodiscard]] nlohmann::json to_json() const;
};

}  // namespace diffusionworks
