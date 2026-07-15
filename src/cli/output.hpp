#pragma once

#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace diffusionworks::cli {

/// Serialises build and environment provenance.
[[nodiscard]] nlohmann::json to_json(const BuildInfo& info);

/// Serialises a valuation.
///
/// Fields that do not apply are omitted rather than emitted as null or zero
/// (TECHNICAL-DESIGN section 19). An analytic price has no standard error, and
/// `"standard_error": 0.0` would claim certainty rather than inapplicability.
[[nodiscard]] nlohmann::json to_json(const PricingResult& result);

/// Renders a valuation for a human.
///
/// The console and JSON renderings are two views of one result object, never two
/// computations, so they cannot disagree. Numbers are printed at full double
/// precision: this is a numerics project, and a rounded console figure that
/// silently differs from the artifact would undermine the whole point.
[[nodiscard]] std::string render_console(const nlohmann::json& document);

/// Writes a document to `path`, creating parent directories as needed.
///
/// Reports an unwritable destination as an explicit IoFailure rather than
/// discarding the result (TESTING-STRATEGY section 8).
[[nodiscard]] Status write_document(const std::filesystem::path& path,
                                    const nlohmann::json& document);

}  // namespace diffusionworks::cli
