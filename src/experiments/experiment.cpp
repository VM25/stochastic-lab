#include <diffusionworks/experiments/experiment.hpp>

#include <fmt/format.h>

#include <algorithm>

namespace diffusionworks {

const char* to_string(ExperimentStatus s) noexcept {
    switch (s) {
        case ExperimentStatus::Pass:
            return "pass";
        case ExperimentStatus::Fail:
            return "fail";
        case ExperimentStatus::Warning:
            return "warning";
        case ExperimentStatus::Inconclusive:
            return "inconclusive";
    }
    return "unknown";
}

Result<std::string> CsvTable::render() const {
    if (headers.empty()) {
        return Result<std::string>::failure(ErrorCode::InvalidArgument,
                                            "a table needs headers to be interpretable",
                                            "CsvTable::render");
    }

    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != headers.size()) {
            // A ragged CSV parses without complaint and silently shifts every
            // column after the gap, so a plot would be drawn from misaligned data
            // and look entirely plausible.
            return Result<std::string>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("row {} has {} fields but there are {} headers", i, rows[i].size(),
                            headers.size()),
                "CsvTable::render");
        }
    }

    std::string out;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        out += headers[i];
        out += i + 1 < headers.size() ? ',' : '\n';
    }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            out += row[i];
            out += i + 1 < row.size() ? ',' : '\n';
        }
    }
    return Result<std::string>::success(std::move(out));
}

nlohmann::json ExperimentRecord::to_json() const {
    return nlohmann::json{
        {"id", id},
        {"name", name},
        {"question", question},
        {"status", to_string(status)},
        {"interpretation", interpretation},
        {"limitations", limitations},
        {"reproduction_command", reproduction_command},
        {"configuration", configuration},
        {"results", results},
        {"runtime_seconds", runtime_seconds},
        // Provenance is part of the record, not an annotation on it. A convergence
        // slope produced under different flags is a different result.
        {"build_metadata", diffusionworks::to_json(collect_build_info())},
    };
}

}  // namespace diffusionworks
