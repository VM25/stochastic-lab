#include "output.hpp"

#include <fmt/format.h>

#include <fstream>
#include <ios>
#include <string_view>
#include <system_error>
#include <variant>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "output";

/// Renders a diagnostic value without collapsing its type.
nlohmann::json diagnostic_to_json(const Diagnostic::Value& value) {
    return std::visit([](const auto& held) -> nlohmann::json { return nlohmann::json(held); },
                      value);
}

/// Formats a double at round-trip precision.
///
/// 17 significant digits is the shortest decimal that reproduces any double
/// exactly, so a value read back from the console equals the value computed.
std::string format_double(double value) {
    return fmt::format("{:.17g}", value);
}

void append_key_value(std::string& out,
                      std::string_view key,
                      const nlohmann::json& value,
                      int indent = 2) {
    out += std::string(static_cast<std::size_t>(indent), ' ');
    out += key;
    out += ": ";
    if (value.is_number_float()) {
        out += format_double(value.get<double>());
    } else if (value.is_string()) {
        out += value.get<std::string>();
    } else {
        out += value.dump();
    }
    out += "\n";
}

}  // namespace


nlohmann::json to_json(const PricingResult& result) {
    nlohmann::json document{
        {"value", result.value},
        {"method", result.method},
    };

    if (result.standard_error.has_value()) {
        document["standard_error"] = *result.standard_error;
    }

    if (result.confidence_interval.has_value()) {
        document["confidence_interval"] = nlohmann::json{
            {"lower", result.confidence_interval->lower},
            {"upper", result.confidence_interval->upper},
            {"level", result.confidence_interval->level},
            {"width", result.confidence_interval->width()},
        };
    }

    if (result.greeks.has_value()) {
        const Greeks& greeks = *result.greeks;

        // Units travel with the numbers. A bare "vega": 12.1 is ambiguous
        // between per-unit and per-point, and the two differ by 100x.
        nlohmann::json rendered{
            {"units",
             nlohmann::json{
                 {"delta", "per unit of spot"},
                 {"gamma", "per unit of spot squared"},
                 {"vega", "per unit of volatility (divide by 100 for a 1% move)"},
                 {"theta", "per year (calendar time)"},
                 {"rho", "per unit of rate (divide by 10000 for a basis point)"},
             }},
        };

        // A Greek that does not exist is omitted rather than emitted as null or
        // zero, and its reason is reported beside the others. A consumer reading
        // greeks["gamma"] therefore either finds a real derivative or finds
        // nothing -- never a fabricated finite value standing in for a Dirac
        // mass.
        const auto emit = [&rendered](const char* name, const std::optional<double>& value) {
            if (value.has_value()) {
                rendered[name] = *value;
            }
        };
        emit("delta", greeks.delta);
        emit("gamma", greeks.gamma);
        emit("vega", greeks.vega);
        emit("theta", greeks.theta);
        emit("rho", greeks.rho);

        if (!greeks.undefined.empty()) {
            nlohmann::json undefined = nlohmann::json::object();
            for (const UndefinedGreek& entry : greeks.undefined) {
                undefined[entry.name] = entry.reason;
            }
            rendered["undefined"] = std::move(undefined);
        }

        document["greeks"] = std::move(rendered);
    }

    nlohmann::json diagnostics = nlohmann::json::object();
    for (const auto& diagnostic : result.diagnostics) {
        diagnostics[diagnostic.name] = diagnostic_to_json(diagnostic.value);
    }
    document["diagnostics"] = std::move(diagnostics);

    document["warnings"] = result.warnings;
    document["runtime_seconds"] = result.runtime_seconds;

    return document;
}

std::string render_console(const nlohmann::json& document) {
    std::string out;

    const auto status = document.value("status", std::string{"unknown"});
    out += fmt::format("status   : {}\n", status);
    out += fmt::format("command  : {}\n", document.value("command", std::string{"unknown"}));

    if (document.contains("market")) {
        out += "\nmarket\n";
        for (const auto& item : document["market"].items()) {
            append_key_value(out, item.key(), item.value());
        }
    }

    if (document.contains("instrument")) {
        out += "\ninstrument\n";
        for (const auto& item : document["instrument"].items()) {
            append_key_value(out, item.key(), item.value());
        }
    }

    if (document.contains("model")) {
        out += "\nmodel\n";
        for (const auto& item : document["model"].items()) {
            append_key_value(out, item.key(), item.value());
        }
    }

    if (document.contains("result")) {
        const nlohmann::json& result = document["result"];
        out += "\nresult\n";
        append_key_value(out, "value", result["value"]);
        append_key_value(out, "method", result["method"]);

        if (result.contains("standard_error")) {
            append_key_value(out, "standard_error", result["standard_error"]);
        }
        if (result.contains("confidence_interval")) {
            const nlohmann::json& ci = result["confidence_interval"];
            out += fmt::format("  confidence_interval: [{}, {}] at level {}\n",
                               format_double(ci["lower"].get<double>()),
                               format_double(ci["upper"].get<double>()),
                               format_double(ci["level"].get<double>()));
        }

        if (result.contains("greeks")) {
            const nlohmann::json& greeks = result["greeks"];
            out += "\ngreeks\n";
            for (const std::string_view view : {"delta", "gamma", "vega", "theta", "rho"}) {
                const std::string name(view);
                if (greeks.contains(name)) {
                    out += fmt::format("  {:<6}: {:>24}   [{}]\n",
                                       name,
                                       format_double(greeks[name].get<double>()),
                                       greeks["units"][name].get<std::string>());
                } else {
                    // The word "undefined" is printed where the number would be,
                    // so a reader scanning the column cannot mistake an absent
                    // Greek for one that was simply not requested.
                    out += fmt::format("  {:<6}: {:>24}\n", name, "undefined");
                }
            }

            if (greeks.contains("undefined")) {
                for (const auto& entry : greeks["undefined"].items()) {
                    out += fmt::format(
                        "    {} is undefined: {}\n", entry.key(), entry.value().get<std::string>());
                }
            }
        }

        if (result.contains("diagnostics") && !result["diagnostics"].empty()) {
            out += "\ndiagnostics\n";
            for (const auto& item : result["diagnostics"].items()) {
                append_key_value(out, item.key(), item.value());
            }
        }

        // Warnings are printed after the number, not before, so that a reader who
        // stops at the value still sees them, and never in place of a value.
        if (result.contains("warnings") && !result["warnings"].empty()) {
            out += "\nwarnings\n";
            for (const auto& warning : result["warnings"]) {
                out += fmt::format("  - {}\n", warning.get<std::string>());
            }
        }

        if (result.contains("runtime_seconds")) {
            out += fmt::format("\nruntime  : {} s\n",
                               format_double(result["runtime_seconds"].get<double>()));
        }
    }

    if (document.contains("build_metadata")) {
        const nlohmann::json& build = document["build_metadata"];
        out += "\nprovenance\n";
        out += fmt::format("  commit : {}{}\n",
                           build.value("git_commit_short", "unknown"),
                           build.value("git_dirty", false) ? " (dirty working tree)" : "");
        out += fmt::format("  build  : {} {} / {}\n",
                           build.value("compiler_id", "unknown"),
                           build.value("compiler_version", "unknown"),
                           build.value("build_type", "unknown"));
        out += fmt::format("  host   : {} {} on {}\n",
                           build.value("os_name", "unknown"),
                           build.value("os_version", "unknown"),
                           build.value("cpu_brand", "unknown"));
        out += fmt::format("  when   : {}\n", build.value("timestamp_utc", "unknown"));
    }

    return out;
}

Status write_document(const std::filesystem::path& path, const nlohmann::json& document) {
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            return Status::failure(ErrorCode::IoFailure,
                                   fmt::format("cannot create output directory '{}': {}",
                                               path.parent_path().string(),
                                               ec.message()),
                                   kContext);
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Status::failure(ErrorCode::IoFailure,
                               fmt::format("cannot open '{}' for writing", path.string()),
                               kContext);
    }

    out << document.dump(2) << "\n";
    out.flush();

    // A stream that failed mid-write must not be reported as a successful run;
    // the artifact would be truncated and the result silently lost.
    if (!out) {
        return Status::failure(
            ErrorCode::IoFailure, fmt::format("error while writing '{}'", path.string()), kContext);
    }

    return Status::success();
}

}  // namespace diffusionworks::cli
