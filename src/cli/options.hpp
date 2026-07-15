#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace diffusionworks::cli {

/// Process exit codes.
///
/// TESTING-STRATEGY section 10 requires meaningful exit codes, and FAILURE-MODES
/// section 16 requires that a numerical failure never look like success. The
/// distinction between usage errors and runtime failures lets a script tell
/// "you invoked this wrong" apart from "the computation did not converge".
enum class ExitCode : int {
    Success = 0,
    RuntimeFailure = 1,
    UsageError = 2,
};

[[nodiscard]] constexpr int to_int(ExitCode code) noexcept {
    return static_cast<int>(code);
}

/// The subcommands required by PROJECT-SPEC "Interfaces".
enum class CommandKind {
    Price,
    Simulate,
    Validate,
    Experiment,
    Calibrate,
    Benchmark,
};

[[nodiscard]] std::string_view to_string(CommandKind kind) noexcept;

[[nodiscard]] std::optional<CommandKind> parse_command(std::string_view text) noexcept;

/// Rendering of a command's result.
///
/// Console output is for humans; json and csv are the machine-readable
/// artifacts. TECHNICAL-DESIGN section 19 requires that the human and machine
/// renderings report consistent values, so they are two views of one result
/// object rather than two independent formatting paths.
enum class OutputFormat {
    Console,
    Json,
    Csv,
};

[[nodiscard]] std::string_view to_string(OutputFormat format) noexcept;

[[nodiscard]] std::optional<OutputFormat> parse_output_format(std::string_view text) noexcept;

/// A parsed command line.
///
/// Options are optional here rather than defaulted, because a value supplied on
/// the command line must override the configuration file while an absent one
/// must leave the file's value alone. Collapsing "absent" into a default would
/// silently override the configuration.
struct Options {
    CommandKind command{CommandKind::Price};
    std::optional<std::filesystem::path> config;
    std::optional<std::filesystem::path> output;
    std::optional<std::uint64_t> seed;
    std::optional<unsigned int> threads;
    std::optional<OutputFormat> format;
    bool help{false};
};

/// Parses argv into Options.
///
/// Returns a failure for an unknown command, an unknown flag, a flag missing its
/// value, or a malformed value. `--help` short-circuits parsing and yields
/// Options with help=true.
[[nodiscard]] Result<Options> parse_arguments(const std::vector<std::string_view>& args);

/// Top-level usage text, listing every command.
[[nodiscard]] std::string usage_text();

/// Usage text for one command.
[[nodiscard]] std::string command_usage_text(CommandKind kind);

}  // namespace diffusionworks::cli
