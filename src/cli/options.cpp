#include "options.hpp"

#include <diffusionworks/core/error.hpp>

#include <fmt/format.h>

#include <charconv>
#include <string>
#include <system_error>
#include <utility>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "cli";

/// Parses an unsigned integer, rejecting negatives, overflow, and trailing text.
/// std::from_chars is used rather than std::stoull because it does not throw and
/// does not silently accept "12abc".
template<typename T>
[[nodiscard]] Result<T> parse_unsigned(std::string_view text, std::string_view flag) {
    T value{};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec == std::errc::result_out_of_range) {
        return Result<T>::failure(ErrorCode::InvalidArgument,
                                  fmt::format("{} value '{}' is out of range", flag, text),
                                  kContext);
    }
    if (ec != std::errc{} || ptr != end) {
        return Result<T>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("{} expects a non-negative integer but received '{}'", flag, text),
            kContext);
    }
    return Result<T>::success(value);
}

}  // namespace

std::string_view to_string(CommandKind kind) noexcept {
    switch (kind) {
        case CommandKind::Price:
            return "price";
        case CommandKind::Simulate:
            return "simulate";
        case CommandKind::Greeks:
            return "greeks";
        case CommandKind::Validate:
            return "validate";
        case CommandKind::Experiment:
            return "experiment";
        case CommandKind::Calibrate:
            return "calibrate";
    }
    return "unknown";
}

std::optional<CommandKind> parse_command(std::string_view text) noexcept {
    if (text == "price") {
        return CommandKind::Price;
    }
    if (text == "simulate") {
        return CommandKind::Simulate;
    }
    if (text == "greeks") {
        return CommandKind::Greeks;
    }
    if (text == "validate") {
        return CommandKind::Validate;
    }
    if (text == "experiment") {
        return CommandKind::Experiment;
    }
    if (text == "calibrate") {
        return CommandKind::Calibrate;
    }
    return std::nullopt;
}

std::string_view to_string(OutputFormat format) noexcept {
    switch (format) {
        case OutputFormat::Console:
            return "console";
        case OutputFormat::Json:
            return "json";
        case OutputFormat::Csv:
            return "csv";
    }
    return "unknown";
}

std::optional<OutputFormat> parse_output_format(std::string_view text) noexcept {
    if (text == "console") {
        return OutputFormat::Console;
    }
    if (text == "json") {
        return OutputFormat::Json;
    }
    if (text == "csv") {
        return OutputFormat::Csv;
    }
    return std::nullopt;
}

Result<Options> parse_arguments(const std::vector<std::string_view>& args) {
    Options options;

    if (args.empty()) {
        return Result<Options>::failure(
            ErrorCode::InvalidArgument, "no command supplied", kContext);
    }

    const std::string_view command_text = args.front();

    // A bare `--help` or `-h` before any command asks for top-level usage.
    if (command_text == "--help" || command_text == "-h") {
        options.help = true;
        return Result<Options>::success(std::move(options));
    }

    const auto command = parse_command(command_text);
    if (!command.has_value()) {
        return Result<Options>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("unknown command '{}'; expected one of: price, simulate, greeks, validate, "
                        "experiment, calibrate",
                        command_text),
            kContext);
    }
    options.command = *command;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];

        if (arg == "--help" || arg == "-h") {
            options.help = true;
            return Result<Options>::success(std::move(options));
        }

        // Every remaining flag takes a value. Supporting `--flag=value` as well
        // as `--flag value` keeps the CLI usable from both shells and scripts.
        std::string_view name = arg;
        std::optional<std::string_view> inline_value;
        if (const auto eq = arg.find('='); eq != std::string_view::npos) {
            name = arg.substr(0, eq);
            inline_value = arg.substr(eq + 1);
        }

        const auto take_value = [&](std::string_view flag) -> Result<std::string_view> {
            if (inline_value.has_value()) {
                if (inline_value->empty()) {
                    return Result<std::string_view>::failure(
                        ErrorCode::InvalidArgument,
                        fmt::format("{} requires a value", flag),
                        kContext);
                }
                return Result<std::string_view>::success(*inline_value);
            }
            if (i + 1 >= args.size()) {
                return Result<std::string_view>::failure(
                    ErrorCode::InvalidArgument, fmt::format("{} requires a value", flag), kContext);
            }
            ++i;
            return Result<std::string_view>::success(args[i]);
        };

        if (name == "--config" || name == "-c") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            options.config = std::filesystem::path(std::string(value.value()));
        } else if (name == "--output" || name == "-o") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            options.output = std::filesystem::path(std::string(value.value()));
        } else if (name == "--seed") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            auto parsed = parse_unsigned<std::uint64_t>(value.value(), "--seed");
            if (!parsed) {
                return Result<Options>::failure(std::move(parsed).error());
            }
            options.seed = parsed.value();
        } else if (name == "--threads" || name == "-t") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            auto parsed = parse_unsigned<unsigned int>(value.value(), "--threads");
            if (!parsed) {
                return Result<Options>::failure(std::move(parsed).error());
            }
            if (parsed.value() == 0) {
                return Result<Options>::failure(
                    ErrorCode::InvalidArgument,
                    "--threads must be at least 1; thread count is never inferred implicitly "
                    "because it is part of a run's published metadata",
                    kContext);
            }
            options.threads = parsed.value();
        } else if (name == "--format" || name == "-f") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            const auto format = parse_output_format(value.value());
            if (!format.has_value()) {
                return Result<Options>::failure(
                    ErrorCode::InvalidArgument,
                    fmt::format("unknown --format '{}'; expected one of: console, json, csv",
                                value.value()),
                    kContext);
            }
            options.format = format;
        } else if (name == "--id") {
            auto value = take_value(name);
            if (!value) {
                return Result<Options>::failure(std::move(value).error());
            }
            if (options.command != CommandKind::Experiment) {
                // Accepted only where it means something. Silently ignoring a flag
                // on the wrong command is how a run ends up not doing what its
                // command line says it did.
                return Result<Options>::failure(
                    ErrorCode::InvalidArgument,
                    fmt::format("--id applies to the experiment command, not '{}'",
                                to_string(options.command)),
                    kContext);
            }
            options.experiment_id = std::string(value.value());
        } else {
            return Result<Options>::failure(ErrorCode::InvalidArgument,
                                            fmt::format("unknown option '{}' for command '{}'",
                                                        name,
                                                        to_string(options.command)),
                                            kContext);
        }
    }

    return Result<Options>::success(std::move(options));
}

std::string usage_text() {
    return R"(DiffusionWorks: Stochastic Derivatives Numerics
A C++ Stochastic Derivatives Modeling, Validation and Performance Engine

Usage:
  diffusionworks <command> [options]

Commands:
  price       Value an instrument with a chosen model and method
  simulate    Generate paths and report simulation diagnostics
  greeks      Estimate a Greek by finite difference, pathwise, or likelihood ratio
  validate    Check results against references and financial invariants
  experiment  Run a configured numerical experiment
  calibrate   Fit Heston parameters to an implied-volatility surface

Options:
  -c, --config <path>    Configuration file (JSON)
  -o, --output <path>    Write the result artifact to this path
      --seed <n>         Master random seed; overrides the configuration
  -t, --threads <n>      Worker thread count; overrides the configuration
  -f, --format <fmt>     Output format: console, json, csv
  -h, --help             Show help for a command

Exit codes:
  0  success
  1  runtime or numerical failure
  2  usage error

Run 'diffusionworks <command> --help' for command-specific help.
)";
}

std::string command_usage_text(CommandKind kind) {
    std::string description;
    std::string detail;

    switch (kind) {
        case CommandKind::Price:
            description = "Value an instrument with a chosen model and method.";
            detail = "Reports the value together with its uncertainty, diagnostics, runtime, and\n"
                     "the metadata needed to reproduce the run.";
            break;
        case CommandKind::Simulate:
            description = "Generate paths and report simulation diagnostics.";
            detail = "Reports path statistics and scheme diagnostics. Non-finite path states are\n"
                     "reported as failures rather than dropped from the sample.";
            break;
        case CommandKind::Greeks:
            description = "Estimate a Greek by finite difference, pathwise, or likelihood ratio.";
            detail =
                "Runs the chosen estimator across a seed set and reports the estimate with its\n"
                "across-seed uncertainty, the bump used, runtime, status, and any warnings.";
            break;
        case CommandKind::Validate:
            description = "Check results against references and financial invariants.";
            detail = "Each check reports PASS, FAIL, WARNING, or INCONCLUSIVE. A FAIL exits\n"
                     "nonzero.";
            break;
        case CommandKind::Experiment:
            description = "Run a configured numerical experiment.";
            detail = "Runs a convergence, stability, or estimator-comparison study from a stored\n"
                     "configuration and writes machine-readable results.";
            break;
        case CommandKind::Calibrate:
            description = "Fit Heston parameters to an implied-volatility surface.";
            detail = "Reports residuals, convergence status, evaluation count, and sensitivity to\n"
                     "the initial guess. A converged optimizer is not by itself a successful\n"
                     "calibration.";
            break;
    }

    // Only the experiment command takes --id, so only its help mentions it.
    const std::string command_options =
        kind == CommandKind::Experiment
            ? "      --id <EXP-NN>      Catalog experiment to run (required)\n"
            : "";

    return fmt::format(R"(diffusionworks {0}

{1}

{2}

Usage:
  diffusionworks {0} [options]

Options:
  -c, --config <path>    Configuration file (JSON)
  -o, --output <path>    Write the result artifact to this path
{3}      --seed <n>         Master random seed; overrides the configuration
  -t, --threads <n>      Worker thread count; overrides the configuration
  -f, --format <fmt>     Output format: console, json, csv
  -h, --help             Show this help
)",
                       to_string(kind),
                       description,
                       detail,
                       command_options);
}

}  // namespace diffusionworks::cli
