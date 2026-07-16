#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"
#include "output.hpp"
#include "price_command.hpp"
#include "simulate_command.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dw = diffusionworks;
namespace cli = diffusionworks::cli;

namespace {

/// Runs one command and returns its result document.
///
/// Commands not yet wired to an engine report ErrorCode::NotImplemented. That is
/// deliberate and temporary: a declared failure state, not a plausible number.
/// Returning 0.0 until the engine lands is exactly the failure mode
/// FAILURE-MODES section 16 prohibits. Each command sheds this branch in its own
/// phase.
[[nodiscard]] dw::Result<nlohmann::json> run_command(const cli::Options& options) {
    if (!options.config.has_value()) {
        return dw::Result<nlohmann::json>::failure(
            dw::ErrorCode::InvalidArgument,
            fmt::format("command '{}' requires --config; a run is defined by a stored "
                        "configuration so that it can be reproduced",
                        cli::to_string(options.command)),
            "cli");
    }

    auto config = dw::load_config_file(*options.config);
    if (!config) {
        return dw::Result<nlohmann::json>::failure(std::move(config).error());
    }

    switch (options.command) {
        case cli::CommandKind::Price:
            return cli::run_price(config.value(), options);
        case cli::CommandKind::Simulate:
            return cli::run_simulate(config.value(), options);
        case cli::CommandKind::Validate:
        case cli::CommandKind::Experiment:
        case cli::CommandKind::Calibrate:
        case cli::CommandKind::Benchmark:
            break;
    }

    return dw::Result<nlohmann::json>::failure(
        dw::ErrorCode::NotImplemented,
        fmt::format("command '{}' is not yet implemented in this build",
                    cli::to_string(options.command)),
        "cli");
}

/// Emits a result document in the requested format.
[[nodiscard]] dw::Status emit(const nlohmann::json& document, const cli::Options& options) {
    const cli::OutputFormat format = options.format.value_or(cli::OutputFormat::Console);

    if (format == cli::OutputFormat::Csv) {
        return dw::Status::failure(
            dw::ErrorCode::UnsupportedCombination,
            "--format csv is not supported for a single valuation, which is not tabular. Use "
            "console or json.",
            "cli");
    }

    if (options.output.has_value()) {
        // A file destination always receives JSON: the artifact is the
        // machine-readable record, and the console rendering is a view of it.
        // Not const: it is returned by value on the failure path, and const
        // would force a copy where a move would do.
        dw::Status written = cli::write_document(*options.output, document);
        if (!written) {
            return written;
        }
        fmt::print(stdout, "wrote {}\n", options.output->string());
        return dw::Status::success();
    }

    if (format == cli::OutputFormat::Json) {
        fmt::print(stdout, "{}\n", document.dump(2));
    } else {
        fmt::print(stdout, "{}", cli::render_console(document));
    }

    return dw::Status::success();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // argv is a raw pointer-and-count pair; wrapping it in a span makes the
        // bound explicit and keeps the indexing checked, rather than reaching
        // through the pointer directly.
        const std::span<char*> raw_arguments(argv, static_cast<std::size_t>(argc));

        std::vector<std::string_view> args;
        args.reserve(raw_arguments.empty() ? 0 : raw_arguments.size() - 1);
        // Skip argv[0], the program name.
        for (char* argument : raw_arguments.subspan(1)) {
            args.emplace_back(argument);
        }

        if (args.empty()) {
            fmt::print(stderr, "{}", cli::usage_text());
            return cli::to_int(cli::ExitCode::UsageError);
        }

        auto parsed = cli::parse_arguments(args);
        if (!parsed) {
            fmt::print(stderr, "error: {}\n\n", parsed.error().message);
            fmt::print(stderr, "{}", cli::usage_text());
            return cli::to_int(cli::ExitCode::UsageError);
        }

        const cli::Options options = std::move(parsed).value();

        if (options.help) {
            // A bare `--help` with no command yields Options defaulted to
            // `price`; distinguish that from `price --help` by re-reading argv.
            const bool bare_help = (args.front() == "--help" || args.front() == "-h");
            fmt::print(stdout,
                       "{}",
                       bare_help ? cli::usage_text() : cli::command_usage_text(options.command));
            return cli::to_int(cli::ExitCode::Success);
        }

        auto document = run_command(options);
        if (!document) {
            fmt::print(stderr, "error: {}\n", document.error().describe());
            return cli::to_int(cli::ExitCode::RuntimeFailure);
        }

        const dw::Status emitted = emit(document.value(), options);
        if (!emitted) {
            fmt::print(stderr, "error: {}\n", emitted.error().describe());
            return cli::to_int(cli::ExitCode::RuntimeFailure);
        }

        return cli::to_int(cli::ExitCode::Success);
    } catch (const dw::DiffusionWorksError& e) {
        fmt::print(stderr, "fatal: {}\n", e.error().describe());
        return cli::to_int(cli::ExitCode::RuntimeFailure);
    } catch (const std::exception& e) {
        fmt::print(stderr, "fatal: unhandled exception: {}\n", e.what());
        return cli::to_int(cli::ExitCode::RuntimeFailure);
    } catch (...) {
        fmt::print(stderr, "fatal: unhandled non-standard exception\n");
        return cli::to_int(cli::ExitCode::RuntimeFailure);
    }
}
