#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>
#include <diffusionworks/core/result.hpp>

#include "options.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace dw = diffusionworks;
namespace cli = diffusionworks::cli;

namespace {

/// Runs one command.
///
/// Phase 0 delivers the CLI shell: argument parsing, configuration loading, and
/// error reporting are real; the numerical work is not yet wired in. Each
/// command therefore loads and validates its configuration (so a malformed file
/// fails explicitly, per the Phase 0 exit gate) and then reports
/// ErrorCode::NotImplemented.
///
/// NotImplemented is deliberate and temporary. It is a declared failure state,
/// not a plausible number: the alternative -- returning 0.0 until the engine
/// lands -- is exactly the failure mode FAILURE-MODES section 16 prohibits.
/// Every command is wired to its engine in a later phase.
[[nodiscard]] dw::Status run_command(const cli::Options& options) {
    if (options.config.has_value()) {
        auto document = dw::load_config_file(*options.config);
        if (!document) {
            return dw::Status::failure(std::move(document).error());
        }
    }

    return dw::Status::failure(dw::ErrorCode::NotImplemented,
                               fmt::format("command '{}' is not yet implemented in this build",
                                           cli::to_string(options.command)),
                               "cli");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string_view> args;
        args.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
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

        const dw::Status status = run_command(options);
        if (!status) {
            fmt::print(stderr, "error: {}\n", status.error().describe());
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
