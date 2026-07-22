#include <diffusionworks/core/error.hpp>

#include "options.hpp"

#include <gtest/gtest.h>

#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace diffusionworks::cli {
namespace {

Result<Options> parse(std::initializer_list<std::string_view> args) {
    return parse_arguments(std::vector<std::string_view>(args));
}

constexpr std::array<CommandKind, 5> kAllCommands{
    CommandKind::Price,
    CommandKind::Simulate,
    CommandKind::Validate,
    CommandKind::Experiment,
    CommandKind::Calibrate,
};

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

// PROJECT-SPEC "Interfaces" fixes these five command names. They are part of the
// published interface, so the round trip is pinned by test.
TEST(CommandTest, EveryRequiredCommandParsesAndRoundTrips) {
    std::set<std::string_view> seen;

    for (const CommandKind kind : kAllCommands) {
        const std::string_view name = to_string(kind);
        EXPECT_NE(name, "unknown");
        EXPECT_TRUE(seen.insert(name).second) << "duplicate command name: " << name;

        const auto parsed = parse_command(name);
        ASSERT_TRUE(parsed.has_value()) << name;
        EXPECT_EQ(*parsed, kind);
    }

    EXPECT_EQ(seen.size(), kAllCommands.size());
}

TEST(CommandTest, SpecRequiredNamesAreExact) {
    EXPECT_TRUE(parse_command("price").has_value());
    EXPECT_TRUE(parse_command("simulate").has_value());
    EXPECT_TRUE(parse_command("validate").has_value());
    EXPECT_TRUE(parse_command("experiment").has_value());
    EXPECT_TRUE(parse_command("calibrate").has_value());
}

TEST(CommandTest, RejectsUnknownCommand) {
    EXPECT_FALSE(parse_command("compute").has_value());
    EXPECT_FALSE(parse_command("Price").has_value());
    EXPECT_FALSE(parse_command("").has_value());
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

TEST(ParseArgumentsTest, RejectsEmptyArguments) {
    const auto parsed = parse({});

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
}

TEST(ParseArgumentsTest, ParsesBareCommand) {
    const auto parsed = parse({"price"});

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    EXPECT_EQ(parsed.value().command, CommandKind::Price);
    EXPECT_FALSE(parsed.value().config.has_value());
    EXPECT_FALSE(parsed.value().help);
}

TEST(ParseArgumentsTest, ReportsUnknownCommandWithSuggestions) {
    const auto parsed = parse({"frobnicate"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_NE(parsed.error().message.find("unknown command"), std::string::npos);
    EXPECT_NE(parsed.error().message.find("price"), std::string::npos);
}

TEST(ParseArgumentsTest, ParsesAllOptions) {
    const auto parsed = parse({"simulate",
                               "--config",
                               "a.json",
                               "--output",
                               "b.json",
                               "--seed",
                               "42",
                               "--threads",
                               "8",
                               "--format",
                               "json"});

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    const Options& options = parsed.value();
    EXPECT_EQ(options.command, CommandKind::Simulate);
    EXPECT_EQ(options.config->string(), "a.json");
    EXPECT_EQ(options.output->string(), "b.json");
    EXPECT_EQ(options.seed.value(), 42U);
    EXPECT_EQ(options.threads.value(), 8U);
    EXPECT_EQ(options.format.value(), OutputFormat::Json);
}

TEST(ParseArgumentsTest, ParsesShortFlags) {
    const auto parsed = parse({"price", "-c", "a.json", "-o", "b.json", "-t", "4", "-f", "csv"});

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    EXPECT_EQ(parsed.value().config->string(), "a.json");
    EXPECT_EQ(parsed.value().output->string(), "b.json");
    EXPECT_EQ(parsed.value().threads.value(), 4U);
    EXPECT_EQ(parsed.value().format.value(), OutputFormat::Csv);
}

TEST(ParseArgumentsTest, ParsesInlineEqualsForm) {
    const auto parsed = parse({"price", "--config=a.json", "--seed=99", "--format=json"});

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    EXPECT_EQ(parsed.value().config->string(), "a.json");
    EXPECT_EQ(parsed.value().seed.value(), 99U);
    EXPECT_EQ(parsed.value().format.value(), OutputFormat::Json);
}

// An option left unset must stay unset rather than collapse to a default, so
// that a configuration file's value survives when the flag is absent.
TEST(ParseArgumentsTest, AbsentOptionsRemainUnset) {
    const auto parsed = parse({"price"});

    ASSERT_TRUE(parsed.ok());
    EXPECT_FALSE(parsed.value().seed.has_value());
    EXPECT_FALSE(parsed.value().threads.has_value());
    EXPECT_FALSE(parsed.value().format.has_value());
    EXPECT_FALSE(parsed.value().output.has_value());
}

TEST(ParseArgumentsTest, RejectsUnknownOption) {
    const auto parsed = parse({"price", "--bogus", "1"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
    EXPECT_NE(parsed.error().message.find("--bogus"), std::string::npos);
}

TEST(ParseArgumentsTest, RejectsFlagMissingItsValue) {
    for (const std::string_view flag :
         {"--config", "--output", "--seed", "--threads", "--format"}) {
        const auto parsed = parse_arguments({"price", flag});
        ASSERT_FALSE(parsed.ok()) << flag;
        EXPECT_NE(parsed.error().message.find("requires a value"), std::string::npos) << flag;
    }
}

TEST(ParseArgumentsTest, RejectsEmptyInlineValue) {
    const auto parsed = parse({"price", "--config="});

    ASSERT_FALSE(parsed.ok());
    EXPECT_NE(parsed.error().message.find("requires a value"), std::string::npos);
}

TEST(ParseArgumentsTest, RejectsNonNumericSeed) {
    const auto parsed = parse({"price", "--seed", "abc"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
}

// "12abc" must not silently become 12; a partially parsed seed would make a run
// irreproducible while appearing to succeed.
TEST(ParseArgumentsTest, RejectsSeedWithTrailingText) {
    const auto parsed = parse({"price", "--seed", "12abc"});

    ASSERT_FALSE(parsed.ok());
}

TEST(ParseArgumentsTest, RejectsNegativeSeed) {
    const auto parsed = parse({"price", "--seed", "-1"});

    ASSERT_FALSE(parsed.ok());
}

TEST(ParseArgumentsTest, RejectsOverflowingSeed) {
    const auto parsed = parse({"price", "--seed", "99999999999999999999999999"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_NE(parsed.error().message.find("out of range"), std::string::npos);
}

TEST(ParseArgumentsTest, AcceptsLargeSeedWithinRange) {
    const auto parsed = parse({"price", "--seed", "18446744073709551615"});

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    EXPECT_EQ(parsed.value().seed.value(), 18446744073709551615ULL);
}

TEST(ParseArgumentsTest, RejectsZeroThreads) {
    const auto parsed = parse({"price", "--threads", "0"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_NE(parsed.error().message.find("at least 1"), std::string::npos);
}

TEST(ParseArgumentsTest, RejectsUnknownFormat) {
    const auto parsed = parse({"price", "--format", "xml"});

    ASSERT_FALSE(parsed.ok());
    EXPECT_NE(parsed.error().message.find("xml"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

TEST(ParseArgumentsTest, BareHelpRequestsTopLevelUsage) {
    for (const std::string_view flag : {"--help", "-h"}) {
        const auto parsed = parse_arguments({flag});
        ASSERT_TRUE(parsed.ok()) << flag;
        EXPECT_TRUE(parsed.value().help) << flag;
    }
}

TEST(ParseArgumentsTest, CommandHelpSetsHelpAndKeepsCommand) {
    const auto parsed = parse({"calibrate", "--help"});

    ASSERT_TRUE(parsed.ok());
    EXPECT_TRUE(parsed.value().help);
    EXPECT_EQ(parsed.value().command, CommandKind::Calibrate);
}

// `price --help --bogus` must print help rather than fail: a user asking for
// help is not asking for their typo to be adjudicated first.
TEST(ParseArgumentsTest, HelpShortCircuitsRemainingArguments) {
    const auto parsed = parse({"price", "--help", "--bogus"});

    ASSERT_TRUE(parsed.ok());
    EXPECT_TRUE(parsed.value().help);
}

TEST(UsageTest, TopLevelUsageListsEveryCommand) {
    const std::string usage = usage_text();

    for (const CommandKind kind : kAllCommands) {
        EXPECT_NE(usage.find(to_string(kind)), std::string::npos)
            << "usage omits command: " << to_string(kind);
    }
}

TEST(UsageTest, TopLevelUsageListsEveryOptionAndExitCode) {
    const std::string usage = usage_text();

    for (const std::string_view flag :
         {"--config", "--output", "--seed", "--threads", "--format", "--help"}) {
        EXPECT_NE(usage.find(flag), std::string::npos) << "usage omits option: " << flag;
    }
    EXPECT_NE(usage.find("Exit codes"), std::string::npos);
}

TEST(UsageTest, CommandUsageNamesItsCommandAndOptions) {
    for (const CommandKind kind : kAllCommands) {
        const std::string usage = command_usage_text(kind);
        EXPECT_NE(usage.find(to_string(kind)), std::string::npos);
        EXPECT_NE(usage.find("--config"), std::string::npos);
        EXPECT_NE(usage.find("--seed"), std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Formats and exit codes
// ---------------------------------------------------------------------------

TEST(OutputFormatTest, RoundTrips) {
    for (const OutputFormat format :
         {OutputFormat::Console, OutputFormat::Json, OutputFormat::Csv}) {
        const auto parsed = parse_output_format(to_string(format));
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, format);
    }
}

TEST(OutputFormatTest, RejectsUnknown) {
    EXPECT_FALSE(parse_output_format("yaml").has_value());
    EXPECT_FALSE(parse_output_format("JSON").has_value());
}

// A script distinguishes "you invoked this wrong" from "the computation failed"
// by exit code, so the numeric values are part of the interface.
TEST(ExitCodeTest, HasStableNumericValues) {
    EXPECT_EQ(to_int(ExitCode::Success), 0);
    EXPECT_EQ(to_int(ExitCode::RuntimeFailure), 1);
    EXPECT_EQ(to_int(ExitCode::UsageError), 2);
}

}  // namespace
}  // namespace diffusionworks::cli
