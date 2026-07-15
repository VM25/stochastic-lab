#include <diffusionworks/core/config.hpp>
#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace diffusionworks {
namespace {

/// Wraps a body in the minimal valid document envelope.
std::string document(const std::string& body) {
    return "{\"schema_version\": 1" + (body.empty() ? std::string{} : ", " + body) + "}";
}

// ---------------------------------------------------------------------------
// Document-level parsing
//
// These cover the Phase 0 exit gate: a malformed configuration must fail
// explicitly rather than fall back to defaults.
// ---------------------------------------------------------------------------

TEST(ConfigParseTest, AcceptsMinimalValidDocument) {
    const auto parsed = parse_config(document(""));

    ASSERT_TRUE(parsed.ok()) << parsed.error().describe();
    EXPECT_EQ(parsed.value().schema_version(), kConfigSchemaVersion);
}

TEST(ConfigParseTest, RejectsMalformedJson) {
    const auto parsed = parse_config("{\"schema_version\": 1,");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::ParseFailure);
}

TEST(ConfigParseTest, RejectsEmptyInput) {
    const auto parsed = parse_config("");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::ParseFailure);
}

TEST(ConfigParseTest, RejectsNonObjectRoot) {
    const auto parsed = parse_config("[1, 2, 3]");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(parsed.error().message.find("must be an object"), std::string::npos);
}

TEST(ConfigParseTest, RejectsMissingSchemaVersion) {
    const auto parsed = parse_config("{\"market\": {}}");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(parsed.error().message.find("schema_version"), std::string::npos);
}

TEST(ConfigParseTest, RejectsNonIntegerSchemaVersion) {
    const auto parsed = parse_config("{\"schema_version\": \"1\"}");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidConfiguration);
}

// Guessing at the meaning of fields from an unknown schema version would produce
// a run the author never described.
TEST(ConfigParseTest, RejectsUnsupportedSchemaVersion) {
    const auto parsed = parse_config("{\"schema_version\": 99}");

    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(parsed.error().message.find("unsupported schema_version 99"), std::string::npos);
}

TEST(ConfigParseTest, PreservesOriginalJsonForArtifacts) {
    const auto parsed = parse_config(document("\"seed\": 12345"));

    ASSERT_TRUE(parsed.ok());
    EXPECT_EQ(parsed.value().json()["seed"].get<std::int64_t>(), 12345);
}

// ---------------------------------------------------------------------------
// Required accessors
// ---------------------------------------------------------------------------

TEST(ConfigNodeTest, ReadsNumber) {
    const auto parsed = parse_config(document("\"spot\": 100.5"));
    ASSERT_TRUE(parsed.ok());

    const auto spot = parsed.value().root().number("spot");
    ASSERT_TRUE(spot.ok()) << spot.error().describe();
    EXPECT_DOUBLE_EQ(spot.value(), 100.5);
}

TEST(ConfigNodeTest, ReadsIntegerAsNumber) {
    const auto parsed = parse_config(document("\"spot\": 100"));
    ASSERT_TRUE(parsed.ok());

    const auto spot = parsed.value().root().number("spot");
    ASSERT_TRUE(spot.ok());
    EXPECT_DOUBLE_EQ(spot.value(), 100.0);
}

TEST(ConfigNodeTest, ReportsMissingFieldWithPath) {
    const auto parsed = parse_config(document(""));
    ASSERT_TRUE(parsed.ok());

    const auto spot = parsed.value().root().number("spot");
    ASSERT_FALSE(spot.ok());
    EXPECT_EQ(spot.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(spot.error().message.find("'spot'"), std::string::npos);
}

TEST(ConfigNodeTest, ReportsWrongTypeWithPath) {
    const auto parsed = parse_config(document("\"spot\": \"one hundred\""));
    ASSERT_TRUE(parsed.ok());

    const auto spot = parsed.value().root().number("spot");
    ASSERT_FALSE(spot.ok());
    EXPECT_NE(spot.error().message.find("must be a number"), std::string::npos);
    EXPECT_NE(spot.error().message.find("string"), std::string::npos);
}

// A nested field's error must name its full path; "missing 'spot'" is not
// actionable when three sections could each contain a spot.
TEST(ConfigNodeTest, NestedErrorsCarryDottedPath) {
    const auto parsed = parse_config(document("\"market\": {\"rate\": \"x\"}"));
    ASSERT_TRUE(parsed.ok());

    const auto market = parsed.value().root().object("market");
    ASSERT_TRUE(market.ok());
    EXPECT_EQ(market.value().path(), "market");

    const auto rate = market.value().number("rate");
    ASSERT_FALSE(rate.ok());
    EXPECT_NE(rate.error().message.find("'market.rate'"), std::string::npos);
}

TEST(ConfigNodeTest, ReadsString) {
    const auto parsed = parse_config(document("\"method\": \"monte_carlo\""));
    ASSERT_TRUE(parsed.ok());

    const auto method = parsed.value().root().string("method");
    ASSERT_TRUE(method.ok());
    EXPECT_EQ(method.value(), "monte_carlo");
}

TEST(ConfigNodeTest, ReadsBoolean) {
    const auto parsed = parse_config(document("\"antithetic\": true"));
    ASSERT_TRUE(parsed.ok());

    const auto flag = parsed.value().root().boolean("antithetic");
    ASSERT_TRUE(flag.ok());
    EXPECT_TRUE(flag.value());
}

TEST(ConfigNodeTest, ReadsInteger) {
    const auto parsed = parse_config(document("\"paths\": 1000000"));
    ASSERT_TRUE(parsed.ok());

    const auto paths = parsed.value().root().integer("paths");
    ASSERT_TRUE(paths.ok());
    EXPECT_EQ(paths.value(), 1000000);
}

// 1e5 is a natural way to write a path count and is exactly representable, so it
// is accepted; 1000.5 is a mistake and is not.
TEST(ConfigNodeTest, AcceptsIntegralValuedFloatAsInteger) {
    const auto parsed = parse_config(document("\"paths\": 1e5"));
    ASSERT_TRUE(parsed.ok());

    const auto paths = parsed.value().root().integer("paths");
    ASSERT_TRUE(paths.ok());
    EXPECT_EQ(paths.value(), 100000);
}

TEST(ConfigNodeTest, RejectsFractionalFloatAsInteger) {
    const auto parsed = parse_config(document("\"paths\": 1000.5"));
    ASSERT_TRUE(parsed.ok());

    const auto paths = parsed.value().root().integer("paths");
    ASSERT_FALSE(paths.ok());
    EXPECT_NE(paths.error().message.find("must be an integer"), std::string::npos);
}

TEST(ConfigNodeTest, ObjectAccessorRejectsNonObject) {
    const auto parsed = parse_config(document("\"market\": 5"));
    ASSERT_TRUE(parsed.ok());

    const auto market = parsed.value().root().object("market");
    ASSERT_FALSE(market.ok());
    EXPECT_NE(market.error().message.find("must be an object"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Range-checked accessors
// ---------------------------------------------------------------------------

TEST(ConfigRangeTest, PositiveNumberAcceptsPositive) {
    const auto parsed = parse_config(document("\"vol\": 0.2"));
    ASSERT_TRUE(parsed.ok());

    EXPECT_TRUE(parsed.value().root().positive_number("vol").ok());
}

TEST(ConfigRangeTest, PositiveNumberRejectsZero) {
    const auto parsed = parse_config(document("\"vol\": 0.0"));
    ASSERT_TRUE(parsed.ok());

    const auto vol = parsed.value().root().positive_number("vol");
    ASSERT_FALSE(vol.ok());
    EXPECT_NE(vol.error().message.find("strictly positive"), std::string::npos);
}

TEST(ConfigRangeTest, PositiveNumberRejectsNegative) {
    const auto parsed = parse_config(document("\"vol\": -0.2"));
    ASSERT_TRUE(parsed.ok());

    EXPECT_FALSE(parsed.value().root().positive_number("vol").ok());
}

TEST(ConfigRangeTest, NumberInAcceptsBounds) {
    const auto parsed = parse_config(document("\"rho\": -1.0"));
    ASSERT_TRUE(parsed.ok());

    // Correlation of exactly -1 is admissible, so the interval is closed.
    EXPECT_TRUE(parsed.value().root().number_in("rho", -1.0, 1.0).ok());
}

TEST(ConfigRangeTest, NumberInRejectsOutOfRange) {
    const auto parsed = parse_config(document("\"rho\": 1.5"));
    ASSERT_TRUE(parsed.ok());

    const auto rho = parsed.value().root().number_in("rho", -1.0, 1.0);
    ASSERT_FALSE(rho.ok());
    EXPECT_NE(rho.error().message.find("must lie in"), std::string::npos);
}

TEST(ConfigRangeTest, PositiveIntegerRejectsZeroAndNegative) {
    const auto zero = parse_config(document("\"paths\": 0"));
    ASSERT_TRUE(zero.ok());
    EXPECT_FALSE(zero.value().root().positive_integer("paths").ok());

    const auto negative = parse_config(document("\"paths\": -5"));
    ASSERT_TRUE(negative.ok());
    EXPECT_FALSE(negative.value().root().positive_integer("paths").ok());
}

TEST(ConfigRangeTest, IntegerInEnforcesBounds) {
    const auto parsed = parse_config(document("\"threads\": 9999"));
    ASSERT_TRUE(parsed.ok());

    EXPECT_FALSE(parsed.value().root().integer_in("threads", 1, 256).ok());
    EXPECT_TRUE(parsed.value().root().integer_in("threads", 1, 100000).ok());
}

// ---------------------------------------------------------------------------
// Optional accessors
// ---------------------------------------------------------------------------

TEST(ConfigOptionalTest, ReturnsFallbackWhenAbsent) {
    const auto parsed = parse_config(document(""));
    ASSERT_TRUE(parsed.ok());

    const auto threads = parsed.value().root().integer_or("threads", 1);
    ASSERT_TRUE(threads.ok());
    EXPECT_EQ(threads.value(), 1);
}

TEST(ConfigOptionalTest, ReturnsValueWhenPresent) {
    const auto parsed = parse_config(document("\"threads\": 8"));
    ASSERT_TRUE(parsed.ok());

    const auto threads = parsed.value().root().integer_or("threads", 1);
    ASSERT_TRUE(threads.ok());
    EXPECT_EQ(threads.value(), 8);
}

// An optional field still has a type. A present-but-wrong value is an error, not
// a reason to fall back.
TEST(ConfigOptionalTest, StillRejectsWrongTypeWhenPresent) {
    const auto parsed = parse_config(document("\"threads\": \"eight\""));
    ASSERT_TRUE(parsed.ok());

    EXPECT_FALSE(parsed.value().root().integer_or("threads", 1).ok());
}

// ---------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------

TEST(ConfigArrayTest, ReadsNumericArray) {
    const auto parsed = parse_config(document("\"strikes\": [90, 100, 110]"));
    ASSERT_TRUE(parsed.ok());

    const auto strikes = parsed.value().root().array("strikes");
    ASSERT_TRUE(strikes.ok());
    EXPECT_EQ(strikes.value().size(), 3U);
    EXPECT_DOUBLE_EQ(strikes.value().number_at(1).value(), 100.0);
}

TEST(ConfigArrayTest, RejectsOutOfRangeIndex) {
    const auto parsed = parse_config(document("\"strikes\": [90]"));
    ASSERT_TRUE(parsed.ok());

    const auto strikes = parsed.value().root().array("strikes");
    ASSERT_TRUE(strikes.ok());

    const auto missing = strikes.value().number_at(5);
    ASSERT_FALSE(missing.ok());
    EXPECT_NE(missing.error().message.find("out of range"), std::string::npos);
}

TEST(ConfigArrayTest, RejectsNonNumericElement) {
    const auto parsed = parse_config(document("\"strikes\": [90, \"x\"]"));
    ASSERT_TRUE(parsed.ok());

    const auto strikes = parsed.value().root().array("strikes");
    ASSERT_TRUE(strikes.ok());
    EXPECT_FALSE(strikes.value().number_at(1).ok());
}

// ---------------------------------------------------------------------------
// Unknown-key rejection
//
// This is the check that turns a typo into an error. Without it, `volatilty`
// silently leaves volatility at its default and the run produces a number that
// looks entirely reasonable but answers a different question.
// ---------------------------------------------------------------------------

TEST(ConfigStrictTest, AcceptsOnlyKnownKeys) {
    const auto parsed = parse_config(document("\"spot\": 100, \"strike\": 95"));
    ASSERT_TRUE(parsed.ok());

    const Status status =
        parsed.value().root().reject_unknown_keys({"schema_version", "spot", "strike"});
    EXPECT_TRUE(status.ok()) << status.error().describe();
}

TEST(ConfigStrictTest, RejectsTypo) {
    const auto parsed = parse_config(document("\"volatilty\": 0.2"));
    ASSERT_TRUE(parsed.ok());

    const Status status =
        parsed.value().root().reject_unknown_keys({"schema_version", "volatility"});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.error().code, ErrorCode::InvalidConfiguration);
    EXPECT_NE(status.error().message.find("volatilty"), std::string::npos);
}

TEST(ConfigStrictTest, ErrorListsAcceptedKeys) {
    const auto parsed = parse_config(document("\"bogus\": 1"));
    ASSERT_TRUE(parsed.ok());

    const Status status =
        parsed.value().root().reject_unknown_keys({"schema_version", "spot", "strike"});
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.error().message.find("spot"), std::string::npos);
    EXPECT_NE(status.error().message.find("strike"), std::string::npos);
}

TEST(ConfigStrictTest, ReportsEveryUnknownKey) {
    const auto parsed = parse_config(document("\"a\": 1, \"b\": 2"));
    ASSERT_TRUE(parsed.ok());

    const Status status = parsed.value().root().reject_unknown_keys({"schema_version"});
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.error().message.find("'a'"), std::string::npos);
    EXPECT_NE(status.error().message.find("'b'"), std::string::npos);
}

// ---------------------------------------------------------------------------
// File loading
// ---------------------------------------------------------------------------

class ConfigFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("dw_config_test_" +
                std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path write(const std::string& name, const std::string& contents) {
        const std::filesystem::path path = dir_ / name;
        std::ofstream out(path);
        out << contents;
        out.close();
        return path;
    }

    std::filesystem::path dir_;
};

TEST_F(ConfigFileTest, LoadsValidFile) {
    const auto path = write("valid.json", document("\"seed\": 7"));

    const auto loaded = load_config_file(path);
    ASSERT_TRUE(loaded.ok()) << loaded.error().describe();
    EXPECT_EQ(loaded.value().source(), path);
}

// A missing file is an IO problem, not a schema problem. Keeping the codes
// distinct lets a caller tell "you gave me the wrong path" apart from "your
// configuration is wrong".
TEST_F(ConfigFileTest, ReportsMissingFileAsIoFailure) {
    const auto loaded = load_config_file(dir_ / "does_not_exist.json");

    ASSERT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.error().code, ErrorCode::IoFailure);
}

TEST_F(ConfigFileTest, ReportsDirectoryAsIoFailure) {
    const auto loaded = load_config_file(dir_);

    ASSERT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.error().code, ErrorCode::IoFailure);
}

TEST_F(ConfigFileTest, ReportsMalformedFileAsParseFailure) {
    const auto path = write("bad.json", "{not json");

    const auto loaded = load_config_file(path);
    ASSERT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.error().code, ErrorCode::ParseFailure);
}

}  // namespace
}  // namespace diffusionworks
