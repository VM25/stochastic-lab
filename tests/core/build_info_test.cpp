#include <diffusionworks/core/build_info.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

// Every field here is required by PROJECT-SPEC "Required Output Metadata". An
// empty field would silently strip provenance from a published artifact, so the
// test asserts presence rather than any particular value.
TEST(BuildInfoTest, PopulatesCompileTimeProvenance) {
    const BuildInfo info = collect_build_info();

    EXPECT_FALSE(info.version.empty());
    EXPECT_FALSE(info.compiler_id.empty());
    EXPECT_FALSE(info.compiler_version.empty());
    EXPECT_FALSE(info.build_type.empty());
    EXPECT_FALSE(info.cxx_standard.empty());
}

TEST(BuildInfoTest, ReportsProjectCxxStandard) {
    const BuildInfo info = collect_build_info();

    EXPECT_EQ(info.cxx_standard, "20");
}

TEST(BuildInfoTest, PopulatesGitProvenance) {
    const BuildInfo info = collect_build_info();

    EXPECT_FALSE(info.git_commit.empty());
    EXPECT_FALSE(info.git_commit_short.empty());
    EXPECT_FALSE(info.git_branch.empty());
}

// A 40-hex-character commit is what makes a result traceable to a specific tree.
// "unknown" is tolerated only where git itself is unavailable (for example a
// source tarball), which CI is not.
TEST(BuildInfoTest, GitCommitIsAFullHashOrExplicitlyUnknown) {
    const BuildInfo info = collect_build_info();

    if (info.git_commit == "unknown") {
        GTEST_SKIP() << "built outside a git working tree";
    }

    EXPECT_EQ(info.git_commit.size(), 40U);
    for (const char c : info.git_commit) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << "commit hash contains a non-hex character: " << c;
    }
    EXPECT_TRUE(info.git_commit.rfind(info.git_commit_short, 0) == 0)
        << "short hash must prefix the full hash";
}

TEST(BuildInfoTest, PopulatesRuntimeEnvironment) {
    const BuildInfo info = collect_build_info();

    EXPECT_FALSE(info.os_name.empty());
    EXPECT_FALSE(info.os_version.empty());
    EXPECT_FALSE(info.cpu_brand.empty());
    EXPECT_FALSE(info.hostname.empty());
    EXPECT_GT(info.logical_cores, 0U);
}

TEST(BuildInfoTest, TimestampIsIso8601Utc) {
    const BuildInfo info = collect_build_info();

    // "2026-07-15T12:34:56Z"
    ASSERT_EQ(info.timestamp_utc.size(), 20U);
    EXPECT_EQ(info.timestamp_utc[4], '-');
    EXPECT_EQ(info.timestamp_utc[7], '-');
    EXPECT_EQ(info.timestamp_utc[10], 'T');
    EXPECT_EQ(info.timestamp_utc[13], ':');
    EXPECT_EQ(info.timestamp_utc[16], ':');
    EXPECT_EQ(info.timestamp_utc[19], 'Z');
}

// Strings crossing into JSON must not carry an embedded NUL, which sysctl's
// reported size would introduce if it were not trimmed.
TEST(BuildInfoTest, StringsHaveNoEmbeddedNul) {
    const BuildInfo info = collect_build_info();

    EXPECT_EQ(info.cpu_brand.find('\0'), std::string::npos);
    EXPECT_EQ(info.os_name.find('\0'), std::string::npos);
    EXPECT_EQ(info.hostname.find('\0'), std::string::npos);
}

}  // namespace
}  // namespace diffusionworks
