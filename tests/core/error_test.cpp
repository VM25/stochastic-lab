#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <array>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace diffusionworks {
namespace {

constexpr std::array<ErrorCode, 12> kAllCodes{
    ErrorCode::InvalidArgument,
    ErrorCode::InvalidConfiguration,
    ErrorCode::ParseFailure,
    ErrorCode::UnsupportedCombination,
    ErrorCode::NonFiniteValue,
    ErrorCode::ConvergenceFailure,
    ErrorCode::RootNotBracketed,
    ErrorCode::IntegrationFailure,
    ErrorCode::UnstableScheme,
    ErrorCode::PathFailure,
    ErrorCode::IoFailure,
    ErrorCode::NotImplemented,
};

// Error codes are serialized into JSON artifacts, so their spellings are part of
// the published output schema. A duplicate would make two distinct failures
// indistinguishable to any downstream consumer.
TEST(ErrorTest, EveryCodeHasDistinctNonEmptyName) {
    std::set<std::string_view> seen;

    for (const ErrorCode code : kAllCodes) {
        const std::string_view name = to_string(code);
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, "unknown") << "code is missing a case in to_string";
        EXPECT_TRUE(seen.insert(name).second) << "duplicate error code name: " << name;
    }

    EXPECT_EQ(seen.size(), kAllCodes.size());
}

TEST(ErrorTest, DescribeIncludesContextMessageAndCode) {
    const Error error{ErrorCode::ConvergenceFailure, "exceeded 100 iterations", "brent_solver"};

    EXPECT_EQ(error.describe(), "brent_solver: exceeded 100 iterations [convergence_failure]");
}

TEST(ErrorTest, DescribeOmitsEmptyContext) {
    const Error error{ErrorCode::InvalidArgument, "strike must be positive"};

    EXPECT_EQ(error.describe(), "strike must be positive [invalid_argument]");
}

TEST(ErrorTest, StreamsThroughDescribe) {
    const Error error{ErrorCode::IoFailure, "disk full", "writer"};

    std::ostringstream out;
    out << error;

    EXPECT_EQ(out.str(), error.describe());
}

TEST(ErrorTest, ExceptionMessageMatchesDescribe) {
    const Error error{ErrorCode::NonFiniteValue, "variance became NaN", "heston"};
    const DiffusionWorksError exception{error};

    EXPECT_EQ(std::string(exception.what()), error.describe());
    EXPECT_EQ(exception.error().code, ErrorCode::NonFiniteValue);
}

}  // namespace
}  // namespace diffusionworks
