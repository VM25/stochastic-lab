#include <diffusionworks/core/error.hpp>
#include <diffusionworks/core/result.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

TEST(ResultTest, HoldsValueOnSuccess) {
    const auto result = Result<double>::success(42.5);

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(static_cast<bool>(result));
    EXPECT_DOUBLE_EQ(result.value(), 42.5);
}

TEST(ResultTest, HoldsErrorOnFailure) {
    const auto result =
        Result<double>::failure(ErrorCode::ConvergenceFailure, "did not converge", "solver");

    ASSERT_FALSE(result.ok());
    EXPECT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(result.error().code, ErrorCode::ConvergenceFailure);
    EXPECT_EQ(result.error().message, "did not converge");
    EXPECT_EQ(result.error().context, "solver");
}

TEST(ResultTest, ImplicitConstructionFromValue) {
    const Result<int> result = 7;

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value(), 7);
}

TEST(ResultTest, ImplicitConstructionFromError) {
    const Result<int> result = Error{ErrorCode::InvalidArgument, "bad input"};

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

// Reading a value that does not exist is a programming error, not a numerical
// one. It must be loud rather than return a default-constructed value that would
// then flow into a computation.
TEST(ResultTest, ReadingValueOfFailedResultThrows) {
    const auto result = Result<double>::failure(ErrorCode::NonFiniteValue, "nan produced");

    EXPECT_THROW((void)result.value(), DiffusionWorksError);
}

TEST(ResultTest, ReadingErrorOfSuccessfulResultThrows) {
    const auto result = Result<double>::success(1.0);

    EXPECT_THROW((void)result.error(), DiffusionWorksError);
}

TEST(ResultTest, ThrownErrorCarriesOriginalDiagnostics) {
    const auto result =
        Result<double>::failure(ErrorCode::IntegrationFailure, "tolerance not met", "quadrature");

    try {
        (void)result.value();
        FAIL() << "expected DiffusionWorksError";
    } catch (const DiffusionWorksError& e) {
        EXPECT_EQ(e.error().code, ErrorCode::IntegrationFailure);
        EXPECT_EQ(e.error().message, "tolerance not met");
        EXPECT_EQ(e.error().context, "quadrature");
    }
}

TEST(ResultTest, ValueOrReturnsFallbackOnFailure) {
    const auto failed = Result<double>::failure(ErrorCode::InvalidArgument, "bad");
    const auto succeeded = Result<double>::success(3.0);

    EXPECT_DOUBLE_EQ(failed.value_or(-1.0), -1.0);
    EXPECT_DOUBLE_EQ(succeeded.value_or(-1.0), 3.0);
}

TEST(ResultTest, SupportsMoveOnlyTypes) {
    auto result = Result<std::unique_ptr<int>>::success(std::make_unique<int>(5));

    ASSERT_TRUE(result.ok());
    const std::unique_ptr<int> extracted = std::move(result).value();
    ASSERT_NE(extracted, nullptr);
    EXPECT_EQ(*extracted, 5);
}

TEST(ResultTest, SupportsNonTrivialTypes) {
    const auto result = Result<std::vector<double>>::success(std::vector<double>{1.0, 2.0, 3.0});

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().size(), 3U);
    EXPECT_DOUBLE_EQ(result.value()[2], 3.0);
}

TEST(ResultTest, ArrowOperatorReachesValueMembers) {
    const auto result = Result<std::string>::success("diffusion");

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 9U);
    EXPECT_EQ(*result, "diffusion");
}

TEST(ResultTest, MoveExtractsErrorWithoutCopy) {
    auto result = Result<double>::failure(ErrorCode::PathFailure, "non-finite state", "gbm");

    const Error error = std::move(result).error();
    EXPECT_EQ(error.code, ErrorCode::PathFailure);
    EXPECT_EQ(error.message, "non-finite state");
}

// --- Result<void> -----------------------------------------------------------

TEST(StatusTest, SuccessHasNoError) {
    const Status status = Status::success();

    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(static_cast<bool>(status));
    EXPECT_THROW((void)status.error(), DiffusionWorksError);
}

TEST(StatusTest, FailureCarriesError) {
    const Status status =
        Status::failure(ErrorCode::UnstableScheme, "CFL condition violated", "explicit_fd");

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.error().code, ErrorCode::UnstableScheme);
    EXPECT_EQ(status.error().context, "explicit_fd");
}

TEST(StatusTest, DefaultConstructedStatusIsSuccess) {
    const Status status;

    EXPECT_TRUE(status.ok());
}

TEST(StatusTest, ImplicitConstructionFromError) {
    const Status status = Error{ErrorCode::IoFailure, "cannot write"};

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.error().code, ErrorCode::IoFailure);
}

}  // namespace
}  // namespace diffusionworks
