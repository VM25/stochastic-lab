#include <diffusionworks/core/error.hpp>
#include <diffusionworks/instruments/european_option.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace diffusionworks {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

TEST(OptionTypeTest, RoundTrips) {
    for (const OptionType type : {OptionType::Call, OptionType::Put}) {
        const auto parsed = parse_option_type(to_string(type));
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, type);
    }
}

TEST(OptionTypeTest, RejectsUnknown) {
    EXPECT_FALSE(parse_option_type("straddle").has_value());
    EXPECT_FALSE(parse_option_type("Call").has_value());
    EXPECT_FALSE(parse_option_type("").has_value());
}

TEST(EuropeanOptionTest, AcceptsValidTerms) {
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0);

    ASSERT_TRUE(option.ok()) << option.error().describe();
    EXPECT_EQ(option.value().type(), OptionType::Call);
    EXPECT_DOUBLE_EQ(option.value().strike(), 100.0);
    EXPECT_DOUBLE_EQ(option.value().maturity(), 1.0);
}

TEST(EuropeanOptionTest, RejectsNonPositiveStrike) {
    for (const double strike : {0.0, -1.0}) {
        const auto option = EuropeanOption::create(OptionType::Call, strike, 1.0);
        ASSERT_FALSE(option.ok()) << "strike = " << strike;
        EXPECT_EQ(option.error().code, ErrorCode::InvalidArgument);
    }
}

TEST(EuropeanOptionTest, RejectsNegativeMaturity) {
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, -0.5);

    ASSERT_FALSE(option.ok());
    EXPECT_EQ(option.error().code, ErrorCode::InvalidArgument);
}

// T = 0 is a required limiting case, not an error: an expired option is worth
// its intrinsic value.
TEST(EuropeanOptionTest, AcceptsZeroMaturity) {
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 0.0);

    ASSERT_TRUE(option.ok()) << option.error().describe();
    EXPECT_DOUBLE_EQ(option.value().maturity(), 0.0);
}

TEST(EuropeanOptionTest, RejectsNonFiniteTerms) {
    EXPECT_FALSE(EuropeanOption::create(OptionType::Call, kNaN, 1.0).ok());
    EXPECT_FALSE(EuropeanOption::create(OptionType::Call, kInf, 1.0).ok());
    EXPECT_FALSE(EuropeanOption::create(OptionType::Call, 100.0, kNaN).ok());
    EXPECT_FALSE(EuropeanOption::create(OptionType::Call, 100.0, kInf).ok());
}

TEST(EuropeanOptionTest, CallPayoff) {
    const auto call = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    EXPECT_DOUBLE_EQ(call.payoff(120.0), 20.0);
    EXPECT_DOUBLE_EQ(call.payoff(100.0), 0.0);
    EXPECT_DOUBLE_EQ(call.payoff(80.0), 0.0);
}

TEST(EuropeanOptionTest, PutPayoff) {
    const auto put = EuropeanOption::create(OptionType::Put, 100.0, 1.0).value();

    EXPECT_DOUBLE_EQ(put.payoff(80.0), 20.0);
    EXPECT_DOUBLE_EQ(put.payoff(100.0), 0.0);
    EXPECT_DOUBLE_EQ(put.payoff(120.0), 0.0);
}

TEST(EuropeanOptionTest, PayoffsAreNonNegative) {
    const auto call = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto put = EuropeanOption::create(OptionType::Put, 100.0, 1.0).value();

    for (double spot = 0.0; spot <= 200.0; spot += 5.0) {
        EXPECT_GE(call.payoff(spot), 0.0) << "spot = " << spot;
        EXPECT_GE(put.payoff(spot), 0.0) << "spot = " << spot;
    }
}

// Payoff parity: (S-K)^+ - (K-S)^+ = S - K for every S, which is the terminal
// statement of put-call parity.
TEST(EuropeanOptionTest, PayoffsSatisfyParity) {
    const auto call = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto put = EuropeanOption::create(OptionType::Put, 100.0, 1.0).value();

    for (double spot = 1.0; spot <= 200.0; spot += 7.0) {
        EXPECT_NEAR(call.payoff(spot) - put.payoff(spot), spot - 100.0, 1e-13) << "spot = " << spot;
    }
}

}  // namespace
}  // namespace diffusionworks
