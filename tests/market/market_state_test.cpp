#include <diffusionworks/core/error.hpp>
#include <diffusionworks/market/market_state.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

TEST(MarketStateTest, AcceptsValidInputs) {
    const auto market = MarketState::create(100.0, 0.05, 0.02);

    ASSERT_TRUE(market.ok()) << market.error().describe();
    EXPECT_DOUBLE_EQ(market.value().spot(), 100.0);
    EXPECT_DOUBLE_EQ(market.value().rate(), 0.05);
    EXPECT_DOUBLE_EQ(market.value().dividend_yield(), 0.02);
}

TEST(MarketStateTest, RejectsNonPositiveSpot) {
    for (const double spot : {0.0, -1.0, -100.0}) {
        const auto market = MarketState::create(spot, 0.05, 0.0);
        ASSERT_FALSE(market.ok()) << "spot = " << spot;
        EXPECT_EQ(market.error().code, ErrorCode::InvalidArgument);
    }
}

TEST(MarketStateTest, RejectsNonFiniteInputs) {
    EXPECT_FALSE(MarketState::create(kNaN, 0.05, 0.0).ok());
    EXPECT_FALSE(MarketState::create(kInf, 0.05, 0.0).ok());
    EXPECT_FALSE(MarketState::create(100.0, kNaN, 0.0).ok());
    EXPECT_FALSE(MarketState::create(100.0, kInf, 0.0).ok());
    EXPECT_FALSE(MarketState::create(100.0, 0.05, kNaN).ok());
    EXPECT_FALSE(MarketState::create(100.0, 0.05, kInf).ok());
}

// Negative policy rates exist. Rejecting them would encode a market regime as a
// mathematical constraint, which is not this type's job.
TEST(MarketStateTest, AcceptsNegativeRates) {
    const auto market = MarketState::create(100.0, -0.005, -0.01);

    ASSERT_TRUE(market.ok()) << market.error().describe();
    EXPECT_DOUBLE_EQ(market.value().rate(), -0.005);
    EXPECT_DOUBLE_EQ(market.value().dividend_yield(), -0.01);
}

TEST(MarketStateTest, ComputesDiscountFactors) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();

    EXPECT_NEAR(market.discount_factor(1.0), std::exp(-0.05), 1e-15);
    EXPECT_NEAR(market.dividend_discount_factor(1.0), std::exp(-0.02), 1e-15);
    EXPECT_DOUBLE_EQ(market.discount_factor(0.0), 1.0);
    EXPECT_DOUBLE_EQ(market.dividend_discount_factor(0.0), 1.0);
}

TEST(MarketStateTest, ComputesForward) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();

    EXPECT_NEAR(market.forward(1.0), 100.0 * std::exp(0.03), 1e-12);
    EXPECT_DOUBLE_EQ(market.forward(0.0), 100.0);
}

// F = S e^{(r-q)T} and the two discount factors are three views of the same
// carry, so they must agree: S e^{-qT} = F e^{-rT}.
TEST(MarketStateTest, ForwardIsConsistentWithDiscountFactors) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();

    for (const double maturity : {0.25, 1.0, 5.0}) {
        const double spot_leg = market.spot() * market.dividend_discount_factor(maturity);
        const double forward_leg = market.forward(maturity) * market.discount_factor(maturity);
        EXPECT_NEAR(spot_leg, forward_leg, 1e-12) << "maturity = " << maturity;
    }
}

}  // namespace
}  // namespace diffusionworks
