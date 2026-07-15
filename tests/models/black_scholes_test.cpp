#include <diffusionworks/core/error.hpp>
#include <diffusionworks/models/black_scholes.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

TEST(BlackScholesModelTest, AcceptsPositiveVolatility) {
    const auto model = BlackScholesModel::create(0.2);

    ASSERT_TRUE(model.ok()) << model.error().describe();
    EXPECT_DOUBLE_EQ(model.value().volatility(), 0.2);
}

// sigma = 0 is the deterministic limit and a required validation case.
TEST(BlackScholesModelTest, AcceptsZeroVolatility) {
    const auto model = BlackScholesModel::create(0.0);

    ASSERT_TRUE(model.ok()) << model.error().describe();
    EXPECT_DOUBLE_EQ(model.value().volatility(), 0.0);
}

// Only sigma^2 enters the dynamics, so a negative sigma would silently behave as
// its absolute value. Rejecting it keeps the parameter's meaning unambiguous.
TEST(BlackScholesModelTest, RejectsNegativeVolatility) {
    const auto model = BlackScholesModel::create(-0.2);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.error().code, ErrorCode::InvalidArgument);
}

TEST(BlackScholesModelTest, RejectsNonFiniteVolatility) {
    EXPECT_FALSE(BlackScholesModel::create(std::numeric_limits<double>::quiet_NaN()).ok());
    EXPECT_FALSE(BlackScholesModel::create(std::numeric_limits<double>::infinity()).ok());
}

TEST(BlackScholesModelTest, ComputesTotalVolatility) {
    const auto model = BlackScholesModel::create(0.2).value();

    EXPECT_NEAR(model.total_volatility(1.0), 0.2, 1e-15);
    EXPECT_NEAR(model.total_volatility(4.0), 0.4, 1e-15);
    EXPECT_NEAR(model.total_volatility(0.25), 0.1, 1e-15);
}

// Both degenerate routes must land on exactly zero, since that is the value the
// engine branches on.
TEST(BlackScholesModelTest, TotalVolatilityIsZeroInDegenerateLimits) {
    EXPECT_DOUBLE_EQ(BlackScholesModel::create(0.2).value().total_volatility(0.0), 0.0);
    EXPECT_DOUBLE_EQ(BlackScholesModel::create(0.0).value().total_volatility(1.0), 0.0);
    EXPECT_DOUBLE_EQ(BlackScholesModel::create(0.0).value().total_volatility(0.0), 0.0);
}

// Total volatility scales as sqrt(T): doubling it requires four times the
// maturity. This is the property every sqrt(T) elsewhere in the engine relies on.
TEST(BlackScholesModelTest, TotalVolatilityScalesAsSqrtTime) {
    const auto model = BlackScholesModel::create(0.3).value();

    EXPECT_NEAR(model.total_volatility(4.0) / model.total_volatility(1.0), 2.0, 1e-14);
    EXPECT_NEAR(model.total_volatility(9.0) / model.total_volatility(1.0), 3.0, 1e-14);
}

}  // namespace
}  // namespace diffusionworks
