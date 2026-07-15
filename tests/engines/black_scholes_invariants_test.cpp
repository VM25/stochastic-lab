#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace diffusionworks {
namespace {

double price(OptionType type,
             double spot,
             double strike,
             double rate,
             double dividend_yield,
             double volatility,
             double maturity) {
    const auto market = MarketState::create(spot, rate, dividend_yield).value();
    const auto option = EuropeanOption::create(type, strike, maturity).value();
    const auto model = BlackScholesModel::create(volatility).value();
    const auto result = BlackScholesAnalyticEngine::price(market, option, model);
    EXPECT_TRUE(result.ok()) << result.error().describe();
    return result.value().value;
}

/// A spread of regimes: in/at/out of the money, short and long dated, low and
/// high volatility, zero and non-zero dividend yield, and a negative rate.
/// VALIDATION-PLAN section 3 requires this coverage; a single at-the-money case
/// would pass while the tails were wrong.
struct Scenario {
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
};

std::vector<Scenario> scenarios() {
    return {
        {100.0, 100.0, 0.05, 0.00, 0.20, 1.00},   // at the money
        {120.0, 100.0, 0.05, 0.00, 0.20, 1.00},   // in the money (call)
        {80.0, 100.0, 0.05, 0.00, 0.20, 1.00},    // out of the money (call)
        {100.0, 100.0, 0.05, 0.03, 0.20, 1.00},   // with dividend yield
        {100.0, 100.0, 0.05, 0.00, 0.05, 1.00},   // low volatility
        {100.0, 100.0, 0.05, 0.00, 0.80, 1.00},   // high volatility
        {100.0, 100.0, 0.05, 0.00, 0.20, 0.01},   // very short dated
        {100.0, 100.0, 0.05, 0.00, 0.20, 30.0},   // very long dated
        {100.0, 100.0, -0.01, 0.00, 0.20, 1.00},  // negative rate
        {100.0, 100.0, 0.00, 0.00, 0.20, 1.00},   // zero rate
        {50.0, 200.0, 0.05, 0.00, 0.20, 1.00},    // deep out of the money (call)
        {200.0, 50.0, 0.05, 0.00, 0.20, 1.00},    // deep in the money (call)
    };
}

// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

// C - P = S e^{-qT} - K e^{-rT}.
//
// This is a model-free no-arbitrage identity, so it holds to floating-point
// accuracy rather than to some modelling tolerance. The bound is scaled by the
// magnitude of the terms being differenced: an absolute tolerance would be far
// too loose for the deep out-of-the-money case and too tight for the long-dated
// one.
TEST(BlackScholesInvariantsTest, PutCallParityHolds) {
    for (const auto& s : scenarios()) {
        const double call = price(
            OptionType::Call, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);
        const double put = price(
            OptionType::Put, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);

        const double discounted_spot = s.spot * std::exp(-s.dividend_yield * s.maturity);
        const double discounted_strike = s.strike * std::exp(-s.rate * s.maturity);

        const double scale = std::max(discounted_spot, discounted_strike);
        EXPECT_NEAR(call - put, discounted_spot - discounted_strike, 1e-12 * scale)
            << "spot=" << s.spot << " strike=" << s.strike << " vol=" << s.volatility
            << " T=" << s.maturity;
    }
}

// ---------------------------------------------------------------------------
// Arbitrage bounds
// ---------------------------------------------------------------------------

TEST(BlackScholesInvariantsTest, CallRespectsArbitrageBounds) {
    for (const auto& s : scenarios()) {
        const double call = price(
            OptionType::Call, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);

        const double discounted_spot = s.spot * std::exp(-s.dividend_yield * s.maturity);
        const double discounted_strike = s.strike * std::exp(-s.rate * s.maturity);
        const double lower = std::max(0.0, discounted_spot - discounted_strike);

        // The tolerance absorbs rounding in the bound itself, not slack in the
        // inequality.
        EXPECT_GE(call, lower - 1e-12 * discounted_spot) << "spot=" << s.spot;
        EXPECT_LE(call, discounted_spot + 1e-12 * discounted_spot) << "spot=" << s.spot;
    }
}

TEST(BlackScholesInvariantsTest, PutRespectsArbitrageBounds) {
    for (const auto& s : scenarios()) {
        const double put = price(
            OptionType::Put, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);

        const double discounted_spot = s.spot * std::exp(-s.dividend_yield * s.maturity);
        const double discounted_strike = s.strike * std::exp(-s.rate * s.maturity);
        const double lower = std::max(0.0, discounted_strike - discounted_spot);

        EXPECT_GE(put, lower - 1e-12 * discounted_strike) << "spot=" << s.spot;
        EXPECT_LE(put, discounted_strike + 1e-12 * discounted_strike) << "spot=" << s.spot;
    }
}

TEST(BlackScholesInvariantsTest, PricesAreNonNegative) {
    for (const auto& s : scenarios()) {
        EXPECT_GE(price(OptionType::Call,
                        s.spot,
                        s.strike,
                        s.rate,
                        s.dividend_yield,
                        s.volatility,
                        s.maturity),
                  0.0);
        EXPECT_GE(price(OptionType::Put,
                        s.spot,
                        s.strike,
                        s.rate,
                        s.dividend_yield,
                        s.volatility,
                        s.maturity),
                  0.0);
    }
}

// ---------------------------------------------------------------------------
// Monotonicity
// ---------------------------------------------------------------------------

TEST(BlackScholesInvariantsTest, CallIncreasesWithSpot) {
    double previous = -1.0;
    for (double spot = 10.0; spot <= 200.0; spot += 5.0) {
        const double current = price(OptionType::Call, spot, 100.0, 0.05, 0.0, 0.2, 1.0);
        EXPECT_GT(current, previous) << "spot = " << spot;
        previous = current;
    }
}

TEST(BlackScholesInvariantsTest, PutDecreasesWithSpot) {
    double previous = std::numeric_limits<double>::infinity();
    for (double spot = 10.0; spot <= 200.0; spot += 5.0) {
        const double current = price(OptionType::Put, spot, 100.0, 0.05, 0.0, 0.2, 1.0);
        EXPECT_LT(current, previous) << "spot = " << spot;
        previous = current;
    }
}

TEST(BlackScholesInvariantsTest, CallDecreasesWithStrike) {
    double previous = std::numeric_limits<double>::infinity();
    for (double strike = 50.0; strike <= 150.0; strike += 5.0) {
        const double current = price(OptionType::Call, 100.0, strike, 0.05, 0.0, 0.2, 1.0);
        EXPECT_LT(current, previous) << "strike = " << strike;
        previous = current;
    }
}

TEST(BlackScholesInvariantsTest, PutIncreasesWithStrike) {
    double previous = -1.0;
    for (double strike = 50.0; strike <= 150.0; strike += 5.0) {
        const double current = price(OptionType::Put, 100.0, strike, 0.05, 0.0, 0.2, 1.0);
        EXPECT_GT(current, previous) << "strike = " << strike;
        previous = current;
    }
}

// Both calls and puts are long volatility: vega is positive for either.
TEST(BlackScholesInvariantsTest, BothTypesIncreaseWithVolatility) {
    for (const OptionType type : {OptionType::Call, OptionType::Put}) {
        double previous = -1.0;
        for (double volatility = 0.01; volatility <= 1.5; volatility += 0.05) {
            const double current = price(type, 100.0, 100.0, 0.05, 0.0, volatility, 1.0);
            EXPECT_GT(current, previous) << to_string(type) << " vol = " << volatility;
            previous = current;
        }
    }
}

// ---------------------------------------------------------------------------
// Limiting cases (MATHEMATICAL-SPEC section 18)
// ---------------------------------------------------------------------------

TEST(BlackScholesLimitsTest, ZeroMaturityReturnsIntrinsicValue) {
    EXPECT_NEAR(price(OptionType::Call, 120.0, 100.0, 0.05, 0.0, 0.2, 0.0), 20.0, 1e-13);
    EXPECT_NEAR(price(OptionType::Call, 80.0, 100.0, 0.05, 0.0, 0.2, 0.0), 0.0, 1e-13);
    EXPECT_NEAR(price(OptionType::Put, 80.0, 100.0, 0.05, 0.0, 0.2, 0.0), 20.0, 1e-13);
    EXPECT_NEAR(price(OptionType::Put, 120.0, 100.0, 0.05, 0.0, 0.2, 0.0), 0.0, 1e-13);
}

TEST(BlackScholesLimitsTest, ZeroVolatilityReturnsDiscountedForwardIntrinsic) {
    // sigma = 0: S_T = S e^{(r-q)T} with certainty, so the call is worth
    // e^{-rT} (F - K)^+ = (S e^{-qT} - K e^{-rT})^+.
    const double spot = 100.0;
    const double strike = 90.0;
    const double rate = 0.05;
    const double maturity = 2.0;

    const double expected = spot - strike * std::exp(-rate * maturity);
    EXPECT_NEAR(price(OptionType::Call, spot, strike, rate, 0.0, 0.0, maturity), expected, 1e-12);

    // The matching put is worthless: the forward finishes above the strike.
    EXPECT_NEAR(price(OptionType::Put, spot, strike, rate, 0.0, 0.0, maturity), 0.0, 1e-13);
}

// The degenerate price is exact, not approximate, so it must be reported with a
// warning rather than an error -- but the warning must be there, because a price
// with no time value is a materially different object from a normal one.
TEST(BlackScholesLimitsTest, DegenerateCasesWarnButSucceed) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model_zero_vol = BlackScholesModel::create(0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 90.0, 1.0).value();

    const auto result = BlackScholesAnalyticEngine::price(market, option, model_zero_vol);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_TRUE(result.value().has_warnings());

    const auto expired = EuropeanOption::create(OptionType::Call, 90.0, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto expired_result = BlackScholesAnalyticEngine::price(market, expired, model);
    ASSERT_TRUE(expired_result.ok()) << expired_result.error().describe();
    EXPECT_TRUE(expired_result.value().has_warnings());
}

// The one place the closed form has no answer. Delta jumps by exp(-qT) across
// the kink and gamma is a Dirac mass, so any finite number returned here would
// be an invention.
TEST(BlackScholesLimitsTest, GreeksFailAtTheDegeneratePayoffKink) {
    // r = q makes the forward equal spot, so F = K = 100 exactly.
    const auto market = MarketState::create(100.0, 0.05, 0.05).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.0).value();

    const BlackScholesTerms t = BlackScholesAnalyticEngine::terms(market, option, model);
    ASSERT_TRUE(t.degenerate);
    ASSERT_DOUBLE_EQ(t.log_moneyness, 0.0) << "test setup must place the forward exactly at strike";

    const auto greeks = BlackScholesAnalyticEngine::greeks(market, option, model);
    ASSERT_FALSE(greeks.ok()) << "Greeks must not be invented at the kink";
    EXPECT_EQ(greeks.error().code, ErrorCode::InvalidArgument);

    // The price itself is still exact there: max(0, 0) = 0.
    const auto priced = BlackScholesAnalyticEngine::price(market, option, model);
    ASSERT_TRUE(priced.ok());
    EXPECT_DOUBLE_EQ(priced.value().value, 0.0);
}

// A resolvable tail must stay strictly positive. Here d2 ~ -3.57, so N(d2) is
// about 1.8e-4: small, but far above anything that could underflow. A price of
// zero would mean the tail had been lost.
TEST(BlackScholesLimitsTest, DeepOutOfTheMoneyCallIsSmallButPositive) {
    const double value = price(OptionType::Call, 100.0, 200.0, 0.0, 0.0, 0.2, 1.0);

    EXPECT_GT(value, 0.0) << "out-of-the-money value collapsed to zero";
    EXPECT_LT(value, 0.01);
    EXPECT_TRUE(std::isfinite(value));
}

// Far enough out, the honest answer is zero.
//
// At K = 1000 with sigma*sqrt(T) ~ 0.032, d1 ~ -72.6 and the tail probability is
// around 1e-1150 -- some 800 orders of magnitude below the smallest positive
// double. Underflow to exactly zero is not a defect but the correctly rounded
// result: no double represents that number. What matters is that the underflow
// is graceful. The value must be exactly zero rather than negative (which
// cancellation could produce and which the engine rejects) or NaN.
TEST(BlackScholesLimitsTest, UnrepresentablyDeepOutOfTheMoneyCallUnderflowsToZero) {
    const double value = price(OptionType::Call, 100.0, 1000.0, 0.05, 0.0, 0.1, 0.1);

    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0) << "underflow must not produce a negative price";
    EXPECT_DOUBLE_EQ(value, 0.0);

    // The matching put is worth its full discounted strike less the spot, and
    // that side stays perfectly well conditioned. Parity therefore still holds
    // exactly, which is what confirms the zero is a genuine limit rather than a
    // lost computation.
    const double put = price(OptionType::Put, 100.0, 1000.0, 0.05, 0.0, 0.1, 0.1);
    const double expected_put = 1000.0 * std::exp(-0.05 * 0.1) - 100.0;
    EXPECT_NEAR(put, expected_put, 1e-10);
}

TEST(BlackScholesLimitsTest, DeepInTheMoneyCallApproachesForwardIntrinsic) {
    const double spot = 1000.0;
    const double strike = 100.0;
    const double rate = 0.05;
    const double maturity = 0.1;

    const double value = price(OptionType::Call, spot, strike, rate, 0.0, 0.1, maturity);
    const double intrinsic = spot - strike * std::exp(-rate * maturity);

    EXPECT_NEAR(value, intrinsic, 1e-8);
}

TEST(BlackScholesLimitsTest, ExtremeParametersStayFinite) {
    EXPECT_TRUE(std::isfinite(price(OptionType::Call, 1e-6, 1e6, 0.05, 0.0, 0.2, 1.0)));
    EXPECT_TRUE(std::isfinite(price(OptionType::Call, 1e6, 1e-6, 0.05, 0.0, 0.2, 1.0)));
    EXPECT_TRUE(std::isfinite(price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 5.0, 1.0)));
    EXPECT_TRUE(std::isfinite(price(OptionType::Call, 100.0, 100.0, 0.05, 0.0, 0.2, 100.0)));
    EXPECT_TRUE(std::isfinite(price(OptionType::Put, 1e-6, 1e6, 0.05, 0.0, 0.2, 1.0)));
    EXPECT_TRUE(std::isfinite(price(OptionType::Put, 1e6, 1e-6, 0.05, 0.0, 0.2, 1.0)));
}

}  // namespace
}  // namespace diffusionworks
