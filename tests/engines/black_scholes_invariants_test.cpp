#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
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
TEST(BlackScholesLimitsTest, PriceIsExactAtTheDegeneratePayoffKinkWhereGreeksAreNot) {
    // r = q makes the forward equal spot, so F = K = 100 exactly.
    const auto market = MarketState::create(100.0, 0.05, 0.05).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.0).value();

    const BlackScholesTerms t = BlackScholesAnalyticEngine::terms(market, option, model);
    ASSERT_TRUE(t.degenerate);
    ASSERT_DOUBLE_EQ(t.log_moneyness, 0.0) << "test setup must place the forward exactly at strike";

    // The price is exact there: max(0, 0) = 0.
    const auto priced = BlackScholesAnalyticEngine::price(market, option, model);
    ASSERT_TRUE(priced.ok());
    EXPECT_DOUBLE_EQ(priced.value().value, 0.0);

    // The Greek request succeeds, because existence is per Greek: gamma is a
    // Dirac mass and must never be invented, while vega is a genuine one-sided
    // derivative and must not be discarded. See black_scholes_greeks_test.cpp
    // for the full classification.
    const auto greeks = BlackScholesAnalyticEngine::greeks(market, option, model);
    ASSERT_TRUE(greeks.ok()) << greeks.error().describe();
    EXPECT_FALSE(greeks.value().gamma.has_value()) << "gamma must never be invented at the kink";
    EXPECT_FALSE(greeks.value().delta.has_value());
    EXPECT_TRUE(greeks.value().vega.has_value());
    EXPECT_FALSE(greeks.value().undefined.empty()) << "absent Greeks must carry reasons";
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

/// A rigorous upper bound on log(call price), computed entirely in the log domain.
///
/// From the Mills ratio inequality N(-x) < phi(x)/x for x > 0, and dropping the
/// (non-negative) subtracted strike term:
///
///     C = S e^{-qT} N(d1) - K e^{-rT} N(d2) <= S e^{-qT} N(d1)
///                                           <  S e^{-qT} phi(|d1|)/|d1|
///
/// Taking logs gives a bound that never evaluates N itself, so it stays exact
/// where N(d1) underflows to zero:
///
///     log C < log S - qT - d1^2/2 - log(sqrt(2 pi)) - log|d1|
///
/// Valid only for d1 < 0, which is the case of interest.
[[nodiscard]] double log_upper_bound_on_call_price(double spot,
                                                   double strike,
                                                   double rate,
                                                   double dividend_yield,
                                                   double volatility,
                                                   double maturity) {
    const double total_vol = volatility * std::sqrt(maturity);
    const double d1 = (std::log(spot / strike) +
                       (rate - dividend_yield + 0.5 * volatility * volatility) * maturity) /
                      total_vol;
    EXPECT_LT(d1, 0.0) << "the bound requires a negative d1";

    const double abs_d1 = std::abs(d1);
    return std::log(spot) - dividend_yield * maturity - 0.5 * d1 * d1 -
           0.5 * std::log(2.0 * std::numbers::pi) - std::log(abs_d1);
}

// Far enough out, zero is the correct answer -- established by a log-domain
// bound, not by parity.
//
// Parity alone cannot settle this. It constrains only C - P, so a call tail lost
// to underflow together with a compensating error in the put would satisfy it
// exactly. The question "is the true price below what a double can represent?"
// has to be answered without computing the price in double at all.
//
// The Mills ratio gives a rigorous upper bound on log(C) that never evaluates N,
// so it remains exact precisely where N(d1) underflows. If that bound falls below
// log(denorm_min), no double represents the true value and zero is the correctly
// rounded result rather than a lost computation.
TEST(BlackScholesLimitsTest, UnrepresentablyDeepOutOfTheMoneyCallUnderflowsToZero) {
    constexpr double kSpot = 100.0;
    constexpr double kStrike = 1000.0;
    constexpr double kRate = 0.05;
    constexpr double kDividendYield = 0.0;
    constexpr double kVolatility = 0.1;
    constexpr double kMaturity = 0.1;

    const double log_bound = log_upper_bound_on_call_price(
        kSpot, kStrike, kRate, kDividendYield, kVolatility, kMaturity);
    const double log_smallest_double = std::log(std::numeric_limits<double>::denorm_min());

    // The bound lands near -2639 in natural logs (about 1e-1146), while the
    // smallest positive subnormal double is near -744 (about 4.9e-324). The true
    // price is some 800 orders of magnitude below anything double can hold.
    EXPECT_LT(log_bound, log_smallest_double)
        << "this test only establishes graceful underflow if the true price is provably "
           "unrepresentable\n  log(upper bound on C) = "
        << log_bound << "\n  log(denorm_min)       = " << log_smallest_double;

    // Given that, zero is the correctly rounded result. What the engine owes is
    // that the underflow is graceful: exactly zero, never negative (which
    // cancellation could produce, and which the engine rejects outright) and
    // never NaN.
    const double value =
        price(OptionType::Call, kSpot, kStrike, kRate, kDividendYield, kVolatility, kMaturity);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0) << "underflow must not produce a negative price";
    EXPECT_DOUBLE_EQ(value, 0.0);

    // Documented double-precision limitation: for moneyness this extreme the
    // engine cannot distinguish a genuinely worthless option from one worth
    // 1e-1146. Both are zero in double. Resolving them would require a
    // log-domain or extended-precision pricer, which Version 1.0 does not
    // provide and does not claim to.
}

// The representable boundary, approached from the side where the price still
// exists. This pins where the engine stops resolving, so the limitation above is
// a measured statement rather than a vague one.
TEST(BlackScholesLimitsTest, PriceRemainsResolvableUntilTheRepresentableBoundary) {
    // Chosen so d1 ~ -20 and the price is ~1e-86: far into the tail, still a
    // normal double, and exactly zero under a cancelling CDF.
    const double value = price(OptionType::Call, 100.0, 300.0, 0.0, 0.0, 0.05, 1.0);

    EXPECT_GT(value, 0.0) << "the tail collapsed while still representable";
    EXPECT_LT(value, 1e-50);
    EXPECT_TRUE(std::isfinite(value));
}

// The put side of the same unrepresentable case stays perfectly well conditioned:
// it is worth its full discounted strike less the spot. This is a supporting
// observation, not the proof -- it confirms the pair behaves sensibly once the
// log-domain bound has established what the call's true value is.
TEST(BlackScholesLimitsTest, PutSideOfAnUnrepresentableCallRemainsExact) {
    const double put = price(OptionType::Put, 100.0, 1000.0, 0.05, 0.0, 0.1, 0.1);
    const double expected = 1000.0 * std::exp(-0.05 * 0.1) - 100.0;

    EXPECT_NEAR(put, expected, 1e-10);
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
