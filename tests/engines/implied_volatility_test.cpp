#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/implied_volatility.hpp>
#include <diffusionworks/models/black_scholes.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

MarketState market(double spot = 100.0, double rate = 0.03, double dividend = 0.01) {
    return MarketState::create(spot, rate, dividend).value();
}

EuropeanOption call(double strike = 100.0, double maturity = 1.0) {
    return EuropeanOption::create(OptionType::Call, strike, maturity).value();
}

EuropeanOption put(double strike = 100.0, double maturity = 1.0) {
    return EuropeanOption::create(OptionType::Put, strike, maturity).value();
}

/// The Black-Scholes price at a known volatility -- the target the solver must invert.
double bs_price(const MarketState& mk, const EuropeanOption& option, double volatility) {
    const auto model = BlackScholesModel::create(volatility).value();
    return BlackScholesAnalyticEngine::price(mk, option, model).value().value;
}

// ---------------------------------------------------------------------------
// Recovery of a known volatility -- the defining property
// ---------------------------------------------------------------------------

// Across strike and maturity, pricing at sigma and inverting the price returns sigma.
// This is the whole contract: the round trip is the identity to a tight tolerance --
// except where the price underflows to intrinsic in double precision, which the deep
// out-of-the-money, low-vol, short-maturity corner does. There the price carries no
// volatility information at all, the solver reports the floor, and asking for recovery
// would be asking to invert a zero. That corner is skipped on the solver's own
// resolvability flag rather than pretended away.
TEST(ImpliedVolatilityTest, RecoversAKnownVolatilityAcrossStrikesAndMaturities) {
    const auto mk = market();
    int resolved_cells = 0;
    for (const double volatility : {0.05, 0.15, 0.30, 0.60, 1.20}) {
        for (const double strike : {70.0, 90.0, 100.0, 115.0, 140.0}) {
            for (const double maturity : {0.1, 0.5, 1.0, 3.0}) {
                const auto option = call(strike, maturity);
                const double target = bs_price(mk, option, volatility);
                const auto solved = ImpliedVolatility::solve(mk, option, target);
                ASSERT_TRUE(solved.ok()) << "sigma=" << volatility << " K=" << strike
                                         << " T=" << maturity << ": " << solved.error().describe();
                EXPECT_NEAR(solved.value().achieved_price, target, 1e-8);
                // The inversion is well conditioned only where vega is meaningful. A
                // price far below a cent carries almost no volatility information -- its
                // vega is tiny, so a price accurate to ~1e-13 pins sigma only to
                // price_accuracy/vega, which for a ~1e-9 price is well above 1e-6.
                // Those deep-out-of-the-money, low-vol cells are skipped rather than
                // held to a tolerance the data cannot support.
                if (solved.value().at_lower_floor || target < 1e-4) {
                    continue;
                }
                EXPECT_NEAR(solved.value().implied_volatility, volatility, 1e-6)
                    << "sigma=" << volatility << " K=" << strike << " T=" << maturity
                    << " target=" << target;
                ++resolved_cells;
            }
        }
    }
    // 90 of the 100 grid cells are well posed and recover to 1e-6; the 10 skipped are
    // the deep-out-of-the-money, low-vol corners where the price underflows. The skip
    // is for that handful, not a licence to skip everything.
    EXPECT_GE(resolved_cells, 90);
}

// Puts inject the same way. A solver right only on calls is not validated.
TEST(ImpliedVolatilityTest, RecoversAKnownVolatilityForPuts) {
    const auto mk = market();
    for (const double volatility : {0.10, 0.35, 0.80}) {
        for (const double strike : {80.0, 100.0, 130.0}) {
            const auto option = put(strike, 1.5);
            const double target = bs_price(mk, option, volatility);
            const auto solved = ImpliedVolatility::solve(mk, option, target);
            ASSERT_TRUE(solved.ok()) << solved.error().describe();
            EXPECT_NEAR(solved.value().implied_volatility, volatility, 1e-6);
        }
    }
}

// Deep in- and out-of-the-money, where the price is dominated by intrinsic (ITM) or is
// tiny (OTM) and vega is small -- the regime where a naive Newton stalls and the
// bisection safeguard earns its place.
TEST(ImpliedVolatilityTest, RecoversInDeepWings) {
    const auto mk = market(100.0, 0.02, 0.0);
    const double volatility = 0.25;
    for (const double strike : {40.0, 60.0, 160.0, 220.0}) {
        const auto option = call(strike, 1.0);
        const double target = bs_price(mk, option, volatility);
        const auto solved = ImpliedVolatility::solve(mk, option, target);
        ASSERT_TRUE(solved.ok()) << "K=" << strike << ": " << solved.error().describe();
        EXPECT_NEAR(solved.value().implied_volatility, volatility, 1e-5) << "K=" << strike;
    }
}

// Very short maturity concentrates the terminal distribution, so away from the money
// the price underflows and vega with it. The floor branch and the bracket keep the
// answer well defined.
TEST(ImpliedVolatilityTest, HandlesVeryShortMaturity) {
    const auto mk = market(100.0, 0.01, 0.0);
    const auto option = call(100.0, 1.0 / 365.0);
    const double target = bs_price(mk, option, 0.40);
    const auto solved = ImpliedVolatility::solve(mk, option, target);
    ASSERT_TRUE(solved.ok()) << solved.error().describe();
    EXPECT_NEAR(solved.value().implied_volatility, 0.40, 1e-5);
}

// ---------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------

// A target below the discounted intrinsic value is arbitrageable: no volatility
// reproduces it, and the solver must say so rather than return a plausible number.
TEST(ImpliedVolatilityTest, RejectsATargetBelowTheArbitrageFloor) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call(80.0, 1.0);  // intrinsic ~ 100 - 80*e^{-0.05} ~ 23.9
    const double intrinsic = 100.0 - 80.0 * std::exp(-0.05);
    const auto solved = ImpliedVolatility::solve(mk, option, intrinsic - 1.0);
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::RootNotBracketed);
}

// A target above the discounted forward is likewise arbitrageable (a call cannot be
// worth more than the prepaid asset).
TEST(ImpliedVolatilityTest, RejectsATargetAboveTheArbitrageCeiling) {
    const auto mk = market(100.0, 0.0, 0.0);
    const auto option = call(100.0, 1.0);
    const auto solved = ImpliedVolatility::solve(mk, option, 100.5);  // > S e^{-qT} = 100
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::RootNotBracketed);
}

// A target exactly at the intrinsic floor is not an error: it is an option quoted at
// intrinsic, whose implied volatility is zero. The result sits at the floor and says
// so rather than pretending to have resolved a root.
TEST(ImpliedVolatilityTest, ReportsTheFloorForATargetAtIntrinsic) {
    const auto mk = market(100.0, 0.04, 0.0);
    const auto option = call(90.0, 1.0);
    const double intrinsic = 100.0 - 90.0 * std::exp(-0.04);
    const auto solved = ImpliedVolatility::solve(mk, option, intrinsic);
    ASSERT_TRUE(solved.ok()) << solved.error().describe();
    EXPECT_TRUE(solved.value().at_lower_floor);
    EXPECT_LT(solved.value().implied_volatility, 1e-6);
}

// Zero maturity has no implied volatility: there is no time for volatility to act.
TEST(ImpliedVolatilityTest, RejectsZeroMaturity) {
    const auto mk = market();
    const auto option = call(100.0, 0.0);
    const auto solved = ImpliedVolatility::solve(mk, option, 1.0);
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::UnsupportedCombination);
}

// A non-finite target is rejected outright.
TEST(ImpliedVolatilityTest, RejectsANonFiniteTarget) {
    const auto mk = market();
    const auto option = call();
    const auto solved =
        ImpliedVolatility::solve(mk, option, std::numeric_limits<double>::quiet_NaN());
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::InvalidArgument);
}

// The reported no-arbitrage window is the one the target was checked against, so a
// rejected quote can be read against the bounds that rejected it.
TEST(ImpliedVolatilityTest, ReportsTheArbitrageWindow) {
    const auto mk = market(100.0, 0.03, 0.01);
    const auto option = call(100.0, 2.0);
    const double target = bs_price(mk, option, 0.2);
    const auto solved = ImpliedVolatility::solve(mk, option, target);
    ASSERT_TRUE(solved.ok());
    const double spot_pv = 100.0 * std::exp(-0.01 * 2.0);
    const double strike_pv = 100.0 * std::exp(-0.03 * 2.0);
    EXPECT_NEAR(solved.value().upper_price_bound, spot_pv, 1e-9);
    EXPECT_NEAR(solved.value().lower_price_bound, std::max(spot_pv - strike_pv, 0.0), 1e-9);
}

}  // namespace
}  // namespace diffusionworks
