#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace diffusionworks {
namespace {

BarrierOption continuous(BarrierType type, double strike, double barrier, double maturity = 1.0) {
    return BarrierOption::create(OptionType::Call,
                                 type,
                                 strike,
                                 barrier,
                                 maturity,
                                 MonitoringConvention::Continuous,
                                 std::nullopt)
        .value();
}

double vanilla_price(const MarketState& market,
                     const BlackScholesModel& model,
                     double strike,
                     double maturity = 1.0) {
    return BlackScholesAnalyticEngine::price(
               market, EuropeanOption::create(OptionType::Call, strike, maturity).value(), model)
        .value()
        .value;
}

// ---------------------------------------------------------------------------
// Against an independent reference
//
// 8.665471658245668 is the Reiner-Rubinstein value at S=K=100, B=90, r=0.05,
// q=0, sigma=0.2, T=1, computed to 40 digits by mpmath. It was *separately*
// confirmed by a Brownian-bridge-corrected simulation -- a route that shares no
// algebra with the closed form -- which agreed to within 1.2 standard errors at
// three different monitoring frequencies. So this constant is not a transcription
// of the same formula checked against itself.
// ---------------------------------------------------------------------------

TEST(BarrierAnalyticTest, MatchesTheHighPrecisionReference) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced = BarrierAnalyticEngine::price(
        market, continuous(BarrierType::DownAndOut, 100.0, 90.0), model);
    ASSERT_TRUE(priced.ok()) << priced.error().describe();

    EXPECT_NEAR(priced.value().value, 8.665471658245668, 1e-12);
}

// In-out parity is an identity, not an approximation: a knock-in and a knock-out
// on the same terms partition every path, so together they are the vanilla. It
// holds exactly, independently of the model, and is the sharpest check available
// on a barrier engine.
TEST(BarrierAnalyticTest, KnockInPlusKnockOutEqualsVanilla) {
    const auto model = BlackScholesModel::create(0.25).value();

    for (const double spot : {95.0, 100.0, 120.0, 150.0}) {
        for (const double barrier : {70.0, 80.0, 90.0}) {
            for (const double dividend : {0.0, 0.03}) {
                const auto market = MarketState::create(spot, 0.05, dividend).value();

                const auto out = BarrierAnalyticEngine::price(
                    market, continuous(BarrierType::DownAndOut, 100.0, barrier), model);
                const auto in = BarrierAnalyticEngine::price(
                    market, continuous(BarrierType::DownAndIn, 100.0, barrier), model);
                ASSERT_TRUE(out.ok()) << out.error().describe();
                ASSERT_TRUE(in.ok()) << in.error().describe();

                EXPECT_NEAR(out.value().value + in.value().value,
                            vanilla_price(market, model, 100.0),
                            1e-10)
                    << "S=" << spot << " B=" << barrier << " q=" << dividend;
            }
        }
    }
}

// The exit gate's requirement, stated as a test: a barrier price never exceeds
// its vanilla. A knock-out pays on a subset of the vanilla's paths and a knock-in
// on the complement, so neither can be worth more.
TEST(BarrierAnalyticTest, NeverExceedsTheCorrespondingVanilla) {
    const auto model = BlackScholesModel::create(0.3).value();

    for (const double spot : {91.0, 95.0, 100.0, 130.0}) {
        for (const double barrier : {60.0, 80.0, 90.0}) {
            const auto market = MarketState::create(spot, 0.05, 0.0).value();
            const double vanilla = vanilla_price(market, model, 100.0);

            for (const auto type : {BarrierType::DownAndOut, BarrierType::DownAndIn}) {
                const auto priced =
                    BarrierAnalyticEngine::price(market, continuous(type, 100.0, barrier), model);
                ASSERT_TRUE(priced.ok()) << priced.error().describe();

                EXPECT_LE(priced.value().value, vanilla + 1e-10)
                    << to_string(type) << " at S=" << spot << " B=" << barrier << " is worth "
                    << priced.value().value << " against a vanilla of " << vanilla;
                EXPECT_GE(priced.value().value, -1e-10) << to_string(type);
            }
        }
    }
}

// A knock-out is worth more the further the barrier is from the spot, because
// fewer paths reach it. Monotonicity the formula must exhibit and a sign error
// would break.
TEST(BarrierAnalyticTest, KnockOutRisesAsTheBarrierRecedes) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    double previous = -1.0;
    for (const double barrier : {95.0, 90.0, 80.0, 70.0, 50.0, 20.0}) {
        const auto priced = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndOut, 100.0, barrier), model);
        ASSERT_TRUE(priced.ok()) << "B=" << barrier << ": " << priced.error().describe();

        EXPECT_GT(priced.value().value, previous) << "at B=" << barrier;
        previous = priced.value().value;
    }

    // A barrier far enough away is never reached, so the knock-out becomes the
    // vanilla. The limit the formula must approach.
    const auto distant = BarrierAnalyticEngine::price(
        market, continuous(BarrierType::DownAndOut, 100.0, 1.0), model);
    ASSERT_TRUE(distant.ok());
    EXPECT_NEAR(distant.value().value, vanilla_price(market, model, 100.0), 1e-6);
}

// ---------------------------------------------------------------------------
// Already breached: a price, not an error
// ---------------------------------------------------------------------------

TEST(BarrierAnalyticTest, BreachedKnockOutIsExactlyWorthless) {
    const auto model = BlackScholesModel::create(0.2).value();

    for (const double spot : {89.9, 85.0, 50.0, 1.0}) {
        const auto market = MarketState::create(spot, 0.05, 0.0).value();
        const auto priced = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndOut, 100.0, 90.0), model);

        ASSERT_TRUE(priced.ok()) << "S=" << spot << ": a breached barrier is a price, not an error";
        EXPECT_DOUBLE_EQ(priced.value().value, 0.0) << "S=" << spot;
        EXPECT_TRUE(priced.value().has_warnings())
            << "the reader must be told the contract had already resolved";
    }
}

TEST(BarrierAnalyticTest, BreachedKnockInIsExactlyTheVanilla) {
    const auto model = BlackScholesModel::create(0.2).value();
    const auto market = MarketState::create(85.0, 0.05, 0.0).value();

    const auto priced = BarrierAnalyticEngine::price(
        market, continuous(BarrierType::DownAndIn, 100.0, 90.0), model);
    ASSERT_TRUE(priced.ok());

    // Exactly, not nearly: once knocked in, the contract *is* the vanilla.
    EXPECT_DOUBLE_EQ(priced.value().value, vanilla_price(market, model, 100.0));
    EXPECT_TRUE(priced.value().has_warnings());
}

// Touching the barrier counts as breaching it. The convention is stated in the
// instrument, and it is pinned here because the alternative is defensible and
// gives a different answer.
TEST(BarrierAnalyticTest, TouchingTheBarrierCountsAsBreaching) {
    const auto model = BlackScholesModel::create(0.2).value();
    const auto at_barrier = MarketState::create(90.0, 0.05, 0.0).value();

    const auto priced = BarrierAnalyticEngine::price(
        at_barrier, continuous(BarrierType::DownAndOut, 100.0, 90.0), model);
    ASSERT_TRUE(priced.ok());
    EXPECT_DOUBLE_EQ(priced.value().value, 0.0)
        << "S = B exactly: the standard convention treats a touch as a breach";
}

// ---------------------------------------------------------------------------
// Near-barrier behaviour, documented rather than smoothed
// ---------------------------------------------------------------------------

// Approaching the barrier from above, the knock-out must fall continuously to
// zero. The reflected term nearly cancels the vanilla here, which is where the
// formula is most fragile -- so the limit is checked rather than assumed.
TEST(BarrierAnalyticTest, KnockOutFallsContinuouslyToZeroAtTheBarrier) {
    const auto model = BlackScholesModel::create(0.2).value();

    double previous = 1e9;
    for (const double spot : {95.0, 92.0, 91.0, 90.5, 90.1, 90.01, 90.001}) {
        const auto market = MarketState::create(spot, 0.05, 0.0).value();
        const auto priced = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndOut, 100.0, 90.0), model);
        ASSERT_TRUE(priced.ok()) << "S=" << spot << ": " << priced.error().describe();

        EXPECT_GE(priced.value().value, 0.0) << "S=" << spot;
        EXPECT_LT(priced.value().value, previous)
            << "S=" << spot << ": must fall as S approaches B";
        previous = priced.value().value;
    }

    // At one basis point above the barrier the option is nearly worthless, and the
    // formula still says so rather than losing the value to cancellation.
    const auto near = MarketState::create(90.001, 0.05, 0.0).value();
    const double value =
        BarrierAnalyticEngine::price(near, continuous(BarrierType::DownAndOut, 100.0, 90.0), model)
            .value()
            .value;
    EXPECT_LT(value, 0.01) << "the knock-out at S/B = 1.00001 should be nearly worthless";
}

// ---------------------------------------------------------------------------
// Degenerate limits
// ---------------------------------------------------------------------------

// With no time or no diffusion the path cannot move, so an unbreached barrier is
// never breached and the knock-out is simply the vanilla.
TEST(BarrierAnalyticTest, DegenerateLimitsCollapseToTheVanillaOrZero) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();

    // sigma -> 0
    {
        const auto still = BlackScholesModel::create(0.0).value();
        const auto out = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndOut, 100.0, 90.0), still);
        const auto in = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndIn, 100.0, 90.0), still);
        ASSERT_TRUE(out.ok());
        ASSERT_TRUE(in.ok());

        EXPECT_DOUBLE_EQ(out.value().value, vanilla_price(market, still, 100.0));
        EXPECT_DOUBLE_EQ(in.value().value, 0.0);
        EXPECT_TRUE(out.value().has_warnings());
    }

    // T -> 0
    {
        const auto model = BlackScholesModel::create(0.2).value();
        const auto out = BarrierAnalyticEngine::price(
            market, continuous(BarrierType::DownAndOut, 100.0, 90.0, 0.0), model);
        ASSERT_TRUE(out.ok());
        EXPECT_DOUBLE_EQ(out.value().value, vanilla_price(market, model, 100.0, 0.0));
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

// A discretely monitored barrier is a different contract, worth more than its
// continuous value for a knock-out, and the gap closes only as O(1/sqrt(m)).
// Returning the continuous number for it would be a plausible answer to a
// question nobody asked.
TEST(BarrierAnalyticTest, RefusesDiscretelyMonitoredContracts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    for (const auto convention :
         {MonitoringConvention::Discrete, MonitoringConvention::BrownianBridge}) {
        const auto option =
            BarrierOption::create(
                OptionType::Call, BarrierType::DownAndOut, 100.0, 90.0, 1.0, convention, 252)
                .value();

        const auto priced = BarrierAnalyticEngine::price(market, option, model);
        ASSERT_FALSE(priced.ok()) << to_string(convention);
        EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
    }
}

// The B > K branch needs a second pair of terms. Pricing it with the implemented
// branch would return a plausible wrong number, so it is refused.
//
// The spot must be above the barrier for this to be reachable at all: a
// down-and-out with S below B is already breached and resolves before the branch
// matters.
TEST(BarrierAnalyticTest, RefusesTheUnimplementedBranchWhereBarrierExceedsStrike) {
    const auto market = MarketState::create(120.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced = BarrierAnalyticEngine::price(
        market, continuous(BarrierType::DownAndOut, 100.0, 110.0), model);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::NotImplemented);
}

TEST(BarrierAnalyticTest, RefusesUpBarriersAndPuts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto up = BarrierOption::create(OptionType::Call,
                                          BarrierType::UpAndOut,
                                          100.0,
                                          120.0,
                                          1.0,
                                          MonitoringConvention::Continuous,
                                          std::nullopt)
                        .value();
    EXPECT_FALSE(BarrierAnalyticEngine::price(market, up, model).ok());

    const auto put = BarrierOption::create(OptionType::Put,
                                           BarrierType::DownAndOut,
                                           100.0,
                                           90.0,
                                           1.0,
                                           MonitoringConvention::Continuous,
                                           std::nullopt)
                         .value();
    EXPECT_FALSE(BarrierAnalyticEngine::price(market, put, model).ok());
}

// ---------------------------------------------------------------------------
// The contract itself
// ---------------------------------------------------------------------------

TEST(BarrierOptionTest, RequiresAMonitoringCountExactlyWhenItMeansSomething) {
    // Continuous monitoring has no observation dates; supplying a count would let
    // a caller believe theirs was used.
    EXPECT_FALSE(BarrierOption::create(OptionType::Call,
                                       BarrierType::DownAndOut,
                                       100.0,
                                       90.0,
                                       1.0,
                                       MonitoringConvention::Continuous,
                                       252)
                     .ok());
    EXPECT_TRUE(BarrierOption::create(OptionType::Call,
                                      BarrierType::DownAndOut,
                                      100.0,
                                      90.0,
                                      1.0,
                                      MonitoringConvention::Continuous,
                                      std::nullopt)
                    .ok());

    // Discrete monitoring without dates is meaningless.
    EXPECT_FALSE(BarrierOption::create(OptionType::Call,
                                       BarrierType::DownAndOut,
                                       100.0,
                                       90.0,
                                       1.0,
                                       MonitoringConvention::Discrete,
                                       std::nullopt)
                     .ok());
    EXPECT_FALSE(BarrierOption::create(OptionType::Call,
                                       BarrierType::DownAndOut,
                                       100.0,
                                       90.0,
                                       1.0,
                                       MonitoringConvention::Discrete,
                                       0)
                     .ok());
}

TEST(BarrierOptionTest, PayoffFollowsTheKnockDirection) {
    const auto out = continuous(BarrierType::DownAndOut, 100.0, 90.0);
    EXPECT_DOUBLE_EQ(out.payoff(120.0, false), 20.0) << "unhit knock-out pays";
    EXPECT_DOUBLE_EQ(out.payoff(120.0, true), 0.0) << "hit knock-out pays nothing";

    const auto in = continuous(BarrierType::DownAndIn, 100.0, 90.0);
    EXPECT_DOUBLE_EQ(in.payoff(120.0, true), 20.0) << "hit knock-in pays";
    EXPECT_DOUBLE_EQ(in.payoff(120.0, false), 0.0) << "unhit knock-in pays nothing";

    // The partition: for any terminal value and any hit outcome, exactly one of
    // the pair pays, and their sum is the vanilla payoff.
    for (const double terminal : {50.0, 100.0, 150.0}) {
        for (const bool hit : {false, true}) {
            EXPECT_DOUBLE_EQ(out.payoff(terminal, hit) + in.payoff(terminal, hit),
                             std::max(terminal - 100.0, 0.0))
                << "terminal=" << terminal << " hit=" << hit;
        }
    }
}

TEST(BarrierOptionTest, BreachIsInclusiveOfTheBarrierItself) {
    const auto down = continuous(BarrierType::DownAndOut, 100.0, 90.0);
    EXPECT_TRUE(down.breaches(89.9));
    EXPECT_TRUE(down.breaches(90.0)) << "touching counts";
    EXPECT_FALSE(down.breaches(90.1));

    const auto up = BarrierOption::create(OptionType::Call,
                                          BarrierType::UpAndOut,
                                          100.0,
                                          110.0,
                                          1.0,
                                          MonitoringConvention::Continuous,
                                          std::nullopt)
                        .value();
    EXPECT_TRUE(up.breaches(110.1));
    EXPECT_TRUE(up.breaches(110.0)) << "touching counts";
    EXPECT_FALSE(up.breaches(109.9));
}

// An already-breached contract is a limiting case with a well-defined price, not
// an invalid contract. The spot is not a contract term, so the instrument does not
// see it.
TEST(BarrierOptionTest, DoesNotValidateTheBarrierAgainstASpotItCannotSee) {
    // A "down-and-out" with a barrier above any plausible spot is a perfectly
    // well-formed contract; it simply resolves immediately.
    EXPECT_TRUE(BarrierOption::create(OptionType::Call,
                                      BarrierType::DownAndOut,
                                      100.0,
                                      1000.0,
                                      1.0,
                                      MonitoringConvention::Continuous,
                                      std::nullopt)
                    .ok());
}

}  // namespace
}  // namespace diffusionworks
