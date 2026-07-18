#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/barrier_monte_carlo.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>

#include "support/thread_agreement.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>

namespace diffusionworks {
namespace {

MarketState market() {
    return MarketState::create(100.0, 0.05, 0.0).value();
}

BlackScholesModel model(double volatility = 0.2) {
    return BlackScholesModel::create(volatility).value();
}

BarrierOption monitored(BarrierType type,
                        MonitoringConvention convention,
                        double barrier,
                        std::int64_t dates,
                        double strike = 100.0) {
    return BarrierOption::create(OptionType::Call, type, strike, barrier, 1.0, convention, dates)
        .value();
}

BarrierMonteCarloConfig with_paths(std::int64_t paths, std::uint64_t seed = 20260716) {
    BarrierMonteCarloConfig config;
    config.paths = paths;
    config.seed = seed;
    return config;
}

/// The continuously monitored closed form, which agrees with mpmath to 1e-15 and
/// QuantLib to 1e-9.
double analytic_continuous(double barrier, double volatility = 0.2, double strike = 100.0) {
    const auto option = BarrierOption::create(OptionType::Call,
                                              BarrierType::DownAndOut,
                                              strike,
                                              barrier,
                                              1.0,
                                              MonitoringConvention::Continuous,
                                              std::nullopt)
                            .value();
    return BarrierAnalyticEngine::price(market(), option, model(volatility)).value().value;
}

std::int64_t integer_diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& diagnostic : result.diagnostics) {
        if (diagnostic.name == name) {
            return std::get<std::int64_t>(diagnostic.value);
        }
    }
    ADD_FAILURE() << "no integer diagnostic named " << name;
    return -1;
}

// ---------------------------------------------------------------------------
// bridge_crossing_probability
//
// The bridge law is where the whole correction lives, so it is tested directly
// rather than only through the prices it produces. A price test would confirm the
// formula and the engine together and blame neither.
// ---------------------------------------------------------------------------

// exp(-2 * 0.1 * 0.1 / (0.04 * 0.5)) = exp(-1.0) = 0.36787944117144233, computed
// by hand from the definition and confirmed against Python. An independent value,
// not a re-run of the same expression.
TEST(BridgeCrossingProbabilityTest, MatchesAHandComputedValue) {
    const double p = bridge_crossing_probability(
        /*log_start=*/0.1,
        /*log_end=*/0.1,
        /*log_barrier=*/0.0,
        /*variance_rate=*/0.04,
        /*dt=*/0.5,
        /*down_barrier=*/true);
    EXPECT_NEAR(p, 0.36787944117144233, 1e-15);
}

// An endpoint at or past the barrier is a certain crossing: the observation itself
// caught it, and the bridge has nothing left to decide.
TEST(BridgeCrossingProbabilityTest, IsCertainWhenAnEndpointHasBreached) {
    EXPECT_EQ(bridge_crossing_probability(-0.01, 0.5, 0.0, 0.04, 1.0, true), 1.0);
    EXPECT_EQ(bridge_crossing_probability(0.5, -0.01, 0.0, 0.04, 1.0, true), 1.0);
    EXPECT_EQ(bridge_crossing_probability(0.0, 0.5, 0.0, 0.04, 1.0, true), 1.0);

    EXPECT_EQ(bridge_crossing_probability(0.01, -0.5, 0.0, 0.04, 1.0, false), 1.0);
    EXPECT_EQ(bridge_crossing_probability(-0.5, 0.01, 0.0, 0.04, 1.0, false), 1.0);
}

// The up-barrier case is the down case reflected about the barrier. Both factors
// change sign, so their product -- and hence the probability -- is unchanged. That
// symmetry is the reason one expression serves both directions, so it is asserted
// rather than assumed.
TEST(BridgeCrossingProbabilityTest, IsSymmetricUnderReflection) {
    for (const double a : {0.05, 0.1, 0.3, 0.8}) {
        for (const double b : {0.05, 0.2, 0.5, 1.0}) {
            const double down = bridge_crossing_probability(a, b, 0.0, 0.04, 0.25, true);
            const double up = bridge_crossing_probability(-a, -b, 0.0, 0.04, 0.25, false);
            EXPECT_DOUBLE_EQ(down, up) << "a=" << a << " b=" << b;
        }
    }
}

// Farther from the barrier is less likely to have crossed it, and the decay is in
// the exponent -- which is why the correction matters at coarse monitoring and
// vanishes at fine monitoring.
TEST(BridgeCrossingProbabilityTest, DecaysWithDistanceFromTheBarrier) {
    double previous = 1.0;
    for (const double distance : {0.01, 0.05, 0.1, 0.2, 0.4}) {
        const double p = bridge_crossing_probability(distance, distance, 0.0, 0.04, 0.1, true);
        EXPECT_LT(p, previous) << "distance=" << distance;
        EXPECT_GT(p, 0.0);
        previous = p;
    }
    EXPECT_LT(previous, 1e-10);
}

// Shorter intervals leave less room to wander, so the same endpoints imply a
// smaller crossing probability. In the limit there is no room at all.
TEST(BridgeCrossingProbabilityTest, VanishesAsTheIntervalOrTheDiffusionDoes) {
    EXPECT_EQ(bridge_crossing_probability(0.1, 0.1, 0.0, 0.04, 0.0, true), 0.0);
    EXPECT_EQ(bridge_crossing_probability(0.1, 0.1, 0.0, 0.0, 0.5, true), 0.0);

    double previous = bridge_crossing_probability(0.1, 0.1, 0.0, 0.04, 1.0, true);
    for (const double dt : {0.5, 0.1, 0.01, 0.001}) {
        const double p = bridge_crossing_probability(0.1, 0.1, 0.0, 0.04, dt, true);
        EXPECT_LT(p, previous) << "dt=" << dt;
        previous = p;
    }
}

// ---------------------------------------------------------------------------
// What the engine refuses
// ---------------------------------------------------------------------------

// No simulation observes a barrier continuously. Monitoring very finely and
// reporting the result as continuous would report a discretely monitored price
// under the wrong name -- and at O(1/sqrt(m)) "very finely" is not as close as it
// sounds. EXP-07 measures exactly how far off it is.
TEST(BarrierMonteCarloTest, RefusesContinuousMonitoring) {
    const auto option = BarrierOption::create(OptionType::Call,
                                              BarrierType::DownAndOut,
                                              100.0,
                                              90.0,
                                              1.0,
                                              MonitoringConvention::Continuous,
                                              std::nullopt)
                            .value();

    const auto priced = BarrierMonteCarloEngine::price(market(), option, model(), with_paths(1000));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

TEST(BarrierMonteCarloTest, RefusesFewerThanTwoPaths) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 12);

    const auto priced = BarrierMonteCarloEngine::price(market(), option, model(), with_paths(1));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------

// In-out parity is an identity, not an approximation: a knock-in and a knock-out on
// the same terms partition every path. Under a shared seed the two runs see the same
// paths *and* the same bridge draws, so the identity holds path by path and the
// sampling error cancels exactly. That makes this a far sharper test than comparing
// either leg to a reference -- it holds to floating point, not to a standard error.
TEST(BarrierMonteCarloTest, KnockInPlusKnockOutEqualsVanillaUnderASharedSeed) {
    for (const auto convention :
         {MonitoringConvention::Discrete, MonitoringConvention::BrownianBridge}) {
        const auto knock_out = monitored(BarrierType::DownAndOut, convention, 90.0, 25);
        const auto knock_in = monitored(BarrierType::DownAndIn, convention, 90.0, 25);

        const auto out =
            BarrierMonteCarloEngine::price(market(), knock_out, model(), with_paths(20000));
        const auto in =
            BarrierMonteCarloEngine::price(market(), knock_in, model(), with_paths(20000));
        ASSERT_TRUE(out.ok()) << out.error().describe();
        ASSERT_TRUE(in.ok()) << in.error().describe();

        // The vanilla under the same paths. Priced through the same generator with
        // the same seed, so this too shares the draws rather than approximating them.
        MonteCarloConfig vanilla_config;
        vanilla_config.paths = 20000;
        vanilla_config.seed = 20260716;
        vanilla_config.steps = 25;
        vanilla_config.scheme = DiscretizationScheme::Exact;
        const auto vanilla =
            MonteCarloEngine::price(market(),
                                    EuropeanOption::create(OptionType::Call, 100.0, 1.0).value(),
                                    model(),
                                    vanilla_config);
        ASSERT_TRUE(vanilla.ok()) << vanilla.error().describe();

        EXPECT_NEAR(out.value().value + in.value().value, vanilla.value().value, 1e-9)
            << "convention " << to_string(convention);
    }
}

// A knock-out can only ever pay what the vanilla pays, and sometimes pays nothing
// instead. The price can therefore never exceed the vanilla's -- a Phase 7 exit
// criterion.
//
// The comparison is against the vanilla priced on the *same paths*, not against the
// analytic vanilla, because the inequality is pathwise rather than distributional.
// Against the analytic value this would be a claim about two estimators' expectations
// and a sampled knock-out with a distant barrier sits above it about half the time --
// a failure that would say nothing about the engine. Under common random numbers the
// inequality holds draw by draw, so this asserts the real property and asserts it
// exactly.
TEST(BarrierMonteCarloTest, KnockOutNeverExceedsTheVanillaOnTheSamePaths) {
    for (const auto convention :
         {MonitoringConvention::Discrete, MonitoringConvention::BrownianBridge}) {
        for (const double barrier : {50.0, 70.0, 90.0, 95.0, 99.0}) {
            for (const std::int64_t dates : {5, 50, 250}) {
                const auto option = monitored(BarrierType::DownAndOut, convention, barrier, dates);
                const auto priced =
                    BarrierMonteCarloEngine::price(market(), option, model(), with_paths(20000));
                ASSERT_TRUE(priced.ok()) << priced.error().describe();

                MonteCarloConfig vanilla_config;
                vanilla_config.paths = 20000;
                vanilla_config.seed = 20260716;
                vanilla_config.steps = dates;
                vanilla_config.scheme = DiscretizationScheme::Exact;
                const auto vanilla = MonteCarloEngine::price(
                    market(),
                    EuropeanOption::create(OptionType::Call, 100.0, 1.0).value(),
                    model(),
                    vanilla_config);
                ASSERT_TRUE(vanilla.ok()) << vanilla.error().describe();

                EXPECT_LE(priced.value().value, vanilla.value().value)
                    << to_string(convention) << " B=" << barrier << " m=" << dates;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Against the analytic reference
// ---------------------------------------------------------------------------

// The headline claim: the bridge converges to the *continuous* price, and does so
// at every monitoring frequency rather than only at fine ones. That is what makes
// it a correction rather than a refinement -- and it is what discrete monitoring
// conspicuously fails to do at m = 5.
TEST(BarrierMonteCarloTest, BridgeAgreesWithTheAnalyticContinuousPriceAtEveryFrequency) {
    const double reference = analytic_continuous(90.0);

    for (const std::int64_t dates : {5, 12, 50, 250}) {
        const auto option =
            monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, dates);
        const auto priced =
            BarrierMonteCarloEngine::price(market(), option, model(), with_paths(200000));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        ASSERT_TRUE(priced.value().standard_error.has_value());

        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, 4.0 * *priced.value().standard_error)
            << "m=" << dates << " price=" << priced.value().value << " reference=" << reference;
    }
}

// The up-barrier analytic formulae, checked by a route that shares no algebra with
// them. The bridge simulation reaches the continuous price through the reflection
// principle applied per-interval; Reiner-Rubinstein reaches it through the closed
// form. Agreement is evidence about both, and this is the only check on the
// up-barrier assembly that is not another evaluation of the same expression.
TEST(BarrierMonteCarloTest, BridgeAgreesWithTheAnalyticUpAndOutPrice) {
    const auto up_option = [](double strike, double barrier, std::int64_t dates) {
        return BarrierOption::create(OptionType::Call,
                                     BarrierType::UpAndOut,
                                     strike,
                                     barrier,
                                     1.0,
                                     MonitoringConvention::BrownianBridge,
                                     dates)
            .value();
    };
    const auto analytic = [](double strike, double barrier) {
        const auto option = BarrierOption::create(OptionType::Call,
                                                  BarrierType::UpAndOut,
                                                  strike,
                                                  barrier,
                                                  1.0,
                                                  MonitoringConvention::Continuous,
                                                  std::nullopt)
                                .value();
        return BarrierAnalyticEngine::price(market(), option, model()).value().value;
    };

    for (const double barrier : {110.0, 120.0, 140.0}) {
        for (const std::int64_t dates : {12, 100}) {
            const auto priced = BarrierMonteCarloEngine::price(
                market(), up_option(100.0, barrier, dates), model(), with_paths(200000));
            ASSERT_TRUE(priced.ok()) << priced.error().describe();
            ASSERT_TRUE(priced.value().standard_error.has_value());

            const double reference = analytic(100.0, barrier);
            EXPECT_LT(std::abs(priced.value().value - reference),
                      4.0 * *priced.value().standard_error)
                << "B=" << barrier << " m=" << dates << " mc=" << priced.value().value
                << " analytic=" << reference;
        }
    }
}

// The combinatorial case, through the simulation rather than the formula. An
// up-and-out call struck above its barrier cannot pay: every path that finishes
// above the strike passed the barrier on the way. The simulation must find zero
// paying paths, not merely few.
TEST(BarrierMonteCarloTest, AnUpAndOutCallStruckAboveItsBarrierNeverPays) {
    const auto breached_market = MarketState::create(80.0, 0.05, 0.0).value();
    const auto option = BarrierOption::create(OptionType::Call,
                                              BarrierType::UpAndOut,
                                              100.0,
                                              90.0,
                                              1.0,
                                              MonitoringConvention::BrownianBridge,
                                              50)
                            .value();

    const auto priced =
        BarrierMonteCarloEngine::price(breached_market, option, model(), with_paths(50000));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_EQ(priced.value().value, 0.0);
    EXPECT_EQ(integer_diagnostic(priced.value(), "paid"), 0);
}

// The bias has a direction, and it is not the intuitive one. A barrier unobserved
// between fixes lets a knock-out survive paths that would have killed it, so the
// discretely monitored contract is worth *more* than the continuous one. Asserted
// as a resolved inequality -- several standard errors, not merely a sign -- so it
// cannot pass on noise.
TEST(BarrierMonteCarloTest, DiscreteMonitoringIsBiasedAboveTheContinuousPrice) {
    const double reference = analytic_continuous(90.0);

    for (const std::int64_t dates : {5, 12, 25, 50}) {
        const auto option =
            monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, dates);
        const auto priced =
            BarrierMonteCarloEngine::price(market(), option, model(), with_paths(200000));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        ASSERT_TRUE(priced.value().standard_error.has_value());

        EXPECT_GT(priced.value().value - reference, 5.0 * *priced.value().standard_error)
            << "m=" << dates;
    }
}

// The bias shrinks with frequency, but only as O(1/sqrt(m)): the point of EXP-07.
// Asserted as a trend rather than a rate here -- the rate needs multi-seed evidence,
// which is the experiment's job, not a unit test's.
TEST(BarrierMonteCarloTest, DiscreteBiasShrinksWithMonitoringFrequency) {
    const double reference = analytic_continuous(90.0);

    double previous = std::numeric_limits<double>::infinity();
    for (const std::int64_t dates : {5, 25, 100, 250}) {
        const auto option =
            monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, dates);
        const auto priced =
            BarrierMonteCarloEngine::price(market(), option, model(), with_paths(200000));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();

        const double bias = priced.value().value - reference;
        EXPECT_LT(bias, previous) << "m=" << dates;
        previous = bias;
    }
}

// A barrier the price cannot plausibly reach knocks out almost nothing, so the
// contract is the vanilla. A limiting case per MATHEMATICAL-SPEC section 18, and a
// check that the barrier machinery does not cost anything where it should do
// nothing.
TEST(BarrierMonteCarloTest, ADistantBarrierReproducesTheVanilla) {
    const double vanilla =
        BlackScholesAnalyticEngine::price(
            market(), EuropeanOption::create(OptionType::Call, 100.0, 1.0).value(), model())
            .value()
            .value;

    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 5.0, 50);
    const auto priced =
        BarrierMonteCarloEngine::price(market(), option, model(), with_paths(50000));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();

    EXPECT_NEAR(priced.value().value, vanilla, 4.0 * *priced.value().standard_error);
}

// ---------------------------------------------------------------------------
// The two conventions are different contracts
// ---------------------------------------------------------------------------

// A spot already past the barrier is where the conventions visibly diverge, and
// both answers are right for the contract they describe. A discretely monitored
// contract with dates t_1..t_m simply does not observe t_0, so the option is alive
// until the first fix. A continuously monitored one is already dead. If the two
// agreed here, one of them would be wrong.
TEST(BarrierMonteCarloTest, TheConventionsDisagreeWhenTheSpotHasAlreadyBreached) {
    const auto breached = MarketState::create(85.0, 0.05, 0.0).value();

    const auto bridge =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 12);
    const auto bridge_priced =
        BarrierMonteCarloEngine::price(breached, bridge, model(), with_paths(5000));
    ASSERT_TRUE(bridge_priced.ok()) << bridge_priced.error().describe();
    // Certain crossing on the first interval, for every path: the contract is dead
    // before it starts.
    EXPECT_EQ(bridge_priced.value().value, 0.0);

    const auto discrete =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 12);
    const auto discrete_priced =
        BarrierMonteCarloEngine::price(breached, discrete, model(), with_paths(5000));
    ASSERT_TRUE(discrete_priced.ok()) << discrete_priced.error().describe();
    EXPECT_GT(discrete_priced.value().value, 0.0);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// knocked_by_bridge_only counts the population discrete monitoring silently misses:
// paths whose observed values never breached but which crossed in between. It is
// the correction made countable, and it must be exactly zero under discrete
// monitoring -- a non-zero count there would mean the bridge ran when it was not
// asked to, which is the failure mode this diagnostic exists to expose.
TEST(BarrierMonteCarloTest, OnlyTheBridgeKnocksPathsItDidNotObserveBreaching) {
    const auto discrete =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 12);
    const auto discrete_priced =
        BarrierMonteCarloEngine::price(market(), discrete, model(), with_paths(20000));
    ASSERT_TRUE(discrete_priced.ok()) << discrete_priced.error().describe();
    EXPECT_EQ(integer_diagnostic(discrete_priced.value(), "knocked_by_bridge_only"), 0);
    EXPECT_GT(integer_diagnostic(discrete_priced.value(), "knocked_at_observation"), 0);

    const auto bridge =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 12);
    const auto bridge_priced =
        BarrierMonteCarloEngine::price(market(), bridge, model(), with_paths(20000));
    ASSERT_TRUE(bridge_priced.ok()) << bridge_priced.error().describe();
    EXPECT_GT(integer_diagnostic(bridge_priced.value(), "knocked_by_bridge_only"), 0);
}

// A discrete price is a different contract from the continuous one, and quoting it
// as though it were the same is the error EXP-07 quantifies. The warning is how the
// result says so at the point of use.
TEST(BarrierMonteCarloTest, DiscreteMonitoringCarriesAWarningAndTheBridgeDoesNot) {
    const auto discrete =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 250);
    const auto discrete_priced =
        BarrierMonteCarloEngine::price(market(), discrete, model(), with_paths(5000));
    ASSERT_TRUE(discrete_priced.ok()) << discrete_priced.error().describe();
    EXPECT_TRUE(discrete_priced.value().has_warnings());

    const auto bridge =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 250);
    const auto bridge_priced =
        BarrierMonteCarloEngine::price(market(), bridge, model(), with_paths(5000));
    ASSERT_TRUE(bridge_priced.ok()) << bridge_priced.error().describe();
    EXPECT_FALSE(bridge_priced.value().has_warnings());
}

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

TEST(BarrierMonteCarloTest, IsBitwiseReproducibleUnderTheSameSeed) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 25);

    const auto first =
        BarrierMonteCarloEngine::price(market(), option, model(), with_paths(10000, 7));
    const auto second =
        BarrierMonteCarloEngine::price(market(), option, model(), with_paths(10000, 7));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());

    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
}

// Two seeds must produce two answers. A degenerate stream -- the bridge drawing the
// same uniforms for every path, say -- would leave the engine reproducible and
// wrong, and reproducibility alone cannot tell the difference.
TEST(BarrierMonteCarloTest, DifferentSeedsProduceDifferentEstimates) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 25);

    const auto first =
        BarrierMonteCarloEngine::price(market(), option, model(), with_paths(10000, 1));
    const auto second =
        BarrierMonteCarloEngine::price(market(), option, model(), with_paths(10000, 2));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());

    EXPECT_NE(first.value().value, second.value().value);
}

// ---------------------------------------------------------------------------
// Multithreading (Phase 12)
//
// The path loop runs across deterministic worker partitions with thread-local
// accumulators and diagnostics reduced in block order (ADR-011). One thread is the
// sequential reference; a fixed thread count is reproducible; different counts agree
// up to the reassociation of the merge. The early-knockout break lives inside one
// path's own loop and every stream is keyed by (seed, purpose, index), so
// partitioning by index cannot change which bridge uniform an interval is tested
// against or when a path stops drawing. The knock counts, which the early break
// drives, are exact integer reductions and so must be *identical* at every thread
// count -- the sharpest evidence that partitioning and early knockout do not interact.
// ---------------------------------------------------------------------------

BarrierMonteCarloConfig
threaded(int threads, std::int64_t paths = 120000, std::uint64_t seed = 20260716) {
    BarrierMonteCarloConfig config = with_paths(paths, seed);
    config.threads = threads;
    return config;
}

double double_diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& diagnostic : result.diagnostics) {
        if (diagnostic.name == name) {
            return std::get<double>(diagnostic.value);
        }
    }
    ADD_FAILURE() << "no double diagnostic named " << name;
    return std::numeric_limits<double>::quiet_NaN();
}

// The bridge convention exercises both the early-knockout break and the bridge CRN
// comparison. The price and its standard error agree up to the documented
// reassociation of the merge, and the knock counts -- which the early break drives --
// are bit-identical, because they are exact integer reductions of decisions each path
// makes in isolation from its own keyed streams.
TEST(BarrierMonteCarloThreadingTest, BridgeConventionAgreesAcrossThreadCounts) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 25);

    const auto one = BarrierMonteCarloEngine::price(market(), option, model(), threaded(1));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    const std::int64_t at_obs = integer_diagnostic(one.value(), "knocked_at_observation");
    const std::int64_t by_bridge = integer_diagnostic(one.value(), "knocked_by_bridge_only");
    const std::int64_t paid = integer_diagnostic(one.value(), "paid");
    EXPECT_GT(at_obs, 0);
    EXPECT_GT(by_bridge, 0)
        << "the setup should knock some paths only by the bridge for the check to bite";

    for (const int threads : {2, 4, 8}) {
        const auto many =
            BarrierMonteCarloEngine::price(market(), option, model(), threaded(threads));
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        const std::string tag = "threads=" + std::to_string(threads);
        test::expect_mean_agrees(many.value().value, one.value().value, tag + " value");
        test::expect_error_agrees(
            *many.value().standard_error, *one.value().standard_error, tag + " standard error");

        // The knock counts are exact integer reductions: identical, not merely close.
        EXPECT_EQ(integer_diagnostic(many.value(), "knocked_at_observation"), at_obs)
            << "threads=" << threads << ": an observation-knock count moved under partitioning";
        EXPECT_EQ(integer_diagnostic(many.value(), "knocked_by_bridge_only"), by_bridge)
            << "threads=" << threads << ": a bridge-only knock count moved under partitioning";
        EXPECT_EQ(integer_diagnostic(many.value(), "paid"), paid) << "threads=" << threads;
    }
}

// Discrete monitoring has no bridge stream, but it still terminates early at the first
// observed breach. The knock count and the price are the same at any thread count.
TEST(BarrierMonteCarloThreadingTest, DiscreteConventionAgreesAcrossThreadCounts) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 25);

    const auto one = BarrierMonteCarloEngine::price(market(), option, model(), threaded(1));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    const std::int64_t at_obs = integer_diagnostic(one.value(), "knocked_at_observation");
    EXPECT_GT(at_obs, 0);

    for (const int threads : {2, 4, 8}) {
        const auto many =
            BarrierMonteCarloEngine::price(market(), option, model(), threaded(threads));
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        const std::string tag = "threads=" + std::to_string(threads);
        test::expect_mean_agrees(many.value().value, one.value().value, tag + " value");
        EXPECT_EQ(integer_diagnostic(many.value(), "knocked_at_observation"), at_obs)
            << "threads=" << threads;
        EXPECT_EQ(integer_diagnostic(many.value(), "knocked_by_bridge_only"), 0)
            << "discrete monitoring never knocks by bridge";
    }
}

TEST(BarrierMonteCarloThreadingTest, IsBitReproducibleAtAFixedThreadCount) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 25);
    const auto first = BarrierMonteCarloEngine::price(market(), option, model(), threaded(4));
    const auto second = BarrierMonteCarloEngine::price(market(), option, model(), threaded(4));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
    // The mean bridge probability is an OnlineMoments mean: bit-identical at a fixed count.
    EXPECT_EQ(double_diagnostic(first.value(), "mean_bridge_probability"),
              double_diagnostic(second.value(), "mean_bridge_probability"));
}

TEST(BarrierMonteCarloThreadingTest, RejectsAnInvalidThreadCount) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::Discrete, 90.0, 25);
    const auto priced = BarrierMonteCarloEngine::price(market(), option, model(), threaded(0));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

// More workers than paths must clamp rather than crash, even with the early-knockout
// break in play: each path's monitoring loop is self-contained, so oversubscription
// only changes the partition. The knock counts stay bit-identical and the price agrees
// to the scale-aware tolerance.
TEST(BarrierMonteCarloThreadingTest, HandlesMoreThreadsThanPaths) {
    const auto option =
        monitored(BarrierType::DownAndOut, MonitoringConvention::BrownianBridge, 90.0, 25);

    const auto one = BarrierMonteCarloEngine::price(market(), option, model(), threaded(1, 64));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    const auto many = BarrierMonteCarloEngine::price(market(), option, model(), threaded(1024, 64));
    ASSERT_TRUE(many.ok()) << "threads far exceeding paths must clamp, not fail: "
                           << many.error().describe();
    test::expect_mean_agrees(many.value().value, one.value().value, "value with threads > paths");
    EXPECT_EQ(integer_diagnostic(many.value(), "knocked_at_observation"),
              integer_diagnostic(one.value(), "knocked_at_observation"))
        << "the observation-knock count moved under oversubscription";
    EXPECT_EQ(integer_diagnostic(many.value(), "knocked_by_bridge_only"),
              integer_diagnostic(one.value(), "knocked_by_bridge_only"))
        << "the bridge-only knock count moved under oversubscription";
}

}  // namespace
}  // namespace diffusionworks
