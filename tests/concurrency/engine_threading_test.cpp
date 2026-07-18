// Fast, concurrency-labelled threading smoke tests for every parallelised Monte Carlo
// engine, so ThreadSanitizer in CI exercises each engine's *own* path loop -- not only
// the shared parallel_reduce facility (parallel_reduce_test.cpp). The detailed
// statistical threading tests live in the per-engine suites (labelled statistical, run
// under the debug/release CI jobs); these are deliberately small -- a few thousand paths
// -- so they run under TSan's slowdown, because their job is to catch a data race, not
// to validate a price. Each still checks that more threads agree with one thread to the
// scale-aware tolerance, so a race that corrupted the reduction would show up as a
// disagreement even if TSan somehow missed it.

#include <diffusionworks/engines/barrier_monte_carlo.hpp>
#include <diffusionworks/engines/greeks_monte_carlo.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>

#include "support/thread_agreement.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260718;

// --- European --------------------------------------------------------------

TEST(EngineThreadingSmokeTest, EuropeanIsRaceFreeAndAgreesAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig base;
    base.paths = 8000;
    base.steps = 1;
    base.seed = kSeed;

    MonteCarloConfig one = base;
    one.threads = 1;
    const auto ref = MonteCarloEngine::price(market, option, model, one);
    ASSERT_TRUE(ref.ok()) << ref.error().describe();

    for (const int threads : {2, 4}) {
        MonteCarloConfig cfg = base;
        cfg.threads = threads;
        const auto many = MonteCarloEngine::price(market, option, model, cfg);
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        test::expect_mean_agrees(
            many.value().value, ref.value().value, "european threads=" + std::to_string(threads));
    }
}

// --- Arithmetic Asian, with the control-variate pilot ------------------------

TEST(EngineThreadingSmokeTest, AsianControlVariateIsRaceFreeAndAgreesAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig base;
    base.paths = 8000;
    base.steps = 12;
    base.seed = kSeed;
    base.variance_reduction.control_variate = true;

    MonteCarloConfig one = base;
    one.threads = 1;
    const auto ref = MonteCarloEngine::price(market, option, model, one);
    ASSERT_TRUE(ref.ok()) << ref.error().describe();

    for (const int threads : {2, 4}) {
        MonteCarloConfig cfg = base;
        cfg.threads = threads;
        const auto many = MonteCarloEngine::price(market, option, model, cfg);
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        test::expect_mean_agrees(
            many.value().value, ref.value().value, "asian threads=" + std::to_string(threads));
    }
}

// --- Heston, including the data-dependent Feller warning ---------------------

// The stressed regime violates Feller, so every run carries the Feller warning whose
// text embeds the negative-variance fraction. That fraction is a thread-invariant
// diagnostic, so the warning string must be *identical* across thread counts -- the
// direct check that warnings, not only numeric diagnostics, are thread-invariant.
TEST(EngineThreadingSmokeTest, HestonIsRaceFreeAndWarningsAreIdenticalAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto stressed = HestonModel::create(0.04, 2.0, 0.04, 1.0, -0.7).value();

    HestonMonteCarloConfig base;
    base.paths = 4000;
    base.steps = 40;
    base.seed = kSeed;

    HestonMonteCarloConfig one = base;
    one.threads = 1;
    const auto ref = HestonMonteCarloEngine::price(market, option, stressed, one);
    ASSERT_TRUE(ref.ok()) << ref.error().describe();
    ASSERT_FALSE(ref.value().warnings.empty())
        << "the stressed regime must carry the Feller warning";

    for (const int threads : {2, 4}) {
        HestonMonteCarloConfig cfg = base;
        cfg.threads = threads;
        const auto many = HestonMonteCarloEngine::price(market, option, stressed, cfg);
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        test::expect_mean_agrees(
            many.value().value, ref.value().value, "heston threads=" + std::to_string(threads));
        EXPECT_EQ(many.value().warnings, ref.value().warnings)
            << "threads=" << threads << ": a warning string moved under partitioning";
    }
}

// --- Greeks, every estimator's own loop --------------------------------------

TEST(EngineThreadingSmokeTest, GreeksAreRaceFreeAndAgreeAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const std::tuple<GreekName, GreekMethod> cases[] = {
        {GreekName::Delta, GreekMethod::FiniteDifference},
        {GreekName::Vega, GreekMethod::Pathwise},
        {GreekName::Delta, GreekMethod::LikelihoodRatio},
    };

    for (const auto& [greek, method] : cases) {
        GreeksMonteCarloConfig base;
        base.paths = 8000;
        base.seed = kSeed;

        GreeksMonteCarloConfig one = base;
        one.threads = 1;
        const auto ref =
            GreeksMonteCarloEngine::estimate(market, option, model, greek, method, one);
        ASSERT_TRUE(ref.ok()) << ref.error().describe();

        for (const int threads : {2, 4}) {
            GreeksMonteCarloConfig cfg = base;
            cfg.threads = threads;
            const auto many =
                GreeksMonteCarloEngine::estimate(market, option, model, greek, method, cfg);
            ASSERT_TRUE(many.ok())
                << to_string(method) << " threads=" << threads << ": " << many.error().describe();
            test::expect_mean_agrees(
                many.value().value, ref.value().value, std::string(to_string(method)) + " value");
        }
    }
}

// --- Barrier, the early-knockout loop and the bridge stream ------------------

TEST(EngineThreadingSmokeTest, BarrierBridgeIsRaceFreeAndAgreesAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = BarrierOption::create(OptionType::Call,
                                              BarrierType::DownAndOut,
                                              100.0,
                                              90.0,
                                              1.0,
                                              MonitoringConvention::BrownianBridge,
                                              25)
                            .value();
    const auto model = BlackScholesModel::create(0.2).value();

    BarrierMonteCarloConfig base;
    base.paths = 8000;
    base.seed = kSeed;

    BarrierMonteCarloConfig one = base;
    one.threads = 1;
    const auto ref = BarrierMonteCarloEngine::price(market, option, model, one);
    ASSERT_TRUE(ref.ok()) << ref.error().describe();

    for (const int threads : {2, 4}) {
        BarrierMonteCarloConfig cfg = base;
        cfg.threads = threads;
        const auto many = BarrierMonteCarloEngine::price(market, option, model, cfg);
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        test::expect_mean_agrees(
            many.value().value, ref.value().value, "barrier threads=" + std::to_string(threads));
    }
}

// A single oversubscription smoke, exercising the empty-block-free clamp under TSan.
TEST(EngineThreadingSmokeTest, OversubscriptionIsRaceFree) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig cfg;
    cfg.paths = 32;
    cfg.steps = 1;
    cfg.seed = kSeed;
    cfg.threads = 256;  // 8x more workers than paths
    const auto priced = MonteCarloEngine::price(market, option, model, cfg);
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
}

}  // namespace
}  // namespace diffusionworks
