#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/greeks_monte_carlo.hpp>

#include "support/thread_agreement.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace diffusionworks {
namespace {

MarketState market(double spot = 100.0) {
    return MarketState::create(spot, 0.05, 0.0).value();
}

BlackScholesModel model(double volatility = 0.2) {
    return BlackScholesModel::create(volatility).value();
}

EuropeanOption call(double strike = 100.0, double maturity = 1.0) {
    return EuropeanOption::create(OptionType::Call, strike, maturity).value();
}

GreeksMonteCarloConfig config(std::int64_t paths = 200000, std::uint64_t seed = 20260717) {
    GreeksMonteCarloConfig c;
    c.paths = paths;
    c.seed = seed;
    return c;
}

Greeks analytic(const MarketState& mk, const EuropeanOption& option, const BlackScholesModel& md) {
    return BlackScholesAnalyticEngine::greeks(mk, option, md).value();
}

/// Asserts an estimate is consistent with the analytic value *given its own reported
/// uncertainty*. This is the right criterion for a Monte Carlo Greek: a fixed
/// absolute tolerance would either reject a correct low-variance estimator or accept
/// a wrong high-variance one, while "within k standard errors" scales with the
/// precision the estimator actually achieved. k = 4 makes a false failure a
/// ~1-in-16000 event per assertion.
void expect_consistent(const GreekEstimate& estimate, double reference) {
    ASSERT_GT(estimate.standard_error, 0.0);
    EXPECT_LT(std::abs(estimate.value - reference), 4.0 * estimate.standard_error)
        << to_string(estimate.method) << " " << to_string(estimate.greek) << ": " << estimate.value
        << " vs analytic " << reference << " (se " << estimate.standard_error << ")";
}

// ---------------------------------------------------------------------------
// Every estimator agrees with the analytic Greek
//
// The analytic sensitivities are the reference (validated against Hull, Haug,
// mpmath, and QuantLib), so each Monte Carlo estimator's job is to reproduce them
// to within its sampling error. Checked against the *reported* standard error, so
// the test measures whether the estimator is unbiased rather than whether it hit an
// arbitrary tolerance.
// ---------------------------------------------------------------------------

TEST(GreeksMonteCarloTest, FiniteDifferenceMatchesAnalyticDeltaGammaVega) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();
    const Greeks g = analytic(mk, option, md);

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, config());
    const auto gamma = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Gamma, GreekMethod::FiniteDifference, config());
    const auto vega = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Vega, GreekMethod::FiniteDifference, config());
    ASSERT_TRUE(delta.ok()) << delta.error().describe();
    ASSERT_TRUE(gamma.ok()) << gamma.error().describe();
    ASSERT_TRUE(vega.ok()) << vega.error().describe();

    expect_consistent(delta.value(), *g.delta);
    expect_consistent(gamma.value(), *g.gamma);
    expect_consistent(vega.value(), *g.vega);
}

TEST(GreeksMonteCarloTest, PathwiseMatchesAnalyticDeltaAndVega) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();
    const Greeks g = analytic(mk, option, md);

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::Pathwise, config());
    const auto vega = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Vega, GreekMethod::Pathwise, config());
    ASSERT_TRUE(delta.ok()) << delta.error().describe();
    ASSERT_TRUE(vega.ok()) << vega.error().describe();

    expect_consistent(delta.value(), *g.delta);
    expect_consistent(vega.value(), *g.vega);
}

TEST(GreeksMonteCarloTest, LikelihoodRatioMatchesAnalyticDelta) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();
    const Greeks g = analytic(mk, option, md);

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::LikelihoodRatio, config());
    ASSERT_TRUE(delta.ok()) << delta.error().describe();
    expect_consistent(delta.value(), *g.delta);
}

// Across moneyness and maturity: an estimator that is right only at the money is not
// validated. The pathwise delta is checked over a grid, against the analytic value
// at each point.
TEST(GreeksMonteCarloTest, PathwiseDeltaMatchesAcrossMoneynessAndMaturity) {
    const auto md = model();
    for (const double spot : {80.0, 95.0, 100.0, 110.0, 130.0}) {
        for (const double maturity : {0.25, 1.0, 3.0}) {
            const auto mk = market(spot);
            const auto option = call(100.0, maturity);
            const double reference = *analytic(mk, option, md).delta;

            const auto delta = GreeksMonteCarloEngine::estimate(
                mk, option, md, GreekName::Delta, GreekMethod::Pathwise, config());
            ASSERT_TRUE(delta.ok()) << "S=" << spot << " T=" << maturity;
            EXPECT_LT(std::abs(delta.value().value - reference), 4.0 * delta.value().standard_error)
                << "S=" << spot << " T=" << maturity;
        }
    }
}

// A put has negative delta. Exercises the put branch of the pathwise slope, which a
// call-only test would leave unchecked.
TEST(GreeksMonteCarloTest, PathwiseDeltaHandlesAPut) {
    const auto mk = market();
    const auto md = model();
    const auto put = EuropeanOption::create(OptionType::Put, 100.0, 1.0).value();
    const double reference = *analytic(mk, put, md).delta;
    ASSERT_LT(reference, 0.0);

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, put, md, GreekName::Delta, GreekMethod::Pathwise, config());
    ASSERT_TRUE(delta.ok()) << delta.error().describe();
    expect_consistent(delta.value(), reference);
}

// ---------------------------------------------------------------------------
// The estimators trade off differently -- no one is best
//
// For a smooth payoff's delta, the pathwise estimator has far lower variance than
// the likelihood-ratio one, because LR multiplies the payoff by a score that grows
// without bound while pathwise differentiates the bounded payoff slope. This is the
// concrete form of "no estimator is universally superior": LR loses here, but it is
// the only one of the three that survives a discontinuous payoff.
// ---------------------------------------------------------------------------

TEST(GreeksMonteCarloTest, LikelihoodRatioDeltaIsNoisierThanPathwise) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    const auto pathwise = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::Pathwise, config());
    const auto likelihood = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::LikelihoodRatio, config());
    ASSERT_TRUE(pathwise.ok());
    ASSERT_TRUE(likelihood.ok());

    // Both are unbiased for delta, so this is purely about variance. The margin is
    // large (about 2.5x in standard error) and stable, so a modest factor is a safe
    // structural assertion rather than a coincidence of one seed.
    EXPECT_GT(likelihood.value().standard_error, 1.5 * pathwise.value().standard_error)
        << "likelihood-ratio delta was not materially noisier than pathwise";
}

// ---------------------------------------------------------------------------
// Common random numbers
// ---------------------------------------------------------------------------

// The whole point of CRN bumping: the finite-difference delta must have a standard
// error far smaller than the option price's own sampling error, because the shared
// draws cancel in the difference. Without CRN the difference of two independent
// estimates would have variance O(1/h^2) -- here, thousands of times larger. A
// loose ceiling catches the catastrophic regression (independent draws) without
// pinning the exact variance.
TEST(GreeksMonteCarloTest, CommonRandomNumbersKeepFiniteDifferenceVarianceLow) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, config());
    ASSERT_TRUE(delta.ok());
    // Delta is ~0.64; with CRN its standard error is ~1e-3 at 200k paths. Independent
    // bumping at a 1% spot bump would put it in the tens. Anything below 0.05 proves
    // the draws are shared.
    EXPECT_LT(delta.value().standard_error, 0.05)
        << "finite-difference delta variance is far too high; common random numbers are not being "
           "shared across the bump";
}

TEST(GreeksMonteCarloTest, IsReproducibleUnderTheSameSeed) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    const auto first = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, config(50000, 7));
    const auto second = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, config(50000, 7));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(first.value().standard_error, second.value().standard_error);
}

// ---------------------------------------------------------------------------
// The bump-size trade-off, demonstrated
//
// Finite-difference gamma is the sharpest illustration: the 1/h^2 denominator
// amplifies the noise in the second difference, so a smaller bump is noisier, while
// a larger bump biases the estimate. The *rate* at which the variance grows with a
// shrinking bump is not a universal power law -- it depends on the payoff regularity
// and how strongly the shared draws couple the three re-prices -- so this test
// asserts only the direction, which is theoretically robust, and leaves the
// empirical scaling to EXP-08, where it is measured and fitted rather than assumed.
// ---------------------------------------------------------------------------

TEST(GreeksMonteCarloTest, SmallerBumpsMakeFiniteDifferenceGammaNoisier) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    double previous_error = 0.0;
    for (const double fraction : {0.05, 0.02, 0.01, 0.005}) {
        GreeksMonteCarloConfig c = config();
        c.spot_bump_fraction = fraction;
        const auto gamma = GreeksMonteCarloEngine::estimate(
            mk, option, md, GreekName::Gamma, GreekMethod::FiniteDifference, c);
        ASSERT_TRUE(gamma.ok()) << "fraction=" << fraction;
        EXPECT_GT(gamma.value().standard_error, previous_error)
            << "shrinking the bump from a larger value did not raise the gamma variance at "
            << fraction;
        previous_error = gamma.value().standard_error;
    }
}

// ---------------------------------------------------------------------------
// Rejected combinations
// ---------------------------------------------------------------------------

TEST(GreeksMonteCarloTest, RejectsPathwiseGamma) {
    const auto priced = GreeksMonteCarloEngine::estimate(
        market(), call(), model(), GreekName::Gamma, GreekMethod::Pathwise, config());
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

TEST(GreeksMonteCarloTest, RejectsUnimplementedLikelihoodRatioGreeks) {
    for (const auto greek : {GreekName::Gamma, GreekName::Vega}) {
        const auto priced = GreeksMonteCarloEngine::estimate(
            market(), call(), model(), greek, GreekMethod::LikelihoodRatio, config());
        ASSERT_FALSE(priced.ok()) << to_string(greek);
        EXPECT_EQ(priced.error().code, ErrorCode::NotImplemented) << to_string(greek);
    }
}

TEST(GreeksMonteCarloTest, OnePathIsRejected) {
    const auto one_path = GreeksMonteCarloEngine::estimate(
        market(), call(), model(), GreekName::Delta, GreekMethod::Pathwise, config(1));
    ASSERT_FALSE(one_path.ok());
    EXPECT_EQ(one_path.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Degenerate inputs are handled per method, not per request
//
// The correction: a zero volatility or maturity is fatal to the likelihood-ratio
// and pathwise formulas but only degenerate for finite difference, which stays a
// meaningful deterministic difference of the price. A request for one method must
// not be rejected because a *different* method would fail on the same inputs.
// ---------------------------------------------------------------------------

TEST(GreeksMonteCarloTest, LikelihoodRatioAndPathwiseAreRefusedAtDegenerateBoundaries) {
    // Bind by const reference: the elements are const std::tuple in an
    // initializer_list, and a by-value structured binding copies each -- which g++-13
    // flags under -Werror=range-loop-construct.
    for (const auto& [spot, maturity, volatility] :
         {std::tuple{100.0, 0.0, 0.2}, std::tuple{100.0, 1.0, 0.0}}) {
        const auto mk = market(spot);
        const auto md = model(volatility);
        const auto option = call(100.0, maturity);
        for (const auto method : {GreekMethod::LikelihoodRatio, GreekMethod::Pathwise}) {
            const auto priced = GreeksMonteCarloEngine::estimate(
                mk, option, md, GreekName::Delta, method, config());
            ASSERT_FALSE(priced.ok())
                << to_string(method) << " T=" << maturity << " s=" << volatility;
            EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination) << to_string(method);
        }
    }
}

TEST(GreeksMonteCarloTest, FiniteDifferenceSurvivesDegenerateBoundariesWithAWarning) {
    // Zero volatility: the price is deterministic, so finite-difference delta is the
    // exact analytic delta away from the kink, returned with a warning and a zero
    // standard error. That the likelihood-ratio method fails here (previous test)
    // must not deny the caller this.
    const auto mk = market(120.0);  // in the money, away from the strike kink
    const auto md = model(0.0);
    const auto option = call(100.0, 1.0);

    const auto delta = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, config());
    ASSERT_TRUE(delta.ok()) << delta.error().describe();
    // Deep in the money with no diffusion: the forward is above the strike with
    // certainty, so delta is exactly the discounted growth factor e^{-qT}=1 here.
    EXPECT_NEAR(delta.value().value, 1.0, 1e-9);
    EXPECT_EQ(delta.value().standard_error, 0.0);
    EXPECT_FALSE(delta.value().warnings.empty())
        << "a deterministic degenerate estimate must say so";
}

// A volatility bump that would carry the central difference into negative volatility
// is refused: negative sigma mirrors the terminal law, collapsing the estimate. A
// real defect, caught as a regression.
TEST(GreeksMonteCarloTest, RejectsAVegaBumpThatCrossesZeroVolatility) {
    const auto mk = market();
    const auto md = model(0.005);  // volatility below the default bump of 0.01
    const auto option = call();

    const auto vega = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Vega, GreekMethod::FiniteDifference, config());
    ASSERT_FALSE(vega.ok());
    EXPECT_EQ(vega.error().code, ErrorCode::InvalidArgument);
}

// Every successful estimate carries its uncertainty and its cost -- the exit gate's
// requirement, since the whole comparison is meaningless without them.
TEST(GreeksMonteCarloTest, EveryEstimateCarriesUncertaintyAndRuntime) {
    const auto estimate = GreeksMonteCarloEngine::estimate(
        market(), call(), model(), GreekName::Delta, GreekMethod::Pathwise, config());
    ASSERT_TRUE(estimate.ok());
    EXPECT_GT(estimate.value().standard_error, 0.0);
    EXPECT_GE(estimate.value().runtime_seconds, 0.0);
    EXPECT_EQ(estimate.value().paths, 200000);
}

// ---------------------------------------------------------------------------
// Multithreading (Phase 12)
//
// The path loop runs across deterministic worker partitions with thread-local
// accumulators reduced in block order (ADR-011). One thread is the sequential
// reference; a fixed thread count is reproducible; different counts agree up to the
// reassociation of the merge. Crucially, each path's whole common-random-number
// contribution -- for the finite-difference estimators the up and down (and, for
// gamma, mid) re-prices against the path's single shared draw -- is computed inside
// one `contribution` call, so it stays within one worker and one accumulator. Were
// the paired re-prices split across threads they would decorrelate and the estimate
// would move far more than reassociation, which the 1e-9 cross-count agreement rules
// out.
// ---------------------------------------------------------------------------

GreeksMonteCarloConfig
threaded(int threads, std::int64_t paths = 120000, std::uint64_t seed = 20260717) {
    GreeksMonteCarloConfig c = config(paths, seed);
    c.threads = threads;
    return c;
}

TEST(GreeksMonteCarloThreadingTest, EveryEstimatorAgreesAcrossThreadCounts) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    const std::tuple<GreekName, GreekMethod> cases[] = {
        {GreekName::Delta, GreekMethod::FiniteDifference},
        {GreekName::Gamma, GreekMethod::FiniteDifference},
        {GreekName::Vega, GreekMethod::FiniteDifference},
        {GreekName::Delta, GreekMethod::Pathwise},
        {GreekName::Vega, GreekMethod::Pathwise},
        {GreekName::Delta, GreekMethod::LikelihoodRatio},
    };

    for (const auto& [greek, method] : cases) {
        const auto one =
            GreeksMonteCarloEngine::estimate(mk, option, md, greek, method, threaded(1));
        ASSERT_TRUE(one.ok()) << one.error().describe();

        for (const int threads : {2, 4, 8}) {
            const auto many =
                GreeksMonteCarloEngine::estimate(mk, option, md, greek, method, threaded(threads));
            ASSERT_TRUE(many.ok()) << to_string(method) << " " << to_string(greek)
                                   << " threads=" << threads << ": " << many.error().describe();
            const std::string tag = std::string(to_string(method)) + " " +
                                    std::string(to_string(greek)) +
                                    " threads=" + std::to_string(threads);
            // Scale-aware: a Greek's magnitude ranges from ~0.02 (gamma) to ~40 (vega),
            // so a relative tolerance is the only one that means the same thing for each.
            test::expect_mean_agrees(many.value().value, one.value().value, tag + " value");
            test::expect_error_agrees(
                many.value().standard_error, one.value().standard_error, tag + " standard error");
        }
    }
}

// More workers than paths must clamp rather than crash: the CRN estimator for each path
// is a self-contained contribution, so oversubscription only changes the partition, and
// the result agrees with the single-thread run to the scale-aware tolerance.
TEST(GreeksMonteCarloThreadingTest, HandlesMoreThreadsThanPaths) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();

    const auto one = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, threaded(1, 64));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    const auto many = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, threaded(1024, 64));
    ASSERT_TRUE(many.ok()) << "threads far exceeding paths must clamp, not fail: "
                           << many.error().describe();
    test::expect_mean_agrees(many.value().value, one.value().value, "value with threads > paths");
    test::expect_error_agrees(many.value().standard_error,
                              one.value().standard_error,
                              "standard error with threads > paths");
}

TEST(GreeksMonteCarloThreadingTest, IsBitReproducibleAtAFixedThreadCount) {
    const auto mk = market();
    const auto md = model();
    const auto option = call();
    const auto first = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, threaded(4));
    const auto second = GreeksMonteCarloEngine::estimate(
        mk, option, md, GreekName::Delta, GreekMethod::FiniteDifference, threaded(4));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(first.value().standard_error, second.value().standard_error);
}

TEST(GreeksMonteCarloThreadingTest, RejectsAnInvalidThreadCount) {
    const auto estimate = GreeksMonteCarloEngine::estimate(
        market(), call(), model(), GreekName::Delta, GreekMethod::Pathwise, threaded(0));
    ASSERT_FALSE(estimate.ok());
    EXPECT_EQ(estimate.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks
