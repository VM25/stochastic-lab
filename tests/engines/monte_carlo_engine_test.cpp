#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>

#include "support/thread_agreement.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <variant>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

MonteCarloConfig config_with(std::int64_t paths,
                             std::int64_t steps,
                             std::uint64_t seed,
                             DiscretizationScheme scheme = DiscretizationScheme::Exact) {
    MonteCarloConfig config;
    config.paths = paths;
    config.steps = steps;
    config.seed = seed;
    config.scheme = scheme;
    return config;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

TEST(MonteCarloConfigTest, RejectsConfigurationsThatCannotReportUncertainty) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    // One path has no dispersion, so it cannot report a standard error. A price
    // without one is not a Monte Carlo result.
    EXPECT_FALSE(MonteCarloEngine::price(market, option, model, config_with(1, 1, kSeed)).ok());
    EXPECT_FALSE(MonteCarloEngine::price(market, option, model, config_with(0, 1, kSeed)).ok());
    EXPECT_FALSE(MonteCarloEngine::price(market, option, model, config_with(100, 0, kSeed)).ok());

    MonteCarloConfig bad_level = config_with(100, 1, kSeed);
    bad_level.confidence_level = 1.5;
    EXPECT_FALSE(MonteCarloEngine::price(market, option, model, bad_level).ok());
}

// ---------------------------------------------------------------------------
// Convergence to the analytic engine
//
// The Phase 3 exit gate. Validated against the Phase 1 engine, which is itself
// validated against Hull, Haug, a 50-digit mpmath oracle, and QuantLib -- so this
// is a chain to published values rather than to another piece of this project.
// ---------------------------------------------------------------------------

struct Scenario {
    const char* name;
    OptionType type;
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
};

std::vector<Scenario> scenarios() {
    return {
        {"at_the_money_call", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"in_the_money_call", OptionType::Call, 120.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"out_of_the_money_call", OptionType::Call, 80.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"at_the_money_put", OptionType::Put, 100.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"with_dividend", OptionType::Call, 100.0, 100.0, 0.05, 0.03, 0.20, 1.0},
        {"high_volatility", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.60, 1.0},
    };
}

// Every scenario's analytic value must lie inside the Monte Carlo interval. The
// test is stated in terms of the estimator's own reported uncertainty rather than
// an absolute tolerance: an absolute bound would be arbitrary, and would pass or
// fail for reasons unrelated to whether the estimator is correct.
TEST(MonteCarloEngineTest, ExactSchemeBracketsTheAnalyticValue) {
    for (const auto& s : scenarios()) {
        const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield).value();
        const auto option = EuropeanOption::create(s.type, s.strike, s.maturity).value();
        const auto model = BlackScholesModel::create(s.volatility).value();

        const auto analytic = BlackScholesAnalyticEngine::price(market, option, model);
        ASSERT_TRUE(analytic.ok()) << analytic.error().describe();

        const auto simulated =
            MonteCarloEngine::price(market, option, model, config_with(200000, 1, kSeed));
        ASSERT_TRUE(simulated.ok()) << simulated.error().describe();

        ASSERT_TRUE(simulated.value().confidence_interval.has_value());
        const ConfidenceInterval& interval = *simulated.value().confidence_interval;

        EXPECT_TRUE(interval.contains(analytic.value().value))
            << s.name << ": the 95% interval [" << interval.lower << ", " << interval.upper
            << "] excludes the analytic value " << analytic.value().value << " (estimate "
            << simulated.value().value << ")";
    }
}

// Sampling error must fall as N^{-1/2}. Checked over a 64-fold increase in paths
// so the expected 8-fold reduction is far larger than the noise in the estimate.
TEST(MonteCarloEngineTest, StandardErrorFallsAsInverseRootPaths) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto few = MonteCarloEngine::price(market, option, model, config_with(5000, 1, kSeed));
    const auto many = MonteCarloEngine::price(market, option, model, config_with(320000, 1, kSeed));
    ASSERT_TRUE(few.ok());
    ASSERT_TRUE(many.ok());

    const double ratio = *few.value().standard_error / *many.value().standard_error;

    // 64x the paths should give 8x the precision. The bound is generous because
    // the sample standard deviation is itself estimated, but a scheme converging
    // at any other rate would miss it by far more.
    EXPECT_NEAR(ratio, 8.0, 1.0) << "standard error scaled by " << ratio << " for 64x the paths";
}

// The estimate must move toward the analytic value as paths increase, not merely
// report a smaller interval around a wrong number.
TEST(MonteCarloEngineTest, ErrorShrinksWithPaths) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const double reference = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    // Averaged over independent seeds: a single run's error is one draw from a
    // distribution and can shrink or grow by luck, so comparing two single runs
    // would be a coin flip dressed as evidence (FAILURE-MODES section 8).
    const auto mean_absolute_error = [&](std::int64_t paths) {
        OnlineMoments errors;
        for (std::uint64_t seed = 0; seed < 24; ++seed) {
            const auto priced =
                MonteCarloEngine::price(market, option, model, config_with(paths, 1, 5000 + seed));
            EXPECT_TRUE(priced.ok()) << priced.error().describe();
            errors.add(std::abs(priced.value().value - reference));
        }
        return errors.mean();
    };

    const double coarse = mean_absolute_error(2000);
    const double fine = mean_absolute_error(128000);

    EXPECT_LT(fine, coarse * 0.35)
        << "64x the paths should cut the mean absolute error by ~8x: " << coarse << " -> " << fine;
}

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

TEST(MonteCarloEngineTest, SameSeedGivesTheSameEstimate) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto first = MonteCarloEngine::price(market, option, model, config_with(20000, 4, kSeed));
    const auto second =
        MonteCarloEngine::price(market, option, model, config_with(20000, 4, kSeed));

    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value) << "a fixed seed must reproduce exactly";
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
}

TEST(MonteCarloEngineTest, DifferentSeedsGiveDifferentEstimates) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto first = MonteCarloEngine::price(market, option, model, config_with(20000, 1, 1));
    const auto second = MonteCarloEngine::price(market, option, model, config_with(20000, 1, 2));

    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_NE(first.value().value, second.value().value);
}

// The exact transition is exact over any horizon, so for a European payoff the
// step count changes the cost and nothing else. Same seed, same paths, different
// steps: the shocks differ, so the estimates differ -- but both must bracket the
// same analytic value, and neither is "more converged" than the other.
TEST(MonteCarloEngineTest, StepCountDoesNotBiasTheExactScheme) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const double reference = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    for (const std::int64_t steps : {1, 4, 32}) {
        const auto priced =
            MonteCarloEngine::price(market, option, model, config_with(200000, steps, kSeed));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        EXPECT_TRUE(priced.value().confidence_interval->contains(reference))
            << "steps = " << steps << " excluded the analytic value";
    }
}

// ---------------------------------------------------------------------------
// Result completeness
// ---------------------------------------------------------------------------

TEST(MonteCarloEngineTest, ReportsUncertaintyAndProvenance) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced =
        MonteCarloEngine::price(market, option, model, config_with(10000, 2, kSeed));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    const PricingResult& result = priced.value();

    // A Monte Carlo point estimate without a standard error is not a price.
    ASSERT_TRUE(result.standard_error.has_value());
    EXPECT_GT(*result.standard_error, 0.0);
    ASSERT_TRUE(result.confidence_interval.has_value());
    EXPECT_GT(result.confidence_interval->width(), 0.0);
    EXPECT_TRUE(result.confidence_interval->contains(result.value));

    EXPECT_EQ(result.method, "monte_carlo_exact");

    // The seed and path count are what make the number reproducible, so they
    // travel with it.
    const auto has_diagnostic = [&](const std::string& name) {
        for (const Diagnostic& d : result.diagnostics) {
            if (d.name == name) {
                return true;
            }
        }
        return false;
    };
    for (const std::string name : {"paths", "steps", "seed", "scheme", "non_positive_states"}) {
        EXPECT_TRUE(has_diagnostic(name)) << "missing diagnostic: " << name;
    }
}

// The exact scheme cannot cross zero, so a clean run must not warn.
TEST(MonteCarloEngineTest, ExactSchemeRunsWithoutWarnings) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced =
        MonteCarloEngine::price(market, option, model, config_with(50000, 8, kSeed));
    ASSERT_TRUE(priced.ok());
    EXPECT_FALSE(priced.value().has_warnings());
}

// Euler can cross zero, and the payoff then clamps to a perfectly ordinary
// number. The warning is the only visible trace of the resulting bias, so its
// absence would be the defect.
TEST(MonteCarloEngineTest, EulerWarnsWhenPathsCrossZero) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(1.5).value();

    const auto priced = MonteCarloEngine::price(
        market, option, model, config_with(20000, 4, kSeed, DiscretizationScheme::EulerMaruyama));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();

    EXPECT_TRUE(priced.value().has_warnings())
        << "Euler crossed zero under these parameters without saying so";

    bool mentions_crossing = false;
    for (const std::string& warning : priced.value().warnings) {
        if (warning.find("negative") != std::string::npos) {
            mentions_crossing = true;
        }
    }
    EXPECT_TRUE(mentions_crossing);
}

// ---------------------------------------------------------------------------
// Confidence coverage
//
// VALIDATION-PLAN section 7. A plausible interval is not enough: the interval
// must actually contain the truth at its nominal rate. Systematic undercoverage
// would make every published uncertainty a fiction, and no single run can reveal
// it.
// ---------------------------------------------------------------------------

TEST(MonteCarloCoverageTest, NinetyFivePercentIntervalsCoverAtTheirNominalRate) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const double reference = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    constexpr int kTrials = 400;
    constexpr std::int64_t kPathsPerTrial = 4000;

    int covered = 0;
    for (std::uint64_t trial = 0; trial < kTrials; ++trial) {
        const auto priced = MonteCarloEngine::price(
            market, option, model, config_with(kPathsPerTrial, 1, 900000 + trial));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        if (priced.value().confidence_interval->contains(reference)) {
            ++covered;
        }
    }

    const double rate = static_cast<double>(covered) / kTrials;

    // Coverage is itself a binomial estimate: with 400 trials at p = 0.95 its
    // standard error is sqrt(0.95*0.05/400) = 1.09%. The bound is ~3.7 standard
    // errors, wide enough not to be flaky and narrow enough to catch the
    // undercoverage a skewed payoff would cause.
    EXPECT_NEAR(rate, 0.95, 0.04) << covered << " of " << kTrials
                                  << " intervals covered the analytic value (" << rate * 100
                                  << "%)";
}

// A deep out-of-the-money call is the hard case: most paths pay nothing and a few
// pay a lot, so the payoff distribution is severely skewed and the central limit
// theorem bites late. This is exactly where a normal-approximation interval would
// under-cover, so it is tested rather than assumed away.
TEST(MonteCarloCoverageTest, SkewedPayoffsStillCoverNearTheirNominalRate) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 160.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const double reference = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    constexpr int kTrials = 400;
    int covered = 0;
    for (std::uint64_t trial = 0; trial < kTrials; ++trial) {
        const auto priced =
            MonteCarloEngine::price(market, option, model, config_with(20000, 1, 700000 + trial));
        ASSERT_TRUE(priced.ok());
        if (priced.value().confidence_interval->contains(reference)) {
            ++covered;
        }
    }

    const double rate = static_cast<double>(covered) / kTrials;

    // Deliberately looser than the at-the-money case, and asymmetric in intent:
    // some undercoverage is expected here and is a property of the payoff rather
    // than a defect. What would be a defect is gross undercoverage, which this
    // catches. EXP-15 quantifies the shortfall properly.
    EXPECT_GT(rate, 0.90) << covered << " of " << kTrials << " covered (" << rate * 100
                          << "%): a skewed payoff undercovers, but not this far";
    EXPECT_LT(rate, 0.99) << "coverage far above nominal suggests the interval is too wide";
}

// ---------------------------------------------------------------------------
// Asian options
// ---------------------------------------------------------------------------

TEST(AsianOptionTest, ValidatesItsTerms) {
    EXPECT_TRUE(
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).ok());
    EXPECT_FALSE(
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, -1.0, 1.0, 12).ok());
    EXPECT_FALSE(
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 0.0, 12).ok());
    EXPECT_FALSE(
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 0).ok());
}

// The monitoring convention: M equally spaced dates ending at maturity, with the
// initial spot excluded. Leaving this implicit is itself a failure
// (VALIDATION-PLAN section 11), so it is pinned.
TEST(AsianOptionTest, MonitoringDatesEndAtMaturityAndExcludeInception) {
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 4).value();

    EXPECT_NEAR(option.monitoring_time(1), 0.25, 1e-15);
    EXPECT_NEAR(option.monitoring_time(2), 0.50, 1e-15);
    EXPECT_NEAR(option.monitoring_time(3), 0.75, 1e-15);
    EXPECT_DOUBLE_EQ(option.monitoring_time(4), 1.0) << "the last date must be maturity exactly";
}

TEST(AsianOptionTest, PayoffMatchesTheContract) {
    const auto call =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 4).value();
    EXPECT_DOUBLE_EQ(call.payoff(120.0), 20.0);
    EXPECT_DOUBLE_EQ(call.payoff(80.0), 0.0);

    const auto put =
        AsianOption::create(OptionType::Put, AveragingType::Arithmetic, 100.0, 1.0, 4).value();
    EXPECT_DOUBLE_EQ(put.payoff(80.0), 20.0);
    EXPECT_DOUBLE_EQ(put.payoff(120.0), 0.0);
}

// The grid must resolve the monitoring dates. Monitoring at the nearest grid
// point instead would price a different contract, quietly.
TEST(MonteCarloAsianTest, RejectsAGridThatDoesNotResolveTheMonitoringDates) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced = MonteCarloEngine::price(market, option, model, config_with(1000, 5, kSeed));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

TEST(MonteCarloAsianTest, AcceptsAGridThatIsAMultipleOfTheMonitoringCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 4).value();
    const auto model = BlackScholesModel::create(0.2).value();

    for (const std::int64_t steps : {4, 8, 12}) {
        const auto priced =
            MonteCarloEngine::price(market, option, model, config_with(5000, steps, kSeed));
        EXPECT_TRUE(priced.ok()) << "steps = " << steps << ": " << priced.error().describe();
    }
}

// An average is less volatile than the terminal value it averages, so an
// arithmetic Asian call is worth strictly less than the European call with the
// same strike. This is a model-free ordering, and it is the sharpest cheap check
// available before the geometric control variate arrives in Phase 4.
TEST(MonteCarloAsianTest, ArithmeticAsianIsCheaperThanTheEuropean) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();

    const auto asian =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto european = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    const auto asian_price =
        MonteCarloEngine::price(market, asian, model, config_with(200000, 12, kSeed));
    ASSERT_TRUE(asian_price.ok()) << asian_price.error().describe();

    const double european_price =
        BlackScholesAnalyticEngine::price(market, european, model).value().value;

    EXPECT_LT(asian_price.value().value, european_price)
        << "the average is less volatile than the terminal value, so the Asian must be cheaper";
    EXPECT_GT(asian_price.value().value, 0.0);
}

// The geometric mean never exceeds the arithmetic mean, so with the same strike
// the geometric Asian call is worth no more. Compared on the same seed, so the
// two estimates share their paths and the ordering is not a sampling accident.
TEST(MonteCarloAsianTest, GeometricAsianIsNoDearerThanArithmetic) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();

    const auto arithmetic =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto geometric =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 12).value();

    const auto arithmetic_price =
        MonteCarloEngine::price(market, arithmetic, model, config_with(100000, 12, kSeed));
    const auto geometric_price =
        MonteCarloEngine::price(market, geometric, model, config_with(100000, 12, kSeed));

    ASSERT_TRUE(arithmetic_price.ok());
    ASSERT_TRUE(geometric_price.ok());
    EXPECT_LE(geometric_price.value().value, arithmetic_price.value().value)
        << "AM-GM requires the geometric average to be no larger";
}

TEST(MonteCarloAsianTest, ReportsItsMonitoringConvention) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 6).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto priced = MonteCarloEngine::price(market, option, model, config_with(2000, 6, kSeed));
    ASSERT_TRUE(priced.ok());

    bool has_monitoring = false;
    bool has_averaging = false;
    for (const Diagnostic& d : priced.value().diagnostics) {
        has_monitoring = has_monitoring || d.name == "monitoring_count";
        has_averaging = has_averaging || d.name == "averaging";
    }
    EXPECT_TRUE(has_monitoring);
    EXPECT_TRUE(has_averaging) << "an Asian price without its averaging convention is ambiguous";
}

// A non-positive state has no logarithm, so a geometric average cannot be formed
// from it. Reported rather than skipped: dropping the path would silently bias
// the estimator toward the paths that happened to survive.
TEST(MonteCarloAsianTest, GeometricAveragingFailsExplicitlyWhenAPathCrossesZero) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 4).value();
    const auto model = BlackScholesModel::create(1.5).value();

    const auto priced = MonteCarloEngine::price(
        market, option, model, config_with(20000, 4, kSeed, DiscretizationScheme::EulerMaruyama));

    ASSERT_FALSE(priced.ok()) << "a path crossed zero; its geometric average has no logarithm";
    EXPECT_EQ(priced.error().code, ErrorCode::PathFailure);
}

// ---------------------------------------------------------------------------
// Multi-seed aggregation
// ---------------------------------------------------------------------------

// The estimator's realised dispersion across seeds should agree with the standard
// error a single run reports. That agreement is the entire justification for
// reporting a standard error, so it is measured rather than assumed
// (FAILURE-MODES section 8).
TEST(MonteCarloEngineTest, ReportedStandardErrorMatchesTheRealisedDispersion) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const double reference = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    constexpr int kSeeds = 40;
    std::vector<SeedResult> results;
    double reported_standard_error = 0.0;

    for (std::uint64_t seed = 0; seed < kSeeds; ++seed) {
        const auto priced =
            MonteCarloEngine::price(market, option, model, config_with(20000, 1, 300000 + seed));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        results.push_back(SeedResult{300000 + seed, priced.value().value});
        reported_standard_error += *priced.value().standard_error;
    }
    reported_standard_error /= kSeeds;

    const auto summary = summarize_seeds(results, reference);
    ASSERT_TRUE(summary.ok()) << summary.error().describe();

    // The realised standard deviation across seeds should match the average
    // self-reported standard error. Its own sampling error is ~1/sqrt(2(k-1)),
    // about 11% at 40 seeds, so the bound checks agreement in magnitude rather
    // than to three digits.
    EXPECT_NEAR(
        summary.value().standard_deviation, reported_standard_error, 0.4 * reported_standard_error)
        << "the estimator's realised dispersion disagrees with what it reports";

    // And it must be centred: the exact scheme has no discretisation bias, so any
    // systematic offset would be an error in the estimator itself.
    ASSERT_TRUE(summary.value().bias.has_value());
    EXPECT_LT(std::abs(*summary.value().bias), 3.0 * summary.value().standard_error)
        << "the exact scheme should be unbiased, but the mean error is " << *summary.value().bias;
}

// ---------------------------------------------------------------------------
// Multithreading (Phase 12)
//
// The path loop runs across deterministic worker partitions with thread-local
// accumulators reduced in block order (ADR-011). One thread is the sequential
// reference; a fixed thread count is reproducible; different counts agree up to the
// floating-point reassociation of the exact merge, and nothing races.
// ---------------------------------------------------------------------------

MonteCarloConfig threaded_config(int threads, bool antithetic = false) {
    MonteCarloConfig config = config_with(200000, 1, kSeed);
    config.threads = threads;
    config.variance_reduction.antithetic = antithetic;
    return config;
}

// One thread is the sequential engine; more threads change the answer only by the
// reassociation of the merge. The agreement is checked against a scale-aware tolerance
// (support/thread_agreement.hpp): relative to the price and the standard error, not a
// universal absolute constant, because reassociation scales with the magnitude of the
// reduced quantity.
TEST(MonteCarloThreadingTest, MatchesTheSingleThreadedResultAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto one = MonteCarloEngine::price(market, option, model, threaded_config(1));
    ASSERT_TRUE(one.ok()) << one.error().describe();

    for (const int threads : {2, 4, 8}) {
        const auto many = MonteCarloEngine::price(market, option, model, threaded_config(threads));
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();
        const std::string tag = "threads=" + std::to_string(threads);
        test::expect_mean_agrees(many.value().value, one.value().value, tag + " value");
        test::expect_error_agrees(
            *many.value().standard_error, *one.value().standard_error, tag + " standard error");
    }
}

// More workers than paths must be valid, not a crash: effective_worker_count clamps
// the worker count to the path count, so there are no empty blocks, and the result
// agrees with the single-thread run to the scale-aware tolerance. 1024 workers over 64
// paths is a 16x oversubscription.
TEST(MonteCarloThreadingTest, HandlesMoreThreadsThanPaths) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig few = config_with(64, 1, kSeed);
    few.threads = 1;
    const auto one = MonteCarloEngine::price(market, option, model, few);
    ASSERT_TRUE(one.ok()) << one.error().describe();

    MonteCarloConfig oversubscribed = config_with(64, 1, kSeed);
    oversubscribed.threads = 1024;
    const auto many = MonteCarloEngine::price(market, option, model, oversubscribed);
    ASSERT_TRUE(many.ok()) << "threads far exceeding paths must clamp, not fail: "
                           << many.error().describe();
    test::expect_mean_agrees(many.value().value, one.value().value, "value with threads > paths");
    test::expect_error_agrees(*many.value().standard_error,
                              *one.value().standard_error,
                              "standard error with threads > paths");
}

// A fixed thread count is bit-for-bit reproducible: the partition and the reduction
// order are fixed, so the operations are identical.
TEST(MonteCarloThreadingTest, IsBitReproducibleAtAFixedThreadCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto first = MonteCarloEngine::price(market, option, model, threaded_config(4));
    const auto second = MonteCarloEngine::price(market, option, model, threaded_config(4));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
}

// Antithetic pairing is thread-safe and equally deterministic: each pair is drawn
// inside one path index, so partitioning by index keeps pairs together.
TEST(MonteCarloThreadingTest, AntitheticIsConsistentAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto one = MonteCarloEngine::price(market, option, model, threaded_config(1, true));
    const auto many = MonteCarloEngine::price(market, option, model, threaded_config(8, true));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    ASSERT_TRUE(many.ok()) << many.error().describe();
    test::expect_mean_agrees(many.value().value, one.value().value, "antithetic value");
}

TEST(MonteCarloThreadingTest, RejectsAnInvalidThreadCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig zero = threaded_config(0);
    const auto priced = MonteCarloEngine::price(market, option, model, zero);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Multithreading: the arithmetic Asian control-variate engine (Phase 12)
//
// Only the main sample loop is parallelised. The control-variate pilot is a
// sequential pre-pass over path indices [paths, paths + pilot_paths) -- disjoint
// from the production indices [0, paths) drawn by the main loop -- so beta and the
// control expectation are constants at any thread count. Because the pilot never
// runs in parallel, its variance-reduction statistics must be *bit-identical*
// across thread counts, and the price agrees up to the reassociation of the merge.
// ---------------------------------------------------------------------------

MonteCarloConfig threaded_asian_config(int threads, bool antithetic, bool control) {
    MonteCarloConfig config = config_with(120000, 12, kSeed);
    config.threads = threads;
    config.variance_reduction.antithetic = antithetic;
    config.variance_reduction.control_variate = control;
    return config;
}

// Extracts a double-valued diagnostic by name, failing the test if it is absent or
// not a double -- used to compare the pilot's statistics across thread counts.
double double_diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& d : result.diagnostics) {
        if (d.name == name) {
            if (const double* v = std::get_if<double>(&d.value)) {
                return *v;
            }
        }
    }
    ADD_FAILURE() << "missing double diagnostic: " << name;
    return std::numeric_limits<double>::quiet_NaN();
}

TEST(MonteCarloAsianThreadingTest, ControlVariateAgreesAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto one =
        MonteCarloEngine::price(market, option, model, threaded_asian_config(1, false, true));
    ASSERT_TRUE(one.ok()) << one.error().describe();

    for (const int threads : {2, 4, 8}) {
        const auto many = MonteCarloEngine::price(
            market, option, model, threaded_asian_config(threads, false, true));
        ASSERT_TRUE(many.ok()) << "threads=" << threads << ": " << many.error().describe();

        // The same production paths, so the price and its standard error agree up to
        // the documented reassociation of the reduction, to a scale-aware tolerance.
        const std::string tag = "threads=" + std::to_string(threads);
        test::expect_mean_agrees(many.value().value, one.value().value, tag + " value");
        test::expect_error_agrees(
            *many.value().standard_error, *one.value().standard_error, tag + " standard error");

        // The pilot is sequential, so beta, the control expectation, and the pilot
        // correlation are the *same bits* regardless of how the main loop partitions.
        EXPECT_EQ(double_diagnostic(many.value(), "control_beta"),
                  double_diagnostic(one.value(), "control_beta"))
            << "threads=" << threads << ": the sequential pilot's beta must not move";
        EXPECT_EQ(double_diagnostic(many.value(), "control_expectation"),
                  double_diagnostic(one.value(), "control_expectation"))
            << "threads=" << threads;
        EXPECT_EQ(double_diagnostic(many.value(), "control_pilot_correlation"),
                  double_diagnostic(one.value(), "control_pilot_correlation"))
            << "threads=" << threads;
    }
}

TEST(MonteCarloAsianThreadingTest, ControlVariateIsBitReproducibleAtAFixedThreadCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto first =
        MonteCarloEngine::price(market, option, model, threaded_asian_config(4, false, true));
    const auto second =
        MonteCarloEngine::price(market, option, model, threaded_asian_config(4, false, true));
    ASSERT_TRUE(first.ok()) << first.error().describe();
    ASSERT_TRUE(second.ok()) << second.error().describe();
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
}

// The control correction and antithetic pairing together are still deterministic:
// the correction uses the constant pilot beta, and each pair is drawn inside one
// path index, so partitioning by index keeps both intact.
TEST(MonteCarloAsianThreadingTest, ControlAndAntitheticCombinedAgreeAcrossThreadCounts) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    const auto one =
        MonteCarloEngine::price(market, option, model, threaded_asian_config(1, true, true));
    const auto many =
        MonteCarloEngine::price(market, option, model, threaded_asian_config(8, true, true));
    ASSERT_TRUE(one.ok()) << one.error().describe();
    ASSERT_TRUE(many.ok()) << many.error().describe();
    test::expect_mean_agrees(many.value().value, one.value().value, "control+antithetic value");
    test::expect_error_agrees(*many.value().standard_error,
                              *one.value().standard_error,
                              "control+antithetic standard error");
}

// The non-positive-state diagnostic must survive partitioning: the pilot's
// excursions are folded into the reduced main-sample count, and workers sum their
// own counts, so the total is the same at any thread count. Euler at high vol
// crosses zero often enough to make the count non-trivial.
TEST(MonteCarloAsianThreadingTest, NonPositiveStateCountIsInvariantToThreadCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(1.2).value();

    const auto count_non_positive = [&](int threads) -> std::int64_t {
        MonteCarloConfig config =
            config_with(60000, 12, kSeed, DiscretizationScheme::EulerMaruyama);
        config.threads = threads;
        const auto priced = MonteCarloEngine::price(market, option, model, config);
        EXPECT_TRUE(priced.ok()) << priced.error().describe();
        for (const Diagnostic& d : priced.value().diagnostics) {
            if (d.name == "non_positive_states") {
                return std::get<std::int64_t>(d.value);
            }
        }
        ADD_FAILURE() << "missing non_positive_states diagnostic";
        return -1;
    };

    const std::int64_t reference = count_non_positive(1);
    EXPECT_GT(reference, 0) << "the regime should produce excursions for the check to bite";
    for (const int threads : {2, 4, 8}) {
        EXPECT_EQ(count_non_positive(threads), reference)
            << "threads=" << threads << ": the excursion count changed under partitioning";
    }
}

}  // namespace
}  // namespace diffusionworks
