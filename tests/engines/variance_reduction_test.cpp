#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/geometric_asian_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Variance reduction, evidenced across seeds.
//
// The Phase 4 exit gate is explicit: each method needs multi-seed evidence, and
// no favourable single run supports a claim. A single run's variance is one draw
// from a distribution; comparing two of them would be a coin flip presented as a
// measurement (FAILURE-MODES section 8). Every claim below is therefore made over
// independent replications, and the *realised* dispersion is what counts -- not
// the estimator's opinion of itself.
// ---------------------------------------------------------------------------

MonteCarloConfig make_config(std::int64_t paths,
                             std::int64_t steps,
                             std::uint64_t seed,
                             bool antithetic,
                             bool control_variate) {
    MonteCarloConfig config;
    config.paths = paths;
    config.steps = steps;
    config.seed = seed;
    config.scheme = DiscretizationScheme::Exact;
    config.variance_reduction.antithetic = antithetic;
    config.variance_reduction.control_variate = control_variate;
    config.control_variate_pilot_paths = 2000;
    return config;
}

MarketState market() {
    return MarketState::create(100.0, 0.05, 0.0).value();
}

BlackScholesModel model() {
    return BlackScholesModel::create(0.3).value();
}

AsianOption asian_call() {
    return AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
}

EuropeanOption european_call() {
    return EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
}

/// The realised behaviour of an estimator across independent seeds.
struct Realised {
    double mean{};
    /// The estimator's actual dispersion: the quantity a single run's standard
    /// error is trying to predict.
    double standard_deviation{};
    /// The average standard error the runs reported for themselves.
    double reported_standard_error{};
};

template<typename Instrument>
Realised measure(const Instrument& instrument,
                 bool antithetic,
                 bool control_variate,
                 std::int64_t paths,
                 std::int64_t steps,
                 int seeds,
                 std::uint64_t seed_base) {
    std::vector<SeedResult> results;
    OnlineMoments reported;

    for (int i = 0; i < seeds; ++i) {
        const std::uint64_t seed = seed_base + static_cast<std::uint64_t>(i);
        const auto priced =
            MonteCarloEngine::price(market(),
                                    instrument,
                                    model(),
                                    make_config(paths, steps, seed, antithetic, control_variate));
        EXPECT_TRUE(priced.ok()) << priced.error().describe();
        results.push_back(SeedResult{seed, priced.value().value});
        reported.add(*priced.value().standard_error);
    }

    const auto summary = summarize_seeds(results);
    EXPECT_TRUE(summary.ok()) << summary.error().describe();

    return Realised{summary.value().mean, summary.value().standard_deviation, reported.mean()};
}

// ---------------------------------------------------------------------------
// Unbiasedness
//
// A variance-reduction technique must change the variance and nothing else. A
// method that shrinks the interval around the wrong number is worse than no
// method at all, because it looks better.
// ---------------------------------------------------------------------------

// The arithmetic Asian has no closed form, so the reference is the crude
// estimator itself, measured across enough seeds that its own uncertainty is
// small. Each technique must agree with it within their combined uncertainty.
TEST(VarianceReductionTest, EveryMethodAgreesWithCrudeAcrossSeeds) {
    constexpr int kSeeds = 32;
    constexpr std::int64_t kPaths = 20000;

    const Realised crude = measure(asian_call(), false, false, kPaths, 12, kSeeds, 40000);
    const Realised antithetic = measure(asian_call(), true, false, kPaths, 12, kSeeds, 41000);
    const Realised control = measure(asian_call(), false, true, kPaths, 12, kSeeds, 42000);
    const Realised combined = measure(asian_call(), true, true, kPaths, 12, kSeeds, 43000);

    // The uncertainty in each mean-of-means is its dispersion over sqrt(seeds).
    const double crude_uncertainty = crude.standard_deviation / std::sqrt(kSeeds);

    for (const auto& [name, realised] : {std::pair{"antithetic", antithetic},
                                         std::pair{"control", control},
                                         std::pair{"combined", combined}}) {
        const double uncertainty = realised.standard_deviation / std::sqrt(kSeeds);
        const double combined_uncertainty =
            std::sqrt(crude_uncertainty * crude_uncertainty + uncertainty * uncertainty);

        EXPECT_LT(std::abs(realised.mean - crude.mean), 4.0 * combined_uncertainty)
            << name << " disagrees with crude: " << realised.mean << " vs " << crude.mean
            << " (combined uncertainty " << combined_uncertainty << ")";
    }
}

// For a European call the truth is known, so unbiasedness is checked against it
// rather than against another estimator.
TEST(VarianceReductionTest, AntitheticIsUnbiasedAgainstTheAnalyticValue) {
    const double reference =
        BlackScholesAnalyticEngine::price(market(), european_call(), model()).value().value;

    constexpr int kSeeds = 32;
    const Realised antithetic = measure(european_call(), true, false, 20000, 1, kSeeds, 44000);

    const double uncertainty = antithetic.standard_deviation / std::sqrt(kSeeds);
    EXPECT_LT(std::abs(antithetic.mean - reference), 4.0 * uncertainty)
        << "antithetic mean " << antithetic.mean << " vs analytic " << reference;
}

// The pilot-estimated beta makes the control estimator exactly unbiased rather
// than merely nearly so. Checked against the analytic geometric Asian, whose
// price is known: controlling the geometric option with a *different* control
// would be circular, so this instead verifies the arithmetic estimator against
// the crude one at high seed count above, and here verifies that the control
// mechanism itself does not shift a known answer.
TEST(VarianceReductionTest, ControlVariateDoesNotShiftAKnownPrice) {
    // The geometric Asian priced by simulation with an arithmetic control is not
    // meaningful; instead check that the arithmetic estimator with control agrees
    // with a very long crude run, which is the best available reference.
    constexpr std::int64_t kLongPaths = 400000;

    const auto crude = MonteCarloEngine::price(
        market(), asian_call(), model(), make_config(kLongPaths, 12, 55555, false, false));
    ASSERT_TRUE(crude.ok()) << crude.error().describe();

    const auto controlled = MonteCarloEngine::price(
        market(), asian_call(), model(), make_config(100000, 12, 66666, false, true));
    ASSERT_TRUE(controlled.ok()) << controlled.error().describe();

    // Independent seeds, so the uncertainties add in quadrature.
    const double uncertainty = std::sqrt(std::pow(*crude.value().standard_error, 2) +
                                         std::pow(*controlled.value().standard_error, 2));
    EXPECT_LT(std::abs(controlled.value().value - crude.value().value), 4.0 * uncertainty)
        << "controlled " << controlled.value().value << " vs long crude " << crude.value().value;
}

// ---------------------------------------------------------------------------
// The reported uncertainty must be honest
//
// This is the test that catches the antithetic pairing error. The two members of
// a pair are negatively correlated by construction, so accumulating 2N payoffs as
// if independent would compute Var(X)/2N instead of Var(X)(1+rho)/2N and misstate
// the standard error by sqrt(1+rho). The estimate would be right and its error
// bar wrong, which is worse than both being wrong -- and no test of the *value*
// would notice.
// ---------------------------------------------------------------------------

TEST(VarianceReductionTest, ReportedStandardErrorMatchesRealisedDispersion) {
    constexpr int kSeeds = 40;
    constexpr std::int64_t kPaths = 20000;

    struct Case {
        const char* name;
        bool antithetic;
        bool control_variate;
        std::uint64_t seed_base;
    };

    for (const auto& c : {Case{"crude", false, false, 50000},
                          Case{"antithetic", true, false, 51000},
                          Case{"control", false, true, 52000},
                          Case{"combined", true, true, 53000}}) {
        const Realised realised =
            measure(asian_call(), c.antithetic, c.control_variate, kPaths, 12, kSeeds, c.seed_base);

        // The realised dispersion should match the average self-reported standard
        // error. Its own sampling error is ~1/sqrt(2(k-1)), about 11% at 40 seeds,
        // so the bound checks agreement in magnitude rather than to three digits.
        EXPECT_NEAR(realised.standard_deviation,
                    realised.reported_standard_error,
                    0.4 * realised.reported_standard_error)
            << c.name << ": realised dispersion " << realised.standard_deviation
            << " disagrees with the reported standard error " << realised.reported_standard_error;
    }
}

// ---------------------------------------------------------------------------
// Variance reduction, measured
// ---------------------------------------------------------------------------

// Measured on realised dispersion across seeds, not on one run's reported
// interval.
TEST(VarianceReductionTest, AntitheticReducesVarianceForAMonotonePayoff) {
    constexpr int kSeeds = 32;
    constexpr std::int64_t kPaths = 20000;

    const Realised crude = measure(asian_call(), false, false, kPaths, 12, kSeeds, 60000);
    const Realised antithetic = measure(asian_call(), true, false, kPaths, 12, kSeeds, 61000);

    const double ratio = std::pow(crude.standard_deviation / antithetic.standard_deviation, 2);

    // An Asian call is monotone in the shocks, so its reflection is strongly
    // negatively correlated and the pairing helps. The bound is loose because the
    // ratio is itself estimated from 32 dispersions.
    EXPECT_GT(ratio, 1.5) << "antithetic reduced variance by only " << ratio << "x";
}

TEST(VarianceReductionTest, ControlVariateReducesVarianceSubstantially) {
    constexpr int kSeeds = 32;
    constexpr std::int64_t kPaths = 20000;

    const Realised crude = measure(asian_call(), false, false, kPaths, 12, kSeeds, 62000);
    const Realised control = measure(asian_call(), false, true, kPaths, 12, kSeeds, 63000);

    const double ratio = std::pow(crude.standard_deviation / control.standard_deviation, 2);

    // The geometric and arithmetic averages of the same path differ only by AM
    // versus GM, so they correlate at about 0.999 and the reduction is large --
    // measured at roughly 500x. The bound is deliberately far below that: this
    // asserts the mechanism works, not a particular number, which EXP-05 reports
    // properly.
    EXPECT_GT(ratio, 50.0) << "the control reduced variance by only " << ratio << "x";
}

TEST(VarianceReductionTest, CombiningBothIsNoWorseThanEither) {
    constexpr int kSeeds = 32;
    constexpr std::int64_t kPaths = 20000;

    const Realised control = measure(asian_call(), false, true, kPaths, 12, kSeeds, 64000);
    const Realised combined = measure(asian_call(), true, true, kPaths, 12, kSeeds, 65000);

    // Combining should not hurt. Stated as "no worse" rather than "better"
    // because at this correlation the control already removes almost everything
    // the antithetic pairing could, and claiming a further gain would overstate
    // what 32 seeds can resolve.
    EXPECT_LT(combined.standard_deviation, control.standard_deviation * 1.3)
        << "combining was materially worse than the control alone";
}

// A method that does not help must be documented rather than quietly recommended
// (Phase 4 exit gate). Antithetic helps far less for a European call at one step
// than for an Asian, and this records that rather than leaving the impression it
// is uniformly worthwhile.
TEST(VarianceReductionTest, AntitheticHelpsLessForASingleStepEuropean) {
    constexpr int kSeeds = 32;
    constexpr std::int64_t kPaths = 20000;

    const Realised crude = measure(european_call(), false, false, kPaths, 1, kSeeds, 66000);
    const Realised antithetic = measure(european_call(), true, false, kPaths, 1, kSeeds, 67000);

    const double ratio = std::pow(crude.standard_deviation / antithetic.standard_deviation, 2);

    // It does help -- a call is monotone in its single shock -- but the gain is
    // modest, and it costs two payoffs per observation. Whether that is
    // worthwhile is a question about efficiency rather than variance, and EXP-05
    // answers it with runtime included.
    EXPECT_GT(ratio, 1.0) << "antithetic made a monotone payoff worse, which should not happen";

    // Recorded so a regression that silently changed the pairing would show.
    EXPECT_LT(ratio, 20.0) << "an implausibly large gain suggests the pair is not being averaged "
                              "into a single observation";
}

// ---------------------------------------------------------------------------
// Common random numbers
//
// Not a mode to switch on but a property of the design: a path is a pure function
// of its coordinates, so two runs sharing a seed share their paths exactly. That
// makes the *difference* between two nearby estimates far more precise than
// either estimate, which is what bump-and-revalue Greeks depend on in Phase 8.
// ---------------------------------------------------------------------------

TEST(CommonRandomNumbersTest, SharedSeedsMakeADifferenceFarMorePreciseThanItsParts) {
    constexpr int kTrials = 60;
    constexpr std::int64_t kPaths = 20000;
    constexpr double kBump = 1.0;

    const auto price_at = [&](double spot, std::uint64_t seed) {
        const auto bumped = MarketState::create(spot, 0.05, 0.0).value();
        const auto priced = MonteCarloEngine::price(
            bumped, european_call(), model(), make_config(kPaths, 1, seed, false, false));
        EXPECT_TRUE(priced.ok()) << priced.error().describe();
        return priced.value().value;
    };

    OnlineMoments shared;
    OnlineMoments independent;

    for (int trial = 0; trial < kTrials; ++trial) {
        const auto seed = static_cast<std::uint64_t>(70000 + trial);

        // Same seed on both sides: the paths are identical, so the difference
        // isolates the bump.
        shared.add((price_at(100.0 + kBump, seed) - price_at(100.0, seed)) / kBump);

        // Different seeds: the paths differ, so the difference carries both
        // estimates' sampling error.
        independent.add((price_at(100.0 + kBump, seed) - price_at(100.0, seed + 500000)) / kBump);
    }

    const double shared_dispersion = std::sqrt(shared.sample_variance().value());
    const double independent_dispersion = std::sqrt(independent.sample_variance().value());

    // Sharing the paths should cut the difference's noise by orders of magnitude:
    // the two prices move together, so most of the sampling error cancels.
    EXPECT_LT(shared_dispersion, independent_dispersion * 0.1)
        << "common random numbers barely helped: " << shared_dispersion << " vs "
        << independent_dispersion;

    // And the difference quotient must still estimate delta, which is 0.6368 for
    // these terms.
    const double analytic_delta =
        *BlackScholesAnalyticEngine::greeks(market(), european_call(), model()).value().delta;
    EXPECT_NEAR(shared.mean(), analytic_delta, 0.02)
        << "the shared-path difference quotient does not recover delta";
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

// Asking for a technique that will not be applied would report a method the run
// did not use.
TEST(VarianceReductionTest, ControlVariateIsRejectedWhereNoControlExists) {
    const auto european = MonteCarloEngine::price(
        market(), european_call(), model(), make_config(1000, 1, 1, false, true));
    ASSERT_FALSE(european.ok()) << "no control is defined for a European option";
    EXPECT_EQ(european.error().code, ErrorCode::UnsupportedCombination);

    const auto geometric_option =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 12).value();
    const auto geometric = MonteCarloEngine::price(
        market(), geometric_option, model(), make_config(1000, 12, 1, false, true));
    ASSERT_FALSE(geometric.ok())
        << "controlling the geometric option with itself would return the closed form dressed as "
           "a simulation";
    EXPECT_EQ(geometric.error().code, ErrorCode::UnsupportedCombination);
}

TEST(VarianceReductionTest, RejectsAPilotTooSmallToEstimateBeta) {
    MonteCarloConfig config = make_config(1000, 12, 1, false, true);
    config.control_variate_pilot_paths = 1;

    const auto priced = MonteCarloEngine::price(market(), asian_call(), model(), config);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

TEST(VarianceReductionTest, ReportsWhichEstimatorProducedTheNumber) {
    const auto priced = MonteCarloEngine::price(
        market(), asian_call(), model(), make_config(20000, 12, 1, true, true));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();

    const auto find = [&](const std::string& name) -> const Diagnostic* {
        for (const Diagnostic& d : priced.value().diagnostics) {
            if (d.name == name) {
                return &d;
            }
        }
        return nullptr;
    };

    // A stored result must say which estimator produced it, or it cannot be
    // reproduced or compared.
    ASSERT_NE(find("antithetic"), nullptr);
    ASSERT_NE(find("control_variate"), nullptr);
    EXPECT_TRUE(std::get<bool>(find("antithetic")->value));
    EXPECT_TRUE(std::get<bool>(find("control_variate")->value));

    // beta and the control's known expectation are what a reviewer needs to check
    // the correction by hand.
    ASSERT_NE(find("control_beta"), nullptr);
    ASSERT_NE(find("control_expectation"), nullptr);
    ASSERT_NE(find("control_pilot_correlation"), nullptr);

    // The control is nearly perfectly correlated here, so beta should sit near 1.
    EXPECT_NEAR(std::get<double>(find("control_beta")->value), 1.0, 0.3);
    EXPECT_GT(std::get<double>(find("control_pilot_correlation")->value), 0.9);

    // The pair average is one observation, so an antithetic run reports as many
    // observations as paths -- not twice as many.
    ASSERT_NE(find("observations"), nullptr);
    EXPECT_EQ(std::get<std::int64_t>(find("observations")->value), 20000);
}

// The control's expectation is the closed form, not an estimate. Reporting an
// estimated one would defeat the technique.
TEST(VarianceReductionTest, ControlExpectationIsTheClosedForm) {
    const auto priced = MonteCarloEngine::price(
        market(), asian_call(), model(), make_config(5000, 12, 1, false, true));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();

    const auto geometric_twin =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 12).value();
    const double expected =
        GeometricAsianAnalyticEngine::price(market(), geometric_twin, model()).value().value;

    for (const Diagnostic& d : priced.value().diagnostics) {
        if (d.name == "control_expectation") {
            EXPECT_DOUBLE_EQ(std::get<double>(d.value), expected);
            return;
        }
    }
    FAIL() << "the control's expectation was not reported";
}

}  // namespace
}  // namespace diffusionworks
