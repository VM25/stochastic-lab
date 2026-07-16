#include <diffusionworks/core/error.hpp>
#include <diffusionworks/experiments/convergence.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

MarketState baseline_market() {
    return MarketState::create(100.0, 0.05, 0.0).value();
}

BlackScholesModel baseline_model() {
    return BlackScholesModel::create(0.3).value();
}

// The grid runs to M = 1024 because Milstein needs it. A higher-order scheme
// enters its asymptotic regime *later*: at M = 256 Milstein's fitted order is
// still 0.985, and only past M = 128 does the interval cover 1. Euler reaches 0.5
// by M = 64. Truncating the grid at 256 to save time would have produced a real
// measurement of a pre-asymptotic slope and an apparent contradiction with theory.
std::vector<ConvergenceLevel> strong_levels(DiscretizationScheme scheme,
                                            std::uint64_t paths = 40000) {
    std::vector<ConvergenceLevel> levels;
    for (const std::int64_t steps : {16, 32, 64, 128, 256, 512, 1024}) {
        auto level = measure_strong_error(
            baseline_market(), baseline_model(), scheme, 1.0, steps, paths, kSeed);
        EXPECT_TRUE(level.ok()) << level.error().describe();
        levels.push_back(level.value());
    }
    return levels;
}

// ---------------------------------------------------------------------------
// Path coupling: the property the whole strong-error measurement rests on
// ---------------------------------------------------------------------------

// Two schemes at the same (seed, path index) must consume identical shocks. If
// they did not, the "strong error" would be the difference of two independent
// samples -- an O(1) quantity that does not converge at all, and which would
// still produce a plot and a slope. EXP-02 names this as a failure condition.
//
// Tested by exploiting a fact about the schemes: as sigma -> 0 both the exact and
// the explicit schemes become deterministic, so agreement there proves nothing.
// Instead we check that the exact and Milstein paths, driven by the same shocks,
// track each other far more closely than two exact paths driven by *different*
// seeds. The ratio is the evidence.
TEST(ConvergenceTest, CoupledSchemesShareTheirBrownianPath) {
    const auto market = baseline_market();
    const auto model = baseline_model();
    const auto grid = TimeGrid::uniform(1.0, 256).value();

    const GbmPathGenerator exact(market, model, grid, DiscretizationScheme::Exact);
    const GbmPathGenerator milstein(market, model, grid, DiscretizationScheme::Milstein);

    std::vector<double> exact_path(exact.path_size());
    std::vector<double> milstein_path(milstein.path_size());
    std::vector<double> decoupled_path(exact.path_size());

    double coupled_difference = 0.0;
    double decoupled_difference = 0.0;
    constexpr std::uint64_t kPaths = 2000;

    for (std::uint64_t i = 0; i < kPaths; ++i) {
        ASSERT_TRUE(exact.generate(kSeed, i, exact_path).ok());
        ASSERT_TRUE(milstein.generate(kSeed, i, milstein_path).ok());
        // A different seed: same law, unrelated Brownian path.
        ASSERT_TRUE(exact.generate(kSeed + 1, i, decoupled_path).ok());

        coupled_difference += std::abs(milstein_path.back() - exact_path.back());
        decoupled_difference += std::abs(decoupled_path.back() - exact_path.back());
    }
    coupled_difference /= kPaths;
    decoupled_difference /= kPaths;

    // At M = 256 the coupled difference is a discretisation error of order dt;
    // the decoupled one is the spread of the terminal law itself. Orders of
    // magnitude apart.
    EXPECT_LT(coupled_difference, 0.05);
    EXPECT_GT(decoupled_difference, 10.0);
    EXPECT_GT(decoupled_difference / coupled_difference, 100.0)
        << "the schemes do not appear to share a Brownian path: coupled difference "
        << coupled_difference << " vs decoupled " << decoupled_difference;
}

// The exact scheme is the reference, so its strong error against itself is
// identically zero. Fitting that would report infinite order from a tautology.
TEST(ConvergenceTest, RefusesTheStrongErrorOfTheExactSchemeAgainstItself) {
    const auto level = measure_strong_error(
        baseline_market(), baseline_model(), DiscretizationScheme::Exact, 1.0, 32, 1000, kSeed);
    ASSERT_FALSE(level.ok());
    EXPECT_EQ(level.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Strong convergence orders
//
// The headline result of EXP-02. The verdict rests on the asymptotic window
// because the theoretical order describes the limit dt -> 0, and the coarse end
// of any finite grid is not in it.
// ---------------------------------------------------------------------------

TEST(ConvergenceTest, EulerMaruyamaAttainsStrongOrderOneHalf) {
    const auto study = fit_convergence("euler_maruyama",
                                       "strong_error",
                                       strong_levels(DiscretizationScheme::EulerMaruyama),
                                       0.5,
                                       /*asymptotic_level_count=*/4);
    ASSERT_TRUE(study.ok()) << study.error().describe();

    EXPECT_TRUE(study.value().verdict == ConvergenceVerdict::Consistent ||
                study.value().verdict == ConvergenceVerdict::ConsistentAsymptotically)
        << "verdict was " << to_string(study.value().verdict) << "; asymptotic slope "
        << study.value().asymptotic_fit.slope << " CI ["
        << study.value().asymptotic_slope_interval.lower << ", "
        << study.value().asymptotic_slope_interval.upper << "]";

    EXPECT_TRUE(study.value().asymptotic_slope_interval.contains(0.5));
    EXPECT_GT(study.value().full_fit.r_squared, 0.99);
}

TEST(ConvergenceTest, MilsteinAttainsStrongOrderOne) {
    const auto study = fit_convergence("milstein",
                                       "strong_error",
                                       strong_levels(DiscretizationScheme::Milstein),
                                       1.0,
                                       /*asymptotic_level_count=*/4);
    ASSERT_TRUE(study.ok()) << study.error().describe();

    EXPECT_TRUE(study.value().verdict == ConvergenceVerdict::Consistent ||
                study.value().verdict == ConvergenceVerdict::ConsistentAsymptotically)
        << "verdict was " << to_string(study.value().verdict) << "; asymptotic slope "
        << study.value().asymptotic_fit.slope << " CI ["
        << study.value().asymptotic_slope_interval.lower << ", "
        << study.value().asymptotic_slope_interval.upper << "]";

    EXPECT_TRUE(study.value().asymptotic_slope_interval.contains(1.0));
}

// The local orders must climb toward the theoretical one as the grid refines.
// This is the evidence that the full-range slope's shortfall is pre-asymptotic
// contamination rather than a wrong order: a scheme that genuinely converged at
// order 0.96 would show local orders scattered about 0.96, not marching to 1.
TEST(ConvergenceTest, LocalOrdersApproachTheTheoreticalOrderAsTheGridRefines) {
    const auto study = fit_convergence(
        "milstein", "strong_error", strong_levels(DiscretizationScheme::Milstein), 1.0, 4);
    ASSERT_TRUE(study.ok());
    ASSERT_GE(study.value().local_orders.size(), 4U);

    const auto& orders = study.value().local_orders;
    EXPECT_LT(orders.front().order, 0.95) << "the coarsest pair should still be pre-asymptotic";
    EXPECT_GT(orders.back().order, 0.96) << "the finest pair should be close to order 1";
    EXPECT_GT(orders.back().order, orders.front().order)
        << "the local order must improve as the grid refines";
}

// Milstein must beat Euler, and the gap must widen as the grid refines: that is
// what a higher order means. Comparing at one grid spacing would show only that
// one constant is smaller than another.
TEST(ConvergenceTest, MilsteinOutperformsEulerAndTheGapWidensWithRefinement) {
    const auto euler = strong_levels(DiscretizationScheme::EulerMaruyama);
    const auto milstein = strong_levels(DiscretizationScheme::Milstein);
    ASSERT_EQ(euler.size(), milstein.size());

    double previous_ratio = 0.0;
    for (std::size_t i = 0; i < euler.size(); ++i) {
        ASSERT_EQ(euler[i].steps, milstein[i].steps);
        EXPECT_LT(milstein[i].error, euler[i].error) << "at M = " << euler[i].steps;

        const double ratio = euler[i].error / milstein[i].error;
        EXPECT_GT(ratio, previous_ratio)
            << "the advantage must grow with refinement; at M = " << euler[i].steps
            << " the ratio was " << ratio << " against " << previous_ratio << " at the previous"
            << " level";
        previous_ratio = ratio;
    }

    // Half an order of separation over a 32x refinement: the ratio should grow
    // roughly as sqrt(dt), i.e. by about sqrt(32) ~ 5.7 across this range.
    EXPECT_GT(previous_ratio, 20.0);
}

// ---------------------------------------------------------------------------
// Weak convergence
// ---------------------------------------------------------------------------

// The exact weak error, from the closed-form scheme moments. No sampling
// uncertainty exists here, so the verdict must be Consistent outright.
TEST(ConvergenceTest, AnalyticWeakErrorGivesOrderOneForBothSchemes) {
    for (const auto scheme :
         {DiscretizationScheme::EulerMaruyama, DiscretizationScheme::Milstein}) {
        for (const auto test_function : {WeakTestFunction::Identity, WeakTestFunction::Square}) {
            std::vector<ConvergenceLevel> levels;
            for (const std::int64_t steps : {32, 64, 128, 256, 512}) {
                auto level = weak_error_analytic(
                    baseline_market(), baseline_model(), scheme, 1.0, test_function, steps);
                ASSERT_TRUE(level.ok()) << level.error().describe();
                EXPECT_EQ(level.value().source, ErrorSource::Analytic);
                EXPECT_FALSE(level.value().error_standard_error.has_value())
                    << "an exact computation must not report a sampling error";
                levels.push_back(level.value());
            }

            const auto study = fit_convergence(
                std::string(to_string(scheme)), "weak_error", std::move(levels), 1.0, 3);
            ASSERT_TRUE(study.ok()) << study.error().describe();
            EXPECT_EQ(study.value().verdict, ConvergenceVerdict::Consistent)
                << to_string(scheme) << " / " << to_string(test_function) << ": slope "
                << study.value().full_fit.slope;
            EXPECT_NEAR(study.value().asymptotic_fit.slope, 1.0, 0.01)
                << to_string(scheme) << " / " << to_string(test_function);
        }
    }
}

// The explicit schemes under-state E[S_T] here, and the sign carries information
// that |error| would discard.
TEST(ConvergenceTest, WeakErrorSignIsReportedAndNegativeUnderPositiveCarry) {
    const auto level = weak_error_analytic(baseline_market(),
                                           baseline_model(),
                                           DiscretizationScheme::EulerMaruyama,
                                           1.0,
                                           WeakTestFunction::Identity,
                                           16);
    ASSERT_TRUE(level.ok());
    ASSERT_TRUE(level.value().signed_error.has_value());
    EXPECT_LT(*level.value().signed_error, 0.0);
    EXPECT_DOUBLE_EQ(level.value().error, std::abs(*level.value().signed_error));
}

// The call payoff has no closed-form expectation under a discretised scheme.
// Returning an approximation would let a caller mistake it for an exact value.
TEST(ConvergenceTest, RefusesAnalyticWeakErrorForTheCallPayoff) {
    const auto level = weak_error_analytic(baseline_market(),
                                           baseline_model(),
                                           DiscretizationScheme::EulerMaruyama,
                                           1.0,
                                           WeakTestFunction::CallPayoff,
                                           16);
    ASSERT_FALSE(level.ok());
    EXPECT_EQ(level.error().code, ErrorCode::UnsupportedCombination);
}

// The simulated weak error must reproduce the analytic one where both exist.
// This is what licenses the simulated estimator for the call payoff, where no
// closed form is available to check it against.
TEST(ConvergenceTest, SimulatedWeakErrorAgreesWithTheAnalyticOne) {
    for (const std::int64_t steps : {4, 8, 16}) {
        const auto analytic = weak_error_analytic(baseline_market(),
                                                  baseline_model(),
                                                  DiscretizationScheme::EulerMaruyama,
                                                  1.0,
                                                  WeakTestFunction::Identity,
                                                  steps);
        ASSERT_TRUE(analytic.ok());

        const auto simulated = measure_weak_error(baseline_market(),
                                                  baseline_model(),
                                                  DiscretizationScheme::EulerMaruyama,
                                                  1.0,
                                                  100.0,
                                                  WeakTestFunction::Identity,
                                                  steps,
                                                  400000,
                                                  kSeed);
        ASSERT_TRUE(simulated.ok()) << simulated.error().describe();
        ASSERT_TRUE(simulated.value().signed_error.has_value());
        ASSERT_TRUE(simulated.value().error_standard_error.has_value());

        EXPECT_NEAR(*simulated.value().signed_error,
                    *analytic.value().signed_error,
                    3.0 * *simulated.value().error_standard_error)
            << "at M = " << steps;
    }
}

// Pairing against the exact scheme on common paths is what makes the bias
// visible. Without it the estimator reports noise.
TEST(ConvergenceTest, PairingReducesTheWeakErrorVarianceSubstantially) {
    const auto market = baseline_market();
    const auto model = baseline_model();
    const auto grid = TimeGrid::uniform(1.0, 16).value();

    const GbmPathGenerator euler(market, model, grid, DiscretizationScheme::EulerMaruyama);
    const GbmPathGenerator exact(market, model, grid, DiscretizationScheme::Exact);
    std::vector<double> euler_path(euler.path_size());
    std::vector<double> exact_path(exact.path_size());

    OnlineMoments unpaired;
    OnlineMoments paired;
    constexpr std::uint64_t kPaths = 100000;
    for (std::uint64_t i = 0; i < kPaths; ++i) {
        ASSERT_TRUE(euler.generate(kSeed, i, euler_path).ok());
        ASSERT_TRUE(exact.generate(kSeed, i, exact_path).ok());
        unpaired.add(euler_path.back());
        paired.add(euler_path.back() - exact_path.back());
    }

    // Measured at roughly 9x on the standard error for these parameters. The
    // bound is loose because the exact factor is a property of the parameters,
    // not a guarantee; what is being pinned is that pairing helps by a lot.
    EXPECT_GT(unpaired.standard_error().value() / paired.standard_error().value(), 5.0);
}

// A study whose levels are not resolved above their own sampling noise must
// report NoiseDominated rather than a slope. A fit through noise still returns a
// number, and that number looks exactly like a measurement.
TEST(ConvergenceTest, ReportsNoiseDominatedRatherThanFittingThroughNoise) {
    std::vector<ConvergenceLevel> levels;
    for (const std::int64_t steps : {4, 8, 16, 32}) {
        ConvergenceLevel level;
        level.steps = steps;
        level.step_size = 1.0 / static_cast<double>(steps);
        level.source = ErrorSource::Simulated;
        level.error = 1e-3 / static_cast<double>(steps);
        // A standard error the same size as the error itself: nothing here is
        // resolved.
        level.error_standard_error = level.error;
        level.paths = 1000;
        levels.push_back(level);
    }

    const auto study = fit_convergence("euler_maruyama", "weak_error", levels, 1.0, 3);
    ASSERT_TRUE(study.ok()) << study.error().describe();
    EXPECT_EQ(study.value().verdict, ConvergenceVerdict::NoiseDominated);
    ASSERT_TRUE(study.value().worst_resolution.has_value());
    EXPECT_LT(*study.value().worst_resolution, kResolutionThreshold);
}

// A scheme claiming an order it does not have must be contradicted, not
// accommodated. This is the guard that makes a passing verdict mean something.
TEST(ConvergenceTest, ReportsInconsistentWhenTheOrderIsWrong) {
    const auto study = fit_convergence("euler_maruyama",
                                       "strong_error",
                                       strong_levels(DiscretizationScheme::EulerMaruyama),
                                       /*theoretical_order=*/1.0,  // false: Euler is 0.5
                                       4);
    ASSERT_TRUE(study.ok());
    EXPECT_EQ(study.value().verdict, ConvergenceVerdict::Inconsistent);
}

// ---------------------------------------------------------------------------
// Rejection
// ---------------------------------------------------------------------------

TEST(ConvergenceTest, RefusesTooFewLevels) {
    std::vector<ConvergenceLevel> levels(2);
    levels[0] = {8, 0.125, ErrorSource::Simulated, 1.0, {}, {}, {}, {}, 100, 0};
    levels[1] = {16, 0.0625, ErrorSource::Simulated, 0.5, {}, {}, {}, {}, 100, 0};

    const auto study = fit_convergence("s", "strong_error", levels, 0.5, 3);
    ASSERT_FALSE(study.ok());
    EXPECT_EQ(study.error().code, ErrorCode::InvalidArgument);
}

TEST(ConvergenceTest, RefusesAnAsymptoticWindowThatIsTooSmallOrTooLarge) {
    const auto levels = strong_levels(DiscretizationScheme::Milstein, 2000);

    EXPECT_FALSE(fit_convergence("m", "strong_error", levels, 1.0, 2).ok());
    EXPECT_FALSE(fit_convergence("m", "strong_error", levels, 1.0, levels.size() + 1).ok());
    EXPECT_TRUE(fit_convergence("m", "strong_error", levels, 1.0, levels.size()).ok());
}

TEST(ConvergenceTest, RefusesTooFewPaths) {
    EXPECT_FALSE(
        measure_strong_error(
            baseline_market(), baseline_model(), DiscretizationScheme::Milstein, 1.0, 16, 1, kSeed)
            .ok());
    EXPECT_FALSE(measure_weak_error(baseline_market(),
                                    baseline_model(),
                                    DiscretizationScheme::Milstein,
                                    1.0,
                                    100.0,
                                    WeakTestFunction::Identity,
                                    16,
                                    1,
                                    kSeed)
                     .ok());
}

}  // namespace
}  // namespace diffusionworks
