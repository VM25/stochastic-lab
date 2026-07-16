#include <diffusionworks/core/error.hpp>
#include <diffusionworks/experiments/scheme_moments.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

// ---------------------------------------------------------------------------
// The claim these formulas make is that no simulation is needed. So the first
// thing to check is that they agree with simulation.
//
// This is the sharp test of the whole EXP-03 approach: the closed forms are
// derived by hand, and a derivation error would silently produce a clean,
// convergent, wrong answer. Monte Carlo cannot confirm them to 12 digits, but it
// can confirm them to within its own standard error -- and a wrong coefficient
// would miss by far more than that.
// ---------------------------------------------------------------------------

struct MomentCase {
    const char* name;
    double spot;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
    std::int64_t steps;
};

std::vector<MomentCase> moment_cases() {
    return {
        {"baseline", 100.0, 0.05, 0.00, 0.30, 1.0, 8},
        {"coarse_grid", 100.0, 0.05, 0.00, 0.30, 1.0, 2},
        {"with_dividend", 100.0, 0.05, 0.03, 0.20, 2.0, 4},
        {"high_volatility", 100.0, 0.02, 0.00, 0.60, 1.0, 8},
        {"negative_carry", 80.0, 0.01, 0.07, 0.25, 1.5, 6},
    };
}

TEST(SchemeMomentsTest, ClosedFormsAgreeWithSimulation) {
    for (const auto& c : moment_cases()) {
        const auto market = MarketState::create(c.spot, c.rate, c.dividend_yield).value();
        const auto model = BlackScholesModel::create(c.volatility).value();
        const auto grid = TimeGrid::uniform(c.maturity, c.steps).value();

        for (const auto scheme : {DiscretizationScheme::EulerMaruyama,
                                  DiscretizationScheme::Milstein,
                                  DiscretizationScheme::Exact}) {
            const auto predicted =
                scheme_terminal_moments(market, model, scheme, c.maturity, c.steps);
            ASSERT_TRUE(predicted.ok()) << predicted.error().describe();

            const GbmPathGenerator generator(market, model, grid, scheme);
            std::vector<double> path(generator.path_size());

            OnlineMoments first;
            OnlineMoments second;
            constexpr std::uint64_t kPaths = 400000;
            for (std::uint64_t i = 0; i < kPaths; ++i) {
                ASSERT_TRUE(generator.generate(kSeed, i, path).ok());
                first.add(path.back());
                second.add(path.back() * path.back());
            }

            // Three standard errors: the formulas are either right or wrong by a
            // structural amount, so this separates them without being flaky.
            const double first_tolerance = 3.0 * first.standard_error().value();
            const double second_tolerance = 3.0 * second.standard_error().value();

            EXPECT_NEAR(first.mean(), predicted.value().first, first_tolerance)
                << c.name << " / " << to_string(scheme) << ": E[S_T]";
            EXPECT_NEAR(second.mean(), predicted.value().second, second_tolerance)
                << c.name << " / " << to_string(scheme) << ": E[S_T^2]";
        }
    }
}

// ---------------------------------------------------------------------------
// Structure the formulas must have
// ---------------------------------------------------------------------------

// The exact scheme samples the true terminal law, so its moments must not depend
// on the step count at all. If they did, it would not be exact.
TEST(SchemeMomentsTest, ExactSchemeMomentsAreIndependentOfStepCount) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();
    const auto model = BlackScholesModel::create(0.3).value();

    const auto analytic = analytic_terminal_moments(market, model, 1.0);
    ASSERT_TRUE(analytic.ok());

    for (const std::int64_t steps : {1, 2, 10, 1000}) {
        const auto moments =
            scheme_terminal_moments(market, model, DiscretizationScheme::Exact, 1.0, steps);
        ASSERT_TRUE(moments.ok());
        EXPECT_DOUBLE_EQ(moments.value().first, analytic.value().first) << "steps = " << steps;
        EXPECT_DOUBLE_EQ(moments.value().second, analytic.value().second) << "steps = " << steps;
    }
}

// The Milstein correction has mean zero, so it cannot move E[S]. Euler and
// Milstein therefore share a first moment exactly -- Milstein's strong order 1
// buys nothing at all in the weak error of the mean.
//
// This is the counter-intuitive result EXP-03 exists to surface, so it is pinned
// as an equality rather than left as a remark.
TEST(SchemeMomentsTest, MilsteinAndEulerShareTheirFirstMomentExactly) {
    for (const auto& c : moment_cases()) {
        const auto market = MarketState::create(c.spot, c.rate, c.dividend_yield).value();
        const auto model = BlackScholesModel::create(c.volatility).value();

        const auto euler = scheme_terminal_moments(
            market, model, DiscretizationScheme::EulerMaruyama, c.maturity, c.steps);
        const auto milstein = scheme_terminal_moments(
            market, model, DiscretizationScheme::Milstein, c.maturity, c.steps);
        ASSERT_TRUE(euler.ok());
        ASSERT_TRUE(milstein.ok());

        EXPECT_DOUBLE_EQ(euler.value().first, milstein.value().first) << c.name;

        // The second moments must differ: the correction contributes
        // (sigma^4/2) dt^2 per step to E[u^2]. Equality here would mean the
        // Milstein branch is not being taken.
        EXPECT_GT(milstein.value().second, euler.value().second) << c.name;
    }
}

// Both schemes' terminal moments must approach the true ones as the grid refines,
// and the bias must halve per grid doubling. That ratio *is* weak order 1, and it
// is established here exactly rather than regressed through sampling noise.
TEST(SchemeMomentsTest, BiasHalvesPerGridDoublingForBothSchemes) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();
    const auto exact = analytic_terminal_moments(market, model, 1.0).value();

    for (const auto scheme :
         {DiscretizationScheme::EulerMaruyama, DiscretizationScheme::Milstein}) {
        double previous_first = 0.0;
        double previous_second = 0.0;

        for (const std::int64_t steps : {64, 128, 256, 512}) {
            const auto moments = scheme_terminal_moments(market, model, scheme, 1.0, steps);
            ASSERT_TRUE(moments.ok());

            const double first_bias = std::abs(moments.value().first - exact.first);
            const double second_bias = std::abs(moments.value().second - exact.second);

            if (previous_first > 0.0) {
                // Order 1 means the ratio is 2. The window is tight because these
                // are exact numbers: the only slack is the residual second-order
                // term, which is already below 0.3% at M = 64.
                EXPECT_NEAR(previous_first / first_bias, 2.0, 0.01)
                    << to_string(scheme) << " first moment at M = " << steps;
                EXPECT_NEAR(previous_second / second_bias, 2.0, 0.02)
                    << to_string(scheme) << " second moment at M = " << steps;
            }
            previous_first = first_bias;
            previous_second = second_bias;
        }
    }
}

// The explicit schemes under-state both moments here, and the sign is stable
// rather than incidental: (1+a dt)^M < e^{aT} for a > 0 by the strict convexity
// of the exponential.
TEST(SchemeMomentsTest, ExplicitSchemesUnderstateTheMomentsUnderPositiveCarry) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();
    const auto exact = analytic_terminal_moments(market, model, 1.0).value();

    for (const auto scheme :
         {DiscretizationScheme::EulerMaruyama, DiscretizationScheme::Milstein}) {
        for (const std::int64_t steps : {4, 16, 64, 256}) {
            const auto moments = scheme_terminal_moments(market, model, scheme, 1.0, steps);
            ASSERT_TRUE(moments.ok());
            EXPECT_LT(moments.value().first, exact.first)
                << to_string(scheme) << " at M = " << steps;
            EXPECT_LT(moments.value().second, exact.second)
                << to_string(scheme) << " at M = " << steps;
        }
    }
}

// The variance implied by the two moments must be positive. A formula error that
// preserved both moments' magnitudes but not their relationship would show up
// here and nowhere else.
TEST(SchemeMomentsTest, ImpliedVarianceIsPositive) {
    for (const auto& c : moment_cases()) {
        const auto market = MarketState::create(c.spot, c.rate, c.dividend_yield).value();
        const auto model = BlackScholesModel::create(c.volatility).value();

        for (const auto scheme : {DiscretizationScheme::EulerMaruyama,
                                  DiscretizationScheme::Milstein,
                                  DiscretizationScheme::Exact}) {
            const auto moments =
                scheme_terminal_moments(market, model, scheme, c.maturity, c.steps);
            ASSERT_TRUE(moments.ok());
            EXPECT_GT(moments.value().variance(), 0.0) << c.name << " / " << to_string(scheme);
        }
    }
}

// Zero volatility leaves the price deterministic: S_T = S_0 e^{aT} and the
// variance collapses. The explicit schemes still carry their drift bias, which is
// the point -- the bias is a property of the drift discretisation, not of the noise.
TEST(SchemeMomentsTest, ZeroVolatilityLeavesTheDriftBiasIntact) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.0).value();

    const auto euler =
        scheme_terminal_moments(market, model, DiscretizationScheme::EulerMaruyama, 1.0, 4);
    ASSERT_TRUE(euler.ok());

    // (1 + 0.05/4)^4 * 100, exactly.
    EXPECT_NEAR(euler.value().first, 100.0 * std::pow(1.0125, 4), 1e-12);
    EXPECT_NEAR(euler.value().variance(), 0.0, 1e-9);

    const auto exact = scheme_terminal_moments(market, model, DiscretizationScheme::Exact, 1.0, 4);
    ASSERT_TRUE(exact.ok());
    EXPECT_NEAR(exact.value().first, 100.0 * std::exp(0.05), 1e-12);
}

TEST(SchemeMomentsTest, RejectsInvalidGrids) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();

    EXPECT_FALSE(
        scheme_terminal_moments(market, model, DiscretizationScheme::EulerMaruyama, 1.0, 0).ok());
    EXPECT_FALSE(
        scheme_terminal_moments(market, model, DiscretizationScheme::EulerMaruyama, -1.0, 4).ok());
}

}  // namespace
}  // namespace diffusionworks
