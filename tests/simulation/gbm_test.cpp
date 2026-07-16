#include <diffusionworks/core/error.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

struct Fixture {
    MarketState market;
    BlackScholesModel model;
    TimeGrid grid;
};

Fixture make_fixture(double spot,
                     double rate,
                     double dividend_yield,
                     double volatility,
                     double maturity,
                     std::int64_t steps) {
    return Fixture{MarketState::create(spot, rate, dividend_yield).value(),
                   BlackScholesModel::create(volatility).value(),
                   TimeGrid::uniform(maturity, steps).value()};
}

// ---------------------------------------------------------------------------
// TimeGrid
// ---------------------------------------------------------------------------

TEST(TimeGridTest, AcceptsValidGrids) {
    const auto grid = TimeGrid::uniform(2.0, 4);

    ASSERT_TRUE(grid.ok()) << grid.error().describe();
    EXPECT_DOUBLE_EQ(grid.value().maturity(), 2.0);
    EXPECT_EQ(grid.value().steps(), 4);
    EXPECT_DOUBLE_EQ(grid.value().step_size(), 0.5);
    EXPECT_NEAR(grid.value().sqrt_step_size(), std::sqrt(0.5), 1e-15);
}

TEST(TimeGridTest, RejectsInvalidGrids) {
    EXPECT_FALSE(TimeGrid::uniform(0.0, 4).ok()) << "a grid over an instant has no steps";
    EXPECT_FALSE(TimeGrid::uniform(-1.0, 4).ok());
    EXPECT_FALSE(TimeGrid::uniform(1.0, 0).ok());
    EXPECT_FALSE(TimeGrid::uniform(1.0, -5).ok());
    EXPECT_FALSE(TimeGrid::uniform(std::numeric_limits<double>::quiet_NaN(), 4).ok());
    EXPECT_FALSE(TimeGrid::uniform(std::numeric_limits<double>::infinity(), 4).ok());
}

TEST(TimeGridTest, EndpointsAreExact) {
    const auto grid = TimeGrid::uniform(1.0, 3).value();

    EXPECT_DOUBLE_EQ(grid.time_at(0), 0.0);
    // Exactly maturity, not the rounding of 3*(1/3): a payoff at expiry must see
    // the contract's maturity.
    EXPECT_DOUBLE_EQ(grid.time_at(3), 1.0);
}

// Times are computed as i*T/M rather than accumulated, which would drift. Over
// 10^6 steps the accumulated error is visible; multiplying by the index cannot
// drift at all.
TEST(TimeGridTest, TimesDoNotDriftOverManySteps) {
    constexpr std::int64_t kSteps = 1000000;
    const auto grid = TimeGrid::uniform(1.0, kSteps).value();

    EXPECT_DOUBLE_EQ(grid.time_at(kSteps), 1.0);
    EXPECT_NEAR(grid.time_at(kSteps / 2), 0.5, 1e-15);

    // What accumulation would have produced, demonstrated rather than asserted
    // from theory.
    double accumulated = 0.0;
    for (std::int64_t i = 0; i < kSteps; ++i) {
        accumulated += grid.step_size();
    }
    EXPECT_GT(std::abs(accumulated - 1.0), 0.0)
        << "accumulation was expected to drift; if it no longer does, this test no longer "
           "demonstrates why time_at multiplies";
    EXPECT_LT(std::abs(grid.time_at(kSteps) - 1.0), std::abs(accumulated - 1.0));
}

TEST(TimeGridTest, TimesAreMonotone) {
    const auto grid = TimeGrid::uniform(5.0, 100).value();

    double previous = -1.0;
    for (std::int64_t i = 0; i <= 100; ++i) {
        const double time = grid.time_at(i);
        EXPECT_GT(time, previous) << "at index " << i;
        previous = time;
    }
}

// ---------------------------------------------------------------------------
// Schemes
// ---------------------------------------------------------------------------

TEST(DiscretizationSchemeTest, RoundTrips) {
    for (const DiscretizationScheme scheme : {DiscretizationScheme::Exact,
                                              DiscretizationScheme::EulerMaruyama,
                                              DiscretizationScheme::Milstein}) {
        const auto parsed = parse_discretization_scheme(to_string(scheme));
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(*parsed, scheme);
    }

    EXPECT_FALSE(parse_discretization_scheme("runge_kutta").has_value());
    EXPECT_FALSE(parse_discretization_scheme("Exact").has_value());
}

// The exact scheme steps log-spot and carries the Ito correction; the explicit
// schemes step spot and do not. Using one drift for the other is a classic error
// that leaves prices plausible but biased, so the distinction is pinned.
TEST(GbmStepperTest, ExactSchemeCarriesTheItoCorrection) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.02, 0.2, 1.0, 4);

    const GbmStepper exact = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);
    const GbmStepper euler = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama);

    const double carry = 0.05 - 0.02;
    const double dt = 0.25;

    EXPECT_NEAR(exact.drift_term(), (carry - 0.5 * 0.2 * 0.2) * dt, 1e-15);
    EXPECT_NEAR(euler.drift_term(), carry * dt, 1e-15);
    EXPECT_NE(exact.drift_term(), euler.drift_term());

    // Both share the diffusion term.
    EXPECT_NEAR(exact.diffusion_term(), 0.2 * std::sqrt(dt), 1e-15);
    EXPECT_NEAR(euler.diffusion_term(), 0.2 * std::sqrt(dt), 1e-15);
}

TEST(GbmStepperTest, MatchesTheClosedFormOfEachScheme) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 1);
    const double dt = 1.0;
    const double z = 0.7;
    constexpr double kSpot = 100.0;

    const double exact_expected =
        kSpot * std::exp((0.05 - 0.5 * 0.04) * dt + 0.2 * std::sqrt(dt) * z);
    EXPECT_NEAR(
        GbmStepper::create(fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact)
            .advance(kSpot, z),
        exact_expected,
        1e-12);

    const double euler_expected = kSpot * (1.0 + 0.05 * dt + 0.2 * std::sqrt(dt) * z);
    EXPECT_NEAR(
        GbmStepper::create(
            fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama)
            .advance(kSpot, z),
        euler_expected,
        1e-12);

    const double milstein_expected =
        kSpot * (1.0 + 0.05 * dt + 0.2 * std::sqrt(dt) * z + 0.5 * 0.04 * dt * (z * z - 1.0));
    EXPECT_NEAR(GbmStepper::create(
                    fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Milstein)
                    .advance(kSpot, z),
                milstein_expected,
                1e-12);
}

// Milstein reduces to Euler exactly when the correction vanishes, at Z^2 = 1.
// This is a structural check on the correction term rather than on its size.
TEST(GbmStepperTest, MilsteinReducesToEulerWhereItsCorrectionVanishes) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.3, 1.0, 10);

    const GbmStepper euler = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama);
    const GbmStepper milstein = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Milstein);

    for (const double z : {-1.0, 1.0}) {
        EXPECT_NEAR(milstein.advance(100.0, z), euler.advance(100.0, z), 1e-12) << "z = " << z;
    }

    // And differs everywhere else.
    EXPECT_NE(milstein.advance(100.0, 2.0), euler.advance(100.0, 2.0));
}

// The exact scheme steps log-spot, so it cannot cross zero however large the
// shock. This is the property that makes it usable as a reference path.
TEST(GbmStepperTest, ExactSchemeStaysPositiveUnderAnyShock) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.5, 1.0, 1);
    const GbmStepper exact = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    for (const double z : {-10.0, -5.0, 0.0, 5.0, 10.0}) {
        EXPECT_GT(exact.advance(100.0, z), 0.0) << "z = " << z;
    }
}

// Euler crosses zero for a large enough negative shock. Demonstrated rather than
// asserted, because the resulting bias is what EXP-04 measures and what the
// engine warns about.
TEST(GbmStepperTest, EulerCanCrossZeroUnderALargeNegativeShock) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.5, 1.0, 1);
    const GbmStepper euler = GbmStepper::create(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama);

    // 1 + 0.05 + 0.5*z < 0 requires z < -2.1.
    EXPECT_LT(euler.advance(100.0, -5.0), 0.0)
        << "Euler was expected to cross zero here; that it did not means this test no longer "
           "demonstrates the discretisation's known weakness";
    EXPECT_GT(euler.advance(100.0, -2.0), 0.0);
}

// ---------------------------------------------------------------------------
// Path generation
// ---------------------------------------------------------------------------

TEST(GbmPathGeneratorTest, PathStartsAtSpotAndHasTheRightLength) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 8);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    ASSERT_EQ(generator.path_size(), 9U);

    std::vector<double> path(generator.path_size());
    const auto diagnostics = generator.generate(kSeed, 0, path);

    ASSERT_TRUE(diagnostics.ok()) << diagnostics.error().describe();
    EXPECT_DOUBLE_EQ(path.front(), 100.0);
    for (const double state : path) {
        EXPECT_TRUE(std::isfinite(state));
    }
}

TEST(GbmPathGeneratorTest, RejectsAMisSizedBuffer) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 8);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    std::vector<double> too_small(4);
    const auto result = generator.generate(kSeed, 0, too_small);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(GbmPathGeneratorTest, PathsAreReproducible) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 16);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    std::vector<double> first(generator.path_size());
    std::vector<double> second(generator.path_size());

    ASSERT_TRUE(generator.generate(kSeed, 7, first).ok());
    ASSERT_TRUE(generator.generate(kSeed, 7, second).ok());

    EXPECT_EQ(first, second);
}

TEST(GbmPathGeneratorTest, DifferentPathsDiffer) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 16);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    std::vector<double> first(generator.path_size());
    std::vector<double> second(generator.path_size());

    ASSERT_TRUE(generator.generate(kSeed, 0, first).ok());
    ASSERT_TRUE(generator.generate(kSeed, 1, second).ok());

    EXPECT_NE(first, second);
}

// The property that makes strong convergence measurable at all: two schemes on
// the same path index must consume the *same* Brownian increments. Comparing
// paths built from different draws would measure sampling noise rather than
// discretisation error (FAILURE-MODES section 6).
TEST(GbmPathGeneratorTest, SchemesShareBrownianIncrementsOnTheSamePath) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 32);

    const GbmPathGenerator exact(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);
    const GbmPathGenerator euler(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama);

    std::vector<double> exact_path(exact.path_size());
    std::vector<double> euler_path(euler.path_size());
    ASSERT_TRUE(exact.generate(kSeed, 3, exact_path).ok());
    ASSERT_TRUE(euler.generate(kSeed, 3, euler_path).ok());

    // The paths differ -- that is the discretisation error -- but they must be
    // driven by identical shocks. Recover each scheme's implied z from its own
    // step and require them to agree: the shocks are shared, the schemes are not.
    const GbmStepper& exact_stepper = exact.stepper();
    const GbmStepper& euler_stepper = euler.stepper();

    for (std::size_t i = 0; i < 32; ++i) {
        // Exact: S_{i+1} = S_i exp(drift + diff*z)  =>  z = (log(S_{i+1}/S_i) - drift)/diff
        const double z_from_exact =
            (std::log(exact_path[i + 1] / exact_path[i]) - exact_stepper.drift_term()) /
            exact_stepper.diffusion_term();

        // Euler: S_{i+1} = S_i (1 + drift + diff*z)  =>  z = (S_{i+1}/S_i - 1 - drift)/diff
        const double z_from_euler =
            ((euler_path[i + 1] / euler_path[i]) - 1.0 - euler_stepper.drift_term()) /
            euler_stepper.diffusion_term();

        EXPECT_NEAR(z_from_exact, z_from_euler, 1e-9)
            << "the two schemes saw different shocks at step " << i;
    }
}

// A path is a pure function of its index, so generating them out of order, or
// interleaved with other paths, cannot change any of them.
TEST(GbmPathGeneratorTest, PathsAreIndependentOfGenerationOrder) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.2, 1.0, 8);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    std::vector<std::vector<double>> forward;
    for (std::uint64_t index = 0; index < 16; ++index) {
        std::vector<double> path(generator.path_size());
        ASSERT_TRUE(generator.generate(kSeed, index, path).ok());
        forward.push_back(path);
    }

    for (std::uint64_t index = 16; index-- > 0;) {
        std::vector<double> path(generator.path_size());
        ASSERT_TRUE(generator.generate(kSeed, index, path).ok());
        EXPECT_EQ(path, forward[index]) << "path " << index << " changed when generated in reverse";
    }
}

// The exact scheme cannot leave the positive half-line, so it never reports an
// excursion.
TEST(GbmPathGeneratorTest, ExactSchemeNeverReportsNonPositiveStates) {
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 0.8, 2.0, 64);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    std::vector<double> path(generator.path_size());
    for (std::uint64_t index = 0; index < 2000; ++index) {
        const auto diagnostics = generator.generate(kSeed, index, path);
        ASSERT_TRUE(diagnostics.ok()) << diagnostics.error().describe();
        ASSERT_EQ(diagnostics.value().non_positive_states, 0) << "at path " << index;
    }
}

// Euler does leave it, and the count must be reported rather than swallowed. The
// payoff would clamp a negative terminal price to zero and return an entirely
// ordinary number, so this count is the only visible trace of the bias.
TEST(GbmPathGeneratorTest, EulerReportsNonPositiveStatesRatherThanHidingThem) {
    // A coarse grid and a large volatility make the crossing likely: one step of
    // 0.25 years at 150% volatility needs z < -1.4 or so.
    const Fixture fixture = make_fixture(100.0, 0.05, 0.0, 1.5, 1.0, 4);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::EulerMaruyama);

    std::vector<double> path(generator.path_size());
    std::int64_t total = 0;
    for (std::uint64_t index = 0; index < 5000; ++index) {
        const auto diagnostics = generator.generate(kSeed, index, path);
        ASSERT_TRUE(diagnostics.ok()) << diagnostics.error().describe();
        total += diagnostics.value().non_positive_states;
    }

    EXPECT_GT(total, 0)
        << "Euler was expected to cross zero under these parameters; if it no longer does, this "
           "test no longer demonstrates that the excursions are counted";
}

// ---------------------------------------------------------------------------
// Theoretical moments of the exact scheme
//
// The Phase 3 exit gate. The exact transition is the true log-normal law, so its
// sample moments must match the theory to within sampling error -- and the bounds
// below are stated in standard errors rather than chosen to pass.
// ---------------------------------------------------------------------------

class ExactGbmMomentsTest : public ::testing::TestWithParam<std::int64_t> {};

TEST_P(ExactGbmMomentsTest, TerminalMomentsMatchTheory) {
    constexpr double kSpot = 100.0;
    constexpr double kRate = 0.05;
    constexpr double kDividend = 0.02;
    constexpr double kVolatility = 0.3;
    constexpr double kMaturity = 2.0;
    constexpr int kPaths = 400000;

    const std::int64_t steps = GetParam();
    const Fixture fixture = make_fixture(kSpot, kRate, kDividend, kVolatility, kMaturity, steps);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    OnlineMoments terminal;
    std::vector<double> path(generator.path_size());
    for (std::uint64_t index = 0; index < kPaths; ++index) {
        ASSERT_TRUE(generator.generate(kSeed, index, path).ok());
        terminal.add(path.back());
    }

    // E[S_T] = S_0 e^{(r-q)T}
    const double carry = kRate - kDividend;
    const double expected_mean = kSpot * std::exp(carry * kMaturity);

    // Var[S_T] = S_0^2 e^{2(r-q)T} (e^{sigma^2 T} - 1)
    const double expected_variance = kSpot * kSpot * std::exp(2.0 * carry * kMaturity) *
                                     std::expm1(kVolatility * kVolatility * kMaturity);

    // The sample mean's own standard error, from the theoretical variance. The
    // bound is 4 of them: tight enough to catch a real error, loose enough that
    // the test is not a coin flip.
    const double mean_standard_error = std::sqrt(expected_variance / kPaths);
    EXPECT_NEAR(terminal.mean(), expected_mean, 4.0 * mean_standard_error) << "steps = " << steps;

    // The sample variance of a lognormal has standard error
    // sqrt((mu_4 - sigma^4)/N). For a lognormal with parameters (m, s),
    // the fourth central moment is large; rather than derive it, the variance is
    // checked in relative terms with a bound reflecting its heavier tail.
    const auto variance = terminal.sample_variance();
    ASSERT_TRUE(variance.ok()) << variance.error().describe();
    EXPECT_NEAR(variance.value() / expected_variance, 1.0, 0.05) << "steps = " << steps;
}

// The exact transition is exact over any horizon, so the step count changes the
// cost and nothing else. Running the same moment check at 1, 4 and 64 steps
// makes that a measurement rather than a claim.
INSTANTIATE_TEST_SUITE_P(StepCounts, ExactGbmMomentsTest, ::testing::Values(1, 4, 64));

// The terminal law is lognormal, so log(S_T/S_0) is exactly normal with known
// parameters. Checking the log-domain moments is a sharper test than the price
// moments: it isolates the drift and diffusion terms without the lognormal's
// heavy tail inflating the sampling error.
TEST(ExactGbmTest, LogReturnsAreNormalWithTheRightParameters) {
    constexpr double kSpot = 100.0;
    constexpr double kRate = 0.05;
    constexpr double kDividend = 0.02;
    constexpr double kVolatility = 0.3;
    constexpr double kMaturity = 2.0;
    constexpr int kPaths = 500000;

    const Fixture fixture = make_fixture(kSpot, kRate, kDividend, kVolatility, kMaturity, 8);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    OnlineMoments log_returns;
    std::vector<double> path(generator.path_size());
    for (std::uint64_t index = 0; index < kPaths; ++index) {
        ASSERT_TRUE(generator.generate(kSeed, index, path).ok());
        log_returns.add(std::log(path.back() / kSpot));
    }

    // log(S_T/S_0) ~ N((r - q - sigma^2/2)T, sigma^2 T)
    const double expected_mean = (kRate - kDividend - 0.5 * kVolatility * kVolatility) * kMaturity;
    const double expected_variance = kVolatility * kVolatility * kMaturity;

    const double mean_standard_error = std::sqrt(expected_variance / kPaths);
    EXPECT_NEAR(log_returns.mean(), expected_mean, 4.0 * mean_standard_error);

    // A normal sample variance has standard error sigma^2 sqrt(2/N).
    const double variance_standard_error = expected_variance * std::sqrt(2.0 / kPaths);
    EXPECT_NEAR(
        log_returns.sample_variance().value(), expected_variance, 4.0 * variance_standard_error);
}

// The martingale property under the risk-neutral measure: the discounted,
// dividend-adjusted price has expectation S_0. This is the identity every
// risk-neutral price rests on, so it is checked directly rather than inferred
// from a price agreeing with a formula.
TEST(ExactGbmTest, DiscountedPriceIsAMartingale) {
    constexpr double kSpot = 100.0;
    constexpr double kRate = 0.05;
    constexpr double kDividend = 0.03;
    constexpr double kVolatility = 0.25;
    constexpr double kMaturity = 1.5;
    constexpr int kPaths = 400000;

    const Fixture fixture = make_fixture(kSpot, kRate, kDividend, kVolatility, kMaturity, 4);
    const GbmPathGenerator generator(
        fixture.market, fixture.model, fixture.grid, DiscretizationScheme::Exact);

    OnlineMoments discounted;
    std::vector<double> path(generator.path_size());
    for (std::uint64_t index = 0; index < kPaths; ++index) {
        ASSERT_TRUE(generator.generate(kSeed, index, path).ok());
        // e^{-(r-q)T} S_T has expectation S_0.
        discounted.add(std::exp(-(kRate - kDividend) * kMaturity) * path.back());
    }

    const auto interval = discounted.confidence_interval(0.99);
    ASSERT_TRUE(interval.ok()) << interval.error().describe();
    EXPECT_TRUE(interval.value().contains(kSpot))
        << "the 99% interval for the discounted price [" << interval.value().lower << ", "
        << interval.value().upper << "] excludes S_0 = " << kSpot;
}

}  // namespace
}  // namespace diffusionworks
