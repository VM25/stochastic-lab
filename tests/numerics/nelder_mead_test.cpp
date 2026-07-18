#include <diffusionworks/core/error.hpp>
#include <diffusionworks/numerics/nelder_mead.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace diffusionworks {
namespace {

// A quadratic bowl with a known minimum at (1, -2, 3). The simplex must find it.
double bowl(std::span<const double> x) {
    const double a = x[0] - 1.0;
    const double b = x[1] + 2.0;
    const double c = x[2] - 3.0;
    return a * a + 2.0 * b * b + 0.5 * c * c + 1.0;
}

// The Rosenbrock function: a curved valley whose minimum at (1, 1) is easy to fall
// into but slow to reach. The classic torture test for a derivative-free method.
double rosenbrock(std::span<const double> x) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

TEST(NelderMeadTest, FindsTheMinimumOfAQuadraticBowl) {
    const std::array<double, 3> start{0.0, 0.0, 0.0};
    const auto result = nelder_mead(bowl, start);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_EQ(result.value().status, NelderMeadStatus::Converged);
    EXPECT_NEAR(result.value().point[0], 1.0, 1e-4);
    EXPECT_NEAR(result.value().point[1], -2.0, 1e-4);
    EXPECT_NEAR(result.value().point[2], 3.0, 1e-4);
    EXPECT_NEAR(result.value().value, 1.0, 1e-6);
}

TEST(NelderMeadTest, SolvesRosenbrockFromAFarStart) {
    const std::array<double, 2> start{-1.2, 1.0};
    NelderMeadConfig config;
    config.max_iterations = 5000;
    const auto result = nelder_mead(rosenbrock, start, config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_EQ(result.value().status, NelderMeadStatus::Converged);
    EXPECT_NEAR(result.value().point[0], 1.0, 1e-3);
    EXPECT_NEAR(result.value().point[1], 1.0, 1e-3);
}

// Determinism: the same objective and start give byte-identical results, so a
// calibration is reproducible from its configuration.
TEST(NelderMeadTest, IsDeterministic) {
    const std::array<double, 3> start{0.5, 0.5, 0.5};
    const auto a = nelder_mead(bowl, start);
    const auto b = nelder_mead(bowl, start);
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(a.value().value, b.value().value);
    EXPECT_EQ(a.value().point, b.value().point);
    EXPECT_EQ(a.value().function_evaluations, b.value().function_evaluations);
}

// A non-finite objective in part of the space is treated as +infinity, so the simplex
// retreats from it and still finds the minimum in the feasible region rather than
// being derailed by a NaN.
TEST(NelderMeadTest, RetreatsFromANonFiniteRegion) {
    // Minimum of (x-2)^2 at x = 2, but the objective is +inf for x < 0.
    const auto guarded = [](std::span<const double> x) -> double {
        if (x[0] < 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return (x[0] - 2.0) * (x[0] - 2.0);
    };
    const std::array<double, 1> start{1.0};
    const auto result = nelder_mead(guarded, start);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_NEAR(result.value().point[0], 2.0, 1e-4);
}

// Non-convergence is a reported status, not an error: a tight budget returns the best
// point so far with MaxIterationsReached, so the caller can see it did not converge.
TEST(NelderMeadTest, ReportsNonConvergenceRatherThanFailing) {
    const std::array<double, 2> start{-1.2, 1.0};
    NelderMeadConfig config;
    config.max_iterations = 5;  // far too few for Rosenbrock
    const auto result = nelder_mead(rosenbrock, start, config);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().status, NelderMeadStatus::MaxIterationsReached);
    EXPECT_LE(result.value().iterations, 5);
}

TEST(NelderMeadTest, ReportsTheEvaluationCount) {
    const std::array<double, 3> start{0.0, 0.0, 0.0};
    const auto result = nelder_mead(bowl, start);
    ASSERT_TRUE(result.ok());
    // At least the initial simplex of n+1 = 4 evaluations happened.
    EXPECT_GE(result.value().function_evaluations, 4);
}

TEST(NelderMeadTest, RejectsAnEmptyStart) {
    const std::vector<double> empty;
    const auto result = nelder_mead(bowl, empty);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(NelderMeadTest, RejectsABadConfiguration) {
    const std::array<double, 2> start{0.0, 0.0};
    NelderMeadConfig config;
    config.function_tolerance = -1.0;
    const auto result = nelder_mead(rosenbrock, start, config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks
