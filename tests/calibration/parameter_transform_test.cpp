#include <diffusionworks/calibration/parameter_transform.hpp>
#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace diffusionworks {
namespace {

HestonParameters params(double v0, double kappa, double theta, double xi, double rho) {
    return HestonParameters{.initial_variance = v0,
                            .mean_reversion = kappa,
                            .long_run_variance = theta,
                            .vol_of_variance = xi,
                            .correlation = rho};
}

// The forward map followed by the inverse is the identity for any point strictly
// inside the box -- the property that lets the optimizer work in the unconstrained
// space and read results back as parameters.
TEST(ParameterTransformTest, RoundTripsThroughTheUnconstrainedSpace) {
    const auto bounds = HestonParameterBounds::defaults();
    for (const auto& p : {params(0.04, 2.0, 0.04, 0.5, -0.7),
                          params(0.01, 0.5, 0.09, 1.2, 0.3),
                          params(0.25, 10.0, 0.2, 3.0, -0.95)}) {
        const auto x = to_unconstrained(p, bounds);
        ASSERT_TRUE(x.ok()) << x.error().describe();
        const auto back = to_constrained(x.value(), bounds);
        EXPECT_NEAR(back.initial_variance, p.initial_variance, 1e-12);
        EXPECT_NEAR(back.mean_reversion, p.mean_reversion, 1e-11);
        EXPECT_NEAR(back.long_run_variance, p.long_run_variance, 1e-12);
        EXPECT_NEAR(back.vol_of_variance, p.vol_of_variance, 1e-11);
        EXPECT_NEAR(back.correlation, p.correlation, 1e-12);
    }
}

// For moderate unconstrained coordinates the inverse map lands strictly inside the
// box -- the property the optimizer relies on while it searches near the interior.
TEST(ParameterTransformTest, InverseMapLandsStrictlyInsideForModerateCoordinates) {
    const auto bounds = HestonParameterBounds::defaults();
    for (const double coordinate : {-15.0, -3.0, 0.0, 4.5, 15.0}) {
        const std::array<double, 5> point{
            coordinate, coordinate, coordinate, coordinate, coordinate};
        const auto p = to_constrained(point, bounds);
        EXPECT_GT(p.initial_variance, bounds.lower.initial_variance);
        EXPECT_LT(p.initial_variance, bounds.upper.initial_variance);
        EXPECT_GT(p.correlation, bounds.lower.correlation);
        EXPECT_LT(p.correlation, bounds.upper.correlation);
        EXPECT_GT(p.vol_of_variance, bounds.lower.vol_of_variance);
        EXPECT_LT(p.vol_of_variance, bounds.upper.vol_of_variance);
    }
}

// The inverse map is total: however far the optimizer steps, the result stays within
// the closed box and always builds a valid model. At extreme coordinates the logistic
// saturates to a bound in double precision -- that is a feasible parameter set (the
// bounds are chosen so the boundary is admissible), not an out-of-box escape.
TEST(ParameterTransformTest, InverseMapStaysFeasibleEvenAtExtremeCoordinates) {
    const auto bounds = HestonParameterBounds::defaults();
    for (const double coordinate : {-1000.0, -50.0, 50.0, 1000.0}) {
        const std::array<double, 5> point{
            coordinate, coordinate, coordinate, coordinate, coordinate};
        const auto p = to_constrained(point, bounds);
        EXPECT_GE(p.initial_variance, bounds.lower.initial_variance);
        EXPECT_LE(p.initial_variance, bounds.upper.initial_variance);
        EXPECT_GE(p.correlation, bounds.lower.correlation);
        EXPECT_LE(p.correlation, bounds.upper.correlation);
        EXPECT_GE(p.vol_of_variance, bounds.lower.vol_of_variance);
        EXPECT_LE(p.vol_of_variance, bounds.upper.vol_of_variance);
        // The feasible point always builds a valid model, saturated or not.
        EXPECT_TRUE(p.to_model().ok());
    }
}

// A parameter on or past a bound has no finite unconstrained image and is rejected,
// rather than mapped to an infinity that would poison the optimizer.
TEST(ParameterTransformTest, RejectsAParameterOutsideTheBox) {
    const auto bounds = HestonParameterBounds::defaults();
    const auto on_bound = params(bounds.upper.initial_variance, 2.0, 0.04, 0.5, -0.7);
    const auto solved = to_unconstrained(on_bound, bounds);
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::InvalidArgument);
}

// An inside-out box is refused before any mapping is attempted.
TEST(ParameterTransformTest, RejectsAnInsideOutBox) {
    HestonParameterBounds bounds = HestonParameterBounds::defaults();
    std::swap(bounds.lower.mean_reversion, bounds.upper.mean_reversion);
    EXPECT_FALSE(bounds.valid());
    const auto solved = to_unconstrained(params(0.04, 2.0, 0.04, 0.5, -0.7), bounds);
    ASSERT_FALSE(solved.ok());
    EXPECT_EQ(solved.error().code, ErrorCode::InvalidArgument);
}

// The midpoint of the box maps to the origin of the unconstrained space, and the
// origin maps back to the midpoint -- the anchor that makes the transform easy to
// reason about.
TEST(ParameterTransformTest, BoxMidpointMapsToTheOrigin) {
    const auto bounds = HestonParameterBounds::defaults();
    const auto mid = params(0.5 * (bounds.lower.initial_variance + bounds.upper.initial_variance),
                            0.5 * (bounds.lower.mean_reversion + bounds.upper.mean_reversion),
                            0.5 * (bounds.lower.long_run_variance + bounds.upper.long_run_variance),
                            0.5 * (bounds.lower.vol_of_variance + bounds.upper.vol_of_variance),
                            0.5 * (bounds.lower.correlation + bounds.upper.correlation));
    const auto x = to_unconstrained(mid, bounds);
    ASSERT_TRUE(x.ok());
    for (const double coordinate : x.value()) {
        EXPECT_NEAR(coordinate, 0.0, 1e-12);
    }
}

}  // namespace
}  // namespace diffusionworks
