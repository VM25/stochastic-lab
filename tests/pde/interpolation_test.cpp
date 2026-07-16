#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace diffusionworks {
namespace {

/// Samples `f` on the grid.
std::vector<double> sample(const AssetGrid& grid, double (*f)(double)) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(grid.nodes()));
    for (std::int64_t i = 0; i < grid.nodes(); ++i) {
        values.push_back(f(grid.at(i)));
    }
    return values;
}

// ---------------------------------------------------------------------------
// Exactness on a node
//
// The one place interpolation has no error at all. If this fails, every
// interpolated price carries an offset that no amount of grid refinement removes.
// ---------------------------------------------------------------------------

TEST(InterpolationTest, RecoversNodeValuesExactly) {
    const auto grid = AssetGrid::uniform(400.0, 41).value();
    const auto values = sample(grid, [](double s) { return 3.0 * s - 7.0; });

    for (std::int64_t i = 0; i < grid.nodes(); ++i) {
        const auto v = interpolate_linear(grid, values, grid.at(i));
        ASSERT_TRUE(v.ok()) << "at node " << i << ": " << v.error().describe();
        EXPECT_DOUBLE_EQ(v.value(), values[static_cast<std::size_t>(i)]) << "at node " << i;
    }
}

// Node-exactness must hold on an awkward spacing too, where s/dS lands an ulp
// below an integer and floor() picks the interval to the left.
TEST(InterpolationTest, RecoversNodeValuesOnAnAwkwardSpacing) {
    for (const std::int64_t nodes : {7, 13, 101, 1001}) {
        const auto grid = AssetGrid::uniform(300.0, nodes).value();
        const auto values = sample(grid, [](double s) { return s * s; });

        for (std::int64_t i = 0; i < grid.nodes(); ++i) {
            const auto v = interpolate_linear(grid, values, grid.at(i));
            ASSERT_TRUE(v.ok());
            EXPECT_NEAR(v.value(),
                        values[static_cast<std::size_t>(i)],
                        1e-9 * (1.0 + values[static_cast<std::size_t>(i)]))
                << "nodes " << nodes << " at node " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Off-node behaviour
// ---------------------------------------------------------------------------

// A linear function is reproduced exactly everywhere, not only on nodes: linear
// interpolation of a linear function is that function. This separates "the
// interpolation is right" from "the grid is fine enough".
TEST(InterpolationTest, ReproducesALinearFunctionExactlyOffNode) {
    const auto grid = AssetGrid::uniform(400.0, 21).value();
    const auto values = sample(grid, [](double s) { return 2.5 * s + 11.0; });

    for (const double s : {0.0, 1.0, 17.3, 99.9, 200.0, 333.33, 400.0}) {
        const auto v = interpolate_linear(grid, values, s);
        ASSERT_TRUE(v.ok()) << "at s = " << s;
        EXPECT_NEAR(v.value(), 2.5 * s + 11.0, 1e-10) << "at s = " << s;
    }
}

// The midpoint of a span is the average of its endpoints. Hand-checkable, and it
// catches a weight computed from the wrong end.
TEST(InterpolationTest, MidpointIsTheAverageOfItsNodes) {
    const auto grid = AssetGrid::uniform(100.0, 11).value();  // spacing 10
    std::vector<double> values(11);
    for (std::size_t i = 0; i < 11; ++i) {
        values[i] = static_cast<double>(i * i);  // 0, 1, 4, 9, ...
    }

    // s = 25 sits midway between node 2 (S=20, v=4) and node 3 (S=30, v=9).
    const auto v = interpolate_linear(grid, values, 25.0);
    ASSERT_TRUE(v.ok());
    EXPECT_DOUBLE_EQ(v.value(), 6.5);

    // And a quarter of the way: 22.5 -> 4 + 0.25*(9-4) = 5.25.
    EXPECT_DOUBLE_EQ(interpolate_linear(grid, values, 22.5).value(), 5.25);
}

// Linear interpolation cannot overshoot: the result is bounded by the two nodes it
// spans. That is the property it was chosen for over a cubic, which is more
// accurate on smooth data but can introduce oscillation into an answer the scheme
// did not oscillate -- and telling scheme error from artifact is the whole point.
TEST(InterpolationTest, NeverOvershootsItsBracketingNodes) {
    const auto grid = AssetGrid::uniform(100.0, 11).value();
    // A deliberately vicious profile: a spike that a cubic would ring around.
    const std::vector<double> values{0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (double s = 0.0; s <= 100.0; s += 0.37) {
        const auto v = interpolate_linear(grid, values, s);
        ASSERT_TRUE(v.ok()) << "at s = " << s;
        EXPECT_GE(v.value(), 0.0) << "at s = " << s << ": undershoot below both nodes";
        EXPECT_LE(v.value(), 100.0) << "at s = " << s << ": overshoot above both nodes";
    }
}

// ---------------------------------------------------------------------------
// Boundaries
// ---------------------------------------------------------------------------

TEST(InterpolationTest, IsExactAtBothBoundaries) {
    const auto grid = AssetGrid::uniform(400.0, 41).value();
    const auto values = sample(grid, [](double s) { return std::sqrt(s); });

    EXPECT_DOUBLE_EQ(interpolate_linear(grid, values, 0.0).value(), values.front());
    EXPECT_DOUBLE_EQ(interpolate_linear(grid, values, 400.0).value(), values.back());
}

// Just inside the top boundary, where floor(s/dS) would index the last node and
// the span [n-1, n] does not exist. The clamp to n-2 is what keeps this in bounds,
// and this test is why it is there.
TEST(InterpolationTest, HandlesASpotJustInsideTheTopBoundary) {
    const auto grid = AssetGrid::uniform(400.0, 41).value();
    const auto values = sample(grid, [](double s) { return 2.0 * s; });

    const double just_inside = std::nextafter(400.0, 0.0);
    const auto v = interpolate_linear(grid, values, just_inside);
    ASSERT_TRUE(v.ok());
    EXPECT_NEAR(v.value(), 2.0 * just_inside, 1e-8);
}

// Outside the grid the PDE was never solved. Refused rather than clamped or
// extrapolated: answering about the boundary instead would return a number with no
// relationship to the question.
TEST(InterpolationTest, RefusesASpotOutsideTheGrid) {
    const auto grid = AssetGrid::uniform(400.0, 41).value();
    const auto values = sample(grid, [](double s) { return s; });

    EXPECT_FALSE(interpolate_linear(grid, values, -1e-9).ok());
    EXPECT_FALSE(interpolate_linear(grid, values, 400.001).ok());
    EXPECT_FALSE(interpolate_linear(grid, values, std::numeric_limits<double>::quiet_NaN()).ok());
    EXPECT_FALSE(interpolate_linear(grid, values, std::numeric_limits<double>::infinity()).ok());
}

TEST(InterpolationTest, RefusesAMismatchedValueVector) {
    const auto grid = AssetGrid::uniform(400.0, 41).value();
    const std::vector<double> too_short(10, 1.0);

    const auto v = interpolate_linear(grid, too_short, 100.0);
    ASSERT_FALSE(v.ok());
    EXPECT_EQ(v.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Interpolation error under refinement
//
// Linear interpolation is second order in dS on smooth data. That matters here
// because it is the *same* order as the schemes: the interpolation is not a
// negligible afterthought, it contributes error at the same rate the PDE solve
// does, and refining the grid improves both together.
// ---------------------------------------------------------------------------

TEST(InterpolationTest, ErrorApproachesSecondOrderInSpacingOnSmoothData) {
    // A smooth, curved function. exp is reproduced exactly by no finite
    // interpolant, so the error is a real measurement rather than a rounding
    // artifact.
    const auto f = [](double s) { return std::exp(0.01 * s); };

    std::vector<double> spacings;
    std::vector<double> errors;

    for (const std::int64_t nodes : {26, 51, 101, 201, 401, 801, 1601, 3201}) {
        const auto grid = AssetGrid::uniform(400.0, nodes).value();
        std::vector<double> values;
        for (std::int64_t i = 0; i < grid.nodes(); ++i) {
            values.push_back(f(grid.at(i)));
        }

        // Sampled at the midpoint of every span, where the error is largest. A
        // sweep that happened to land on nodes would measure zero and report an
        // infinite order.
        double worst = 0.0;
        const double offset = grid.spacing() * 0.5;
        for (std::int64_t i = 0; i + 1 < grid.nodes(); ++i) {
            const double s = grid.at(i) + offset;
            const auto v = interpolate_linear(grid, values, s);
            ASSERT_TRUE(v.ok());
            worst = std::max(worst, std::abs(v.value() - f(s)));
        }

        spacings.push_back(grid.spacing());
        errors.push_back(worst);
    }

    // The local order between adjacent spacings, which is what shows the trend.
    //
    // Why not simply require a fitted interval to contain 2.0: the fit is
    // extraordinarily precise here (R^2 = 0.9999999) and its interval is
    // consequently narrower than the model's own misspecification. Linear
    // interpolation's error is (1/8) f''(xi) dS^2 only to leading order; the
    // neglected cubic term leaves the fitted slope a hair below 2 at every
    // spacing, and the interval -- which measures how well the *line* is
    // determined, not how well the model fits -- excludes 2.0 even over the
    // finest window. That is the same phenomenon EXP-02 documents for the SDE
    // schemes, and the same answer applies: the evidence for the order is that
    // the local orders climb monotonically toward it.
    std::vector<double> local_orders;
    for (std::size_t i = 1; i < spacings.size(); ++i) {
        local_orders.push_back(std::log(errors[i - 1] / errors[i]) /
                               std::log(spacings[i - 1] / spacings[i]));
    }

    for (std::size_t i = 1; i < local_orders.size(); ++i) {
        EXPECT_GT(local_orders[i], local_orders[i - 1])
            << "the local order must improve monotonically as the spacing refines; at index " << i
            << " it went from " << local_orders[i - 1] << " to " << local_orders[i];
    }

    // Approaching 2 from below, as the cubic term implies, and close by the
    // finest spacing.
    EXPECT_LT(local_orders.back(), 2.0)
        << "the neglected cubic term keeps the measured order just below 2";
    EXPECT_GT(local_orders.back(), 1.99)
        << "the finest local order should be within 1% of 2, but it is " << local_orders.back();

    // And the fit is still a good description of the data -- just of a curve that
    // is very slightly not a straight line.
    const auto fit = fit_power_law(spacings, errors);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();
    EXPECT_GT(fit.value().r_squared, 0.9999);
    EXPECT_NEAR(fit.value().slope, 2.0, 0.02);
}

// On a node the error is zero at every spacing, so refinement cannot improve it.
// This is the property that makes strike alignment worth doing: it puts the kink
// -- the one place the value function is least well approximated -- where the
// interpolation contributes nothing.
TEST(InterpolationTest, NodeErrorIsZeroAtEverySpacing) {
    const auto f = [](double s) { return std::exp(0.01 * s); };

    for (const std::int64_t nodes : {26, 51, 101, 201}) {
        const auto grid = AssetGrid::uniform(400.0, nodes).value();
        std::vector<double> values;
        for (std::int64_t i = 0; i < grid.nodes(); ++i) {
            values.push_back(f(grid.at(i)));
        }

        for (std::int64_t i = 0; i < grid.nodes(); i += 7) {
            const auto v = interpolate_linear(grid, values, grid.at(i));
            ASSERT_TRUE(v.ok());
            EXPECT_DOUBLE_EQ(v.value(), f(grid.at(i))) << "nodes " << nodes << " node " << i;
        }
    }
}

}  // namespace
}  // namespace diffusionworks
