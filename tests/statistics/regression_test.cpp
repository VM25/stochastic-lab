#include <diffusionworks/core/error.hpp>
#include <diffusionworks/statistics/distributions.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Exactness on a line
//
// A regression through collinear points must return the generating line and a
// zero residual. This is the check that catches a transposed sum or a swapped
// x/y: an almost-right implementation still produces an almost-right slope on
// noisy data, but only the right one is exact here.
// ---------------------------------------------------------------------------

TEST(RegressionTest, RecoversAnExactLine) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y;
    for (const double xi : x) {
        y.push_back(3.0 * xi - 7.0);
    }

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();

    EXPECT_NEAR(fit.value().slope, 3.0, 1e-13);
    EXPECT_NEAR(fit.value().intercept, -7.0, 1e-13);
    EXPECT_NEAR(fit.value().r_squared, 1.0, 1e-13);
    EXPECT_NEAR(fit.value().residual_standard_deviation, 0.0, 1e-13);
    EXPECT_NEAR(fit.value().slope_standard_error, 0.0, 1e-13);
}

// A small case checked against an independent implementation.
//
// x = {1,2,3,4}, y = {2,4,5,4}. Centred sums: Sxx = 5, Sxy = 3.5, Syy = 4.75, so
// slope = 0.7 and intercept = 3.75 - 0.7*2.5 = 2.0. Residuals are
// {-0.7, 0.6, 0.9, -0.8}, SSres = 2.30, s^2 = 1.15, SE(slope) = sqrt(1.15/5).
//
// These match scipy.stats.linregress exactly (slope 0.7, intercept 2.0,
// stderr 0.4795831523312719, rvalue^2 0.5157894736842106). They are written out
// as literals so the test does not depend on scipy being installed, but they were
// not derived here -- an earlier draft of this test asserted slope 0.8 from an
// arithmetic slip in Sxy, and the C++ was correct. Checking against a separate
// implementation is what distinguished the two.
TEST(RegressionTest, MatchesAnIndependentlyComputedFit) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{2.0, 4.0, 5.0, 4.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();

    EXPECT_NEAR(fit.value().slope, 0.7, 1e-13);
    EXPECT_NEAR(fit.value().intercept, 2.0, 1e-13);
    EXPECT_NEAR(fit.value().slope_standard_error, 0.4795831523312719, 1e-13);
    EXPECT_NEAR(fit.value().residual_standard_deviation, 1.072380529476361, 1e-13);
    EXPECT_NEAR(fit.value().r_squared, 0.5157894736842106, 1e-13);
    EXPECT_EQ(fit.value().observations, 4U);
    EXPECT_DOUBLE_EQ(fit.value().degrees_of_freedom(), 2.0);
}

// ---------------------------------------------------------------------------
// The power-law path, which is what the convergence studies actually use
// ---------------------------------------------------------------------------

TEST(RegressionTest, RecoversAPowerLawExponent) {
    // y = 2.5 * x^0.5, sampled on a log-spaced grid like a real convergence study.
    std::vector<double> x;
    std::vector<double> y;
    for (int k = 0; k < 6; ++k) {
        const double xi = std::pow(2.0, -k);
        x.push_back(xi);
        y.push_back(2.5 * std::sqrt(xi));
    }

    const auto fit = fit_power_law(x, y);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();

    EXPECT_NEAR(fit.value().slope, 0.5, 1e-12);
    EXPECT_NEAR(std::exp(fit.value().intercept), 2.5, 1e-11);
}

TEST(RegressionTest, RecoversFirstOrderExponent) {
    std::vector<double> x;
    std::vector<double> y;
    for (int k = 0; k < 6; ++k) {
        const double xi = std::pow(2.0, -k);
        x.push_back(xi);
        y.push_back(0.3 * xi);
    }

    const auto fit = fit_power_law(x, y);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();
    EXPECT_NEAR(fit.value().slope, 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// The interval
// ---------------------------------------------------------------------------

// The critical value must be Student-t, not normal. With 2 degrees of freedom
// they differ by more than a factor of two, which is the difference between a
// slope that appears to exclude the theoretical order and one that does not.
//
// The reference is exact rather than tabulated. Student-t at nu = 2 has an
// elementary CDF, F(t) = 1/2 + t/(2*sqrt(2+t^2)), which inverts in closed form to
//
//     t_p = 2c*sqrt(2)/sqrt(1-4c^2),   c = p - 1/2,
//
// giving t_{0.975,2} = 4.3026527297494638523... to 50 digits under mpmath. This
// matters: scipy.stats.t.ppf returns 4.302652729696142, which is wrong at 5e-11,
// and a widely reproduced table value is wrong at 1.6e-10. Neither is accurate
// enough to be the oracle here, so the closed form is used and the tolerance is
// set by *our* quantile's accuracy rather than by scipy's.
TEST(RegressionTest, SlopeIntervalUsesStudentTNotNormal) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{2.0, 4.0, 5.0, 4.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok());

    const auto interval = fit.value().slope_interval(0.95);
    ASSERT_TRUE(interval.ok()) << interval.error().describe();

    const double t_critical = 4.3026527297494638;
    const double expected_half_width = t_critical * fit.value().slope_standard_error;

    EXPECT_NEAR(interval.value().lower, 0.7 - expected_half_width, 1e-12);
    EXPECT_NEAR(interval.value().upper, 0.7 + expected_half_width, 1e-12);
    EXPECT_DOUBLE_EQ(interval.value().level, 0.95);

    // The normal quantile would give a far narrower interval. If this ever fails,
    // someone swapped in 1.96 and every convergence verdict in the project became
    // over-confident.
    const double normal_half_width = 1.959963984540054 * fit.value().slope_standard_error;
    EXPECT_GT(interval.value().width() / (2.0 * normal_half_width), 2.0)
        << "the interval is too narrow to be a t interval at 2 degrees of freedom";
}

// The quantile our interval depends on, checked against the nu = 2 closed form.
// This is the one degrees-of-freedom value where an exact elementary reference
// exists, so it pins the continued-fraction inversion to something that is not
// another numerical library.
TEST(RegressionTest, StudentTQuantileMatchesTheClosedFormAtTwoDegreesOfFreedom) {
    for (const double p : {0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999}) {
        const double c = p - 0.5;
        const double exact = 2.0 * c * std::sqrt(2.0) / std::sqrt(1.0 - 4.0 * c * c);

        const auto computed = student_t_quantile(p, 2.0);
        ASSERT_TRUE(computed.ok()) << computed.error().describe();
        EXPECT_NEAR(computed.value(), exact, 1e-12 * std::max(1.0, std::abs(exact)))
            << "at p = " << p;
    }
}

TEST(RegressionTest, WiderLevelGivesWiderInterval) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y{2.0, 4.0, 5.0, 4.0, 6.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok());

    const double narrow = fit.value().slope_interval(0.90).value().width();
    const double wide = fit.value().slope_interval(0.99).value().width();
    EXPECT_GT(wide, narrow);
}

// ---------------------------------------------------------------------------
// Rejection
// ---------------------------------------------------------------------------

// Two points fit exactly, leaving no residual and hence no standard error. A
// slope reported from them would carry a zero uncertainty and so would look like
// the most reliable point in a study. Refused instead.
TEST(RegressionTest, RefusesTwoPoints) {
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> y{3.0, 5.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_FALSE(fit.ok());
    EXPECT_EQ(fit.error().code, ErrorCode::InvalidArgument);
}

TEST(RegressionTest, RefusesMismatchedLengths) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> y{3.0, 5.0};

    EXPECT_FALSE(ordinary_least_squares(x, y).ok());
    EXPECT_FALSE(fit_power_law(x, y).ok());
}

// A vertical scatter has no slope. Returning infinity would propagate into a
// convergence order that looks like a very fast method.
TEST(RegressionTest, RefusesConstantX) {
    const std::vector<double> x{2.0, 2.0, 2.0, 2.0};
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_FALSE(fit.ok());
    EXPECT_EQ(fit.error().code, ErrorCode::InvalidArgument);
}

TEST(RegressionTest, RefusesNonFiniteData) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> y{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_FALSE(fit.ok());
    EXPECT_EQ(fit.error().code, ErrorCode::NonFiniteValue);
}

// A zero or negative error at one grid level is a finding -- an underflow, or a
// level where the scheme happens to be exact. Dropping it would fit a different
// study than the one that ran, so the fit is refused and the caller must look.
TEST(RegressionTest, RefusesNonPositiveDataInPowerLawRatherThanDroppingIt) {
    const std::vector<double> x{1.0, 0.5, 0.25, 0.125};
    const std::vector<double> y{1.0, 0.5, 0.0, 0.125};

    const auto fit = fit_power_law(x, y);
    ASSERT_FALSE(fit.ok());
    EXPECT_EQ(fit.error().code, ErrorCode::InvalidArgument);
    // The message must say which level, or it cannot be acted on.
    EXPECT_NE(fit.error().describe().find("observation 2"), std::string::npos)
        << fit.error().describe();
}

TEST(RegressionTest, RejectsInvalidConfidenceLevel) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{2.0, 4.0, 5.0, 4.0};

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok());

    EXPECT_FALSE(fit.value().slope_interval(0.0).ok());
    EXPECT_FALSE(fit.value().slope_interval(1.0).ok());
    EXPECT_FALSE(fit.value().slope_interval(-0.5).ok());
    EXPECT_FALSE(fit.value().slope_interval(std::numeric_limits<double>::quiet_NaN()).ok());
}

// ---------------------------------------------------------------------------
// Numerical conditioning
//
// The centred formulation exists because the textbook one cancels. Convergence
// studies regress on log step sizes, which are large in magnitude and narrow in
// spread -- exactly the case that breaks sum(x^2) - n*x_bar^2.
// ---------------------------------------------------------------------------

TEST(RegressionTest, StaysAccurateOnPoorlyConditionedAbscissae) {
    // x values near 1e8 with unit spread: the uncentred Sxx would lose about 16
    // digits to cancellation and return garbage or a negative variance.
    std::vector<double> x;
    std::vector<double> y;
    for (int k = 0; k < 5; ++k) {
        const double xi = 1e8 + static_cast<double>(k);
        x.push_back(xi);
        y.push_back(2.0 * xi + 1.0);
    }

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok()) << fit.error().describe();
    EXPECT_NEAR(fit.value().slope, 2.0, 1e-7);
}

// A near-perfect fit must not produce a negative residual variance and hence a
// NaN standard deviation, which is what SSres = Syy - slope*Sxy does here.
TEST(RegressionTest, NearPerfectFitDoesNotProduceNaNUncertainty) {
    std::vector<double> x;
    std::vector<double> y;
    for (int k = 0; k < 8; ++k) {
        const double xi = static_cast<double>(k + 1);
        x.push_back(xi);
        y.push_back(1.5 * xi + 2.0);
    }

    const auto fit = ordinary_least_squares(x, y);
    ASSERT_TRUE(fit.ok());
    EXPECT_TRUE(std::isfinite(fit.value().residual_standard_deviation));
    EXPECT_GE(fit.value().residual_standard_deviation, 0.0);
    EXPECT_TRUE(std::isfinite(fit.value().slope_standard_error));
    EXPECT_GE(fit.value().slope_standard_error, 0.0);
}

}  // namespace
}  // namespace diffusionworks
