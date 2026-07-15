#include <diffusionworks/numerics/normal.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace diffusionworks {
namespace {

// Reference values from Abramowitz & Stegun, "Handbook of Mathematical
// Functions" (1964), Table 26.1, which tabulates the standard normal
// distribution to 15 decimal places. Values below are the exact analytic
// identities and A&S tabulations.

TEST(NormalPdfTest, MatchesClosedFormAtZero) {
    // phi(0) = 1/sqrt(2*pi)
    EXPECT_NEAR(norm_pdf(0.0), 0.3989422804014327, 1e-15);
}

TEST(NormalPdfTest, IsSymmetric) {
    for (const double x : {0.1, 0.5, 1.0, 2.0, 3.5, 7.0}) {
        EXPECT_DOUBLE_EQ(norm_pdf(x), norm_pdf(-x)) << "x = " << x;
    }
}

TEST(NormalPdfTest, MatchesTabulatedValues) {
    // phi(1) = exp(-0.5)/sqrt(2 pi)
    EXPECT_NEAR(norm_pdf(1.0), 0.24197072451914337, 1e-15);
    // phi(2) = exp(-2)/sqrt(2 pi)
    EXPECT_NEAR(norm_pdf(2.0), 0.05399096651318806, 1e-15);
}

TEST(NormalPdfTest, UnderflowsToZeroWithoutProducingNonFinite) {
    EXPECT_DOUBLE_EQ(norm_pdf(100.0), 0.0);
    EXPECT_TRUE(std::isfinite(norm_pdf(1e300)));
}

TEST(NormalCdfTest, IsOneHalfAtZero) {
    EXPECT_DOUBLE_EQ(norm_cdf(0.0), 0.5);
}

TEST(NormalCdfTest, MatchesTabulatedValues) {
    // A&S Table 26.1. N(1) and N(2) to 15 significant figures.
    EXPECT_NEAR(norm_cdf(1.0), 0.8413447460685429, 1e-15);
    EXPECT_NEAR(norm_cdf(2.0), 0.9772498680518208, 1e-15);
    EXPECT_NEAR(norm_cdf(3.0), 0.9986501019683699, 1e-15);
    EXPECT_NEAR(norm_cdf(-1.0), 0.15865525393145705, 1e-15);
    EXPECT_NEAR(norm_cdf(-2.0), 0.022750131948179195, 1e-16);
}

TEST(NormalCdfTest, SatisfiesReflectionIdentity) {
    // N(x) + N(-x) = 1
    for (const double x : {0.0, 0.25, 1.0, 2.5, 4.0, 6.0}) {
        EXPECT_NEAR(norm_cdf(x) + norm_cdf(-x), 1.0, 1e-15) << "x = " << x;
    }
}

TEST(NormalCdfTest, IsMonotone) {
    double previous = norm_cdf(-10.0);
    for (double x = -9.9; x <= 10.0; x += 0.1) {
        const double current = norm_cdf(x);
        EXPECT_GE(current, previous) << "CDF decreased at x = " << x;
        previous = current;
    }
}

TEST(NormalCdfTest, StaysInUnitInterval) {
    for (const double x : {-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0}) {
        EXPECT_GE(norm_cdf(x), 0.0) << "x = " << x;
        EXPECT_LE(norm_cdf(x), 1.0) << "x = " << x;
    }
}

// This is the reason norm_cdf routes through erfc rather than
// 0.5*(1 + erf(x/sqrt(2))). In the left tail the answer *is* the small quantity
// that the erf form cancels away. N(d2) for a deep out-of-the-money option is
// exactly this tail probability, so the difference is a pricing difference, not
// a curiosity.
TEST(NormalCdfTest, RetainsRelativeAccuracyInTheFarLeftTail) {
    // Reference values computed from the asymptotic expansion
    //   N(-x) ~ phi(x)/x * (1 - 1/x^2 + 3/x^4 - ...)
    // and cross-checked against A&S 26.2.12. Compared in relative terms because
    // an absolute tolerance is meaningless at 1e-24.
    struct Case {
        double x;
        double expected;
    };

    constexpr Case cases[] = {
        {-5.0, 2.866515718791939e-7},
        {-8.0, 6.220960574271786e-16},
        {-10.0, 7.619853024160525e-24},
    };

    for (const auto& c : cases) {
        const double actual = norm_cdf(c.x);
        EXPECT_GT(actual, 0.0) << "tail underflowed to zero at x = " << c.x;
        EXPECT_NEAR(actual / c.expected, 1.0, 1e-12)
            << "x = " << c.x << " expected " << c.expected << " got " << actual;
    }
}

// The naive form would return exactly 1.0 here and lose the tail entirely; the
// point of this test is that the complementary side stays resolvable.
TEST(NormalCdfTest, ApproachesOneFromBelowInTheRightTail) {
    EXPECT_LT(norm_cdf(5.0), 1.0);
    EXPECT_NEAR(1.0 - norm_cdf(5.0), 2.866515718791939e-7, 1e-15);
}

TEST(NormalCdfTest, SaturatesWithoutProducingNonFinite) {
    EXPECT_DOUBLE_EQ(norm_cdf(-100.0), 0.0);
    EXPECT_DOUBLE_EQ(norm_cdf(100.0), 1.0);
    EXPECT_TRUE(std::isfinite(norm_cdf(-1e300)));
    EXPECT_TRUE(std::isfinite(norm_cdf(1e300)));
}

}  // namespace
}  // namespace diffusionworks
