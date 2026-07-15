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

// ---------------------------------------------------------------------------
// Direct tail behaviour
//
// The erfc formulation is load-bearing for the pricing engine: N(d2) for a deep
// out-of-the-money option *is* the tail probability being priced, so a CDF that
// loses the tail loses the price. These tests exercise that region directly
// rather than inferring it from a price, and they are what would fail first if
// norm_cdf were ever rewritten in the algebraically equivalent but numerically
// cancelling form 0.5*(1 + erf(x/sqrt(2))).
// ---------------------------------------------------------------------------

// Monotone across the whole tail, including where values are subnormal. A
// non-monotone CDF would break every root-finder and bracketing argument built
// on it, and cancellation typically shows up as a flat or jittering region long
// before it shows up as an obviously wrong number.
TEST(NormalCdfTailTest, IsStrictlyMonotoneThroughTheLeftTail) {
    double previous = norm_cdf(-37.0);
    EXPECT_GE(previous, 0.0);

    for (double x = -37.0; x <= -1.0; x += 0.05) {
        const double current = norm_cdf(x);
        EXPECT_GE(current, previous) << "CDF decreased at x = " << x;
        EXPECT_TRUE(std::isfinite(current)) << "non-finite at x = " << x;
        previous = current;
    }
}

TEST(NormalCdfTailTest, IsStrictlyMonotoneThroughTheRightTail) {
    double previous = norm_cdf(1.0);

    for (double x = 1.0; x <= 37.0; x += 0.05) {
        const double current = norm_cdf(x);
        EXPECT_GE(current, previous) << "CDF decreased at x = " << x;
        EXPECT_LE(current, 1.0) << "CDF exceeded 1 at x = " << x;
        previous = current;
    }
}

// N(x) + N(-x) = 1 must hold deep into both tails. The naive form satisfies this
// trivially by returning 0 and 1, so this test is paired with the tail-value
// tests below: together they require the identity to hold *and* the small side to
// remain resolvable.
TEST(NormalCdfTailTest, ComplementIdentityHoldsIntoBothTails) {
    for (double x = 0.0; x <= 40.0; x += 0.25) {
        const double sum = norm_cdf(x) + norm_cdf(-x);
        EXPECT_NEAR(sum, 1.0, 1e-15) << "complement identity broke at x = " << x;
    }
}

// The left tail must stay strictly positive as far as double can represent it.
// N(-37) is about 5.7e-300, still a normal double; if this returns zero the CDF
// has lost roughly 290 orders of magnitude of information.
TEST(NormalCdfTailTest, LeftTailStaysPositiveWhileRepresentable) {
    for (const double x : {-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -37.0}) {
        EXPECT_GT(norm_cdf(x), 0.0) << "tail collapsed to zero at x = " << x;
        EXPECT_TRUE(std::isfinite(norm_cdf(x))) << "x = " << x;
    }
}

// Below roughly x = -38.5 the true value drops under the smallest subnormal
// double (~4.9e-324) and zero becomes the correctly rounded answer. Underflow
// must be graceful: zero, never negative and never NaN.
TEST(NormalCdfTailTest, UnderflowsGracefullyBeyondRepresentableRange) {
    for (const double x : {-40.0, -50.0, -100.0, -1e10, -1e300}) {
        const double value = norm_cdf(x);
        EXPECT_TRUE(std::isfinite(value)) << "x = " << x;
        EXPECT_GE(value, 0.0) << "underflow produced a negative probability at x = " << x;
        EXPECT_DOUBLE_EQ(value, 0.0) << "x = " << x;
    }
}

// The right tail saturates at exactly 1, and that is correct: past x ~ 8.3 the
// true value is within half an ulp of 1, so 1.0 is the nearest double. Nothing is
// wrong here, and nothing can be done about it -- N(x) simply stops carrying tail
// information.
TEST(NormalCdfTailTest, RightTailSaturatesAtOne) {
    // Still resolvable just below the boundary.
    EXPECT_LT(norm_cdf(8.0), 1.0);
    EXPECT_LT(norm_cdf(8.2), 1.0);

    // Saturated from ~8.3 on. The spacing of doubles near 1 is 2.2e-16, while
    // N(-8.3) ~ 5.2e-17, so no double lies between N(8.3) and 1.
    for (const double x : {8.3, 9.0, 10.0, 30.0, 40.0, 100.0, 1e10, 1e300}) {
        const double value = norm_cdf(x);
        EXPECT_TRUE(std::isfinite(value)) << "x = " << x;
        EXPECT_DOUBLE_EQ(value, 1.0) << "x = " << x;
    }
}

// The consequence, and the reason the pricing formulas never write 1 - N(d).
//
// Past saturation, 1 - N(x) is exactly zero while the same probability read from
// the mirrored tail, N(-x), is still resolvable 180 orders of magnitude further
// down. The degradation starts well before saturation: at x = 8 the subtraction
// is already ~7% wrong.
//
// This is why the put price uses N(-d2), N(-d1) and put delta uses -e^{-qT}N(-d1)
// rather than their algebraically identical complements. The numbers below are
// measured, not asserted from theory.
TEST(NormalCdfTailTest, ComplementBySubtractionLosesTheTailThatMirroringKeeps) {
    // Reference values from the asymptotic expansion, cross-checked against
    // A&S 26.2.12.
    struct Case {
        double x;
        double expected_tail;
    };

    constexpr Case cases[] = {
        {8.0, 6.220960574271786e-16},
        {10.0, 7.619853024160525e-24},
        {30.0, 4.906713927148187e-198},
    };

    for (const auto& c : cases) {
        const double by_subtraction = 1.0 - norm_cdf(c.x);
        const double by_mirroring = norm_cdf(-c.x);

        EXPECT_NEAR(by_mirroring / c.expected_tail, 1.0, 1e-12)
            << "the mirrored tail must stay accurate at x = " << c.x;

        // The subtraction is never more accurate, and is usually catastrophically
        // worse.
        const double mirror_error = std::abs(by_mirroring - c.expected_tail) / c.expected_tail;
        const double subtraction_error =
            std::abs(by_subtraction - c.expected_tail) / c.expected_tail;
        EXPECT_LE(mirror_error, subtraction_error) << "x = " << c.x;
    }

    // At x = 8, before saturation, the subtraction already carries several
    // percent relative error: 6.66e-16 against a true 6.22e-16.
    EXPECT_GT(std::abs((1.0 - norm_cdf(8.0)) - 6.220960574271786e-16) / 6.220960574271786e-16, 0.05)
        << "expected the subtraction to be materially wrong at x = 8";

    // Past saturation it returns exactly zero while mirroring still resolves the
    // value.
    EXPECT_DOUBLE_EQ(1.0 - norm_cdf(30.0), 0.0);
    EXPECT_GT(norm_cdf(-30.0), 0.0);
}

// Extreme finite inputs, both signs, must produce finite in-range values rather
// than NaN from an intermediate overflow.
TEST(NormalCdfTailTest, ExtremeInputsStayInRange) {
    constexpr double kLargest = std::numeric_limits<double>::max();

    for (const double x : {-kLargest, -1e300, -1e100, 1e100, 1e300, kLargest}) {
        const double cdf = norm_cdf(x);
        EXPECT_TRUE(std::isfinite(cdf)) << "norm_cdf not finite at x = " << x;
        EXPECT_GE(cdf, 0.0) << "x = " << x;
        EXPECT_LE(cdf, 1.0) << "x = " << x;

        const double pdf = norm_pdf(x);
        EXPECT_TRUE(std::isfinite(pdf)) << "norm_pdf not finite at x = " << x;
        EXPECT_GE(pdf, 0.0) << "x = " << x;
    }
}

// A reference implemented as 0.5*(1 + erf(x/sqrt(2))) is compared against the
// erfc form to record, in the test suite itself, exactly what the choice buys.
// This is the regression that would fire if someone "simplified" norm_cdf.
TEST(NormalCdfTailTest, ErfcFormOutperformsTheCancellingAlternative) {
    const auto cancelling_form = [](double x) { return 0.5 * (1.0 + std::erf(x * kInvSqrt2)); };

    // Near the centre the two agree: there is nothing to cancel.
    for (const double x : {-1.0, 0.0, 1.0}) {
        EXPECT_NEAR(norm_cdf(x), cancelling_form(x), 1e-15) << "x = " << x;
    }

    // In the tail the alternative collapses to exactly zero while the erfc form
    // still resolves the value. This is the defect, demonstrated rather than
    // asserted.
    EXPECT_DOUBLE_EQ(cancelling_form(-10.0), 0.0);
    EXPECT_GT(norm_cdf(-10.0), 0.0);
    EXPECT_NEAR(norm_cdf(-10.0) / 7.619853024160525e-24, 1.0, 1e-12);
}

}  // namespace
}  // namespace diffusionworks
