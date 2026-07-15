#include <diffusionworks/core/error.hpp>
#include <diffusionworks/numerics/normal.hpp>
#include <diffusionworks/statistics/distributions.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <string>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Validation against high-precision references.
//
// data/references/distributions.json is computed with mpmath at 50 digits from
// the definitions -- root-finding on the CDF for the inverse normal, the
// regularized incomplete beta for Student-t -- so it is independent of the
// algorithms used here (AS 241 and a continued fraction). A reference that
// shared those approximations would confirm the transcription while missing an
// error in the method.
//
// The fixture generator also cross-checks against scipy and records the result.
// Notably scipy's t quantile agrees only to ~4e-11, which is scipy's own
// inversion accuracy: mpmath is the oracle here, not scipy.
// ---------------------------------------------------------------------------

class DistributionReferenceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::filesystem::path path =
            std::filesystem::path(DW_TEST_DATA_DIR) / "references" / "distributions.json";

        std::ifstream stream(path);
        ASSERT_TRUE(stream.is_open())
            << "reference fixture not found: " << path
            << "\nregenerate with: python3 python/generate_distribution_references.py > "
               "data/references/distributions.json";

        ASSERT_NO_THROW(document_ = nlohmann::json::parse(stream));
    }

    static double reference(const nlohmann::json& value) {
        return std::stod(value.get<std::string>());
    }

    static void expect_close(double actual,
                             double expected,
                             double rel_tolerance,
                             double abs_tolerance,
                             const std::string& what) {
        const double allowed = abs_tolerance + rel_tolerance * std::abs(expected);
        EXPECT_NEAR(actual, expected, allowed)
            << what << "\n  engine    : " << actual << "\n  reference : " << expected
            << "\n  allowed   : " << allowed;
    }

    static nlohmann::json document_;
};

nlohmann::json DistributionReferenceTest::document_;

TEST_F(DistributionReferenceTest, FixtureRecordsItsProvenance) {
    ASSERT_TRUE(document_.contains("provenance"));
    EXPECT_GE(document_["provenance"]["mpmath_working_digits"].get<int>(), 30);
    EXPECT_TRUE(document_["provenance"].contains("cross_checked_against"));
}

TEST_F(DistributionReferenceTest, InverseNormalCdfMatchesTheReference) {
    ASSERT_FALSE(document_["inverse_norm_cdf"].empty());

    for (const auto& c : document_["inverse_norm_cdf"]) {
        const double p = c["p"].get<double>();
        const double expected = reference(c["expected"]);

        // AS 241 claims about 1e-16 relative accuracy. 1e-14 leaves two orders
        // of margin for the rounding of the rational evaluation itself, and is
        // still far tighter than any real transcription error.
        expect_close(inverse_norm_cdf(p),
                     expected,
                     1e-14,
                     1e-15,
                     "inverse_norm_cdf(" + std::to_string(p) + ")");
    }
}

// The tails are the point. A rational approximation degrades there, and it is
// exactly where a deep out-of-the-money path lives, so a fixture of central
// values would pass while the far quantiles were wrong.
TEST_F(DistributionReferenceTest, InverseNormalCdfIsAccurateInTheFarTails) {
    int tail_cases = 0;

    for (const auto& c : document_["inverse_norm_cdf"]) {
        const double p = c["p"].get<double>();
        if (p > 1e-6 && p < 1.0 - 1e-6) {
            continue;
        }
        ++tail_cases;
        expect_close(inverse_norm_cdf(p),
                     reference(c["expected"]),
                     1e-14,
                     1e-15,
                     "far tail at p = " + std::to_string(p));
    }

    EXPECT_GE(tail_cases, 6) << "the fixture does not exercise the tails";
}

TEST_F(DistributionReferenceTest, StudentTCdfMatchesTheReference) {
    ASSERT_FALSE(document_["student_t_cdf"].empty());

    for (const auto& c : document_["student_t_cdf"]) {
        const double t = c["t"].get<double>();
        const double nu = c["degrees_of_freedom"].get<double>();

        const auto actual = student_t_cdf(t, nu);
        ASSERT_TRUE(actual.ok()) << actual.error().describe();

        // 1e-11 rather than machine precision, because accuracy degrades with
        // nu in a way that is measured and documented rather than absorbed --
        // see StudentTAccuracyTest below for the reason and the shape.
        expect_close(actual.value(),
                     reference(c["expected"]),
                     1e-11,
                     1e-15,
                     "student_t_cdf(" + std::to_string(t) + ", " + std::to_string(nu) + ")");
    }
}

// ---------------------------------------------------------------------------
// A measured limitation, recorded rather than hidden behind a loose tolerance.
//
// Accuracy is ~1e-15 up to nu = 100 and degrades to ~5e-12 by nu = 10000. The
// cause is cancellation in the prefactor lgamma(a+b) - lgamma(a) - lgamma(b).
// At nu = 10000 the individual terms are ~37582 and carry a relative error of
// ~1e-16, hence an absolute error of ~4e-12; their difference is only ~3.6, so
// that absolute error becomes a ~1e-12 relative error, which exp() then carries
// straight through.
//
// This is immaterial in use: a confidence interval's half-width is itself a
// statistical estimate with ~1/sqrt(2n) relative uncertainty -- around 0.2% at
// 100,000 paths -- so a 1e-12 error in the critical value is nine orders of
// magnitude below the noise it multiplies. It is asserted here anyway, so the
// limitation is a measurement rather than an assumption, and so that a
// regression which made it worse would be caught.
// ---------------------------------------------------------------------------

TEST_F(DistributionReferenceTest, StudentTAccuracyDegradesPredictablyWithDegreesOfFreedom) {
    struct Band {
        double max_degrees_of_freedom;
        double tolerance;
    };

    // Measured worst-case relative error per band, with roughly an order of
    // magnitude of headroom.
    constexpr Band bands[] = {
        {100.0, 1e-13},
        {1000.0, 1e-11},
        {100000.0, 1e-10},
    };

    for (const auto& c : document_["student_t_cdf"]) {
        const double expected = reference(c["expected"]);
        if (expected == 0.0 || expected == 1.0) {
            continue;
        }
        const double nu = c["degrees_of_freedom"].get<double>();

        double tolerance = 1e-10;
        for (const auto& band : bands) {
            if (nu <= band.max_degrees_of_freedom) {
                tolerance = band.tolerance;
                break;
            }
        }

        const auto actual = student_t_cdf(c["t"].get<double>(), nu);
        ASSERT_TRUE(actual.ok()) << actual.error().describe();

        const double relative = std::abs(actual.value() - expected) / std::abs(expected);
        EXPECT_LT(relative, tolerance)
            << "student_t_cdf accuracy at nu = " << nu << " is worse than measured: " << relative;
    }
}

TEST_F(DistributionReferenceTest, StudentTQuantileMatchesTheReference) {
    ASSERT_FALSE(document_["student_t_quantile"].empty());

    for (const auto& c : document_["student_t_quantile"]) {
        const double p = c["p"].get<double>();
        const double nu = c["degrees_of_freedom"].get<double>();

        const auto actual = student_t_quantile(p, nu);
        ASSERT_TRUE(actual.ok()) << actual.error().describe();

        // Bounded by the bisection bracket (1e-13 relative) and, at large nu,
        // by the CDF's own lgamma cancellation; see
        // StudentTAccuracyDegradesPredictablyWithDegreesOfFreedom.
        expect_close(actual.value(),
                     reference(c["expected"]),
                     1e-10,
                     1e-13,
                     "student_t_quantile(" + std::to_string(p) + ", " + std::to_string(nu) + ")");
    }
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

// Round-tripping x -> N(x) -> Phi^-1 recovers x, but only to the accuracy the
// intermediate probability can carry, and that accuracy is wildly asymmetric.
//
// The error is amplified by dx/dp = 1/phi(x). In the left tail N(x) is a small
// number held with full relative precision, so its ULP is tiny and the round trip
// is near-exact. In the right tail N(x) approaches 1, its ULP is a fixed 2.2e-16
// regardless of how close to 1 it is, and dividing that by phi(6) = 6.1e-9
// inflates it to ~3e-8.
//
// So the tolerance is derived from the conditioning rather than picked. A single
// fixed bound would either be too loose to mean anything at x = -6 or fail at
// x = +6 for reasons that have nothing to do with this function. The asymmetry is
// the same one that makes the pricing formulas evaluate N(-d) instead of
// 1 - N(d).
TEST(InverseNormalCdfTest, InvertsTheCdfToTheAccuracyTheProbabilityCarries) {
    for (const double x : {-6.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 6.0}) {
        const double p = norm_cdf(x);

        // Error propagated from p's own resolution: ulp(p) amplified by dx/dp.
        const double p_ulp = std::nextafter(p, 2.0) - p;
        const double density = norm_pdf(x);
        const double propagated = 8.0 * p_ulp / density;

        // Floor at x's own resolution. In the left tail the propagated bound is
        // smaller than the spacing of doubles near x -- at x = -6 it comes out at
        // 2.7e-16 against an ulp of 8.9e-16 -- and no round trip can land nearer
        // than the grid it returns to. Without this the test would be demanding
        // exact equality while appearing to allow a tolerance.
        const double representable =
            8.0 * std::numeric_limits<double>::epsilon() * std::max(std::abs(x), 1.0);

        const double tolerance = std::max(propagated, representable);

        EXPECT_NEAR(inverse_norm_cdf(p), x, tolerance)
            << "x = " << x << "\n  p           = " << p << "\n  ulp(p)      = " << p_ulp
            << "\n  1/phi(x)    = " << 1.0 / density << "\n  propagated  = " << propagated
            << "\n  representable = " << representable << "\n  tolerance   = " << tolerance;
    }
}

// The left tail, where the round trip really is near-exact. This is the half that
// matters for pricing, and holding it to 1e-13 rather than to the conditioned
// bound above keeps the test sharp.
TEST(InverseNormalCdfTest, RoundTripIsNearExactInTheLeftTail) {
    for (const double x : {-8.0, -6.0, -4.0, -2.0, -1.0}) {
        EXPECT_NEAR(inverse_norm_cdf(norm_cdf(x)), x, 1e-13 * std::abs(x)) << "x = " << x;
    }
}

// Phi^-1(p) = -Phi^-1(1-p) exactly in real arithmetic, but `1 - p` is not exact
// in floating point for small p: it loses every digit below 1e-16 absolute. At
// p = 1e-10 the subtraction perturbs the argument by ~5e-17, which the quantile's
// 4e9 sensitivity turns into ~2e-7 -- so testing oddness there would be measuring
// the subtraction, not the function.
//
// Restricted to p where 1-p is well conditioned, and held tight there.
TEST(InverseNormalCdfTest, IsOddAboutOneHalf) {
    for (const double p : {0.01, 0.1, 0.25, 0.4, 0.499}) {
        const double lower = inverse_norm_cdf(p);
        const double upper = inverse_norm_cdf(1.0 - p);
        EXPECT_NEAR(lower, -upper, 1e-12 * std::max(1.0, std::abs(lower))) << "p = " << p;
    }
}

TEST(InverseNormalCdfTest, IsZeroAtOneHalf) {
    EXPECT_DOUBLE_EQ(inverse_norm_cdf(0.5), 0.0);
}

// Monotonicity is what makes the antithetic map u -> 1-u a genuine reflection.
TEST(InverseNormalCdfTest, IsMonotone) {
    double previous = inverse_norm_cdf(1e-12);
    for (double p = 1e-4; p < 1.0; p += 1e-4) {
        const double current = inverse_norm_cdf(p);
        ASSERT_GE(current, previous) << "not monotone at p = " << p;
        previous = current;
    }
}

TEST(InverseNormalCdfTest, ReturnsInfinityAtTheEndpoints) {
    EXPECT_EQ(inverse_norm_cdf(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(inverse_norm_cdf(1.0), std::numeric_limits<double>::infinity());
}

TEST(InverseNormalCdfTest, PropagatesNaNAndRejectsOutOfRange) {
    EXPECT_TRUE(std::isnan(inverse_norm_cdf(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(inverse_norm_cdf(-0.5)));
    EXPECT_TRUE(std::isnan(inverse_norm_cdf(1.5)));
}

// ---------------------------------------------------------------------------
// Incomplete beta
// ---------------------------------------------------------------------------

TEST(IncompleteBetaTest, MatchesKnownClosedForms) {
    // I_x(1,1) = x, since Beta(1,1) is uniform.
    for (const double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const auto result = regularized_incomplete_beta(1.0, 1.0, x);
        ASSERT_TRUE(result.ok()) << result.error().describe();
        EXPECT_NEAR(result.value(), x, 1e-14) << "x = " << x;
    }

    // I_x(1,2) = 1 - (1-x)^2, and I_x(2,1) = x^2.
    const auto a = regularized_incomplete_beta(1.0, 2.0, 0.3);
    ASSERT_TRUE(a.ok());
    EXPECT_NEAR(a.value(), 1.0 - 0.7 * 0.7, 1e-14);

    const auto b = regularized_incomplete_beta(2.0, 1.0, 0.3);
    ASSERT_TRUE(b.ok());
    EXPECT_NEAR(b.value(), 0.09, 1e-14);
}

TEST(IncompleteBetaTest, SatisfiesTheSymmetryRelation) {
    // I_x(a,b) = 1 - I_{1-x}(b,a). The implementation switches between the two
    // sides depending on x, so this checks the seam.
    for (const double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        const auto forward = regularized_incomplete_beta(2.5, 3.5, x);
        const auto mirrored = regularized_incomplete_beta(3.5, 2.5, 1.0 - x);
        ASSERT_TRUE(forward.ok());
        ASSERT_TRUE(mirrored.ok());
        EXPECT_NEAR(forward.value(), 1.0 - mirrored.value(), 1e-13) << "x = " << x;
    }
}

TEST(IncompleteBetaTest, IsMonotoneAndBounded) {
    double previous = -1.0;
    for (double x = 0.0; x <= 1.0; x += 0.01) {
        const auto result = regularized_incomplete_beta(3.0, 4.0, x);
        ASSERT_TRUE(result.ok()) << "x = " << x;
        EXPECT_GE(result.value(), previous) << "not monotone at x = " << x;
        EXPECT_GE(result.value(), 0.0);
        EXPECT_LE(result.value(), 1.0);
        previous = result.value();
    }
}

TEST(IncompleteBetaTest, RejectsInvalidArguments) {
    EXPECT_FALSE(regularized_incomplete_beta(0.0, 1.0, 0.5).ok());
    EXPECT_FALSE(regularized_incomplete_beta(-1.0, 1.0, 0.5).ok());
    EXPECT_FALSE(regularized_incomplete_beta(1.0, 0.0, 0.5).ok());
    EXPECT_FALSE(regularized_incomplete_beta(1.0, 1.0, -0.1).ok());
    EXPECT_FALSE(regularized_incomplete_beta(1.0, 1.0, 1.1).ok());
    EXPECT_FALSE(
        regularized_incomplete_beta(1.0, 1.0, std::numeric_limits<double>::quiet_NaN()).ok());
}

// Large parameters overflow tgamma long before the answer stops being
// representable, which is why the prefactor is formed in logs.
TEST(IncompleteBetaTest, HandlesLargeParametersWithoutOverflow) {
    const auto result = regularized_incomplete_beta(5000.0, 0.5, 0.5);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_TRUE(std::isfinite(result.value()));
    EXPECT_GE(result.value(), 0.0);
    EXPECT_LE(result.value(), 1.0);
}

// ---------------------------------------------------------------------------
// Student-t
// ---------------------------------------------------------------------------

TEST(StudentTTest, CdfIsOneHalfAtZero) {
    for (const double nu : {1.0, 2.0, 5.0, 30.0, 1000.0}) {
        const auto result = student_t_cdf(0.0, nu);
        ASSERT_TRUE(result.ok());
        EXPECT_NEAR(result.value(), 0.5, 1e-14) << "nu = " << nu;
    }
}

TEST(StudentTTest, CdfIsSymmetric) {
    for (const double nu : {1.0, 5.0, 100.0}) {
        for (const double t : {0.5, 1.0, 2.0, 5.0}) {
            const auto positive = student_t_cdf(t, nu);
            const auto negative = student_t_cdf(-t, nu);
            ASSERT_TRUE(positive.ok());
            ASSERT_TRUE(negative.ok());
            EXPECT_NEAR(positive.value() + negative.value(), 1.0, 1e-13)
                << "t = " << t << " nu = " << nu;
        }
    }
}

// At one degree of freedom the t is Cauchy, whose CDF has a closed form. This is
// an independent check of the incomplete beta path at the hardest parameter.
TEST(StudentTTest, MatchesTheCauchyClosedFormAtOneDegreeOfFreedom) {
    for (const double t : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        const auto result = student_t_cdf(t, 1.0);
        ASSERT_TRUE(result.ok());
        const double cauchy = 0.5 + std::atan(t) / std::numbers::pi;
        EXPECT_NEAR(result.value(), cauchy, 1e-13) << "t = " << t;
    }
}

// As nu grows the t approaches the normal. This is why a normal approximation is
// adequate at production path counts -- and why it is not at small ones.
TEST(StudentTTest, ApproachesTheNormalAsDegreesOfFreedomGrow) {
    const double normal_quantile = inverse_norm_cdf(0.975);

    const auto small = student_t_quantile(0.975, 5.0);
    const auto large = student_t_quantile(0.975, 100000.0);
    ASSERT_TRUE(small.ok());
    ASSERT_TRUE(large.ok());

    // At 5 degrees of freedom the t is materially wider: 2.571 against 1.960.
    EXPECT_GT(small.value(), normal_quantile + 0.5);
    // At 1e5 it is indistinguishable.
    EXPECT_NEAR(large.value(), normal_quantile, 1e-4);
}

// The t has heavier tails than the normal at every finite nu, which is exactly
// why using the normal at small samples under-covers.
TEST(StudentTTest, QuantileAlwaysExceedsTheNormalQuantile) {
    const double normal_quantile = inverse_norm_cdf(0.975);

    for (const double nu : {1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0}) {
        const auto result = student_t_quantile(0.975, nu);
        ASSERT_TRUE(result.ok());
        EXPECT_GT(result.value(), normal_quantile) << "nu = " << nu;
    }
}

TEST(StudentTTest, QuantileInvertsTheCdf) {
    for (const double nu : {1.0, 3.0, 10.0, 500.0}) {
        for (const double p : {0.01, 0.1, 0.5, 0.9, 0.99}) {
            const auto t = student_t_quantile(p, nu);
            ASSERT_TRUE(t.ok()) << "p = " << p << " nu = " << nu;
            const auto back = student_t_cdf(t.value(), nu);
            ASSERT_TRUE(back.ok());
            EXPECT_NEAR(back.value(), p, 1e-11) << "p = " << p << " nu = " << nu;
        }
    }
}

TEST(StudentTTest, QuantileIsZeroAtOneHalf) {
    for (const double nu : {1.0, 5.0, 1000.0}) {
        const auto result = student_t_quantile(0.5, nu);
        ASSERT_TRUE(result.ok());
        EXPECT_DOUBLE_EQ(result.value(), 0.0);
    }
}

// An infinite quantile is not a useful confidence bound, so it is a failure
// rather than a returned infinity a caller might publish.
TEST(StudentTTest, QuantileRejectsTheEndpoints) {
    EXPECT_FALSE(student_t_quantile(0.0, 5.0).ok());
    EXPECT_FALSE(student_t_quantile(1.0, 5.0).ok());
    EXPECT_FALSE(student_t_quantile(-0.1, 5.0).ok());
    EXPECT_FALSE(student_t_quantile(1.1, 5.0).ok());
    EXPECT_FALSE(student_t_quantile(std::numeric_limits<double>::quiet_NaN(), 5.0).ok());
}

TEST(StudentTTest, RejectsInvalidDegreesOfFreedom) {
    EXPECT_FALSE(student_t_cdf(1.0, 0.0).ok());
    EXPECT_FALSE(student_t_cdf(1.0, -5.0).ok());
    EXPECT_FALSE(student_t_quantile(0.5, 0.0).ok());
    EXPECT_FALSE(student_t_quantile(0.5, std::numeric_limits<double>::infinity()).ok());
}

// Fractional degrees of freedom are admissible: the definition does not require
// an integer, and Welch-style effective degrees of freedom are fractional.
TEST(StudentTTest, AcceptsFractionalDegreesOfFreedom) {
    const auto result = student_t_quantile(0.975, 7.3);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_GT(result.value(), inverse_norm_cdf(0.975));
}

// Bracketing must widen rather than assume the normal quantile is adequate: at
// one degree of freedom the 0.975 quantile is 12.7 against the normal's 1.96.
TEST(StudentTTest, QuantileBracketsHeavyTails) {
    const auto result = student_t_quantile(0.975, 1.0);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_NEAR(result.value(), 12.706204736174705, 1e-9);
}

TEST(StudentTTest, CdfHandlesInfiniteArguments) {
    const auto positive = student_t_cdf(std::numeric_limits<double>::infinity(), 5.0);
    ASSERT_TRUE(positive.ok());
    EXPECT_DOUBLE_EQ(positive.value(), 1.0);

    const auto negative = student_t_cdf(-std::numeric_limits<double>::infinity(), 5.0);
    ASSERT_TRUE(negative.ok());
    EXPECT_DOUBLE_EQ(negative.value(), 0.0);
}

}  // namespace
}  // namespace diffusionworks
