#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

std::vector<double> draw_normals(RandomStream stream, int count) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        values.push_back(stream.next_normal());
    }
    return values;
}

// ---------------------------------------------------------------------------
// Reproducibility
//
// The Phase 2 exit gate: repeated fixed configurations must reproduce results.
// ---------------------------------------------------------------------------

TEST(RandomStreamTest, SameCoordinatesGiveTheSameDraws) {
    const auto first = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 100);
    const auto second = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 100);

    EXPECT_EQ(first, second);
}

TEST(RandomStreamTest, DifferentSeedsGiveDifferentDraws) {
    const auto first = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 50);
    const auto second = draw_normals(RandomStream(kSeed + 1, StreamPurpose::AssetShock, 0), 50);

    EXPECT_NE(first, second);
}

TEST(RandomStreamTest, DifferentPathsGiveDifferentDraws) {
    const auto first = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 50);
    const auto second = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 1), 50);

    EXPECT_NE(first, second);
}

// If the asset and variance streams shared draws, Heston's correlated shocks
// would be the same number rather than two correlated ones, and the correlation
// would be fictitious.
TEST(RandomStreamTest, DifferentPurposesGiveDifferentDraws) {
    const auto asset = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 50);
    const auto variance = draw_normals(RandomStream(kSeed, StreamPurpose::VarianceShock, 0), 50);

    EXPECT_NE(asset, variance);
    for (std::size_t i = 0; i < asset.size(); ++i) {
        EXPECT_NE(asset[i], variance[i]) << "streams coincide at draw " << i;
    }
}

// A draw is a pure function of its coordinates, so seeking to a position must
// give exactly what sequential reading gives. This is what makes a path
// resumable and, in Phase 12, schedule-independent.
TEST(RandomStreamTest, SeekAgreesWithSequentialReading) {
    const auto sequential = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 3), 40);

    for (std::uint64_t position = 0; position < 40; ++position) {
        RandomStream stream(kSeed, StreamPurpose::AssetShock, 3);
        stream.seek(position);
        EXPECT_DOUBLE_EQ(stream.next_normal(), sequential[position])
            << "seek to " << position << " disagrees with sequential reading";
    }
}

TEST(RandomStreamTest, ResetReturnsToTheStart) {
    RandomStream stream(kSeed, StreamPurpose::AssetShock, 0);
    const double first = stream.next_normal();
    for (int i = 0; i < 10; ++i) {
        (void)stream.next_normal();
    }

    stream.reset();
    EXPECT_DOUBLE_EQ(stream.next_normal(), first);
    EXPECT_EQ(stream.position(), 1U);
}

TEST(RandomStreamTest, PositionTracksDraws) {
    RandomStream stream(kSeed, StreamPurpose::AssetShock, 0);
    EXPECT_EQ(stream.position(), 0U);

    for (std::uint64_t i = 1; i <= 10; ++i) {
        (void)stream.next_normal();
        EXPECT_EQ(stream.position(), i);
    }
}

// Draws must not depend on the block boundary. Philox emits four words at a
// time, and an off-by-one in the buffer would show up exactly here, at the
// seam -- while leaving the moments untouched.
TEST(RandomStreamTest, DrawsAreContinuousAcrossBufferBoundaries) {
    const auto reference = draw_normals(RandomStream(kSeed, StreamPurpose::AssetShock, 0), 32);

    // Read the same positions one at a time, forcing a fresh block for each.
    for (std::uint64_t position = 0; position < 32; ++position) {
        RandomStream stream(kSeed, StreamPurpose::AssetShock, 0);
        stream.seek(position);
        EXPECT_DOUBLE_EQ(stream.next_normal(), reference[position]) << "at position " << position;
    }
}

// The design's central claim, tested directly: a path's draws cannot depend on
// how work was scheduled. Reading the same coordinates from several threads must
// give identical values, because there is no shared state to race on.
TEST(RandomStreamTest, DrawsAreIdenticalAcrossThreads) {
    constexpr int kPaths = 64;
    constexpr int kDraws = 16;

    std::vector<std::vector<double>> single_threaded;
    single_threaded.reserve(kPaths);
    for (int path = 0; path < kPaths; ++path) {
        single_threaded.push_back(draw_normals(
            RandomStream(kSeed, StreamPurpose::AssetShock, static_cast<std::uint64_t>(path)),
            kDraws));
    }

    std::vector<std::vector<double>> multi_threaded(kPaths);
    {
        std::vector<std::thread> workers;
        constexpr int kThreads = 8;
        workers.reserve(kThreads);
        for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
            workers.emplace_back([&, thread_index] {
                for (int path = thread_index; path < kPaths; path += kThreads) {
                    multi_threaded[static_cast<std::size_t>(path)] = draw_normals(
                        RandomStream(
                            kSeed, StreamPurpose::AssetShock, static_cast<std::uint64_t>(path)),
                        kDraws);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    EXPECT_EQ(single_threaded, multi_threaded)
        << "a path's draws changed with the thread count, which the counter-based design "
           "exists to make impossible";
}

// ---------------------------------------------------------------------------
// Uniform transform
// ---------------------------------------------------------------------------

// Both endpoints must be excluded. u = 0 maps to -infinity under the inverse
// CDF, so a single unlucky draw would poison a path with a non-finite state.
TEST(UniformTransformTest, ExcludesBothEndpoints) {
    EXPECT_GT(uniform_from_bits(0), 0.0) << "all-zero bits produced exactly 0";
    EXPECT_LT(uniform_from_bits(~std::uint64_t{0}), 1.0) << "all-one bits produced exactly 1";

    // Extremes stay strictly inside, so the inverse CDF is always finite.
    EXPECT_TRUE(std::isfinite(inverse_norm_cdf(uniform_from_bits(0))));
    EXPECT_TRUE(std::isfinite(inverse_norm_cdf(uniform_from_bits(~std::uint64_t{0}))));
}

TEST(UniformTransformTest, IsMonotoneInItsBits) {
    // Only the top 52 bits survive the shift, so bits below position 12 cannot
    // move the result; above it, each higher bit must strictly increase it.
    double previous = uniform_from_bits(0);
    for (int shift = 12; shift < 64; ++shift) {
        const double current = uniform_from_bits(std::uint64_t{1} << shift);
        EXPECT_GT(current, previous) << "shift " << shift;
        previous = current;
    }
}

// The endpoint bug this transform exists to avoid, pinned as a regression.
//
// The natural 53-bit form, ((bits >> 11) + 0.5) * 2^-53, returns exactly 1.0 for
// all-ones input: 2^53 - 1 + 0.5 is unrepresentable and rounds up to 2^53. Once
// in 2^53 draws it would hand inverse_norm_cdf a 1.0 and produce +infinity.
TEST(UniformTransformTest, AllOnesDoesNotReachOne) {
    const double u = uniform_from_bits(~std::uint64_t{0});

    EXPECT_LT(u, 1.0) << "the transform returned 1.0, which maps to +infinity";
    EXPECT_GT(u, 1.0 - 1e-15);
    EXPECT_TRUE(std::isfinite(inverse_norm_cdf(u)));

    // Demonstrates the defect rather than asserting it from theory.
    const double naive = (static_cast<double>((~std::uint64_t{0}) >> 11U) + 0.5) * 0x1.0p-53;
    EXPECT_DOUBLE_EQ(naive, 1.0)
        << "the 53-bit form was expected to round up to exactly 1.0; if it no longer does, this "
           "test no longer demonstrates the reason for using 52 bits";
}

TEST(RandomStreamTest, UniformsStayStrictlyInsideTheUnitInterval) {
    RandomStream stream(kSeed, StreamPurpose::Diagnostic, 0);

    for (int i = 0; i < 100000; ++i) {
        const double u = stream.next_uniform();
        ASSERT_GT(u, 0.0) << "draw " << i;
        ASSERT_LT(u, 1.0) << "draw " << i;
    }
}

TEST(RandomStreamTest, NormalsAreAlwaysFinite) {
    RandomStream stream(kSeed, StreamPurpose::Diagnostic, 1);

    for (int i = 0; i < 100000; ++i) {
        ASSERT_TRUE(std::isfinite(stream.next_normal())) << "draw " << i;
    }
}

// ---------------------------------------------------------------------------
// Antithetic pairing
// ---------------------------------------------------------------------------

// Exactness matters. Antithetic variates rely on the pair being a reflection of
// the same draw; an approximate negation would leave a residual that biases the
// estimator in a way no moment test would isolate.
TEST(RandomStreamTest, AntitheticNormalIsTheExactNegation) {
    RandomStream forward(kSeed, StreamPurpose::AssetShock, 0);
    RandomStream mirrored(kSeed, StreamPurpose::AssetShock, 0);

    for (int i = 0; i < 1000; ++i) {
        const double z = forward.next_normal();
        const double antithetic = mirrored.next_antithetic_normal();

        // inverse_norm_cdf is odd about u = 0.5, so inverting 1-u gives exactly
        // -inverse_norm_cdf(u) up to the rounding of 1-u itself.
        EXPECT_NEAR(antithetic, -z, 1e-12 * std::max(1.0, std::abs(z))) << "draw " << i;
    }
}

// ---------------------------------------------------------------------------
// Statistical validation
//
// These use enough samples that the acceptance bounds are far from the sampling
// noise; a flaky stochastic test is a test-design failure (TESTING-STRATEGY
// section 6), so the thresholds below are stated in standard errors rather than
// chosen to pass.
// ---------------------------------------------------------------------------

TEST(RandomStreamStatisticsTest, NormalsHaveTheRightFirstTwoMoments) {
    constexpr int kSamples = 2000000;

    OnlineMoments moments;
    RandomStream stream(kSeed, StreamPurpose::Diagnostic, 0);
    for (int i = 0; i < kSamples; ++i) {
        moments.add(stream.next_normal());
    }

    // The sample mean of N standard normals has standard error 1/sqrt(N)
    // = 7.1e-4 here, so 5 standard errors is 3.5e-3. A generator biased enough
    // to matter for pricing would fail this by orders of magnitude.
    EXPECT_NEAR(moments.mean(), 0.0, 5.0 / std::sqrt(static_cast<double>(kSamples)));

    const auto variance = moments.sample_variance();
    ASSERT_TRUE(variance.ok()) << variance.error().describe();
    // The sample variance of normals has standard error sqrt(2/N).
    EXPECT_NEAR(variance.value(), 1.0, 5.0 * std::sqrt(2.0 / static_cast<double>(kSamples)));
}

TEST(RandomStreamStatisticsTest, NormalsHaveTheRightShape) {
    constexpr int kSamples = 2000000;

    RandomStream stream(kSeed, StreamPurpose::Diagnostic, 1);
    double sum_cubed = 0.0;
    double sum_fourth = 0.0;

    for (int i = 0; i < kSamples; ++i) {
        const double z = stream.next_normal();
        const double squared = z * z;
        sum_cubed += squared * z;
        sum_fourth += squared * squared;
    }

    const auto n = static_cast<double>(kSamples);
    const double skewness = sum_cubed / n;
    const double kurtosis = sum_fourth / n;

    // Skewness 0 and kurtosis 3 for a standard normal. Third and fourth moments
    // catch shape errors that the mean and variance cannot: a mixture, a
    // truncated tail, or a Box-Muller sign error can all leave the first two
    // moments intact.
    EXPECT_NEAR(skewness, 0.0, 5.0 * std::sqrt(6.0 / n));
    EXPECT_NEAR(kurtosis, 3.0, 5.0 * std::sqrt(24.0 / n));
}

// A weak generator can have correct moments while its draws remain serially
// dependent, which would make paths correlated and every error estimate wrong.
TEST(RandomStreamStatisticsTest, ConsecutiveDrawsAreUncorrelated) {
    constexpr int kSamples = 1000000;

    OnlineCovariance covariance;
    RandomStream stream(kSeed, StreamPurpose::Diagnostic, 2);

    double previous = stream.next_normal();
    for (int i = 0; i < kSamples; ++i) {
        const double current = stream.next_normal();
        covariance.add(previous, current);
        previous = current;
    }

    const auto correlation = covariance.correlation();
    ASSERT_TRUE(correlation.ok()) << correlation.error().describe();
    // Sample correlation of independent series has standard error ~1/sqrt(N).
    EXPECT_NEAR(correlation.value(), 0.0, 5.0 / std::sqrt(static_cast<double>(kSamples)));
}

// Independence *across* paths is what the whole partitioning rests on. If paths
// 0 and 1 were correlated, averaging over paths would not reduce variance at the
// advertised rate and every confidence interval would be too narrow.
TEST(RandomStreamStatisticsTest, DifferentPathsAreUncorrelated) {
    constexpr int kSamples = 500000;

    for (const std::uint64_t other_path :
         {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{100}}) {
        OnlineCovariance covariance;
        RandomStream first(kSeed, StreamPurpose::Diagnostic, 0);
        RandomStream second(kSeed, StreamPurpose::Diagnostic, other_path);

        for (int i = 0; i < kSamples; ++i) {
            covariance.add(first.next_normal(), second.next_normal());
        }

        const auto correlation = covariance.correlation();
        ASSERT_TRUE(correlation.ok()) << correlation.error().describe();
        EXPECT_NEAR(correlation.value(), 0.0, 5.0 / std::sqrt(static_cast<double>(kSamples)))
            << "path 0 correlates with path " << other_path;
    }
}

// Adjacent path indices are the dangerous case: paths are numbered
// consecutively, so a generator that mixed the path word weakly would correlate
// neighbours specifically.
TEST(RandomStreamStatisticsTest, AdjacentPathsAreUncorrelated) {
    constexpr int kSamples = 200000;

    for (std::uint64_t path = 0; path < 8; ++path) {
        OnlineCovariance covariance;
        RandomStream first(kSeed, StreamPurpose::Diagnostic, path);
        RandomStream second(kSeed, StreamPurpose::Diagnostic, path + 1);

        for (int i = 0; i < kSamples; ++i) {
            covariance.add(first.next_normal(), second.next_normal());
        }

        const auto correlation = covariance.correlation();
        ASSERT_TRUE(correlation.ok()) << correlation.error().describe();
        EXPECT_NEAR(correlation.value(), 0.0, 5.0 / std::sqrt(static_cast<double>(kSamples)))
            << "paths " << path << " and " << path + 1 << " correlate";
    }
}

// ---------------------------------------------------------------------------
// Correlated normals
// ---------------------------------------------------------------------------

TEST(CorrelatedNormalsTest, RecoversTheRequestedCorrelation) {
    constexpr int kSamples = 1000000;

    for (const double rho : {-0.9, -0.5, 0.0, 0.3, 0.7, 0.95}) {
        OnlineCovariance covariance;
        RandomStream asset(kSeed, StreamPurpose::AssetShock, 0);
        RandomStream variance(kSeed, StreamPurpose::VarianceShock, 0);

        for (int i = 0; i < kSamples; ++i) {
            const CorrelatedNormals pair =
                correlate(asset.next_normal(), variance.next_normal(), rho);
            covariance.add(pair.first, pair.second);
        }

        const auto correlation = covariance.correlation();
        ASSERT_TRUE(correlation.ok()) << correlation.error().describe();

        // The sample correlation has standard error (1-rho^2)/sqrt(N).
        const double standard_error = (1.0 - rho * rho) / std::sqrt(static_cast<double>(kSamples));
        EXPECT_NEAR(correlation.value(), rho, 5.0 * std::max(standard_error, 1e-6))
            << "requested rho = " << rho;
    }
}

// Both components must remain standard normal. A Cholesky slip would leave the
// correlation right while rescaling the second marginal, which would silently
// change the variance of the process it drives.
TEST(CorrelatedNormalsTest, BothMarginalsRemainStandardNormal) {
    constexpr int kSamples = 1000000;
    constexpr double rho = 0.7;

    OnlineMoments first_moments;
    OnlineMoments second_moments;
    RandomStream asset(kSeed, StreamPurpose::AssetShock, 5);
    RandomStream variance(kSeed, StreamPurpose::VarianceShock, 5);

    for (int i = 0; i < kSamples; ++i) {
        const CorrelatedNormals pair = correlate(asset.next_normal(), variance.next_normal(), rho);
        first_moments.add(pair.first);
        second_moments.add(pair.second);
    }

    const auto n = static_cast<double>(kSamples);
    EXPECT_NEAR(first_moments.mean(), 0.0, 5.0 / std::sqrt(n));
    EXPECT_NEAR(second_moments.mean(), 0.0, 5.0 / std::sqrt(n));
    EXPECT_NEAR(first_moments.sample_variance().value(), 1.0, 5.0 * std::sqrt(2.0 / n));
    EXPECT_NEAR(second_moments.sample_variance().value(), 1.0, 5.0 * std::sqrt(2.0 / n));
}

// The first component passes through untouched, so switching a model from
// independent to correlated shocks leaves the asset's own draws alone and the
// two runs stay comparable.
TEST(CorrelatedNormalsTest, FirstComponentIsUnchanged) {
    for (const double rho : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
        const CorrelatedNormals pair = correlate(1.234, -0.567, rho);
        EXPECT_DOUBLE_EQ(pair.first, 1.234) << "rho = " << rho;
    }
}

TEST(CorrelatedNormalsTest, HandlesPerfectCorrelation) {
    const CorrelatedNormals positive = correlate(1.5, -2.0, 1.0);
    EXPECT_DOUBLE_EQ(positive.first, 1.5);
    EXPECT_DOUBLE_EQ(positive.second, 1.5) << "rho = 1 must make the components identical";

    const CorrelatedNormals negative = correlate(1.5, -2.0, -1.0);
    EXPECT_DOUBLE_EQ(negative.second, -1.5) << "rho = -1 must make them exact opposites";
}

// Rounding can push 1 - rho^2 marginally negative at |rho| = 1, and an
// unguarded sqrt would return NaN and poison every path that touched it.
TEST(CorrelatedNormalsTest, StaysFiniteAtTheCorrelationBoundary) {
    for (const double rho : {-1.0, 1.0, -0.9999999999999999, 0.9999999999999999}) {
        const CorrelatedNormals pair = correlate(0.5, 0.5, rho);
        EXPECT_TRUE(std::isfinite(pair.first)) << "rho = " << rho;
        EXPECT_TRUE(std::isfinite(pair.second)) << "rho = " << rho;
    }
}

TEST(CorrelatedNormalsTest, ZeroCorrelationPassesBothThrough) {
    const CorrelatedNormals pair = correlate(1.5, -2.5, 0.0);
    EXPECT_DOUBLE_EQ(pair.first, 1.5);
    EXPECT_DOUBLE_EQ(pair.second, -2.5);
}

}  // namespace
}  // namespace diffusionworks
