#include <diffusionworks/core/error.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// OnlineMoments
// ---------------------------------------------------------------------------

TEST(OnlineMomentsTest, StartsEmpty) {
    const OnlineMoments moments;

    EXPECT_EQ(moments.count(), 0U);
    EXPECT_FALSE(moments.sample_variance().ok());
    EXPECT_FALSE(moments.standard_error().ok());
    EXPECT_FALSE(moments.confidence_interval().ok());
}

// A single observation has a mean but no dispersion. Reporting zero variance
// would claim a perfectly precise estimate from one sample, so it is a failure
// rather than a special case.
TEST(OnlineMomentsTest, VarianceIsUndefinedForOneObservation) {
    OnlineMoments moments;
    moments.add(42.0);

    EXPECT_EQ(moments.count(), 1U);
    EXPECT_DOUBLE_EQ(moments.mean(), 42.0);

    const auto variance = moments.sample_variance();
    ASSERT_FALSE(variance.ok());
    EXPECT_EQ(variance.error().code, ErrorCode::InvalidArgument);
}

TEST(OnlineMomentsTest, MatchesKnownMomentsOfASmallSample) {
    // 2, 4, 4, 4, 5, 5, 7, 9: mean 5, sample variance 32/7.
    OnlineMoments moments;
    for (const double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) {
        moments.add(x);
    }

    EXPECT_EQ(moments.count(), 8U);
    EXPECT_DOUBLE_EQ(moments.mean(), 5.0);

    const auto variance = moments.sample_variance();
    ASSERT_TRUE(variance.ok()) << variance.error().describe();
    EXPECT_NEAR(variance.value(), 32.0 / 7.0, 1e-14);
    EXPECT_NEAR(moments.standard_error().value(), std::sqrt((32.0 / 7.0) / 8.0), 1e-14);
}

TEST(OnlineMomentsTest, ConstantSampleHasZeroVariance) {
    OnlineMoments moments;
    for (int i = 0; i < 100; ++i) {
        moments.add(7.5);
    }

    EXPECT_DOUBLE_EQ(moments.mean(), 7.5);
    const auto variance = moments.sample_variance();
    ASSERT_TRUE(variance.ok());
    EXPECT_DOUBLE_EQ(variance.value(), 0.0);
}

// The reason Welford is used at all.
//
// The textbook form, (sum x^2 - n*mean^2)/(n-1), subtracts two nearly equal
// quantities. With values near 1e9 and a spread of 1, the leading digits cancel
// and the result is dominated by rounding -- it can even come out negative. This
// is not a contrived case: a discounted payoff of 10.45 with a standard error of
// 0.001 has the same structure, and every confidence interval depends on getting
// it right.
TEST(OnlineMomentsTest, StaysAccurateWhenTheMeanDwarfsTheSpread) {
    constexpr double kOffset = 1e9;
    const std::vector<double> sample{kOffset + 4.0, kOffset + 7.0, kOffset + 13.0, kOffset + 16.0};

    OnlineMoments moments;
    for (const double x : sample) {
        moments.add(x);
    }

    // Shifting by a constant leaves the variance unchanged, so the exact answer
    // is the variance of {4, 7, 13, 16}, which is 30.
    const auto variance = moments.sample_variance();
    ASSERT_TRUE(variance.ok()) << variance.error().describe();
    EXPECT_NEAR(variance.value(), 30.0, 1e-6);

    // The naive computation loses everything here. Demonstrated rather than
    // asserted, so the reason for Welford is recorded in the suite.
    double sum = 0.0;
    double sum_squares = 0.0;
    for (const double x : sample) {
        sum += x;
        sum_squares += x * x;
    }
    const double n = static_cast<double>(sample.size());
    const double naive = (sum_squares - sum * sum / n) / (n - 1.0);
    EXPECT_GT(std::abs(naive - 30.0), 1e-3)
        << "the naive form was expected to lose precision here; if it did not, this test no "
           "longer demonstrates anything";
}

// ---------------------------------------------------------------------------
// Merging
//
// ADR-011 requires thread-local accumulation and a reduction in a defined order.
// merge() is that reduction, so it must be exact rather than approximate.
// ---------------------------------------------------------------------------

TEST(OnlineMomentsTest, MergeMatchesSequentialAccumulation) {
    RandomStream stream(99, StreamPurpose::Diagnostic, 0);
    std::vector<double> sample;
    sample.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        sample.push_back(stream.next_normal() * 3.0 + 10.0);
    }

    OnlineMoments sequential;
    for (const double x : sample) {
        sequential.add(x);
    }

    // Split at several points, including lopsided ones: the merge weights by
    // count, and an equal split would not exercise that.
    for (const std::size_t split :
         {std::size_t{1}, std::size_t{7}, std::size_t{500}, std::size_t{999}}) {
        OnlineMoments left;
        OnlineMoments right;
        for (std::size_t i = 0; i < split; ++i) {
            left.add(sample[i]);
        }
        for (std::size_t i = split; i < sample.size(); ++i) {
            right.add(sample[i]);
        }
        left.merge(right);

        EXPECT_EQ(left.count(), sequential.count()) << "split at " << split;
        EXPECT_NEAR(left.mean(), sequential.mean(), 1e-12) << "split at " << split;
        EXPECT_NEAR(left.sample_variance().value(), sequential.sample_variance().value(), 1e-10)
            << "split at " << split;
    }
}

// The property the parallel reduction rests on: many partial accumulators must
// combine to the same answer as one. If this failed, the thread count would
// change the reported variance.
TEST(OnlineMomentsTest, ManyPartialsMergeToTheSameResult) {
    RandomStream stream(1234, StreamPurpose::Diagnostic, 0);
    std::vector<double> sample;
    sample.reserve(1024);
    for (int i = 0; i < 1024; ++i) {
        sample.push_back(stream.next_normal());
    }

    OnlineMoments sequential;
    for (const double x : sample) {
        sequential.add(x);
    }

    for (const std::size_t partitions : {2U, 4U, 8U, 16U, 64U}) {
        std::vector<OnlineMoments> partials(partitions);
        for (std::size_t i = 0; i < sample.size(); ++i) {
            partials[i % partitions].add(sample[i]);
        }

        // Reduced in index order, which is what makes the result deterministic.
        OnlineMoments combined;
        for (const OnlineMoments& partial : partials) {
            combined.merge(partial);
        }

        EXPECT_EQ(combined.count(), sequential.count()) << partitions << " partitions";
        EXPECT_NEAR(combined.mean(), sequential.mean(), 1e-12) << partitions << " partitions";
        EXPECT_NEAR(combined.sample_variance().value(), sequential.sample_variance().value(), 1e-11)
            << partitions << " partitions";
    }
}

TEST(OnlineMomentsTest, MergingAnEmptyAccumulatorChangesNothing) {
    OnlineMoments moments;
    moments.add(1.0);
    moments.add(2.0);
    moments.add(3.0);

    const double mean_before = moments.mean();
    const double variance_before = moments.sample_variance().value();

    moments.merge(OnlineMoments{});

    EXPECT_EQ(moments.count(), 3U);
    EXPECT_DOUBLE_EQ(moments.mean(), mean_before);
    EXPECT_DOUBLE_EQ(moments.sample_variance().value(), variance_before);
}

TEST(OnlineMomentsTest, MergingIntoAnEmptyAccumulatorAdopts) {
    OnlineMoments source;
    source.add(1.0);
    source.add(2.0);

    OnlineMoments target;
    target.merge(source);

    EXPECT_EQ(target.count(), 2U);
    EXPECT_DOUBLE_EQ(target.mean(), 1.5);
}

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

TEST(OnlineMomentsTest, ConfidenceIntervalIsCentredOnTheMean) {
    RandomStream stream(7, StreamPurpose::Diagnostic, 0);
    OnlineMoments moments;
    for (int i = 0; i < 1000; ++i) {
        moments.add(stream.next_normal());
    }

    const auto interval = moments.confidence_interval(0.95);
    ASSERT_TRUE(interval.ok()) << interval.error().describe();

    EXPECT_NEAR(0.5 * (interval.value().lower + interval.value().upper), moments.mean(), 1e-12);
    EXPECT_DOUBLE_EQ(interval.value().level, 0.95);
    EXPECT_GT(interval.value().width(), 0.0);
    EXPECT_TRUE(interval.value().contains(moments.mean()));
}

TEST(OnlineMomentsTest, HigherConfidenceGivesAWiderInterval) {
    RandomStream stream(8, StreamPurpose::Diagnostic, 0);
    OnlineMoments moments;
    for (int i = 0; i < 500; ++i) {
        moments.add(stream.next_normal());
    }

    const double narrow = moments.confidence_interval(0.90).value().width();
    const double medium = moments.confidence_interval(0.95).value().width();
    const double wide = moments.confidence_interval(0.99).value().width();

    EXPECT_LT(narrow, medium);
    EXPECT_LT(medium, wide);
}

// The interval must use t rather than a normal approximation. At 4 degrees of
// freedom the difference is 40%, so a normal-based interval would be materially
// too narrow -- exactly the systematic under-coverage EXP-15 exists to detect.
TEST(OnlineMomentsTest, ConfidenceIntervalUsesStudentTAtSmallSamples) {
    OnlineMoments moments;
    for (const double x : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        moments.add(x);
    }

    const auto interval = moments.confidence_interval(0.95);
    ASSERT_TRUE(interval.ok()) << interval.error().describe();

    // mean 3, sample variance 2.5, standard error sqrt(2.5/5) = 0.7071.
    // t(0.975, 4) = 2.776445, so the half-width is 1.9635. A normal-based
    // interval would use 1.96 and give 1.3859: 29% too narrow.
    EXPECT_NEAR(interval.value().upper - 3.0, 1.9635, 1e-3);
    EXPECT_NEAR(3.0 - interval.value().lower, 1.9635, 1e-3);
}

// At production path counts t and normal agree, so the correctness at small
// samples costs nothing where it does not matter.
TEST(OnlineMomentsTest, ConfidenceIntervalApproachesTheNormalAtLargeSamples) {
    RandomStream stream(9, StreamPurpose::Diagnostic, 0);
    OnlineMoments moments;
    for (int i = 0; i < 100000; ++i) {
        moments.add(stream.next_normal());
    }

    const auto interval = moments.confidence_interval(0.95);
    ASSERT_TRUE(interval.ok());

    const double half_width = 0.5 * interval.value().width();
    const double normal_half_width = 1.959963984540054 * moments.standard_error().value();
    EXPECT_NEAR(half_width, normal_half_width, 1e-4 * normal_half_width);
}

TEST(OnlineMomentsTest, ConfidenceIntervalRejectsInvalidLevels) {
    OnlineMoments moments;
    moments.add(1.0);
    moments.add(2.0);

    EXPECT_FALSE(moments.confidence_interval(0.0).ok());
    EXPECT_FALSE(moments.confidence_interval(1.0).ok());
    EXPECT_FALSE(moments.confidence_interval(-0.5).ok());
    EXPECT_FALSE(moments.confidence_interval(1.5).ok());
}

// ---------------------------------------------------------------------------
// OnlineCovariance
// ---------------------------------------------------------------------------

TEST(OnlineCovarianceTest, RecoversAKnownCovariance) {
    // y = 2x exactly, so cov(x,y) = 2 var(x) and the correlation is 1.
    OnlineCovariance covariance;
    for (const double x : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        covariance.add(x, 2.0 * x);
    }

    const auto cov = covariance.covariance();
    ASSERT_TRUE(cov.ok()) << cov.error().describe();
    EXPECT_NEAR(cov.value(), 2.0 * 2.5, 1e-13);

    const auto correlation = covariance.correlation();
    ASSERT_TRUE(correlation.ok());
    EXPECT_NEAR(correlation.value(), 1.0, 1e-13);
}

TEST(OnlineCovarianceTest, RecoversPerfectNegativeCorrelation) {
    OnlineCovariance covariance;
    for (const double x : {1.0, 2.0, 3.0, 4.0, 5.0}) {
        covariance.add(x, -3.0 * x);
    }

    EXPECT_NEAR(covariance.correlation().value(), -1.0, 1e-13);
}

TEST(OnlineCovarianceTest, IndependentSeriesHaveNearZeroCorrelation) {
    OnlineCovariance covariance;
    RandomStream first(11, StreamPurpose::Diagnostic, 0);
    RandomStream second(11, StreamPurpose::Diagnostic, 1);

    constexpr int kSamples = 200000;
    for (int i = 0; i < kSamples; ++i) {
        covariance.add(first.next_normal(), second.next_normal());
    }

    EXPECT_NEAR(
        covariance.correlation().value(), 0.0, 5.0 / std::sqrt(static_cast<double>(kSamples)));
}

TEST(OnlineCovarianceTest, MergeMatchesSequentialAccumulation) {
    RandomStream stream(13, StreamPurpose::Diagnostic, 0);
    std::vector<std::pair<double, double>> sample;
    sample.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const double x = stream.next_normal();
        const double y = 0.5 * x + stream.next_normal();
        sample.emplace_back(x, y);
    }

    OnlineCovariance sequential;
    for (const auto& [x, y] : sample) {
        sequential.add(x, y);
    }

    for (const std::size_t split : {std::size_t{1}, std::size_t{333}, std::size_t{999}}) {
        OnlineCovariance left;
        OnlineCovariance right;
        for (std::size_t i = 0; i < split; ++i) {
            left.add(sample[i].first, sample[i].second);
        }
        for (std::size_t i = split; i < sample.size(); ++i) {
            right.add(sample[i].first, sample[i].second);
        }
        left.merge(right);

        EXPECT_EQ(left.count(), sequential.count()) << "split at " << split;
        EXPECT_NEAR(left.covariance().value(), sequential.covariance().value(), 1e-11)
            << "split at " << split;
        EXPECT_NEAR(left.correlation().value(), sequential.correlation().value(), 1e-11)
            << "split at " << split;
    }
}

TEST(OnlineCovarianceTest, CovarianceIsUndefinedBelowTwoObservations) {
    OnlineCovariance covariance;
    EXPECT_FALSE(covariance.covariance().ok());

    covariance.add(1.0, 2.0);
    EXPECT_FALSE(covariance.covariance().ok());

    covariance.add(2.0, 3.0);
    EXPECT_TRUE(covariance.covariance().ok());
}

// A constant series has no direction for the other to align with, so the
// correlation genuinely does not exist. Returning 0 would assert independence
// that has not been established.
TEST(OnlineCovarianceTest, CorrelationIsUndefinedWhenAMarginalIsConstant) {
    OnlineCovariance covariance;
    for (const double y : {1.0, 2.0, 3.0, 4.0}) {
        covariance.add(5.0, y);
    }

    const auto correlation = covariance.correlation();
    ASSERT_FALSE(correlation.ok());
    EXPECT_EQ(correlation.error().code, ErrorCode::InvalidArgument);

    // The covariance itself is still well defined, and is zero.
    ASSERT_TRUE(covariance.covariance().ok());
    EXPECT_NEAR(covariance.covariance().value(), 0.0, 1e-14);
}

TEST(OnlineCovarianceTest, CorrelationStaysWithinItsBounds) {
    RandomStream stream(17, StreamPurpose::Diagnostic, 0);

    for (const double loading : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
        OnlineCovariance covariance;
        for (int i = 0; i < 10000; ++i) {
            const double x = stream.next_normal();
            covariance.add(x, loading * x + 0.1 * stream.next_normal());
        }

        const auto correlation = covariance.correlation();
        ASSERT_TRUE(correlation.ok()) << correlation.error().describe();
        EXPECT_GE(correlation.value(), -1.0) << "loading = " << loading;
        EXPECT_LE(correlation.value(), 1.0) << "loading = " << loading;
    }
}

TEST(OnlineCovarianceTest, TracksMarginalMoments) {
    OnlineCovariance covariance;
    for (const double x : {1.0, 2.0, 3.0}) {
        covariance.add(x, 10.0 * x);
    }

    EXPECT_DOUBLE_EQ(covariance.mean_x(), 2.0);
    EXPECT_DOUBLE_EQ(covariance.mean_y(), 20.0);
    EXPECT_DOUBLE_EQ(covariance.moments_x().mean(), 2.0);
    EXPECT_DOUBLE_EQ(covariance.moments_y().mean(), 20.0);
    EXPECT_NEAR(covariance.moments_x().sample_variance().value(), 1.0, 1e-14);
}

}  // namespace
}  // namespace diffusionworks
