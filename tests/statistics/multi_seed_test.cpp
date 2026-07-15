#include <diffusionworks/core/error.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace diffusionworks {
namespace {

std::vector<SeedResult> make_results(const std::vector<double>& estimates) {
    std::vector<SeedResult> results;
    results.reserve(estimates.size());
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        results.push_back(SeedResult{static_cast<std::uint64_t>(i + 1), estimates[i]});
    }
    return results;
}

TEST(MultiSeedTest, SummarisesDispersionAcrossSeeds) {
    const auto summary = summarize_seeds(make_results({10.0, 12.0, 14.0, 16.0, 18.0}));

    ASSERT_TRUE(summary.ok()) << summary.error().describe();
    EXPECT_EQ(summary.value().seed_count, 5U);
    EXPECT_DOUBLE_EQ(summary.value().mean, 14.0);
    EXPECT_NEAR(summary.value().standard_deviation, std::sqrt(10.0), 1e-13);
    EXPECT_NEAR(summary.value().standard_error, std::sqrt(10.0 / 5.0), 1e-13);
    EXPECT_DOUBLE_EQ(summary.value().minimum, 10.0);
    EXPECT_DOUBLE_EQ(summary.value().maximum, 18.0);
}

// One run cannot measure its own dispersion. Reporting zero would assert a
// reliability that has not been established, which is the failure this type
// exists to prevent -- so it is a rejection, not a special case.
TEST(MultiSeedTest, RejectsFewerThanTwoSeeds) {
    EXPECT_FALSE(summarize_seeds({}).ok());

    const auto single = summarize_seeds(make_results({10.0}));
    ASSERT_FALSE(single.ok());
    EXPECT_EQ(single.error().code, ErrorCode::InvalidArgument);
    EXPECT_NE(single.error().message.find("independent"), std::string::npos);
}

// A repeated seed is not an independent replication. Left unchecked it would
// shrink the apparent dispersion, making the estimator look more reliable
// precisely because the evidence is weaker.
TEST(MultiSeedTest, RejectsRepeatedSeeds) {
    const std::vector<SeedResult> results{{1, 10.0}, {2, 11.0}, {1, 10.0}};

    const auto summary = summarize_seeds(results);
    ASSERT_FALSE(summary.ok());
    EXPECT_EQ(summary.error().code, ErrorCode::InvalidArgument);
    EXPECT_NE(summary.error().message.find("more than once"), std::string::npos);
}

TEST(MultiSeedTest, RejectsNonFiniteEstimates) {
    const std::vector<SeedResult> with_nan{{1, 10.0},
                                           {2, std::numeric_limits<double>::quiet_NaN()}};
    const auto summary = summarize_seeds(with_nan);

    ASSERT_FALSE(summary.ok());
    EXPECT_EQ(summary.error().code, ErrorCode::NonFiniteValue);
}

// RMSE against nothing is not zero error; it is an unanswerable question, so the
// field stays absent rather than being filled with a number.
TEST(MultiSeedTest, OmitsErrorMetricsWithoutAReference) {
    const auto summary = summarize_seeds(make_results({10.0, 12.0}));

    ASSERT_TRUE(summary.ok());
    EXPECT_FALSE(summary.value().rmse.has_value());
    EXPECT_FALSE(summary.value().bias.has_value());
}

TEST(MultiSeedTest, ComputesRmseAndBiasAgainstAReference) {
    // Errors against 10: -1, +1, +3, +5. Mean error 2, mean squared error 9.
    const auto summary = summarize_seeds(make_results({9.0, 11.0, 13.0, 15.0}), 10.0);

    ASSERT_TRUE(summary.ok()) << summary.error().describe();
    ASSERT_TRUE(summary.value().rmse.has_value());
    ASSERT_TRUE(summary.value().bias.has_value());

    EXPECT_NEAR(*summary.value().bias, 2.0, 1e-13);
    EXPECT_NEAR(*summary.value().rmse, 3.0, 1e-13);
}

// RMSE and bias answer different questions, which is why both are reported. An
// estimator can be centred and noisy, or tight and wrong; only the pair
// distinguishes them, and discretization bias looks like the second.
TEST(MultiSeedTest, BiasAndRmseSeparateCentringFromSpread) {
    // Unbiased but noisy: errors -5, +5.
    const auto noisy = summarize_seeds(make_results({5.0, 15.0}), 10.0);
    ASSERT_TRUE(noisy.ok());
    EXPECT_NEAR(*noisy.value().bias, 0.0, 1e-13) << "symmetric errors must cancel in the bias";
    EXPECT_NEAR(*noisy.value().rmse, 5.0, 1e-13) << "but must not cancel in the RMSE";

    // Biased but tight: errors +3, +3.
    const auto biased = summarize_seeds(make_results({13.0, 13.0}), 10.0);
    ASSERT_TRUE(biased.ok());
    EXPECT_NEAR(*biased.value().bias, 3.0, 1e-13);
    EXPECT_NEAR(*biased.value().rmse, 3.0, 1e-13);
    EXPECT_NEAR(biased.value().standard_deviation, 0.0, 1e-13)
        << "a systematically biased estimator can still look perfectly stable across seeds, "
           "which is why dispersion alone is not evidence of accuracy";
}

TEST(MultiSeedTest, RmseIsZeroForAnExactEstimator) {
    const auto summary = summarize_seeds(make_results({10.0, 10.0}), 10.0);

    ASSERT_TRUE(summary.ok());
    EXPECT_NEAR(*summary.value().rmse, 0.0, 1e-15);
    EXPECT_NEAR(*summary.value().bias, 0.0, 1e-15);
}

// The realised across-seed dispersion should agree with what a single run's
// standard error predicts. That agreement is the whole justification for
// reporting a standard error at all, and it is checked here rather than assumed.
TEST(MultiSeedTest, AcrossSeedDispersionMatchesTheWithinRunStandardError) {
    constexpr int kSeeds = 40;
    constexpr int kSamplesPerSeed = 20000;

    std::vector<SeedResult> results;
    double predicted_standard_error = 0.0;

    for (int seed = 0; seed < kSeeds; ++seed) {
        OnlineMoments moments;
        RandomStream stream(static_cast<std::uint64_t>(1000 + seed), StreamPurpose::Diagnostic, 0);
        for (int i = 0; i < kSamplesPerSeed; ++i) {
            moments.add(stream.next_normal());
        }
        results.push_back(SeedResult{static_cast<std::uint64_t>(1000 + seed), moments.mean()});
        predicted_standard_error += moments.standard_error().value();
    }
    predicted_standard_error /= static_cast<double>(kSeeds);

    const auto summary = summarize_seeds(results, 0.0);
    ASSERT_TRUE(summary.ok()) << summary.error().describe();

    // The realised standard deviation of the estimate across seeds should match
    // the average self-reported standard error. Its own sampling error is
    // ~1/sqrt(2(k-1)) relative, about 11% at 40 seeds, so the bound is 40%: this
    // is checking that the two agree in magnitude, not that they agree to three
    // digits.
    EXPECT_NEAR(summary.value().standard_deviation,
                predicted_standard_error,
                0.4 * predicted_standard_error)
        << "the estimator's realised dispersion disagrees with its self-reported standard error";
}

}  // namespace
}  // namespace diffusionworks
