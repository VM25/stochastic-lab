#include <diffusionworks/experiments/coverage_experiments.hpp>
#include <diffusionworks/experiments/experiment.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

namespace diffusionworks {
namespace {

// At-the-money and deep-out-of-the-money, a small and a large sample. The Monte
// Carlo uses fixed per-trial seeds, so the coverage is deterministic and these
// assertions are not flaky. Reduced trial count, still enough to resolve the
// deep-out-of-the-money under-coverage.
CoverageExperimentConfig small_config() {
    CoverageExperimentConfig config;
    config.strikes = {100.0, 160.0};
    config.sample_sizes = {2000, 50000};
    config.trial_count = 300;
    return config;
}

class CoverageExperimentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto record = run_confidence_coverage(small_config());
        ASSERT_TRUE(record.ok()) << record.error().describe();
        record_ = new ExperimentRecord(std::move(record).value());
    }

    static void TearDownTestSuite() {
        delete record_;
        record_ = nullptr;
    }

    static ExperimentRecord* record_;
};

ExperimentRecord* CoverageExperimentTest::record_ = nullptr;

// The methodology is sound where the central limit theorem holds (the large sample),
// and the sweep reaches the small-sample skewed regime where coverage degrades, so
// the record passes rather than being inconclusive.
TEST_F(CoverageExperimentTest, Passes) {
    EXPECT_EQ(record_->status, ExperimentStatus::Pass);
    const auto& summary = record_->results.at("summary");
    EXPECT_TRUE(summary.at("methodology_sound_at_largest_sample").get<bool>());
    EXPECT_TRUE(summary.at("observed_under_coverage_somewhere").get<bool>());
}

// The payoff skewness is an exact, deterministic property of the payoff and grows
// with moneyness: the deep-out-of-the-money call is far more right-skewed than the
// at-the-money one. This is the quantity that explains the coverage degradation.
TEST_F(CoverageExperimentTest, PayoffSkewnessGrowsWithMoneyness) {
    double atm_skew = -1.0;
    double deep_skew = -1.0;
    for (const auto& c : record_->results.at("cells")) {
        if (c.at("strike").get<double>() == 100.0) {
            atm_skew = c.at("payoff_skewness").get<double>();
        } else if (c.at("strike").get<double>() == 160.0) {
            deep_skew = c.at("payoff_skewness").get<double>();
        }
    }
    ASSERT_GT(atm_skew, 0.0);
    ASSERT_GT(deep_skew, 0.0);
    EXPECT_GT(deep_skew, atm_skew) << "the deep-OTM payoff must be more skewed than the ATM one";
    EXPECT_GT(deep_skew, 3.0 * atm_skew) << "and markedly so";
}

// At the largest sample every moneyness is defensible; at the small sample the deep-
// out-of-the-money call under-covers, which is the defect the experiment detects.
TEST_F(CoverageExperimentTest, CoverageIsDefensibleAtLargeSampleAndDegradesForSkewedSmallSample) {
    bool large_sample_all_defensible = true;
    bool deep_small_under_covers = false;
    for (const auto& c : record_->results.at("cells")) {
        if (c.at("is_largest_sample").get<bool>() && !c.at("defensible").get<bool>()) {
            large_sample_all_defensible = false;
        }
        if (c.at("strike").get<double>() == 160.0 &&
            c.at("sample_size").get<std::int64_t>() == 2000) {
            deep_small_under_covers = c.at("under_covers").get<bool>();
        }
    }
    EXPECT_TRUE(large_sample_all_defensible);
    EXPECT_TRUE(deep_small_under_covers)
        << "the deep-OTM call at the small sample is expected to under-cover";
}

}  // namespace
}  // namespace diffusionworks
