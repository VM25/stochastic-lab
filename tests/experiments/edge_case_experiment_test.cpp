#include <diffusionworks/experiments/edge_case_experiments.hpp>
#include <diffusionworks/experiments/experiment.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

// The edge cases are analytic and deterministic, so the default config runs
// instantly; there is no need to shrink it.
class EdgeCaseExperimentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto record = run_edge_cases(EdgeCaseExperimentConfig{});
        ASSERT_TRUE(record.ok()) << record.error().describe();
        record_ = new ExperimentRecord(std::move(record).value());
    }

    static void TearDownTestSuite() {
        delete record_;
        record_ = nullptr;
    }

    static ExperimentRecord* record_;
};

ExperimentRecord* EdgeCaseExperimentTest::record_ = nullptr;

// Every edge case resolves and, above all, no non-finite value escapes.
TEST_F(EdgeCaseExperimentTest, EveryCaseResolvesAndNoNonFiniteEscapes) {
    EXPECT_EQ(record_->status, ExperimentStatus::Pass);
    const auto& summary = record_->results.at("summary");
    EXPECT_FALSE(summary.at("any_non_finite_escaped").get<bool>());
    EXPECT_TRUE(summary.at("all_resolved").get<bool>());
    EXPECT_EQ(summary.at("passed").get<std::int64_t>(),
              summary.at("total_cases").get<std::int64_t>());
}

// The degenerate Heston regions refuse (they do not return a plausible number), and
// the invalid inputs are rejected at construction (they never reach an engine).
TEST_F(EdgeCaseExperimentTest, DegenerateCasesRefuseAndInvalidInputsAreRejected) {
    int refusals = 0;
    int rejections = 0;
    for (const auto& c : record_->results.at("cases")) {
        const auto category = c.at("category").get<std::string>();
        if (category == "degenerate_refusal") {
            EXPECT_EQ(c.at("behavior").get<std::string>(), "refused") << c.at("case");
            EXPECT_TRUE(c.at("passed").get<bool>());
            ++refusals;
        } else if (category == "invalid_input_rejected") {
            EXPECT_EQ(c.at("behavior").get<std::string>(), "rejected") << c.at("case");
            EXPECT_TRUE(c.at("passed").get<bool>());
            ++rejections;
        }
    }
    EXPECT_GT(refusals, 0);
    EXPECT_GT(rejections, 0);
}

// The limiting and already-breached cases produce finite, correct values.
TEST_F(EdgeCaseExperimentTest, LimitingAndBreachedCasesAreCorrectAndFinite) {
    for (const auto& c : record_->results.at("cases")) {
        const auto category = c.at("category").get<std::string>();
        if (category == "limiting_behavior" || category == "already_breached") {
            EXPECT_EQ(c.at("behavior").get<std::string>(), "priced") << c.at("case");
            ASSERT_TRUE(c.contains("finite"));
            EXPECT_TRUE(c.at("finite").get<bool>()) << c.at("case");
            EXPECT_TRUE(c.at("passed").get<bool>()) << c.at("case");
        }
    }
}

}  // namespace
}  // namespace diffusionworks
