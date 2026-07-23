#include <diffusionworks/experiments/cross_method_experiments.hpp>
#include <diffusionworks/experiments/experiment.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

// A small config: one Black-Scholes regime (so one European and one Asian family
// cell) and two Heston regimes straddling the Feller boundary, at reduced path and
// grid counts. The agreement gates are on 5-sigma point differences and a grid-based
// relative tolerance, both robust at this size; the tests assert those invariants,
// not empirical rankings.
CrossMethodAgreementExperimentConfig small_config() {
    CrossMethodAgreementExperimentConfig config;
    config.volatilities = {0.2};
    config.maturities = {1.0};
    config.monte_carlo_paths = 50000;
    config.fd_asset_nodes = 201;
    config.fd_time_steps = 200;
    config.heston_maturities = {1.0};
    config.heston_vol_of_variances = {0.3, 1.0};
    config.heston_monte_carlo_paths = 50000;
    config.heston_monte_carlo_steps = 200;
    return config;
}

class CrossMethodExperimentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto record = run_cross_method_agreement(small_config());
        ASSERT_TRUE(record.ok()) << record.error().describe();
        record_ = new ExperimentRecord(std::move(record).value());
    }

    static void TearDownTestSuite() {
        delete record_;
        record_ = nullptr;
    }

    static ExperimentRecord* record_;
};

ExperimentRecord* CrossMethodExperimentTest::record_ = nullptr;

TEST_F(CrossMethodExperimentTest, AllFamiliesAgreeAndThePassStands) {
    EXPECT_EQ(record_->status, ExperimentStatus::Pass);
}

// Family 1: three independent methods (analytic, Monte Carlo, finite difference)
// agree with the exact analytic price for the European option.
TEST_F(CrossMethodExperimentTest, BlackScholesEuropeanMethodsAgreeWithTheAnalyticReference) {
    const auto& cells = record_->results.at("black_scholes_european");
    ASSERT_FALSE(cells.empty());
    for (const auto& c : cells) {
        EXPECT_TRUE(c.at("all_agree").get<bool>());
        bool saw_fd = false;
        bool saw_mc = false;
        for (const auto& m : c.at("methods")) {
            const auto method = m.at("method").get<std::string>();
            if (method == "finite_difference_crank_nicolson") {
                saw_fd = true;
                EXPECT_TRUE(m.at("agrees").get<bool>());
            } else if (method.rfind("monte_carlo", 0) == 0) {
                saw_mc = true;
                EXPECT_TRUE(m.at("agrees").get<bool>());
            }
        }
        EXPECT_TRUE(saw_fd) << "the finite-difference method must be compared here";
        EXPECT_TRUE(saw_mc) << "the Monte Carlo methods must be compared here";
    }
}

// Family 2: the Heston Monte Carlo agrees with the characteristic-function price in
// both a Feller-satisfying and a Feller-violating regime.
TEST_F(CrossMethodExperimentTest, HestonMonteCarloAgreesWithTheCharacteristicFunction) {
    const auto& cells = record_->results.at("heston_european");
    ASSERT_EQ(cells.size(), 2U);
    bool saw_satisfied = false;
    bool saw_violated = false;
    for (const auto& c : cells) {
        EXPECT_TRUE(c.at("agrees").get<bool>())
            << "T=" << c.at("maturity") << " xi=" << c.at("vol_of_variance") << " sigmas "
            << c.at("sigmas");
        if (c.at("satisfies_feller").get<bool>()) {
            saw_satisfied = true;
        } else {
            saw_violated = true;
        }
    }
    EXPECT_TRUE(saw_satisfied);
    EXPECT_TRUE(saw_violated);
}

// Family 3: with no closed form, the crude, antithetic, and control-variate
// estimators of the arithmetic Asian agree with one another.
TEST_F(CrossMethodExperimentTest, ArithmeticAsianEstimatorsAgreeWithEachOther) {
    const auto& cells = record_->results.at("arithmetic_asian");
    ASSERT_FALSE(cells.empty());
    for (const auto& c : cells) {
        EXPECT_TRUE(c.at("all_agree").get<bool>());
        EXPECT_EQ(c.at("pairwise_agreement").size(), 3U);  // three estimators -> three pairs
        for (const auto& p : c.at("pairwise_agreement")) {
            EXPECT_TRUE(p.at("agrees").get<bool>()) << p.at("pair");
        }
    }
}

}  // namespace
}  // namespace diffusionworks
