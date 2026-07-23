#include <diffusionworks/experiments/experiment.hpp>
#include <diffusionworks/experiments/variance_reduction_experiments.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <string>

namespace diffusionworks {
namespace {

// A deliberately small configuration: one regime, few paths and seeds. The tests
// assert mathematical invariants (the deterministic work model, the definition of
// efficiency) and unbiasedness, not empirical estimator rankings -- so they need
// only enough sampling to resolve unbiasedness, not the full published budget.
VarianceReductionExperimentConfig small_config() {
    VarianceReductionExperimentConfig config;
    config.spots = {100.0};
    config.volatilities = {0.2};
    config.paths = 10000;
    config.control_variate_pilot_paths = 2000;
    config.asian_monitoring_count = 12;
    config.reference_paths = 60000;
    config.seed_count = 6;
    config.master_seed = 20260717;
    return config;
}

// Runs the experiment once for the whole suite; the tests only read its record.
class VarianceReductionExperimentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto record = run_variance_reduction_efficiency(small_config());
        ASSERT_TRUE(record.ok()) << record.error().describe();
        record_ = new ExperimentRecord(std::move(record).value());
    }

    static void TearDownTestSuite() {
        delete record_;
        record_ = nullptr;
    }

    static const nlohmann::json& cells() { return record_->results.at("cells"); }

    static nlohmann::json cell(const std::string& instrument, const std::string& estimator) {
        for (const auto& c : cells()) {
            if (c.at("instrument") == instrument && c.at("estimator") == estimator) {
                return c;
            }
        }
        ADD_FAILURE() << "no cell for " << instrument << "/" << estimator;
        return nlohmann::json::object();
    }

    static ExperimentRecord* record_;
};

ExperimentRecord* VarianceReductionExperimentTest::record_ = nullptr;

TEST_F(VarianceReductionExperimentTest, PassesAndValidatesTheControlExpectation) {
    EXPECT_EQ(record_->status, ExperimentStatus::Pass);

    // Identity: the geometric control's closed-form price must equal an independent
    // Monte Carlo estimate of the same average, within the resolution threshold.
    const auto& validations = record_->results.at("control_validation");
    ASSERT_FALSE(validations.empty());
    for (const auto& v : validations) {
        EXPECT_TRUE(v.at("validated").get<bool>())
            << "geometric control gap " << v.at("gap_over_se") << " standard errors";
    }
}

// The work model is a deterministic function of the configuration and the
// estimator -- independent of the seed, the machine, and the run. These are the
// exact operation counts the design documents, checked against the formula:
// path-leg simulations + arithmetic averages (+ geometric averages under the
// control), with an antithetic observation simulating two paths and the control
// adding a pilot.
TEST_F(VarianceReductionExperimentTest, WorkModelIsDeterministicAndMatchesTheFormula) {
    const std::int64_t n = 10000;  // production paths (config.paths)
    const std::int64_t p = 2000;   // control-variate pilot
    const std::int64_t m = 12;     // Asian monitoring steps

    // European (M = 1): crude simulates N paths; antithetic simulates 2N.
    const auto ec = cell("european", "crude");
    EXPECT_EQ(ec.at("simulated_production_paths").get<std::int64_t>(), n);
    EXPECT_EQ(ec.at("pilot_paths").get<std::int64_t>(), 0);
    EXPECT_EQ(ec.at("work_units").get<std::int64_t>(), 2 * n);  // simulate + arithmetic average
    const auto ea = cell("european", "antithetic");
    EXPECT_EQ(ea.at("simulated_production_paths").get<std::int64_t>(), 2 * n);
    EXPECT_EQ(ea.at("work_units").get<std::int64_t>(), 4 * n);

    // Arithmetic Asian (M = 12).
    EXPECT_EQ(cell("arithmetic_asian", "crude").at("work_units").get<std::int64_t>(), 2 * n * m);
    EXPECT_EQ(cell("arithmetic_asian", "antithetic").at("work_units").get<std::int64_t>(),
              4 * n * m);

    // The control variate adds a pilot and a geometric average per simulated path.
    const auto ct = cell("arithmetic_asian", "control_variate");
    EXPECT_EQ(ct.at("pilot_paths").get<std::int64_t>(), p);
    EXPECT_EQ(ct.at("geometric_average_ops").get<std::int64_t>(), (n + p) * m);
    EXPECT_EQ(ct.at("work_units").get<std::int64_t>(), 3 * (n + p) * m);

    // Combined = antithetic production (2N simulated) plus the control's pilot.
    const auto cm = cell("arithmetic_asian", "combined");
    EXPECT_EQ(cm.at("simulated_production_paths").get<std::int64_t>(), 2 * n);
    EXPECT_EQ(cm.at("pilot_paths").get<std::int64_t>(), p);
    EXPECT_EQ(cm.at("work_units").get<std::int64_t>(), 3 * (2 * n + p) * m);
}

// Definitional identity: efficiency is exactly 1 / (variance * work units), with no
// wall-clock time in it.
TEST_F(VarianceReductionExperimentTest, EfficiencyIsTheReciprocalOfVarianceTimesWork) {
    for (const auto& c : cells()) {
        const double variance = c.at("variance").get<double>();
        const auto work = c.at("work_units").get<std::int64_t>();
        const double reported = c.at("work_normalised_efficiency").get<double>();
        if (variance > 0.0 && work > 0) {
            const double expected = 1.0 / (variance * static_cast<double>(work));
            EXPECT_NEAR(reported, expected, expected * 1e-12)
                << c.at("instrument") << "/" << c.at("estimator");
        }
    }
}

// Unbiasedness: crude and antithetic are unbiased estimators of the European price,
// so their across-seed mean must sit within the resolution threshold of the exact
// Black-Scholes value. (The record stores the signed bias and the across-seed
// dispersion; the standard error of the mean is dispersion / sqrt(seed_count).)
TEST_F(VarianceReductionExperimentTest, EuropeanEstimatorsAreUnbiasedAgainstBlackScholes) {
    for (const char* estimator : {"crude", "antithetic"}) {
        const auto c = cell("european", estimator);
        const double bias = std::abs(c.at("bias").get<double>());
        const double sd = c.at("across_seed_standard_deviation").get<double>();
        const auto seeds = c.at("seed_count").get<std::int64_t>();
        ASSERT_GT(seeds, 0);
        const double standard_error = sd / std::sqrt(static_cast<double>(seeds));
        if (standard_error > 0.0) {
            EXPECT_LT(bias / standard_error, 4.0)
                << "European " << estimator << " shows a resolved bias against Black-Scholes";
        }
    }
}

}  // namespace
}  // namespace diffusionworks
