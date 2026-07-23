#include <diffusionworks/experiments/experiment.hpp>
#include <diffusionworks/experiments/heston_cf_validation_experiments.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

// A small sweep -- one strike, maturity, and correlation, and two vol-of-variances
// straddling the Feller boundary (with kappa = 2, theta = 0.04: xi = 0.3 satisfies
// it, xi = 1.0 violates it). The reference cases, the convergence study, and the
// pathological corner are fixed inside the experiment and run regardless, so this
// only shrinks the property sweep. The tests assert the characteristic function's
// exact analytic identities and the record's pass gates.
HestonCfValidationExperimentConfig small_config() {
    HestonCfValidationExperimentConfig config;
    config.strikes = {100.0};
    config.maturities = {1.0};
    config.correlations = {-0.5};
    config.vol_of_variances = {0.3, 1.0};
    config.mean_reversions = {2.0};
    config.cf_grid_points = 200;
    config.quadrature_node_counts = {16, 64, 256, 1024};
    return config;
}

class HestonCfValidationExperimentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto record = run_heston_cf_validation(small_config());
        ASSERT_TRUE(record.ok()) << record.error().describe();
        record_ = new ExperimentRecord(std::move(record).value());
    }

    static void TearDownTestSuite() {
        delete record_;
        record_ = nullptr;
    }

    static ExperimentRecord* record_;
};

ExperimentRecord* HestonCfValidationExperimentTest::record_ = nullptr;

TEST_F(HestonCfValidationExperimentTest, Passes) {
    EXPECT_EQ(record_->status, ExperimentStatus::Pass);
}

// The characteristic function's exact analytic identities, checked as reported by
// the experiment: phi(0) = 1, conjugate symmetry, the martingale identity, |phi| <=
// 1, and finiteness, across a sweep that exercises both Feller statuses.
TEST_F(HestonCfValidationExperimentTest, CharacteristicFunctionSatisfiesItsIdentities) {
    const auto& p = record_->results.at("characteristic_function_properties");
    EXPECT_GT(p.at("feller_satisfied").get<int>(), 0);
    EXPECT_GT(p.at("feller_violated").get<int>(), 0);
    EXPECT_LT(p.at("max_phi_zero_deviation").get<double>(), 1e-10);
    EXPECT_LT(p.at("max_conjugate_symmetry_residual").get<double>(), 1e-9);
    EXPECT_LT(p.at("max_martingale_identity_relative_error").get<double>(), 1e-8);
    EXPECT_LE(p.at("max_modulus").get<double>(), 1.0 + 1e-9);
    EXPECT_TRUE(p.at("all_finite").get<bool>());
    EXPECT_TRUE(p.at("passes").get<bool>());
}

// Every external reference price is recovered within its stated tolerance, and each
// is labelled by provenance -- exactly one published value, the rest independently
// generated (not called literature).
TEST_F(HestonCfValidationExperimentTest, RecoversExternalReferencesWithHonestProvenance) {
    const auto& refs = record_->results.at("external_references");
    ASSERT_FALSE(refs.empty());
    int published = 0;
    int independent = 0;
    for (const auto& c : refs) {
        EXPECT_TRUE(c.at("passes").get<bool>())
            << c.at("case").get<std::string>() << " error " << c.at("absolute_error").get<double>();
        const auto category = c.at("provenance_category").get<std::string>();
        if (category == "published") {
            ++published;
        } else if (category == "independently_generated") {
            ++independent;
        } else {
            ADD_FAILURE() << "unexpected provenance category: " << category;
        }
    }
    EXPECT_EQ(published, 1) << "only Fang-Oosterlee (2008) is a published literature value";
    EXPECT_GT(independent, 0);
}

TEST_F(HestonCfValidationExperimentTest, AnalyticInvariantsHold) {
    const auto& inv = record_->results.at("analytic_invariants");
    EXPECT_TRUE(inv.at("put_call_parity_passes").get<bool>());
    EXPECT_LT(inv.at("put_call_parity_max_residual").get<double>(), 1e-9);
    EXPECT_TRUE(inv.at("black_scholes_limit_passes").get<bool>());
}

// The integration converges as the grid refines, and the pathological corner is
// refused with the exact ConvergenceFailure status rather than returned as a price.
TEST_F(HestonCfValidationExperimentTest, IntegrationConvergesAndPathologicalCornerIsRefused) {
    const auto& ic = record_->results.at("internal_numerical_checks");
    EXPECT_TRUE(ic.at("integration_converges").get<bool>());
    EXPECT_LT(ic.at("final_integration_error").get<double>(), 1e-8);
    EXPECT_TRUE(ic.at("pathological_refused_correctly").get<bool>());
    EXPECT_EQ(ic.at("pathological_actual_status").get<std::string>(), "convergence_failure");
}

}  // namespace
}  // namespace diffusionworks
