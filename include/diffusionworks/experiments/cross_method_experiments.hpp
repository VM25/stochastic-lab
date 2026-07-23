#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-13: cross-method accuracy and agreement.
///
/// Where independent methods price the same instrument, do they agree within their
/// justified tolerances? The experiment groups methods by the instrument they
/// support -- each method is used only where it is defined -- and checks agreement
/// against the appropriate reference: an exact analytic value where one exists, and
/// cross-estimator agreement where it does not. It makes no timing or performance
/// claim; the question is accuracy and agreement, not cost.
struct CrossMethodAgreementExperimentConfig {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double strike{100.0};

    /// The Black-Scholes sweep: for the European family the reference is the exact
    /// analytic price, and Monte Carlo (crude and antithetic) and the finite-
    /// difference solver must agree with it; the arithmetic Asian family reuses the
    /// same volatilities and maturities and checks the Monte Carlo estimators
    /// against each other, since it has no closed form.
    std::vector<double> volatilities{0.2, 0.4};
    std::vector<double> maturities{0.5, 2.0};

    std::int64_t monte_carlo_paths{400000};
    std::uint64_t monte_carlo_seed{20260717};
    std::int64_t asian_monitoring_count{12};
    std::int64_t control_variate_pilot_paths{2000};

    /// Crank-Nicolson finite-difference grid. Fine enough that its own
    /// discretization error is comfortably inside the agreement tolerance.
    std::int64_t fd_asset_nodes{401};
    std::int64_t fd_time_steps{400};

    /// The Heston European family: the characteristic-function price is the
    /// reference and the Heston Monte Carlo simulation must agree with it. One
    /// Feller-satisfying and one violating regime.
    std::vector<double> heston_maturities{0.5, 2.0};
    double heston_initial_variance{0.04};
    double heston_mean_reversion{2.0};
    double heston_long_run_variance{0.04};
    std::vector<double> heston_vol_of_variances{0.3, 1.0};
    double heston_correlation{-0.7};
    std::int64_t heston_monte_carlo_paths{200000};
    std::int64_t heston_monte_carlo_steps{200};
    std::uint64_t heston_monte_carlo_seed{20260717};

    /// A Monte Carlo estimate agrees with a reference when their difference is
    /// within this many combined standard errors. Conservative, so an unbiased
    /// estimator effectively never fails by chance while a real bias is caught.
    double agreement_sigma{5.0};

    /// The finite-difference price is deterministic, so it is held to a relative
    /// tolerance justified by its grid rather than to a standard error.
    double fd_relative_tolerance{5e-3};
};

/// EXP-13: do independent methods agree where they price the same instrument?
[[nodiscard]] Result<ExperimentRecord>
run_cross_method_agreement(const CrossMethodAgreementExperimentConfig& config);

}  // namespace diffusionworks
