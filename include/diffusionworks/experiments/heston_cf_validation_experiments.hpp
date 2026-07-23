#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-09: Heston characteristic-function validation.
///
/// The experiment validates the characteristic function the price is built from, not
/// only the prices, and it keeps three kinds of evidence separate and honestly
/// labelled: the characteristic function's own analytic properties, external
/// reference prices, and internal numerical checks. The sweep axes below drive the
/// property and invariant checks across regimes -- strike, maturity, correlation,
/// vol-of-variance, and mean-reversion, spanning both Feller-satisfying and
/// Feller-violating parameters. The external reference cases are fixed published or
/// independently-generated constants and are built into the experiment, not
/// configured.
struct HestonCfValidationExperimentConfig {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};

    /// Held fixed across the sweep so the varied axes are isolated.
    double initial_variance{0.04};
    double long_run_variance{0.04};

    std::vector<double> strikes{80.0, 100.0, 120.0};
    std::vector<double> maturities{0.25, 1.0, 5.0};
    std::vector<double> correlations{-0.7, 0.0, 0.7};

    /// Spans the Feller boundary: with kappa and theta fixed, a larger xi pushes
    /// 2 kappa theta below xi^2, so both statuses are exercised.
    std::vector<double> vol_of_variances{0.3, 0.6, 1.0};
    std::vector<double> mean_reversions{1.0, 2.0};

    /// The dense real-u grid on which finiteness and continuity of the
    /// characteristic function are checked.
    double cf_grid_max_u{50.0};
    std::int64_t cf_grid_points{1000};

    /// Quadrature node counts for the integration-convergence study, coarse to fine.
    std::vector<std::int64_t> quadrature_node_counts{16, 32, 64, 128, 256, 512};

    /// Nodes used to price the external reference cases, generous enough that their
    /// own integration error is far below the reference tolerances.
    std::int64_t reference_quadrature_nodes{1024};
};

/// EXP-09: does the characteristic function have the properties it must, and does
/// the price built from it reproduce trusted references?
[[nodiscard]] Result<ExperimentRecord>
run_heston_cf_validation(const HestonCfValidationExperimentConfig& config);

}  // namespace diffusionworks
