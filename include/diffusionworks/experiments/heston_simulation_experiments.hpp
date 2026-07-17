#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-10: Heston variance discretization.
///
/// The two regimes the catalog asks for -- one comfortably Feller-satisfying, one
/// violating -- differ only in the vol-of-variance. Everything else is held fixed so
/// the sweep isolates the one parameter that pushes the Feller ratio across 1, which
/// is exactly the boundary the experiment studies. Configurable rather than fixed
/// because EXPERIMENT-CATALOG requires every experiment to run from a configuration.
struct HestonSimulationExperimentConfig {
    double spot{100.0};
    double strike{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double maturity{1.0};

    /// Shared Heston parameters. The regimes differ only in the vol-of-variance
    /// below; these are common to all of them.
    double initial_variance{0.04};
    double mean_reversion{2.0};
    double long_run_variance{0.04};
    double correlation{-0.7};

    /// One regime per vol-of-variance. With the shared parameters above, 0.3 gives a
    /// Feller ratio of 1.78 (satisfied) and 1.0 gives 0.16 (violated), so the default
    /// pair straddles the boundary. The list is swept, so more regimes can be added.
    std::vector<double> vol_of_variance{0.3, 1.0};

    /// Time-step counts, coarse to fine. The bias is O(dt), so this is the axis the
    /// discretization study turns; a wide range is needed to see the decay before it
    /// sinks below the sampling noise.
    std::vector<std::int64_t> step_counts{5, 10, 20, 40, 80, 160, 320};

    std::int64_t paths{40000};

    /// Independent replications. The bias is measured across seeds, so a single seed
    /// cannot separate a real discretization bias from a lucky draw -- the same
    /// reason the convergence and Greek experiments aggregate seeds.
    std::uint64_t seed_count{8};

    std::uint64_t master_seed{20260717};

    /// A bias counts as resolved, and so is eligible for the decay-order fit, only
    /// when it clears this many across-seed standard errors. Below it, the bias is
    /// indistinguishable from sampling noise and fitting an order to it would be
    /// fitting the noise -- which is refused rather than reported.
    double bias_resolution{3.0};
};

/// EXP-10: how does full-truncation Euler behave across stable and difficult regimes?
///
/// Compares the naive Euler scheme, full-truncation Euler, and the semi-analytic
/// characteristic-function reference, measuring the negative pre-truncation frequency,
/// the minimum variance, the pricing bias and its decay with the step, the path
/// failures, and the runtime. Full truncation is expected to price every regime with
/// no non-finite paths; the naive scheme is expected to fail, which is the point.
[[nodiscard]] Result<ExperimentRecord>
run_heston_variance_discretization(const HestonSimulationExperimentConfig& config);

}  // namespace diffusionworks
