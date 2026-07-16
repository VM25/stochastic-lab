#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-06: PDE stability and grid convergence.
struct PdeExperimentConfig {
    double spot{100.0};
    double strike{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.20};
    double maturity{1.0};

    // --- Space sweep -------------------------------------------------------
    /// Node counts to refine over.
    std::vector<std::int64_t> space_nodes{101, 201, 401, 801};

    /// Time steps held fixed during the space sweep, large enough that the
    /// temporal error is negligible.
    ///
    /// The premise is checked, not assumed: `dtau_leakage_fraction` in the record
    /// reports how much doubling this moves the finest answer, as a fraction of
    /// the smallest spatial error being fitted. The implicit scheme is first order
    /// in time and needs a far larger value here than Crank-Nicolson does -- at
    /// 8000 its leakage is 8.8% and the fitted "spatial" order is 1.916 rather
    /// than the true 1.997.
    std::int64_t space_sweep_time_steps{128000};

    // --- Time sweep --------------------------------------------------------
    /// Asset nodes held fixed during the time sweep.
    std::int64_t time_sweep_nodes{401};

    /// Step counts to refine over.
    std::vector<std::int64_t> time_steps{10, 20, 40, 80, 160};

    /// The reference for the time sweep, on the *same* spatial grid.
    ///
    /// Same-grid so the spatial error cancels exactly rather than sitting under
    /// the measurement as a floor a fit would read as a convergence stall.
    std::int64_t time_sweep_reference_steps{20000};

    // --- S_max sweep -------------------------------------------------------
    /// S_max values, as multiples of the strike.
    std::vector<double> s_max_multiples{1.5, 2.0, 3.0, 4.0, 6.0, 8.0};

    /// The spacing held fixed across the S_max sweep.
    ///
    /// Fixed dS is the whole point. A sweep at constant node count would change
    /// dS = S_max/(N-1) as S_max grows, so the boundary's contribution and the
    /// grid's resolution would move together and neither could be attributed. The
    /// node count therefore grows with S_max to keep dS where it is.
    double s_max_sweep_spacing{0.5};

    std::int64_t s_max_sweep_time_steps{2000};

    // --- Strike alignment --------------------------------------------------
    /// Fractional offsets of the strike within its cell, in [0, 1).
    ///
    /// 0 is exact alignment. The rest place the kink progressively further from a
    /// node. Swept rather than compared at a single offset, because the error's
    /// dependence on the offset is the phenomenon: it oscillates rather than
    /// degrading monotonically, so one off-node sample could land anywhere and
    /// support any conclusion.
    std::vector<double> strike_offsets{0.0, 0.1, 0.25, 0.5, 0.75, 0.9};

    std::int64_t alignment_nodes{401};
    std::int64_t alignment_time_steps{2000};

    // --- Rannacher ---------------------------------------------------------
    /// Smoothing-step counts to compare.
    std::vector<std::int64_t> rannacher_counts{0, 1, 2, 4};

    // --- Explicit stability ------------------------------------------------
    /// Target dtau/dtau_max ratios to probe, spanning the bound.
    std::vector<double> stability_ratios{0.25, 0.5, 0.9, 0.99, 1.01, 1.5, 2.0};

    std::int64_t stability_nodes{101};

    // --- Sensitivity -------------------------------------------------------
    std::vector<double> volatilities{0.1, 0.2, 0.4, 0.8};
    std::vector<double> maturities{0.25, 1.0, 3.0};
};

/// EXP-06: how do explicit, implicit, and Crank-Nicolson differ in stability,
/// accuracy, and cost?
[[nodiscard]] Result<ExperimentRecord>
run_pde_stability_and_convergence(const PdeExperimentConfig& config);

}  // namespace diffusionworks
