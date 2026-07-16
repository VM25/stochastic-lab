#pragma once

#include <diffusionworks/experiments/convergence.hpp>
#include <diffusionworks/experiments/experiment.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters shared by the convergence experiments.
///
/// Stored rather than hard-coded: EXPERIMENT-CATALOG requires every experiment to
/// run from a configuration, and a number typed into a source file is not one.
struct ConvergenceExperimentConfig {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.30};
    double maturity{1.0};
    double strike{100.0};

    std::uint64_t master_seed{20260715};

    /// Independent replications, for the experiments whose claims are stochastic.
    std::uint64_t seed_count{32};

    /// Grid levels for the strong and weak studies, coarse to fine.
    std::vector<std::int64_t> step_counts{16, 32, 64, 128, 256, 512, 1024};

    /// How many of the finest levels form the asymptotic window.
    ///
    /// Declared here, in the configuration, because it must be fixed before the
    /// results are seen. Choosing it afterwards would be choosing the verdict.
    std::size_t asymptotic_level_count{4};

    /// Paths per grid level in the strong study.
    std::uint64_t strong_paths{100000};

    /// Paths per grid level for EXP-03's call-payoff arm.
    ///
    /// Far larger than `strong_paths` because that arm is the one with no closed
    /// form. Its bias is O(dt) while even the paired standard error is
    /// O(sqrt(dt)/sqrt(N)), so resolution goes like sqrt(dt*N): millions of paths
    /// buy only a factor of a few, and the requirement grows as the grid refines.
    ///
    /// Configurable rather than fixed so a test can exercise the wiring without
    /// running the physics. A test that had to spend 4e6 paths per level to check
    /// that a JSON field is populated would simply not be run.
    std::uint64_t call_payoff_paths{4000000};

    /// Path counts for the sampling-convergence sweep.
    std::vector<std::uint64_t> path_counts{1000, 4000, 16000, 64000, 256000, 1024000};
};

/// EXP-01: does Monte Carlo pricing error decay as N^{-1/2}?
///
/// Measures RMSE across independent seeds against the analytic price, and fits the
/// log-log slope against the expected -0.5.
///
/// RMSE across seeds rather than the error of a single run: one run's error is one
/// draw from a distribution centred near zero, so it can be small by luck, and a
/// sweep of single runs produces a scatter with no reliable slope at all. The
/// catalog names using one seed as a failure condition for exactly this reason.
[[nodiscard]] Result<ExperimentRecord>
run_sampling_convergence(const ConvergenceExperimentConfig& config);

/// EXP-02: do Euler-Maruyama and Milstein attain their strong orders?
///
/// Measures E|S_T^dt - S_T^exact| on common Brownian paths and fits the order.
[[nodiscard]] Result<ExperimentRecord>
run_strong_convergence(const ConvergenceExperimentConfig& config);

/// EXP-03: do the schemes reproduce expected values at their weak rate?
///
/// Uses the closed-form scheme moments for f(S)=S and f(S)=S^2, where the weak
/// error is exact, and the paired simulated estimator for the call payoff, where
/// it is not.
[[nodiscard]] Result<ExperimentRecord>
run_weak_convergence(const ConvergenceExperimentConfig& config);

/// EXP-04: when does increasing the path count stop helping?
///
/// Sweeps paths and steps jointly and separates the two error sources: the bias
/// from the analytic scheme moment, the sampling error from the run itself.
[[nodiscard]] Result<ExperimentRecord>
run_bias_variance_tradeoff(const ConvergenceExperimentConfig& config);

}  // namespace diffusionworks
