#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-07: barrier monitoring bias.
struct BarrierExperimentConfig {
    double spot{100.0};
    double strike{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.20};
    double maturity{1.0};

    /// Barriers, as distances below the spot. The bias grows as the barrier
    /// approaches the spot, so a single barrier would measure one point of a
    /// curve and imply it was the whole story.
    std::vector<double> barriers{70.0, 80.0, 90.0, 95.0};

    /// Monitoring frequencies. 250 is roughly daily.
    std::vector<std::int64_t> monitoring_counts{5, 12, 25, 50, 100, 250};

    std::int64_t paths{200000};

    /// Independent replications.
    ///
    /// The bias is a difference of two numbers each carrying sampling error, so a
    /// single seed cannot separate a real bias from a lucky draw -- and the bridge
    /// arm's residual is small enough that this is the whole question there.
    std::uint64_t seed_count{16};

    std::uint64_t master_seed{20260716};

    /// Volatilities for the sensitivity arm. The bias scales with sigma*sqrt(dt),
    /// so volatility moves it as directly as monitoring frequency does.
    std::vector<double> volatilities{0.1, 0.2, 0.4};
};

/// EXP-07: how much error results from discrete barrier monitoring?
[[nodiscard]] Result<ExperimentRecord>
run_barrier_monitoring_bias(const BarrierExperimentConfig& config);

}  // namespace diffusionworks
