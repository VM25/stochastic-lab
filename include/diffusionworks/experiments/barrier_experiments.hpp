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

    /// Down-barriers, below the spot. The bias grows as the barrier approaches the
    /// spot, so a single barrier would measure one point of a curve and imply it was
    /// the whole story.
    std::vector<double> barriers{70.0, 80.0, 90.0, 95.0};

    /// Up-barriers, above the spot, for the up-and-out arm.
    ///
    /// The catalog asks for both directions, and they are not redundant: an
    /// up-and-out call knocks out in exactly the states where it would have paid,
    /// so its bias is driven by a different part of the distribution than a
    /// down-and-out's. Kept below the strike-versus-barrier split (B > K = 100), the
    /// branch where the contract is not identically zero.
    std::vector<double> up_barriers{105.0, 110.0, 120.0, 140.0};

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
