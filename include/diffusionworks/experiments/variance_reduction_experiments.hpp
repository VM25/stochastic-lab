#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-05: variance-reduction efficiency.
///
/// The experiment compares crude Monte Carlo against antithetic variates, a
/// control variate, and their combination, on a European call and an arithmetic
/// Asian call. Its headline metric is efficiency -- error per unit of computation
/// -- not raw variance, because a technique that halves the variance while
/// doubling the work has bought nothing. Every estimator runs on the *same* path
/// budget and the *same* seeds so the comparison is fair (the catalog's first exit
/// criterion), and the arithmetic Asian's reference, which has no closed form, is a
/// high-path-count run reported with its own uncertainty rather than asserted.
struct VarianceReductionExperimentConfig {
    double rate{0.05};
    double dividend_yield{0.0};
    double strike{100.0};

    /// Spots swept as regimes for a call struck at 100: out-of-, at-, and
    /// in-the-money. Antithetic variates help most where the payoff is smooth and
    /// monotone in the shock, and that changes with moneyness, so a single spot
    /// would license a universal claim the exit gate forbids.
    std::vector<double> spots{90.0, 100.0, 110.0};

    /// Low and high volatility. Volatility scales the diffusion and so the payoff's
    /// curvature, which is where antithetic sampling's benefit erodes.
    std::vector<double> volatilities{0.2, 0.4};

    double maturity{1.0};

    /// Monitoring dates for the arithmetic Asian. The geometric control is taken on
    /// the same dates, so the two averages differ only by AM versus GM.
    std::int64_t asian_monitoring_count{12};

    /// Paths per estimator per seed. Identical across estimators: the comparison is
    /// at an equal path budget, so an efficiency difference is the estimator's, not
    /// the sample size's.
    std::int64_t paths{100000};

    /// Pilot paths for the control-variate coefficient, drawn from indices disjoint
    /// from the main sample so the estimator stays exactly unbiased.
    std::int64_t control_variate_pilot_paths{2000};

    /// Paths for the arithmetic Asian reference price, which has no closed form. Far
    /// larger than the comparison budget, with the best available estimator, so the
    /// RMSE the estimators are measured against is dominated by their own error and
    /// not the reference's.
    std::int64_t reference_paths{2000000};

    /// Independent replications. Variance, RMSE, and efficiency are all measured
    /// across seeds, so a single seed cannot separate a real reduction from a lucky
    /// draw.
    std::uint64_t seed_count{12};

    std::uint64_t master_seed{20260717};
};

/// EXP-05: which estimator produces the lowest error per unit of computation?
[[nodiscard]] Result<ExperimentRecord>
run_variance_reduction_efficiency(const VarianceReductionExperimentConfig& config);

}  // namespace diffusionworks
