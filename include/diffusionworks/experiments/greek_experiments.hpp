#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-08: Greek estimator comparison.
///
/// The grid is deliberately wide. The whole point of the experiment is that no
/// estimator is best everywhere, so a single scenario would license exactly the
/// universal ranking the exit gate forbids. Moneyness, maturity, and volatility are
/// all swept, and the finite-difference bump is swept within each cell.
struct GreekExperimentConfig {
    double strike{100.0};
    double rate{0.05};
    double dividend_yield{0.0};

    /// Deep out-of-the-money, at-the-money, deep in-the-money for a call at K = 100.
    /// The pathwise and finite-difference estimators behave very differently across
    /// moneyness -- the payoff slope is almost always 0 or 1 deep OTM/ITM, which is
    /// where the estimators' variances part company.
    std::vector<double> spots{70.0, 100.0, 130.0};

    /// Short and long. Short maturity concentrates the terminal distribution, which
    /// sharpens the finite-difference bias and the likelihood-ratio variance.
    std::vector<double> maturities{0.1, 2.0};

    /// Low and high. Volatility scales the diffusion, so it moves the estimators'
    /// variances directly.
    std::vector<double> volatilities{0.1, 0.5};

    /// Spot bump fractions for finite-difference delta and gamma. Swept so the
    /// bias-variance trade-off is measured rather than assumed, and so the empirical
    /// variance-versus-bump scaling can be fitted rather than asserted.
    std::vector<double> spot_bump_fractions{0.05, 0.02, 0.01, 0.005};

    /// Volatility bumps for finite-difference vega.
    std::vector<double> volatility_bumps{0.02, 0.01, 0.005};

    std::int64_t paths{100000};

    /// Independent replications. Bias, variance, and RMSE are all measured across
    /// seeds, so a single seed cannot separate a real bias from a lucky draw.
    std::uint64_t seed_count{16};

    std::uint64_t master_seed{20260717};
};

/// EXP-08: which Greek estimator is most accurate, and where?
[[nodiscard]] Result<ExperimentRecord>
run_greek_estimator_comparison(const GreekExperimentConfig& config);

}  // namespace diffusionworks
