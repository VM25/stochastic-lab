#pragma once

#include <diffusionworks/experiments/experiment.hpp>

#include <cstdint>
#include <vector>

namespace diffusionworks {

/// Parameters for EXP-14: statistical confidence coverage.
///
/// Do the reported Monte Carlo confidence intervals achieve their intended coverage?
/// The experiment repeats independent simulations, records how often the exact
/// analytic price falls inside each interval, and compares that observed coverage to
/// the nominal level. It sweeps the two things that break coverage: the sample size
/// (the central limit theorem needs enough paths) and the payoff skewness (a
/// deep-out-of-the-money call pays on almost no paths, so its payoff distribution is
/// far from normal). The point is not only to confirm coverage where it should hold
/// but to locate and explain where it does not.
struct CoverageExperimentConfig {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.2};
    double maturity{1.0};

    /// At-the-money, out-of-the-money, and deep out-of-the-money. Moneyness controls
    /// the payoff skewness: the deeper out of the money, the more skewed the
    /// discounted payoff and the harder the interval's normal approximation.
    std::vector<double> strikes{100.0, 130.0, 160.0};

    /// A small sample (where the central limit theorem is weak, especially for a
    /// skewed payoff) and a large one (where it holds).
    std::vector<std::int64_t> sample_sizes{2000, 50000};

    /// Independent replications per cell. Coverage is a fraction over these trials,
    /// so its own standard error is sqrt(p(1-p)/trials); enough trials are needed to
    /// tell a real under-coverage from noise.
    std::int64_t trial_count{400};

    std::uint64_t master_seed{20260717};

    /// Nominal coverage of the intervals under test.
    double confidence_level{0.95};
};

/// EXP-14: do the Monte Carlo confidence intervals cover the true value as often as
/// they claim?
[[nodiscard]] Result<ExperimentRecord>
run_confidence_coverage(const CoverageExperimentConfig& config);

}  // namespace diffusionworks
