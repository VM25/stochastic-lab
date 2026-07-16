#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/asian_option.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/simulation/gbm_stepper.hpp>

#include <cstdint>

namespace diffusionworks {

/// Everything a Monte Carlo run needs beyond the market, instrument, and model.
///
/// Every field is explicit. There is no default seed and no default path count:
/// both are published metadata, and a run whose seed was chosen implicitly cannot
/// be reproduced from its own record (ADR-010).
struct MonteCarloConfig {
    /// Number of independent paths.
    std::int64_t paths{0};

    /// Time steps per path.
    ///
    /// For a European option under the exact scheme, one step is not an
    /// approximation but the whole answer: the log-normal transition is exact
    /// over any horizon, so a single step to maturity samples the true terminal
    /// distribution. More steps cost time and change nothing. They matter only
    /// for path-dependent payoffs, or for the explicit schemes whose error is
    /// the object of study.
    std::int64_t steps{1};

    /// Master seed. Part of the reproducibility contract.
    std::uint64_t seed{0};

    DiscretizationScheme scheme{DiscretizationScheme::Exact};

    /// Nominal coverage of the reported interval.
    double confidence_level{0.95};
};

/// Monte Carlo pricing under geometric Brownian motion.
///
/// Estimator (MATHEMATICAL-SPEC section 6): the discounted payoff is averaged
/// across paths,
///
/// \f[ \hat{V}_N = \frac{1}{N}\sum_{i=1}^{N} e^{-rT} f(S^{(i)}), \f]
///
/// with the sample standard error \f$ s/\sqrt{N} \f$ and a Student-t interval.
///
/// Every result carries its uncertainty. A Monte Carlo point estimate without a
/// standard error is not a price, it is a number that happens to be near one, and
/// reporting it alone is what VALIDATION-PLAN section 6 refuses.
class MonteCarloEngine {
public:
    /// Prices a European option.
    ///
    /// Under the exact scheme with any step count this converges to the analytic
    /// Black-Scholes value as paths increase, with error \f$ O(N^{-1/2}) \f$ and
    /// no discretisation bias. Under Euler or Milstein it converges to a
    /// *different* value at fixed step count -- the discretisation bias -- which
    /// vanishes only as steps increase. Both effects are separated and measured
    /// in EXP-04.
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const EuropeanOption& option,
                                                     const BlackScholesModel& model,
                                                     const MonteCarloConfig& config);

    /// Prices an Asian option.
    ///
    /// The grid must resolve the monitoring dates: `steps` must be a positive
    /// multiple of the option's monitoring count, so that every monitoring date
    /// falls exactly on a grid point. Interpolating a monitored state, or
    /// silently monitoring at the nearest grid point instead, would price a
    /// different contract than the one described.
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const AsianOption& option,
                                                     const BlackScholesModel& model,
                                                     const MonteCarloConfig& config);
};

}  // namespace diffusionworks
