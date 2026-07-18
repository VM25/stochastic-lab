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

/// Which variance-reduction techniques to apply.
///
/// Off by default. A technique that changes the estimator must be asked for, so
/// that a stored configuration says plainly which estimator produced its number.
struct VarianceReduction {
    /// Pair each path with its reflection, and average the two payoffs into one
    /// observation.
    ///
    /// \f[ \hat{V}_{\text{anti}} = \frac{1}{N}\sum_{i=1}^{N}
    ///     \frac{X_i(Z_i) + X_i(-Z_i)}{2} \f]
    ///
    /// The pair average is *one* observation, not two. This is the detail that
    /// makes the reported uncertainty honest: the two members are negatively
    /// correlated by construction, so treating 2N payoffs as independent would
    /// compute Var(X)/2N instead of Var(X)(1+rho)/2N and misstate the standard
    /// error by a factor of sqrt(1+rho). The estimate would be right and its
    /// error bar wrong, which is worse than both being wrong.
    ///
    /// Reduces variance only when the payoff is monotone in the shock. It is not
    /// universally helpful, and where it is not, this build says so rather than
    /// quietly costing twice the work for nothing.
    bool antithetic{false};

    /// Subtract a control whose expectation is known:
    ///
    /// \f[ \hat{V}_{CV} = \bar{X} - \beta(\bar{Y} - \mu_Y) \f]
    ///
    /// For an arithmetic Asian the control is the geometric Asian on the same
    /// paths, whose price is known in closed form. The two track each other
    /// closely -- same paths, same monitoring, and averages that differ only by
    /// AM versus GM -- so the correlation is high and the reduction large.
    ///
    /// Only applies to arithmetic Asian options. Requesting it elsewhere is
    /// rejected rather than ignored.
    bool control_variate{false};

    [[nodiscard]] bool any() const noexcept { return antithetic || control_variate; }
};

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

    VarianceReduction variance_reduction{};

    /// Paths used to estimate the control-variate coefficient, drawn from indices
    /// disjoint from the main sample.
    ///
    /// Why a pilot rather than the same sample
    /// ---------------------------------------
    /// MATHEMATICAL-SPEC section 11 requires beta to be estimated "without
    /// introducing undocumented look-ahead bias". Estimating beta from the very
    /// sample it then corrects makes beta and the sample mean dependent, which
    /// biases the estimator by O(1/N). At production path counts that bias is
    /// negligible -- but negligible and absent are different claims, and only one
    /// of them needs no footnote.
    ///
    /// A pilot on disjoint path indices removes it entirely: beta is then a
    /// constant with respect to the main sample, so the estimator is exactly
    /// unbiased. The cost is the pilot's paths and a slightly noisier beta, which
    /// costs a little of the variance reduction rather than any of the
    /// correctness. Counter-based addressing makes the disjointness free: the
    /// pilot simply reads indices [paths, paths + pilot_paths).
    std::int64_t control_variate_pilot_paths{2000};

    /// Worker threads for the path loop. One is sequential and bit-identical to the
    /// single-threaded engine (ADR-011). More partition the paths deterministically
    /// across fixed workers with thread-local accumulators reduced in block order, so a
    /// fixed thread count is reproducible and different counts differ only by the
    /// floating-point reassociation of the merge -- a documented effect, not a race.
    /// The counter-based generator makes this safe: every path is a pure function of
    /// (seed, index), so there is no shared mutable RNG to contend on.
    int threads{1};
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
