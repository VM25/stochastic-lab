#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/barrier_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

#include <cstdint>

namespace diffusionworks {

struct BarrierMonteCarloConfig {
    std::int64_t paths{200000};
    std::uint64_t seed{20260716};
    double confidence_level{0.95};
};

/// What the barrier simulation observed about itself.
struct BarrierMonteCarloDiagnostics {
    std::int64_t paths{};
    std::int64_t monitoring_dates{};

    /// Paths knocked at an observation date.
    std::int64_t knocked_at_observation{};

    /// Paths knocked only by the bridge -- that is, paths whose observed values
    /// never breached, but which the bridge judged to have crossed in between.
    ///
    /// The whole correction, made countable. Zero under discrete monitoring by
    /// construction; under the bridge it is the population that discrete
    /// monitoring silently misses, and its size is the monitoring bias.
    std::int64_t knocked_by_bridge_only{};

    /// Mean bridge crossing probability over the survivors it was applied to.
    double mean_bridge_probability{};

    /// Paths that finished in the money and were not knocked.
    std::int64_t paid{};
};

/// Prices a barrier option by simulation, under discrete or bridge monitoring.
///
/// Why the path is simulated *exactly at the monitoring dates*
/// ----------------------------------------------------------
/// The barrier introduces two distinct discretisation errors, and conflating them
/// makes both unmeasurable:
///
///   1. **Path discretisation** -- the simulated path is not the true path.
///   2. **Monitoring discretisation** -- the barrier is observed at m dates
///      rather than continuously.
///
/// This engine eliminates the first entirely. Log-GBM is Brownian motion with
/// drift, so the exact scheme samples the *exact* joint law of the price at any
/// finite set of dates, however coarse. Stepping exactly at the m monitoring
/// dates therefore introduces no path error at all: what remains is monitoring
/// bias and Monte Carlo noise, nothing else.
///
/// That is what makes EXP-07's measurement clean. An Euler path on a finer grid
/// would add a bias of its own that the O(1/sqrt(m)) monitoring bias would then be
/// measured through.
class BarrierMonteCarloEngine {
public:
    /// Prices the option.
    ///
    /// Refuses continuous monitoring: no simulation observes a barrier
    /// continuously, and pretending otherwise by monitoring very finely would
    /// report a discretely monitored price as a continuous one. Use
    /// BarrierAnalyticEngine, or the bridge convention, which is what a simulation
    /// can honestly say about continuous monitoring.
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const BarrierOption& option,
                                                     const BlackScholesModel& model,
                                                     const BarrierMonteCarloConfig& config);
};

/// The probability that a Brownian bridge between two observed log-prices crossed
/// the barrier in between.
///
/// For log-prices a and b at the ends of an interval of length dt, with a log
/// barrier at `log_barrier` and diffusion sigma, the minimum of the bridge crosses
/// with probability
///
///     p = exp( -2 (a - lb)(b - lb) / (sigma^2 dt) )
///
/// when both endpoints are above lb, and 1 when either is at or below it. The
/// symmetric expression holds for a maximum against an upper barrier, with the
/// signs of both factors reversed -- which leaves the product, and hence the
/// formula, unchanged.
///
/// Exact for this model rather than an approximation: log-GBM *is* Brownian motion
/// with drift, and the drift cancels out of the bridge law once both endpoints are
/// conditioned on. That is why the correction converges to the continuous price
/// rather than to something near it.
///
/// It corrects the discretisation of the *observation*, not of the path. Those are
/// different, and a bridge applied to an Euler path would be correcting the wrong
/// one.
[[nodiscard]] double bridge_crossing_probability(double log_start,
                                                 double log_end,
                                                 double log_barrier,
                                                 double variance_rate,
                                                 double dt,
                                                 bool down_barrier) noexcept;

}  // namespace diffusionworks
