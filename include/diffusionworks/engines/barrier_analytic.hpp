#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/barrier_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

namespace diffusionworks {

/// Closed-form barrier pricing under Black-Scholes, continuous monitoring.
///
/// The Reiner-Rubinstein formulae, which follow from the reflection principle: a
/// geometric Brownian motion's running extremum has a known joint law with its
/// terminal value, so the knock-out value is the vanilla less a reflected vanilla
/// weighted by (B/S)^(2(mu+1)).
///
/// Covers all four call cases -- down and up, knock-out and knock-in -- on both
/// branches of the strike-versus-barrier split. Puts are not implemented.
///
/// **This prices continuous monitoring, and almost no traded barrier is
/// continuously monitored.** A real contract observes the barrier at fixes,
/// usually daily closes. Continuous monitoring finds every excursion; discrete
/// monitoring misses those between observations. So this value is systematically
/// *lower* than the discretely monitored contract for a knock-out, and the gap
/// closes only as O(1/sqrt(m)) -- slowly enough that daily monitoring is not
/// "close enough" to continuous, which is the point EXP-07 measures.
///
/// Used here as a *reference for the continuous convention*, not as the price of a
/// discretely monitored contract. The engine refuses a discretely monitored
/// option rather than returning this number for it.
class BarrierAnalyticEngine {
public:
    /// Prices a continuously monitored barrier option.
    ///
    /// Fails on a discrete or bridge convention: those are different contracts,
    /// and returning the continuous value for them would be answering a question
    /// nobody asked with a number that looks right.
    ///
    /// An already-breached barrier is priced, not refused. A knock-out whose spot
    /// is past the barrier is worth zero; a knock-in past its barrier is worth the
    /// vanilla. Both are correct prices for a real situation
    /// (MATHEMATICAL-SPEC section 18).
    ///
    /// An up-and-out call struck above its barrier is likewise priced, at exactly
    /// zero: paying requires the terminal price to exceed the strike and hence the
    /// barrier, so every path that would pay knocked out first. Exact, and carries
    /// a warning, because a zero price otherwise reads as a defect.
    [[nodiscard]] static Result<PricingResult>
    price(const MarketState& market, const BarrierOption& option, const BlackScholesModel& model);
};

}  // namespace diffusionworks
