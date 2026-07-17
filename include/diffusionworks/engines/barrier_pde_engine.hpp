#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/barrier_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

namespace diffusionworks {

/// Prices a continuously monitored knock-out call by solving the Black-Scholes PDE
/// with an absorbing Dirichlet boundary at the barrier.
///
/// What it prices, and against what
/// --------------------------------
/// This is the *continuous* monitoring convention -- the barrier is watched at every
/// instant, which is what the absorbing boundary V(B, tau) = 0 enforces. It shares
/// that convention with `BarrierAnalyticEngine`, and validating against it is the
/// point: the analytic engine gives the closed-form continuous price, so the PDE's
/// job is to converge to it as the grid refines. EXP-07's PDE arm measures that
/// convergence.
///
/// It is a *different question* from the Monte Carlo engine's. That one measures the
/// bias of *discrete* monitoring against the continuous price and the Brownian-bridge
/// correction for it. This one prices the continuous contract directly. The two arms
/// of Phase 7 answer different questions and must not be conflated.
///
/// Why it reuses the vanilla engine
/// --------------------------------
/// A knock-out is the same PDE as its vanilla, solved on a smaller domain with one
/// boundary replaced by an absorbing barrier. So this shares the vanilla engine's
/// time-stepping exactly, through the same `solve_core`; only the grid, the live
/// index range, the terminal zeroing, and one boundary value differ. Nothing about
/// the scheme, the tridiagonal solve, or the diagnostics is reimplemented, and the
/// vanilla path is provably unchanged by the sharing.
///
/// The barrier is aligned to a grid node by construction (it is a Dirichlet
/// boundary, and half a cell of misplacement prices a different contract). The
/// down-and-out keeps the [0, S_max] span with the barrier on an interior node and
/// the sub-barrier nodes carried dead; the up-and-out's domain top is the barrier
/// itself. `AssetGrid::with_barrier_on_node` and the design note in
/// `finite_difference_engine.cpp` explain why the down-and-out does not simply span
/// [B, S_max].
class BarrierPdeEngine {
public:
    /// Prices the option.
    ///
    /// Refuses:
    ///   - any monitoring convention but continuous -- discrete and bridge are the
    ///     Monte Carlo engine's, and returning the continuous price for them would
    ///     answer a different question with a plausible number;
    ///   - puts -- only calls are implemented, and the terminal condition and
    ///     boundaries here are call-shaped;
    ///   - knock-ins -- the PDE solves the knock-out directly; a knock-in is
    ///     vanilla minus knock-out (in-out parity) and is left to that composition
    ///     rather than given a second, separately-wrong solve here.
    ///
    /// Prices, rather than refusing:
    ///   - an already-breached spot (a knock-out past its barrier is worth zero); it
    ///     is an exact price with a warning, not an error;
    ///   - an up-and-out struck at or above its barrier (worth exactly zero: any
    ///     path finishing above the strike has crossed the barrier).
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const BarrierOption& option,
                                                     const BlackScholesModel& model,
                                                     const PdeConfig& config);

    /// The full solution on the grid, for tests and convergence studies. Reuses the
    /// vanilla engine's Solution so a caller can inspect the grid, the values, and
    /// the same diagnostics.
    [[nodiscard]] static Result<FiniteDifferenceEngine::Solution>
    solve(const MarketState& market,
          const BarrierOption& option,
          const BlackScholesModel& model,
          const PdeConfig& config);
};

}  // namespace diffusionworks
