#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>

namespace diffusionworks {

/// Bounds and tolerances for the implied-volatility inversion.
struct ImpliedVolatilityConfig {
    /// The volatility floor of the bracket. Not zero: at exactly zero the option is
    /// worth its discounted intrinsic value and the density is a spike, so the search
    /// runs from a small positive floor and a target at or below the floor's price is
    /// reported as sitting at the floor rather than solved past it.
    double lower_volatility{1.0e-8};

    /// The initial volatility ceiling of the bracket. Expanded up to `max_volatility`
    /// when the target price sits above the price at this ceiling.
    double upper_volatility{5.0};

    /// The hard ceiling. A target that still is not bracketed here is refused rather
    /// than chased to an ever-larger volatility -- a price that requires a 50-vol to
    /// match is a data problem, not a solver problem.
    double max_volatility{50.0};

    /// Convergence is declared when the price residual falls below this.
    double price_tolerance{1.0e-10};

    /// Or when the bracket width falls below this -- the safeguard that guarantees
    /// termination even where vega is so small the price is flat in volatility.
    double volatility_tolerance{1.0e-12};

    int max_iterations{128};
};

/// A solved implied volatility, with the evidence behind it.
struct ImpliedVolatilityResult {
    double implied_volatility{};

    /// The Black-Scholes price at `implied_volatility`. Reported so a reader can see
    /// the residual against the target rather than trust that it is small.
    double achieved_price{};
    double target_price{};

    int iterations{};

    /// Whether the result sits at the volatility floor because the target was at or
    /// below its price -- an option quoted essentially at intrinsic. Not a failure,
    /// but the volatility is a floor rather than a resolved root and says so.
    bool at_lower_floor{};

    /// The no-arbitrage price window the target had to lie in: for a call,
    /// [S e^{-qT} - K e^{-rT}]^+ to S e^{-qT}. Reported so a rejected target can be
    /// seen against the bounds that rejected it.
    double lower_price_bound{};
    double upper_price_bound{};
};

/// Inverts the Black-Scholes price for the volatility that reproduces a target price.
///
/// The method is a safeguarded Newton iteration inside a maintained bracket: a Newton
/// step is taken when it stays inside the bracket and vega is usable, and a bisection
/// step is taken otherwise. Because the Black-Scholes price is strictly increasing in
/// volatility, the bracket always contains the unique root and the bisection fallback
/// guarantees termination even where vega underflows.
///
/// What it refuses (MATHEMATICAL-SPEC section 17)
/// ----------------------------------------------
///   - A target outside the no-arbitrage price window has no volatility that
///     reproduces it: RootNotBracketed, with the window reported.
///   - A non-positive maturity has no implied volatility -- there is no time for
///     volatility to act: UnsupportedCombination.
///   - A target above the price at `max_volatility` is refused rather than matched by
///     an unbounded volatility: RootNotBracketed.
///   - Exhausting the iteration budget without converging is ConvergenceFailure, not a
///     returned best-so-far. A value is never returned when no valid root was found.
class ImpliedVolatility {
public:
    [[nodiscard]] static Result<ImpliedVolatilityResult>
    solve(const MarketState& market,
          const EuropeanOption& option,
          double target_price,
          const ImpliedVolatilityConfig& config = {});
};

}  // namespace diffusionworks
