#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

namespace diffusionworks {

/// Intermediate quantities of the Black-Scholes formula.
///
/// Exposed because they are the evidence behind a price: d2 is the risk-neutral
/// probability of finishing in the money, and a reviewer checking a suspicious
/// value looks here first. They are also reported as diagnostics.
struct BlackScholesTerms {
    double d1{};
    double d2{};

    /// \f$ \sigma\sqrt{T} \f$, the total standard deviation of log-spot.
    double total_volatility{};

    /// \f$ F = S_0 e^{(r-q)T} \f$.
    double forward{};

    /// \f$ \ln(F/K) \f$. Positive when the forward is in the money for a call.
    double log_moneyness{};

    double discount_factor{};
    double dividend_discount_factor{};

    /// True when \f$ \sigma\sqrt{T} = 0 \f$, i.e. T = 0 or sigma = 0.
    ///
    /// In this limit the asset is deterministic, d1 and d2 are not finite, and
    /// the closed form must not be evaluated as written.
    bool degenerate{};
};

/// Closed-form Black-Scholes-Merton pricing and sensitivities.
///
/// Implements MATHEMATICAL-SPEC sections 2 and 3 for European calls and puts.
/// This engine is the project's analytic reference: Monte Carlo, finite
/// difference, and Greek estimators are all validated against it, so its
/// accuracy bounds everything downstream.
///
/// Degenerate limits (sigma*sqrt(T) = 0) are handled explicitly rather than by
/// evaluating the formula and hoping. The option is then worth the discounted
/// intrinsic value of its forward, which is well defined and returned with a
/// warning. The single exception is the payoff kink, F = K, where gamma diverges
/// and delta jumps; there the price is still exact (zero) but the sensitivities
/// do not exist, and greeks() reports a failure rather than a number.
class BlackScholesAnalyticEngine {
public:
    /// Computes the shared intermediate terms.
    [[nodiscard]] static BlackScholesTerms terms(const MarketState& market,
                                                 const EuropeanOption& option,
                                                 const BlackScholesModel& model) noexcept;

    /// Prices a European option.
    ///
    /// \f[ C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2) \f]
    /// \f[ P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1) \f]
    ///
    /// Never fails for a valid market, option, and model: the domain types
    /// already exclude the inputs that would make the price undefined. The
    /// Result return is kept for interface consistency and because a non-finite
    /// intermediate is reported rather than returned.
    [[nodiscard]] static Result<PricingResult>
    price(const MarketState& market, const EuropeanOption& option, const BlackScholesModel& model);

    /// Computes analytic delta, gamma, vega, theta, and rho.
    ///
    /// See Greeks for units; they are per unit of the underlying variable, not
    /// per volatility point or per basis point.
    ///
    /// Fails with InvalidArgument at the degenerate payoff kink (sigma*sqrt(T) = 0
    /// and F = K), where gamma is unbounded and delta is discontinuous. Returning
    /// any finite number there would be a fabrication.
    [[nodiscard]] static Result<Greeks>
    greeks(const MarketState& market, const EuropeanOption& option, const BlackScholesModel& model);
};

}  // namespace diffusionworks
