#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/heston.hpp>

#include <complex>
#include <cstdint>

namespace diffusionworks {

/// Controls the numerical integration of the Heston pricing formula.
struct HestonAnalyticConfig {
    /// Gauss-Legendre nodes for the base quadrature. The price is also computed at
    /// twice this many, and the two are compared: their difference is the reported
    /// integration error, and if it exceeds the tolerance the price is refused
    /// rather than returned.
    ///
    /// 1024 resolves the ordinary and moderately-hard regimes -- across moneyness out
    /// to twice the strike, short and long maturity, Feller-satisfying and not,
    /// including deep out-of-the-money at quarter-year maturity -- to well below the
    /// tolerance. It is deliberately generous: the semi-infinite integrand's tail is
    /// the slow part, and under-resolving it produces a smooth wrong price that only
    /// the doubling check exposes, so paying for nodes is cheaper than trusting a
    /// plausible number. The pathological corner (very short maturity with large
    /// vol-of-variance) does not converge at any practical node count and is refused
    /// rather than approximated -- which is the point of the check.
    std::int64_t quadrature_nodes{1024};

    /// The largest acceptable |price(2N) - price(N)|, in price units. A run whose
    /// doubling disagreement exceeds this has not converged, and a non-converged
    /// integral must not masquerade as a price (FAILURE-MODES: a failure that
    /// returns a plausible number is worse than one that stops).
    double convergence_tolerance{1e-8};
};

/// Prices European options under the Heston model by numerically integrating the
/// characteristic-function pricing formula.
///
/// Conventions, documented because MATHEMATICAL-SPEC section 4 requires it
/// -------------------------------------------------------------------------
///   - **Characteristic function:** the "little Heston trap" formulation of
///     Albrecher, Mayer, Schoutens and Tistaert (2007). It is algebraically
///     identical to Heston's original but writes the complex discriminant and the
///     `g` factor so that the principal branch of the complex logarithm is the
///     correct one for all maturities. Heston's original form crosses a branch cut
///     as T grows and returns a discontinuous, wrong price there -- the "trap" -- so
///     the choice is not cosmetic.
///   - **Pricing formula:** the two-probability decomposition
///     \f$ C = S e^{-qT} P_1 - K e^{-rT} P_2 \f$, with each \f$P_j\f$ a
///     \f$\tfrac12 + \tfrac1\pi\int_0^\infty\f$ integral of the real part of a
///     characteristic-function ratio.
///   - **Integration bounds:** the semi-infinite domain \f$[0,\infty)\f$ is mapped
///     to \f$[0,1)\f$ by \f$u = x/(1-x)\f$, so no truncation point is chosen and no
///     tail is discarded -- the transform carries the whole domain.
///   - **Quadrature:** Gauss-Legendre on the transformed interval, with the node
///     count and the doubling tolerance in HestonAnalyticConfig.
///   - **The u -> 0 singularity** of the integrand is removable and is never
///     evaluated: the Gauss-Legendre nodes are strictly interior.
///
/// A put is priced from the call by put-call parity, which is exact and independent
/// of the model, rather than by a second integral.
class HestonAnalyticEngine {
public:
    /// Prices the option.
    ///
    /// Refuses a non-finite or non-positive spot, strike, or maturity through the
    /// domain types. Reports rather than refuses a Feller-condition violation: the
    /// pricing integral is well defined whether or not the variance can reach zero,
    /// so the violation is a diagnostic and a warning, not an error.
    ///
    /// Refuses to return a price when the doubling check fails: an integral that has
    /// not converged is a failure, not a number near the answer.
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const EuropeanOption& option,
                                                     const HestonModel& model,
                                                     const HestonAnalyticConfig& config = {});

    /// Evaluates the little-trap characteristic function of the log-price directly,
    /// exposed so the object the price is built from can be validated in its own
    /// right (EXP-09), not only through the price it produces.
    ///
    /// Returns \f$ \phi_j(u) = \mathbb{E}^{j}[e^{i u \ln S_T}] \f$ for the
    /// two-probability decomposition index `j`. Index 2 is the genuine risk-neutral
    /// log-price characteristic function; index 1 is its counterpart under the share
    /// measure. The argument `u` is complex so that identities off the real axis can
    /// be checked -- in particular the martingale identity
    /// \f$ \phi_2(-i) = S_0 e^{(r-q)T} \f$, which pins the drift.
    ///
    /// Refuses a non-positive maturity and a non-positive vol-of-variance for the
    /// same reasons `price` does: the formula divides by \f$\xi^2\f$, and at
    /// \f$\xi = 0\f$ the variance is deterministic and this integral does not
    /// represent the model. `index` must be 1 or 2.
    [[nodiscard]] static Result<std::complex<double>>
    log_price_characteristic_function(const MarketState& market,
                                      const HestonModel& model,
                                      double maturity,
                                      std::complex<double> u,
                                      int index = 2);
};

}  // namespace diffusionworks
