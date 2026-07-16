#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/asian_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>

namespace diffusionworks {

/// The lognormal law of a discretely monitored geometric average under GBM.
///
/// Exposed because it is the whole reason the geometric average has a closed form
/// while the arithmetic one does not, and because a reviewer checking a
/// suspicious control-variate price looks here first.
struct GeometricAverageLaw {
    /// \f$ E[\ln G] \f$.
    double log_mean{};

    /// \f$ \operatorname{Var}[\ln G] \f$.
    double log_variance{};

    /// \f$ E[G] = e^{\mu + \sigma^2/2} \f$, the average's forward.
    double forward{};
};

/// Closed-form pricing of a discretely monitored geometric Asian option.
///
/// Why this exists at all
/// ----------------------
/// The arithmetic average of lognormals is not lognormal, so an arithmetic Asian
/// has no closed form and needs Monte Carlo. The *geometric* average is
/// lognormal, because a sum of normals is normal and the log of a geometric
/// average is exactly such a sum. That single fact gives an exact price, and it
/// is what makes the geometric Asian the natural control variate for the
/// arithmetic one: the two track each other closely, and one of them is known.
///
/// The law (Kemna and Vorst, "A Pricing Method for Options Based on Average Asset
/// Values", Journal of Banking and Finance 14, 1990), discretised over M dates
/// \f$ t_i = iT/M \f$:
///
/// \f[ \ln G = \frac{1}{M}\sum_{i=1}^{M} \ln S_{t_i} \f]
///
/// \f[ E[\ln G] = \ln S_0 + \Big(r-q-\tfrac{1}{2}\sigma^2\Big)\frac{T(M+1)}{2M} \f]
///
/// \f[ \operatorname{Var}[\ln G] = \frac{\sigma^2 T (M+1)(2M+1)}{6M^2} \f]
///
/// The variance follows from \f$ \operatorname{Var}(\sum_i W_{t_i}) = \sum_{i,j}\min(t_i,t_j) \f$
/// and \f$ \sum_{i,j}\min(i,j) = M(M+1)(2M+1)/6 \f$.
///
/// Two limits pin the formula, and both are tested:
///   - M = 1 places the only monitoring date at maturity, so G = S_T and the
///     price must equal Black-Scholes *exactly*. This ties the formula to the
///     Phase 1 engine, which is itself validated against Hull, Haug, mpmath and
///     QuantLib.
///   - M -> infinity gives the continuous case, where the variance tends to
///     \f$ \sigma^2 T/3 \f$ -- the classical result.
class GeometricAsianAnalyticEngine {
public:
    /// The average's lognormal law.
    ///
    /// Fails only if the monitoring count is such that the law degenerates; for
    /// any valid AsianOption it succeeds.
    [[nodiscard]] static GeometricAverageLaw law(const MarketState& market,
                                                 const AsianOption& option,
                                                 const BlackScholesModel& model) noexcept;

    /// Prices a geometric Asian option.
    ///
    /// Rejects an arithmetic option rather than pricing it as geometric, which
    /// would silently answer a different question with a plausible number.
    [[nodiscard]] static Result<PricingResult>
    price(const MarketState& market, const AsianOption& option, const BlackScholesModel& model);
};

}  // namespace diffusionworks
