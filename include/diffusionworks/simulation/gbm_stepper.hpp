#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/simulation/time_grid.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

namespace diffusionworks {

/// How a step of the SDE is approximated.
enum class DiscretizationScheme : std::uint8_t {
    /// The exact log-normal transition. No discretisation error: the simulated
    /// distribution is the true one at every step size.
    Exact,

    /// Euler-Maruyama. Strong order 0.5, weak order 1.
    EulerMaruyama,

    /// Milstein. Strong order 1, weak order 1.
    Milstein,
};

[[nodiscard]] std::string_view to_string(DiscretizationScheme scheme) noexcept;

[[nodiscard]] std::optional<DiscretizationScheme>
parse_discretization_scheme(std::string_view text) noexcept;

/// Advances geometric Brownian motion by one step of a fixed grid.
///
/// Under the risk-neutral measure (MATHEMATICAL-SPEC section 2):
///
/// \f[ dS_t = (r-q) S_t\,dt + \sigma S_t\,dW_t \f]
///
/// The three schemes, with \f$ Z \sim N(0,1) \f$ and \f$ \Delta W = \sqrt{\Delta t}\,Z \f$:
///
/// Exact:
/// \f[ S_{t+\Delta} = S_t \exp\!\big((r-q-\tfrac{1}{2}\sigma^2)\Delta t
///                                   + \sigma\sqrt{\Delta t}\,Z\big) \f]
///
/// Euler-Maruyama:
/// \f[ S_{t+\Delta} = S_t\big(1 + (r-q)\Delta t + \sigma\sqrt{\Delta t}\,Z\big) \f]
///
/// Milstein:
/// \f[ S_{t+\Delta} = S_t\big(1 + (r-q)\Delta t + \sigma\sqrt{\Delta t}\,Z
///                             + \tfrac{1}{2}\sigma^2\Delta t\,(Z^2-1)\big) \f]
///
/// The Milstein correction is \f$ \tfrac{1}{2} b b' ((\Delta W)^2 - \Delta t) \f$
/// with \f$ b(S) = \sigma S \f$ and \f$ b'(S) = \sigma \f$, which for GBM reduces
/// to the term above.
///
/// Why the exact scheme exists alongside the others
/// ------------------------------------------------
/// Not to price with -- for a European option the analytic formula already does
/// that, and better. It exists to be the *reference path*. Strong convergence is
/// the error of a discretised path against the true path driven by the same
/// Brownian motion, so measuring it requires that true path. GBM is one of the
/// few models where it is available in closed form, which is what makes EXP-02
/// possible at all (VALIDATION-PLAN section 8).
///
/// Constants are precomputed once per simulation rather than per step: a step
/// runs 10^6 to 10^9 times per experiment, and recomputing sqrt(dt) inside it
/// would dominate the arithmetic that matters.
class GbmStepper {
public:
    [[nodiscard]] static GbmStepper create(const MarketState& market,
                                           const BlackScholesModel& model,
                                           const TimeGrid& grid,
                                           DiscretizationScheme scheme) noexcept;

    /// Advances one step given a standard normal shock.
    ///
    /// Unchecked and branch-light by design: this is the innermost loop of every
    /// experiment. Validity of the resulting state is the caller's concern, and
    /// GbmPathGenerator checks it once per step rather than hiding a check here.
    [[nodiscard]] double advance(double spot, double z) const noexcept {
        switch (scheme_) {
            case DiscretizationScheme::Exact:
                return spot * std::exp(drift_term_ + diffusion_term_ * z);
            case DiscretizationScheme::EulerMaruyama:
                return spot * (1.0 + drift_term_ + diffusion_term_ * z);
            case DiscretizationScheme::Milstein:
                return spot *
                       (1.0 + drift_term_ + diffusion_term_ * z + milstein_term_ * (z * z - 1.0));
        }
        return spot;
    }

    [[nodiscard]] DiscretizationScheme scheme() const noexcept { return scheme_; }

    /// \f$ (r-q-\tfrac{1}{2}\sigma^2)\Delta t \f$ for the exact scheme, or
    /// \f$ (r-q)\Delta t \f$ for the others. They differ because the exact scheme
    /// steps log-spot while the others step spot, and the Ito correction belongs
    /// to the former only.
    [[nodiscard]] double drift_term() const noexcept { return drift_term_; }

    /// \f$ \sigma\sqrt{\Delta t} \f$.
    [[nodiscard]] double diffusion_term() const noexcept { return diffusion_term_; }

private:
    GbmStepper(DiscretizationScheme scheme,
               double drift_term,
               double diffusion_term,
               double milstein_term) noexcept
        : scheme_(scheme), drift_term_(drift_term), diffusion_term_(diffusion_term),
          milstein_term_(milstein_term) {}

    DiscretizationScheme scheme_;
    double drift_term_;
    double diffusion_term_;

    /// \f$ \tfrac{1}{2}\sigma^2\Delta t \f$. Zero for the other schemes.
    double milstein_term_;
};

}  // namespace diffusionworks
