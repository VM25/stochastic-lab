#pragma once

#include <diffusionworks/core/result.hpp>

namespace diffusionworks {

/// Heston stochastic-volatility model parameters.
///
/// Under the risk-neutral measure (MATHEMATICAL-SPEC section 4):
///
/// \f[ dS_t = (r-q) S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \f]
/// \f[ dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v \f]
/// \f[ dW_t^S\,dW_t^v = \rho\,dt \f]
///
/// The variance follows a Cox-Ingersoll-Ross process: mean-reverting to \f$\theta\f$
/// at speed \f$\kappa\f$, with volatility-of-variance \f$\xi\f$. As with
/// BlackScholesModel, the rates r and q are market inputs and live in MarketState,
/// so one market state serves every model.
///
/// The Feller condition
/// --------------------
/// \f$ 2\kappa\theta \ge \xi^2 \f$ is the condition under which the variance stays
/// strictly positive; when it fails the process can touch zero. This model
/// *reports* whether it holds and never rejects a parameter set for failing it
/// (MATHEMATICAL-SPEC section 4): a Feller-violating set is a legitimate, frequently
/// calibrated regime, and refusing it would encode a numerical preference as a
/// validity rule. The consequences of the violation are the simulation's problem,
/// not the parameters'.
class HestonModel {
public:
    /// Validates and constructs.
    ///
    /// Requires, all finite: v0 >= 0, kappa >= 0, theta >= 0, xi >= 0, and
    /// rho in [-1, 1]. The boundaries are admitted rather than excluded --
    /// v0 = 0 (starting at zero variance), xi = 0 (deterministic variance,
    /// reducing to time-dependent Black-Scholes), and |rho| = 1 (perfectly
    /// correlated shocks) are all limiting cases a reference engine must handle
    /// rather than refuse.
    ///
    /// The Feller condition is *not* a validation: a set that violates it is
    /// constructed successfully and reports the violation through feller_ratio and
    /// satisfies_feller.
    [[nodiscard]] static Result<HestonModel> create(double initial_variance,
                                                    double mean_reversion,
                                                    double long_run_variance,
                                                    double vol_of_variance,
                                                    double correlation);

    [[nodiscard]] double initial_variance() const noexcept { return initial_variance_; }

    [[nodiscard]] double mean_reversion() const noexcept { return mean_reversion_; }

    [[nodiscard]] double long_run_variance() const noexcept { return long_run_variance_; }

    [[nodiscard]] double vol_of_variance() const noexcept { return vol_of_variance_; }

    [[nodiscard]] double correlation() const noexcept { return correlation_; }

    /// The ratio \f$ 2\kappa\theta / \xi^2 \f$. At least 1 exactly when the Feller
    /// condition holds; below 1 when it fails. Reported as a number rather than a
    /// bare flag so a caller can see *how far* a set is from the boundary, which is
    /// what governs how badly a naive simulation misbehaves.
    ///
    /// Infinite when xi = 0 (the variance is deterministic, so it cannot reach
    /// zero from diffusion), which is the correct limit.
    [[nodiscard]] double feller_ratio() const noexcept;

    /// Whether \f$ 2\kappa\theta \ge \xi^2 \f$.
    [[nodiscard]] bool satisfies_feller() const noexcept;

private:
    HestonModel(double initial_variance,
                double mean_reversion,
                double long_run_variance,
                double vol_of_variance,
                double correlation) noexcept
        : initial_variance_(initial_variance), mean_reversion_(mean_reversion),
          long_run_variance_(long_run_variance), vol_of_variance_(vol_of_variance),
          correlation_(correlation) {}

    double initial_variance_;
    double mean_reversion_;
    double long_run_variance_;
    double vol_of_variance_;
    double correlation_;
};

}  // namespace diffusionworks
