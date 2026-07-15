#pragma once

#include <diffusionworks/core/result.hpp>

namespace diffusionworks {

/// Black-Scholes-Merton model parameters.
///
/// \f[ dS_t = (r-q) S_t\,dt + \sigma S_t\,dW_t \f]
///
/// The model contributes exactly one parameter, the volatility; r and q are
/// market inputs and live in MarketState. Splitting them this way keeps a single
/// market state usable across models.
class BlackScholesModel {
public:
    /// Validates and constructs.
    ///
    /// Requires volatility >= 0 and finite.
    ///
    /// Zero volatility is admitted rather than rejected: sigma -> 0 is a required
    /// limiting case (MATHEMATICAL-SPEC section 18) and is well defined -- the
    /// asset grows deterministically at the carry rate and the option is worth
    /// the discounted intrinsic value of its forward. Engines handle it
    /// explicitly rather than dividing by sigma*sqrt(T).
    [[nodiscard]] static Result<BlackScholesModel> create(double volatility);

    [[nodiscard]] double volatility() const noexcept { return volatility_; }

    /// Total standard deviation of log-spot over [0, T]: \f$ \sigma\sqrt{T} \f$.
    ///
    /// This grouping, not sigma alone, is what the pricing formulas divide by,
    /// so it is the quantity whose degeneracy engines must guard.
    [[nodiscard]] double total_volatility(double maturity) const noexcept;

private:
    explicit BlackScholesModel(double volatility) noexcept : volatility_(volatility) {}

    double volatility_;
};

}  // namespace diffusionworks
