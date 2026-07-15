#pragma once

#include <diffusionworks/core/result.hpp>

namespace diffusionworks {

/// Observable market inputs, independent of any model or instrument.
///
/// Holds only what the market quotes: a spot level and two continuously
/// compounded carry rates. Volatility is deliberately absent -- it is a model
/// parameter, not a market observable, and lives in BlackScholesModel or
/// HestonModel. Keeping them apart is what lets one market state price against
/// several models.
///
/// Conventions:
///   - `rate` and `dividend_yield` are continuously compounded and annualised.
///   - `dividend_yield` is the continuous yield q. For an FX option it is the
///     foreign rate; for an equity index it approximates the dividend stream.
///
/// Construction is validated (ADR-006): use create(). There is no public
/// constructor, so a MarketState in hand is always a valid one.
class MarketState {
public:
    /// Validates and constructs.
    ///
    /// Requires spot > 0 and finite. Rates may be negative -- negative policy
    /// rates are real, and rejecting them would encode a market regime as a
    /// mathematical constraint -- but must be finite.
    [[nodiscard]] static Result<MarketState>
    create(double spot, double rate, double dividend_yield);

    [[nodiscard]] double spot() const noexcept { return spot_; }

    [[nodiscard]] double rate() const noexcept { return rate_; }

    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

    /// Discount factor \f$ e^{-rT} \f$.
    [[nodiscard]] double discount_factor(double maturity) const noexcept;

    /// Dividend discount factor \f$ e^{-qT} \f$.
    [[nodiscard]] double dividend_discount_factor(double maturity) const noexcept;

    /// Forward price \f$ F = S_0 e^{(r-q)T} \f$.
    [[nodiscard]] double forward(double maturity) const noexcept;

private:
    MarketState(double spot, double rate, double dividend_yield) noexcept
        : spot_(spot), rate_(rate), dividend_yield_(dividend_yield) {}

    double spot_;
    double rate_;
    double dividend_yield_;
};

}  // namespace diffusionworks
