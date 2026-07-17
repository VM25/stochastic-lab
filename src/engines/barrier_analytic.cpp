#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BarrierAnalyticEngine";

/// The vanilla value, from the Phase 1 engine.
///
/// Reused rather than reimplemented. That engine is validated against Hull, Haug,
/// a 50-digit mpmath oracle, and QuantLib, so every barrier price inherits that
/// chain -- and a second inline Black-Scholes here would be a second thing to keep
/// right.
Result<double> vanilla_value(const MarketState& market,
                             const BlackScholesModel& model,
                             OptionType type,
                             double strike,
                             double maturity) {
    const auto option = EuropeanOption::create(type, strike, maturity);
    if (!option.ok()) {
        return Result<double>::failure(option.error());
    }
    const auto priced = BlackScholesAnalyticEngine::price(market, option.value(), model);
    if (!priced.ok()) {
        return Result<double>::failure(priced.error());
    }
    return Result<double>::success(priced.value().value);
}

}  // namespace

Result<PricingResult> BarrierAnalyticEngine::price(const MarketState& market,
                                                   const BarrierOption& option,
                                                   const BlackScholesModel& model) {
    const auto start = std::chrono::steady_clock::now();

    if (option.convention() != MonitoringConvention::Continuous) {
        // Refused rather than approximated. A discretely monitored barrier is a
        // different contract worth a different amount -- more, for a knock-out,
        // because the observation dates miss excursions -- and the gap closes only
        // as O(1/sqrt(m)). Returning this value for it would be a plausible answer
        // to a question that was not asked.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("this engine prices continuous monitoring, but the option specifies {} "
                        "monitoring. Those are different contracts: discrete observation misses "
                        "excursions between fixes, so a knock-out is worth more than its "
                        "continuous value, and the difference decays only as 1/sqrt(m). Use the "
                        "Monte Carlo engine.",
                        to_string(option.convention())),
            kContext);
    }
    if (option.type() != OptionType::Call) {
        return Result<PricingResult>::failure(
            ErrorCode::NotImplemented,
            "only barrier calls are implemented; the put formulae are a separate set of cases",
            kContext);
    }

    const double spot = market.spot();
    const double strike = option.strike();
    const double barrier = option.barrier();
    const double maturity = option.maturity();
    const double rate = market.rate();
    const double dividend = market.dividend_yield();
    const double sigma = model.volatility();

    PricingResult result;
    result.method = "barrier_analytic_reiner_rubinstein";
    result.add_diagnostic("monitoring", std::string(to_string(option.convention())));
    result.add_diagnostic("barrier_type", std::string(to_string(option.barrier_type())));
    result.add_diagnostic("barrier", barrier);

    // --- Already breached ---------------------------------------------------
    //
    // A price, not an error. The contract has already resolved: a knock-out is
    // dead and a knock-in has become the vanilla.
    if (option.breaches(spot)) {
        if (is_knock_out(option.barrier_type())) {
            result.value = 0.0;
            result.add_warning(fmt::format(
                "the barrier at {} is already breached by the spot at {}, so this knock-out is "
                "worthless; the price is exact rather than approximate",
                barrier,
                spot));
        } else {
            const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
            if (!vanilla.ok()) {
                return Result<PricingResult>::failure(vanilla.error());
            }
            result.value = vanilla.value();
            result.add_warning(fmt::format(
                "the barrier at {} is already breached by the spot at {}, so this knock-in has "
                "become the vanilla; the price is exact rather than approximate",
                barrier,
                spot));
        }
        result.add_diagnostic("already_breached", true);
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

    // --- Degenerate limits --------------------------------------------------
    if (maturity <= 0.0 || sigma <= 0.0) {
        // With no time or no diffusion the path cannot move, so an unbreached
        // barrier is never breached: a knock-out is the vanilla and a knock-in is
        // worthless. The Phase 1 engine handles both degenerate cases of the
        // vanilla itself.
        const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
        if (!vanilla.ok()) {
            return Result<PricingResult>::failure(vanilla.error());
        }
        result.value = is_knock_out(option.barrier_type()) ? vanilla.value() : 0.0;
        result.add_warning(fmt::format(
            "degenerate limit (maturity={}, sigma={}): the path cannot reach the barrier, so the "
            "knock-{} is worth {}",
            maturity,
            sigma,
            is_knock_out(option.barrier_type()) ? "out" : "in",
            is_knock_out(option.barrier_type()) ? "the vanilla" : "zero"));
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

    // --- Reiner-Rubinstein --------------------------------------------------
    //
    // Four building blocks, from which every single-barrier case is assembled.
    // Write s = sigma sqrt(T), b = r - q for the cost of carry,
    // mu = (b - sigma^2/2)/sigma^2, and eta = +1 for a down barrier, -1 for an up
    // barrier. Let w_S = (B/S)^{2(mu+1)} and w_K = (B/S)^{2mu} be the reflection
    // weights. Then:
    //
    //   x1 = ln(S/K)/s      + (1+mu) s        y1 = ln(B^2/(SK))/s + (1+mu) s
    //   x2 = ln(S/B)/s      + (1+mu) s        y2 = ln(B/S)/s      + (1+mu) s
    //
    //   A = S e^{-qT} N(x1)     - K e^{-rT} N(x1 - s)
    //   B = S e^{-qT} N(x2)     - K e^{-rT} N(x2 - s)
    //   C = S e^{-qT} w_S N(eta y1) - K e^{-rT} w_K N(eta y1 - eta s)
    //   D = S e^{-qT} w_S N(eta y2) - K e^{-rT} w_K N(eta y2 - eta s)
    //
    // (Calls only here, so Haug's phi = +1 throughout and drops out.)
    //
    // A is the vanilla: x1 is d1 once (1+mu) sigma sqrt T is expanded. C and D are
    // the reflected vanilla -- the reflection principle maps paths touching B onto
    // paths started at B^2/S, and (B/S)^{2(mu+1)} is the Radon-Nikodym factor
    // between them. B and D are the same pair with the *barrier* rather than the
    // strike setting the boundary of the payoff region, which is why the assembly
    // below turns on K vs B rather than on the barrier direction alone.
    //
    // Knock-out calls:
    //
    //   down, K > B : A - C          up, K > B : 0
    //   down, K < B : B - D          up, K < B : A - B + C - D
    //
    // The up, K > B case is exactly zero rather than approximately so: paying needs
    // S_T > K > B, and any such path crossed B and knocked out. It is a price.
    //
    // Validated at 1e-14 against QuantLib on both branches of both directions, and
    // separately against a 40-digit mpmath evaluation. The down-and-out K > B value
    // 8.665471658245668 is additionally confirmed by a bridge-corrected simulation
    // -- a route sharing no algebra with this formula.
    const double variance_rate = sigma * sigma;
    const double sqrt_maturity = std::sqrt(maturity);
    const double sigma_root_t = sigma * sqrt_maturity;
    const double mu = (rate - dividend - 0.5 * variance_rate) / variance_rate;
    const double drift_adjustment = (1.0 + mu) * sigma_root_t;

    // The one parameter that reflects every formula. Down barriers read the lower
    // tail, up barriers the upper one.
    const double eta = is_down_barrier(option.barrier_type()) ? 1.0 : -1.0;

    const double x1 = std::log(spot / strike) / sigma_root_t + drift_adjustment;
    const double x2 = std::log(spot / barrier) / sigma_root_t + drift_adjustment;
    const double y1 =
        std::log(barrier * barrier / (spot * strike)) / sigma_root_t + drift_adjustment;
    const double y2 = std::log(barrier / spot) / sigma_root_t + drift_adjustment;

    const double asset_discount = std::exp(-dividend * maturity);
    const double cash_discount = market.discount_factor(maturity);
    const double ratio = barrier / spot;
    const double reflected_asset_weight = std::pow(ratio, 2.0 * (mu + 1.0));
    const double reflected_cash_weight = std::pow(ratio, 2.0 * mu);

    const auto direct_term = [&](double x) {
        return spot * asset_discount * norm_cdf(x) -
               strike * cash_discount * norm_cdf(x - sigma_root_t);
    };
    const auto reflected_term = [&](double y) {
        return spot * asset_discount * reflected_asset_weight * norm_cdf(eta * y) -
               strike * cash_discount * reflected_cash_weight *
                   norm_cdf(eta * y - eta * sigma_root_t);
    };

    const double a_term = direct_term(x1);
    const double b_term = direct_term(x2);
    const double c_term = reflected_term(y1);
    const double d_term = reflected_term(y2);

    const bool strike_above_barrier = strike > barrier;
    double knock_out = 0.0;
    if (is_down_barrier(option.barrier_type())) {
        knock_out = strike_above_barrier ? a_term - c_term : b_term - d_term;
    } else {
        knock_out = strike_above_barrier ? 0.0 : a_term - b_term + c_term - d_term;
    }

    if (is_knock_out(option.barrier_type())) {
        result.value = knock_out;
        if (!is_down_barrier(option.barrier_type()) && strike_above_barrier) {
            // Zero can look like a defect, so the record says why it is not one.
            result.add_warning(fmt::format(
                "an up-and-out call struck at {} above its barrier at {} is worth exactly zero: "
                "paying requires the terminal price to exceed the strike and hence the barrier, "
                "and every such path knocked out first. This is a price, not a failure to compute "
                "one.",
                strike,
                barrier));
        }
    } else {
        // In-out parity: knock-in = vanilla - knock-out, exactly, when the
        // contracts and conventions match. Computed rather than given its own
        // formula, so the identity cannot be violated by a transcription error in
        // a second closed form.
        //
        // It also *is* the published formula, not merely a shortcut to it: A is the
        // vanilla, so vanilla - (A - B + C - D) collapses to B - C + D, which is
        // Haug's up-and-in call, and vanilla - 0 collapses to A, which is his other
        // branch. The identity and the closed form agree by construction.
        const auto vanilla = vanilla_value(market, model, option.type(), strike, maturity);
        if (!vanilla.ok()) {
            return Result<PricingResult>::failure(vanilla.error());
        }
        result.value = vanilla.value() - knock_out;
        result.add_diagnostic("vanilla", vanilla.value());
        result.add_diagnostic("knock_out", knock_out);
    }

    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the barrier price is not finite (S={}, K={}, B={}, T={}, sigma={})",
                        spot,
                        strike,
                        barrier,
                        maturity,
                        sigma),
            kContext);
    }
    if (result.value < 0.0) {
        // The reflected term is subtracted, so cancellation can drive the result
        // slightly negative near the barrier where the two nearly balance.
        // Reported rather than clamped: a negative barrier price is impossible,
        // and clamping would hide exactly the near-barrier fragility Phase 7
        // exists to document.
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the barrier price is negative ({}), which the contract cannot be. The "
                        "reflected term nearly cancels the vanilla near the barrier (S={}, B={}, "
                        "S/B={}), and this is that cancellation.",
                        result.value,
                        spot,
                        barrier,
                        spot / barrier),
            kContext);
    }

    result.add_diagnostic("x1", x1);
    result.add_diagnostic("x2", x2);
    result.add_diagnostic("y1", y1);
    result.add_diagnostic("y2", y2);
    result.add_diagnostic("reflection_exponent", 2.0 * (mu + 1.0));
    result.add_diagnostic("spot_to_barrier_ratio", spot / barrier);
    // Which of Haug's branches produced this number. The two differ by which of the
    // strike and the barrier bounds the payoff region, and a reader comparing
    // against a reference needs to know which one was taken.
    result.add_diagnostic(
        "branch",
        std::string(strike_above_barrier ? "strike_above_barrier" : "barrier_above_strike"));

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
