#include <diffusionworks/simulation/gbm_stepper.hpp>

namespace diffusionworks {

std::string_view to_string(DiscretizationScheme scheme) noexcept {
    switch (scheme) {
        case DiscretizationScheme::Exact:
            return "exact";
        case DiscretizationScheme::EulerMaruyama:
            return "euler_maruyama";
        case DiscretizationScheme::Milstein:
            return "milstein";
    }
    return "unknown";
}

std::optional<DiscretizationScheme> parse_discretization_scheme(std::string_view text) noexcept {
    if (text == "exact") {
        return DiscretizationScheme::Exact;
    }
    if (text == "euler_maruyama") {
        return DiscretizationScheme::EulerMaruyama;
    }
    if (text == "milstein") {
        return DiscretizationScheme::Milstein;
    }
    return std::nullopt;
}

GbmStepper GbmStepper::create(const MarketState& market,
                              const BlackScholesModel& model,
                              const TimeGrid& grid,
                              DiscretizationScheme scheme) noexcept {
    const double carry = market.rate() - market.dividend_yield();
    const double volatility = model.volatility();
    const double step_size = grid.step_size();

    // The exact scheme steps log-spot, so its drift carries the Ito correction
    // -sigma^2/2. Euler and Milstein step spot itself, where the drift is the
    // SDE's own (r-q) and no correction applies. Using one for the other is a
    // classic error that leaves prices plausible but biased, so the two are
    // formed separately rather than by adjusting a shared value.
    const double drift_term = (scheme == DiscretizationScheme::Exact)
                                  ? (carry - 0.5 * volatility * volatility) * step_size
                                  : carry * step_size;

    const double diffusion_term = volatility * grid.sqrt_step_size();

    const double milstein_term = (scheme == DiscretizationScheme::Milstein)
                                     ? 0.5 * volatility * volatility * step_size
                                     : 0.0;

    return {scheme, drift_term, diffusion_term, milstein_term};
}

}  // namespace diffusionworks
