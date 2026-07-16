#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/pde/boundary_conditions.hpp>

#include <fmt/format.h>

#include <cmath>
#include <utility>

namespace diffusionworks {

Result<std::vector<double>> terminal_condition(const EuropeanOption& option,
                                               const AssetGrid& grid) {
    const auto nodes = static_cast<std::size_t>(grid.nodes());
    std::vector<double> values(nodes, 0.0);

    for (std::size_t i = 0; i < nodes; ++i) {
        values[i] = option.payoff(grid.at(static_cast<std::int64_t>(i)));
        if (!std::isfinite(values[i])) {
            return Result<std::vector<double>>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("the payoff is not finite at node {} (S = {})",
                            i,
                            grid.at(static_cast<std::int64_t>(i))),
                "terminal_condition");
        }
    }

    return Result<std::vector<double>>::success(std::move(values));
}

double lower_boundary(const EuropeanOption& option, const MarketState& market, double tau) {
    switch (option.type()) {
        case OptionType::Call:
            // S = 0 is absorbing, so the call expires worthless with certainty.
            return 0.0;
        case OptionType::Put:
            // The holder receives K with certainty, but at expiry: discounted over
            // the remaining life tau, not over the elapsed time and not at all.
            return option.strike() * market.discount_factor(tau);
    }
    return 0.0;
}

double upper_boundary(const EuropeanOption& option,
                      const MarketState& market,
                      const AssetGrid& grid,
                      double tau) {
    switch (option.type()) {
        case OptionType::Call:
            // Asymptotic: the forward less the discounted strike. The dividend
            // discount applies to the asset leg and the rate discount to the cash
            // leg -- they are different rates and swapping them is a silent error
            // that only shows up when q != r.
            return grid.s_max() * std::exp(-market.dividend_yield() * tau) -
                   option.strike() * market.discount_factor(tau);
        case OptionType::Put:
            // Deep in the asset, the put is nearly certain to expire worthless.
            return 0.0;
    }
    return 0.0;
}

Result<double> upper_boundary_truncation_error(const EuropeanOption& option,
                                               const MarketState& market,
                                               const BlackScholesModel& model,
                                               const AssetGrid& grid,
                                               double tau) {
    // The true value at the boundary, from the analytic engine: a European option
    // struck at K with tau remaining, evaluated at S = S_max.
    const auto boundary_market =
        MarketState::create(grid.s_max(), market.rate(), market.dividend_yield());
    if (!boundary_market.ok()) {
        return Result<double>::failure(boundary_market.error());
    }
    const auto boundary_option = EuropeanOption::create(option.type(), option.strike(), tau);
    if (!boundary_option.ok()) {
        return Result<double>::failure(boundary_option.error());
    }

    const auto exact =
        BlackScholesAnalyticEngine::price(boundary_market.value(), boundary_option.value(), model);
    if (!exact.ok()) {
        return Result<double>::failure(exact.error());
    }

    return Result<double>::success(upper_boundary(option, market, grid, tau) - exact.value().value);
}

}  // namespace diffusionworks
