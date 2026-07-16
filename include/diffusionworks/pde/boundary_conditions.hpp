#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/black_scholes.hpp>
#include <diffusionworks/pde/asset_grid.hpp>

#include <span>
#include <vector>

namespace diffusionworks {

/// The terminal condition: the payoff, sampled on the grid.
///
/// In backward time this is the *initial* condition, at tau = 0. The naming
/// follows the finance rather than the PDE -- it is the option's terminal payoff --
/// but the solver starts here and marches tau upward.
[[nodiscard]] Result<std::vector<double>> terminal_condition(const EuropeanOption& option,
                                                             const AssetGrid& grid);

/// The value at the S = 0 boundary, at backward time tau.
///
/// At S = 0 the asset is absorbed: geometric Brownian motion started at zero stays
/// at zero, so the terminal price is certainly zero. Hence
///
///     call: V(0, tau) = 0
///     put:  V(0, tau) = K e^{-r tau}
///
/// The put's boundary is the discounted strike, not the strike: the holder will
/// certainly receive K, but at expiry, so its value today discounts over the
/// remaining life tau. Using K undiscounted is a classic error that produces a
/// visibly wrong price only for large r*tau, and a subtly wrong one otherwise.
[[nodiscard]] double
lower_boundary(const EuropeanOption& option, const MarketState& market, double tau);

/// The value at the S = S_max boundary, at backward time tau.
///
/// This is an *asymptotic* condition, not an exact one, and that distinction is
/// the whole reason S_max sensitivity has to be measured (EXP-06).
///
/// For S_max far above the strike a call is nearly certain to finish in the money,
/// so its value approaches the forward less the discounted strike:
///
///     call: V(S_max, tau) ~ S_max e^{-q tau} - K e^{-r tau}
///     put:  V(S_max, tau) ~ 0
///
/// "Nearly certain" is doing the work. The true value at S_max is strictly greater
/// than this for a call -- the option still has some probability of finishing out
/// of the money, and that optionality has value -- so the condition truncates the
/// domain and introduces an error that decays as S_max grows but never vanishes.
/// MATHEMATICAL-SPEC section 12 writes it as an approximation for exactly this
/// reason, and the size of the truncation error is a measured quantity here rather
/// than an assumed-negligible one.
[[nodiscard]] double upper_boundary(const EuropeanOption& option,
                                    const MarketState& market,
                                    const AssetGrid& grid,
                                    double tau);

/// The exact truncation error of the upper boundary at backward time tau.
///
/// The difference between the asymptotic condition applied at S_max and the true
/// Black-Scholes value there. Computable because the true value is known
/// analytically for a European option -- which is precisely why the vanilla case is
/// the right place to measure how much the truncation costs, before Phase 7 relies
/// on the same boundary for barriers where no such reference exists.
[[nodiscard]] Result<double> upper_boundary_truncation_error(const EuropeanOption& option,
                                                             const MarketState& market,
                                                             const BlackScholesModel& model,
                                                             const AssetGrid& grid,
                                                             double tau);

}  // namespace diffusionworks
