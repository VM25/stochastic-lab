#include <diffusionworks/pde/black_scholes_operator.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "black_scholes_operator";

}  // namespace

Result<OperatorCoefficients> black_scholes_coefficients(const MarketState& market,
                                                        const BlackScholesModel& model,
                                                        const AssetGrid& grid) {
    const auto nodes = static_cast<std::size_t>(grid.nodes());
    const double variance_rate = model.volatility() * model.volatility();
    const double carry = market.rate() - market.dividend_yield();
    const double rate = market.rate();

    OperatorCoefficients coefficients;
    coefficients.a.assign(nodes, 0.0);
    coefficients.b.assign(nodes, 0.0);
    coefficients.c.assign(nodes, 0.0);

    // Interior nodes only. Node 0 and node N-1 carry boundary conditions, not
    // operator rows, and their coefficients stay zero so that an index means the
    // same thing here as in the grid.
    for (std::size_t i = 1; i + 1 < nodes; ++i) {
        const auto index = static_cast<double>(i);

        // In terms of the node index, not the price: sigma^2 S_i^2 / dS^2 is
        // exactly sigma^2 i^2 on a uniform grid with S_0 = 0, and the division by
        // dS^2 -- which would dominate the rounding error at fine grids -- cancels
        // analytically rather than numerically.
        const double diffusion = 0.5 * variance_rate * index * index;
        const double convection = 0.5 * carry * index;

        coefficients.a[i] = diffusion - convection;
        coefficients.b[i] = -2.0 * diffusion - rate;
        coefficients.c[i] = diffusion + convection;
    }

    for (std::size_t i = 0; i < nodes; ++i) {
        if (!std::isfinite(coefficients.a[i]) || !std::isfinite(coefficients.b[i]) ||
            !std::isfinite(coefficients.c[i])) {
            return Result<OperatorCoefficients>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("the operator coefficients are not finite at node {} (a={}, b={}, "
                            "c={}); check sigma={} and the grid's node count ({})",
                            i,
                            coefficients.a[i],
                            coefficients.b[i],
                            coefficients.c[i],
                            model.volatility(),
                            grid.nodes()),
                kContext);
        }
    }

    return Result<OperatorCoefficients>::success(std::move(coefficients));
}

OperatorDiagnostics diagnose_operator(const MarketState& market,
                                      const BlackScholesModel& model,
                                      const AssetGrid& grid,
                                      const OperatorCoefficients& coefficients) {
    OperatorDiagnostics diagnostics;

    const double variance_rate = model.volatility() * model.volatility();
    const double carry = market.rate() - market.dividend_yield();

    // a_i >= 0 requires (1/2) sigma^2 i^2 >= (1/2) (r-q) i, i.e. i >= (r-q)/sigma^2.
    // The cell-Peclet condition, in index form.
    //
    // At sigma = 0 the operator has no diffusion at all and every interior node is
    // pure convection: the threshold is infinite, which is the honest answer rather
    // than a division by zero.
    diagnostics.peclet_threshold_index =
        variance_rate > 0.0 ? carry / variance_rate : std::numeric_limits<double>::infinity();

    const auto nodes = static_cast<std::size_t>(grid.nodes());
    for (std::size_t i = 1; i + 1 < nodes; ++i) {
        if (coefficients.a[i] < 0.0) {
            diagnostics.negative_sub_diagonal_nodes.push_back(static_cast<std::int64_t>(i));
        }
        if (coefficients.c[i] < 0.0) {
            diagnostics.negative_super_diagonal_nodes.push_back(static_cast<std::int64_t>(i));
        }
    }

    return diagnostics;
}

Result<double> explicit_stability_limit(const OperatorCoefficients& coefficients) {
    const std::size_t n = coefficients.b.size();
    if (n < 3) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the operator has {} nodes; at least 3 are needed for an interior row", n),
            "explicit_stability_limit");
    }

    // Read off the assembled diagonal rather than restating a textbook formula.
    // The two agree for this discretisation -- max|b_i| is exactly sigma^2 N'^2 + r
    // for the largest interior index N' -- but only one of them stays correct if
    // the discretisation changes, and it is not the formula.
    double worst = 0.0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        worst = std::max(worst, std::abs(coefficients.b[i]));
    }

    if (worst <= 0.0) {
        // Every interior diagonal vanished: sigma = 0 and r = 0, so the operator is
        // pure convection with no zeroth-order term. The explicit iteration has no
        // amplification to bound and the limit is unbounded. Reported as infinity
        // rather than as a failure, since it is a true statement about a degenerate
        // but valid operator.
        return Result<double>::success(std::numeric_limits<double>::infinity());
    }

    return Result<double>::success(1.0 / worst);
}

}  // namespace diffusionworks
