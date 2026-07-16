#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/numerics/tridiagonal.hpp>
#include <diffusionworks/pde/black_scholes_operator.hpp>
#include <diffusionworks/pde/boundary_conditions.hpp>
#include <diffusionworks/simulation/time_grid.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "FiniteDifferenceEngine";

/// The default upper truncation, as a multiple of the strike.
///
/// Four standard deviations of log-price above the strike would be the principled
/// choice, but it depends on sigma and T and would make the grid's geometry vary
/// with the model -- which makes a convergence sweep across parameters compare
/// different domains. A fixed multiple keeps the domain fixed and the comparison
/// clean, and EXP-06 measures what the truncation actually costs rather than
/// assuming this is enough.
constexpr double kDefaultSmaxMultiple = 4.0;

}  // namespace

const char* to_string(PdeScheme scheme) noexcept {
    switch (scheme) {
        case PdeScheme::Explicit:
            return "explicit";
        case PdeScheme::Implicit:
            return "implicit";
        case PdeScheme::CrankNicolson:
            return "crank_nicolson";
    }
    return "unknown";
}

Result<double> interpolate_linear(const AssetGrid& grid, std::span<const double> values, double s) {
    if (values.size() != static_cast<std::size_t>(grid.nodes())) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the value vector has {} entries but the grid has {} nodes",
                        values.size(),
                        grid.nodes()),
            "interpolate_linear");
    }
    if (!std::isfinite(s) || s < 0.0 || s > grid.s_max()) {
        // Refused rather than clamped or extrapolated. A spot outside the grid was
        // never solved for, and answering about the nearest boundary instead would
        // return a number with no relationship to the question.
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("s = {} lies outside the grid [0, {}]; the PDE was not solved there",
                        s,
                        grid.s_max()),
            "interpolate_linear");
    }

    const double position = s / grid.spacing();
    auto lower = static_cast<std::int64_t>(std::floor(position));
    lower = std::clamp(lower, std::int64_t{0}, grid.nodes() - 2);

    const double s_lower = grid.at(lower);
    const double s_upper = grid.at(lower + 1);
    const double v_lower = values[static_cast<std::size_t>(lower)];
    const double v_upper = values[static_cast<std::size_t>(lower + 1)];

    // Guard the degenerate span rather than dividing by it. Cannot arise on a
    // validated grid, but the division would produce a NaN price if it did.
    const double span = s_upper - s_lower;
    if (!(span > 0.0)) {
        return Result<double>::success(v_lower);
    }

    const double weight = (s - s_lower) / span;
    return Result<double>::success(v_lower + weight * (v_upper - v_lower));
}

Result<FiniteDifferenceEngine::Solution>
FiniteDifferenceEngine::solve(const MarketState& market,
                              const EuropeanOption& option,
                              const BlackScholesModel& model,
                              const PdeConfig& config) {
    if (option.maturity() <= 0.0) {
        return Result<Solution>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("the PDE cannot be stepped over a zero maturity ({}); the payoff is the "
                        "price at expiry and the analytic engine returns it exactly",
                        option.maturity()),
            kContext);
    }
    if (config.rannacher.count > 0 && config.scheme != PdeScheme::CrankNicolson) {
        // Refused rather than ignored. Rannacher smoothing means "take the first
        // steps implicitly instead of Crank-Nicolson"; asking for it on a scheme
        // that is already implicit, or on the explicit scheme, is a
        // misunderstanding, and silently dropping it would leave the caller
        // believing they got something they did not.
        return Result<Solution>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("Rannacher smoothing applies to Crank-Nicolson, not to the {} scheme: it "
                        "replaces the first Crank-Nicolson steps with implicit ones",
                        to_string(config.scheme)),
            kContext);
    }
    if (config.rannacher.count > config.time_steps) {
        return Result<Solution>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("{} Rannacher steps were requested but the run has only {} steps; the "
                        "result would be a fully implicit solve reported as Crank-Nicolson",
                        config.rannacher.count,
                        config.time_steps),
            kContext);
    }

    const double s_max = config.s_max.value_or(kDefaultSmaxMultiple * option.strike());

    auto grid = config.align_strike_to_node
                    ? AssetGrid::with_strike_on_node(s_max, config.asset_nodes, option.strike())
                    : AssetGrid::uniform(s_max, config.asset_nodes);
    if (!grid.ok()) {
        return Result<Solution>::failure(grid.error());
    }

    const auto time = TimeGrid::uniform(option.maturity(), config.time_steps);
    if (!time.ok()) {
        return Result<Solution>::failure(time.error());
    }

    const auto coefficients = black_scholes_coefficients(market, model, grid.value());
    if (!coefficients.ok()) {
        return Result<Solution>::failure(coefficients.error());
    }
    const auto operator_diagnostics =
        diagnose_operator(market, model, grid.value(), coefficients.value());

    auto values = terminal_condition(option, grid.value());
    if (!values.ok()) {
        return Result<Solution>::failure(values.error());
    }
    std::vector<double> current = std::move(values).value();

    const double dtau = time.value().step_size();
    const auto stability_limit = explicit_stability_limit(coefficients.value());
    if (!stability_limit.ok()) {
        return Result<Solution>::failure(stability_limit.error());
    }

    PdeDiagnostics diagnostics;
    diagnostics.asset_nodes = grid.value().nodes();
    diagnostics.time_steps = config.time_steps;
    diagnostics.s_max = grid.value().s_max();
    diagnostics.asset_spacing = grid.value().spacing();
    diagnostics.time_step = dtau;
    diagnostics.explicit_stability_ratio = dtau / stability_limit.value();
    diagnostics.sign_structure_holds = operator_diagnostics.sign_structure_holds();
    diagnostics.peclet_violating_nodes =
        static_cast<std::int64_t>(operator_diagnostics.negative_sub_diagonal_nodes.size() +
                                  operator_diagnostics.negative_super_diagonal_nodes.size());

    // The highest violating node, in price terms. This is the quantity that makes
    // the violation judgeable: it shrinks with dS, so refinement confines it to an
    // ever-smaller neighbourhood of S = 0.
    std::int64_t highest_violating_node = 0;
    for (const std::int64_t node : operator_diagnostics.negative_sub_diagonal_nodes) {
        highest_violating_node = std::max(highest_violating_node, node);
    }
    for (const std::int64_t node : operator_diagnostics.negative_super_diagonal_nodes) {
        highest_violating_node = std::max(highest_violating_node, node);
    }
    diagnostics.peclet_violating_max_s =
        highest_violating_node > 0 ? grid.value().at(highest_violating_node) : 0.0;
    diagnostics.rannacher_steps = config.rannacher.count;

    if (config.scheme == PdeScheme::Explicit) {
        diagnostics.explicit_stable = diagnostics.explicit_stability_ratio <= 1.0;
    }

    const auto n = static_cast<std::size_t>(grid.value().nodes());
    const auto& a = coefficients.value().a;
    const auto& b = coefficients.value().b;
    const auto& c = coefficients.value().c;

    std::vector<double> next(n, 0.0);
    double worst_pivot = std::numeric_limits<double>::infinity();
    double worst_residual = 0.0;
    bool solved_any = false;
    diagnostics.most_negative_value =
        std::min(0.0, *std::min_element(current.begin(), current.end()));

    for (std::int64_t step = 1; step <= config.time_steps; ++step) {
        const double tau = time.value().time_at(step);
        const double lower = lower_boundary(option, market, tau);
        const double upper = upper_boundary(option, market, grid.value(), tau);

        // The first `rannacher.count` steps run fully implicit; the rest run under
        // the configured scheme. With count = 0 this is exactly the configured
        // scheme throughout.
        const PdeScheme step_scheme =
            (step <= config.rannacher.count) ? PdeScheme::Implicit : config.scheme;

        if (step_scheme == PdeScheme::Explicit) {
            next[0] = lower;
            next[n - 1] = upper;
            for (std::size_t i = 1; i + 1 < n; ++i) {
                next[i] = current[i] + dtau * (a[i] * current[i - 1] + b[i] * current[i] +
                                               c[i] * current[i + 1]);
            }
        } else {
            const double theta = (step_scheme == PdeScheme::CrankNicolson) ? 0.5 : 1.0;

            TridiagonalSystem system;
            system.lower.assign(n, 0.0);
            system.diagonal.assign(n, 0.0);
            system.upper.assign(n, 0.0);
            system.rhs.assign(n, 0.0);

            // The boundary rows are Dirichlet: identity on the diagonal, the
            // boundary value on the right. Written as rows of the system rather
            // than substituted into the neighbours, so the matrix the solver sees
            // is the matrix the scheme describes.
            system.diagonal[0] = 1.0;
            system.rhs[0] = lower;
            system.diagonal[n - 1] = 1.0;
            system.rhs[n - 1] = upper;

            for (std::size_t i = 1; i + 1 < n; ++i) {
                // (I - theta*dtau*L) V^{n+1} = (I + (1-theta)*dtau*L) V^n
                system.lower[i] = -theta * dtau * a[i];
                system.diagonal[i] = 1.0 - theta * dtau * b[i];
                system.upper[i] = -theta * dtau * c[i];

                const double explicit_part =
                    (1.0 - theta) * dtau *
                    (a[i] * current[i - 1] + b[i] * current[i] + c[i] * current[i + 1]);
                system.rhs[i] = current[i] + explicit_part;
            }

            const auto solution = solve_tridiagonal(system);
            if (!solution.ok()) {
                return Result<Solution>::failure(solution.error());
            }
            solved_any = true;
            worst_pivot = std::min(worst_pivot, solution.value().diagnostics.smallest_pivot_ratio);

            const auto residual = tridiagonal_residual(system, solution.value().values);
            if (!residual.ok()) {
                return Result<Solution>::failure(residual.error());
            }
            worst_residual = std::max(worst_residual, residual.value());

            next = solution.value().values;
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(next[i])) {
                return Result<Solution>::failure(
                    ErrorCode::NonFiniteValue,
                    fmt::format(
                        "the solution went non-finite at node {} on step {} of {} (value {}). The "
                        "{} scheme was running at dtau/dtau_max = {:.3f}; for the explicit scheme "
                        "a ratio above 1 is exactly the instability this indicates.",
                        i,
                        step,
                        config.time_steps,
                        next[i],
                        to_string(step_scheme),
                        diagnostics.explicit_stability_ratio),
                    kContext);
            }
        }

        diagnostics.most_negative_value =
            std::min(diagnostics.most_negative_value, *std::min_element(next.begin(), next.end()));

        current.swap(next);
    }

    if (solved_any) {
        diagnostics.worst_pivot_ratio = worst_pivot;
        diagnostics.worst_residual = worst_residual;
    }

    // Convexity. A vanilla European value function is convex in S, so its second
    // difference is non-negative. Ringing shows up as violations; counting them
    // distinguishes an oscillating scheme from a merely inaccurate one, which a
    // price alone cannot.
    //
    // The tolerance scales with the value: a strict > 0 test would count rounding
    // noise in the flat far-field region, where the second difference is
    // legitimately zero to many digits.
    const double value_scale = std::max(1.0, *std::max_element(current.begin(), current.end()));
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double second_difference = current[i + 1] - 2.0 * current[i] + current[i - 1];
        if (second_difference < -1e-9 * value_scale) {
            ++diagnostics.convexity_violations;
        }
    }

    return Result<Solution>::success(
        Solution{.grid = grid.value(), .values = std::move(current), .diagnostics = diagnostics});
}

Result<PricingResult> FiniteDifferenceEngine::price(const MarketState& market,
                                                    const EuropeanOption& option,
                                                    const BlackScholesModel& model,
                                                    const PdeConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    const auto solution = solve(market, option, model, config);
    if (!solution.ok()) {
        return Result<PricingResult>::failure(solution.error());
    }

    const auto value =
        interpolate_linear(solution.value().grid, solution.value().values, market.spot());
    if (!value.ok()) {
        return Result<PricingResult>::failure(value.error());
    }

    PricingResult result;
    result.method = fmt::format("finite_difference_{}", to_string(config.scheme));
    result.value = value.value();

    const PdeDiagnostics& d = solution.value().diagnostics;
    result.add_diagnostic("asset_nodes", d.asset_nodes);
    result.add_diagnostic("time_steps", d.time_steps);
    result.add_diagnostic("s_max", d.s_max);
    result.add_diagnostic("asset_spacing", d.asset_spacing);
    result.add_diagnostic("time_step", d.time_step);
    result.add_diagnostic("explicit_stability_ratio", d.explicit_stability_ratio);
    result.add_diagnostic("sign_structure_holds", d.sign_structure_holds);
    result.add_diagnostic("peclet_violating_nodes", d.peclet_violating_nodes);
    result.add_diagnostic("peclet_violating_max_s", d.peclet_violating_max_s);
    result.add_diagnostic("most_negative_value", d.most_negative_value);
    result.add_diagnostic("convexity_violations", d.convexity_violations);
    result.add_diagnostic("rannacher_steps", d.rannacher_steps);
    if (d.worst_pivot_ratio.has_value()) {
        result.add_diagnostic("worst_pivot_ratio", *d.worst_pivot_ratio);
    }
    if (d.worst_residual.has_value()) {
        result.add_diagnostic("worst_residual", *d.worst_residual);
    }

    if (d.explicit_stable.has_value() && !*d.explicit_stable) {
        result.add_warning(fmt::format(
            "the explicit scheme is running at dtau/dtau_max = {:.3f}, beyond its stability bound. "
            "The result may be finite and is not therefore trustworthy.",
            d.explicit_stability_ratio));
    }
    if (!d.sign_structure_holds) {
        result.add_warning(fmt::format(
            "the spatial operator loses its M-matrix sign structure at {} interior node(s), for S "
            "up to {} against a strike of {}: convection dominates diffusion there (the "
            "cell-Peclet condition, threshold index (r-q)/sigma^2). Positivity is not guaranteed "
            "in that region. It occupies S < dS*(r-q)/sigma^2 and so shrinks as the grid refines; "
            "the spacing here is {}.",
            d.peclet_violating_nodes,
            d.peclet_violating_max_s,
            option.strike(),
            d.asset_spacing));
    }
    if (d.most_negative_value < 0.0) {
        result.add_warning(fmt::format(
            "the solution reached {} somewhere on the grid, but a vanilla payoff is non-negative "
            "and so is its price; this indicates oscillation rather than rounding",
            d.most_negative_value));
    }
    if (d.convexity_violations > 0) {
        result.add_warning(
            fmt::format("the final solution violates convexity at {} nodes; a vanilla European "
                        "value function is convex in S, so this indicates spatial oscillation",
                        d.convexity_violations));
    }

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
