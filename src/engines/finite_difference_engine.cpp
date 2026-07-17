#include <diffusionworks/engines/barrier_pde_engine.hpp>
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
#include <optional>
#include <ranges>
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

namespace {

/// A knock-out barrier for the PDE solve: the level, and which side is live.
struct BarrierSpec {
    double level;  ///< the barrier B, aligned to a grid node
    bool is_down;  ///< down-and-out (live above B) versus up-and-out (live below B)
};

/// The shared backward solve, parameterised by an optional knock-out barrier.
///
/// With no barrier this is the vanilla solve, byte for byte: `first`/`last` are
/// 0 and n-1, no terminal node is forced to zero, and the boundary values are the
/// vanilla ones, so every expression reduces to exactly what it was. That identity
/// is the point. The barrier support is added without disturbing the validated
/// vanilla path, and a regression test asserts the vanilla grid outputs are
/// unchanged to the bit.
Result<FiniteDifferenceEngine::Solution> solve_core(const MarketState& market,
                                                    const EuropeanOption& option,
                                                    const BlackScholesModel& model,
                                                    const PdeConfig& config,
                                                    const std::optional<BarrierSpec>& barrier) {
    using Solution = FiniteDifferenceEngine::Solution;
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

    // The grid, and the index range the PDE is solved on.
    //
    //   vanilla       : [0, S_max],   live [0, n-1]
    //   down-and-out  : [0, S_max],   live [barrier_index, n-1], B on an interior
    //                   node, S_max above it; the nodes below B are dead
    //   up-and-out    : [0, B],       live [0, n-1], B on the last node
    //
    // The down-and-out keeps the [0, S_max] span rather than [B, S_max] on purpose:
    // black_scholes_coefficients builds its rows from the node index as sigma^2 i^2
    // exactly, which cancels the 1/dS^2 that would otherwise dominate the rounding
    // at fine grids -- and that cancellation needs S_0 = 0. A grid starting at B
    // would forfeit it. So B sits on an interior node and the sub-barrier nodes are
    // carried as dead rows. The up-and-out's domain top *is* the barrier, so there
    // are no dead nodes and the barrier lands on the last node.
    Result<AssetGrid> grid =
        Result<AssetGrid>::failure(ErrorCode::InvalidArgument, "grid not constructed", kContext);
    if (!barrier.has_value()) {
        const double s_max = config.s_max.value_or(kDefaultSmaxMultiple * option.strike());
        grid = config.align_strike_to_node
                   ? AssetGrid::with_strike_on_node(s_max, config.asset_nodes, option.strike())
                   : AssetGrid::uniform(s_max, config.asset_nodes);
    } else if (barrier->is_down) {
        // The barrier must sit strictly below S_max, or it is not an interior node.
        const double s_max = config.s_max.value_or(kDefaultSmaxMultiple * option.strike());
        grid = AssetGrid::with_barrier_on_node(s_max, config.asset_nodes, barrier->level);
    } else {
        // Up-and-out: the domain is exactly [0, B], so the barrier is the top node.
        // A uniform grid pins it there exactly. That alignment is mandatory -- the
        // barrier is a Dirichlet boundary -- and takes precedence over strike
        // alignment, which cannot generally be had at the same time on one uniform
        // spacing. The strike therefore lands off-node in general, the same tradeoff
        // the down-and-out makes, and the payoff-kink error that leaves is a spatial
        // O(dS) term the convergence study measures rather than assumes away.
        //
        // Aligning the strike and putting the barrier at the *nearest multiple* to B
        // instead was the first attempt and was wrong: the top node then sits a
        // rounding error away from B, so the absorbing boundary prices a barrier that
        // is not the one asked for. A permanent test pins the up-and-out barrier on
        // its top node for exactly this reason.
        grid = AssetGrid::uniform(barrier->level, config.asset_nodes);
    }
    if (!grid.ok()) {
        return Result<Solution>::failure(grid.error());
    }

    // The live range and, for a down barrier, the node the barrier sits on.
    std::size_t first = 0;
    std::size_t last = static_cast<std::size_t>(grid.value().nodes()) - 1;
    if (barrier.has_value()) {
        const auto barrier_index = grid.value().nearest_index(barrier->level);
        if (!barrier_index.has_value()) {
            return Result<Solution>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("the barrier at {} is not on the grid it was aligned to; this is an "
                            "internal inconsistency in the barrier grid construction",
                            barrier->level),
                kContext);
        }
        if (barrier->is_down) {
            first = static_cast<std::size_t>(*barrier_index);
        } else {
            last = static_cast<std::size_t>(*barrier_index);
        }
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

    // A knock-out is worth zero at and beyond its barrier at expiry, so the barrier
    // node and every dead node carry zero rather than the vanilla payoff. This is
    // not merely cosmetic: the first live interior node reads its neighbour across
    // the barrier, so a non-zero payoff left there would leak into the solution on
    // the first step.
    if (barrier.has_value()) {
        if (barrier->is_down) {
            for (std::size_t i = 0; i <= first; ++i) {
                current[i] = 0.0;
            }
        } else {
            for (std::size_t i = last; i < current.size(); ++i) {
                current[i] = 0.0;
            }
        }
    }

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
        static_cast<std::int64_t>(operator_diagnostics.negative_sub_diagonal_nodes.size()) +
        static_cast<std::int64_t>(operator_diagnostics.negative_super_diagonal_nodes.size());

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

    // `first` and `last` -- the live range and its two boundary nodes -- were
    // computed above from the barrier (or 0 and n-1 for a vanilla). Everything below
    // steps only the interior between them and pins the two ends each step.
    std::vector<double> next(n, 0.0);
    double worst_pivot = std::numeric_limits<double>::infinity();
    double worst_residual = 0.0;
    bool solved_any = false;
    diagnostics.most_negative_value = std::min(0.0, *std::ranges::min_element(current));

    for (std::int64_t step = 1; step <= config.time_steps; ++step) {
        const double tau = time.value().time_at(step);

        // The two ends of the live range. For a vanilla these are the S=0 and S=S_max
        // boundaries. A down-and-out replaces the lower end with the Dirichlet barrier
        // (V(B, tau) = 0) and keeps the vanilla asymptotic at the top -- valid because
        // far above both barrier and strike the down-and-out is nearly the vanilla. An
        // up-and-out keeps the vanilla S=0 value (a call there is worthless) and
        // replaces the top with the Dirichlet barrier.
        double lower{};
        double upper{};
        if (!barrier.has_value()) {
            lower = lower_boundary(option, market, tau);
            upper = upper_boundary(option, market, grid.value(), tau);
        } else if (barrier->is_down) {
            lower = 0.0;
            upper = upper_boundary(option, market, grid.value(), tau);
        } else {
            lower = lower_boundary(option, market, tau);
            upper = 0.0;
        }

        // The first `rannacher.count` steps run fully implicit; the rest run under
        // the configured scheme. With count = 0 this is exactly the configured
        // scheme throughout.
        const PdeScheme step_scheme =
            (step <= config.rannacher.count) ? PdeScheme::Implicit : config.scheme;

        if (step_scheme == PdeScheme::Explicit) {
            std::ranges::fill(next, 0.0);
            next[first] = lower;
            next[last] = upper;
            for (std::size_t i = first + 1; i + 1 <= last; ++i) {
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
            // Dead rows first: identity with a zero right-hand side, so the solver
            // returns exactly zero there. They are rows of the system rather than a
            // separate case because a tridiagonal solve of an identity row is free
            // and perfectly conditioned, and because keeping one matrix means the
            // residual check below covers the whole grid rather than part of it.
            for (std::size_t i = 0; i < first; ++i) {
                system.diagonal[i] = 1.0;
            }
            for (std::size_t i = last + 1; i < n; ++i) {
                system.diagonal[i] = 1.0;
            }

            system.diagonal[first] = 1.0;
            system.rhs[first] = lower;
            system.diagonal[last] = 1.0;
            system.rhs[last] = upper;

            for (std::size_t i = first + 1; i + 1 <= last; ++i) {
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
            std::min(diagnostics.most_negative_value, *std::ranges::min_element(next));

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
    //
    // Restricted to the live interior, and skipped entirely for an up-and-out. A
    // down-and-out call is still convex, so the check applies over [first, last]. An
    // up-and-out call is not: it rises off S = 0 and falls back to zero at the
    // barrier, so it is genuinely concave near the top, and counting that as
    // oscillation would be a false positive. For barriers the primary oscillation
    // guard is non-negativity above -- a knock-out price cannot be negative, so
    // most_negative_value < 0 is unambiguous ringing whichever direction the barrier.
    const double value_scale = std::max(1.0, *std::ranges::max_element(current));
    const bool check_convexity = !barrier.has_value() || barrier->is_down;
    if (check_convexity) {
        for (std::size_t i = first + 1; i + 1 <= last; ++i) {
            const double second_difference = current[i + 1] - 2.0 * current[i] + current[i - 1];
            if (second_difference < -1e-9 * value_scale) {
                ++diagnostics.convexity_violations;
            }
        }
    }

    return Result<Solution>::success(
        Solution{.grid = grid.value(), .values = std::move(current), .diagnostics = diagnostics});
}

/// Copies the solve diagnostics onto a result and raises the same warnings for
/// both the vanilla and the barrier engine.
///
/// `strike` only labels the Peclet message. The non-negativity and convexity
/// warnings are worded for "the payoff" rather than "a vanilla payoff" because a
/// knock-out price is non-negative too, and the down-and-out is convex; the
/// up-and-out simply reports no convexity violations because its check is skipped.
void annotate_pde_result(PricingResult& result, const PdeDiagnostics& d, double strike) {
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
            strike,
            d.asset_spacing));
    }
    if (d.most_negative_value < 0.0) {
        result.add_warning(fmt::format(
            "the solution reached {} somewhere on the grid, but the payoff is non-negative and so "
            "is its price; this indicates oscillation rather than rounding",
            d.most_negative_value));
    }
    if (d.convexity_violations > 0) {
        result.add_warning(
            fmt::format("the final solution violates convexity at {} nodes; the value function is "
                        "convex in S here, so this indicates spatial oscillation",
                        d.convexity_violations));
    }
}

}  // namespace

Result<FiniteDifferenceEngine::Solution>
FiniteDifferenceEngine::solve(const MarketState& market,
                              const EuropeanOption& option,
                              const BlackScholesModel& model,
                              const PdeConfig& config) {
    return solve_core(market, option, model, config, std::nullopt);
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
    annotate_pde_result(result, solution.value().diagnostics, option.strike());

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

// ---------------------------------------------------------------------------
// BarrierPdeEngine
//
// Defined in this translation unit rather than its own so it can share the
// file-local solve_core and annotate_pde_result -- the barrier is the same PDE on a
// smaller domain, and duplicating the stepper to give it a separate .cpp would
// invite exactly the drift the sharing prevents.
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kBarrierContext = "BarrierPdeEngine";

/// The contracts the PDE barrier engine does not price, checked in one place so
/// price() and solve() refuse identically -- and, critically, so price() refuses
/// *before* its already-breached short-circuit. An up-and-in whose spot sits above
/// the barrier is "breached", but a breached knock-in becomes the vanilla, not zero;
/// refusing it first stops the engine reporting a plausible wrong price for a
/// contract it does not implement.
std::optional<Error> barrier_pde_refusal(const BarrierOption& option) {
    if (option.convention() != MonitoringConvention::Continuous) {
        return Error(
            ErrorCode::UnsupportedCombination,
            fmt::format("the PDE barrier engine prices continuous monitoring, but the "
                        "option specifies {} monitoring. Discrete and bridge monitoring "
                        "are the Monte Carlo engine's; returning this continuous price for "
                        "them would answer a different question.",
                        to_string(option.convention())),
            kBarrierContext);
    }
    if (option.type() != OptionType::Call) {
        return Error(ErrorCode::NotImplemented,
                     "only barrier calls are implemented; the put terminal condition and "
                     "boundaries are a separate set of cases",
                     kBarrierContext);
    }
    if (!is_knock_out(option.barrier_type())) {
        return Error(ErrorCode::NotImplemented,
                     fmt::format("the PDE engine solves knock-outs directly; {} is a knock-in, "
                                 "which is the vanilla less the knock-out by in-out parity rather "
                                 "than a second solve here",
                                 to_string(option.barrier_type())),
                     kBarrierContext);
    }
    return std::nullopt;
}

}  // namespace

Result<FiniteDifferenceEngine::Solution> BarrierPdeEngine::solve(const MarketState& market,
                                                                 const BarrierOption& option,
                                                                 const BlackScholesModel& model,
                                                                 const PdeConfig& config) {
    using Solution = FiniteDifferenceEngine::Solution;

    if (const auto refusal = barrier_pde_refusal(option); refusal.has_value()) {
        return Result<Solution>::failure(*refusal);
    }

    const double barrier = option.barrier();
    const bool is_down = is_down_barrier(option.barrier_type());
    const EuropeanOption vanilla = option.vanilla();

    // An up-and-out call struck at or above its barrier cannot pay: any path
    // finishing above the strike has finished above the barrier and knocked out. The
    // domain [0, B] carries the zero payoff everywhere, so the price is exactly zero
    // -- a price, returned on a real grid so a caller inspecting the solution sees a
    // consistent object, not a special-cased scalar.
    if (!is_down && vanilla.strike() >= barrier) {
        const auto grid = AssetGrid::uniform(barrier, config.asset_nodes);
        if (!grid.ok()) {
            return Result<Solution>::failure(grid.error());
        }
        std::vector<double> values(static_cast<std::size_t>(grid.value().nodes()), 0.0);
        PdeDiagnostics diagnostics;
        diagnostics.asset_nodes = grid.value().nodes();
        diagnostics.s_max = grid.value().s_max();
        diagnostics.asset_spacing = grid.value().spacing();
        return Result<Solution>::success(Solution{
            .grid = grid.value(), .values = std::move(values), .diagnostics = diagnostics});
    }

    return solve_core(
        market, vanilla, model, config, BarrierSpec{.level = barrier, .is_down = is_down});
}

Result<PricingResult> BarrierPdeEngine::price(const MarketState& market,
                                              const BarrierOption& option,
                                              const BlackScholesModel& model,
                                              const PdeConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    // Refuse before anything else, in particular before the breach check below: a
    // breached knock-in is the vanilla rather than zero, so the engine must decline
    // it rather than short-circuit it to a wrong price.
    if (const auto refusal = barrier_pde_refusal(option); refusal.has_value()) {
        return Result<PricingResult>::failure(*refusal);
    }

    // Already breached: a price, not an error. A knock-out whose spot sits past the
    // barrier is dead and worth zero, exactly. Handled before the solve because for
    // an up-and-out the breached spot lies above the grid's top, where interpolation
    // would rightly refuse.
    if (option.breaches(market.spot())) {
        PricingResult result;
        result.method = fmt::format("barrier_pde_{}", to_string(config.scheme));
        result.value = 0.0;
        result.add_diagnostic("already_breached", true);
        result.add_warning(fmt::format(
            "the spot {} already breaches the barrier {}, so this knock-out is worth exactly zero; "
            "the price is exact rather than a grid approximation",
            market.spot(),
            option.barrier()));
        result.runtime_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return Result<PricingResult>::success(std::move(result));
    }

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
    result.method = fmt::format("barrier_pde_{}", to_string(config.scheme));
    result.value = value.value();
    result.add_diagnostic("barrier", option.barrier());
    result.add_diagnostic("barrier_type", std::string(to_string(option.barrier_type())));
    result.add_diagnostic("monitoring", std::string(to_string(option.convention())));
    annotate_pde_result(result, solution.value().diagnostics, option.strike());

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
