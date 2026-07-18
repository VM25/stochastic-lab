#include <diffusionworks/numerics/nelder_mead.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "NelderMead";

using Point = std::vector<double>;

/// A vertex of the simplex: a point and its objective value.
struct Vertex {
    Point point;
    double value{};
};

}  // namespace

const char* to_string(NelderMeadStatus status) noexcept {
    switch (status) {
        case NelderMeadStatus::Converged:
            return "converged";
        case NelderMeadStatus::MaxIterationsReached:
            return "max_iterations_reached";
    }
    return "unknown";
}

Result<NelderMeadResult>
nelder_mead(const std::function<double(std::span<const double>)>& objective,
            std::span<const double> initial,
            const NelderMeadConfig& config) {
    const std::size_t n = initial.size();
    if (n == 0) {
        return Result<NelderMeadResult>::failure(
            ErrorCode::InvalidArgument, "Nelder-Mead needs at least one dimension", kContext);
    }
    if (config.function_tolerance <= 0.0 || config.simplex_tolerance <= 0.0 ||
        config.initial_step <= 0.0 || config.max_iterations < 1) {
        return Result<NelderMeadResult>::failure(
            ErrorCode::InvalidArgument,
            "Nelder-Mead needs positive tolerances, a positive initial step, and at least one "
            "iteration",
            kContext);
    }

    int evaluations = 0;
    // A non-finite objective is treated as +infinity so the simplex retreats from it
    // rather than sorting a NaN. The evaluation is still counted -- it happened.
    const auto evaluate = [&](const Point& p) -> double {
        const double v = objective(p);
        ++evaluations;
        return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };

    // Copy the start into a local vector element by element. Assigning a vector from a
    // span's iterators lets GCC's optimiser reason that the span *could* be empty
    // (null data), so it flags a potential null dereference in the inlined copy; the
    // indexed loop, guarded by the n > 0 established above, does not.
    Point start(n);
    for (std::size_t d = 0; d < n; ++d) {
        start[d] = initial[d];
    }

    // The initial simplex: the start point and one neighbour per axis.
    std::vector<Vertex> simplex(n + 1);
    simplex[0].point = start;
    simplex[0].value = evaluate(simplex[0].point);
    for (std::size_t i = 0; i < n; ++i) {
        Point p = start;
        p[i] += config.initial_step;
        simplex[i + 1].point = std::move(p);
        simplex[i + 1].value = evaluate(simplex[i + 1].point);
    }

    const auto by_value = [](const Vertex& a, const Vertex& b) { return a.value < b.value; };

    const auto centroid_of_best = [&](const std::vector<Vertex>& s) -> Point {
        // The centroid of every vertex but the worst (the last after sorting).
        Point c(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < n; ++d) {
                c[d] += s[i].point[d];
            }
        }
        for (double& value : c) {
            value /= static_cast<double>(n);
        }
        return c;
    };

    // The simplex "size": the largest distance from the centroid to any vertex. Small
    // means the vertices have collapsed together, which -- with a small spread of
    // values -- is convergence.
    const auto simplex_size = [&](const std::vector<Vertex>& s, const Point& centroid) -> double {
        double worst = 0.0;
        for (const Vertex& v : s) {
            double sum = 0.0;
            for (std::size_t d = 0; d < n; ++d) {
                const double delta = v.point[d] - centroid[d];
                sum += delta * delta;
            }
            worst = std::max(worst, std::sqrt(sum));
        }
        return worst;
    };

    NelderMeadStatus status = NelderMeadStatus::MaxIterationsReached;
    int iteration = 0;
    for (; iteration < config.max_iterations; ++iteration) {
        std::ranges::sort(simplex, by_value);

        const double best = simplex.front().value;
        const double worst = simplex.back().value;
        // The value spread and the point spread must both be small. A convergence
        // judged on the objective alone would stop on a flat plateau far from a
        // minimum; one judged on the simplex alone would stop on a cliff.
        const double spread = std::isfinite(worst) && std::isfinite(best)
                                  ? worst - best
                                  : std::numeric_limits<double>::infinity();
        const Point full_centroid = [&] {
            Point c(n, 0.0);
            for (const Vertex& v : simplex) {
                for (std::size_t d = 0; d < n; ++d) {
                    c[d] += v.point[d];
                }
            }
            for (double& value : c) {
                value /= static_cast<double>(n + 1);
            }
            return c;
        }();
        if (spread <= config.function_tolerance &&
            simplex_size(simplex, full_centroid) <= config.simplex_tolerance) {
            status = NelderMeadStatus::Converged;
            break;
        }

        const Point centroid = centroid_of_best(simplex);
        Vertex& worst_vertex = simplex.back();
        const double second_worst = simplex[n - 1].value;

        // Reflection.
        Point reflected(n);
        for (std::size_t d = 0; d < n; ++d) {
            reflected[d] = centroid[d] + config.reflection * (centroid[d] - worst_vertex.point[d]);
        }
        const double reflected_value = evaluate(reflected);

        if (reflected_value < best) {
            // Expansion: the reflection improved on the best, so try stepping further.
            Point expanded(n);
            for (std::size_t d = 0; d < n; ++d) {
                expanded[d] = centroid[d] + config.expansion * (reflected[d] - centroid[d]);
            }
            const double expanded_value = evaluate(expanded);
            if (expanded_value < reflected_value) {
                worst_vertex.point = std::move(expanded);
                worst_vertex.value = expanded_value;
            } else {
                worst_vertex.point = std::move(reflected);
                worst_vertex.value = reflected_value;
            }
        } else if (reflected_value < second_worst) {
            // Reflection is a middling improvement: accept it.
            worst_vertex.point = std::move(reflected);
            worst_vertex.value = reflected_value;
        } else {
            // Contraction, on whichever side of the centroid is lower.
            const bool outside = reflected_value < worst_vertex.value;
            Point contracted(n);
            for (std::size_t d = 0; d < n; ++d) {
                const double target = outside ? reflected[d] : worst_vertex.point[d];
                contracted[d] = centroid[d] + config.contraction * (target - centroid[d]);
            }
            const double contracted_value = evaluate(contracted);
            const double compare = outside ? reflected_value : worst_vertex.value;
            if (contracted_value < compare) {
                worst_vertex.point = std::move(contracted);
                worst_vertex.value = contracted_value;
            } else {
                // Shrink every vertex toward the best.
                const Point best_point = simplex.front().point;
                for (std::size_t i = 1; i < simplex.size(); ++i) {
                    for (std::size_t d = 0; d < n; ++d) {
                        simplex[i].point[d] =
                            best_point[d] + config.shrink * (simplex[i].point[d] - best_point[d]);
                    }
                    simplex[i].value = evaluate(simplex[i].point);
                }
            }
        }
    }

    std::ranges::sort(simplex, by_value);

    NelderMeadResult result;
    result.point = simplex.front().point;
    result.value = simplex.front().value;
    result.iterations = iteration;
    result.function_evaluations = evaluations;
    result.status = status;
    return Result<NelderMeadResult>::success(std::move(result));
}

}  // namespace diffusionworks
