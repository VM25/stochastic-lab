#include <diffusionworks/concurrency/parallel_reduce.hpp>
#include <diffusionworks/engines/barrier_monte_carlo.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BarrierMonteCarloEngine";

}  // namespace

double bridge_crossing_probability(double log_start,
                                   double log_end,
                                   double log_barrier,
                                   double variance_rate,
                                   double dt,
                                   bool down_barrier) noexcept {
    // Distances from the barrier, signed so that "on the safe side" is positive
    // for both directions. The product below is then the same expression for an
    // upper and a lower barrier.
    const double from_start = down_barrier ? log_start - log_barrier : log_barrier - log_start;
    const double from_end = down_barrier ? log_end - log_barrier : log_barrier - log_end;

    // An endpoint already at or past the barrier is a certain crossing: the
    // observation itself caught it, and the bridge has nothing left to decide.
    if (from_start <= 0.0 || from_end <= 0.0) {
        return 1.0;
    }
    if (!(variance_rate > 0.0) || !(dt > 0.0)) {
        // No diffusion or no elapsed time leaves no room to wander between the
        // endpoints, so a path that ends on the safe side stayed there.
        return 0.0;
    }

    const double exponent = -2.0 * from_start * from_end / (variance_rate * dt);
    // exp of a large negative is 0, which is correct and needs no guard: an
    // interval whose endpoints are far from the barrier relative to sqrt(dt) is
    // one the path did not cross.
    return std::exp(exponent);
}

Result<PricingResult> BarrierMonteCarloEngine::price(const MarketState& market,
                                                     const BarrierOption& option,
                                                     const BlackScholesModel& model,
                                                     const BarrierMonteCarloConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    if (option.convention() == MonitoringConvention::Continuous) {
        // No simulation observes a barrier continuously. Monitoring very finely
        // and calling the result continuous would report a discretely monitored
        // price under the wrong name -- and the O(1/sqrt(m)) bias means "very
        // finely" is not as close as it sounds.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "a simulation cannot observe a barrier continuously. Monitoring on a fine grid and "
            "reporting it as continuous would misname a discretely monitored price, and the bias "
            "decays only as 1/sqrt(m). Use BarrierAnalyticEngine for the continuous contract, or "
            "the brownian_bridge convention, which is what a simulation can honestly say about "
            "it.",
            kContext);
    }
    if (config.paths < 2) {
        return Result<PricingResult>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a Monte Carlo price needs at least 2 paths to have a standard error, got "
                        "{}",
                        config.paths),
            kContext);
    }
    if (option.maturity() <= 0.0) {
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "a zero-maturity barrier has no path to monitor; the payoff is the price, and the "
            "analytic engine returns it exactly",
            kContext);
    }
    if (config.threads < 1 || config.threads > 1024) {
        return Result<PricingResult>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("threads must be between 1 and 1024 but is {}", config.threads),
            kContext);
    }

    const std::int64_t dates = *option.monitoring_count();
    const auto grid = TimeGrid::uniform(option.maturity(), dates);
    if (!grid.ok()) {
        return Result<PricingResult>::failure(grid.error());
    }

    // The exact scheme, at exactly the monitoring dates. Log-GBM is Brownian
    // motion with drift, so this samples the true joint law of the observed
    // prices however coarse the grid -- leaving monitoring bias and sampling noise
    // as the only errors, which is what makes the bias measurable.
    const GbmPathGenerator generator(market, model, grid.value(), DiscretizationScheme::Exact);

    const bool use_bridge = option.convention() == MonitoringConvention::BrownianBridge;
    const bool down = is_down_barrier(option.barrier_type());
    const double log_barrier = std::log(option.barrier());
    const double variance_rate = model.volatility() * model.volatility();
    const double dt = grid.value().step_size();
    const double discount = market.discount_factor(option.maturity());

    // A worker's thread-local state: its own payoff and bridge-probability
    // accumulators, its own diagnostic counts, and its own path buffer (allocated once
    // per worker, never per path -- ADR-013), so nothing is shared and mutated across
    // workers (ADR-011).
    struct Local {
        OnlineMoments payoffs;
        OnlineMoments bridge_probabilities;
        std::int64_t knocked_at_observation{0};
        std::int64_t knocked_by_bridge_only{0};
        std::int64_t paid{0};
        std::vector<double> path;
    };

    const auto make_local = [&] {
        Local local;
        local.path.resize(generator.path_size());
        return local;
    };

    // The counts are integer sums and the two OnlineMoments merge in block order, so
    // every path's contribution reduces exactly once and no diagnostic is lost.
    const auto reduce = [](Local& into, const Local& from) {
        into.payoffs.merge(from.payoffs);
        into.bridge_probabilities.merge(from.bridge_probabilities);
        into.knocked_at_observation += from.knocked_at_observation;
        into.knocked_by_bridge_only += from.knocked_by_bridge_only;
        into.paid += from.paid;
    };

    // One path. The asset shocks come from (seed, index) through the generator, and the
    // bridge uniforms from (seed, BarrierBridge, index); the early-knockout break lives
    // entirely inside this path's own monitoring loop. So which worker owns the path
    // cannot change the coordinates it draws, the order it draws them in, or when it
    // stops -- the parallel run tests every interval against exactly the bridge uniform
    // the sequential run did.
    const auto body = [&](std::int64_t i, Local& local) -> Status {
        const auto generated =
            generator.generate(config.seed, static_cast<std::uint64_t>(i), local.path);
        if (!generated.ok()) {
            return Status::failure(generated.error());
        }

        bool knocked = false;
        bool knocked_at_date = false;

        // The bridge's uniforms come from their own stream. They must not be drawn
        // from the asset shocks that produced these endpoints: the crossing
        // probability is a function of those shocks, so sharing a stream would test
        // each interval against a number derived from the interval itself.
        RandomStream bridge_stream(
            config.seed, StreamPurpose::BarrierBridge, static_cast<std::uint64_t>(i));

        for (std::size_t k = 1; k < local.path.size(); ++k) {
            // Node 0 is the initial spot. Whether the *starting* spot breaches is
            // the caller's question, not a monitoring event -- the analytic engine
            // handles an already-breached contract -- so monitoring starts at the
            // first date.
            if (option.breaches(local.path[k])) {
                knocked = true;
                knocked_at_date = true;
                break;
            }

            if (use_bridge) {
                const double p = bridge_crossing_probability(std::log(local.path[k - 1]),
                                                             std::log(local.path[k]),
                                                             log_barrier,
                                                             variance_rate,
                                                             dt,
                                                             down);
                local.bridge_probabilities.add(p);
                if (bridge_stream.next_uniform() < p) {
                    knocked = true;
                    break;
                }
            }
        }

        if (knocked) {
            if (knocked_at_date) {
                ++local.knocked_at_observation;
            } else {
                ++local.knocked_by_bridge_only;
            }
        }

        const double payoff = option.payoff(local.path.back(), knocked);
        if (payoff > 0.0) {
            ++local.paid;
        }
        local.payoffs.add(discount * payoff);
        return Status::success();
    };

    auto reduced = parallel_reduce<Local>(config.paths, config.threads, make_local, body, reduce);
    if (!reduced.ok()) {
        return Result<PricingResult>::failure(reduced.error());
    }
    const OnlineMoments& payoffs = reduced.value().payoffs;

    BarrierMonteCarloDiagnostics diagnostics;
    diagnostics.paths = config.paths;
    diagnostics.monitoring_dates = dates;
    diagnostics.knocked_at_observation = reduced.value().knocked_at_observation;
    diagnostics.knocked_by_bridge_only = reduced.value().knocked_by_bridge_only;
    diagnostics.paid = reduced.value().paid;

    const auto standard_error = payoffs.standard_error();
    if (!standard_error.ok()) {
        return Result<PricingResult>::failure(standard_error.error());
    }
    const auto interval = payoffs.confidence_interval(config.confidence_level);
    if (!interval.ok()) {
        return Result<PricingResult>::failure(interval.error());
    }

    diagnostics.mean_bridge_probability = reduced.value().bridge_probabilities.count() > 0
                                              ? reduced.value().bridge_probabilities.mean()
                                              : 0.0;

    PricingResult result;
    result.method = fmt::format("barrier_monte_carlo_{}", to_string(option.convention()));
    result.value = payoffs.mean();
    result.standard_error = standard_error.value();
    result.confidence_interval = interval.value();

    result.add_diagnostic("paths", diagnostics.paths);
    result.add_diagnostic("monitoring_dates", diagnostics.monitoring_dates);
    result.add_diagnostic("monitoring", std::string(to_string(option.convention())));
    result.add_diagnostic("knocked_at_observation", diagnostics.knocked_at_observation);
    result.add_diagnostic("knocked_by_bridge_only", diagnostics.knocked_by_bridge_only);
    result.add_diagnostic("mean_bridge_probability", diagnostics.mean_bridge_probability);
    result.add_diagnostic("paid", diagnostics.paid);
    result.add_diagnostic("knock_fraction",
                          static_cast<double>(diagnostics.knocked_at_observation +
                                              diagnostics.knocked_by_bridge_only) /
                              static_cast<double>(config.paths));

    if (option.convention() == MonitoringConvention::Discrete) {
        result.add_warning(fmt::format(
            "discrete monitoring at {} dates: the barrier is unobserved between fixes, so a "
            "knock-out is biased *high* against its continuously monitored value. The bias decays "
            "only as 1/sqrt(m); at m={} it is not negligible. The brownian_bridge convention "
            "corrects it.",
            dates,
            dates));
    }

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<PricingResult>::success(std::move(result));
}

}  // namespace diffusionworks
