#include <diffusionworks/concurrency/parallel_reduce.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/simulation/time_grid.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "HestonMonteCarloEngine";

}  // namespace

const char* to_string(HestonVarianceScheme scheme) noexcept {
    switch (scheme) {
        case HestonVarianceScheme::FullTruncation:
            return "full_truncation";
        case HestonVarianceScheme::NaiveEuler:
            return "naive_euler";
    }
    return "unknown";
}

Result<HestonSimulationOutcome>
HestonMonteCarloEngine::simulate(const MarketState& market,
                                 const EuropeanOption& option,
                                 const HestonModel& model,
                                 const HestonMonteCarloConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    if (config.paths < 2) {
        return Result<HestonSimulationOutcome>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a Monte Carlo price needs at least 2 paths for a standard error, got {}",
                        config.paths),
            kContext);
    }
    if (option.maturity() <= 0.0) {
        return Result<HestonSimulationOutcome>::failure(
            ErrorCode::UnsupportedCombination,
            "a zero-maturity option has no path to simulate; its payoff is the price",
            kContext);
    }
    if (config.threads < 1 || config.threads > 1024) {
        return Result<HestonSimulationOutcome>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("threads must be between 1 and 1024 but is {}", config.threads),
            kContext);
    }

    const auto grid = TimeGrid::uniform(option.maturity(), config.steps);
    if (!grid.ok()) {
        return Result<HestonSimulationOutcome>::failure(grid.error());
    }
    const auto rho = CorrelationCoefficient::create(model.correlation());
    if (!rho.ok()) {
        return Result<HestonSimulationOutcome>::failure(rho.error());
    }

    const double dt = grid.value().step_size();
    const double sqrt_dt = std::sqrt(dt);
    const double drift = market.rate() - market.dividend_yield();
    const double kappa = model.mean_reversion();
    const double theta = model.long_run_variance();
    const double xi = model.vol_of_variance();
    const double v0 = model.initial_variance();
    const double log_spot0 = std::log(market.spot());
    const double discount = market.discount_factor(option.maturity());
    const bool full_truncation = config.scheme == HestonVarianceScheme::FullTruncation;

    // A worker's thread-local state: its own payoff accumulator and its own copies
    // of the accumulating diagnostics, so nothing is shared and mutated across
    // workers (ADR-011). `minimum_variance` starts at v0 in every worker, exactly as
    // the sequential run seeds it, so the reducing `min` reproduces the sequential
    // depth regardless of the partition.
    struct Local {
        OnlineMoments payoffs;
        std::int64_t negative_variance_events{0};
        std::int64_t non_finite_paths{0};
        double minimum_variance{0.0};
    };

    const auto make_local = [&] {
        Local local;
        local.minimum_variance = v0;
        return local;
    };

    // The excursion counts sum and the depth mins -- both exact, order-independent
    // reductions -- so no worker's diagnostics are ever lost, and the totals are the
    // same at any thread count. The payoffs merge with the reassociation the mean
    // and variance carry.
    const auto reduce = [](Local& into, const Local& from) {
        into.payoffs.merge(from.payoffs);
        into.negative_variance_events += from.negative_variance_events;
        into.non_finite_paths += from.non_finite_paths;
        into.minimum_variance = std::min(into.minimum_variance, from.minimum_variance);
    };

    // One path. The two shock streams are keyed by (seed, purpose, index), so a path
    // draws the same coordinates in whichever worker owns it -- path ownership and the
    // RNG addressing are unchanged by partitioning. A non-finite path is a diagnostic,
    // not an error: it is counted and the body still succeeds, exactly as the
    // sequential loop continued past it, because the price is blocked afterwards on the
    // reduced count rather than on the first path to diverge.
    const auto body = [&](std::int64_t i, Local& local) -> Status {
        // The two shock streams are independent by construction; the correlation is
        // imposed by the Cholesky combination below, never by sharing a stream. The
        // variance's own shock drives the pair (passes through unchanged), so a run
        // switching rho on or off leaves the variance path's driving draws fixed.
        RandomStream variance_stream(
            config.seed, StreamPurpose::VarianceShock, static_cast<std::uint64_t>(i));
        RandomStream asset_stream(
            config.seed, StreamPurpose::AssetShock, static_cast<std::uint64_t>(i));

        double log_spot = log_spot0;
        double variance = v0;
        bool path_finite = true;

        for (std::int64_t step = 0; step < config.steps; ++step) {
            // The only difference between the schemes is which variance feeds the
            // square root and the drift correction. Full truncation floors it; the
            // naive scheme uses the raw value, so a negative variance makes `vol`
            // not-a-number and the path is lost on this very step.
            const double variance_used = full_truncation ? std::max(variance, 0.0) : variance;
            const double vol = std::sqrt(variance_used);

            const CorrelatedNormals shocks =
                correlate(variance_stream.next_normal(), asset_stream.next_normal(), rho.value());
            const double z_variance = shocks.first;
            const double z_spot = shocks.second;

            log_spot += (drift - 0.5 * variance_used) * dt + vol * sqrt_dt * z_spot;
            variance += kappa * (theta - variance_used) * dt + xi * vol * sqrt_dt * z_variance;

            // The pre-truncation variance going negative is the event the diagnostic
            // exists to count, and the depth it reaches is the minimum-variance
            // diagnostic. Both are recorded before any flooring, so they describe the
            // scheme's real excursion rather than its cleaned-up state.
            if (std::isfinite(variance)) {
                if (variance < 0.0) {
                    ++local.negative_variance_events;
                }
                local.minimum_variance = std::min(local.minimum_variance, variance);
            }
            if (!std::isfinite(log_spot) || !std::isfinite(variance)) {
                path_finite = false;
                break;
            }
        }

        if (!path_finite) {
            ++local.non_finite_paths;
            return Status::success();
        }

        const double terminal_spot = std::exp(log_spot);
        if (!std::isfinite(terminal_spot)) {
            ++local.non_finite_paths;
            return Status::success();
        }
        local.payoffs.add(discount * option.payoff(terminal_spot));
        return Status::success();
    };

    auto reduced = parallel_reduce<Local>(config.paths, config.threads, make_local, body, reduce);
    if (!reduced.ok()) {
        return Result<HestonSimulationOutcome>::failure(reduced.error());
    }
    const OnlineMoments& payoffs = reduced.value().payoffs;

    HestonSimulationDiagnostics diagnostics;
    diagnostics.paths = config.paths;
    diagnostics.steps = config.steps;
    diagnostics.negative_variance_events = reduced.value().negative_variance_events;
    diagnostics.minimum_variance = reduced.value().minimum_variance;
    diagnostics.non_finite_paths = reduced.value().non_finite_paths;

    HestonSimulationOutcome outcome;
    outcome.diagnostics = diagnostics;

    // A non-finite path leaves no admissible price. The outcome still carries the
    // diagnostics -- for the naive scheme they are the finding -- so the caller can
    // quantify the failure rather than only learn that it happened.
    if (diagnostics.non_finite_paths > 0) {
        return Result<HestonSimulationOutcome>::success(std::move(outcome));
    }

    const auto standard_error = payoffs.standard_error();
    if (!standard_error.ok()) {
        return Result<HestonSimulationOutcome>::failure(standard_error.error());
    }
    const auto interval = payoffs.confidence_interval(config.confidence_level);
    if (!interval.ok()) {
        return Result<HestonSimulationOutcome>::failure(interval.error());
    }

    PricingResult result;
    result.method = fmt::format("heston_monte_carlo_{}", to_string(config.scheme));
    result.value = payoffs.mean();
    result.standard_error = standard_error.value();
    result.confidence_interval = interval.value();

    result.add_diagnostic("paths", diagnostics.paths);
    result.add_diagnostic("steps", diagnostics.steps);
    result.add_diagnostic("negative_variance_events", diagnostics.negative_variance_events);
    const double negative_fraction =
        static_cast<double>(diagnostics.negative_variance_events) /
        (static_cast<double>(config.paths) * static_cast<double>(config.steps));
    result.add_diagnostic("negative_variance_fraction", negative_fraction);
    result.add_diagnostic("minimum_variance", diagnostics.minimum_variance);
    result.add_diagnostic("non_finite_paths", diagnostics.non_finite_paths);
    result.add_diagnostic("feller_ratio", model.feller_ratio());
    result.add_diagnostic("satisfies_feller", model.satisfies_feller());

    if (!model.satisfies_feller()) {
        result.add_warning(fmt::format(
            "the Feller condition is violated (ratio {:.4f}): the true variance can approach zero "
            "and the full-truncation scheme floored a negative pre-truncation variance on {:.2f}% "
            "of steps. The price carries the discretisation bias this induces; EXP-10 quantifies "
            "it against the semi-analytic reference.",
            model.feller_ratio(),
            100.0 * negative_fraction));
    }

    result.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    outcome.price = std::move(result);
    return Result<HestonSimulationOutcome>::success(std::move(outcome));
}

Result<PricingResult> HestonMonteCarloEngine::price(const MarketState& market,
                                                    const EuropeanOption& option,
                                                    const HestonModel& model,
                                                    const HestonMonteCarloConfig& config) {
    auto outcome = simulate(market, option, model, config);
    if (!outcome.ok()) {
        return Result<PricingResult>::failure(std::move(outcome).error());
    }
    if (!outcome.value().price.has_value()) {
        // A non-finite path is a failure, not a sample to drop. Averaging the
        // survivors would report a clean price for a simulation that blew up on some
        // paths, hiding exactly the instability the exit gate says must block
        // completion.
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("{} of {} Heston paths went non-finite; the price is blocked rather than "
                        "averaged over the survivors. This is a hard regime for full-truncation "
                        "Euler -- a larger vol-of-variance or a coarser step -- and the diverged "
                        "paths are evidence of failure, not samples to discard.",
                        outcome.value().diagnostics.non_finite_paths,
                        config.paths),
            kContext);
    }
    return Result<PricingResult>::success(std::move(outcome).value().price.value());
}

}  // namespace diffusionworks
