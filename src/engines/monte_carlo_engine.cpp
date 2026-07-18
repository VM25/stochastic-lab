#include <diffusionworks/concurrency/parallel_reduce.hpp>
#include <diffusionworks/engines/geometric_asian_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "MonteCarloEngine";

class ScopedTimer {
public:
    [[nodiscard]] double elapsed_seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

[[nodiscard]] Status validate(const MonteCarloConfig& config) {
    if (config.paths < 2) {
        return Status::failure(
            ErrorCode::InvalidArgument,
            fmt::format("paths must be at least 2 but is {}; a single path has no dispersion and "
                        "cannot report a standard error",
                        config.paths),
            kContext);
    }
    if (config.steps < 1) {
        return Status::failure(ErrorCode::InvalidArgument,
                               fmt::format("steps must be at least 1 but is {}", config.steps),
                               kContext);
    }
    // Negated conjunction rather than the DeMorgan form, so that NaN is rejected:
    // every IEEE comparison against NaN is false, which would make
    // `level <= 0.0 || level >= 1.0` false for NaN and let it through.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(config.confidence_level > 0.0 && config.confidence_level < 1.0)) {
        return Status::failure(
            ErrorCode::InvalidArgument,
            fmt::format("confidence_level must lie strictly inside (0, 1) but is {}",
                        config.confidence_level),
            kContext);
    }
    if (config.variance_reduction.control_variate && config.control_variate_pilot_paths < 2) {
        return Status::failure(
            ErrorCode::InvalidArgument,
            fmt::format("control_variate_pilot_paths must be at least 2 but is {}; beta is a "
                        "covariance and needs dispersion to estimate",
                        config.control_variate_pilot_paths),
            kContext);
    }
    if (config.threads < 1 || config.threads > 1024) {
        return Status::failure(
            ErrorCode::InvalidArgument,
            fmt::format("threads must be between 1 and 1024 but is {}", config.threads),
            kContext);
    }
    return Status::success();
}

/// One path's contribution: the payoff, and the control's payoff when one exists.
struct PathSample {
    double payoff{};
    double control{};
};

/// Records the discretisation excursions seen across a run.
struct RunDiagnostics {
    std::int64_t non_positive_states{0};
};

/// A worker's thread-local state: its own accumulator, diagnostics, and path buffers,
/// so nothing is shared and mutated across workers (ADR-011, ADR-013).
struct Local {
    OnlineMoments observations;
    RunDiagnostics diagnostics;
    std::vector<double> path;
    std::vector<double> reflected;
};

/// Merges one worker's local state into another, in block order.
void reduce_local(Local& into, const Local& from) {
    into.observations.merge(from.observations);
    into.diagnostics.non_positive_states += from.diagnostics.non_positive_states;
}

/// Assembles the parts of the result every Monte Carlo run shares.
[[nodiscard]] Result<PricingResult> summarize(const OnlineMoments& observations,
                                              const RunDiagnostics& diagnostics,
                                              const MonteCarloConfig& config,
                                              double runtime) {
    auto error = observations.standard_error();
    if (!error) {
        return Result<PricingResult>::failure(std::move(error).error());
    }
    auto interval = observations.confidence_interval(config.confidence_level);
    if (!interval) {
        return Result<PricingResult>::failure(std::move(interval).error());
    }

    PricingResult result;
    result.method = fmt::format("monte_carlo_{}", to_string(config.scheme));
    result.value = observations.mean();
    result.standard_error = error.value();
    result.confidence_interval = interval.value();
    result.runtime_seconds = runtime;

    result.add_diagnostic("observations", static_cast<std::int64_t>(observations.count()));
    result.add_diagnostic("paths", config.paths);
    result.add_diagnostic("steps", config.steps);
    result.add_diagnostic("seed", static_cast<std::int64_t>(config.seed));
    result.add_diagnostic("scheme", std::string(to_string(config.scheme)));
    result.add_diagnostic("antithetic", config.variance_reduction.antithetic);
    result.add_diagnostic("control_variate", config.variance_reduction.control_variate);
    result.add_diagnostic("non_positive_states", diagnostics.non_positive_states);

    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the estimator is not finite after {} observations", observations.count()),
            kContext);
    }

    // A control-variate estimator can legitimately go slightly negative when the
    // true price is near zero, since it subtracts a correction. A crude or
    // antithetic estimator cannot: every payoff is non-negative. The check is
    // therefore applied only where it is a genuine invariant.
    if (!config.variance_reduction.control_variate && result.value < 0.0) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the estimator is negative ({}), which non-negative payoffs cannot "
                        "produce",
                        result.value),
            kContext);
    }

    if (diagnostics.non_positive_states > 0) {
        result.add_warning(fmt::format(
            "{} state(s) reached zero or went negative under the {} scheme. Euler-Maruyama and "
            "Milstein step the price itself and can cross zero; the payoff then clamps, biasing "
            "this estimate. Increase steps or use the exact scheme.",
            diagnostics.non_positive_states,
            to_string(config.scheme)));
    }

    return Result<PricingResult>::success(std::move(result));
}

}  // namespace

// ---------------------------------------------------------------------------
// European
// ---------------------------------------------------------------------------

Result<PricingResult> MonteCarloEngine::price(const MarketState& market,
                                              const EuropeanOption& option,
                                              const BlackScholesModel& model,
                                              const MonteCarloConfig& config) {
    const ScopedTimer timer;

    const Status valid = validate(config);
    if (!valid) {
        return Result<PricingResult>::failure(valid.error());
    }
    if (config.variance_reduction.control_variate) {
        // The geometric Asian control has no bearing on a European payoff, and no
        // other control is implemented. Refused rather than silently ignored: a
        // configuration asking for a technique it does not get would report a
        // method it did not use.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "control variates are implemented for arithmetic Asian options only, where the "
            "geometric Asian provides a control with a known expectation. No control is defined "
            "for a European option in this build.",
            kContext);
    }

    auto grid = TimeGrid::uniform(option.maturity(), config.steps);
    if (!grid) {
        return Result<PricingResult>::failure(std::move(grid).error());
    }

    const GbmPathGenerator generator(market, model, grid.value(), config.scheme);
    const double discount = market.discount_factor(option.maturity());
    const bool antithetic = config.variance_reduction.antithetic;

    // Each worker gets its own path buffers, allocated once (ADR-013), never shared.
    const auto make_local = [&] {
        Local local;
        local.path.resize(generator.path_size());
        local.reflected.resize(antithetic ? generator.path_size() : 0);
        return local;
    };

    // One path's contribution, accumulated into the worker's own state. The generator
    // is a pure function of (seed, index), so no two workers touch shared RNG state.
    const auto body = [&](std::int64_t index, Local& local) -> Status {
        const auto path_index = static_cast<std::uint64_t>(index);

        auto primary =
            generator.generate(config.seed, path_index, local.path, PathVariate::Primary);
        if (!primary) {
            return Status::failure(std::move(primary).error());
        }
        local.diagnostics.non_positive_states += primary.value().non_positive_states;
        const double payoff = discount * option.payoff(local.path.back());

        if (!antithetic) {
            local.observations.add(payoff);
            return Status::success();
        }

        auto mirrored =
            generator.generate(config.seed, path_index, local.reflected, PathVariate::Antithetic);
        if (!mirrored) {
            return Status::failure(std::move(mirrored).error());
        }
        local.diagnostics.non_positive_states += mirrored.value().non_positive_states;

        // One observation, not two: the pair is negatively correlated by construction,
        // so adding the members separately would misstate the standard error.
        const double reflected_payoff = discount * option.payoff(local.reflected.back());
        local.observations.add(0.5 * (payoff + reflected_payoff));
        return Status::success();
    };

    auto reduced =
        parallel_reduce<Local>(config.paths, config.threads, make_local, body, reduce_local);
    if (!reduced) {
        return Result<PricingResult>::failure(std::move(reduced).error());
    }

    return summarize(
        reduced.value().observations, reduced.value().diagnostics, config, timer.elapsed_seconds());
}

// ---------------------------------------------------------------------------
// Asian
// ---------------------------------------------------------------------------

namespace {

/// Reduces one path to its average, in whichever sense the option specifies.
///
/// Returns an empty optional when a geometric average cannot be formed because a
/// state is non-positive; the caller reports that rather than dropping the path.
[[nodiscard]] std::optional<double> average_of(const std::vector<double>& path,
                                               const AsianOption& option,
                                               std::int64_t stride,
                                               AveragingType averaging) {
    double accumulator = 0.0;

    // Monitoring runs i = 1..M, excluding the initial spot: it is known at
    // inception and averaging it in would price a different contract.
    for (std::int64_t i = 1; i <= option.monitoring_count(); ++i) {
        const double state = path[static_cast<std::size_t>(i * stride)];

        if (averaging == AveragingType::Arithmetic) {
            accumulator += state;
        } else {
            if (state <= 0.0) {
                return std::nullopt;
            }
            accumulator += std::log(state);
        }
    }

    const auto count = static_cast<double>(option.monitoring_count());
    return (averaging == AveragingType::Arithmetic) ? accumulator / count
                                                    : std::exp(accumulator / count);
}

/// Draws one path and reduces it to a payoff, and to the control's payoff when a
/// control is in use.
///
/// The control is evaluated on the *same path* as the payoff. That is the whole
/// mechanism: the correlation between them is what the correction exploits, and
/// an independently drawn control would be useless.
[[nodiscard]] Result<PathSample> sample_asian(const GbmPathGenerator& generator,
                                              const AsianOption& option,
                                              double discount,
                                              std::int64_t stride,
                                              std::uint64_t seed,
                                              std::uint64_t path_index,
                                              PathVariate variate,
                                              bool with_control,
                                              std::vector<double>& path,
                                              RunDiagnostics& diagnostics) {
    auto generated = generator.generate(seed, path_index, path, variate);
    if (!generated) {
        return Result<PathSample>::failure(std::move(generated).error());
    }
    diagnostics.non_positive_states += generated.value().non_positive_states;

    const std::optional<double> average = average_of(path, option, stride, option.averaging());
    if (!average.has_value()) {
        return Result<PathSample>::failure(
            ErrorCode::PathFailure,
            fmt::format("path {} reached a non-positive state, so its geometric average has no "
                        "logarithm",
                        path_index),
            kContext);
    }

    PathSample sample;
    sample.payoff = discount * option.payoff(*average);

    if (with_control) {
        const std::optional<double> geometric =
            average_of(path, option, stride, AveragingType::Geometric);
        if (!geometric.has_value()) {
            return Result<PathSample>::failure(
                ErrorCode::PathFailure,
                fmt::format("path {} reached a non-positive state, so the geometric control has "
                            "no logarithm. Use the exact scheme or more steps.",
                            path_index),
                kContext);
        }
        sample.control = discount * option.payoff(*geometric);
    }

    return Result<PathSample>::success(sample);
}

}  // namespace

Result<PricingResult> MonteCarloEngine::price(const MarketState& market,
                                              const AsianOption& option,
                                              const BlackScholesModel& model,
                                              const MonteCarloConfig& config) {
    const ScopedTimer timer;

    const Status valid = validate(config);
    if (!valid) {
        return Result<PricingResult>::failure(valid.error());
    }

    if (config.steps % option.monitoring_count() != 0) {
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("steps ({}) must be a positive multiple of the option's monitoring count "
                        "({}) so that every monitoring date falls on a grid point",
                        config.steps,
                        option.monitoring_count()),
            kContext);
    }

    const bool use_control = config.variance_reduction.control_variate;
    if (use_control && option.averaging() != AveragingType::Arithmetic) {
        // Controlling the geometric Asian with itself would give a correlation of
        // one and an estimator that simply returns the closed form -- true, but
        // not a Monte Carlo result, and misleading to report as one.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "the control variate is the geometric Asian, which controls the arithmetic average. "
            "Applying it to a geometric option would control it with itself and return the "
            "closed form dressed as a simulation.",
            kContext);
    }

    auto grid = TimeGrid::uniform(option.maturity(), config.steps);
    if (!grid) {
        return Result<PricingResult>::failure(std::move(grid).error());
    }

    const GbmPathGenerator generator(market, model, grid.value(), config.scheme);
    const double discount = market.discount_factor(option.maturity());
    const std::int64_t stride = config.steps / option.monitoring_count();

    std::vector<double> path(generator.path_size());
    std::vector<double> reflected(config.variance_reduction.antithetic ? generator.path_size() : 0);

    RunDiagnostics diagnostics;

    // --- Control expectation and pilot beta ---------------------------------

    double control_mean = 0.0;
    double beta = 0.0;
    double pilot_correlation = 0.0;

    if (use_control) {
        // The control's expectation is not estimated. It is known in closed form,
        // which is the entire reason this control was chosen.
        const auto geometric_twin = AsianOption::create(option.type(),
                                                        AveragingType::Geometric,
                                                        option.strike(),
                                                        option.maturity(),
                                                        option.monitoring_count());
        if (!geometric_twin) {
            return Result<PricingResult>::failure(std::move(geometric_twin).error());
        }
        auto control_price =
            GeometricAsianAnalyticEngine::price(market, geometric_twin.value(), model);
        if (!control_price) {
            return Result<PricingResult>::failure(std::move(control_price).error());
        }
        control_mean = control_price.value().value;

        // beta is estimated on a pilot drawn from path indices disjoint from the
        // main sample, so it is a constant with respect to that sample and the
        // estimator is exactly unbiased. Estimating it in-sample would couple
        // beta to the mean it corrects and bias the result by O(1/N).
        OnlineCovariance pilot;
        for (std::int64_t index = 0; index < config.control_variate_pilot_paths; ++index) {
            const auto pilot_index = static_cast<std::uint64_t>(config.paths + index);
            auto sample = sample_asian(generator,
                                       option,
                                       discount,
                                       stride,
                                       config.seed,
                                       pilot_index,
                                       PathVariate::Primary,
                                       true,
                                       path,
                                       diagnostics);
            if (!sample) {
                return Result<PricingResult>::failure(std::move(sample).error());
            }
            pilot.add(sample.value().payoff, sample.value().control);
        }

        auto covariance = pilot.covariance();
        if (!covariance) {
            return Result<PricingResult>::failure(std::move(covariance).error());
        }
        auto control_variance = pilot.moments_y().sample_variance();
        if (!control_variance) {
            return Result<PricingResult>::failure(std::move(control_variance).error());
        }

        if (control_variance.value() <= 0.0) {
            // A constant control carries no information and cannot be divided by.
            return Result<PricingResult>::failure(
                ErrorCode::InvalidArgument,
                "the control has zero variance on the pilot sample, so beta is undefined and the "
                "control carries no information",
                kContext);
        }

        beta = covariance.value() / control_variance.value();

        auto correlation = pilot.correlation();
        if (correlation) {
            pilot_correlation = correlation.value();
        }
    }

    // --- Main sample --------------------------------------------------------

    OnlineMoments observations;

    for (std::int64_t index = 0; index < config.paths; ++index) {
        const auto path_index = static_cast<std::uint64_t>(index);

        auto primary = sample_asian(generator,
                                    option,
                                    discount,
                                    stride,
                                    config.seed,
                                    path_index,
                                    PathVariate::Primary,
                                    use_control,
                                    path,
                                    diagnostics);
        if (!primary) {
            return Result<PricingResult>::failure(std::move(primary).error());
        }

        const auto corrected = [&](const PathSample& sample) {
            return use_control ? sample.payoff - beta * (sample.control - control_mean)
                               : sample.payoff;
        };

        if (!config.variance_reduction.antithetic) {
            observations.add(corrected(primary.value()));
            continue;
        }

        auto mirrored = sample_asian(generator,
                                     option,
                                     discount,
                                     stride,
                                     config.seed,
                                     path_index,
                                     PathVariate::Antithetic,
                                     use_control,
                                     reflected,
                                     diagnostics);
        if (!mirrored) {
            return Result<PricingResult>::failure(std::move(mirrored).error());
        }

        // The pair average is one observation. Applied after the control
        // correction so that both members are corrected on their own control.
        observations.add(0.5 * (corrected(primary.value()) + corrected(mirrored.value())));
    }

    auto result = summarize(observations, diagnostics, config, timer.elapsed_seconds());
    if (!result) {
        return result;
    }

    result.value().add_diagnostic("monitoring_count", option.monitoring_count());
    result.value().add_diagnostic("averaging", std::string(to_string(option.averaging())));

    if (use_control) {
        result.value().add_diagnostic("control_beta", beta);
        result.value().add_diagnostic("control_expectation", control_mean);
        result.value().add_diagnostic("control_pilot_paths", config.control_variate_pilot_paths);
        result.value().add_diagnostic("control_pilot_correlation", pilot_correlation);

        // A weak control does not help, and saying so is the point: the technique
        // still costs a second payoff per path, so a low correlation means paying
        // for nothing. Reported rather than left for the reader to infer from a
        // standard error that failed to shrink.
        if (std::abs(pilot_correlation) < 0.5) {
            result.value().add_warning(fmt::format(
                "the control correlates with the payoff at only {:.3f} on the pilot sample, so "
                "the variance reduction will be modest ({:.0f}% at best) while still costing a "
                "second payoff per path",
                pilot_correlation,
                100.0 * (1.0 - (1.0 - pilot_correlation * pilot_correlation))));
        }
    }

    return result;
}

}  // namespace diffusionworks
