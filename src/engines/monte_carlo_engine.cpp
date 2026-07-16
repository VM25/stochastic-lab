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
        // Two is the floor for a variance, and a price without uncertainty is not
        // a Monte Carlo result.
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
    return Status::success();
}

/// Assembles the parts of the result every Monte Carlo run shares.
[[nodiscard]] Result<PricingResult> summarize(const OnlineMoments& payoffs,
                                              std::int64_t non_positive_states,
                                              const MonteCarloConfig& config,
                                              double runtime) {
    auto error = payoffs.standard_error();
    if (!error) {
        return Result<PricingResult>::failure(std::move(error).error());
    }
    auto interval = payoffs.confidence_interval(config.confidence_level);
    if (!interval) {
        return Result<PricingResult>::failure(std::move(interval).error());
    }

    PricingResult result;
    result.method = fmt::format("monte_carlo_{}", to_string(config.scheme));
    result.value = payoffs.mean();
    result.standard_error = error.value();
    result.confidence_interval = interval.value();
    result.runtime_seconds = runtime;

    result.add_diagnostic("paths", static_cast<std::int64_t>(payoffs.count()));
    result.add_diagnostic("steps", config.steps);
    result.add_diagnostic("seed", static_cast<std::int64_t>(config.seed));
    result.add_diagnostic("scheme", std::string(to_string(config.scheme)));
    result.add_diagnostic("non_positive_states", non_positive_states);

    if (!std::isfinite(result.value)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the estimator is not finite after {} paths", payoffs.count()),
            kContext);
    }

    // A discounted payoff is non-negative on every path, so its mean cannot be
    // negative in exact arithmetic. A negative here means the accumulation has
    // broken, and reporting it is the point: clamping would turn a broken
    // estimator into a confident zero.
    if (result.value < 0.0) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the estimator is negative ({}), which non-negative payoffs cannot "
                        "produce",
                        result.value),
            kContext);
    }

    if (non_positive_states > 0) {
        // Surfaced rather than buried in the diagnostics. The explicit schemes
        // can drive the price through zero, and a call payoff clamps the result
        // to a perfectly ordinary number -- so the bias is invisible in the price
        // and must be announced.
        result.add_warning(fmt::format(
            "{} state(s) across {} paths reached zero or went negative under the {} scheme. "
            "Euler-Maruyama and Milstein step the price itself and can cross zero; the payoff "
            "then clamps, biasing this estimate. Increase steps or use the exact scheme.",
            non_positive_states,
            payoffs.count(),
            to_string(config.scheme)));
    }

    return Result<PricingResult>::success(std::move(result));
}

}  // namespace

Result<PricingResult> MonteCarloEngine::price(const MarketState& market,
                                              const EuropeanOption& option,
                                              const BlackScholesModel& model,
                                              const MonteCarloConfig& config) {
    const ScopedTimer timer;

    const Status valid = validate(config);
    if (!valid) {
        return Result<PricingResult>::failure(valid.error());
    }

    auto grid = TimeGrid::uniform(option.maturity(), config.steps);
    if (!grid) {
        return Result<PricingResult>::failure(std::move(grid).error());
    }

    const GbmPathGenerator generator(market, model, grid.value(), config.scheme);
    const double discount = market.discount_factor(option.maturity());

    // Allocated once for the whole run, reused by every path
    // (TECHNICAL-DESIGN section 11).
    std::vector<double> path(generator.path_size());

    OnlineMoments payoffs;
    std::int64_t non_positive_states = 0;

    for (std::int64_t index = 0; index < config.paths; ++index) {
        auto diagnostics = generator.generate(config.seed, static_cast<std::uint64_t>(index), path);
        if (!diagnostics) {
            return Result<PricingResult>::failure(std::move(diagnostics).error());
        }
        non_positive_states += diagnostics.value().non_positive_states;

        payoffs.add(discount * option.payoff(path.back()));
    }

    return summarize(payoffs, non_positive_states, config, timer.elapsed_seconds());
}

Result<PricingResult> MonteCarloEngine::price(const MarketState& market,
                                              const AsianOption& option,
                                              const BlackScholesModel& model,
                                              const MonteCarloConfig& config) {
    const ScopedTimer timer;

    const Status valid = validate(config);
    if (!valid) {
        return Result<PricingResult>::failure(valid.error());
    }

    // Every monitoring date must land exactly on a grid point. Monitoring at the
    // nearest point instead, or interpolating between two, would price a
    // different contract than the one described -- quietly, and by an amount
    // nobody would notice.
    if (config.steps % option.monitoring_count() != 0) {
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("steps ({}) must be a positive multiple of the option's monitoring count "
                        "({}) so that every monitoring date falls on a grid point",
                        config.steps,
                        option.monitoring_count()),
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

    OnlineMoments payoffs;
    std::int64_t non_positive_states = 0;

    for (std::int64_t index = 0; index < config.paths; ++index) {
        auto diagnostics = generator.generate(config.seed, static_cast<std::uint64_t>(index), path);
        if (!diagnostics) {
            return Result<PricingResult>::failure(std::move(diagnostics).error());
        }
        non_positive_states += diagnostics.value().non_positive_states;

        // Monitoring runs i = 1..M, excluding the initial spot: it is known at
        // inception and carries no uncertainty, so averaging it in would price a
        // different contract.
        double accumulator = 0.0;
        bool log_domain_failed = false;

        for (std::int64_t i = 1; i <= option.monitoring_count(); ++i) {
            const double state = path[static_cast<std::size_t>(i * stride)];

            switch (option.averaging()) {
                case AveragingType::Arithmetic:
                    accumulator += state;
                    break;
                case AveragingType::Geometric:
                    // The geometric average needs a logarithm, which a
                    // non-positive state does not have. The explicit schemes can
                    // produce one, so this is reachable rather than theoretical.
                    if (state <= 0.0) {
                        log_domain_failed = true;
                    } else {
                        accumulator += std::log(state);
                    }
                    break;
            }
        }

        if (log_domain_failed) {
            return Result<PricingResult>::failure(
                ErrorCode::PathFailure,
                fmt::format("path {} reached a non-positive state, so its geometric average has "
                            "no logarithm. The {} scheme steps the price itself and can cross "
                            "zero; use the exact scheme or more steps.",
                            index,
                            to_string(config.scheme)),
                kContext);
        }

        const auto count = static_cast<double>(option.monitoring_count());
        const double average = (option.averaging() == AveragingType::Arithmetic)
                                   ? accumulator / count
                                   : std::exp(accumulator / count);

        payoffs.add(discount * option.payoff(average));
    }

    auto result = summarize(payoffs, non_positive_states, config, timer.elapsed_seconds());
    if (result) {
        result.value().add_diagnostic("monitoring_count", option.monitoring_count());
        result.value().add_diagnostic("averaging", std::string(to_string(option.averaging())));
    }
    return result;
}

}  // namespace diffusionworks
