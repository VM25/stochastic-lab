#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/experiments/convergence.hpp>
#include <diffusionworks/experiments/scheme_moments.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "convergence";

double apply_test_function(WeakTestFunction f, double terminal, double strike) noexcept {
    switch (f) {
        case WeakTestFunction::Identity:
            return terminal;
        case WeakTestFunction::Square:
            return terminal * terminal;
        case WeakTestFunction::CallPayoff:
            return std::max(terminal - strike, 0.0);
    }
    return terminal;
}

}  // namespace

const char* to_string(ErrorSource s) noexcept {
    switch (s) {
        case ErrorSource::Simulated:
            return "simulated";
        case ErrorSource::Analytic:
            return "analytic";
    }
    return "unknown";
}

const char* to_string(WeakTestFunction f) noexcept {
    switch (f) {
        case WeakTestFunction::Identity:
            return "identity";
        case WeakTestFunction::Square:
            return "square";
        case WeakTestFunction::CallPayoff:
            return "call_payoff";
    }
    return "unknown";
}

const char* to_string(ConvergenceVerdict v) noexcept {
    switch (v) {
        case ConvergenceVerdict::Consistent:
            return "consistent";
        case ConvergenceVerdict::ConsistentAsymptotically:
            return "consistent_asymptotically";
        case ConvergenceVerdict::Inconsistent:
            return "inconsistent";
        case ConvergenceVerdict::NoiseDominated:
            return "noise_dominated";
    }
    return "unknown";
}

Result<ConvergenceLevel> measure_strong_error(const MarketState& market,
                                              const BlackScholesModel& model,
                                              DiscretizationScheme scheme,
                                              double maturity,
                                              std::int64_t steps,
                                              std::uint64_t paths,
                                              std::uint64_t seed) {
    if (scheme == DiscretizationScheme::Exact) {
        // Zero by construction, which would plot as convergence of infinite order.
        // Refused rather than returned: the number is a tautology, not a
        // measurement.
        return Result<ConvergenceLevel>::failure(
            ErrorCode::InvalidArgument,
            "the strong error of the exact scheme against itself is identically zero, which is a "
            "tautology rather than a measurement. Pass euler_maruyama or milstein.",
            kContext);
    }
    if (paths < 2) {
        return Result<ConvergenceLevel>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a strong error needs at least 2 paths to have a standard error, got {}",
                        paths),
            kContext);
    }

    const auto grid = TimeGrid::uniform(maturity, steps);
    if (!grid.ok()) {
        return Result<ConvergenceLevel>::failure(grid.error());
    }

    const GbmPathGenerator exact_generator(
        market, model, grid.value(), DiscretizationScheme::Exact);
    const GbmPathGenerator scheme_generator(market, model, grid.value(), scheme);

    std::vector<double> exact_path(exact_generator.path_size());
    std::vector<double> scheme_path(scheme_generator.path_size());

    OnlineMoments absolute_error;
    std::int64_t non_positive = 0;

    for (std::uint64_t i = 0; i < paths; ++i) {
        // The same (seed, path index) under both generators. The stream is
        // addressed by coordinates rather than advanced by use, so both consume an
        // identical Z sequence and the difference below is pathwise.
        const auto exact_diagnostics = exact_generator.generate(seed, i, exact_path);
        if (!exact_diagnostics.ok()) {
            return Result<ConvergenceLevel>::failure(exact_diagnostics.error());
        }
        const auto scheme_diagnostics = scheme_generator.generate(seed, i, scheme_path);
        if (!scheme_diagnostics.ok()) {
            return Result<ConvergenceLevel>::failure(scheme_diagnostics.error());
        }
        non_positive += scheme_diagnostics.value().non_positive_states;

        absolute_error.add(std::abs(scheme_path.back() - exact_path.back()));
    }

    const auto standard_error = absolute_error.standard_error();
    if (!standard_error.ok()) {
        return Result<ConvergenceLevel>::failure(standard_error.error());
    }

    ConvergenceLevel level;
    level.steps = steps;
    level.step_size = grid.value().step_size();
    level.source = ErrorSource::Simulated;
    level.error = absolute_error.mean();
    level.error_standard_error = standard_error.value();
    level.paths = paths;
    level.non_positive_states = non_positive;
    return Result<ConvergenceLevel>::success(level);
}

Result<double> weak_reference(const MarketState& market,
                              const BlackScholesModel& model,
                              double maturity,
                              double strike,
                              WeakTestFunction test_function) {
    if (test_function == WeakTestFunction::CallPayoff) {
        // E[(S_T-K)^+] = e^{rT} * BS, because the analytic engine returns the
        // discounted expectation and this test function is undiscounted.
        const auto option = EuropeanOption::create(OptionType::Call, strike, maturity);
        if (!option.ok()) {
            return Result<double>::failure(option.error());
        }
        const auto priced = BlackScholesAnalyticEngine::price(market, option.value(), model);
        if (!priced.ok()) {
            return Result<double>::failure(priced.error());
        }
        return Result<double>::success(priced.value().value / market.discount_factor(maturity));
    }

    const auto moments = analytic_terminal_moments(market, model, maturity);
    if (!moments.ok()) {
        return Result<double>::failure(moments.error());
    }
    return Result<double>::success(test_function == WeakTestFunction::Identity
                                       ? moments.value().first
                                       : moments.value().second);
}

Result<ConvergenceLevel> weak_error_analytic(const MarketState& market,
                                             const BlackScholesModel& model,
                                             DiscretizationScheme scheme,
                                             double maturity,
                                             WeakTestFunction test_function,
                                             std::int64_t steps) {
    if (test_function == WeakTestFunction::CallPayoff) {
        // The scheme's expected payoff has no elementary form: it is an integral of
        // a kinked function against the scheme's own terminal law, which is not
        // lognormal. Refused rather than approximated, so a caller cannot mistake a
        // simulated number for an exact one.
        return Result<ConvergenceLevel>::failure(
            ErrorCode::UnsupportedCombination,
            "the call payoff has no closed-form expectation under a discretised scheme, whose "
            "terminal law is not lognormal. Use measure_weak_error, which pairs against the exact "
            "scheme on common Brownian paths.",
            "weak_error_analytic");
    }

    const auto grid = TimeGrid::uniform(maturity, steps);
    if (!grid.ok()) {
        return Result<ConvergenceLevel>::failure(grid.error());
    }
    const auto scheme_moments = scheme_terminal_moments(market, model, scheme, maturity, steps);
    if (!scheme_moments.ok()) {
        return Result<ConvergenceLevel>::failure(scheme_moments.error());
    }
    const auto exact_moments = analytic_terminal_moments(market, model, maturity);
    if (!exact_moments.ok()) {
        return Result<ConvergenceLevel>::failure(exact_moments.error());
    }

    const bool first = test_function == WeakTestFunction::Identity;
    const double scheme_value =
        first ? scheme_moments.value().first : scheme_moments.value().second;
    const double exact_value = first ? exact_moments.value().first : exact_moments.value().second;

    ConvergenceLevel level;
    level.steps = steps;
    level.step_size = grid.value().step_size();
    level.source = ErrorSource::Analytic;
    level.signed_error = scheme_value - exact_value;
    level.error = std::abs(*level.signed_error);
    level.estimate = scheme_value;
    level.reference = exact_value;
    // No error_standard_error: this level was computed, not sampled. Reporting a
    // zero would be a different claim -- "measured, and found to be exactly
    // certain" -- and would make resolution() divide by zero.
    level.paths = 0;
    return Result<ConvergenceLevel>::success(level);
}

Result<ConvergenceLevel> measure_weak_error(const MarketState& market,
                                            const BlackScholesModel& model,
                                            DiscretizationScheme scheme,
                                            double maturity,
                                            double strike,
                                            WeakTestFunction test_function,
                                            std::int64_t steps,
                                            std::uint64_t paths,
                                            std::uint64_t seed) {
    if (paths < 2) {
        return Result<ConvergenceLevel>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a weak error needs at least 2 paths to have a standard error, got {}",
                        paths),
            kContext);
    }

    const auto grid = TimeGrid::uniform(maturity, steps);
    if (!grid.ok()) {
        return Result<ConvergenceLevel>::failure(grid.error());
    }
    const auto reference = weak_reference(market, model, maturity, strike, test_function);
    if (!reference.ok()) {
        return Result<ConvergenceLevel>::failure(reference.error());
    }

    const GbmPathGenerator scheme_generator(market, model, grid.value(), scheme);
    const GbmPathGenerator exact_generator(
        market, model, grid.value(), DiscretizationScheme::Exact);

    std::vector<double> scheme_path(scheme_generator.path_size());
    std::vector<double> exact_path(exact_generator.path_size());

    OnlineMoments paired_difference;
    OnlineMoments scheme_sample;
    OnlineMoments exact_sample;
    std::int64_t non_positive = 0;

    for (std::uint64_t i = 0; i < paths; ++i) {
        const auto scheme_diagnostics = scheme_generator.generate(seed, i, scheme_path);
        if (!scheme_diagnostics.ok()) {
            return Result<ConvergenceLevel>::failure(scheme_diagnostics.error());
        }
        const auto exact_diagnostics = exact_generator.generate(seed, i, exact_path);
        if (!exact_diagnostics.ok()) {
            return Result<ConvergenceLevel>::failure(exact_diagnostics.error());
        }
        non_positive += scheme_diagnostics.value().non_positive_states;

        const double scheme_payoff = apply_test_function(test_function, scheme_path.back(), strike);
        const double exact_payoff = apply_test_function(test_function, exact_path.back(), strike);

        scheme_sample.add(scheme_payoff);
        exact_sample.add(exact_payoff);
        // The estimand is E[f(S^dt) - f(S^exact)], which equals the weak error
        // because the exact scheme samples the true terminal law. The pairing
        // cancels the common noise; it does not change what is being estimated.
        paired_difference.add(scheme_payoff - exact_payoff);
    }

    const auto standard_error = paired_difference.standard_error();
    if (!standard_error.ok()) {
        return Result<ConvergenceLevel>::failure(standard_error.error());
    }

    ConvergenceLevel level;
    level.steps = steps;
    level.step_size = grid.value().step_size();
    level.source = ErrorSource::Simulated;
    level.signed_error = paired_difference.mean();
    level.error = std::abs(paired_difference.mean());
    level.error_standard_error = standard_error.value();
    level.estimate = scheme_sample.mean();
    level.reference = reference.value();
    level.paths = paths;
    level.non_positive_states = non_positive;
    return Result<ConvergenceLevel>::success(level);
}

Result<ConvergenceStudy> fit_convergence(std::string scheme,
                                         std::string quantity,
                                         std::vector<ConvergenceLevel> levels,
                                         double theoretical_order,
                                         std::size_t asymptotic_level_count,
                                         double confidence_level) {
    if (levels.size() < 3) {
        return Result<ConvergenceStudy>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a convergence order needs at least 3 grid levels, got {}. Two levels can "
                        "be joined by a line of any slope, so the order would be an assumption "
                        "rather than a measurement.",
                        levels.size()),
            kContext);
    }
    if (asymptotic_level_count < 3 || asymptotic_level_count > levels.size()) {
        return Result<ConvergenceStudy>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the asymptotic window must span between 3 and {} levels, got {}",
                        levels.size(),
                        asymptotic_level_count),
            kContext);
    }

    // Sorted coarse-to-fine so that "the finest levels" is a well-defined suffix
    // regardless of the order the caller measured them in.
    std::ranges::sort(levels, [](const ConvergenceLevel& a, const ConvergenceLevel& b) {
        return a.step_size > b.step_size;
    });

    ConvergenceStudy study;
    study.scheme = std::move(scheme);
    study.quantity = std::move(quantity);
    study.theoretical_order = theoretical_order;
    study.asymptotic_level_count = asymptotic_level_count;

    for (std::size_t i = 1; i < levels.size(); ++i) {
        const ConvergenceLevel& coarse = levels[i - 1];
        const ConvergenceLevel& fine = levels[i];
        if (!(coarse.error > 0.0) || !(fine.error > 0.0)) {
            continue;
        }
        study.local_orders.push_back(
            LocalOrder{.coarse_steps = coarse.steps,
                       .fine_steps = fine.steps,
                       .order = std::log(coarse.error / fine.error) /
                                std::log(coarse.step_size / fine.step_size)});
    }

    for (const ConvergenceLevel& l : levels) {
        const auto resolution = l.resolution();
        if (resolution.has_value()) {
            study.worst_resolution = study.worst_resolution.has_value()
                                         ? std::min(*study.worst_resolution, *resolution)
                                         : *resolution;
        }
    }

    std::vector<double> step_sizes;
    std::vector<double> errors;
    step_sizes.reserve(levels.size());
    errors.reserve(levels.size());
    for (const ConvergenceLevel& l : levels) {
        step_sizes.push_back(l.step_size);
        errors.push_back(l.error);
    }

    const auto full = fit_power_law(step_sizes, errors);
    if (!full.ok()) {
        return Result<ConvergenceStudy>::failure(full.error());
    }
    const auto full_interval = full.value().slope_interval(confidence_level);
    if (!full_interval.ok()) {
        return Result<ConvergenceStudy>::failure(full_interval.error());
    }

    // The finest levels, as a suffix. subspan rather than pointer arithmetic: it
    // carries the bound with it, so an off-by-one here is a contract violation
    // rather than a silent read past the end of the sweep.
    const std::size_t offset = levels.size() - asymptotic_level_count;
    const auto asymptotic = fit_power_law(std::span<const double>(step_sizes).subspan(offset),
                                          std::span<const double>(errors).subspan(offset));
    if (!asymptotic.ok()) {
        return Result<ConvergenceStudy>::failure(asymptotic.error());
    }
    const auto asymptotic_interval = asymptotic.value().slope_interval(confidence_level);
    if (!asymptotic_interval.ok()) {
        return Result<ConvergenceStudy>::failure(asymptotic_interval.error());
    }

    study.full_fit = full.value();
    study.full_slope_interval = full_interval.value();
    study.asymptotic_fit = asymptotic.value();
    study.asymptotic_slope_interval = asymptotic_interval.value();

    // The verdict rests on the asymptotic window, because the theoretical order is
    // a statement about the limit dt -> 0 and the coarse levels are not in it.
    //
    // How consistency is judged depends on where the uncertainty lives. For
    // simulated levels it is sampling noise, which the regression interval is built
    // to quantify, so containment is the right test. For exact levels there is no
    // sampling noise at all: the residual is at rounding level, the interval
    // collapses to a point, and containment would reject every order including the
    // true one. There the only meaningful slack is the neglected higher-order term,
    // so a documented tolerance is used instead.
    const bool exact_study = std::ranges::all_of(
        levels, [](const ConvergenceLevel& l) { return l.source == ErrorSource::Analytic; });

    const auto consistent_with = [&](const LinearFit& fit, const ConfidenceInterval& interval) {
        return exact_study ? std::abs(fit.slope - theoretical_order) <= kAnalyticOrderTolerance
                           : interval.contains(theoretical_order);
    };

    // Noise domination is checked first: an unresolved level makes every slope
    // through it meaningless, so it must not be overridden by a lucky fit.
    if (study.worst_resolution.has_value() && *study.worst_resolution < kResolutionThreshold) {
        study.verdict = ConvergenceVerdict::NoiseDominated;
    } else if (!consistent_with(asymptotic.value(), asymptotic_interval.value())) {
        study.verdict = ConvergenceVerdict::Inconsistent;
    } else if (consistent_with(full.value(), full_interval.value())) {
        study.verdict = ConvergenceVerdict::Consistent;
    } else {
        study.verdict = ConvergenceVerdict::ConsistentAsymptotically;
    }

    study.levels = std::move(levels);
    return Result<ConvergenceStudy>::success(std::move(study));
}

}  // namespace diffusionworks
