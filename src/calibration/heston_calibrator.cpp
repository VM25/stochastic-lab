#include <diffusionworks/calibration/heston_calibrator.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "HestonCalibrator";

/// A full evaluation of a parameter set against the surface.
struct Evaluation {
    std::vector<QuoteResidual> residuals;
    double objective{};
    double price_rmse{};
    double implied_vol_rmse{};

    /// Quotes that could not be priced or inverted at these parameters. Each carried a
    /// penalty into the objective rather than a residual.
    int quotes_failed{};
};

/// The penalty a single unpriceable or uninvertible quote adds to the objective. It is
/// large against a typical squared residual (a per-quote implied-vol error of a
/// percent squares to ~1e-4), so a region our quadrature cannot resolve is strongly
/// discouraged -- but it is *finite*, so the optimizer descends on the quotes that do
/// price rather than being walled onto an infinite plateau. The true parameters price
/// cleanly, so the penalty never moves the minimum; it only shapes the path there.
constexpr double kQuotePenalty = 1.0;

/// Normalised Euclidean distance between two parameter vectors, each coordinate
/// scaled by its bound range so the five parameters are comparable.
[[nodiscard]] double normalized_distance(const HestonParameters& a,
                                         const HestonParameters& b,
                                         const HestonParameterBounds& bounds) {
    const auto term = [](double x, double y, double lower, double upper) {
        const double range = upper - lower;
        const double d = range > 0.0 ? (x - y) / range : 0.0;
        return d * d;
    };
    return std::sqrt(
        term(a.initial_variance,
             b.initial_variance,
             bounds.lower.initial_variance,
             bounds.upper.initial_variance) +
        term(a.mean_reversion,
             b.mean_reversion,
             bounds.lower.mean_reversion,
             bounds.upper.mean_reversion) +
        term(a.long_run_variance,
             b.long_run_variance,
             bounds.lower.long_run_variance,
             bounds.upper.long_run_variance) +
        term(a.vol_of_variance,
             b.vol_of_variance,
             bounds.lower.vol_of_variance,
             bounds.upper.vol_of_variance) +
        term(a.correlation, b.correlation, bounds.lower.correlation, bounds.upper.correlation));
}

}  // namespace

const char* to_string(CalibrationObjectiveType type) noexcept {
    switch (type) {
        case CalibrationObjectiveType::ImpliedVolatility:
            return "implied_volatility";
        case CalibrationObjectiveType::Price:
            return "price";
    }
    return "unknown";
}

Result<CalibrationResult> calibrate_heston(const VolatilitySurface& surface,
                                           const CalibrationConfig& config) {
    if (surface.quotes.empty()) {
        return Result<CalibrationResult>::failure(
            ErrorCode::InvalidArgument, "the surface has no quotes to calibrate to", kContext);
    }
    if (config.initial_guesses.empty()) {
        return Result<CalibrationResult>::failure(
            ErrorCode::InvalidArgument,
            "calibration needs at least one initial guess; the intended use is several, none of "
            "them the true parameters",
            kContext);
    }
    if (!config.bounds.valid()) {
        return Result<CalibrationResult>::failure(
            ErrorCode::InvalidArgument, "the parameter bounds are inside-out", kContext);
    }

    const auto market = surface.market();
    if (!market) {
        return Result<CalibrationResult>::failure(market.error());
    }

    // Precompute the options once; the objective is evaluated thousands of times.
    std::vector<EuropeanOption> options;
    options.reserve(surface.quotes.size());
    for (const SurfaceQuote& quote : surface.quotes) {
        const auto option = EuropeanOption::create(quote.type, quote.strike, quote.maturity);
        if (!option) {
            return Result<CalibrationResult>::failure(option.error());
        }
        options.push_back(option.value());
    }

    const bool objective_is_iv = config.objective == CalibrationObjectiveType::ImpliedVolatility;
    const std::size_t n = surface.quotes.size();

    // Evaluates a parameter set against the surface. Never fails: a quote our
    // quadrature cannot price, or a price we cannot invert, contributes a finite
    // penalty instead of a residual, so the optimizer descends on the quotes that do
    // resolve rather than being trapped on an infinite plateau. The penalised-quote
    // count is reported so a fit that leaned on penalties is not read as clean.
    const auto evaluate = [&](const HestonParameters& parameters, bool need_iv) -> Evaluation {
        Evaluation e;
        const auto model = parameters.to_model();
        if (!model) {
            // Feasible parameters always build a model, so this is a defensive path; a
            // fully penalised objective keeps it finite and repels the optimizer.
            e.objective = kQuotePenalty * static_cast<double>(n) * 10.0;
            e.quotes_failed = static_cast<int>(n);
            return e;
        }
        e.residuals.reserve(n);
        double objective = 0.0;
        double price_squared = 0.0;
        double iv_squared = 0.0;
        std::size_t price_count = 0;
        std::size_t iv_count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto priced = HestonAnalyticEngine::price(
                market.value(), options[i], model.value(), config.pricing);
            if (!priced) {
                objective += kQuotePenalty;
                ++e.quotes_failed;
                continue;
            }
            const double model_price = priced.value().value;
            const double price_residual = model_price - surface.quotes[i].price;
            price_squared += price_residual * price_residual;
            ++price_count;

            QuoteResidual residual;
            residual.type = surface.quotes[i].type;
            residual.strike = surface.quotes[i].strike;
            residual.maturity = surface.quotes[i].maturity;
            residual.weight = surface.quotes[i].weight;
            residual.market_price = surface.quotes[i].price;
            residual.model_price = model_price;
            residual.market_implied_volatility = surface.quotes[i].implied_volatility;
            residual.model_implied_volatility = std::numeric_limits<double>::quiet_NaN();

            bool inverted = false;
            double iv_residual = 0.0;
            if (need_iv) {
                const auto iv = ImpliedVolatility::solve(
                    market.value(), options[i], model_price, config.implied_volatility);
                if (iv) {
                    inverted = true;
                    residual.model_implied_volatility = iv.value().implied_volatility;
                    iv_residual =
                        iv.value().implied_volatility - surface.quotes[i].implied_volatility;
                    iv_squared += iv_residual * iv_residual;
                    ++iv_count;
                }
            }
            e.residuals.push_back(residual);

            if (objective_is_iv) {
                if (inverted) {
                    objective += surface.quotes[i].weight * iv_residual * iv_residual;
                } else {
                    // The IV objective needs an implied volatility this quote could not
                    // supply: penalise it rather than fitting to a missing residual.
                    objective += kQuotePenalty;
                    ++e.quotes_failed;
                }
            } else {
                objective += surface.quotes[i].weight * price_residual * price_residual;
            }
        }
        e.objective = objective;
        e.price_rmse =
            price_count > 0 ? std::sqrt(price_squared / static_cast<double>(price_count)) : 0.0;
        e.implied_vol_rmse =
            iv_count > 0 ? std::sqrt(iv_squared / static_cast<double>(iv_count)) : 0.0;
        return e;
    };

    // The unconstrained objective the optimizer minimises. The Price objective does not
    // need per-evaluation inversions, so it skips them.
    const std::function<double(std::span<const double>)> unconstrained_objective =
        [&](std::span<const double> x) -> double {
        std::array<double, 5> point{x[0], x[1], x[2], x[3], x[4]};
        const HestonParameters parameters = to_constrained(point, config.bounds);
        return evaluate(parameters, objective_is_iv).objective;
    };

    CalibrationResult result;
    result.starts.reserve(config.initial_guesses.size());

    for (const HestonParameters& guess : config.initial_guesses) {
        CalibrationStart start;
        start.initial = guess;

        const auto x0 = to_unconstrained(guess, config.bounds);
        if (!x0) {
            start.started = false;
            start.note = fmt::format("initial guess is not strictly inside the bounds: {}",
                                     x0.error().message);
            result.starts.push_back(std::move(start));
            continue;
        }

        const auto optimized = nelder_mead(
            unconstrained_objective, std::span<const double>(x0.value()), config.optimizer);
        if (!optimized) {
            return Result<CalibrationResult>::failure(optimized.error());
        }

        const std::array<double, 5> best_point{optimized.value().point[0],
                                               optimized.value().point[1],
                                               optimized.value().point[2],
                                               optimized.value().point[3],
                                               optimized.value().point[4]};
        start.calibrated = to_constrained(best_point, config.bounds);
        start.started = true;
        start.status = optimized.value().status;
        start.iterations = optimized.value().iterations;
        start.function_evaluations = optimized.value().function_evaluations;

        // Final evaluation with implied volatilities, for the reported RMSEs and the
        // residual surface -- one full evaluation, cheap against the whole search.
        const Evaluation final_eval = evaluate(start.calibrated, /*need_iv=*/true);
        start.objective_value = final_eval.objective;
        start.price_rmse = final_eval.price_rmse;
        start.implied_vol_rmse = final_eval.implied_vol_rmse;
        start.quotes_failed = final_eval.quotes_failed;
        if (final_eval.quotes_failed > 0) {
            start.note = fmt::format(
                "{} of {} quotes could not be priced or inverted at the calibrated parameters; the "
                "objective carried a penalty for them and the RMSEs are over the rest",
                final_eval.quotes_failed,
                n);
        }
        result.starts.push_back(std::move(start));
    }

    // The best started fit, and the dispersion across the started fits.
    std::optional<std::size_t> best_index;
    double objective_sum = 0.0;
    result.objective_min = std::numeric_limits<double>::infinity();
    result.objective_max = -std::numeric_limits<double>::infinity();
    std::vector<HestonParameters> calibrated;
    for (std::size_t i = 0; i < result.starts.size(); ++i) {
        const CalibrationStart& s = result.starts[i];
        if (!s.started) {
            continue;
        }
        ++result.started_count;
        objective_sum += s.objective_value;
        result.objective_min = std::min(result.objective_min, s.objective_value);
        result.objective_max = std::max(result.objective_max, s.objective_value);
        calibrated.push_back(s.calibrated);
        if (!best_index.has_value() ||
            s.objective_value < result.starts[*best_index].objective_value) {
            best_index = i;
        }
    }

    if (!best_index.has_value()) {
        return Result<CalibrationResult>::failure(
            ErrorCode::InvalidArgument,
            "no initial guess was strictly inside the bounds, so nothing could be calibrated; move "
            "the guesses inside the box",
            kContext);
    }

    result.best = result.starts[*best_index];
    result.objective_mean = objective_sum / static_cast<double>(result.started_count);

    // The residual surface at the best parameters.
    result.best_residuals = evaluate(result.best.calibrated, /*need_iv=*/true).residuals;

    // Parameter dispersion across the started fits.
    const auto mean_of = [&](const std::function<double(const HestonParameters&)>& get) {
        double sum = 0.0;
        for (const HestonParameters& p : calibrated) {
            sum += get(p);
        }
        return sum / static_cast<double>(calibrated.size());
    };
    const auto stddev_of = [&](const std::function<double(const HestonParameters&)>& get,
                               double mean) {
        if (calibrated.size() < 2) {
            return 0.0;
        }
        double sum = 0.0;
        for (const HestonParameters& p : calibrated) {
            const double d = get(p) - mean;
            sum += d * d;
        }
        return std::sqrt(sum / static_cast<double>(calibrated.size() - 1));
    };
    result.parameter_mean = {
        .initial_variance = mean_of([](const HestonParameters& p) { return p.initial_variance; }),
        .mean_reversion = mean_of([](const HestonParameters& p) { return p.mean_reversion; }),
        .long_run_variance = mean_of([](const HestonParameters& p) { return p.long_run_variance; }),
        .vol_of_variance = mean_of([](const HestonParameters& p) { return p.vol_of_variance; }),
        .correlation = mean_of([](const HestonParameters& p) { return p.correlation; })};
    result.parameter_stddev = {
        .initial_variance = stddev_of([](const HestonParameters& p) { return p.initial_variance; },
                                      result.parameter_mean.initial_variance),
        .mean_reversion = stddev_of([](const HestonParameters& p) { return p.mean_reversion; },
                                    result.parameter_mean.mean_reversion),
        .long_run_variance =
            stddev_of([](const HestonParameters& p) { return p.long_run_variance; },
                      result.parameter_mean.long_run_variance),
        .vol_of_variance = stddev_of([](const HestonParameters& p) { return p.vol_of_variance; },
                                     result.parameter_mean.vol_of_variance),
        .correlation = stddev_of([](const HestonParameters& p) { return p.correlation; },
                                 result.parameter_mean.correlation)};

    // Identifiability: among the fits that explain the surface about as well as the
    // best, are any materially far from it in parameter space? Those are the non-unique
    // calibrations a low objective alone would have concealed.
    const double similar_threshold = result.best.objective_value * config.similar_fit_factor;
    for (std::size_t i = 0; i < result.starts.size(); ++i) {
        if (i == *best_index) {
            continue;
        }
        const CalibrationStart& s = result.starts[i];
        if (!s.started || s.objective_value > similar_threshold) {
            continue;
        }
        const double distance =
            normalized_distance(s.calibrated, result.best.calibrated, config.bounds);
        result.max_similar_fit_distance = std::max(result.max_similar_fit_distance, distance);
        if (distance > config.material_parameter_distance) {
            result.non_unique = true;
            result.similar_fits.push_back(s.calibrated);
        }
    }

    return Result<CalibrationResult>::success(std::move(result));
}

}  // namespace diffusionworks
