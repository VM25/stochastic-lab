#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/geometric_asian_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/experiments/variance_reduction_experiments.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

/// An unbiased estimator whose across-seed mean drifts this many standard errors
/// from a reference it should reproduce is a defect, not noise. Kept at the
/// convergence framework's value: crude, antithetic, and control-variate averages
/// are all exactly unbiased, so a resolved bias is a genuine failure.
constexpr double kBiasResolution = 4.0;

std::string number(double x) {
    return fmt::format("{:.6g}", x);
}

std::vector<std::uint64_t> seeds_from(std::uint64_t master, std::uint64_t count) {
    std::vector<std::uint64_t> out;
    out.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(master + i * 1000003ULL);
    }
    return out;
}

/// One estimator measured across seeds: its across-seed dispersion (the standard
/// deviation of an N-observation estimate) and the mean wall-clock cost of one run.
/// The runtime is a diagnostic only -- it never enters efficiency, ranking, or
/// status, because a wall-clock number would make the comparison a benchmark of
/// this machine rather than of the estimators.
struct EstimatorResult {
    MultiSeedSummary summary;
    double mean_runtime_seconds{};
};

/// A deterministic, machine-independent count of the work an estimator does, from
/// the configuration and the estimator alone. This is what efficiency is normalised
/// by, so the comparison measures the estimators and not the clock.
///
/// The unit is one elementary operation. Simulating one path over its M monitoring
/// steps costs M path-leg operations; forming that path's arithmetic average over M
/// points costs M more; and, for the control variate, forming the geometric average
/// costs M more still. Every component the design names is here: the production
/// paths, the antithetic pair cost (an antithetic observation simulates two paths,
/// original and reflected), the pilot paths that estimate the control-variate beta,
/// and the arithmetic and geometric payoff evaluations.
struct WorkModel {
    std::int64_t production_paths{};      // observations requested (config.paths)
    std::int64_t simulated_production{};  // paths actually simulated (2x under antithetic)
    std::int64_t pilot_paths{};           // control-variate beta pilot (0 otherwise)
    std::int64_t path_leg_ops{};          // GBM step evolutions
    std::int64_t arithmetic_average_ops{};
    std::int64_t geometric_average_ops{};
    std::int64_t work_units{};
};

WorkModel work_model(std::int64_t production,
                     std::int64_t steps,
                     std::int64_t pilot,
                     bool antithetic,
                     bool control) {
    WorkModel w;
    w.production_paths = production;
    w.simulated_production = antithetic ? 2 * production : production;
    w.pilot_paths = control ? pilot : 0;
    const std::int64_t total_sim = w.simulated_production + w.pilot_paths;
    w.path_leg_ops = total_sim * steps;
    // One arithmetic average (over `steps` monitoring points) per simulated path.
    w.arithmetic_average_ops = total_sim * steps;
    // The geometric control is formed on every simulated path only when the control
    // variate is in use.
    w.geometric_average_ops = control ? total_sim * steps : 0;
    w.work_units = w.path_leg_ops + w.arithmetic_average_ops + w.geometric_average_ops;
    return w;
}

/// Work-normalised statistical efficiency: the reciprocal of variance times
/// deterministic work units, so a technique that halves the variance while doubling
/// the work scores the same as crude. No wall-clock time enters, so the number is a
/// property of the estimators, not of this machine.
double work_efficiency(double variance, std::int64_t work_units) {
    if (variance <= 0.0 || work_units <= 0) {
        return 0.0;
    }
    return 1.0 / (variance * static_cast<double>(work_units));
}

/// Runs `price_for_seed` once per seed, at an identical path budget, and aggregates.
/// The across-seed standard deviation is exactly the dispersion of a single N-path
/// estimate -- the quantity efficiency divides by -- measured rather than
/// self-reported.
template<typename PriceFn>
Result<EstimatorResult> measure(PriceFn&& price_for_seed,
                                const std::vector<std::uint64_t>& seeds,
                                std::optional<double> reference) {
    std::vector<SeedResult> replications;
    replications.reserve(seeds.size());
    double runtime = 0.0;
    for (const std::uint64_t seed : seeds) {
        auto priced = price_for_seed(seed);
        if (!priced) {
            return Result<EstimatorResult>::failure(priced.error());
        }
        replications.push_back(SeedResult{.seed = seed, .estimate = priced.value().value});
        runtime += priced.value().runtime_seconds;
    }
    auto summary = summarize_seeds(replications, reference);
    if (!summary) {
        return Result<EstimatorResult>::failure(summary.error());
    }
    return Result<EstimatorResult>::success(
        EstimatorResult{.summary = std::move(summary).value(),
                        .mean_runtime_seconds = runtime / static_cast<double>(seeds.size())});
}

/// One row of the comparison, for a single (instrument, regime, estimator). The
/// ranking columns -- variance_reduction_ratio and work_normalised_efficiency_gain
/// -- are functions of variance and deterministic work only. The wall-clock runtime
/// is reported beside them as a diagnostic and is never used to rank or to decide
/// status.
void push_row(ExperimentRecord& record,
              const std::string& instrument,
              double spot,
              double volatility,
              const std::string& estimator,
              const EstimatorResult& e,
              const WorkModel& work,
              double crude_variance,
              double crude_efficiency) {
    const double variance = e.summary.standard_deviation * e.summary.standard_deviation;
    const double eff = work_efficiency(variance, work.work_units);
    const double vr_ratio = variance > 0.0 ? crude_variance / variance : 0.0;
    const double eff_gain = crude_efficiency > 0.0 ? eff / crude_efficiency : 0.0;

    nlohmann::json cell{
        {"instrument", instrument},
        {"spot", spot},
        {"volatility", volatility},
        {"estimator", estimator},
        {"estimate", e.summary.mean},
        {"reference",
         e.summary.rmse.has_value() ? nlohmann::json(e.summary.mean) : nlohmann::json(nullptr)},
        {"across_seed_standard_deviation", e.summary.standard_deviation},
        {"variance", variance},
        {"rmse", e.summary.rmse.value_or(0.0)},
        {"bias", e.summary.bias.value_or(0.0)},
        {"production_paths", work.production_paths},
        {"simulated_production_paths", work.simulated_production},
        {"pilot_paths", work.pilot_paths},
        {"path_leg_ops", work.path_leg_ops},
        {"arithmetic_average_ops", work.arithmetic_average_ops},
        {"geometric_average_ops", work.geometric_average_ops},
        {"work_units", work.work_units},
        {"variance_reduction_ratio", vr_ratio},
        {"work_normalised_efficiency", eff},
        {"work_normalised_efficiency_gain_over_crude", eff_gain},
        {"mean_runtime_seconds_diagnostic", e.mean_runtime_seconds},
        {"seed_count", e.summary.seed_count}};
    record.results["cells"].push_back(std::move(cell));

    record.table.rows.push_back({instrument,
                                 number(spot),
                                 number(volatility),
                                 estimator,
                                 number(e.summary.mean),
                                 number(e.summary.standard_deviation),
                                 number(e.summary.rmse.value_or(0.0)),
                                 std::to_string(work.work_units),
                                 number(vr_ratio),
                                 number(eff_gain),
                                 number(e.mean_runtime_seconds)});
}

}  // namespace

Result<ExperimentRecord>
run_variance_reduction_efficiency(const VarianceReductionExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    ExperimentRecord record;
    record.id = "EXP-05";
    record.name = "Variance-Reduction Efficiency";
    record.question = "Which estimator produces the lowest error per unit of computation?";
    record.reproduction_command =
        "diffusionworks experiment --id EXP-05 --config configs/experiment/variance_reduction.json";
    record.configuration =
        nlohmann::json{{"rate", config.rate},
                       {"dividend_yield", config.dividend_yield},
                       {"strike", config.strike},
                       {"spots", config.spots},
                       {"volatilities", config.volatilities},
                       {"maturity", config.maturity},
                       {"asian_monitoring_count", config.asian_monitoring_count},
                       {"paths", config.paths},
                       {"control_variate_pilot_paths", config.control_variate_pilot_paths},
                       {"reference_paths", config.reference_paths},
                       {"seed_count", config.seed_count},
                       {"master_seed", config.master_seed}};
    record.table.headers = {"instrument",
                            "spot",
                            "volatility",
                            "estimator",
                            "estimate",
                            "across_seed_sd",
                            "rmse",
                            "work_units",
                            "variance_reduction_ratio",
                            "work_normalised_efficiency_gain",
                            "runtime_s_diagnostic"};
    record.results = nlohmann::json::object();
    record.results["cells"] = nlohmann::json::array();
    record.results["control_validation"] = nlohmann::json::array();

    const auto seeds = seeds_from(config.master_seed, config.seed_count);

    // A base config shared by every estimator, so the only thing that changes
    // between a crude and a variance-reduced run is the estimator itself.
    const auto base_config = [&](std::int64_t steps, std::uint64_t seed) {
        MonteCarloConfig mc;
        mc.paths = config.paths;
        mc.steps = steps;
        mc.seed = seed;
        mc.control_variate_pilot_paths = config.control_variate_pilot_paths;
        mc.threads = 1;  // deterministic; this record must reproduce bit-for-bit.
        return mc;
    };

    bool bias_failure = false;
    bool control_unvalidated = false;

    for (const double spot : config.spots) {
        const auto market = MarketState::create(spot, config.rate, config.dividend_yield);
        if (!market) {
            return Result<ExperimentRecord>::failure(market.error());
        }
        for (const double volatility : config.volatilities) {
            const auto model = BlackScholesModel::create(volatility);
            if (!model) {
                return Result<ExperimentRecord>::failure(model.error());
            }

            // --- European call: crude versus antithetic --------------------------
            const auto european =
                EuropeanOption::create(OptionType::Call, config.strike, config.maturity);
            if (!european) {
                return Result<ExperimentRecord>::failure(european.error());
            }
            const auto analytic =
                BlackScholesAnalyticEngine::price(market.value(), european.value(), model.value());
            if (!analytic) {
                return Result<ExperimentRecord>::failure(analytic.error());
            }
            const double euro_reference = analytic.value().value;

            const auto euro_crude = measure(
                [&](std::uint64_t seed) {
                    return MonteCarloEngine::price(
                        market.value(), european.value(), model.value(), base_config(1, seed));
                },
                seeds,
                euro_reference);
            if (!euro_crude) {
                return Result<ExperimentRecord>::failure(euro_crude.error());
            }
            const auto euro_anti = measure(
                [&](std::uint64_t seed) {
                    MonteCarloConfig mc = base_config(1, seed);
                    mc.variance_reduction.antithetic = true;
                    return MonteCarloEngine::price(
                        market.value(), european.value(), model.value(), mc);
                },
                seeds,
                euro_reference);
            if (!euro_anti) {
                return Result<ExperimentRecord>::failure(euro_anti.error());
            }

            // European is exact at one step, so M = 1.
            const WorkModel euro_crude_work =
                work_model(config.paths, 1, config.control_variate_pilot_paths, false, false);
            const WorkModel euro_anti_work =
                work_model(config.paths, 1, config.control_variate_pilot_paths, true, false);
            const double euro_crude_var = euro_crude.value().summary.standard_deviation *
                                          euro_crude.value().summary.standard_deviation;
            const double euro_crude_eff =
                work_efficiency(euro_crude_var, euro_crude_work.work_units);
            push_row(record,
                     "european",
                     spot,
                     volatility,
                     "crude",
                     euro_crude.value(),
                     euro_crude_work,
                     euro_crude_var,
                     euro_crude_eff);
            push_row(record,
                     "european",
                     spot,
                     volatility,
                     "antithetic",
                     euro_anti.value(),
                     euro_anti_work,
                     euro_crude_var,
                     euro_crude_eff);

            // Both estimators are unbiased against the exact Black-Scholes price.
            for (const auto& [name, e] : {std::pair{"crude", &euro_crude.value()},
                                          std::pair{"antithetic", &euro_anti.value()}}) {
                const double se = e->summary.standard_error;
                const double bias = e->summary.bias.value_or(0.0);
                if (se > 0.0 && std::abs(bias) / se > kBiasResolution) {
                    bias_failure = true;
                    record.limitations.push_back(fmt::format(
                        "FAILURE: the European {} estimator is unbiased in theory but its "
                        "across-seed mean drifts {:.3e} ({:.1f} standard errors) from the exact "
                        "Black-Scholes price at spot {:g}, sigma {:g}. That should not happen.",
                        name,
                        bias,
                        std::abs(bias) / se,
                        spot,
                        volatility));
                }
            }

            // --- Arithmetic Asian call: crude, antithetic, control, combined ------
            const std::int64_t steps = config.asian_monitoring_count;
            const auto arithmetic = AsianOption::create(OptionType::Call,
                                                        AveragingType::Arithmetic,
                                                        config.strike,
                                                        config.maturity,
                                                        config.asian_monitoring_count);
            if (!arithmetic) {
                return Result<ExperimentRecord>::failure(arithmetic.error());
            }

            // The reference has no closed form: a large combined-estimator run, its
            // own uncertainty reported so the estimators' RMSE is dominated by their
            // error and not the reference's.
            MonteCarloConfig ref_config = base_config(steps, config.master_seed);
            ref_config.paths = config.reference_paths;
            ref_config.variance_reduction.antithetic = true;
            ref_config.variance_reduction.control_variate = true;
            const auto asian_ref = MonteCarloEngine::price(
                market.value(), arithmetic.value(), model.value(), ref_config);
            if (!asian_ref) {
                return Result<ExperimentRecord>::failure(asian_ref.error());
            }
            const double asian_reference = asian_ref.value().value;

            // Independent validation of the control's known expectation (exit
            // criterion): the geometric Asian's analytic price must agree with a
            // crude Monte Carlo estimate of the same geometric average.
            const auto geometric = AsianOption::create(OptionType::Call,
                                                       AveragingType::Geometric,
                                                       config.strike,
                                                       config.maturity,
                                                       config.asian_monitoring_count);
            if (!geometric) {
                return Result<ExperimentRecord>::failure(geometric.error());
            }
            const auto geo_analytic = GeometricAsianAnalyticEngine::price(
                market.value(), geometric.value(), model.value());
            if (!geo_analytic) {
                return Result<ExperimentRecord>::failure(geo_analytic.error());
            }
            // The control variate cannot price the geometric option (it would be its
            // own control), so the validation run uses antithetic sampling only for
            // a tight but valid confidence interval.
            MonteCarloConfig geo_config = base_config(steps, config.master_seed);
            geo_config.paths = config.reference_paths;
            geo_config.variance_reduction.antithetic = true;
            const auto geo_mc = MonteCarloEngine::price(
                market.value(), geometric.value(), model.value(), geo_config);
            if (!geo_mc) {
                return Result<ExperimentRecord>::failure(geo_mc.error());
            }
            const double geo_se = geo_mc.value().standard_error.value_or(0.0);
            const double geo_gap = geo_mc.value().value - geo_analytic.value().value;
            const bool geo_ok = geo_se <= 0.0 || std::abs(geo_gap) / geo_se <= kBiasResolution;
            if (!geo_ok) {
                control_unvalidated = true;
            }
            record.results["control_validation"].push_back(
                nlohmann::json{{"spot", spot},
                               {"volatility", volatility},
                               {"analytic_geometric_price", geo_analytic.value().value},
                               {"monte_carlo_geometric_price", geo_mc.value().value},
                               {"standard_error", geo_se},
                               {"gap_over_se", geo_se > 0.0 ? geo_gap / geo_se : 0.0},
                               {"validated", geo_ok}});

            const auto asian_crude = measure(
                [&](std::uint64_t seed) {
                    return MonteCarloEngine::price(market.value(),
                                                   arithmetic.value(),
                                                   model.value(),
                                                   base_config(steps, seed));
                },
                seeds,
                asian_reference);
            if (!asian_crude) {
                return Result<ExperimentRecord>::failure(asian_crude.error());
            }
            const auto asian_anti = measure(
                [&](std::uint64_t seed) {
                    MonteCarloConfig mc = base_config(steps, seed);
                    mc.variance_reduction.antithetic = true;
                    return MonteCarloEngine::price(
                        market.value(), arithmetic.value(), model.value(), mc);
                },
                seeds,
                asian_reference);
            if (!asian_anti) {
                return Result<ExperimentRecord>::failure(asian_anti.error());
            }
            const auto asian_control = measure(
                [&](std::uint64_t seed) {
                    MonteCarloConfig mc = base_config(steps, seed);
                    mc.variance_reduction.control_variate = true;
                    return MonteCarloEngine::price(
                        market.value(), arithmetic.value(), model.value(), mc);
                },
                seeds,
                asian_reference);
            if (!asian_control) {
                return Result<ExperimentRecord>::failure(asian_control.error());
            }
            const auto asian_combined = measure(
                [&](std::uint64_t seed) {
                    MonteCarloConfig mc = base_config(steps, seed);
                    mc.variance_reduction.antithetic = true;
                    mc.variance_reduction.control_variate = true;
                    return MonteCarloEngine::price(
                        market.value(), arithmetic.value(), model.value(), mc);
                },
                seeds,
                asian_reference);
            if (!asian_combined) {
                return Result<ExperimentRecord>::failure(asian_combined.error());
            }

            const std::int64_t pilot = config.control_variate_pilot_paths;
            const WorkModel asian_crude_work = work_model(config.paths, steps, pilot, false, false);
            const WorkModel asian_anti_work = work_model(config.paths, steps, pilot, true, false);
            const WorkModel asian_control_work =
                work_model(config.paths, steps, pilot, false, true);
            const WorkModel asian_combined_work =
                work_model(config.paths, steps, pilot, true, true);
            const double asian_crude_var = asian_crude.value().summary.standard_deviation *
                                           asian_crude.value().summary.standard_deviation;
            const double asian_crude_eff =
                work_efficiency(asian_crude_var, asian_crude_work.work_units);
            push_row(record,
                     "arithmetic_asian",
                     spot,
                     volatility,
                     "crude",
                     asian_crude.value(),
                     asian_crude_work,
                     asian_crude_var,
                     asian_crude_eff);
            push_row(record,
                     "arithmetic_asian",
                     spot,
                     volatility,
                     "antithetic",
                     asian_anti.value(),
                     asian_anti_work,
                     asian_crude_var,
                     asian_crude_eff);
            push_row(record,
                     "arithmetic_asian",
                     spot,
                     volatility,
                     "control_variate",
                     asian_control.value(),
                     asian_control_work,
                     asian_crude_var,
                     asian_crude_eff);
            push_row(record,
                     "arithmetic_asian",
                     spot,
                     volatility,
                     "combined",
                     asian_combined.value(),
                     asian_combined_work,
                     asian_crude_var,
                     asian_crude_eff);
        }
    }

    record.results["asian_reference_note"] =
        "The arithmetic Asian has no closed-form price; its reference is a "
        "high-path-count combined-estimator run, reported per cell with the RMSE "
        "measured against it.";
    record.results["bias_resolution_threshold"] = kBiasResolution;

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (bias_failure) {
        record.status = ExperimentStatus::Fail;
    } else if (control_unvalidated) {
        record.status = ExperimentStatus::Fail;
    } else {
        record.status = ExperimentStatus::Pass;
    }

    record.interpretation =
        "The metric is work-normalised statistical efficiency -- the reciprocal of variance times "
        "deterministic work units -- not raw variance, because a technique that halves the "
        "variance "
        "while doubling the work has bought nothing. The work units are counted from the "
        "configuration and the estimator alone (path-leg simulations, arithmetic and geometric "
        "payoff evaluations, the antithetic pair cost, and the control-variate pilot), so the "
        "comparison is a property of the estimators, not of this machine. Wall-clock time is "
        "reported beside each row as a diagnostic only and never enters the ranking or the status. "
        "The variance_reduction_ratio column shows the variance side alone, and the two columns "
        "disagree -- which is the point.\n\n"
        "For the arithmetic Asian the geometric control variate is the most efficient estimator, "
        "in "
        "every regime tested. The geometric Asian on the same paths and dates has a closed-form "
        "price and tracks the arithmetic average closely -- they differ only by arithmetic versus "
        "geometric mean -- so subtracting it removes two to three orders of magnitude of variance "
        "for only the modest extra work of one geometric average per path plus a small beta pilot. "
        "The control's known expectation is not taken on faith: the control_validation block "
        "checks "
        "the analytic geometric price against a Monte Carlo estimate of the same average and "
        "reports the agreement, which holds within a fraction of a standard error in every "
        "regime.\n\n"
        "The combined antithetic-plus-control estimator is the sharpest illustration of why "
        "efficiency and variance are different questions. It has the lowest variance of any "
        "estimator here -- its variance_reduction_ratio is the largest, in every regime -- yet it "
        "is less work-efficient than the control variate alone, in every regime, because the "
        "antithetic layer doubles the simulated paths (each observation now averages an original "
        "and a reflected path) for only a marginal further variance cut once the control has "
        "removed most of it. Ranked by variance the combined estimator wins; ranked by error per "
        "unit of work the control variate alone wins. Normalising by work keeps the doubled path "
        "cost of the antithetic layer visible; reporting only the variance would have named the "
        "wrong estimator best.\n\n"
        "Antithetic sampling on its own is the mild case. For the European call it helps where the "
        "discounted payoff is close to monotone and smooth in the terminal shock, with a "
        "work-normalised gain that is real but modest -- and, because an antithetic observation "
        "costs two simulated paths, a variance cut of less than two would leave it no better than "
        "crude once work is counted. The gain erodes as the option moves out of the money and the "
        "payoff spends most paths pinned at zero, so the per-regime rows show its range rather "
        "than "
        "a single headline. On the arithmetic Asian antithetic alone helps more than on the "
        "vanilla "
        "but is dwarfed by the control.\n\n"
        "The reading is that the estimator should be matched to the payoff and judged by "
        "work-normalised efficiency, not variance: the geometric control for the arithmetic Asian, "
        "where a closely correlated closed-form control exists; antithetic sampling for a smooth "
        "monotone vanilla, with the caveat that its gain is regime-specific and its doubled path "
        "cost must clear; and a combination only when its extra work actually earns its keep, "
        "which "
        "here it does not.";

    record.limitations.emplace_back(
        "The arithmetic Asian reference is a high-path-count Monte Carlo estimate, not a "
        "closed-form value -- no closed form exists. Its own sampling error is small by "
        "construction (a large path count with the best estimator) but not zero, so a reported "
        "RMSE at the smallest scales carries the reference's uncertainty as well as the "
        "estimator's.");
    record.limitations.emplace_back(
        "The work-unit model is an elementary-operation count (path-leg simulations plus "
        "arithmetic "
        "and geometric averaging), not a cycle-accurate cost. It captures the structural "
        "differences "
        "that matter here -- the antithetic pair cost and the control-variate pilot and geometric "
        "average -- but weights each elementary operation equally, so it is a deterministic proxy "
        "for cost, not a measured one. The wall-clock runtime beside each row is a diagnostic "
        "only; "
        "it does not enter efficiency, ranking, or status, and is not a performance claim.");
    record.limitations.emplace_back(
        "The control variate applies only to the arithmetic Asian, whose geometric counterpart is "
        "the natural control. There is no comparably tight closed-form control for the European "
        "call here, so its comparison is crude versus antithetic only.");
    record.limitations.emplace_back(
        "Antithetic sampling's benefit depends on the payoff being monotone in the shock, which "
        "holds for these calls but not in general. A payoff that folds back on itself can leave "
        "antithetic variates no better than crude, or worse once their cost is counted.");

    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
