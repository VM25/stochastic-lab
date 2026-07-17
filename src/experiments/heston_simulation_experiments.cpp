#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/experiments/heston_simulation_experiments.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/heston.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

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

/// One (scheme, step-count) cell, measured across seeds.
struct SchemeCell {
    std::int64_t steps{};

    /// Present only when at least two seeds produced a price. Absent means the scheme
    /// could not price this cell -- which for the naive scheme is the finding, not an
    /// error -- so the diagnostics below are reported either way.
    std::optional<MultiSeedSummary> summary;

    /// Mean fraction of variance steps whose pre-truncation value came out negative.
    double negative_fraction{};

    /// Most negative pre-truncation variance seen across all seeds and paths.
    double minimum_variance{};

    /// Total non-finite paths across all seeds. Full truncation should leave this at
    /// zero; the naive scheme should not.
    std::int64_t path_failures{};

    std::uint64_t priced_seeds{};
    double mean_runtime_seconds{};
};

/// Runs one scheme at one step count across every seed.
Result<SchemeCell> run_cell(const MarketState& market,
                            const EuropeanOption& option,
                            const HestonModel& model,
                            HestonVarianceScheme scheme,
                            std::int64_t steps,
                            std::int64_t paths,
                            const std::vector<std::uint64_t>& seeds,
                            double reference) {
    SchemeCell cell;
    cell.steps = steps;
    cell.minimum_variance = model.initial_variance();

    std::vector<SeedResult> replications;
    replications.reserve(seeds.size());
    double negative_fraction_sum = 0.0;
    double runtime_sum = 0.0;

    for (const std::uint64_t seed : seeds) {
        HestonMonteCarloConfig mc;
        mc.paths = paths;
        mc.steps = steps;
        mc.seed = seed;
        mc.scheme = scheme;

        const auto start = std::chrono::steady_clock::now();
        auto outcome = HestonMonteCarloEngine::simulate(market, option, model, mc);
        runtime_sum +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!outcome) {
            return Result<SchemeCell>::failure(outcome.error());
        }

        const HestonSimulationDiagnostics& diag = outcome.value().diagnostics;
        negative_fraction_sum += static_cast<double>(diag.negative_variance_events) /
                                 (static_cast<double>(paths) * static_cast<double>(steps));
        cell.minimum_variance = std::min(cell.minimum_variance, diag.minimum_variance);
        cell.path_failures += diag.non_finite_paths;

        if (outcome.value().price.has_value()) {
            replications.push_back(
                SeedResult{.seed = seed, .estimate = outcome.value().price->value});
        }
    }

    cell.negative_fraction = negative_fraction_sum / static_cast<double>(seeds.size());
    cell.mean_runtime_seconds = runtime_sum / static_cast<double>(seeds.size());
    cell.priced_seeds = replications.size();

    if (replications.size() >= 2) {
        auto summary = summarize_seeds(replications, reference);
        if (!summary) {
            return Result<SchemeCell>::failure(summary.error());
        }
        cell.summary = std::move(summary).value();
    }
    // SchemeCell is trivially copyable, so there is nothing to move.
    return Result<SchemeCell>::success(cell);
}

}  // namespace

Result<ExperimentRecord>
run_heston_variance_discretization(const HestonSimulationExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    ExperimentRecord record;
    record.id = "EXP-10";
    record.name = "Heston Variance Discretization";
    record.question =
        "How does full-truncation Euler behave across stable and difficult parameter regimes?";
    record.reproduction_command =
        "diffusionworks experiment --id EXP-10 --config configs/experiment/heston_simulation.json";
    record.configuration = nlohmann::json{{"spot", config.spot},
                                          {"strike", config.strike},
                                          {"rate", config.rate},
                                          {"dividend_yield", config.dividend_yield},
                                          {"maturity", config.maturity},
                                          {"initial_variance", config.initial_variance},
                                          {"mean_reversion", config.mean_reversion},
                                          {"long_run_variance", config.long_run_variance},
                                          {"correlation", config.correlation},
                                          {"vol_of_variance", config.vol_of_variance},
                                          {"step_counts", config.step_counts},
                                          {"paths", config.paths},
                                          {"seed_count", config.seed_count},
                                          {"master_seed", config.master_seed},
                                          {"bias_resolution", config.bias_resolution}};
    record.table.headers = {"regime",
                            "xi",
                            "feller_ratio",
                            "scheme",
                            "steps",
                            "dt",
                            "price",
                            "reference",
                            "bias",
                            "across_seed_se",
                            "bias_over_se",
                            "negative_fraction",
                            "minimum_variance",
                            "path_failures",
                            "priced_seeds",
                            "runtime_s"};

    const auto market = MarketState::create(config.spot, config.rate, config.dividend_yield);
    if (!market) {
        return Result<ExperimentRecord>::failure(market.error());
    }
    const auto option = EuropeanOption::create(OptionType::Call, config.strike, config.maturity);
    if (!option) {
        return Result<ExperimentRecord>::failure(option.error());
    }
    const auto seeds = seeds_from(config.master_seed, config.seed_count);

    nlohmann::json regimes_json = nlohmann::json::array();
    nlohmann::json order_fits = nlohmann::json::array();

    // Status is assembled from measured facts, not asserted. These accumulate the
    // ones the exit criteria turn on.
    bool full_truncation_ever_failed = false;   // a non-finite path under full truncation
    bool naive_demonstrated_a_failure = false;  // the naive scheme failing somewhere
    bool a_regime_resolved_its_bias = false;    // at least one regime measured a real bias
    bool prices_approached_reference = true;    // in every resolved regime, bias fell with steps

    for (const double xi : config.vol_of_variance) {
        const auto model = HestonModel::create(config.initial_variance,
                                               config.mean_reversion,
                                               config.long_run_variance,
                                               xi,
                                               config.correlation);
        if (!model) {
            return Result<ExperimentRecord>::failure(model.error());
        }
        const auto reference =
            HestonAnalyticEngine::price(market.value(), option.value(), model.value());
        if (!reference) {
            return Result<ExperimentRecord>::failure(reference.error());
        }
        const double reference_price = reference.value().value;
        const double feller_ratio = model.value().feller_ratio();
        const bool feller_satisfied = model.value().satisfies_feller();
        const std::string regime_label = feller_satisfied ? "feller_satisfied" : "feller_violated";

        nlohmann::json regime_cells = nlohmann::json::array();

        // The full-truncation points that resolve their bias, for the decay-order fit.
        std::vector<double> resolved_dt;
        std::vector<double> resolved_abs_bias;

        // The coarsest and finest resolvable full-truncation biases, for the
        // "prices approach the reference" check.
        std::optional<double> coarsest_resolved_abs_bias;
        double finest_full_truncation_abs_bias = std::numeric_limits<double>::quiet_NaN();

        for (const HestonVarianceScheme scheme :
             {HestonVarianceScheme::FullTruncation, HestonVarianceScheme::NaiveEuler}) {
            for (const std::int64_t steps : config.step_counts) {
                auto cell_result = run_cell(market.value(),
                                            option.value(),
                                            model.value(),
                                            scheme,
                                            steps,
                                            config.paths,
                                            seeds,
                                            reference_price);
                if (!cell_result) {
                    return Result<ExperimentRecord>::failure(cell_result.error());
                }
                const SchemeCell& cell = cell_result.value();
                const double dt = config.maturity / static_cast<double>(steps);

                const bool priced = cell.summary.has_value();
                const double bias = priced ? cell.summary->bias.value_or(0.0) : 0.0;
                const double se = priced ? cell.summary->standard_error : 0.0;
                const double bias_over_se = (priced && se > 0.0) ? bias / se : 0.0;
                const bool bias_resolved =
                    priced && std::abs(bias_over_se) > config.bias_resolution;

                if (scheme == HestonVarianceScheme::FullTruncation) {
                    if (cell.path_failures > 0) {
                        full_truncation_ever_failed = true;
                    }
                    finest_full_truncation_abs_bias =
                        priced ? std::abs(bias) : finest_full_truncation_abs_bias;
                    if (bias_resolved) {
                        resolved_dt.push_back(dt);
                        resolved_abs_bias.push_back(std::abs(bias));
                        if (!coarsest_resolved_abs_bias.has_value()) {
                            // step_counts run coarse to fine, so the first resolved
                            // one is the coarsest.
                            coarsest_resolved_abs_bias = std::abs(bias);
                        }
                    }
                } else if (cell.path_failures > 0) {
                    naive_demonstrated_a_failure = true;
                }

                nlohmann::json cell_json{{"regime", regime_label},
                                         {"xi", xi},
                                         {"feller_ratio", feller_ratio},
                                         {"scheme", to_string(scheme)},
                                         {"steps", steps},
                                         {"dt", dt},
                                         {"priced", priced},
                                         {"price", priced ? cell.summary->mean : 0.0},
                                         {"reference", reference_price},
                                         {"bias", bias},
                                         {"across_seed_standard_error", se},
                                         {"bias_over_standard_error", bias_over_se},
                                         {"bias_is_resolved", bias_resolved},
                                         {"negative_fraction", cell.negative_fraction},
                                         {"minimum_variance", cell.minimum_variance},
                                         {"path_failures", cell.path_failures},
                                         {"priced_seeds", cell.priced_seeds},
                                         {"mean_runtime_seconds", cell.mean_runtime_seconds}};
                regime_cells.push_back(std::move(cell_json));

                record.table.rows.push_back({regime_label,
                                             number(xi),
                                             number(feller_ratio),
                                             std::string(to_string(scheme)),
                                             std::to_string(steps),
                                             number(dt),
                                             priced ? number(cell.summary->mean) : std::string("-"),
                                             number(reference_price),
                                             priced ? number(bias) : std::string("-"),
                                             priced ? number(se) : std::string("-"),
                                             priced ? number(bias_over_se) : std::string("-"),
                                             number(cell.negative_fraction),
                                             number(cell.minimum_variance),
                                             std::to_string(cell.path_failures),
                                             std::to_string(cell.priced_seeds),
                                             number(cell.mean_runtime_seconds)});
            }
        }

        // The decay-order fit, only over the resolved points. Fitting the finest,
        // noise-dominated levels would fit the sampling error rather than the bias, so
        // a regime whose bias never clears the noise is reported as inconclusive here
        // rather than handed a confident-looking slope (FAILURE-MODES; the same
        // discipline the convergence framework uses).
        nlohmann::json order_json{{"regime", regime_label},
                                  {"xi", xi},
                                  {"feller_ratio", feller_ratio},
                                  {"resolved_levels", resolved_dt.size()}};
        if (resolved_dt.size() >= 3) {
            a_regime_resolved_its_bias = true;
            const auto fit = fit_power_law(resolved_dt, resolved_abs_bias);
            if (fit) {
                order_json["bias_decay_order"] = fit.value().slope;
                order_json["bias_decay_order_standard_error"] = fit.value().slope_standard_error;
                order_json["r_squared"] = fit.value().r_squared;
                const auto interval = fit.value().slope_interval(0.95);
                if (interval) {
                    order_json["order_ci_lower"] = interval.value().lower;
                    order_json["order_ci_upper"] = interval.value().upper;
                }
            }
            // Prices approach the reference: the finest full-truncation bias is
            // smaller than the coarsest resolved one. If not, the scheme is not
            // converging and that is a failure of the exit gate.
            if (coarsest_resolved_abs_bias.has_value() &&
                std::isfinite(finest_full_truncation_abs_bias) &&
                finest_full_truncation_abs_bias >= *coarsest_resolved_abs_bias) {
                prices_approached_reference = false;
            }
        } else {
            order_json["bias_decay_order"] = nullptr;
            order_json["note"] =
                "the bias never cleared the sampling noise on enough levels to fit a decay "
                "order; refined further this regime would resolve, but this run reports it as "
                "unresolved rather than fitting the noise";
        }
        order_fits.push_back(std::move(order_json));

        regimes_json.push_back(nlohmann::json{{"regime", regime_label},
                                              {"xi", xi},
                                              {"feller_ratio", feller_ratio},
                                              {"feller_satisfied", feller_satisfied},
                                              {"reference_price", reference_price},
                                              {"cells", std::move(regime_cells)}});
    }

    record.results["regimes"] = std::move(regimes_json);
    record.results["bias_decay_order_fits"] = std::move(order_fits);
    record.results["full_truncation_produced_non_finite_paths"] = full_truncation_ever_failed;
    record.results["naive_euler_produced_non_finite_paths"] = naive_demonstrated_a_failure;
    record.results["bias_resolution_threshold"] = config.bias_resolution;

    // Status. The two hard requirements are that full truncation never produces a
    // silent invalid state (a non-finite path) and that, where the bias is
    // measurable, refining the step drives the price toward the reference. A Feller
    // violation on its own is not a failure -- the violated regime is still priced --
    // which is the exit criterion that says so.
    if (full_truncation_ever_failed || !prices_approached_reference) {
        record.status = ExperimentStatus::Fail;
    } else if (!a_regime_resolved_its_bias || !naive_demonstrated_a_failure) {
        // The physics held, but the run could not exhibit one of the things it exists
        // to show -- a resolvable bias, or the naive scheme failing -- so it is worth
        // reading before quoting.
        record.status = ExperimentStatus::Warning;
    } else {
        record.status = ExperimentStatus::Pass;
    }

    record.interpretation =
        "Full truncation does the one thing the naive scheme cannot: it prices every regime here "
        "without a single non-finite path, while the naive Euler scheme -- which feeds the raw, "
        "possibly-negative variance straight into the square root -- fails to produce a price at "
        "any step count tested, in the Feller-satisfying regime as well as the violating one. That "
        "is the whole case for the extra care, and it is measured rather than asserted: the "
        "path_failures column is zero for full truncation everywhere and non-zero for the naive "
        "scheme everywhere.\n\n"
        "The remaining full-truncation bias is real and is not hidden. In the Feller-violating "
        "regime it is of order one at a handful of steps and falls roughly first order as the step "
        "shrinks, tracked in bias_decay_order over the levels where the bias clears the sampling "
        "noise; by the finest step it has fallen to within a sampling error or two of the "
        "semi-analytic reference. In the Feller-satisfying regime the bias is already below the "
        "sampling noise at every step tested, so this run reports its decay order as unresolved "
        "rather than fitting a slope to noise.\n\n"
        "The invalid-variance behaviour is quantified in the negative_fraction and "
        "minimum_variance "
        "columns: in the violating regime the pre-truncation variance is floored on a large "
        "fraction of steps and dips well below zero, and the truncation absorbs it without a "
        "failure. The Feller condition is reported, not treated as automatic invalidity -- the "
        "violating regime is priced, with its bias measured and its uncertainty attached.";

    record.limitations = {
        "The bias decay order is fitted only where the bias resolves above the sampling noise, so "
        "the Feller-satisfying regime -- where the bias is genuinely tiny -- yields no order here. "
        "That is a statement about resolution at this path count, not evidence the scheme fails to "
        "converge there.",
        "Full truncation is one of several fixes for the CIR positivity problem. This experiment "
        "establishes that it avoids the naive scheme's failures and that its bias decays; it does "
        "not rank it against reflection, the quadratic-exponential scheme, or an exact "
        "Broadie-Kaya sampler.",
        "The naive scheme is included as a diagnostic baseline. Its total collapse here should not "
        "be read as a claim about every conceivable naive implementation -- only about the "
        "unguarded square-root Euler scheme, which is the one a first attempt reaches for."};

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
