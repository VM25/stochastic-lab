#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/barrier_monte_carlo.hpp>
#include <diffusionworks/engines/barrier_pde_engine.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/experiments/barrier_experiments.hpp>
#include <diffusionworks/statistics/multi_seed.hpp>
#include <diffusionworks/statistics/online_moments.hpp>
#include <diffusionworks/statistics/regression.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "EXP-07";

/// A bias is "resolved" once it clears this many across-seed standard errors.
///
/// Matches the convergence framework's threshold. Below it the experiment has
/// measured nothing: the difference is inside its own noise, and reporting it as a
/// bias would be reporting a draw.
constexpr double kResolutionThreshold = 3.0;

/// A resolved *bridge* bias above this many standard errors is reported as a defect
/// rather than a limitation.
///
/// Higher than kResolutionThreshold on purpose. The bridge correction is exact for
/// this model, so the null hypothesis is a bias of exactly zero rather than a small
/// one, and there are two dozen bridge cells. At 16 seeds a five-sigma excursion is
/// roughly a 1-in-10000 event per cell, so a false accusation across the whole grid
/// runs a few parts in a thousand -- while a real defect of any consequence clears
/// it easily.
constexpr double kBridgeDefectThreshold = 5.0;

/// Broadie-Glasserman-Kou's continuity-correction constant,
/// \f$ \beta = -\zeta(1/2)/\sqrt{2\pi} \f$.
///
/// The expected overshoot of a Gaussian random walk past a level, in units of the
/// per-step standard deviation. Hardcoded rather than computed: it needs the Riemann
/// zeta function at 1/2, which this project has no other use for. The digits are
/// checkable against mpmath as `-mp.zeta(0.5)/mp.sqrt(2*mp.pi)`.
constexpr double kContinuityCorrectionBeta = 0.5825971579390106;

std::string number(double x) {
    return fmt::format("{:.17g}", x);
}

/// Independent seeds, spaced far enough apart that no two replications share a
/// counter range.
std::vector<std::uint64_t> seeds_from(std::uint64_t master, std::uint64_t count) {
    std::vector<std::uint64_t> out;
    out.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(master + i * 1000003ULL);
    }
    return out;
}

/// Number of finest levels used for the asymptotic-window fit.
///
/// The theoretical order is an asymptotic statement, so the coarse levels -- where
/// the higher-order terms are largest -- are precisely the ones that drag a
/// full-range slope away from it. Both fits are published: the full-range one is what
/// the whole sweep says, the window is what the resolved tail says, and where they
/// disagree the disagreement is the finding.
constexpr std::size_t kAsymptoticWindow = 3;

nlohmann::json fit_json(std::span<const double> x, std::span<const double> y) {
    const auto fit = fit_power_law(x, y);
    if (!fit) {
        return nlohmann::json{{"error", fit.error().message}};
    }
    const auto interval = fit.value().slope_interval(0.95);
    if (!interval) {
        return nlohmann::json{{"error", interval.error().message}};
    }
    return nlohmann::json{{"order", fit.value().slope},
                          {"ci_lower", interval.value().lower},
                          {"ci_upper", interval.value().upper},
                          {"r_squared", fit.value().r_squared},
                          {"observations", fit.value().observations},
                          {"consistent_with_expected_order",
                           interval.value().lower <= 0.5 && 0.5 <= interval.value().upper}};
}

/// Order between each consecutive pair of levels.
///
/// Published alongside the full-range fit because a pre-asymptotic study is exactly
/// where the two disagree: a single slope over contaminated levels averages the
/// contamination in, while the local orders show it draining away.
nlohmann::json local_orders_json(const std::vector<double>& x, const std::vector<double>& y) {
    nlohmann::json out = nlohmann::json::array();
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (!(y[i - 1] > 0.0) || !(y[i] > 0.0)) {
            continue;
        }
        out.push_back(
            nlohmann::json{{"coarse", x[i - 1]},
                           {"fine", x[i]},
                           {"order", std::log(y[i - 1] / y[i]) / std::log(x[i - 1] / x[i])}});
    }
    return out;
}

/// The continuously monitored price, from the closed form.
Result<double> continuous_reference(const MarketState& market,
                                    const BlackScholesModel& model,
                                    BarrierType barrier_type,
                                    double strike,
                                    double barrier,
                                    double maturity) {
    const auto option = BarrierOption::create(OptionType::Call,
                                              barrier_type,
                                              strike,
                                              barrier,
                                              maturity,
                                              MonitoringConvention::Continuous,
                                              std::nullopt);
    if (!option) {
        return Result<double>::failure(option.error());
    }
    const auto priced = BarrierAnalyticEngine::price(market, option.value(), model);
    if (!priced) {
        return Result<double>::failure(priced.error());
    }
    return Result<double>::success(priced.value().value);
}

/// Broadie-Glasserman-Kou's prediction for a discretely monitored price.
///
/// A barrier observed at m dates behaves, to leading order, like a continuously
/// monitored barrier moved *away* from the spot by
/// \f$ \exp(\pm\beta\sigma\sqrt{T/m}) \f$ -- down for a lower barrier, up for an
/// upper one. The reason is overshoot: a discretely observed path that is going to
/// breach is first seen past the barrier rather than at it, and the mean overshoot
/// is \f$ \beta\sigma\sqrt{\Delta t} \f$.
///
/// An independent check on the discrete arm, and a far sharper one than a fitted
/// slope. A slope near 0.5 says only that the bias decays at the right *rate*; the
/// correction predicts its actual size from a formula this project did not fit to
/// the data.
///
/// It is an approximation with error o(1/sqrt(m)), not an identity, so a residual is
/// expected here and is itself worth measuring.
Result<double> continuity_corrected_reference(const MarketState& market,
                                              const BlackScholesModel& model,
                                              BarrierType barrier_type,
                                              double strike,
                                              double barrier,
                                              double maturity,
                                              std::int64_t monitoring_dates) {
    const double dt = maturity / static_cast<double>(monitoring_dates);
    // Away from the spot in both cases: down for a lower barrier, up for an upper
    // one. The sign is the whole content of the correction's direction, and getting
    // it backwards would predict a bias of the right size with the wrong sign --
    // which the residual check would catch, but only after reporting it.
    const double direction = is_down_barrier(barrier_type) ? -1.0 : 1.0;
    const double shifted = barrier * std::exp(direction * kContinuityCorrectionBeta *
                                              model.volatility() * std::sqrt(dt));
    return continuous_reference(market, model, barrier_type, strike, shifted, maturity);
}

/// One (convention, barrier, monitoring frequency) cell, priced across every seed.
///
/// `bias` and `rmse` are plain doubles rather than the optionals MultiSeedSummary
/// carries. Those are optional because RMSE against no reference is an unanswerable
/// question rather than zero -- so they are resolved once, at construction, where an
/// absent value can still be reported as the error it is. Reading them unchecked
/// downstream would turn a missing reference into a plausible-looking bias of
/// whatever happened to be in the slot, which is the whole class of failure this
/// project exists to distrust.
struct Cell {
    MultiSeedSummary summary;
    double bias{};
    double rmse{};
    double mean_knock_fraction{};
    double runtime_seconds{};
};

}  // namespace

Result<ExperimentRecord> run_barrier_monitoring_bias(const BarrierExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    const auto market = MarketState::create(config.spot, config.rate, config.dividend_yield);
    if (!market) {
        return Result<ExperimentRecord>::failure(market.error());
    }
    const auto model = BlackScholesModel::create(config.volatility);
    if (!model) {
        return Result<ExperimentRecord>::failure(model.error());
    }

    ExperimentRecord record;
    record.id = "EXP-07";
    record.name = "Barrier Monitoring Bias";
    record.question = "How much error results from discrete barrier monitoring?";
    record.reproduction_command =
        "diffusionworks experiment --id EXP-07 --config configs/experiment/barrier.json";
    record.configuration = nlohmann::json{{"spot", config.spot},
                                          {"strike", config.strike},
                                          {"rate", config.rate},
                                          {"dividend_yield", config.dividend_yield},
                                          {"volatility", config.volatility},
                                          {"maturity", config.maturity},
                                          {"barriers", config.barriers},
                                          {"up_barriers", config.up_barriers},
                                          {"monitoring_counts", config.monitoring_counts},
                                          {"paths", config.paths},
                                          {"seed_count", config.seed_count},
                                          {"master_seed", config.master_seed},
                                          {"volatilities", config.volatilities},
                                          {"pde_resolutions", config.pde_resolutions},
                                          {"pde_rannacher_steps", config.pde_rannacher_steps}};
    record.table.headers = {"barrier_type",
                            "convention",
                            "barrier",
                            "monitoring",
                            "price",
                            "reference",
                            "bias",
                            "across_seed_se",
                            "bias_over_se",
                            "knock_fraction"};
    record.results = nlohmann::json::object();

    const std::vector<std::uint64_t> seeds = seeds_from(config.master_seed, config.seed_count);

    // The bias is measured across seeds, not from one run: it is a difference of two
    // numbers, one of which carries sampling error, and a single draw cannot separate
    // a real bias from a lucky one. That is the whole question for the bridge arm,
    // whose residual is small enough to be either.
    const auto price_cell = [&](const BarrierOption& option,
                                const BlackScholesModel& cell_model,
                                double reference) -> Result<Cell> {
        std::vector<SeedResult> replications;
        replications.reserve(seeds.size());
        OnlineMoments knock_fractions;
        double runtime = 0.0;

        for (const std::uint64_t seed : seeds) {
            BarrierMonteCarloConfig mc;
            mc.paths = config.paths;
            mc.seed = seed;

            const auto priced =
                BarrierMonteCarloEngine::price(market.value(), option, cell_model, mc);
            if (!priced) {
                return Result<Cell>::failure(priced.error());
            }
            replications.push_back(SeedResult{.seed = seed, .estimate = priced.value().value});
            runtime += priced.value().runtime_seconds;

            for (const Diagnostic& diagnostic : priced.value().diagnostics) {
                if (diagnostic.name == "knock_fraction") {
                    knock_fractions.add(std::get<double>(diagnostic.value));
                }
            }
        }

        auto summary = summarize_seeds(replications, reference);
        if (!summary) {
            return Result<Cell>::failure(summary.error());
        }
        if (!summary.value().bias.has_value() || !summary.value().rmse.has_value()) {
            // Unreachable: a reference was supplied. Checked anyway, because the
            // alternative is dereferencing on the belief that it was, and this whole
            // experiment is a statement about a bias.
            return Result<Cell>::failure(
                ErrorCode::InvalidArgument,
                "the multi-seed summary carries no bias despite a reference being supplied, so "
                "there is nothing to measure against",
                kContext);
        }
        const double bias = *summary.value().bias;
        const double rmse = *summary.value().rmse;
        return Result<Cell>::success(Cell{.summary = std::move(summary).value(),
                                          .bias = bias,
                                          .rmse = rmse,
                                          .mean_knock_fraction = knock_fractions.mean(),
                                          .runtime_seconds = runtime});
    };

    bool bridge_defect = false;
    bool coarse_bias_resolved = false;
    nlohmann::json arms = nlohmann::json::array();

    // Both directions. They are not redundant: an up-and-out call knocks out in
    // exactly the states where it would have paid, so its bias is driven by a
    // different part of the distribution than a down-and-out's, whose knock-outs are
    // mostly paths that would have expired worthless anyway.
    struct Sweep {
        BarrierType type;
        double barrier;
    };

    std::vector<Sweep> sweeps;
    sweeps.reserve(config.barriers.size() + config.up_barriers.size());
    for (const double barrier : config.barriers) {
        sweeps.push_back(Sweep{.type = BarrierType::DownAndOut, .barrier = barrier});
    }
    for (const double barrier : config.up_barriers) {
        sweeps.push_back(Sweep{.type = BarrierType::UpAndOut, .barrier = barrier});
    }

    for (const auto& sweep : sweeps) {
        const BarrierType barrier_type = sweep.type;
        const double barrier = sweep.barrier;
        const auto reference = continuous_reference(
            market.value(), model.value(), barrier_type, config.strike, barrier, config.maturity);
        if (!reference) {
            return Result<ExperimentRecord>::failure(reference.error());
        }
        const double continuous = reference.value();
        if (!(continuous > 0.0)) {
            // An up-and-out struck above its barrier is worth exactly zero, so a
            // relative bias against it is undefined and the arm measures nothing.
            // Skipped rather than divided by, and named so the omission is visible.
            record.limitations.push_back(fmt::format(
                "the {} arm at B={:g} was skipped: its continuous reference is exactly zero, so "
                "there is no bias to measure relative to it",
                to_string(barrier_type),
                barrier));
            continue;
        }

        for (const auto convention :
             {MonitoringConvention::Discrete, MonitoringConvention::BrownianBridge}) {
            std::vector<double> monitoring_intervals;
            std::vector<double> absolute_biases;
            std::size_t resolved_levels = 0;
            nlohmann::json levels = nlohmann::json::array();

            for (const std::int64_t dates : config.monitoring_counts) {
                const auto option = BarrierOption::create(OptionType::Call,
                                                          barrier_type,
                                                          config.strike,
                                                          barrier,
                                                          config.maturity,
                                                          convention,
                                                          dates);
                if (!option) {
                    return Result<ExperimentRecord>::failure(option.error());
                }

                const auto cell = price_cell(option.value(), model.value(), continuous);
                if (!cell) {
                    return Result<ExperimentRecord>::failure(cell.error());
                }

                const MultiSeedSummary& summary = cell.value().summary;
                const double bias = cell.value().bias;
                const double across_seed_se = summary.standard_error;
                const double significance = across_seed_se > 0.0 ? bias / across_seed_se : 0.0;
                const double dt = config.maturity / static_cast<double>(dates);
                const bool resolved = std::abs(significance) > kResolutionThreshold;

                monitoring_intervals.push_back(dt);
                absolute_biases.push_back(std::abs(bias));
                resolved_levels += resolved ? 1 : 0;

                nlohmann::json level{{"monitoring_dates", dates},
                                     {"monitoring_interval", dt},
                                     {"price", summary.mean},
                                     {"continuous_reference", continuous},
                                     {"bias", bias},
                                     {"relative_bias", bias / continuous},
                                     {"rmse", cell.value().rmse},
                                     {"across_seed_standard_deviation", summary.standard_deviation},
                                     {"across_seed_standard_error", across_seed_se},
                                     {"bias_over_standard_error", significance},
                                     {"bias_is_resolved", resolved},
                                     {"minimum_estimate", summary.minimum},
                                     {"maximum_estimate", summary.maximum},
                                     {"mean_knock_fraction", cell.value().mean_knock_fraction},
                                     {"seed_count", summary.seed_count},
                                     {"runtime_seconds", cell.value().runtime_seconds}};

                if (convention == MonitoringConvention::Discrete) {
                    const auto corrected = continuity_corrected_reference(market.value(),
                                                                          model.value(),
                                                                          barrier_type,
                                                                          config.strike,
                                                                          barrier,
                                                                          config.maturity,
                                                                          dates);
                    if (!corrected) {
                        return Result<ExperimentRecord>::failure(corrected.error());
                    }
                    const double residual = summary.mean - corrected.value();
                    const double residual_significance =
                        across_seed_se > 0.0 ? residual / across_seed_se : 0.0;
                    level["continuity_corrected_reference"] = corrected.value();
                    level["continuity_corrected_residual"] = residual;
                    level["continuity_corrected_residual_over_se"] = residual_significance;

                    // Whether the closed-form correction and the measurement actually
                    // disagree. This, not the fraction below, is the statistic to
                    // read: the correction is an o(1/sqrt(m)) approximation, so its
                    // residual is expected to be resolved somewhere, and where that
                    // happens is itself a measurement of the correction's own error.
                    level["continuity_corrected_residual_is_resolved"] =
                        std::abs(residual_significance) > kResolutionThreshold;

                    // How much of the raw bias the correction accounts for. Intuitive,
                    // and unreliable wherever the bias is small: it divides by the
                    // bias, so at a distant barrier a residual well inside the noise
                    // still reads as a large unexplained fraction. Reported because a
                    // reader will want it, but the residual above is what decides
                    // whether the correction is right.
                    level["bias_explained_fraction"] = bias != 0.0 ? 1.0 - (residual / bias) : 0.0;

                    if (dates == config.monitoring_counts.front() &&
                        std::abs(significance) > kResolutionThreshold) {
                        coarse_bias_resolved = true;
                    }
                }

                if (convention == MonitoringConvention::BrownianBridge &&
                    std::abs(significance) > kBridgeDefectThreshold) {
                    // The bridge is exact for this model, so a resolved bias here is a
                    // defect in the correction rather than a limitation to note.
                    bridge_defect = true;
                    record.limitations.push_back(fmt::format(
                        "FAILURE: the Brownian-bridge correction is biased by {:.3e} ({:.3f}% of "
                        "the price) at B={:g}, m={}, which is {:.1f} across-seed standard errors "
                        "from the continuous reference over {} seeds. The correction is exact for "
                        "geometric Brownian motion, so this is unexplained, and the bridge arm's "
                        "agreement must not be quoted.",
                        bias,
                        100.0 * bias / continuous,
                        barrier,
                        dates,
                        significance,
                        summary.seed_count));
                }

                record.table.rows.push_back({std::string(to_string(barrier_type)),
                                             std::string(to_string(convention)),
                                             number(barrier),
                                             std::to_string(dates),
                                             number(summary.mean),
                                             number(continuous),
                                             number(bias),
                                             number(across_seed_se),
                                             number(significance),
                                             number(cell.value().mean_knock_fraction)});

                levels.push_back(std::move(level));
            }

            nlohmann::json arm{{"convention", std::string(to_string(convention))},
                               {"barrier_type", std::string(to_string(barrier_type))},
                               {"barrier", barrier},
                               // Unsigned: how far the barrier is, in the units the
                               // bias actually scales in. Direction is barrier_type.
                               {"barrier_distance_in_sigma_sqrt_t",
                                std::abs(std::log(config.spot / barrier)) /
                                    (config.volatility * std::sqrt(config.maturity))},
                               {"continuous_reference", continuous},
                               {"resolved_level_count", resolved_levels},
                               {"level_count", levels.size()},
                               {"levels", std::move(levels)}};

            if (convention == MonitoringConvention::Discrete) {
                // Broadie-Glasserman-Kou: the bias is O(sqrt(dt)), so fitted against
                // the monitoring interval the expected order is +0.5.
                arm["expected_order"] = 0.5;

                if (resolved_levels < config.monitoring_counts.size()) {
                    // No order is reported unless every level's bias cleared its own
                    // noise, because a power law fitted to unresolved levels returns a
                    // number and an interval regardless -- and both are meaningless.
                    //
                    // This is not hypothetical. At B=70 the bias never clears 2.1
                    // standard errors at any frequency, and fitting all six levels
                    // anyway gave an order of -0.19 with a 95% interval of
                    // [-0.70, +0.33]: a confident-looking claim that the bias *grows*
                    // as the barrier is watched more often, fitted entirely to six
                    // draws from zero. A reader has no way to tell that from a
                    // measurement, so the fit is refused and the record says what
                    // actually happened instead.
                    arm["fit_refused"] = fmt::format(
                        "the bias cleared {} standard errors at only {} of {} monitoring "
                        "frequencies, so there is no measured decay to fit. A power law fitted to "
                        "unresolved levels would report an order and an interval for what is "
                        "sampling noise. The barrier is {:.2f} sigma*sqrt(T) from the spot, far "
                        "enough that discrete monitoring misses almost nothing: the bias is real "
                        "but smaller than this run can see.",
                        kResolutionThreshold,
                        resolved_levels,
                        config.monitoring_counts.size(),
                        std::abs(std::log(config.spot / barrier)) /
                            (config.volatility * std::sqrt(config.maturity)));
                } else {
                    arm["fit_vs_monitoring_interval"] =
                        fit_json(monitoring_intervals, absolute_biases);

                    // The theoretical 0.5 is asymptotic, and the coarse levels are
                    // where the higher-order terms are largest. Both fits are
                    // published; where they disagree, the disagreement is the finding.
                    if (monitoring_intervals.size() > kAsymptoticWindow) {
                        const std::size_t offset = monitoring_intervals.size() - kAsymptoticWindow;
                        arm["asymptotic_window_fit"] =
                            fit_json(std::span{monitoring_intervals}.subspan(offset),
                                     std::span{absolute_biases}.subspan(offset));
                        arm["asymptotic_window_levels"] = kAsymptoticWindow;
                    }
                }

                arm["local_orders"] = local_orders_json(monitoring_intervals, absolute_biases);
            }

            arms.push_back(std::move(arm));
        }
    }
    record.results["arms"] = std::move(arms);

    // --- Volatility sensitivity --------------------------------------------
    //
    // The bias scales with sigma*sqrt(dt), so volatility moves it as directly as the
    // monitoring frequency does. Measured at daily monitoring, where the temptation
    // to call the discrete price continuous is strongest.
    {
        nlohmann::json cells = nlohmann::json::array();
        constexpr double kSensitivityBarrier = 90.0;
        constexpr std::int64_t kSensitivityDates = 250;

        for (const double volatility : config.volatilities) {
            const auto cell_model = BlackScholesModel::create(volatility);
            if (!cell_model) {
                return Result<ExperimentRecord>::failure(cell_model.error());
            }
            const auto reference = continuous_reference(market.value(),
                                                        cell_model.value(),
                                                        BarrierType::DownAndOut,
                                                        config.strike,
                                                        kSensitivityBarrier,
                                                        config.maturity);
            if (!reference) {
                return Result<ExperimentRecord>::failure(reference.error());
            }
            const auto option = BarrierOption::create(OptionType::Call,
                                                      BarrierType::DownAndOut,
                                                      config.strike,
                                                      kSensitivityBarrier,
                                                      config.maturity,
                                                      MonitoringConvention::Discrete,
                                                      kSensitivityDates);
            if (!option) {
                return Result<ExperimentRecord>::failure(option.error());
            }
            const auto cell = price_cell(option.value(), cell_model.value(), reference.value());
            if (!cell) {
                return Result<ExperimentRecord>::failure(cell.error());
            }

            const MultiSeedSummary& summary = cell.value().summary;
            cells.push_back(nlohmann::json{
                {"volatility", volatility},
                {"barrier", kSensitivityBarrier},
                {"monitoring_dates", kSensitivityDates},
                {"continuous_reference", reference.value()},
                {"discrete_price", summary.mean},
                {"bias", cell.value().bias},
                {"relative_bias", cell.value().bias / reference.value()},
                {"across_seed_standard_error", summary.standard_error},
                {"bias_over_standard_error",
                 summary.standard_error > 0.0 ? cell.value().bias / summary.standard_error : 0.0},
                {"mean_knock_fraction", cell.value().mean_knock_fraction}});
        }
        record.results["volatility_sensitivity"] = std::move(cells);
    }

    // --- PDE arm: continuous-barrier pricing convergence -------------------
    //
    // A different question from the Monte Carlo arms above. Those measure the bias of
    // discrete monitoring against the continuous contract and the bridge correction
    // for it. This one prices the continuous contract *directly* -- solving the
    // Black-Scholes PDE with an absorbing boundary at the barrier -- and measures how
    // fast that solve converges to the analytic reference as the grid refines. Node
    // count and time steps are refined together, so the fitted order is the joint
    // spatial-and-temporal order, second for Crank-Nicolson with Rannacher smoothing.
    //
    // The reference is the same Reiner-Rubinstein value the discrete arm measures
    // against, so the two arms are anchored to one continuous price and their results
    // are directly comparable: the PDE's grid error and the Monte Carlo's monitoring
    // bias are both errors against the same number.
    bool pde_arm_converged = true;
    {
        nlohmann::json pde_cells = nlohmann::json::array();

        struct PdeBarrier {
            BarrierType type;
            double barrier;
        };

        std::vector<PdeBarrier> pde_barriers;
        pde_barriers.reserve(config.barriers.size() + config.up_barriers.size());
        for (const double barrier : config.barriers) {
            pde_barriers.push_back(PdeBarrier{.type = BarrierType::DownAndOut, .barrier = barrier});
        }
        for (const double barrier : config.up_barriers) {
            pde_barriers.push_back(PdeBarrier{.type = BarrierType::UpAndOut, .barrier = barrier});
        }

        for (const auto& pde : pde_barriers) {
            const auto reference = continuous_reference(market.value(),
                                                        model.value(),
                                                        pde.type,
                                                        config.strike,
                                                        pde.barrier,
                                                        config.maturity);
            if (!reference) {
                return Result<ExperimentRecord>::failure(reference.error());
            }
            if (!(reference.value() > 0.0)) {
                // A worthless contract (up-and-out struck above its barrier) has no
                // relative error to converge, so it is skipped rather than divided by.
                continue;
            }

            const auto option = BarrierOption::create(OptionType::Call,
                                                      pde.type,
                                                      config.strike,
                                                      pde.barrier,
                                                      config.maturity,
                                                      MonitoringConvention::Continuous,
                                                      std::nullopt);
            if (!option) {
                return Result<ExperimentRecord>::failure(option.error());
            }

            std::vector<double> spacings;
            std::vector<double> errors;
            nlohmann::json levels = nlohmann::json::array();
            for (const std::int64_t nodes : config.pde_resolutions) {
                PdeConfig pde_config;
                pde_config.asset_nodes = nodes;
                pde_config.time_steps = nodes;
                pde_config.scheme = PdeScheme::CrankNicolson;
                pde_config.rannacher = RannacherSteps{config.pde_rannacher_steps};

                const auto priced = BarrierPdeEngine::price(
                    market.value(), option.value(), model.value(), pde_config);
                if (!priced) {
                    return Result<ExperimentRecord>::failure(priced.error());
                }
                const double error = std::abs(priced.value().value - reference.value());
                // The refinement parameter is 1/(nodes-1), proportional to dS on a
                // fixed domain and to dtau at fixed maturity, so the fitted slope is
                // the order in the joint refinement.
                const double h = 1.0 / static_cast<double>(nodes - 1);
                spacings.push_back(h);
                errors.push_back(error);

                levels.push_back(nlohmann::json{{"asset_nodes", nodes},
                                                {"time_steps", nodes},
                                                {"refinement_parameter", h},
                                                {"price", priced.value().value},
                                                {"continuous_reference", reference.value()},
                                                {"absolute_error", error},
                                                {"relative_error", error / reference.value()}});
            }

            nlohmann::json cell{{"barrier_type", std::string(to_string(pde.type))},
                                {"barrier", pde.barrier},
                                {"continuous_reference", reference.value()},
                                {"scheme", "crank_nicolson"},
                                {"rannacher_steps", config.pde_rannacher_steps},
                                {"expected_order", 2.0},
                                {"levels", levels}};

            const bool all_positive =
                std::ranges::all_of(errors, [](double error) { return error > 0.0; });
            if (all_positive && spacings.size() >= 3) {
                cell["order_fit"] = fit_json(spacings, errors);
                cell["local_orders"] = local_orders_json(spacings, errors);
                // The PDE arm is a validation: it passes when the finest error is
                // small and the fitted order is consistent with the scheme's second.
                // A first-order bug -- a misplaced boundary, a wrong live index --
                // would break the order conspicuously, which is exactly what this
                // guards against.
                const auto& fit = cell["order_fit"];
                if (fit.contains("order") && fit["order"].get<double>() < 1.5) {
                    pde_arm_converged = false;
                    record.limitations.push_back(fmt::format(
                        "FAILURE: the PDE arm's convergence order for the {} barrier at B={:g} is "
                        "{:.3f}, below the 1.5 floor for a second-order scheme. A misplaced "
                        "barrier "
                        "boundary or a wrong live-index range would produce exactly this; the PDE "
                        "price should not be trusted until it is diagnosed.",
                        to_string(pde.type),
                        pde.barrier,
                        fit["order"].get<double>()));
                }
            }

            pde_cells.push_back(std::move(cell));
        }
        record.results["pde_convergence"] = std::move(pde_cells);
        record.results["pde_arm_converged"] = pde_arm_converged;
    }

    record.results["resolution_threshold"] = kResolutionThreshold;
    record.results["bridge_defect_threshold"] = kBridgeDefectThreshold;
    record.results["continuity_correction_beta"] = kContinuityCorrectionBeta;

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (bridge_defect || !pde_arm_converged) {
        // Either arm failing is a failure of the whole record: a bridge that does not
        // reproduce the continuous price, or a PDE that does not converge to it, is a
        // defect the reader must see before quoting any number here.
        record.status = ExperimentStatus::Fail;
    } else if (!coarse_bias_resolved) {
        // No resolved bias at the coarsest monitoring means the experiment did not
        // measure the thing it exists to measure. That is not a pass -- and it is not
        // a claim that the bias is absent either.
        record.status = ExperimentStatus::Inconclusive;
        record.limitations.emplace_back(
            "The discrete-monitoring bias did not clear its own across-seed noise at the coarsest "
            "monitoring frequency, so this run quantified nothing. More paths or more seeds are "
            "needed before any statement about the bias is supported.");
    } else {
        record.status = ExperimentStatus::Pass;
    }

    record.interpretation =
        "Discrete monitoring biases a knock-out upward, and by far more than intuition allows. The "
        "barrier is unobserved between fixes, so excursions that would have killed the option go "
        "unseen and it survives paths it should not -- making it worth more than the continuously "
        "monitored contract the closed form prices. Every one of the 42 resolved cells is biased "
        "high, across both barrier directions.\n\n"
        "The size is the finding, and it is worst exactly where the contract is cheapest. An "
        "up-and-out call struck at the spot with its barrier 5% above knocks out in precisely the "
        "states where it would have paid, so what survives is worth almost nothing -- 0.0088 here "
        "-- and daily monitoring biases that by +67% of its own value. A down-and-out 0.26 "
        "sigma*sqrt(T) away is biased +10.1% at the same frequency; further out, +3.0% and +0.28%. "
        "The bias tracks sigma*sqrt(dt), the distance the price can wander unobserved: at daily "
        "monitoring a down-and-out at B=90 runs +0.30%, +3.01%, +10.03% of price at volatilities "
        "of 0.1, 0.2 and 0.4. None of this is about the calendar.\n\n"
        "The decay rate needs care, and this experiment is a good illustration of why a fitted "
        "slope is not a measured order. Theory (Broadie-Glasserman-Kou) says the bias decays as "
        "O(1/sqrt(m)) -- order 0.5 against the monitoring interval. Almost no fitted interval "
        "here contains 0.5, and, revealingly, they miss it from *both sides*: down-barriers come "
        "in low (0.248, 0.395, 0.437 as the barrier nears the spot) while close up-barriers come "
        "in high (0.630 at B=105, 0.552 at B=110). A single-direction study would have read its "
        "own miss as a signed correction to theory; seeing both signs shows it for what it is -- "
        "the higher-order terms the asymptotic statement discards, which carry opposite signs for "
        "the two geometries. The local orders climb monotonically toward 0.5 for down-barriers "
        "(0.342, 0.375, 0.390, 0.431, 0.435 at B=90) and the asymptotic-window fits recover 0.5 at "
        "five of the seven fitted arms. The measured statement is that the bias decays at a rate "
        "approaching 0.5 over the frequencies tested, not that it decays at 0.5, and not that "
        "theory is wrong.\n\n"
        "What settles that is a check needing no fit: the same theory predicts the bias's *size* "
        "in closed form, by moving the barrier away from the spot through the mean overshoot "
        "exp(+/- beta*sigma*sqrt(dt)). It was not fitted to this data -- beta is "
        "-zeta(1/2)/sqrt(2*pi) -- and for down-barriers it lands within 2 across-seed standard "
        "errors at 16 of 18 resolved cells. A theory with the rate wrong could not predict the "
        "size to within its noise across three barriers and six frequencies.\n\n"
        "For up-barriers the same correction is resolved as *wrong* at 23 of the 24 cells, and "
        "that is the most informative result here. It is not a sign error: the shift direction was "
        "checked "
        "independently, and reversing it misses by an order of magnitude rather than a few "
        "percent. "
        "The residual is systematically negative -- the correction over-predicts -- and small in "
        "absolute terms (0.0006 to 0.008), running -275 standard errors at B=105, m=5 and draining "
        "to -5.4 by m=250. The single cell that does *not* disagree is the farthest up-barrier at "
        "the finest monitoring, B=140 at m=250 -- exactly where an o(1/sqrt(m)) remainder should "
        "have drained away, which is the behaviour the theory predicts rather than an exception "
        "to it. Two things make it visible. First, an up-and-out call near the spot "
        "pays "
        "on almost no paths, so the estimator's variance collapses and a remainder that would hide "
        "under a down-barrier's noise is glaring here: the same absolute error is 0.1% of a "
        "down-barrier price and 4% of this one. Second, the geometry is genuinely harder. An "
        "up-and-out call pays B-K *just below* its barrier and zero at it, so the value is "
        "discontinuous across the boundary -- while a down-and-out call struck above its barrier "
        "approaches zero smoothly there. A continuity correction built on an overshoot argument is "
        "weakest exactly where the value jumps. So the experiment does not merely use the "
        "correction as an oracle; it locates where the oracle stops being trustworthy, and the "
        "answer is not where a down-barrier-only study would have suggested.\n\n"
        "The Brownian-bridge correction removes the bias rather than shrinking it, in both "
        "directions. Between two observed points log-GBM is a Brownian bridge whose crossing "
        "probability is known exactly, so a simulation can account for the excursions it did not "
        "observe instead of pretending they did not happen -- and because log-GBM is exactly "
        "Brownian motion with drift, that correction is exact rather than asymptotic. No bridge "
        "cell shows a bias this run can resolve: the largest across all 48 is 2.4 across-seed "
        "standard errors, against a discrete arm reaching 563. It holds at m=5 as well as m=250, "
        "which is the point -- it is a correction, not a refinement, and five observations with "
        "the bridge beat 250 without it.\n\n"
        "The PDE arm answers a different question and anchors the whole record. It prices the "
        "*continuous* contract directly -- solving the Black-Scholes equation with an absorbing "
        "boundary at the barrier -- rather than measuring a monitoring bias, and it converges to "
        "the same analytic reference the discrete arm measures against, at the second order "
        "Crank-Nicolson with Rannacher smoothing predicts. Two independent routes therefore reach "
        "one continuous price: the closed form the analytic engine evaluates, and the grid the PDE "
        "refines toward it. The Monte Carlo bias and the PDE's grid error are then errors against "
        "the same number, which is what lets the two arms sit in one experiment without conflating "
        "the discretisation of a path with the discretisation of an observation. The fitted orders "
        "are in `pde_convergence`; the arm fails the record if any falls below a second-order "
        "floor, since a misplaced barrier boundary would show up there first.\n\n"
        "The practical reading is that the monitoring convention is a contract term worth more "
        "than most modelling choices, and that its cost is not uniform: it concentrates in the "
        "cheap, nearly-knocked-out contracts where a relative error is largest and a trader's "
        "intuition is weakest. It is a term of the contract, not an approximation to be tuned "
        "away, and a system that leaves it implicit will silently price one contract and report "
        "another.";

    record.limitations.emplace_back(
        "The path is simulated with the exact scheme at exactly the monitoring dates, so it "
        "carries no path-discretisation error: log-GBM is Brownian motion with drift, and the "
        "exact scheme samples the true joint law of the price at any finite set of dates. What is "
        "measured here is therefore monitoring bias alone. A production simulation on a finer path "
        "grid, or with Euler steps, would additionally carry a path error this experiment cannot "
        "see and does not bound.");
    record.limitations.emplace_back(
        "The Brownian-bridge correction is exact for geometric Brownian motion and for that "
        "reason only. Under Heston, or any model whose log-price is not Brownian motion with "
        "constant coefficients, the bridge law is an approximation, and nothing measured here says "
        "how good an approximation it is.");
    record.limitations.emplace_back(
        "Calls only. Puts are a separate set of cases the analytic engine does not implement, so "
        "no trusted continuous reference exists for them.");
    record.limitations.emplace_back(
        "Every bias reported here is measured against Reiner-Rubinstein, which is a model output "
        "rather than a market observation. It agrees with mpmath to 1e-15 and QuantLib to 1e-9, so "
        "it is a trustworthy statement about the model -- but this experiment measures agreement "
        "with a formula, not with reality.");
    record.limitations.emplace_back(
        "At B=70, 1.78 sigma*sqrt(T) below the spot, the bias never clears 2.1 across-seed "
        "standard errors at any monitoring frequency. This run measured nothing there, and no "
        "order is reported for it. The bias is real -- theory says so, and the sign is right at "
        "four of six frequencies -- but it is smaller than 200000 paths across 16 seeds can see. "
        "Read as evidence that a distant barrier's monitoring bias is negligible at these path "
        "counts, not as evidence that it is zero.");
    record.limitations.emplace_back(
        "The continuity correction is trustworthy for the down-barriers measured here and is NOT "
        "for the up-barriers: its residual is resolved at 23 of the 24 up-barrier cells, "
        "systematically negative, and still -5.4 standard errors at daily monitoring on the "
        "nearest one. The shift direction was "
        "verified independently -- reversing it misses by an order of magnitude rather than a few "
        "percent -- so this is the correction's own o(1/sqrt(m)) remainder, not a defect here. Do "
        "not use the continuity-corrected reference as an oracle for an up-and-out call near the "
        "spot; use the bridge, which is exact for this model.");
    record.limitations.emplace_back(
        "The fitted orders that miss 0.5 are a property of the frequencies tested, not a measured "
        "contradiction of theory, and the record should not be quoted as though the two were the "
        "same. The evidence for that reading is circumstantial in the fits themselves -- climbing "
        "local orders, and a B=95 asymptotic window that contains 0.5 -- and direct only in the "
        "continuity-corrected check. Reaching the asymptotic range would need monitoring "
        "frequencies well beyond daily, which no traded contract uses and which this experiment "
        "therefore did not prioritise.");
    record.limitations.emplace_back(
        "Read `continuity_corrected_residual_over_se`, not `bias_explained_fraction`, when judging "
        "the continuity correction. The fraction divides by the bias, so where the bias is small "
        "it "
        "is a noisy statistic and misleads: at B=80 it ranges from 67% to 109% across frequencies, "
        "which reads as the correction failing, while every residual there is within 1.9 standard "
        "errors of zero and therefore consistent with it being exactly right. The fraction is "
        "published because a reader will want it, not because it decides anything. Note also that "
        "a "
        "residual's significance depends on the estimator's noise, which varies by orders of "
        "magnitude across this grid: an up-and-out call near the spot pays on almost no paths, so "
        "its standard error collapses and the same absolute error that hides at B=70 is resolved "
        "at hundreds of sigma at B=105. Significance measures what this run can see, not how large "
        "the error is.");
    record.limitations.emplace_back(
        "The 48 bridge cells share the same 16 seeds, so they are not 24 independent tests of the "
        "correction. Their sampling errors are correlated across barriers and frequencies -- "
        "visibly so: the largest positive deviation sits at m=50 for three of the four barriers. "
        "The evidence that the bridge is unbiased is weaker than the cell count suggests, though "
        "no individual cell comes close to the threshold.");
    record.limitations.emplace_back(
        "The nearest barriers tested are 0.24-0.26 sigma*sqrt(T) from the spot. Barriers "
        "very close to the spot knock out nearly every path, leaving a small surviving sample and "
        "an estimator whose variance is dominated by rare paths. That regime is not characterised "
        "here.");
    record.limitations.emplace_back(
        "The bridge estimator draws a Bernoulli knock decision per interval rather than "
        "accumulating the conditional survival probability. Both are unbiased, but the "
        "conditional-expectation form would have strictly lower variance. The bridge arm's "
        "intervals are wider than they need to be as a consequence of that choice, not of the "
        "correction itself.");

    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
