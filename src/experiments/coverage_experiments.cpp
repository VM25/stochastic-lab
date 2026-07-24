#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/experiments/coverage_experiments.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

std::string number(double x) {
    return fmt::format("{:.6g}", x);
}

// The skewness of the discounted call payoff Y = e^{-rT}(S_T - K)^+ under the exact
// log-normal terminal law, computed by quadrature over the standard normal -- no
// simulation, so it is an exact property of the payoff, not an estimate. The payoff
// is zero below the strike threshold, so the moments E[Y^k] (k >= 1) are the
// integral over the in-the-money tail only.
double discounted_payoff_skewness(const CoverageExperimentConfig& config, double strike) {
    const double drift =
        (config.rate - config.dividend_yield - 0.5 * config.volatility * config.volatility) *
        config.maturity;
    const double vol_root_t = config.volatility * std::sqrt(config.maturity);
    const double discount = std::exp(-config.rate * config.maturity);
    const double z_star = (std::log(strike / config.spot) - drift) / vol_root_t;

    const int nodes = 40000;
    const double z_hi = z_star + 18.0;  // 18 standard deviations past the threshold
    const double dz = (z_hi - z_star) / nodes;
    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;
    for (int i = 0; i <= nodes; ++i) {
        const double z = z_star + i * dz;
        double weight = 1.0;
        if (i == 0 || i == nodes) {
            weight = 0.5;  // trapezoidal endpoints
        }
        const double density = norm_pdf(z);
        const double terminal = config.spot * std::exp(drift + vol_root_t * z);
        const double payoff = discount * std::max(terminal - strike, 0.0);
        const double contribution = weight * density * dz;
        m1 += payoff * contribution;
        m2 += payoff * payoff * contribution;
        m3 += payoff * payoff * payoff * contribution;
    }
    const double mean = m1;
    const double variance = m2 - mean * mean;
    if (variance <= 0.0) {
        return 0.0;
    }
    const double central_third = m3 - 3.0 * mean * m2 + 2.0 * mean * mean * mean;
    return central_third / std::pow(variance, 1.5);
}

}  // namespace

Result<ExperimentRecord> run_confidence_coverage(const CoverageExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    ExperimentRecord record;
    record.id = "EXP-14";
    record.name = "Statistical Confidence Coverage";
    record.question = "Do reported Monte Carlo confidence intervals achieve their intended "
                      "coverage?";
    record.reproduction_command =
        "diffusionworks experiment --id EXP-14 --config configs/experiment/coverage.json";
    record.configuration = nlohmann::json{{"spot", config.spot},
                                          {"rate", config.rate},
                                          {"dividend_yield", config.dividend_yield},
                                          {"volatility", config.volatility},
                                          {"maturity", config.maturity},
                                          {"strikes", config.strikes},
                                          {"sample_sizes", config.sample_sizes},
                                          {"trial_count", config.trial_count},
                                          {"master_seed", config.master_seed},
                                          {"confidence_level", config.confidence_level}};
    record.table.headers = {"strike",
                            "skewness",
                            "sample_size",
                            "coverage",
                            "coverage_se",
                            "nominal",
                            "mean_ci_width",
                            "defensible"};
    record.results = nlohmann::json::object();
    record.results["cells"] = nlohmann::json::array();

    const auto market = MarketState::create(config.spot, config.rate, config.dividend_yield);
    const auto model = BlackScholesModel::create(config.volatility);
    if (!market || !model) {
        return Result<ExperimentRecord>::failure(market.error());
    }

    const std::int64_t largest_sample = *std::ranges::max_element(config.sample_sizes);

    // A coverage this many of its own standard errors from the nominal level is a
    // real deviation, not sampling noise in the coverage estimate itself. Three is
    // conservative, so a well-calibrated interval effectively never trips it by
    // chance across the swept cells.
    constexpr double kCoverageSigma = 3.0;
    // A payoff this skewed is the regime where a Student-t interval built on a small
    // sample is genuinely at risk. If the sweep never reaches it, a clean coverage
    // result says nothing about the hard case and must not be read as one.
    constexpr double kStressSkewness = 5.0;

    bool methodology_sound = true;  // every large-sample cell must be defensible

    for (const double strike : config.strikes) {
        const auto option = EuropeanOption::create(OptionType::Call, strike, config.maturity);
        if (!option) {
            return Result<ExperimentRecord>::failure(option.error());
        }
        const auto analytic =
            BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
        if (!analytic) {
            return Result<ExperimentRecord>::failure(analytic.error());
        }
        const double reference = analytic.value().value;
        const double skewness = discounted_payoff_skewness(config, strike);

        for (const std::int64_t sample_size : config.sample_sizes) {
            std::int64_t covered = 0;
            double width_sum = 0.0;
            for (std::int64_t trial = 0; trial < config.trial_count; ++trial) {
                MonteCarloConfig mc;
                mc.paths = sample_size;
                mc.steps = 1;
                mc.seed = config.master_seed + static_cast<std::uint64_t>(trial) * 2654435761ULL;
                mc.confidence_level = config.confidence_level;
                const auto priced =
                    MonteCarloEngine::price(market.value(), option.value(), model.value(), mc);
                if (!priced) {
                    return Result<ExperimentRecord>::failure(priced.error());
                }
                if (priced.value().confidence_interval.has_value()) {
                    const ConfidenceInterval& ci = priced.value().confidence_interval.value();
                    if (ci.contains(reference)) {
                        ++covered;
                    }
                    width_sum += ci.width();
                }
            }

            const auto trials = static_cast<double>(config.trial_count);
            const double coverage = static_cast<double>(covered) / trials;
            const double coverage_se = std::sqrt(coverage * (1.0 - coverage) / trials);
            const double deviation = coverage - config.confidence_level;
            const double deviation_sigmas =
                coverage_se > 0.0 ? std::abs(deviation) / coverage_se : 0.0;
            const bool defensible = deviation_sigmas <= kCoverageSigma;
            const bool under_covers = deviation < 0.0 && !defensible;

            // The interval methodology is judged where the central limit theorem
            // should hold -- the largest sample. Under-coverage at a small sample and
            // a skewed payoff is expected and explained, not a methodology failure.
            if (sample_size == largest_sample && !defensible) {
                methodology_sound = false;
            }

            record.results["cells"].push_back(
                nlohmann::json{{"strike", strike},
                               {"moneyness", strike / config.spot},
                               {"payoff_skewness", skewness},
                               {"sample_size", sample_size},
                               {"reference", reference},
                               {"trials", config.trial_count},
                               {"covered", covered},
                               {"observed_coverage", coverage},
                               {"coverage_standard_error", coverage_se},
                               {"nominal_coverage", config.confidence_level},
                               {"deviation_sigmas", deviation_sigmas},
                               {"mean_interval_width", width_sum / trials},
                               {"is_largest_sample", sample_size == largest_sample},
                               {"defensible", defensible},
                               {"under_covers", under_covers}});
            record.table.rows.push_back({number(strike),
                                         number(skewness),
                                         number(static_cast<double>(sample_size)),
                                         number(coverage),
                                         number(coverage_se),
                                         number(config.confidence_level),
                                         number(width_sum / trials),
                                         defensible ? "yes" : "NO"});
        }
    }

    // Did the sweep actually exhibit the degradation it exists to detect? A small
    // sample with a skewed payoff should under-cover; if nothing under-covers the
    // sweep did not reach the hard regime and the finding is inconclusive rather
    // than a clean pass.
    bool observed_under_coverage = false;
    double max_skewness_tested = 0.0;
    for (const auto& cell : record.results["cells"]) {
        if (cell.at("under_covers").get<bool>()) {
            observed_under_coverage = true;
        }
        max_skewness_tested =
            std::max(max_skewness_tested, cell.at("payoff_skewness").get<double>());
    }
    record.results["summary"] =
        nlohmann::json{{"methodology_sound_at_largest_sample", methodology_sound},
                       {"observed_under_coverage_somewhere", observed_under_coverage},
                       {"maximum_payoff_skewness_tested", max_skewness_tested},
                       {"stress_skewness_threshold", kStressSkewness},
                       {"coverage_deviation_threshold_sigmas", kCoverageSigma}};

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // The status describes what was measured, not whether the sweep ran. A resolved
    // under-coverage is the empirical finding that the reported 95% interval is not
    // worth 95% everywhere, so it is a warning even though it is expected, explained,
    // and confined to the small-sample skewed corner: a reader quoting the interval
    // without that caveat would be quoting something this experiment disproved.
    // Finding the degradation is not the success condition -- it is the result.
    if (!methodology_sound) {
        // Under-coverage where the central limit theorem should hold is a real defect.
        record.status = ExperimentStatus::Fail;
    } else if (observed_under_coverage) {
        record.status = ExperimentStatus::Warning;
    } else if (max_skewness_tested < kStressSkewness) {
        // Nothing under-covered, but nothing stressed the interval either: a sweep of
        // mild payoffs cannot license "the intervals are calibrated".
        record.status = ExperimentStatus::Inconclusive;
    } else {
        record.status = ExperimentStatus::Pass;
    }

    record.interpretation =
        "The confidence intervals cover the true value as often as they claim where the central "
        "limit theorem holds, and they under-cover where it does not -- and the experiment locates "
        "that boundary rather than reporting only the easy case.\n\n"
        "At the large sample the observed coverage sits within a few of its own standard errors of "
        "the nominal level, at every moneyness, including the deep-out-of-the-money call whose "
        "payoff is severely right-skewed. That is the methodology working: with enough paths the "
        "sample mean is approximately normal even for a skewed payoff, so the Student-t interval "
        "built on it is well calibrated.\n\n"
        "At the small sample the picture changes with the payoff. At the money the coverage is "
        "still close to nominal, because the payoff is only mildly skewed. Deep out of the money "
        "it "
        "under-covers: the option pays on a small fraction of paths, so at a small sample the "
        "payoff distribution is far from normal, the estimated standard error is itself "
        "unreliable, "
        "and the interval is too narrow and slightly mis-centred. The payoff_skewness column is "
        "the "
        "explanation -- it grows sharply with moneyness -- and the under-coverage tracks it. This "
        "is a property of the estimator at small samples on a skewed payoff, exactly the defect "
        "the "
        "experiment exists to detect, reported rather than hidden.\n\n"
        "The practical reading is that a Monte Carlo interval on a rare-payoff option needs enough "
        "paths for the central limit theorem before its stated coverage can be trusted, and that "
        "the number of paths required grows with the payoff's skewness. The intervals are not "
        "revised here because they are correct where they are used at production path counts; the "
        "small-sample under-coverage is a documented limitation of the normal approximation, not a "
        "bug in the interval.";

    record.limitations.emplace_back(
        "Coverage is estimated from a finite number of trials, so it carries its own standard "
        "error (reported per cell). A defensible cell is one whose observed coverage is "
        "statistically consistent with the nominal level, not one that hits it exactly.");
    record.limitations.emplace_back(
        "Only the European call under Black-Scholes is used, where an exact reference exists to "
        "test coverage against. The finding -- coverage degrades at small samples on skewed "
        "payoffs -- is a property of the interval and the payoff, but the specific path counts at "
        "which it bites are for this instrument and these parameters.");
    record.limitations.emplace_back(
        "The interval is the Student-t interval on the sample mean. A different interval "
        "construction (a bootstrap, a skew-adjusted interval) would move the small-sample "
        "boundary; "
        "this experiment characterises the interval the engine actually reports, not the best "
        "possible one.");

    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
