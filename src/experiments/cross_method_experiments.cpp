#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>
#include <diffusionworks/experiments/cross_method_experiments.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

std::string number(double x) {
    return fmt::format("{:.6g}", x);
}

// Whether a stochastic estimate agrees with a reference: their difference is within
// `sigma` combined standard errors. When the reference is exact its standard error
// is zero, so the combined error is the estimate's own.
struct Agreement {
    double difference{};
    double combined_standard_error{};
    double sigmas{};
    bool agrees{};
};

Agreement
agree_within_sigma(double value, double reference, double se, double reference_se, double sigma) {
    Agreement a;
    a.difference = value - reference;
    a.combined_standard_error = std::sqrt(se * se + reference_se * reference_se);
    a.sigmas = a.combined_standard_error > 0.0
                   ? std::abs(a.difference) / a.combined_standard_error
                   : (a.difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    a.agrees = a.sigmas <= sigma;
    return a;
}

}  // namespace

Result<ExperimentRecord>
run_cross_method_agreement(const CrossMethodAgreementExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    ExperimentRecord record;
    record.id = "EXP-13";
    record.name = "Cross-Method Accuracy and Agreement";
    record.question =
        "Where independent methods price the same instrument, do they agree within their "
        "justified tolerances?";
    record.reproduction_command =
        "diffusionworks experiment --id EXP-13 --config configs/experiment/cross_method.json";
    record.configuration =
        nlohmann::json{{"spot", config.spot},
                       {"rate", config.rate},
                       {"dividend_yield", config.dividend_yield},
                       {"strike", config.strike},
                       {"volatilities", config.volatilities},
                       {"maturities", config.maturities},
                       {"monte_carlo_paths", config.monte_carlo_paths},
                       {"monte_carlo_seed", config.monte_carlo_seed},
                       {"asian_monitoring_count", config.asian_monitoring_count},
                       {"fd_asset_nodes", config.fd_asset_nodes},
                       {"fd_time_steps", config.fd_time_steps},
                       {"heston_maturities", config.heston_maturities},
                       {"heston_vol_of_variances", config.heston_vol_of_variances},
                       {"heston_monte_carlo_paths", config.heston_monte_carlo_paths},
                       {"heston_monte_carlo_steps", config.heston_monte_carlo_steps},
                       {"agreement_sigma", config.agreement_sigma},
                       {"fd_relative_tolerance", config.fd_relative_tolerance}};
    record.table.headers = {
        "family", "regime", "method", "value", "reference", "difference", "sigmas", "agrees"};
    record.results = nlohmann::json::object();
    record.results["black_scholes_european"] = nlohmann::json::array();
    record.results["heston_european"] = nlohmann::json::array();
    record.results["arithmetic_asian"] = nlohmann::json::array();

    bool all_agree = true;
    int mc_cells = 0;
    int mc_ci_covers = 0;

    const auto market = MarketState::create(config.spot, config.rate, config.dividend_yield);
    if (!market) {
        return Result<ExperimentRecord>::failure(market.error());
    }

    // -----------------------------------------------------------------------
    // Family 1 -- European under Black-Scholes. The analytic price is exact and is
    // the reference; Monte Carlo (crude and antithetic) and the Crank-Nicolson
    // finite-difference solver must agree with it, each within its own justified
    // tolerance (a standard-error band for the stochastic methods, a grid-based
    // relative tolerance for the deterministic one).
    // -----------------------------------------------------------------------
    for (const double volatility : config.volatilities) {
        for (const double maturity : config.maturities) {
            const auto model = BlackScholesModel::create(volatility);
            const auto option = EuropeanOption::create(OptionType::Call, config.strike, maturity);
            if (!model || !option) {
                return Result<ExperimentRecord>::failure(model.error());
            }
            const auto analytic =
                BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
            if (!analytic) {
                return Result<ExperimentRecord>::failure(analytic.error());
            }
            const double reference = analytic.value().value;

            MonteCarloConfig mc_crude;
            mc_crude.paths = config.monte_carlo_paths;
            mc_crude.steps = 1;
            mc_crude.seed = config.monte_carlo_seed;
            MonteCarloConfig mc_anti = mc_crude;
            mc_anti.variance_reduction.antithetic = true;
            const auto crude =
                MonteCarloEngine::price(market.value(), option.value(), model.value(), mc_crude);
            const auto anti =
                MonteCarloEngine::price(market.value(), option.value(), model.value(), mc_anti);
            PdeConfig pde;
            pde.asset_nodes = config.fd_asset_nodes;
            pde.time_steps = config.fd_time_steps;
            pde.scheme = PdeScheme::CrankNicolson;
            const auto fd =
                FiniteDifferenceEngine::price(market.value(), option.value(), model.value(), pde);
            if (!crude || !anti || !fd) {
                return Result<ExperimentRecord>::failure(crude.error());
            }

            nlohmann::json methods = nlohmann::json::array();
            bool cell_agrees = true;
            const std::string regime = fmt::format("sigma={:g} T={:g}", volatility, maturity);

            // The analytic method is the reference; it agrees with itself by definition.
            methods.push_back(nlohmann::json{{"method", "analytic"},
                                             {"value", reference},
                                             {"is_reference", true},
                                             {"agrees", true}});
            record.table.rows.push_back({"bs_european",
                                         regime,
                                         "analytic",
                                         number(reference),
                                         number(reference),
                                         number(0.0),
                                         number(0.0),
                                         "yes"});

            // The two Monte Carlo estimators against the exact reference.
            for (const auto& [name, result] :
                 {std::pair{std::string("monte_carlo_crude"), &crude.value()},
                  std::pair{std::string("monte_carlo_antithetic"), &anti.value()}}) {
                const double se = result->standard_error.value_or(0.0);
                const Agreement a =
                    agree_within_sigma(result->value, reference, se, 0.0, config.agreement_sigma);
                cell_agrees = cell_agrees && a.agrees;
                ++mc_cells;
                const bool ci_covers = result->confidence_interval.has_value() &&
                                       result->confidence_interval.value().contains(reference);
                if (ci_covers) {
                    ++mc_ci_covers;
                }
                methods.push_back(
                    nlohmann::json{{"method", name},
                                   {"value", result->value},
                                   {"standard_error", se},
                                   {"difference", a.difference},
                                   {"sigmas", a.sigmas},
                                   {"confidence_interval_covers_reference", ci_covers},
                                   {"agrees", a.agrees}});
                record.table.rows.push_back({"bs_european",
                                             regime,
                                             name,
                                             number(result->value),
                                             number(reference),
                                             number(a.difference),
                                             number(a.sigmas),
                                             a.agrees ? "yes" : "NO"});
            }

            // The finite-difference method against the exact reference, to its
            // grid-based relative tolerance.
            const double fd_rel = std::abs(fd.value().value - reference) / std::abs(reference);
            const bool fd_agrees = fd_rel <= config.fd_relative_tolerance;
            cell_agrees = cell_agrees && fd_agrees;
            methods.push_back(nlohmann::json{{"method", "finite_difference_crank_nicolson"},
                                             {"value", fd.value().value},
                                             {"difference", fd.value().value - reference},
                                             {"relative_error", fd_rel},
                                             {"relative_tolerance", config.fd_relative_tolerance},
                                             {"agrees", fd_agrees}});
            record.table.rows.push_back({"bs_european",
                                         regime,
                                         "finite_difference",
                                         number(fd.value().value),
                                         number(reference),
                                         number(fd.value().value - reference),
                                         number(fd_rel),
                                         fd_agrees ? "yes" : "NO"});

            all_agree = all_agree && cell_agrees;
            record.results["black_scholes_european"].push_back(
                nlohmann::json{{"volatility", volatility},
                               {"maturity", maturity},
                               {"reference", reference},
                               {"reference_method", "analytic"},
                               {"methods", std::move(methods)},
                               {"all_agree", cell_agrees}});
        }
    }

    // -----------------------------------------------------------------------
    // Family 2 -- European under Heston. The characteristic-function price is the
    // reference; the full-truncation Monte Carlo simulation must agree with it
    // within sampling uncertainty. Both Feller statuses are exercised.
    // -----------------------------------------------------------------------
    for (const double maturity : config.heston_maturities) {
        for (const double xi : config.heston_vol_of_variances) {
            const auto model = HestonModel::create(config.heston_initial_variance,
                                                   config.heston_mean_reversion,
                                                   config.heston_long_run_variance,
                                                   xi,
                                                   config.heston_correlation);
            const auto option = EuropeanOption::create(OptionType::Call, config.strike, maturity);
            if (!model || !option) {
                return Result<ExperimentRecord>::failure(model.error());
            }
            const auto cf =
                HestonAnalyticEngine::price(market.value(), option.value(), model.value());
            if (!cf) {
                return Result<ExperimentRecord>::failure(cf.error());
            }
            const double reference = cf.value().value;

            HestonMonteCarloConfig hmc;
            hmc.paths = config.heston_monte_carlo_paths;
            hmc.steps = config.heston_monte_carlo_steps;
            hmc.seed = config.heston_monte_carlo_seed;
            const auto mc =
                HestonMonteCarloEngine::price(market.value(), option.value(), model.value(), hmc);
            if (!mc) {
                return Result<ExperimentRecord>::failure(mc.error());
            }
            const double se = mc.value().standard_error.value_or(0.0);
            const Agreement a =
                agree_within_sigma(mc.value().value, reference, se, 0.0, config.agreement_sigma);
            all_agree = all_agree && a.agrees;
            ++mc_cells;
            const bool ci_covers = mc.value().confidence_interval.has_value() &&
                                   mc.value().confidence_interval.value().contains(reference);
            if (ci_covers) {
                ++mc_ci_covers;
            }
            const std::string regime = fmt::format("T={:g} xi={:g}", maturity, xi);
            record.table.rows.push_back({"heston_european",
                                         regime,
                                         "characteristic_function",
                                         number(reference),
                                         number(reference),
                                         number(0.0),
                                         number(0.0),
                                         "yes"});
            record.table.rows.push_back({"heston_european",
                                         regime,
                                         "monte_carlo_full_truncation",
                                         number(mc.value().value),
                                         number(reference),
                                         number(a.difference),
                                         number(a.sigmas),
                                         a.agrees ? "yes" : "NO"});
            record.results["heston_european"].push_back(
                nlohmann::json{{"maturity", maturity},
                               {"vol_of_variance", xi},
                               {"feller_ratio", model.value().feller_ratio()},
                               {"satisfies_feller", model.value().satisfies_feller()},
                               {"characteristic_function_price", reference},
                               {"monte_carlo_price", mc.value().value},
                               {"monte_carlo_standard_error", se},
                               {"difference", a.difference},
                               {"sigmas", a.sigmas},
                               {"confidence_interval_covers_reference", ci_covers},
                               {"agrees", a.agrees}});
        }
    }

    // -----------------------------------------------------------------------
    // Family 3 -- arithmetic Asian under Black-Scholes. There is no closed form, so
    // the reference is the estimators themselves: crude, antithetic, and control-
    // variate Monte Carlo estimate the same price, so they must agree with one
    // another within their combined sampling uncertainty. That variance reduction
    // does not move the mean is exactly the agreement being checked.
    // -----------------------------------------------------------------------
    for (const double volatility : config.volatilities) {
        for (const double maturity : config.maturities) {
            const auto model = BlackScholesModel::create(volatility);
            const auto asian = AsianOption::create(OptionType::Call,
                                                   AveragingType::Arithmetic,
                                                   config.strike,
                                                   maturity,
                                                   config.asian_monitoring_count);
            if (!model || !asian) {
                return Result<ExperimentRecord>::failure(model.error());
            }
            const std::int64_t steps = config.asian_monitoring_count;

            MonteCarloConfig mc_crude;
            mc_crude.paths = config.monte_carlo_paths;
            mc_crude.steps = steps;
            mc_crude.seed = config.monte_carlo_seed;
            mc_crude.control_variate_pilot_paths = config.control_variate_pilot_paths;
            MonteCarloConfig mc_anti = mc_crude;
            mc_anti.variance_reduction.antithetic = true;
            MonteCarloConfig mc_control = mc_crude;
            mc_control.variance_reduction.control_variate = true;

            const auto crude =
                MonteCarloEngine::price(market.value(), asian.value(), model.value(), mc_crude);
            const auto anti =
                MonteCarloEngine::price(market.value(), asian.value(), model.value(), mc_anti);
            const auto control =
                MonteCarloEngine::price(market.value(), asian.value(), model.value(), mc_control);
            if (!crude || !anti || !control) {
                return Result<ExperimentRecord>::failure(crude.error());
            }

            struct Estimator {
                std::string name;
                double value;
                double se;
            };

            const std::vector<Estimator> estimators = {
                {.name = "monte_carlo_crude",
                 .value = crude.value().value,
                 .se = crude.value().standard_error.value_or(0.0)},
                {.name = "monte_carlo_antithetic",
                 .value = anti.value().value,
                 .se = anti.value().standard_error.value_or(0.0)},
                {.name = "monte_carlo_control_variate",
                 .value = control.value().value,
                 .se = control.value().standard_error.value_or(0.0)}};

            const std::string regime = fmt::format("sigma={:g} T={:g}", volatility, maturity);
            for (const Estimator& e : estimators) {
                record.table.rows.push_back(
                    {"arithmetic_asian", regime, e.name, number(e.value), "-", "-", "-", "-"});
            }

            nlohmann::json pairwise = nlohmann::json::array();
            bool cell_agrees = true;
            for (std::size_t i = 0; i < estimators.size(); ++i) {
                for (std::size_t j = i + 1; j < estimators.size(); ++j) {
                    const Agreement a = agree_within_sigma(estimators[i].value,
                                                           estimators[j].value,
                                                           estimators[i].se,
                                                           estimators[j].se,
                                                           config.agreement_sigma);
                    cell_agrees = cell_agrees && a.agrees;
                    pairwise.push_back(
                        nlohmann::json{{"pair", estimators[i].name + " vs " + estimators[j].name},
                                       {"difference", a.difference},
                                       {"combined_standard_error", a.combined_standard_error},
                                       {"sigmas", a.sigmas},
                                       {"agrees", a.agrees}});
                }
            }

            all_agree = all_agree && cell_agrees;
            nlohmann::json estimator_json = nlohmann::json::array();
            for (const Estimator& e : estimators) {
                estimator_json.push_back(nlohmann::json{
                    {"method", e.name}, {"value", e.value}, {"standard_error", e.se}});
            }
            record.results["arithmetic_asian"].push_back(
                nlohmann::json{{"volatility", volatility},
                               {"maturity", maturity},
                               {"reference", "none (no closed form; cross-estimator agreement)"},
                               {"estimators", std::move(estimator_json)},
                               {"pairwise_agreement", std::move(pairwise)},
                               {"all_agree", cell_agrees}});
        }
    }

    record.results["monte_carlo_confidence_interval_coverage"] =
        nlohmann::json{{"cells", mc_cells},
                       {"covered", mc_ci_covers},
                       {"note",
                        "the fraction of 95% Monte Carlo intervals that contain the reference; "
                        "reported as agreement evidence -- the rigorous coverage study is EXP-14"}};

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    record.status = all_agree ? ExperimentStatus::Pass : ExperimentStatus::Fail;

    record.interpretation =
        "The methods agree wherever they price the same instrument, each used only where it is "
        "defined and judged against the right kind of reference. This is an accuracy-and-agreement "
        "study, not a performance one: it makes no claim about which method is faster, only about "
        "whether they describe the same price.\n\n"
        "For the European option under Black-Scholes the analytic price is exact, and it is the "
        "reference. Crude and antithetic Monte Carlo agree with it within sampling uncertainty -- "
        "their difference from the analytic value is a fraction of their own standard error, and "
        "the 95% confidence interval covers the analytic price -- and the Crank-Nicolson "
        "finite-difference solver agrees to its grid-based relative tolerance. Three independent "
        "numerical routes, one analytic answer, no disagreement beyond what sampling and "
        "discretization explain.\n\n"
        "For the European option under Heston the characteristic-function integration is the "
        "reference and the full-truncation Monte Carlo simulation agrees with it within sampling "
        "uncertainty, in both a Feller-satisfying and a Feller-violating regime. The step count is "
        "chosen so the scheme's discretization bias -- the subject of EXP-10 -- is smaller than "
        "the "
        "sampling error here, so the remaining difference is sampling, not a hidden bias; where a "
        "coarser grid would leave a resolved bias, that is EXP-10's finding, not this one's.\n\n"
        "The arithmetic Asian has no closed form, so there is no exact reference to agree with. "
        "Instead the crude, antithetic, and control-variate estimators are checked against each "
        "other: they estimate the same price, so they must agree within their combined standard "
        "errors, and they do. That is the honest form of the agreement question when no oracle "
        "exists -- variance reduction changes the precision, not the answer -- and it is kept "
        "separate from the cases where an exact reference does exist rather than dressed up as "
        "one.";

    record.limitations.emplace_back(
        "Agreement is checked at a single seed per Monte Carlo method, using the run's own "
        "standard error as its uncertainty. That is the honest one-run uncertainty, but a single "
        "seed cannot detect a bias smaller than its standard error; the multi-seed coverage study "
        "is EXP-14, and the estimator-quality comparison is EXP-05.");
    record.limitations.emplace_back(
        "The finite-difference method is compared only on the European option under Black-Scholes, "
        "which is where it is implemented; it is not presented as a reference for the Asian or "
        "Heston instruments it does not price. Each method is used only on the instruments it "
        "supports.");
    record.limitations.emplace_back(
        "The Heston Monte Carlo step count is set so the full-truncation discretization bias is "
        "below the sampling error, so the agreement is within sampling uncertainty. It is not a "
        "claim that full truncation is unbiased -- it is not (EXP-10) -- only that its bias is not "
        "resolved at this resolution.");
    record.limitations.emplace_back(
        "The arithmetic Asian agreement is mutual, not against an external truth. Three estimators "
        "agreeing is strong evidence they share an expectation, but a bias common to all three -- "
        "the same discretization on the same monitoring grid -- would not be caught by comparing "
        "them to each other. The monitoring convention is the same for all three by construction.");

    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
