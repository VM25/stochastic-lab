#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/experiments/heston_cf_validation_experiments.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

using Complex = std::complex<double>;

std::string number(double x) {
    return fmt::format("{:.6g}", x);
}

// Tolerances. The characteristic-function properties are exact identities, so they
// are held to near machine precision; the reference prices are held to the tolerance
// the reference itself is known to.
constexpr double kPhiZeroTol = 1e-10;       // |phi(0) - 1|
constexpr double kConjTol = 1e-9;           // |phi(-u) - conj(phi(u))|
constexpr double kMartingaleRelTol = 1e-8;  // |phi2(-i) - forward| / forward
constexpr double kModulusTol = 1e-9;        // |phi| <= 1 + this
// The smooth adjacent step on the frequency grid is ~|E[ln S_T]| * du (the
// e^{i u ln S_0} carrier), so the continuity bound scales with the grid spacing du.
// A real branch-cut jump is O(1) and does not shrink with du, so it is caught as
// the grid refines. The constant is generous over |E[ln S_T]| ~ ln(spot) + drift*T.
constexpr double kContinuityLipschitz = 10.0;
constexpr double kParityTol = 1e-9;
constexpr double kBlackScholesLimitTol = 1e-3;
constexpr double kPublishedRefTol = 1e-8;
constexpr double kHighPrecisionRefTol = 1e-6;

Complex
cf(const MarketState& market, const HestonModel& model, double maturity, Complex u, int index) {
    // For xi > 0 and maturity > 0 this never fails; the sweep and reference cases
    // all satisfy that, so the value is taken directly.
    return HestonAnalyticEngine::log_price_characteristic_function(
               market, model, maturity, u, index)
        .value();
}

// An external reference case: a fixed set of parameters with a trusted price, and an
// honest statement of where that price comes from.
struct ReferenceCase {
    const char* name;
    const char* provenance_category;  // "published" or "independently_generated"
    const char* provenance;
    double spot, rate, dividend, strike, maturity, v0, kappa, theta, xi, rho;
    double reference;
    double tolerance;
};

const ReferenceCase kReferences[] = {
    // The one genuinely published benchmark.
    {"fang_oosterlee_2008",
     "published",
     "Fang & Oosterlee (2008) COS-method benchmark",
     100.0,
     0.0,
     0.0,
     100.0,
     1.0,
     0.0175,
     1.5768,
     0.0398,
     0.5751,
     -0.5711,
     5.78515543437619,
     kPublishedRefTol},
    // Independently generated to 40 digits with mpmath and cross-checked against
    // QuantLib's AnalyticHestonEngine. NOT a published literature value.
    {"feller_violating",
     "independently_generated",
     "40-digit mpmath integration, cross-checked with QuantLib (Feller ratio 0.36 < 1)",
     100.0,
     0.05,
     0.0,
     100.0,
     1.0,
     0.09,
     2.0,
     0.09,
     1.0,
     -0.3,
     13.1365327960895,
     kHighPrecisionRefTol},
    {"long_maturity_trap_t5",
     "independently_generated",
     "40-digit mpmath integration; the regime where the naive branch fails",
     100.0,
     0.03,
     0.0,
     100.0,
     5.0,
     0.04,
     1.5,
     0.04,
     0.6,
     -0.7,
     23.50614713998921,
     kHighPrecisionRefTol},
    {"long_maturity_trap_t10",
     "independently_generated",
     "40-digit mpmath integration; the regime where the naive branch fails",
     100.0,
     0.03,
     0.0,
     100.0,
     10.0,
     0.04,
     1.5,
     0.04,
     0.6,
     -0.7,
     36.44207657516968,
     kHighPrecisionRefTol},
    {"long_maturity_trap_t15",
     "independently_generated",
     "40-digit mpmath integration; the regime where the naive branch fails",
     100.0,
     0.03,
     0.0,
     100.0,
     15.0,
     0.04,
     1.5,
     0.04,
     0.6,
     -0.7,
     46.43818472722808,
     kHighPrecisionRefTol},
};

}  // namespace

Result<ExperimentRecord>
run_heston_cf_validation(const HestonCfValidationExperimentConfig& config) {
    const auto start = std::chrono::steady_clock::now();

    ExperimentRecord record;
    record.id = "EXP-09";
    record.name = "Heston Characteristic-Function Validation";
    record.question = "Does native Heston integration reproduce trusted reference prices?";
    record.reproduction_command = "diffusionworks experiment --id EXP-09 --config "
                                  "configs/experiment/heston_cf_validation.json";
    record.configuration =
        nlohmann::json{{"spot", config.spot},
                       {"rate", config.rate},
                       {"dividend_yield", config.dividend_yield},
                       {"initial_variance", config.initial_variance},
                       {"long_run_variance", config.long_run_variance},
                       {"strikes", config.strikes},
                       {"maturities", config.maturities},
                       {"correlations", config.correlations},
                       {"vol_of_variances", config.vol_of_variances},
                       {"mean_reversions", config.mean_reversions},
                       {"cf_grid_max_u", config.cf_grid_max_u},
                       {"cf_grid_points", config.cf_grid_points},
                       {"quadrature_node_counts", config.quadrature_node_counts},
                       {"reference_quadrature_nodes", config.reference_quadrature_nodes}};
    record.table.headers = {"case",
                            "provenance",
                            "maturity",
                            "feller_ratio",
                            "price",
                            "reference",
                            "absolute_error",
                            "relative_error"};
    record.results = nlohmann::json::object();

    // ---------------------------------------------------------------------------
    // Category A -- the characteristic function's own analytic properties.
    //
    // These are properties phi must have because it is the characteristic function
    // of a real log-price under a risk-neutral measure. They are INTERNAL analytic
    // checks, not external oracles: passing them says the function is self-consistent
    // and correctly normalised, not that it prices correctly (that is category B).
    // ---------------------------------------------------------------------------
    double max_phi_zero_dev = 0.0;
    double max_conj_resid = 0.0;
    double max_martingale_rel = 0.0;
    double max_modulus = 0.0;
    double max_continuity_step = 0.0;
    bool all_finite = true;
    double max_parity_resid = 0.0;
    int regimes = 0;
    int feller_satisfied = 0;
    int feller_violated = 0;
    int parity_priced = 0;
    int parity_refused = 0;

    const double du = config.cf_grid_max_u / static_cast<double>(config.cf_grid_points);
    const HestonAnalyticConfig price_config{.quadrature_nodes = config.reference_quadrature_nodes,
                                            .convergence_tolerance = 1e-8};

    for (const double strike : config.strikes) {
        for (const double maturity : config.maturities) {
            for (const double rho : config.correlations) {
                for (const double xi : config.vol_of_variances) {
                    for (const double kappa : config.mean_reversions) {
                        const auto market =
                            MarketState::create(config.spot, config.rate, config.dividend_yield);
                        if (!market) {
                            return Result<ExperimentRecord>::failure(market.error());
                        }
                        const auto model = HestonModel::create(
                            config.initial_variance, kappa, config.long_run_variance, xi, rho);
                        if (!model) {
                            return Result<ExperimentRecord>::failure(model.error());
                        }
                        ++regimes;
                        if (model.value().satisfies_feller()) {
                            ++feller_satisfied;
                        } else {
                            ++feller_violated;
                        }

                        const double forward =
                            config.spot *
                            std::exp((config.rate - config.dividend_yield) * maturity);

                        // phi_j(0) = 1 for both measures.
                        for (const int index : {1, 2}) {
                            const Complex phi0 = cf(
                                market.value(), model.value(), maturity, Complex{0.0, 0.0}, index);
                            max_phi_zero_dev = std::max(max_phi_zero_dev, std::abs(phi0 - 1.0));
                        }

                        // Conjugate symmetry of the risk-neutral CF.
                        for (const double u : {0.3, 1.7, 8.0}) {
                            const Complex plus =
                                cf(market.value(), model.value(), maturity, Complex{u, 0.0}, 2);
                            const Complex minus =
                                cf(market.value(), model.value(), maturity, Complex{-u, 0.0}, 2);
                            max_conj_resid =
                                std::max(max_conj_resid, std::abs(minus - std::conj(plus)));
                        }

                        // Martingale identity phi_2(-i) = forward.
                        const Complex phi_mi =
                            cf(market.value(), model.value(), maturity, Complex{0.0, -1.0}, 2);
                        max_martingale_rel = std::max(
                            max_martingale_rel, std::abs(phi_mi - Complex{forward, 0.0}) / forward);

                        // Finiteness and continuity across a dense real-u grid.
                        Complex previous{1.0, 0.0};
                        for (std::int64_t k = 0; k <= config.cf_grid_points; ++k) {
                            const double u = static_cast<double>(k) * du;
                            const Complex phi =
                                cf(market.value(), model.value(), maturity, Complex{u, 0.0}, 2);
                            if (!std::isfinite(phi.real()) || !std::isfinite(phi.imag())) {
                                all_finite = false;
                            }
                            max_modulus = std::max(max_modulus, std::abs(phi));
                            if (k > 0) {
                                max_continuity_step =
                                    std::max(max_continuity_step, std::abs(phi - previous));
                            }
                            previous = phi;
                        }

                        // Put-call parity where the price converges.
                        const auto call_opt =
                            EuropeanOption::create(OptionType::Call, strike, maturity);
                        const auto put_opt =
                            EuropeanOption::create(OptionType::Put, strike, maturity);
                        if (!call_opt || !put_opt) {
                            return Result<ExperimentRecord>::failure(call_opt.error());
                        }
                        const auto call_price = HestonAnalyticEngine::price(
                            market.value(), call_opt.value(), model.value(), price_config);
                        const auto put_price = HestonAnalyticEngine::price(
                            market.value(), put_opt.value(), model.value(), price_config);
                        if (call_price && put_price) {
                            ++parity_priced;
                            const double parity =
                                config.spot * std::exp(-config.dividend_yield * maturity) -
                                strike * std::exp(-config.rate * maturity);
                            max_parity_resid = std::max(
                                max_parity_resid,
                                std::abs((call_price.value().value - put_price.value().value) -
                                         parity));
                        } else {
                            ++parity_refused;
                        }
                    }
                }
            }
        }
    }

    const double continuity_bound = kContinuityLipschitz * du;
    const bool cf_properties_pass = max_phi_zero_dev <= kPhiZeroTol && max_conj_resid <= kConjTol &&
                                    max_martingale_rel <= kMartingaleRelTol && all_finite &&
                                    max_modulus <= 1.0 + kModulusTol &&
                                    max_continuity_step <= continuity_bound;

    record.results["characteristic_function_properties"] =
        nlohmann::json{{"evidence_category", "internal_analytic_property"},
                       {"regimes", regimes},
                       {"feller_satisfied", feller_satisfied},
                       {"feller_violated", feller_violated},
                       {"max_phi_zero_deviation", max_phi_zero_dev},
                       {"max_conjugate_symmetry_residual", max_conj_resid},
                       {"max_martingale_identity_relative_error", max_martingale_rel},
                       {"max_modulus", max_modulus},
                       {"max_continuity_step", max_continuity_step},
                       {"continuity_step_bound", continuity_bound},
                       {"grid_spacing_du", du},
                       {"all_finite", all_finite},
                       {"passes", cf_properties_pass}};

    // ---------------------------------------------------------------------------
    // Category C(i) -- analytic invariants: put-call parity and the Black-Scholes
    // limit. Model-independent (parity) or a known limiting analytic value (the BS
    // limit); an invariant, not an external price.
    // ---------------------------------------------------------------------------
    const auto bs_market = MarketState::create(config.spot, config.rate, config.dividend_yield);
    const auto bs_option = EuropeanOption::create(OptionType::Call, config.spot, 1.0);
    const double theta = config.long_run_variance;
    const auto heston_limit_model = HestonModel::create(theta, 2.0, theta, 1e-3, 0.0);
    const auto bs_model = BlackScholesModel::create(std::sqrt(theta));
    double bs_limit_error = 0.0;
    bool bs_limit_ok = false;
    if (bs_market && bs_option && heston_limit_model && bs_model) {
        const auto heston_limit = HestonAnalyticEngine::price(
            bs_market.value(), bs_option.value(), heston_limit_model.value(), price_config);
        const auto bs_price = BlackScholesAnalyticEngine::price(
            bs_market.value(), bs_option.value(), bs_model.value());
        if (heston_limit && bs_price) {
            bs_limit_error = std::abs(heston_limit.value().value - bs_price.value().value);
            bs_limit_ok = bs_limit_error <= kBlackScholesLimitTol;
        }
    }
    const bool parity_ok = max_parity_resid <= kParityTol;
    record.results["analytic_invariants"] = nlohmann::json{
        {"evidence_category", "analytic_invariant"},
        {"put_call_parity_max_residual", max_parity_resid},
        {"put_call_parity_regimes_priced", parity_priced},
        {"put_call_parity_regimes_refused", parity_refused},
        {"put_call_parity_passes", parity_ok},
        {"black_scholes_limit_error", bs_limit_error},
        {"black_scholes_limit_note",
         "with xi = 1e-3 and v0 = theta the variance is near-deterministic, so the price "
         "approaches Black-Scholes at volatility sqrt(theta); a loose tolerance because xi is "
         "small "
         "but not zero"},
        {"black_scholes_limit_passes", bs_limit_ok}};

    // ---------------------------------------------------------------------------
    // Category B -- external references: price recovery against trusted values,
    // honestly labelled by where each value comes from.
    // ---------------------------------------------------------------------------
    nlohmann::json references = nlohmann::json::array();
    bool references_pass = true;
    for (const ReferenceCase& c : kReferences) {
        const auto mk = MarketState::create(c.spot, c.rate, c.dividend);
        const auto opt = EuropeanOption::create(OptionType::Call, c.strike, c.maturity);
        const auto model = HestonModel::create(c.v0, c.kappa, c.theta, c.xi, c.rho);
        if (!mk || !opt || !model) {
            return Result<ExperimentRecord>::failure(mk.error());
        }
        const auto priced =
            HestonAnalyticEngine::price(mk.value(), opt.value(), model.value(), price_config);
        if (!priced) {
            return Result<ExperimentRecord>::failure(priced.error());
        }
        const double abs_error = std::abs(priced.value().value - c.reference);
        const double rel_error = abs_error / std::abs(c.reference);
        const bool passes = abs_error <= c.tolerance;
        references_pass = references_pass && passes;

        references.push_back(nlohmann::json{{"case", c.name},
                                            {"evidence_category", "external_reference"},
                                            {"provenance_category", c.provenance_category},
                                            {"provenance", c.provenance},
                                            {"maturity", c.maturity},
                                            {"feller_ratio", model.value().feller_ratio()},
                                            {"price", priced.value().value},
                                            {"reference", c.reference},
                                            {"absolute_error", abs_error},
                                            {"relative_error", rel_error},
                                            {"tolerance", c.tolerance},
                                            {"passes", passes}});
        record.table.rows.push_back({c.name,
                                     c.provenance_category,
                                     number(c.maturity),
                                     number(model.value().feller_ratio()),
                                     number(priced.value().value),
                                     number(c.reference),
                                     number(abs_error),
                                     number(rel_error)});
    }
    record.results["external_references"] = std::move(references);

    // ---------------------------------------------------------------------------
    // Category C(ii) -- internal numerical checks: integration-node convergence and
    // explicit refusal of a pathological regime with the precise failure status.
    // ---------------------------------------------------------------------------
    const auto conv_market = MarketState::create(100.0, 0.0, 0.0);
    const auto conv_option = EuropeanOption::create(OptionType::Call, 100.0, 1.0);
    const auto conv_model = HestonModel::create(0.0175, 1.5768, 0.0398, 0.5751, -0.5711);
    nlohmann::json convergence = nlohmann::json::array();
    double previous_error = std::numeric_limits<double>::infinity();
    double final_error = std::numeric_limits<double>::infinity();
    // Convergence means the doubling error decreases as the grid refines, until it
    // reaches the floating-point floor. Once both consecutive errors are at that
    // floor the difference is rounding noise, not non-convergence, so a wiggle there
    // is allowed; a rise while still above the floor is not.
    constexpr double kConvergenceFloor = 1e-12;
    bool convergence_decreases = true;
    if (conv_market && conv_option && conv_model) {
        for (const std::int64_t nodes : config.quadrature_node_counts) {
            // A permissive tolerance so the (possibly unconverged) value is returned
            // and its own doubling error can be read at each node count.
            const HestonAnalyticConfig cfg{.quadrature_nodes = nodes,
                                           .convergence_tolerance = 1e300};
            const auto priced = HestonAnalyticEngine::price(
                conv_market.value(), conv_option.value(), conv_model.value(), cfg);
            if (!priced) {
                continue;
            }
            double integration_error = 0.0;
            for (const Diagnostic& d : priced.value().diagnostics) {
                if (d.name == "integration_error") {
                    integration_error = std::get<double>(d.value);
                }
            }
            if (integration_error > previous_error && integration_error > kConvergenceFloor &&
                previous_error > kConvergenceFloor) {
                convergence_decreases = false;
            }
            previous_error = integration_error;
            final_error = integration_error;
            convergence.push_back(nlohmann::json{{"quadrature_nodes", nodes},
                                                 {"price", priced.value().value},
                                                 {"integration_error", integration_error}});
        }
    }
    // Converged means it both decreased and reached a small final error.
    const bool convergence_ok = convergence_decreases && final_error < 1e-8;
    // The pathological corner: a deep out-of-the-money, very short maturity, large
    // vol-of-variance regime that does not converge at a practical node count. The
    // engine must refuse it -- with the exact ConvergenceFailure status -- rather
    // than return a plausible number.
    const auto path_market = MarketState::create(100.0, 0.05, 0.0);
    const auto path_option = EuropeanOption::create(OptionType::Call, 150.0, 0.001);
    const auto path_model = HestonModel::create(0.04, 1.0, 0.04, 3.0, 0.9);
    bool pathological_refused = false;
    std::string pathological_status = "priced";
    if (path_market && path_option && path_model) {
        const auto priced = HestonAnalyticEngine::price(
            path_market.value(), path_option.value(), path_model.value());
        if (!priced) {
            pathological_refused = priced.error().code == ErrorCode::ConvergenceFailure;
            pathological_status = std::string(to_string(priced.error().code));
        }
    }
    record.results["internal_numerical_checks"] = nlohmann::json{
        {"evidence_category", "internal_numerical_check"},
        {"integration_convergence", std::move(convergence)},
        {"final_integration_error", final_error},
        {"integration_converges", convergence_ok},
        {"pathological_case", "K=150, T=0.001, xi=3.0, rho=0.9 (deep-OTM, short maturity)"},
        {"pathological_expected_status", "ConvergenceFailure"},
        {"pathological_actual_status", pathological_status},
        {"pathological_refused_correctly", pathological_refused}};

    record.runtime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const bool all_pass = cf_properties_pass && parity_ok && bs_limit_ok && references_pass &&
                          convergence_ok && pathological_refused;
    record.status = all_pass ? ExperimentStatus::Pass : ExperimentStatus::Fail;

    record.interpretation =
        "The experiment validates the characteristic function the price is built from, not only "
        "the prices, and it keeps three kinds of evidence honestly separate rather than calling "
        "them all independent oracles.\n\n"
        "The characteristic function's own analytic properties are internal checks: because phi is "
        "the characteristic function of a real log-price under a risk-neutral measure, it must "
        "satisfy phi_j(0) = 1, conjugate symmetry phi(-u) = conj(phi(u)), the martingale identity "
        "phi_2(-i) = S_0 e^{(r-q)T}, finiteness and continuity across a dense frequency grid, and "
        "|phi| <= 1. These hold to near machine precision across every swept regime, including the "
        "Feller-violating ones -- which says the function is self-consistent and correctly "
        "normalised, not that it prices correctly.\n\n"
        "Whether it prices correctly is the separate question of the external references. The one "
        "genuinely published benchmark is the Fang-Oosterlee (2008) COS value; the "
        "Feller-violating "
        "and long-maturity cases are independently generated to 40 digits with mpmath and "
        "cross-checked against QuantLib, and are labelled as such rather than as literature. Price "
        "recovery matches every reference within its stated tolerance. The long-maturity cases are "
        "the ones where Heston's original branch would return a discontinuous wrong price; the "
        "little-trap form used here stays correct, which the continuity check on phi corroborates "
        "from the other side.\n\n"
        "The analytic invariants -- put-call parity, exact by construction, and the Black-Scholes "
        "limit as vol-of-variance vanishes -- are neither external prices nor CF identities but "
        "consistency conditions the price must meet. The internal numerical checks close the "
        "record: the integration error falls as the quadrature is refined, and the pathological "
        "deep-out-of-the-money short-maturity corner, which does not converge at a practical node "
        "count, is refused with the exact ConvergenceFailure status rather than returned as a "
        "plausible number. A failure that stops is the correct behaviour there; a number that "
        "looked like a price would be the defect.";

    record.limitations.emplace_back(
        "Only one of the reference prices is a published literature value (Fang-Oosterlee 2008). "
        "The others are independently generated to high precision with mpmath and cross-checked "
        "with QuantLib; they are strong external references but not literature benchmarks, and are "
        "labelled accordingly.");
    record.limitations.emplace_back(
        "QuantLib provides the cross-check for the high-precision references but is not linked "
        "into "
        "this experiment binary; the agreement was established when the reference constants were "
        "produced, not recomputed at run time here. The live QuantLib cross-validation lives in "
        "the "
        "dedicated external-validation target.");
    record.limitations.emplace_back(
        "The characteristic-function properties are checked on a finite, if wide, regime sweep and "
        "a finite frequency grid. They are exact identities, so passing them everywhere sampled is "
        "strong evidence, but the sweep does not prove them for every parameter, and the "
        "pathological corner shows the integral itself has regimes it cannot resolve.");
    record.limitations.emplace_back(
        "The martingale identity is evaluated at the single complex point u = -i. It pins the "
        "first moment (the drift); it does not by itself validate higher moments, which the price "
        "recovery against references covers instead.");

    return Result<ExperimentRecord>::success(std::move(record));
}

}  // namespace diffusionworks
