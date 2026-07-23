#include <diffusionworks/engines/heston_analytic.hpp>

#include <fmt/format.h>

#include <cmath>
#include <complex>
#include <numbers>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "HestonAnalyticEngine";

using Complex = std::complex<double>;

/// Gauss-Legendre nodes and weights on [0, 1].
///
/// Computed rather than tabulated so the node count is a free parameter and can be
/// doubled for the convergence check. The classical Newton iteration on the Legendre
/// polynomial, then the interval is mapped from [-1, 1] to [0, 1]. This is standard
/// and converges in a few iterations per node.
struct Quadrature {
    std::vector<double> nodes;
    std::vector<double> weights;
};

[[nodiscard]] Quadrature gauss_legendre_unit(std::int64_t count) {
    const auto n = static_cast<std::size_t>(count);
    Quadrature q;
    q.nodes.resize(n);
    q.weights.resize(n);

    // Nodes are symmetric about 0 on [-1, 1]; compute the first half and mirror.
    const std::size_t m = (n + 1) / 2;
    for (std::size_t i = 0; i < m; ++i) {
        // Initial guess (Abramowitz & Stegun) for the i-th root.
        double x = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) /
                            (static_cast<double>(n) + 0.5));
        double dp = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            // Legendre polynomial P_n(x) and its derivative by recurrence.
            double p0 = 1.0;
            double p1 = x;
            for (std::size_t k = 2; k <= n; ++k) {
                const double p2 = ((2.0 * static_cast<double>(k) - 1.0) * x * p1 -
                                   (static_cast<double>(k) - 1.0) * p0) /
                                  static_cast<double>(k);
                p0 = p1;
                p1 = p2;
            }
            dp = static_cast<double>(n) * (x * p1 - p0) / (x * x - 1.0);
            const double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1e-15) {
                break;
            }
        }
        const double weight = 2.0 / ((1.0 - x * x) * dp * dp);
        // Map [-1, 1] -> [0, 1]: t = (x + 1)/2, w -> w/2. The i-th root -x maps to
        // the mirror node.
        q.nodes[i] = 0.5 * (1.0 - x);
        q.nodes[n - 1 - i] = 0.5 * (1.0 + x);
        q.weights[i] = 0.5 * weight;
        q.weights[n - 1 - i] = 0.5 * weight;
    }
    return q;
}

/// Parameters bundled for the characteristic function.
struct HestonCharacteristics {
    double drift;  // r - q
    double maturity;
    double log_spot;
    double kappa;
    double theta;
    double xi;
    double rho;
    double v0;
};

/// The little-trap characteristic function of the log-price, evaluated for the two
/// probabilities P1 (index 1) and P2 (index 2). The argument is complex: the
/// pricing integrand supplies a real `u`, while validation evaluates identities off
/// the real axis (e.g. the martingale point u = -i).
[[nodiscard]] Complex
characteristic_function(Complex u, int index, const HestonCharacteristics& p) {
    const Complex i{0.0, 1.0};
    const double u_j = index == 1 ? 0.5 : -0.5;
    const double b = index == 1 ? p.kappa - p.rho * p.xi : p.kappa;

    const Complex rxiu = p.rho * p.xi * i * u;
    const double xi2 = p.xi * p.xi;
    const Complex d = std::sqrt((rxiu - b) * (rxiu - b) - xi2 * (2.0 * u_j * i * u - u * u));

    // Little trap: numerator uses (b - rxiu - d) and the exp(-d T) branch, which
    // keeps the principal logarithm correct for every maturity. Heston's original
    // (with +d and exp(+d T)) crosses a branch cut for large T.
    const Complex b_minus = b - rxiu - d;
    const Complex b_plus = b - rxiu + d;
    const Complex g = b_minus / b_plus;
    const Complex edt = std::exp(-d * p.maturity);

    const Complex big_c = p.drift * i * u * p.maturity +
                          (p.kappa * p.theta / xi2) *
                              (b_minus * p.maturity - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
    const Complex big_d = (b_minus / xi2) * ((1.0 - edt) / (1.0 - g * edt));

    return std::exp(big_c + big_d * p.v0 + i * u * p.log_spot);
}

/// The integrand of P_j after the change of variables u = x/(1-x), x in [0, 1).
[[nodiscard]] double
probability_integrand(double x, int index, double log_strike, const HestonCharacteristics& p) {
    const double one_minus = 1.0 - x;
    const double u = x / one_minus;
    const Complex i{0.0, 1.0};
    const Complex value = std::exp(-i * u * log_strike) *
                          characteristic_function(Complex{u, 0.0}, index, p) / (i * u);
    return value.real() / (one_minus * one_minus);
}

/// P_j = 1/2 + (1/pi) integral, by Gauss-Legendre on the transformed domain.
[[nodiscard]] double probability(int index,
                                 double log_strike,
                                 const HestonCharacteristics& p,
                                 const Quadrature& quadrature) {
    double integral = 0.0;
    for (std::size_t k = 0; k < quadrature.nodes.size(); ++k) {
        integral += quadrature.weights[k] *
                    probability_integrand(quadrature.nodes[k], index, log_strike, p);
    }
    return 0.5 + integral / std::numbers::pi;
}

[[nodiscard]] double heston_call(const HestonCharacteristics& p,
                                 double spot,
                                 double strike,
                                 double asset_discount,
                                 double cash_discount,
                                 const Quadrature& quadrature) {
    const double log_strike = std::log(strike);
    const double p1 = probability(1, log_strike, p, quadrature);
    const double p2 = probability(2, log_strike, p, quadrature);
    return spot * asset_discount * p1 - strike * cash_discount * p2;
}

}  // namespace

Result<PricingResult> HestonAnalyticEngine::price(const MarketState& market,
                                                  const EuropeanOption& option,
                                                  const HestonModel& model,
                                                  const HestonAnalyticConfig& config) {
    if (config.quadrature_nodes < 8) {
        return Result<PricingResult>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the quadrature needs at least 8 nodes to resolve the integrand, got {}",
                        config.quadrature_nodes),
            kContext);
    }
    if (!(model.vol_of_variance() > 0.0)) {
        // The pricing integral divides by xi^2. At xi = 0 the variance is
        // deterministic and the model reduces to Black-Scholes with the integrated
        // variance, which is a different computation, not this integral. Refused
        // rather than returning a division-by-zero disguised as a price.
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "the semi-analytic engine needs a positive vol-of-variance: at xi = 0 the variance is "
            "deterministic and the model is Black-Scholes with integrated variance, which this "
            "characteristic-function integral does not represent",
            kContext);
    }

    const double maturity = option.maturity();
    if (maturity <= 0.0) {
        return Result<PricingResult>::failure(
            ErrorCode::UnsupportedCombination,
            "the Heston integral is undefined at zero maturity; the payoff is the price at expiry "
            "and the intrinsic value is exact",
            kContext);
    }

    const double spot = market.spot();
    const double strike = option.strike();
    const double asset_discount = market.dividend_discount_factor(maturity);
    const double cash_discount = market.discount_factor(maturity);

    const HestonCharacteristics params{.drift = market.rate() - market.dividend_yield(),
                                       .maturity = maturity,
                                       .log_spot = std::log(spot),
                                       .kappa = model.mean_reversion(),
                                       .theta = model.long_run_variance(),
                                       .xi = model.vol_of_variance(),
                                       .rho = model.correlation(),
                                       .v0 = model.initial_variance()};

    // The price at N nodes and at 2N. Their difference is the integration error, and
    // it is what decides whether the integral converged. An integral reported without
    // that check is a number that looks like a price.
    const Quadrature base = gauss_legendre_unit(config.quadrature_nodes);
    const Quadrature refined = gauss_legendre_unit(2 * config.quadrature_nodes);

    const double call_base = heston_call(params, spot, strike, asset_discount, cash_discount, base);
    const double call_refined =
        heston_call(params, spot, strike, asset_discount, cash_discount, refined);

    if (!std::isfinite(call_refined) || !std::isfinite(call_base)) {
        return Result<PricingResult>::failure(
            ErrorCode::NonFiniteValue,
            fmt::format("the Heston integral produced a non-finite value (N={}: {}, 2N: {}); the "
                        "characteristic function likely overflowed for these parameters",
                        config.quadrature_nodes,
                        call_base,
                        call_refined),
            kContext);
    }

    const double integration_error = std::abs(call_refined - call_base);
    if (integration_error > config.convergence_tolerance) {
        return Result<PricingResult>::failure(
            ErrorCode::ConvergenceFailure,
            fmt::format(
                "the pricing integral did not converge: doubling the quadrature from {} to "
                "{} nodes changed the call by {:.3e}, above the tolerance {:.3e}. This is a "
                "hard regime (short maturity or large vol-of-variance); increase "
                "quadrature_nodes rather than trusting the unconverged value.",
                config.quadrature_nodes,
                2 * config.quadrature_nodes,
                integration_error,
                config.convergence_tolerance),
            kContext);
    }

    double value = call_refined;
    // Put by parity: exact and model-independent, so no second integral.
    if (option.type() == OptionType::Put) {
        value = call_refined - spot * asset_discount + strike * cash_discount;
    }

    if (value < 0.0) {
        // A price cannot be negative. Near-zero deep-out-of-the-money values can
        // round slightly below zero; a materially negative one is an integration
        // failure, and either way it is reported rather than clamped.
        if (value < -config.convergence_tolerance) {
            return Result<PricingResult>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("the Heston price is negative ({:.3e}), which the contract cannot be; "
                            "the integral has not converged for these parameters",
                            value),
                kContext);
        }
        value = 0.0;
    }

    PricingResult result;
    result.method = "heston_analytic";
    result.value = value;
    result.add_diagnostic("integration_error", integration_error);
    result.add_diagnostic("quadrature_nodes", config.quadrature_nodes);
    result.add_diagnostic("feller_ratio", model.feller_ratio());
    result.add_diagnostic("satisfies_feller", model.satisfies_feller());

    if (!model.satisfies_feller()) {
        result.add_warning(fmt::format(
            "the Feller condition 2*kappa*theta >= xi^2 is violated (ratio {:.4f} < 1): the "
            "variance process can reach zero. The price is unaffected -- the integral is well "
            "defined regardless -- but a naive simulation of these parameters will misbehave near "
            "zero variance, which is Phase 10's concern.",
            model.feller_ratio()));
    }

    return Result<PricingResult>::success(std::move(result));
}

Result<std::complex<double>>
HestonAnalyticEngine::log_price_characteristic_function(const MarketState& market,
                                                        const HestonModel& model,
                                                        double maturity,
                                                        std::complex<double> u,
                                                        int index) {
    if (index != 1 && index != 2) {
        return Result<std::complex<double>>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the characteristic-function index must be 1 or 2, got {}", index),
            kContext);
    }
    if (!(maturity > 0.0)) {
        return Result<std::complex<double>>::failure(
            ErrorCode::UnsupportedCombination,
            "the characteristic function is undefined at zero maturity",
            kContext);
    }
    if (!(model.vol_of_variance() > 0.0)) {
        return Result<std::complex<double>>::failure(
            ErrorCode::UnsupportedCombination,
            "the characteristic function needs a positive vol-of-variance: the formula divides by "
            "xi^2, and at xi = 0 the variance is deterministic",
            kContext);
    }

    const HestonCharacteristics params{.drift = market.rate() - market.dividend_yield(),
                                       .maturity = maturity,
                                       .log_spot = std::log(market.spot()),
                                       .kappa = model.mean_reversion(),
                                       .theta = model.long_run_variance(),
                                       .xi = model.vol_of_variance(),
                                       .rho = model.correlation(),
                                       .v0 = model.initial_variance()};
    return Result<std::complex<double>>::success(characteristic_function(u, index, params));
}

}  // namespace diffusionworks
