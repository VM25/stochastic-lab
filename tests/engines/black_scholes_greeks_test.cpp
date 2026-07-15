#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Analytic Greeks versus finite differences of the price formula.
//
// This is an independent check, not self-validation (VALIDATION-PLAN section 1,
// ADR-023). The closed-form Greeks are hand-derived derivatives of the pricing
// formula; differentiating that formula numerically reaches the same quantity by
// an unrelated route. A transcription error in a Greek shows up here, because
// nothing is shared between the two paths but the price function itself.
//
// Tolerances are set by the accuracy of the *finite difference*, not of the
// analytic formula. A plain central difference carries O(h^2) truncation error,
// whose size depends on the scenario: where sigma*sqrt(T) is small the density is
// narrow, the price is sharply curved, and that error grows. Measured against a
// plain difference, low-volatility delta and short-dated gamma disagree at the
// 1e-6 and 1e-5 level respectively -- entirely from the difference quotient.
//
// Rather than loosen the tolerance until those pass, which would blunt the test
// everywhere else, the reference itself is made accurate: Richardson
// extrapolation combines h and h/2 to cancel the h^2 term and leave O(h^4). The
// reference is then good to ~1e-10, and a uniform tight tolerance means
// something at every scenario.
// ---------------------------------------------------------------------------

/// Central first derivative, error O(h^2).
template<typename F>
[[nodiscard]] double central_first_derivative(F f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

/// Central second derivative, error O(h^2).
template<typename F>
[[nodiscard]] double central_second_derivative(F f, double x, double h) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

/// Richardson-extrapolated first derivative, error O(h^4).
///
/// D(h) = D_exact + c2 h^2 + c4 h^4 + ..., so (4 D(h/2) - D(h))/3 cancels c2.
template<typename F>
[[nodiscard]] double richardson_first_derivative(F f, double x, double h) {
    const double coarse = central_first_derivative(f, x, h);
    const double fine = central_first_derivative(f, x, h / 2.0);
    return (4.0 * fine - coarse) / 3.0;
}

/// Richardson-extrapolated second derivative, error O(h^4).
template<typename F>
[[nodiscard]] double richardson_second_derivative(F f, double x, double h) {
    const double coarse = central_second_derivative(f, x, h);
    const double fine = central_second_derivative(f, x, h / 2.0);
    return (4.0 * fine - coarse) / 3.0;
}

struct Scenario {
    std::string name;
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
};

std::vector<Scenario> scenarios() {
    return {
        {"at_the_money", 100.0, 100.0, 0.05, 0.00, 0.20, 1.00},
        {"in_the_money", 130.0, 100.0, 0.05, 0.00, 0.20, 1.00},
        {"out_of_the_money", 70.0, 100.0, 0.05, 0.00, 0.20, 1.00},
        {"with_dividend", 100.0, 100.0, 0.05, 0.03, 0.20, 1.00},
        {"low_volatility", 100.0, 100.0, 0.05, 0.00, 0.05, 1.00},
        {"high_volatility", 100.0, 100.0, 0.05, 0.00, 0.60, 1.00},
        {"short_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 0.08},
        {"long_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 10.0},
        {"negative_rate", 100.0, 100.0, -0.01, 0.00, 0.20, 1.00},
    };
}

double price_at(OptionType type,
                double spot,
                double strike,
                double rate,
                double dividend_yield,
                double volatility,
                double maturity) {
    const auto market = MarketState::create(spot, rate, dividend_yield).value();
    const auto option = EuropeanOption::create(type, strike, maturity).value();
    const auto model = BlackScholesModel::create(volatility).value();
    const auto result = BlackScholesAnalyticEngine::price(market, option, model);
    EXPECT_TRUE(result.ok()) << result.error().describe();
    return result.value().value;
}

Greeks analytic_greeks(OptionType type, const Scenario& s) {
    const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield).value();
    const auto option = EuropeanOption::create(type, s.strike, s.maturity).value();
    const auto model = BlackScholesModel::create(s.volatility).value();
    const auto result = BlackScholesAnalyticEngine::greeks(market, option, model);
    EXPECT_TRUE(result.ok()) << result.error().describe();
    return result.value();
}

/// Compares against a reference whose own error is `tolerance` in relative
/// terms, guarding against a zero reference where a relative test is undefined.
void expect_close(double actual, double expected, double tolerance, const std::string& what) {
    const double scale = std::max(std::abs(expected), 1e-8);
    EXPECT_NEAR(actual, expected, tolerance * scale)
        << what << ": analytic " << actual << " vs finite difference " << expected;
}

TEST(BlackScholesGreeksTest, DeltaMatchesFiniteDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_spot = [&](double spot) {
                return price_at(
                    type, spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);
            };
            const double fd =
                richardson_first_derivative(as_function_of_spot, s.spot, 1e-3 * s.spot);

            expect_close(analytic_greeks(type, s).delta,
                         fd,
                         1e-8,
                         s.name + " " + std::string(to_string(type)) + " delta");
        }
    }
}

TEST(BlackScholesGreeksTest, GammaMatchesFiniteDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_spot = [&](double spot) {
                return price_at(
                    type, spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);
            };
            // A second difference divides by h^2, so rounding error grows as
            // 1/h^2 and h must stay larger here than for delta.
            const double fd =
                richardson_second_derivative(as_function_of_spot, s.spot, 2e-3 * s.spot);

            expect_close(analytic_greeks(type, s).gamma,
                         fd,
                         1e-7,
                         s.name + " " + std::string(to_string(type)) + " gamma");
        }
    }
}

TEST(BlackScholesGreeksTest, VegaMatchesFiniteDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_volatility = [&](double volatility) {
                return price_at(
                    type, s.spot, s.strike, s.rate, s.dividend_yield, volatility, s.maturity);
            };
            const double fd = richardson_first_derivative(
                as_function_of_volatility, s.volatility, 1e-3 * s.volatility);

            expect_close(analytic_greeks(type, s).vega,
                         fd,
                         1e-8,
                         s.name + " " + std::string(to_string(type)) + " vega");
        }
    }
}

TEST(BlackScholesGreeksTest, ThetaMatchesFiniteDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_maturity = [&](double maturity) {
                return price_at(
                    type, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, maturity);
            };
            // theta is dV/dt in calendar time, while the difference is taken in
            // maturity. Time to expiry runs backwards against the clock, hence
            // the sign.
            const double fd = -richardson_first_derivative(
                as_function_of_maturity, s.maturity, 1e-3 * s.maturity);

            expect_close(analytic_greeks(type, s).theta,
                         fd,
                         1e-8,
                         s.name + " " + std::string(to_string(type)) + " theta");
        }
    }
}

TEST(BlackScholesGreeksTest, RhoMatchesFiniteDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_rate = [&](double rate) {
                return price_at(
                    type, s.spot, s.strike, rate, s.dividend_yield, s.volatility, s.maturity);
            };
            // Stepped on an absolute scale: the rate is near zero (and negative
            // in one scenario), so a relative step would collapse.
            const double fd = richardson_first_derivative(as_function_of_rate, s.rate, 1e-4);

            expect_close(analytic_greeks(type, s).rho,
                         fd,
                         1e-8,
                         s.name + " " + std::string(to_string(type)) + " rho");
        }
    }
}

// ---------------------------------------------------------------------------
// Structural relationships between Greeks
// ---------------------------------------------------------------------------

// Differentiating put-call parity in spot: dC/dS - dP/dS = e^{-qT}.
TEST(BlackScholesGreeksTest, DeltasSatisfyParity) {
    for (const auto& s : scenarios()) {
        const double call_delta = analytic_greeks(OptionType::Call, s).delta;
        const double put_delta = analytic_greeks(OptionType::Put, s).delta;
        const double expected = std::exp(-s.dividend_yield * s.maturity);

        EXPECT_NEAR(call_delta - put_delta, expected, 1e-12) << s.name;
    }
}

// Parity is linear in spot and independent of volatility, so its second spot
// derivative and its vega both vanish: calls and puts share gamma and vega.
TEST(BlackScholesGreeksTest, GammaAndVegaAreTypeIndependent) {
    for (const auto& s : scenarios()) {
        const Greeks call = analytic_greeks(OptionType::Call, s);
        const Greeks put = analytic_greeks(OptionType::Put, s);

        EXPECT_NEAR(call.gamma, put.gamma, 1e-14 * std::max(std::abs(call.gamma), 1e-8)) << s.name;
        EXPECT_NEAR(call.vega, put.vega, 1e-12 * std::max(std::abs(call.vega), 1e-8)) << s.name;
    }
}

// Differentiating parity in rate: dC/dr - dP/dr = T K e^{-rT}.
TEST(BlackScholesGreeksTest, RhosSatisfyParity) {
    for (const auto& s : scenarios()) {
        const double call_rho = analytic_greeks(OptionType::Call, s).rho;
        const double put_rho = analytic_greeks(OptionType::Put, s).rho;
        const double expected = s.maturity * s.strike * std::exp(-s.rate * s.maturity);

        EXPECT_NEAR(call_rho - put_rho, expected, 1e-10 * std::max(expected, 1.0)) << s.name;
    }
}

// ---------------------------------------------------------------------------
// Signs and bounds
// ---------------------------------------------------------------------------

TEST(BlackScholesGreeksTest, DeltasLieInTheirTheoreticalRanges) {
    for (const auto& s : scenarios()) {
        const double bound = std::exp(-s.dividend_yield * s.maturity);

        const double call_delta = analytic_greeks(OptionType::Call, s).delta;
        EXPECT_GE(call_delta, 0.0) << s.name;
        EXPECT_LE(call_delta, bound) << s.name;

        const double put_delta = analytic_greeks(OptionType::Put, s).delta;
        EXPECT_LE(put_delta, 0.0) << s.name;
        EXPECT_GE(put_delta, -bound) << s.name;
    }
}

TEST(BlackScholesGreeksTest, GammaAndVegaArePositive) {
    for (const auto& s : scenarios()) {
        const Greeks g = analytic_greeks(OptionType::Call, s);
        EXPECT_GT(g.gamma, 0.0) << s.name;
        EXPECT_GT(g.vega, 0.0) << s.name;
    }
}

// Gamma peaks near the money and decays either side; this is the shape that
// makes it the interesting Greek, and a sign or scaling error would flatten it.
TEST(BlackScholesGreeksTest, GammaPeaksNearTheMoney) {
    const Scenario at_the_money{"atm", 100.0, 100.0, 0.05, 0.0, 0.2, 1.0};
    const Scenario far_in{"itm", 200.0, 100.0, 0.05, 0.0, 0.2, 1.0};
    const Scenario far_out{"otm", 50.0, 100.0, 0.05, 0.0, 0.2, 1.0};

    const double gamma_atm = analytic_greeks(OptionType::Call, at_the_money).gamma;
    EXPECT_GT(gamma_atm, analytic_greeks(OptionType::Call, far_in).gamma);
    EXPECT_GT(gamma_atm, analytic_greeks(OptionType::Call, far_out).gamma);
}

// Away from the kink the degenerate limits are exact, so they are checked
// against the closed forms rather than against a difference quotient.
TEST(BlackScholesGreeksTest, DegenerateGreeksMatchTheirLimits) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();
    const auto model = BlackScholesModel::create(0.0).value();

    // Forward = 100 e^{0.03} = 103.05 > 90, so the call is in the money.
    const auto call = EuropeanOption::create(OptionType::Call, 90.0, 1.0).value();
    const auto g = BlackScholesAnalyticEngine::greeks(market, call, model);
    ASSERT_TRUE(g.ok()) << g.error().describe();

    EXPECT_NEAR(g.value().delta, std::exp(-0.02), 1e-14);
    EXPECT_DOUBLE_EQ(g.value().gamma, 0.0);
    EXPECT_DOUBLE_EQ(g.value().vega, 0.0);
    EXPECT_NEAR(g.value().rho, 90.0 * 1.0 * std::exp(-0.05), 1e-12);

    // Out of the money in the same limit: the option is worthless and inert.
    const auto worthless = EuropeanOption::create(OptionType::Call, 500.0, 1.0).value();
    const auto gw = BlackScholesAnalyticEngine::greeks(market, worthless, model);
    ASSERT_TRUE(gw.ok()) << gw.error().describe();

    EXPECT_DOUBLE_EQ(gw.value().delta, 0.0);
    EXPECT_DOUBLE_EQ(gw.value().gamma, 0.0);
    EXPECT_DOUBLE_EQ(gw.value().vega, 0.0);
    EXPECT_DOUBLE_EQ(gw.value().theta, 0.0);
    EXPECT_DOUBLE_EQ(gw.value().rho, 0.0);
}

}  // namespace
}  // namespace diffusionworks
