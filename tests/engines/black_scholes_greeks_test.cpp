#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/numerics/normal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Internal consistency: analytic Greeks versus differences of the price.
//
// What this establishes, and what it does not
// -------------------------------------------
// It confirms that each closed-form Greek really is the derivative of *this
// engine's price function*, which catches a transcription slip in the
// hand-derived algebra.
//
// It cannot do more, and is deliberately not the Greek oracle. It differentiates
// the very implementation it checks, so a fault shared by the price and its
// Greeks -- a wrong price formula with a consistently wrong delta -- passes
// silently. The independent oracles live in black_scholes_reference_test.cpp:
// published Hull and Haug values, and a 50-digit mpmath table that
// differentiates a separate price implementation. This file is the
// internal-consistency layer of that set, never its foundation.
//
// Reference accuracy
// ------------------
// A plain central difference carries O(h^2) truncation error whose size depends
// on the scenario: where sigma*sqrt(T) is small the density is narrow, the price
// is sharply curved, and the error grows. Measured that way, low-volatility delta
// and short-dated gamma disagreed at 1e-6 and 1e-5 -- entirely from the
// difference quotient, not the engine. Rather than loosen the bound until they
// passed, which would blunt the test everywhere else, the reference is made
// accurate: Richardson extrapolation cancels the h^2 term and leaves O(h^4).
//
// Tolerances
// ----------
// No single threshold is defensible across these scenarios, so each comparison
// uses |actual - expected| <= abs + rel*|expected|:
//
//   - a purely relative bound is meaningless where a Greek is legitimately near
//     zero (deep out-of-the-money delta, or gamma in the degenerate limit);
//   - a purely absolute bound is meaningless where a Greek is large (long-dated
//     rho is order 1e2, vega order 1e1).
//
// The absolute floor is set by the residual rounding of the difference, which
// scales with the *price* (order 1e1 here) divided by the step, not with the
// derivative being estimated. Values are justified per Greek below.
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
    EXPECT_TRUE(result.value().all_defined())
        << "every Greek is defined away from the degenerate kink";
    return result.value();
}

/// Bump size for a variable whose natural scale is `scale`.
///
/// Scale-aware rather than absolute: a step of 0.1 is a rounding error on a spot
/// of 100 but a 200% move in a volatility of 0.05. The fraction is chosen so the
/// O(h^4) truncation left by Richardson and the O(eps*V/h) rounding are both far
/// below the tolerances asserted.
[[nodiscard]] double bump_for(double scale, double fraction) {
    return fraction * std::abs(scale);
}

/// Asserts |actual - expected| <= abs_tolerance + rel_tolerance*|expected|.
void expect_close(double actual,
                  double expected,
                  double rel_tolerance,
                  double abs_tolerance,
                  const std::string& what) {
    const double allowed = abs_tolerance + rel_tolerance * std::abs(expected);
    EXPECT_NEAR(actual, expected, allowed)
        << what << "\n  analytic          : " << actual << "\n  finite difference : " << expected
        << "\n  allowed           : " << allowed << " (abs " << abs_tolerance << " + rel "
        << rel_tolerance << ")";
}

TEST(BlackScholesGreeksConsistencyTest, DeltaMatchesRichardsonDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_spot = [&](double spot) {
                return price_at(
                    type, spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);
            };
            const double fd =
                richardson_first_derivative(as_function_of_spot, s.spot, bump_for(s.spot, 1e-3));

            // abs floor: rounding in the difference is ~eps*price/h ~ 1e-16*1e1/1e-1 = 1e-14;
            // 1e-10 leaves four orders of margin without admitting a real defect,
            // since delta is O(1) and a genuine error would be far larger.
            expect_close(*analytic_greeks(type, s).delta,
                         fd,
                         1e-8,
                         1e-10,
                         s.name + " " + std::string(to_string(type)) + " delta");
        }
    }
}

TEST(BlackScholesGreeksConsistencyTest, GammaMatchesRichardsonDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_spot = [&](double spot) {
                return price_at(
                    type, spot, s.strike, s.rate, s.dividend_yield, s.volatility, s.maturity);
            };
            // A second difference divides by h^2, so rounding grows as 1/h^2 and
            // the step must stay larger than for delta.
            const double fd =
                richardson_second_derivative(as_function_of_spot, s.spot, bump_for(s.spot, 2e-3));

            // Gamma is O(1e-2) here and its difference is the noisiest of the
            // five (~eps*price/h^2 ~ 1e-12), so the absolute floor is looser in
            // absolute terms while still far below the value itself.
            expect_close(*analytic_greeks(type, s).gamma,
                         fd,
                         1e-7,
                         1e-11,
                         s.name + " " + std::string(to_string(type)) + " gamma");
        }
    }
}

TEST(BlackScholesGreeksConsistencyTest, VegaMatchesRichardsonDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_volatility = [&](double volatility) {
                return price_at(
                    type, s.spot, s.strike, s.rate, s.dividend_yield, volatility, s.maturity);
            };
            // Stepped relative to volatility, not spot: 1e-3*100 would be a
            // 2000% volatility move.
            const double fd = richardson_first_derivative(
                as_function_of_volatility, s.volatility, bump_for(s.volatility, 1e-3));

            // Vega is O(1e1), so an absolute floor of 1e-9 is ~1e-10 relative.
            expect_close(*analytic_greeks(type, s).vega,
                         fd,
                         1e-8,
                         1e-9,
                         s.name + " " + std::string(to_string(type)) + " vega");
        }
    }
}

TEST(BlackScholesGreeksConsistencyTest, ThetaMatchesRichardsonDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_maturity = [&](double maturity) {
                return price_at(
                    type, s.spot, s.strike, s.rate, s.dividend_yield, s.volatility, maturity);
            };
            // theta is dV/dt in calendar time while the difference is taken in
            // maturity; time to expiry runs backwards against the clock.
            const double fd = -richardson_first_derivative(
                as_function_of_maturity, s.maturity, bump_for(s.maturity, 1e-3));

            expect_close(*analytic_greeks(type, s).theta,
                         fd,
                         1e-8,
                         1e-9,
                         s.name + " " + std::string(to_string(type)) + " theta");
        }
    }
}

TEST(BlackScholesGreeksConsistencyTest, RhoMatchesRichardsonDifference) {
    for (const auto& s : scenarios()) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const auto as_function_of_rate = [&](double rate) {
                return price_at(
                    type, s.spot, s.strike, rate, s.dividend_yield, s.volatility, s.maturity);
            };
            // The rate's own magnitude is a poor scale: it is 0.05 in most
            // scenarios, negative in one, and could legitimately be zero, where a
            // relative step would collapse to nothing. A fixed 1e-4 of a
            // representative 1e-2 rate scale is used instead.
            const double fd =
                richardson_first_derivative(as_function_of_rate, s.rate, bump_for(0.01, 1e-2));

            // Rho reaches O(1e2) at ten years, so the relative term dominates.
            expect_close(*analytic_greeks(type, s).rho,
                         fd,
                         1e-8,
                         1e-9,
                         s.name + " " + std::string(to_string(type)) + " rho");
        }
    }
}

// ---------------------------------------------------------------------------
// Structural relationships between Greeks
//
// These are model-free consequences of put-call parity, so they hold to
// floating-point accuracy rather than to a modelling tolerance, and they are
// independent of any differencing.
// ---------------------------------------------------------------------------

// Differentiating parity in spot: dC/dS - dP/dS = e^{-qT}.
TEST(BlackScholesGreeksTest, DeltasSatisfyParity) {
    for (const auto& s : scenarios()) {
        const double call_delta = *analytic_greeks(OptionType::Call, s).delta;
        const double put_delta = *analytic_greeks(OptionType::Put, s).delta;
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

        EXPECT_NEAR(*call.gamma, *put.gamma, 1e-14 * std::max(std::abs(*call.gamma), 1e-8))
            << s.name;
        EXPECT_NEAR(*call.vega, *put.vega, 1e-12 * std::max(std::abs(*call.vega), 1e-8)) << s.name;
    }
}

// Differentiating parity in rate: dC/dr - dP/dr = T K e^{-rT}.
TEST(BlackScholesGreeksTest, RhosSatisfyParity) {
    for (const auto& s : scenarios()) {
        const double call_rho = *analytic_greeks(OptionType::Call, s).rho;
        const double put_rho = *analytic_greeks(OptionType::Put, s).rho;
        const double expected = s.maturity * s.strike * std::exp(-s.rate * s.maturity);

        EXPECT_NEAR(call_rho - put_rho, expected, 1e-10 * std::max(expected, 1.0)) << s.name;
    }
}

// vega = gamma * S^2 * sigma * T is an identity of the Black-Scholes density,
// linking a first and a second derivative. It is an independent structural check:
// gamma and vega are computed by different expressions, so a scaling error in
// either breaks it.
TEST(BlackScholesGreeksTest, VegaGammaIdentityHolds) {
    for (const auto& s : scenarios()) {
        const Greeks g = analytic_greeks(OptionType::Call, s);
        const double expected = *g.gamma * s.spot * s.spot * s.volatility * s.maturity;

        EXPECT_NEAR(*g.vega, expected, 1e-10 * std::max(std::abs(expected), 1.0)) << s.name;
    }
}

// ---------------------------------------------------------------------------
// Signs and bounds
// ---------------------------------------------------------------------------

TEST(BlackScholesGreeksTest, DeltasLieInTheirTheoreticalRanges) {
    for (const auto& s : scenarios()) {
        const double bound = std::exp(-s.dividend_yield * s.maturity);

        const double call_delta = *analytic_greeks(OptionType::Call, s).delta;
        EXPECT_GE(call_delta, 0.0) << s.name;
        EXPECT_LE(call_delta, bound) << s.name;

        const double put_delta = *analytic_greeks(OptionType::Put, s).delta;
        EXPECT_LE(put_delta, 0.0) << s.name;
        EXPECT_GE(put_delta, -bound) << s.name;
    }
}

TEST(BlackScholesGreeksTest, GammaAndVegaArePositive) {
    for (const auto& s : scenarios()) {
        const Greeks g = analytic_greeks(OptionType::Call, s);
        EXPECT_GT(*g.gamma, 0.0) << s.name;
        EXPECT_GT(*g.vega, 0.0) << s.name;
    }
}

// Gamma peaks near the money and decays either side; this is the shape that
// makes it the interesting Greek, and a sign or scaling error would flatten it.
TEST(BlackScholesGreeksTest, GammaPeaksNearTheMoney) {
    const Scenario at_the_money{"atm", 100.0, 100.0, 0.05, 0.0, 0.2, 1.0};
    const Scenario far_in{"itm", 200.0, 100.0, 0.05, 0.0, 0.2, 1.0};
    const Scenario far_out{"otm", 50.0, 100.0, 0.05, 0.0, 0.2, 1.0};

    const double gamma_atm = *analytic_greeks(OptionType::Call, at_the_money).gamma;
    EXPECT_GT(gamma_atm, *analytic_greeks(OptionType::Call, far_in).gamma);
    EXPECT_GT(gamma_atm, *analytic_greeks(OptionType::Call, far_out).gamma);
}

// ---------------------------------------------------------------------------
// Degenerate limits
// ---------------------------------------------------------------------------

// Away from the kink every Greek exists, so the request must not be refused: the
// limits are exact and are checked against the closed forms rather than against a
// difference quotient, which cannot straddle the corner anyway.
TEST(BlackScholesDegenerateGreeksTest, AwayFromTheKinkAllGreeksAreDefined) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();
    const auto model = BlackScholesModel::create(0.0).value();

    // Forward = 100 e^{0.03} = 103.05 > 90, so the call is in the money.
    const auto call = EuropeanOption::create(OptionType::Call, 90.0, 1.0).value();
    const auto g = BlackScholesAnalyticEngine::greeks(market, call, model);
    ASSERT_TRUE(g.ok()) << g.error().describe();
    ASSERT_TRUE(g.value().all_defined()) << "limits exist away from the kink and must be reported";
    EXPECT_TRUE(g.value().undefined.empty());

    EXPECT_NEAR(*g.value().delta, std::exp(-0.02), 1e-14);
    EXPECT_DOUBLE_EQ(*g.value().gamma, 0.0);
    EXPECT_DOUBLE_EQ(*g.value().vega, 0.0);
    EXPECT_NEAR(*g.value().rho, 90.0 * 1.0 * std::exp(-0.05), 1e-12);

    // Out of the money in the same limit: the option is worthless and inert.
    const auto worthless = EuropeanOption::create(OptionType::Call, 500.0, 1.0).value();
    const auto gw = BlackScholesAnalyticEngine::greeks(market, worthless, model);
    ASSERT_TRUE(gw.ok()) << gw.error().describe();
    ASSERT_TRUE(gw.value().all_defined());

    EXPECT_DOUBLE_EQ(*gw.value().delta, 0.0);
    EXPECT_DOUBLE_EQ(*gw.value().gamma, 0.0);
    EXPECT_DOUBLE_EQ(*gw.value().vega, 0.0);
    EXPECT_DOUBLE_EQ(*gw.value().theta, 0.0);
    EXPECT_DOUBLE_EQ(*gw.value().rho, 0.0);
}

/// Builds the zero-diffusion payoff kink: sigma = 0 with the forward exactly at
/// the strike.
struct KinkCase {
    MarketState market;
    EuropeanOption option;
    BlackScholesModel model;
};

KinkCase make_kink(double rate, double dividend_yield, double maturity, double volatility = 0.0) {
    // r = q keeps the forward at spot, so F = K = 100 exactly. Otherwise the
    // strike is set to the forward so that log(F/K) is exactly zero.
    const auto market = MarketState::create(100.0, rate, dividend_yield).value();
    const double strike = market.forward(maturity);
    return KinkCase{market,
                    EuropeanOption::create(OptionType::Call, strike, maturity).value(),
                    BlackScholesModel::create(volatility).value()};
}

// The single most important guarantee of the degenerate branch: gamma is a Dirac
// mass at the corner, so no finite value may ever be reported for it.
TEST(BlackScholesDegenerateGreeksTest, GammaIsNeverInventedAtTheKink) {
    for (const double rate : {0.05, 0.00, -0.01}) {
        for (const double dividend_yield : {0.05, 0.00, 0.03}) {
            for (const double maturity : {0.0, 0.5, 2.0}) {
                const KinkCase c = make_kink(rate, dividend_yield, maturity);
                const BlackScholesTerms t =
                    BlackScholesAnalyticEngine::terms(c.market, c.option, c.model);
                ASSERT_TRUE(t.degenerate);
                ASSERT_NEAR(t.log_moneyness, 0.0, 1e-15) << "setup must sit on the kink";

                const auto g = BlackScholesAnalyticEngine::greeks(c.market, c.option, c.model);
                ASSERT_TRUE(g.ok()) << g.error().describe();
                EXPECT_FALSE(g.value().gamma.has_value())
                    << "a finite gamma was invented at the kink (r=" << rate
                    << " q=" << dividend_yield << " T=" << maturity << ")";
            }
        }
    }
}

// The request must not fail wholesale: the price is exact and vega genuinely
// exists, so refusing everything would discard a real number.
TEST(BlackScholesDegenerateGreeksTest, KinkReportsExactPriceAndPartialGreeks) {
    const KinkCase c = make_kink(0.05, 0.05, 1.0);

    const auto priced = BlackScholesAnalyticEngine::price(c.market, c.option, c.model);
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_DOUBLE_EQ(priced.value().value, 0.0) << "the kink price is exactly zero";

    const auto g = BlackScholesAnalyticEngine::greeks(c.market, c.option, c.model);
    ASSERT_TRUE(g.ok()) << "the whole Greek request must not fail at the kink";

    EXPECT_FALSE(g.value().delta.has_value());
    EXPECT_FALSE(g.value().gamma.has_value());
    EXPECT_TRUE(g.value().vega.has_value()) << "vega exists at the kink and must be reported";
}

// Along F = K the price is D[2N(sigma sqrt(T)/2) - 1], whose derivative at
// sigma = 0 is D sqrt(T) phi(0). This is a one-sided derivative, and since
// sigma < 0 is outside the domain it is the derivative.
TEST(BlackScholesDegenerateGreeksTest, VegaAtTheKinkMatchesItsClosedForm) {
    const double maturity = 2.0;
    const KinkCase c = make_kink(0.05, 0.02, maturity);

    const auto g = BlackScholesAnalyticEngine::greeks(c.market, c.option, c.model);
    ASSERT_TRUE(g.ok()) << g.error().describe();
    ASSERT_TRUE(g.value().vega.has_value());

    const double expected = c.market.spot() * std::exp(-c.market.dividend_yield() * maturity) *
                            norm_pdf(0.0) * std::sqrt(maturity);
    EXPECT_NEAR(*g.value().vega, expected, 1e-12);

    // Confirmed independently by a one-sided difference in sigma, which is the
    // only side that exists.
    const auto price_at_vol = [&](double volatility) {
        const auto model = BlackScholesModel::create(volatility).value();
        return BlackScholesAnalyticEngine::price(c.market, c.option, model).value().value;
    };
    const double h = 1e-6;
    const double one_sided = (price_at_vol(h) - price_at_vol(0.0)) / h;
    // A one-sided difference is only O(h) accurate, hence the loose bound; it
    // exists to confirm the magnitude, not to pin the digits.
    EXPECT_NEAR(*g.value().vega, one_sided, 1e-4 * std::abs(expected));
}

// theta exists at the kink only when sigma = 0 and r = q, where F = K forces
// S = K and the value vanishes identically in maturity.
TEST(BlackScholesDegenerateGreeksTest, ThetaAtTheKinkExistsOnlyWhenCarryVanishes) {
    const KinkCase equal_carry = make_kink(0.05, 0.05, 1.0);
    const auto g_equal = BlackScholesAnalyticEngine::greeks(
        equal_carry.market, equal_carry.option, equal_carry.model);
    ASSERT_TRUE(g_equal.ok());
    ASSERT_TRUE(g_equal.value().theta.has_value()) << "r = q makes the value identically zero in T";
    EXPECT_DOUBLE_EQ(*g_equal.value().theta, 0.0);

    // r != q: the one-sided limits in maturity disagree.
    const KinkCase unequal_carry = make_kink(0.05, 0.02, 1.0);
    const auto g_unequal = BlackScholesAnalyticEngine::greeks(
        unequal_carry.market, unequal_carry.option, unequal_carry.model);
    ASSERT_TRUE(g_unequal.ok());
    EXPECT_FALSE(g_unequal.value().theta.has_value());
}

// rho exists at the kink only at expiry, where the payoff does not involve the
// discount factor at all.
TEST(BlackScholesDegenerateGreeksTest, RhoAtTheKinkExistsOnlyAtExpiry) {
    const KinkCase expired = make_kink(0.05, 0.02, 0.0);
    const auto g_expired =
        BlackScholesAnalyticEngine::greeks(expired.market, expired.option, expired.model);
    ASSERT_TRUE(g_expired.ok());
    ASSERT_TRUE(g_expired.value().rho.has_value());
    EXPECT_DOUBLE_EQ(*g_expired.value().rho, 0.0);

    const KinkCase live = make_kink(0.05, 0.02, 1.0);
    const auto g_live = BlackScholesAnalyticEngine::greeks(live.market, live.option, live.model);
    ASSERT_TRUE(g_live.ok());
    EXPECT_FALSE(g_live.value().rho.has_value());
}

// An absent Greek must never be silently absent.
TEST(BlackScholesDegenerateGreeksTest, EveryUndefinedGreekCarriesAReason) {
    const KinkCase c = make_kink(0.05, 0.02, 1.0);
    const auto g = BlackScholesAnalyticEngine::greeks(c.market, c.option, c.model);
    ASSERT_TRUE(g.ok());

    const Greeks& greeks = g.value();
    std::vector<std::string> reported;
    for (const UndefinedGreek& entry : greeks.undefined) {
        EXPECT_FALSE(entry.reason.empty()) << entry.name << " is undefined without a reason";
        reported.push_back(entry.name);
    }

    const auto reason_given = [&](const std::string& name) {
        return std::find(reported.begin(), reported.end(), name) != reported.end();
    };

    if (!greeks.delta.has_value()) {
        EXPECT_TRUE(reason_given("delta"));
    }
    if (!greeks.gamma.has_value()) {
        EXPECT_TRUE(reason_given("gamma"));
    }
    if (!greeks.theta.has_value()) {
        EXPECT_TRUE(reason_given("theta"));
    }
    if (!greeks.rho.has_value()) {
        EXPECT_TRUE(reason_given("rho"));
    }

    // Exactly the absent ones are explained: no reason is recorded for a Greek
    // that was in fact returned.
    EXPECT_EQ(reported.size(),
              static_cast<std::size_t>(!greeks.delta.has_value()) +
                  static_cast<std::size_t>(!greeks.gamma.has_value()) +
                  static_cast<std::size_t>(!greeks.vega.has_value()) +
                  static_cast<std::size_t>(!greeks.theta.has_value()) +
                  static_cast<std::size_t>(!greeks.rho.has_value()));
}

// At expiry with sigma > 0 the at-the-money price behaves like sigma*sqrt(T), so
// theta diverges. It must be reported as absent rather than as a large number.
TEST(BlackScholesDegenerateGreeksTest, ThetaIsUndefinedAtExpiryWithPositiveVolatility) {
    const KinkCase c = make_kink(0.05, 0.05, 0.0, 0.20);

    const auto g = BlackScholesAnalyticEngine::greeks(c.market, c.option, c.model);
    ASSERT_TRUE(g.ok()) << g.error().describe();
    EXPECT_FALSE(g.value().theta.has_value());
    EXPECT_FALSE(g.value().gamma.has_value());
    // vega is zero at expiry: the payoff does not depend on volatility.
    ASSERT_TRUE(g.value().vega.has_value());
    EXPECT_DOUBLE_EQ(*g.value().vega, 0.0);
}

}  // namespace
}  // namespace diffusionworks
