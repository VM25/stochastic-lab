#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>

namespace diffusionworks {
namespace {

MarketState market(double spot = 100.0, double rate = 0.0, double dividend = 0.0) {
    return MarketState::create(spot, rate, dividend).value();
}

EuropeanOption call(double strike = 100.0, double maturity = 1.0) {
    return EuropeanOption::create(OptionType::Call, strike, maturity).value();
}

HestonModel heston(double v0, double kappa, double theta, double xi, double rho) {
    return HestonModel::create(v0, kappa, theta, xi, rho).value();
}

double diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& d : result.diagnostics) {
        if (d.name == name) {
            return std::get<double>(d.value);
        }
    }
    ADD_FAILURE() << "no double diagnostic named " << name;
    return 0.0;
}

std::int64_t int_diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& d : result.diagnostics) {
        if (d.name == name) {
            return std::get<std::int64_t>(d.value);
        }
    }
    ADD_FAILURE() << "no integer diagnostic named " << name;
    return 0;
}

bool bool_diagnostic(const PricingResult& result, const std::string& name) {
    for (const Diagnostic& d : result.diagnostics) {
        if (d.name == name) {
            return std::get<bool>(d.value);
        }
    }
    ADD_FAILURE() << "no boolean diagnostic named " << name;
    return false;
}

// ---------------------------------------------------------------------------
// Against independent references
//
// 5.78515543437619 is the Fang-Oosterlee (2008) COS-method benchmark for
// S=K=100, T=1, r=q=0, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711,
// v0=0.0175. It agrees with a 40-digit mpmath integration of the little-trap
// formula and with QuantLib's AnalyticHestonEngine, both to 1e-13 -- so the
// constant is a published value confirmed by two routes sharing no code with this
// engine, not a transcription of the same integral checked against itself.
// ---------------------------------------------------------------------------

TEST(HestonAnalyticTest, MatchesTheFangOosterleeBenchmark) {
    const auto priced = HestonAnalyticEngine::price(
        market(), call(), heston(0.0175, 1.5768, 0.0398, 0.5751, -0.5711));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_NEAR(priced.value().value, 5.78515543437619, 1e-10);
}

// A Feller-violating regime, priced correctly. The condition failing is a property
// of the variance process, not of the pricing integral, which is well defined
// regardless -- so the price must still match the reference (13.1365327960895, from
// mpmath and QuantLib) and the violation must be reported, not cause a refusal.
TEST(HestonAnalyticTest, PricesAFellerViolatingRegimeAndReportsIt) {
    const auto model = heston(0.09, 2.0, 0.09, 1.0, -0.3);
    ASSERT_FALSE(model.satisfies_feller());  // 2*2*0.09 = 0.36 < xi^2 = 1

    const auto priced = HestonAnalyticEngine::price(market(100.0, 0.05, 0.0), call(), model);
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_NEAR(priced.value().value, 13.1365327960895, 1e-9);

    EXPECT_FALSE(bool_diagnostic(priced.value(), "satisfies_feller"));
    EXPECT_LT(diagnostic(priced.value(), "feller_ratio"), 1.0);
    EXPECT_TRUE(priced.value().has_warnings())
        << "a Feller violation must be surfaced to the reader";
}

// The branch-cut trap. Heston's original characteristic function crosses a branch
// cut as maturity grows and returns a discontinuous wrong price; the little-trap
// formulation this engine uses does not. Checked against long-maturity mpmath
// references, which is exactly where the naive form fails.
TEST(HestonAnalyticTest, StaysCorrectAtLongMaturityWhereTheTrapBites) {
    const auto model = heston(0.04, 1.5, 0.04, 0.6, -0.7);
    const auto mk = market(100.0, 0.03, 0.0);

    struct Case {
        double maturity;
        double reference;
    };

    for (const Case& c : {Case{5.0, 23.50614713998921},
                          Case{10.0, 36.44207657516968},
                          Case{15.0, 46.43818472722808}}) {
        const auto priced = HestonAnalyticEngine::price(mk, call(100.0, c.maturity), model);
        ASSERT_TRUE(priced.ok()) << "T=" << c.maturity << ": " << priced.error().describe();
        EXPECT_NEAR(priced.value().value, c.reference, 1e-8) << "T=" << c.maturity;
    }
}

// The deterministic-variance limit. With a tiny vol-of-variance and v0 = theta, the
// variance barely moves from theta, so the Heston price approaches Black-Scholes at
// volatility sqrt(theta). Not exact -- xi is small but not zero -- so a loose
// tolerance, but a wrong sign or a factor error would break it.
TEST(HestonAnalyticTest, ApproachesBlackScholesAsVolOfVarianceVanishes) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call(100.0, 1.0);
    const double theta = 0.04;  // so sqrt(theta) = 0.2

    const auto heston_price =
        HestonAnalyticEngine::price(mk, option, heston(theta, 2.0, theta, 1e-3, 0.0));
    ASSERT_TRUE(heston_price.ok()) << heston_price.error().describe();

    const double bs = BlackScholesAnalyticEngine::price(
                          mk, option, BlackScholesModel::create(std::sqrt(theta)).value())
                          .value()
                          .value;
    EXPECT_NEAR(heston_price.value().value, bs, 1e-4);
}

// ---------------------------------------------------------------------------
// Put-call parity
// ---------------------------------------------------------------------------

// The put is priced from the call by parity, so parity holds exactly rather than to
// a tolerance. Checked across regimes to exercise the discounting.
TEST(HestonAnalyticTest, PutCallParityHoldsExactly) {
    const auto model = heston(0.04, 2.0, 0.04, 0.5, -0.6);
    for (const double spot : {80.0, 100.0, 120.0}) {
        for (const double rate : {0.0, 0.05}) {
            for (const double dividend : {0.0, 0.03}) {
                const auto mk = market(spot, rate, dividend);
                const auto priced_call = HestonAnalyticEngine::price(mk, call(100.0, 1.0), model);
                const auto priced_put = HestonAnalyticEngine::price(
                    mk, EuropeanOption::create(OptionType::Put, 100.0, 1.0).value(), model);
                ASSERT_TRUE(priced_call.ok());
                ASSERT_TRUE(priced_put.ok());

                // C - P = S e^{-qT} - K e^{-rT}.
                const double parity = spot * std::exp(-dividend) - 100.0 * std::exp(-rate);
                EXPECT_NEAR(priced_call.value().value - priced_put.value().value, parity, 1e-10)
                    << "S=" << spot << " r=" << rate << " q=" << dividend;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structural sanity
// ---------------------------------------------------------------------------

// A call is worth between its discounted intrinsic forward value and the discounted
// spot -- no-arbitrage bounds that hold for any model.
TEST(HestonAnalyticTest, RespectsNoArbitrageBounds) {
    const auto model = heston(0.05, 1.0, 0.05, 0.4, -0.5);
    for (const double spot : {60.0, 100.0, 160.0}) {
        const auto mk = market(spot, 0.04, 0.0);
        const auto priced = HestonAnalyticEngine::price(mk, call(100.0, 1.0), model);
        ASSERT_TRUE(priced.ok()) << "S=" << spot << ": " << priced.error().describe();

        const double value = priced.value().value;
        const double lower = std::max(0.0, spot - 100.0 * std::exp(-0.04));
        EXPECT_GE(value, lower - 1e-9) << "S=" << spot;
        EXPECT_LE(value, spot + 1e-9) << "S=" << spot;
    }
}

// The call rises with spot: a structural monotonicity any correct price obeys.
TEST(HestonAnalyticTest, CallIncreasesWithSpot) {
    const auto model = heston(0.04, 2.0, 0.04, 0.5, -0.6);
    double previous = -1.0;
    for (const double spot : {70.0, 85.0, 100.0, 115.0, 130.0}) {
        const auto priced = HestonAnalyticEngine::price(market(spot), call(), model);
        ASSERT_TRUE(priced.ok());
        EXPECT_GT(priced.value().value, previous) << "S=" << spot;
        previous = priced.value().value;
    }
}

// ---------------------------------------------------------------------------
// Integration convergence and failure
// ---------------------------------------------------------------------------

// The doubling check is the guard against an unconverged integral posing as a price.
// At a comfortable node count the integration error is far below the tolerance and
// the diagnostic records it.
TEST(HestonAnalyticTest, ReportsAConvergedIntegrationError) {
    const auto priced = HestonAnalyticEngine::price(
        market(), call(), heston(0.0175, 1.5768, 0.0398, 0.5751, -0.5711));
    ASSERT_TRUE(priced.ok());
    EXPECT_LT(diagnostic(priced.value(), "integration_error"), 1e-8);
    EXPECT_EQ(int_diagnostic(priced.value(), "quadrature_nodes"), 1024);
}

// The pathological corner: very short maturity with large vol-of-variance. The
// integrand's tail decays too slowly to resolve at any practical node count, so the
// engine must refuse rather than return the smooth wrong number a starved quadrature
// produces. This is "failures never return plausible values" made concrete.
TEST(HestonAnalyticTest, RefusesTheUnresolvablePathologicalRegime) {
    const auto priced = HestonAnalyticEngine::price(
        market(100.0, 0.05, 0.0), call(150.0, 0.02), heston(0.04, 1.0, 0.04, 2.0, -0.5));
    ASSERT_FALSE(priced.ok())
        << "an integral that has not converged must not be returned as a price";
    EXPECT_TRUE(priced.error().code == ErrorCode::ConvergenceFailure ||
                priced.error().code == ErrorCode::NonFiniteValue);
}

// A merely hard regime -- deep out of the money at quarter-year maturity -- converges
// as the quadrature refines. The integration error must fall monotonically with the
// node count: that proves a refusal elsewhere is a resolution limit rather than a
// defect, and that the doubling check measures real convergence. By the default node
// count the same regime has converged to a real price.
TEST(HestonAnalyticTest, IntegrationErrorFallsMonotonicallyWithNodes) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call(130.0, 0.25);
    const auto model = heston(0.04, 1.0, 0.04, 0.8, -0.5);

    double previous = std::numeric_limits<double>::infinity();
    double finest = 0.0;
    for (const std::int64_t nodes : {128, 256, 512, 1024}) {
        HestonAnalyticConfig config;
        config.quadrature_nodes = nodes;
        config.convergence_tolerance = 1.0;  // read the error rather than gate on it
        const auto priced = HestonAnalyticEngine::price(mk, option, model, config);
        ASSERT_TRUE(priced.ok()) << "nodes=" << nodes;
        const double error = diagnostic(priced.value(), "integration_error");
        EXPECT_LT(error, previous) << "error did not fall at nodes=" << nodes;
        previous = error;
        finest = error;
    }
    EXPECT_LT(finest, 1e-8) << "the hard regime did not converge by the default node count";
}

// ---------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------

TEST(HestonAnalyticTest, RefusesZeroVolOfVariance) {
    const auto priced =
        HestonAnalyticEngine::price(market(), call(), heston(0.04, 2.0, 0.04, 0.0, -0.5));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

TEST(HestonAnalyticTest, RefusesZeroMaturity) {
    const auto priced =
        HestonAnalyticEngine::price(market(), call(100.0, 0.0), heston(0.04, 2.0, 0.04, 0.5, -0.5));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

TEST(HestonAnalyticTest, RefusesTooFewQuadratureNodes) {
    HestonAnalyticConfig tiny;
    tiny.quadrature_nodes = 4;
    const auto priced =
        HestonAnalyticEngine::price(market(), call(), heston(0.04, 2.0, 0.04, 0.5, -0.5), tiny);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Model validation and the Feller diagnostic
// ---------------------------------------------------------------------------

TEST(HestonModelTest, ValidatesParameters) {
    EXPECT_TRUE(HestonModel::create(0.04, 2.0, 0.04, 0.5, -0.5).ok());
    // Boundary values are admitted: they are limiting cases, not errors.
    EXPECT_TRUE(HestonModel::create(0.0, 2.0, 0.04, 0.5, 0.0).ok());   // v0 = 0
    EXPECT_TRUE(HestonModel::create(0.04, 2.0, 0.04, 0.5, 1.0).ok());  // rho = 1
    EXPECT_TRUE(HestonModel::create(0.04, 2.0, 0.04, 0.5, -1.0).ok());

    EXPECT_FALSE(HestonModel::create(-0.01, 2.0, 0.04, 0.5, 0.0).ok());  // v0 < 0
    EXPECT_FALSE(HestonModel::create(0.04, -1.0, 0.04, 0.5, 0.0).ok());  // kappa < 0
    EXPECT_FALSE(HestonModel::create(0.04, 2.0, 0.04, 0.5, 1.5).ok());   // |rho| > 1
    EXPECT_FALSE(HestonModel::create(0.04, 2.0, 0.04, 0.5, -1.5).ok());
}

TEST(HestonModelTest, ReportsTheFellerConditionWithoutRejecting) {
    // Satisfied: 2*2*0.09 = 0.36 >= xi^2 = 0.09.
    const auto satisfied = heston(0.09, 2.0, 0.09, 0.3, -0.5);
    EXPECT_TRUE(satisfied.satisfies_feller());
    EXPECT_GE(satisfied.feller_ratio(), 1.0);

    // Violated: the set is still constructed, and reports the violation.
    const auto violated = heston(0.09, 2.0, 0.09, 1.0, -0.5);
    EXPECT_FALSE(violated.satisfies_feller());
    EXPECT_LT(violated.feller_ratio(), 1.0);
}

// ---------------------------------------------------------------------------
// The characteristic function itself
//
// The price is an integral of this object, so validating it directly -- not only
// through the price -- is what EXP-09 rests on. These are properties the function
// must have because it is the characteristic function of a real log-price under a
// risk-neutral measure, checked here as permanent regressions on the engine.
// ---------------------------------------------------------------------------

using Complex = std::complex<double>;

// A spread of regimes, including a Feller violation and a strong correlation.
struct CfRegime {
    double spot, rate, dividend, maturity, v0, kappa, theta, xi, rho;
};

const CfRegime kCfRegimes[] = {
    {100.0, 0.05, 0.02, 1.5, 0.04, 1.5, 0.04, 0.6, -0.7},   // ordinary
    {100.0, 0.03, 0.00, 5.0, 0.09, 2.0, 0.09, 1.0, -0.3},   // Feller-violating (0.36 < 1)
    {80.0, 0.01, 0.04, 0.25, 0.02, 3.0, 0.05, 0.4, 0.5},    // short maturity, positive rho
    {120.0, 0.06, 0.01, 10.0, 0.06, 1.0, 0.05, 0.7, -0.9},  // long maturity, strong rho
};

TEST(HestonCharacteristicFunctionTest, IsOneAtTheOrigin) {
    // phi_j(0) = E^j[1] = 1 for both probability measures.
    for (const CfRegime& r : kCfRegimes) {
        const auto mk = market(r.spot, r.rate, r.dividend);
        const auto model = heston(r.v0, r.kappa, r.theta, r.xi, r.rho);
        for (const int index : {1, 2}) {
            const auto phi = HestonAnalyticEngine::log_price_characteristic_function(
                mk, model, r.maturity, Complex{0.0, 0.0}, index);
            ASSERT_TRUE(phi.ok()) << phi.error().describe();
            EXPECT_NEAR(phi.value().real(), 1.0, 1e-12) << "index " << index;
            EXPECT_NEAR(phi.value().imag(), 0.0, 1e-12) << "index " << index;
        }
    }
}

TEST(HestonCharacteristicFunctionTest, HasConjugateSymmetry) {
    // The log-price is real, so phi_j(-u) = conj(phi_j(u)) for real u.
    for (const CfRegime& r : kCfRegimes) {
        const auto mk = market(r.spot, r.rate, r.dividend);
        const auto model = heston(r.v0, r.kappa, r.theta, r.xi, r.rho);
        for (const int index : {1, 2}) {
            for (const double u : {0.1, 0.7, 2.5, 12.0}) {
                const auto plus = HestonAnalyticEngine::log_price_characteristic_function(
                    mk, model, r.maturity, Complex{u, 0.0}, index);
                const auto minus = HestonAnalyticEngine::log_price_characteristic_function(
                    mk, model, r.maturity, Complex{-u, 0.0}, index);
                ASSERT_TRUE(plus.ok() && minus.ok());
                EXPECT_LT(std::abs(minus.value() - std::conj(plus.value())), 1e-10)
                    << "index " << index << " u " << u;
            }
        }
    }
}

TEST(HestonCharacteristicFunctionTest, SatisfiesTheMartingaleIdentity) {
    // phi_2(-i) = E^Q[e^{ln S_T}] = E^Q[S_T] = S_0 e^{(r-q)T}, the forward. This pins
    // the drift under the engine's exact convention: index 2 is the risk-neutral CF.
    for (const CfRegime& r : kCfRegimes) {
        const auto mk = market(r.spot, r.rate, r.dividend);
        const auto model = heston(r.v0, r.kappa, r.theta, r.xi, r.rho);
        const auto phi = HestonAnalyticEngine::log_price_characteristic_function(
            mk, model, r.maturity, Complex{0.0, -1.0}, 2);
        ASSERT_TRUE(phi.ok()) << phi.error().describe();
        const double forward = r.spot * std::exp((r.rate - r.dividend) * r.maturity);
        EXPECT_NEAR(phi.value().real(), forward, forward * 1e-10);
        EXPECT_NEAR(phi.value().imag(), 0.0, forward * 1e-10);
    }
}

TEST(HestonCharacteristicFunctionTest, IsFiniteAndContinuousAcrossADenseGrid) {
    // Finite everywhere on a dense real-u grid, and continuous. The characteristic
    // function is smooth, but it is not slowly varying: the e^{i u ln S_0} carrier
    // rotates it at frequency ln(S_0) ~ 4-5, so a genuine step over a small du is
    // O(ln(S_0) * du), not zero. A branch-cut crossing -- the "trap" the little-trap
    // form exists to avoid -- would instead be an O(1) discontinuity. The grid is
    // fine enough (du = 0.01) that the smooth rotation stays well under the bound
    // while any real jump would blow through it.
    const double du = 0.01;
    for (const CfRegime& r : kCfRegimes) {
        const auto mk = market(r.spot, r.rate, r.dividend);
        const auto model = heston(r.v0, r.kappa, r.theta, r.xi, r.rho);
        Complex previous{1.0, 0.0};
        for (int k = 0; k <= 2000; ++k) {
            const double u = k * du;
            const auto phi = HestonAnalyticEngine::log_price_characteristic_function(
                mk, model, r.maturity, Complex{u, 0.0}, 2);
            ASSERT_TRUE(phi.ok());
            ASSERT_TRUE(std::isfinite(phi.value().real()) && std::isfinite(phi.value().imag()))
                << "u " << u;
            // |phi| <= 1 for a characteristic function.
            EXPECT_LE(std::abs(phi.value()), 1.0 + 1e-9) << "u " << u;
            if (k > 0) {
                // Smooth rotation is ~ln(S_0)*du ~ 0.05; a discontinuity is O(1).
                EXPECT_LT(std::abs(phi.value() - previous), 0.2) << "jump at u " << u;
            }
            previous = phi.value();
        }
    }
}

TEST(HestonCharacteristicFunctionTest, RefusesDegenerateInputs) {
    const auto mk = market(100.0, 0.05, 0.0);
    // Zero vol-of-variance: the formula divides by xi^2.
    const auto zero_xi = HestonAnalyticEngine::log_price_characteristic_function(
        mk, heston(0.04, 2.0, 0.04, 0.0, -0.5), 1.0, Complex{1.0, 0.0}, 2);
    ASSERT_FALSE(zero_xi.ok());
    EXPECT_EQ(zero_xi.error().code, ErrorCode::UnsupportedCombination);
    // Zero maturity.
    const auto zero_t = HestonAnalyticEngine::log_price_characteristic_function(
        mk, heston(0.04, 2.0, 0.04, 0.5, -0.5), 0.0, Complex{1.0, 0.0}, 2);
    ASSERT_FALSE(zero_t.ok());
    EXPECT_EQ(zero_t.error().code, ErrorCode::UnsupportedCombination);
    // A bad probability index.
    const auto bad_index = HestonAnalyticEngine::log_price_characteristic_function(
        mk, heston(0.04, 2.0, 0.04, 0.5, -0.5), 1.0, Complex{1.0, 0.0}, 3);
    ASSERT_FALSE(bad_index.ok());
    EXPECT_EQ(bad_index.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks
