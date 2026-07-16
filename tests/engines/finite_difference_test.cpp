#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>
#include <diffusionworks/pde/boundary_conditions.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

struct Case {
    const char* name;
    OptionType type;
    double spot;
    double strike;
    double rate;
    double dividend;
    double volatility;
    double maturity;
};

std::vector<Case> reference_cases() {
    return {
        {"atm_call", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"itm_call", OptionType::Call, 120.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"otm_call", OptionType::Call, 80.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"atm_put", OptionType::Put, 100.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"itm_put", OptionType::Put, 80.0, 100.0, 0.05, 0.00, 0.20, 1.0},
        {"with_dividend", OptionType::Call, 100.0, 100.0, 0.05, 0.03, 0.25, 2.0},
        {"high_volatility", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.50, 1.0},
        {"long_maturity", OptionType::Call, 100.0, 100.0, 0.03, 0.00, 0.20, 3.0},
    };
}

double analytic(const Case& c) {
    const auto market = MarketState::create(c.spot, c.rate, c.dividend).value();
    const auto option = EuropeanOption::create(c.type, c.strike, c.maturity).value();
    const auto model = BlackScholesModel::create(c.volatility).value();
    return BlackScholesAnalyticEngine::price(market, option, model).value().value;
}

Result<PricingResult> solve(const Case& c, const PdeConfig& config) {
    const auto market = MarketState::create(c.spot, c.rate, c.dividend).value();
    const auto option = EuropeanOption::create(c.type, c.strike, c.maturity).value();
    const auto model = BlackScholesModel::create(c.volatility).value();
    return FiniteDifferenceEngine::price(market, option, model, config);
}

// ---------------------------------------------------------------------------
// Against the analytic reference
//
// The Phase 1 engine is validated against Hull, Haug, a 50-digit mpmath oracle,
// and QuantLib, so agreement here ties the PDE to that chain rather than to a
// number this file invented.
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceTest, ImplicitConvergesTowardBlackScholes) {
    for (const auto& c : reference_cases()) {
        PdeConfig config;
        config.asset_nodes = 801;
        config.time_steps = 2000;
        config.scheme = PdeScheme::Implicit;

        const auto priced = solve(c, config);
        ASSERT_TRUE(priced.ok()) << c.name << ": " << priced.error().describe();
        EXPECT_NEAR(priced.value().value, analytic(c), 5e-3 * std::max(1.0, analytic(c))) << c.name;
    }
}

TEST(FiniteDifferenceTest, CrankNicolsonConvergesTowardBlackScholes) {
    for (const auto& c : reference_cases()) {
        PdeConfig config;
        config.asset_nodes = 801;
        config.time_steps = 400;
        config.scheme = PdeScheme::CrankNicolson;

        const auto priced = solve(c, config);
        ASSERT_TRUE(priced.ok()) << c.name << ": " << priced.error().describe();
        EXPECT_NEAR(priced.value().value, analytic(c), 5e-3 * std::max(1.0, analytic(c))) << c.name;
    }
}

TEST(FiniteDifferenceTest, ExplicitConvergesTowardBlackScholesWithinItsStabilityBound) {
    for (const auto& c : reference_cases()) {
        PdeConfig config;
        config.asset_nodes = 201;
        // dtau <= 1/(sigma^2 N^2 + r) scales as 1/N^2, so the step count must
        // scale as N^2. This is the explicit scheme's whole problem, stated as a
        // number: 200 nodes needs tens of thousands of steps.
        config.time_steps = 30000;
        config.scheme = PdeScheme::Explicit;

        const auto priced = solve(c, config);
        ASSERT_TRUE(priced.ok()) << c.name << ": " << priced.error().describe();
        EXPECT_NEAR(priced.value().value, analytic(c), 2e-2 * std::max(1.0, analytic(c))) << c.name;
    }
}

// Put-call parity is an arbitrage identity, independent of the model: the PDE must
// satisfy it whatever its discretisation error, because both sides carry the same
// error. Checking it catches a boundary condition applied to the wrong option type
// -- an error that leaves each price individually plausible.
TEST(FiniteDifferenceTest, SatisfiesPutCallParity) {
    const double spot = 100.0;
    const double strike = 100.0;
    const double rate = 0.05;
    const double dividend = 0.03;
    const double maturity = 1.0;

    const auto market = MarketState::create(spot, rate, dividend).value();
    const auto model = BlackScholesModel::create(0.25).value();

    PdeConfig config;
    config.asset_nodes = 801;
    config.time_steps = 400;
    config.scheme = PdeScheme::CrankNicolson;

    const auto call = FiniteDifferenceEngine::price(
        market, EuropeanOption::create(OptionType::Call, strike, maturity).value(), model, config);
    const auto put = FiniteDifferenceEngine::price(
        market, EuropeanOption::create(OptionType::Put, strike, maturity).value(), model, config);
    ASSERT_TRUE(call.ok());
    ASSERT_TRUE(put.ok());

    // C - P = S e^{-qT} - K e^{-rT}
    const double parity =
        spot * std::exp(-dividend * maturity) - strike * std::exp(-rate * maturity);
    EXPECT_NEAR(call.value().value - put.value().value, parity, 1e-3);
}

// ---------------------------------------------------------------------------
// Instability: demonstrated, preserved, detected
// ---------------------------------------------------------------------------

// The explicit scheme beyond its bound does not fail gracefully -- it diverges by
// orders of magnitude. This is evidence, not a bug, and the test pins the
// behaviour rather than avoiding the parameters that produce it.
TEST(FiniteDifferenceTest, ExplicitDivergesBeyondItsStabilityBound) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    PdeConfig config;
    config.asset_nodes = 101;
    config.time_steps = 200;  // ratio ~1.96, well beyond the bound
    config.scheme = PdeScheme::Explicit;

    const auto priced = FiniteDifferenceEngine::price(market, option, model, config);

    // Either it went non-finite -- reported as a failure -- or it returned a finite
    // number that is wildly wrong and carries a warning. Both are acceptable; a
    // *silent plausible answer* is not.
    if (priced.ok()) {
        const double exact = BlackScholesAnalyticEngine::price(market, option, model).value().value;
        EXPECT_GT(std::abs(priced.value().value - exact), 1.0)
            << "the scheme ran beyond its stability bound and returned " << priced.value().value
            << ", which is close to the true " << exact
            << " -- if that is genuinely stable, the bound is wrong";
        EXPECT_TRUE(priced.value().has_warnings())
            << "an unstable run must not be reported without a warning";
    }
}

// The stability ratio is reported for every scheme and is the quantity that
// predicts the divergence above.
TEST(FiniteDifferenceTest, ReportsTheStabilityRatioAndWarnsWhenExceeded) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    const auto ratio_of = [&](std::int64_t steps) {
        PdeConfig config;
        config.asset_nodes = 101;
        config.time_steps = steps;
        config.scheme = PdeScheme::Explicit;
        const auto s = FiniteDifferenceEngine::solve(market, option, model, config);
        EXPECT_TRUE(s.ok()) << s.error().describe();
        return s.value().diagnostics;
    };

    const auto stable = ratio_of(2000);
    EXPECT_LT(stable.explicit_stability_ratio, 1.0);
    ASSERT_TRUE(stable.explicit_stable.has_value());
    EXPECT_TRUE(*stable.explicit_stable);

    const auto unstable = ratio_of(200);
    EXPECT_GT(unstable.explicit_stability_ratio, 1.0);
    ASSERT_TRUE(unstable.explicit_stable.has_value());
    EXPECT_FALSE(*unstable.explicit_stable);
}

// Regression: the stability bound is sufficient, not necessary.
//
// EXP-06 initially asserted that predicted and observed stability coincide, and
// failed: the scheme is stable at every ratio up to 1.60 and diverges from 1.70,
// so the bound is conservative by roughly 1.65x. The bound was not wrong -- the
// assertion was. max|b_i| is a Gershgorin-style row bound, and Gershgorin discs
// contain the spectrum while being strictly larger than it, so a limit built from
// them is smaller than the true one *by construction*.
//
// This is the same error as requiring diagonal dominance of the Thomas algorithm:
// mistaking a sufficient condition for a necessary one. The property that must
// hold is one-directional, and that is what this pins.
TEST(FiniteDifferenceTest, StabilityBoundIsSufficientNotNecessary) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const double exact = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    const auto price_at_ratio = [&](double ratio) {
        // Calibrate the step count to the target ratio from a probe run.
        PdeConfig probe;
        probe.asset_nodes = 101;
        probe.time_steps = 1000;
        probe.scheme = PdeScheme::Explicit;
        const auto s = FiniteDifferenceEngine::solve(market, option, model, probe);
        EXPECT_TRUE(s.ok());
        const double dtau_max = (1.0 / 1000.0) / s.value().diagnostics.explicit_stability_ratio;

        PdeConfig config = probe;
        config.time_steps = std::llround(1.0 / (ratio * dtau_max));
        return FiniteDifferenceEngine::price(market, option, model, config);
    };

    // The guarantee: at or below the bound, stable. This is the only direction the
    // bound claims, and the only one that must hold.
    for (const double ratio : {0.25, 0.5, 0.9, 0.99}) {
        const auto priced = price_at_ratio(ratio);
        ASSERT_TRUE(priced.ok()) << "ratio " << ratio << " is within the bound and must not "
                                 << "diverge: " << priced.error().describe();
        EXPECT_LT(std::abs(priced.value().value - exact), 1.0)
            << "ratio " << ratio << " is within the bound but the answer is far from the truth; "
            << "the bound's guarantee is broken";
    }

    // Above the bound the scheme is observed to survive for a while. That is the
    // bound being conservative, not a defect -- and asserting instability here
    // would be asserting that a sufficient condition is necessary.
    const auto just_above = price_at_ratio(1.5);
    ASSERT_TRUE(just_above.ok());
    EXPECT_LT(std::abs(just_above.value().value - exact), 1.0)
        << "at ratio 1.5 the scheme is measured to be stable; if this now diverges the bound's "
        << "conservatism has changed and the documented 1.65x margin is stale";

    // Far enough above, it does diverge -- catastrophically.
    const auto far_above = price_at_ratio(2.0);
    if (far_above.ok()) {
        EXPECT_GT(std::abs(far_above.value().value - exact), 1e3)
            << "ratio 2.0 must diverge; it returned " << far_above.value().value;
        EXPECT_TRUE(far_above.value().has_warnings());
    }
}

// The implicit and Crank-Nicolson schemes have no stability bound to satisfy, so
// reporting explicit_stable for them would claim they passed a test they never
// took.
TEST(FiniteDifferenceTest, DoesNotClaimStabilityForSchemesThatWereNotTested) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    for (const auto scheme : {PdeScheme::Implicit, PdeScheme::CrankNicolson}) {
        PdeConfig config;
        config.scheme = scheme;
        const auto s = FiniteDifferenceEngine::solve(market, option, model, config);
        ASSERT_TRUE(s.ok());
        EXPECT_FALSE(s.value().diagnostics.explicit_stable.has_value()) << to_string(scheme);

        // But the ratio is still reported, because how far beyond the explicit
        // limit these schemes run is exactly what their unconditional stability
        // buys.
        EXPECT_GT(s.value().diagnostics.explicit_stability_ratio, 0.0) << to_string(scheme);
    }
}

// Unconditional stability is not accuracy. The implicit scheme at one enormous
// time step returns a finite, smooth, non-oscillating answer that is badly wrong --
// which is the more dangerous failure, because nothing about it looks broken.
TEST(FiniteDifferenceTest, StabilityIsNotAccuracy) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const double exact = BlackScholesAnalyticEngine::price(market, option, model).value().value;

    PdeConfig config;
    config.asset_nodes = 201;
    config.time_steps = 1;  // the entire year in one step
    config.scheme = PdeScheme::Implicit;

    const auto priced = FiniteDifferenceEngine::price(market, option, model, config);
    ASSERT_TRUE(priced.ok()) << "the implicit scheme is unconditionally stable, so this must not "
                                "diverge: "
                             << priced.error().describe();

    EXPECT_TRUE(std::isfinite(priced.value().value));
    EXPECT_GT(std::abs(priced.value().value - exact), 0.1)
        << "one step over a whole year returned " << priced.value().value << " against a true "
        << exact << ". If that is accurate, this test's premise is wrong.";
}

// ---------------------------------------------------------------------------
// Structure the solution must have
// ---------------------------------------------------------------------------

// A vanilla payoff is non-negative and so is its price. A negative value anywhere
// on the grid is oscillation, not rounding.
TEST(FiniteDifferenceTest, SolutionStaysNonNegative) {
    for (const auto& c : reference_cases()) {
        for (const auto scheme : {PdeScheme::Implicit, PdeScheme::CrankNicolson}) {
            const auto market = MarketState::create(c.spot, c.rate, c.dividend).value();
            const auto option = EuropeanOption::create(c.type, c.strike, c.maturity).value();
            const auto model = BlackScholesModel::create(c.volatility).value();

            PdeConfig config;
            config.asset_nodes = 401;
            config.time_steps = 200;
            config.scheme = scheme;

            const auto s = FiniteDifferenceEngine::solve(market, option, model, config);
            ASSERT_TRUE(s.ok()) << c.name << ": " << s.error().describe();

            for (std::size_t i = 0; i < s.value().values.size(); ++i) {
                EXPECT_GE(s.value().values[i], -1e-10)
                    << c.name << " / " << to_string(scheme) << " at node " << i;
            }
        }
    }
}

// The value at expiry is the payoff, exactly. This checks the terminal condition
// is what it claims and, at one step, that the scheme has not corrupted it.
TEST(FiniteDifferenceTest, TerminalConditionIsThePayoff) {
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const auto option = EuropeanOption::create(type, 100.0, 1.0).value();
        const auto grid = AssetGrid::uniform(400.0, 41).value();

        const auto terminal = terminal_condition(option, grid);
        ASSERT_TRUE(terminal.ok());

        for (std::int64_t i = 0; i < grid.nodes(); ++i) {
            EXPECT_DOUBLE_EQ(terminal.value()[static_cast<std::size_t>(i)],
                             option.payoff(grid.at(i)))
                << "at node " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Rannacher smoothing: explicit, not silent
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceTest, RannacherIsOffByDefaultAndRecordedWhenOn) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    PdeConfig plain;
    plain.scheme = PdeScheme::CrankNicolson;
    const auto without = FiniteDifferenceEngine::solve(market, option, model, plain);
    ASSERT_TRUE(without.ok());
    EXPECT_EQ(without.value().diagnostics.rannacher_steps, 0)
        << "a Crank-Nicolson run must be Crank-Nicolson unless asked otherwise";

    PdeConfig smoothed = plain;
    smoothed.rannacher.count = 2;
    const auto with = FiniteDifferenceEngine::solve(market, option, model, smoothed);
    ASSERT_TRUE(with.ok());
    EXPECT_EQ(with.value().diagnostics.rannacher_steps, 2)
        << "the record must say the scheme was modified";

    // And it must actually change the answer, or the option does nothing.
    EXPECT_NE(without.value().values, with.value().values);
}

// Asking for Rannacher on a scheme it does not apply to is a misunderstanding, and
// silently dropping it would leave the caller believing they got something else.
TEST(FiniteDifferenceTest, RefusesRannacherOnSchemesItDoesNotApplyTo) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    for (const auto scheme : {PdeScheme::Explicit, PdeScheme::Implicit}) {
        PdeConfig config;
        config.scheme = scheme;
        config.rannacher.count = 2;

        const auto s = FiniteDifferenceEngine::solve(market, option, model, config);
        ASSERT_FALSE(s.ok()) << to_string(scheme);
        EXPECT_EQ(s.error().code, ErrorCode::UnsupportedCombination);
    }
}

TEST(FiniteDifferenceTest, RefusesMoreRannacherStepsThanTheRunHas) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    PdeConfig config;
    config.scheme = PdeScheme::CrankNicolson;
    config.time_steps = 5;
    config.rannacher.count = 10;

    EXPECT_FALSE(FiniteDifferenceEngine::solve(market, option, model, config).ok());
}

// ---------------------------------------------------------------------------
// Rejection
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceTest, RefusesAZeroMaturity) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 0.0).value();

    const auto priced = FiniteDifferenceEngine::price(market, option, model, PdeConfig{});
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

// A spot beyond the grid was never solved for. Refused rather than answered about
// the boundary instead.
TEST(FiniteDifferenceTest, RefusesASpotOutsideTheGrid) {
    const auto market = MarketState::create(1000.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.2).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();

    PdeConfig config;
    config.s_max = 400.0;  // the spot is beyond it

    const auto priced = FiniteDifferenceEngine::price(market, option, model, config);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks
