#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/geometric_asian_analytic.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260715;

struct Scenario {
    const char* name;
    OptionType type;
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
    std::int64_t monitoring;
};

std::vector<Scenario> scenarios() {
    return {
        {"at_the_money_call", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.30, 1.0, 12},
        {"in_the_money_call", OptionType::Call, 120.0, 100.0, 0.05, 0.00, 0.30, 1.0, 12},
        {"out_of_the_money_call", OptionType::Call, 80.0, 100.0, 0.05, 0.00, 0.30, 1.0, 12},
        {"at_the_money_put", OptionType::Put, 100.0, 100.0, 0.05, 0.00, 0.30, 1.0, 12},
        {"with_dividend", OptionType::Call, 100.0, 100.0, 0.05, 0.04, 0.25, 2.0, 4},
        {"daily_monitoring", OptionType::Call, 100.0, 100.0, 0.03, 0.00, 0.20, 1.0, 252},
        {"single_monitoring", OptionType::Call, 100.0, 100.0, 0.05, 0.00, 0.30, 1.0, 1},
    };
}

double price_of(const Scenario& s) {
    const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield).value();
    const auto option =
        AsianOption::create(s.type, AveragingType::Geometric, s.strike, s.maturity, s.monitoring)
            .value();
    const auto model = BlackScholesModel::create(s.volatility).value();

    const auto priced = GeometricAsianAnalyticEngine::price(market, option, model);
    EXPECT_TRUE(priced.ok()) << priced.error().describe();
    return priced.value().value;
}

// ---------------------------------------------------------------------------
// The limit that validates the formula
//
// With a single monitoring date at maturity the geometric average *is* S_T, so
// the price must equal Black-Scholes exactly. This is the sharpest check
// available: it ties Kemna-Vorst to the Phase 1 engine, which is itself validated
// against Hull, Haug, a 50-digit mpmath oracle, and QuantLib. An error anywhere
// in the drift adjustment or the variance sum breaks it.
// ---------------------------------------------------------------------------

TEST(GeometricAsianTest, SingleMonitoringDateReducesToBlackScholesExactly) {
    for (const auto& s : scenarios()) {
        const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield).value();
        const auto model = BlackScholesModel::create(s.volatility).value();

        const auto geometric =
            AsianOption::create(s.type, AveragingType::Geometric, s.strike, s.maturity, 1).value();
        const auto european = EuropeanOption::create(s.type, s.strike, s.maturity).value();

        const auto asian = GeometricAsianAnalyticEngine::price(market, geometric, model);
        const auto black_scholes = BlackScholesAnalyticEngine::price(market, european, model);
        ASSERT_TRUE(asian.ok()) << asian.error().describe();
        ASSERT_TRUE(black_scholes.ok());

        // Not "close": the two compute the same quantity by different routes and
        // must agree to rounding.
        EXPECT_NEAR(asian.value().value,
                    black_scholes.value().value,
                    1e-12 * std::max(1.0, black_scholes.value().value))
            << s.name;
    }
}

// At M = 1 the law must be exactly the Black-Scholes terminal law.
TEST(GeometricAsianTest, SingleMonitoringLawIsTheTerminalLaw) {
    const auto market = MarketState::create(100.0, 0.05, 0.02).value();
    const auto model = BlackScholesModel::create(0.3).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 2.0, 1).value();

    const GeometricAverageLaw law = GeometricAsianAnalyticEngine::law(market, option, model);

    // ln S_T ~ N(ln S_0 + (r - q - sigma^2/2)T, sigma^2 T)
    EXPECT_NEAR(law.log_mean, std::log(100.0) + (0.03 - 0.045) * 2.0, 1e-14);
    EXPECT_NEAR(law.log_variance, 0.09 * 2.0, 1e-14);
}

// As monitoring grows the discrete law approaches the continuous one, whose
// variance is the classical sigma^2 T/3.
TEST(GeometricAsianTest, DenseMonitoringApproachesTheContinuousLimit) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();

    const double continuous_variance = 0.09 * 1.0 / 3.0;

    double previous = 1.0;
    for (const std::int64_t monitoring : {2, 12, 100, 10000}) {
        const auto option =
            AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, monitoring)
                .value();
        const double variance =
            GeometricAsianAnalyticEngine::law(market, option, model).log_variance;

        EXPECT_LT(variance, previous) << "variance must fall as monitoring densifies";
        EXPECT_GT(variance, continuous_variance) << "and approach the limit from above";
        previous = variance;
    }

    const auto dense =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 100000).value();
    EXPECT_NEAR(GeometricAsianAnalyticEngine::law(market, dense, model).log_variance,
                continuous_variance,
                1e-5);

    // The drift adjustment tends to T/2: the average sees the drift for about
    // half the option's life, which is why an Asian is cheaper than a European.
    EXPECT_NEAR(GeometricAsianAnalyticEngine::law(market, dense, model).log_mean,
                std::log(100.0) + (0.05 - 0.045) * 0.5,
                1e-4);
}

// ---------------------------------------------------------------------------
// Independent confirmation by simulation
// ---------------------------------------------------------------------------

// The closed form and the simulation reach the same number by unrelated routes:
// one integrates the lognormal law, the other averages sampled paths. Agreement
// is real evidence rather than a self-check.
TEST(GeometricAsianTest, SimulationAgreesWithTheClosedForm) {
    for (const auto& s : scenarios()) {
        const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield).value();
        const auto option =
            AsianOption::create(
                s.type, AveragingType::Geometric, s.strike, s.maturity, s.monitoring)
                .value();
        const auto model = BlackScholesModel::create(s.volatility).value();

        const auto analytic = GeometricAsianAnalyticEngine::price(market, option, model);
        ASSERT_TRUE(analytic.ok()) << analytic.error().describe();

        MonteCarloConfig config;
        config.paths = 200000;
        config.steps = s.monitoring;
        config.seed = kSeed;
        config.scheme = DiscretizationScheme::Exact;

        const auto simulated = MonteCarloEngine::price(market, option, model, config);
        ASSERT_TRUE(simulated.ok()) << simulated.error().describe();

        EXPECT_TRUE(simulated.value().confidence_interval->contains(analytic.value().value))
            << s.name << ": the 95% interval [" << simulated.value().confidence_interval->lower
            << ", " << simulated.value().confidence_interval->upper << "] excludes the closed form "
            << analytic.value().value;
    }
}

// ---------------------------------------------------------------------------
// Structure and rejection
// ---------------------------------------------------------------------------

// Denser monitoring lowers the average's variance, so a call on it is worth less.
TEST(GeometricAsianTest, PriceFallsAsMonitoringDensifies) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();

    double previous = std::numeric_limits<double>::infinity();
    for (const std::int64_t monitoring : {1, 2, 4, 12, 52, 252}) {
        const auto option =
            AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, monitoring)
                .value();
        const double price =
            GeometricAsianAnalyticEngine::price(market, option, model).value().value;

        EXPECT_LT(price, previous) << "at monitoring = " << monitoring;
        previous = price;
    }
}

TEST(GeometricAsianTest, PricesAreNonNegativeAndBounded) {
    for (const auto& s : scenarios()) {
        const double price = price_of(s);
        EXPECT_GE(price, 0.0) << s.name;
        EXPECT_TRUE(std::isfinite(price)) << s.name;
        EXPECT_LT(price, s.spot + s.strike) << s.name;
    }
}

// Pricing an arithmetic option with this engine would answer a different question
// with a plausible number, so it is refused.
TEST(GeometricAsianTest, RefusesArithmeticAveraging) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();
    const auto arithmetic =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();

    const auto priced = GeometricAsianAnalyticEngine::price(market, arithmetic, model);
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

// Zero volatility leaves the average deterministic. The price is exact, not
// approximate, and is returned with a warning rather than by dividing by a zero
// standard deviation.
TEST(GeometricAsianTest, ZeroVolatilityIsHandledExactly) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 90.0, 1.0, 4).value();

    const auto priced = GeometricAsianAnalyticEngine::price(market, option, model);
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_TRUE(priced.value().has_warnings());
    EXPECT_TRUE(std::isfinite(priced.value().value));

    // With no diffusion the average is exp(ln S_0 + r * T(M+1)/(2M)), known
    // exactly.
    const double average_time = 1.0 * (4.0 + 1.0) / (2.0 * 4.0);
    const double certain_average = 100.0 * std::exp(0.05 * average_time);
    EXPECT_NEAR(
        priced.value().value, std::exp(-0.05) * std::max(certain_average - 90.0, 0.0), 1e-12);
}

TEST(GeometricAsianTest, ReportsItsLawAsDiagnostics) {
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto model = BlackScholesModel::create(0.3).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Geometric, 100.0, 1.0, 12).value();

    const auto priced = GeometricAsianAnalyticEngine::price(market, option, model);
    ASSERT_TRUE(priced.ok());

    const auto has = [&](const std::string& name) {
        for (const Diagnostic& d : priced.value().diagnostics) {
            if (d.name == name) {
                return true;
            }
        }
        return false;
    };
    for (const std::string name : {"d1", "d2", "log_mean", "log_variance", "average_forward"}) {
        EXPECT_TRUE(has(name)) << "missing diagnostic: " << name;
    }
}

}  // namespace
}  // namespace diffusionworks
