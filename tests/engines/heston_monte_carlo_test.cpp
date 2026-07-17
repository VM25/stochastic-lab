#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>

#include <gtest/gtest.h>

#include <cmath>
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

HestonMonteCarloConfig
config(std::int64_t steps, std::int64_t paths = 200000, std::uint64_t seed = 20260717) {
    HestonMonteCarloConfig c;
    c.steps = steps;
    c.paths = paths;
    c.seed = seed;
    return c;
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

double
analytic_price(const MarketState& mk, const EuropeanOption& option, const HestonModel& model) {
    return HestonAnalyticEngine::price(mk, option, model).value().value;
}

// A Feller-satisfying regime (2*3*0.04 = 0.24 >= xi^2 = 0.09), where the Euler
// scheme rarely floors the variance and the bias is small -- the clean case for a
// convergence check.
HestonModel benign() {
    return heston(0.04, 3.0, 0.04, 0.3, -0.6);
}

// A Feller-violating regime (2*2*0.04 = 0.16 << xi^2 = 1.0). Here the discretisation
// bias is large -- of order one at a handful of steps -- so it resolves far above the
// sampling noise, which is what a convergence check needs. The full-truncation scheme
// floors a negative pre-truncation variance on a large fraction of steps but never
// produces a non-finite path; the naive scheme is destroyed here.
HestonModel stressed() {
    return heston(0.04, 2.0, 0.04, 1.0, -0.7);
}

// ---------------------------------------------------------------------------
// Convergence to the semi-analytic reference
//
// The semi-analytic engine is the reference (validated against mpmath, QuantLib, and
// the published benchmark). The simulation's job is to approach it as the step
// shrinks. The full-truncation Euler scheme is biased, so the check is that the bias
// falls with the step, and that at a fine step the price agrees to within a few
// across-path standard errors.
// ---------------------------------------------------------------------------

// The convergence check runs in the stressed regime, not the benign one, on purpose.
// The benign bias is smaller than the sampling noise at any affordable path count, so
// a single-seed "finer beats coarser" comparison there would be comparing noise. In
// the stressed regime the bias is of order one at a few steps and roughly halves each
// time the step doubles, so it clears the noise by a wide margin and the monotone
// decrease is a real signal. That the naive scheme cannot even price this regime is
// covered separately.
TEST(HestonMonteCarloTest, BiasFallsMonotonicallyAsTheStepShrinks) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();
    const auto model = stressed();
    const double reference = analytic_price(mk, option, model);

    double previous_bias = std::numeric_limits<double>::infinity();
    for (const std::int64_t steps : {5, 10, 20, 40}) {
        const auto priced = HestonMonteCarloEngine::price(mk, option, model, config(steps));
        ASSERT_TRUE(priced.ok()) << "steps=" << steps << ": " << priced.error().describe();
        const double bias = std::abs(priced.value().value - reference);
        EXPECT_LT(bias, previous_bias)
            << "refining from a coarser step to " << steps << " did not reduce the bias (" << bias
            << " vs previous " << previous_bias << ")";
        previous_bias = bias;
    }
}

TEST(HestonMonteCarloTest, AgreesWithTheSemiAnalyticPriceAtAFineStep) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();
    const auto model = benign();
    const double reference = analytic_price(mk, option, model);

    const auto priced = HestonMonteCarloEngine::price(mk, option, model, config(320, 400000));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    ASSERT_TRUE(priced.value().standard_error.has_value());

    // At 320 steps the Euler bias for this benign regime is small relative to the
    // sampling error, so the price is within a few standard errors of the reference.
    EXPECT_LT(std::abs(priced.value().value - reference), 5.0 * *priced.value().standard_error)
        << "MC " << priced.value().value << " vs analytic " << reference;
}

// Across moneyness, the fine-step simulation stays close to the reference. An engine
// right only at the money is not validated.
TEST(HestonMonteCarloTest, TracksTheReferenceAcrossMoneyness) {
    const auto md = benign();
    for (const double spot : {80.0, 100.0, 120.0}) {
        const auto mk = market(spot, 0.05, 0.0);
        const auto option = call();
        const double reference = analytic_price(mk, option, md);
        const auto priced = HestonMonteCarloEngine::price(mk, option, md, config(320, 400000));
        ASSERT_TRUE(priced.ok()) << "S=" << spot << ": " << priced.error().describe();
        EXPECT_LT(std::abs(priced.value().value - reference), 6.0 * *priced.value().standard_error)
            << "S=" << spot;
    }
}

// ---------------------------------------------------------------------------
// The variance diagnostic
// ---------------------------------------------------------------------------

// The negative pre-truncation frequency is the signal that the discretisation is
// straining. It must be near zero for a comfortably Feller-satisfying regime and
// substantial for a violating one -- the whole reason full truncation exists.
TEST(HestonMonteCarloTest, NegativeVarianceFrequencyRisesWhenFellerFails) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();

    const auto benign_priced = HestonMonteCarloEngine::price(mk, option, benign(), config(100));
    ASSERT_TRUE(benign_priced.ok());
    const double benign_fraction = diagnostic(benign_priced.value(), "negative_variance_fraction");

    const auto violating_priced =
        HestonMonteCarloEngine::price(mk, option, stressed(), config(100));
    ASSERT_TRUE(violating_priced.ok());
    const double violating_fraction =
        diagnostic(violating_priced.value(), "negative_variance_fraction");

    EXPECT_GT(violating_fraction, benign_fraction);
    EXPECT_GT(violating_fraction, 0.0)
        << "a strongly Feller-violating regime must floor the variance sometimes";
    EXPECT_TRUE(violating_priced.value().has_warnings())
        << "a Feller violation with truncation events must be surfaced";
}

// The hard regime is the one that matters for the exit gate's "prices approach the
// reference as resolution improves": at a fine step, full truncation prices the
// Feller-violating regime to within a few standard errors of the semi-analytic
// reference, even though the coarse-step bias there was of order one.
TEST(HestonMonteCarloTest, StressedRegimeReachesTheReferenceAtAFineStep) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();
    const auto model = stressed();
    const double reference = analytic_price(mk, option, model);

    const auto priced = HestonMonteCarloEngine::price(mk, option, model, config(320, 400000));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    ASSERT_TRUE(priced.value().standard_error.has_value());
    EXPECT_LT(std::abs(priced.value().value - reference), 5.0 * *priced.value().standard_error)
        << "stressed MC " << priced.value().value << " vs analytic " << reference;
}

// ---------------------------------------------------------------------------
// The naive scheme: EXP-10's diagnostic baseline
//
// The naive Euler scheme exists to be shown failing. In a regime full truncation
// prices cleanly, the naive scheme takes the square root of a negative variance,
// produces non-finite paths, and is blocked -- so `simulate` reports the failure
// through diagnostics and `price` returns a hard error rather than a laundered
// average of the survivors.
// ---------------------------------------------------------------------------

TEST(HestonMonteCarloTest, NaiveEulerFailsWhereFullTruncationPrices) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();
    const auto model = stressed();

    HestonMonteCarloConfig full = config(80);
    full.scheme = HestonVarianceScheme::FullTruncation;
    HestonMonteCarloConfig naive = config(80);
    naive.scheme = HestonVarianceScheme::NaiveEuler;

    EXPECT_TRUE(HestonMonteCarloEngine::price(mk, option, model, full).ok())
        << "full truncation must price the stressed regime";

    const auto naive_priced = HestonMonteCarloEngine::price(mk, option, model, naive);
    ASSERT_FALSE(naive_priced.ok()) << "the naive scheme must not silently price this regime";
    EXPECT_EQ(naive_priced.error().code, ErrorCode::NonFiniteValue);
}

// A blocked run is still evidence, not an error to swallow: simulate() reports the
// diagnostics of the failure -- the path-failure count -- and leaves the price absent.
TEST(HestonMonteCarloTest, SimulateReportsDiagnosticsOnABlockedNaiveRun) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();

    HestonMonteCarloConfig naive = config(80);
    naive.scheme = HestonVarianceScheme::NaiveEuler;

    const auto outcome = HestonMonteCarloEngine::simulate(mk, option, stressed(), naive);
    ASSERT_TRUE(outcome.ok()) << "a non-finite run is reported, not an error, at this level";
    EXPECT_FALSE(outcome.value().price.has_value());
    EXPECT_GT(outcome.value().diagnostics.non_finite_paths, 0)
        << "the block must be backed by a counted failure";
}

// Full truncation never turns a negative pre-truncation variance into a lost path:
// the minimum pre-truncation variance goes negative in the stressed regime, and yet
// there are zero non-finite paths. That is the property the scheme exists to provide.
TEST(HestonMonteCarloTest, FullTruncationSurvivesNegativeVarianceThatSinksTheNaiveScheme) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto option = call();

    const auto outcome = HestonMonteCarloEngine::simulate(mk, option, stressed(), config(80));
    ASSERT_TRUE(outcome.ok()) << outcome.error().describe();
    ASSERT_TRUE(outcome.value().price.has_value());
    EXPECT_EQ(outcome.value().diagnostics.non_finite_paths, 0);
    EXPECT_LT(outcome.value().diagnostics.minimum_variance, 0.0)
        << "the stressed regime must drive the pre-truncation variance negative";
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

// Put-call parity, under common random numbers so it holds per path: with the same
// seed the call and put see the same terminal spots, so (S_T-K)^+ - (K-S_T)^+ = S_T-K
// exactly, and the difference of the means is the discounted mean of S_T - K. The
// log-Euler spot step keeps the discounted spot a martingale in expectation, so the
// only gap left is the sampling error in that mean.
TEST(HestonMonteCarloTest, PutCallParityHoldsUnderCommonRandomNumbers) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto model = benign();
    const auto put = EuropeanOption::create(OptionType::Put, 100.0, 1.0).value();

    const auto priced_call = HestonMonteCarloEngine::price(mk, call(), model, config(160));
    const auto priced_put = HestonMonteCarloEngine::price(mk, put, model, config(160));
    ASSERT_TRUE(priced_call.ok());
    ASSERT_TRUE(priced_put.ok());

    const double parity = 100.0 * std::exp(0.0) - 100.0 * std::exp(-0.05);  // S e^{-qT} - K e^{-rT}

    // The gate is uncertainty-aware, not a fixed number of cents. The difference is
    // priced with the same seed, so its true standard error is that of the discounted
    // mean of S_T - K, which is not reported directly. But the payoffs (S_T-K)^+ and
    // (K-S_T)^+ are negatively correlated (one is zero wherever the other is
    // positive), so Cauchy-Schwarz bounds SE[C-P] <= SE_call + SE_put -- a reported,
    // rigorous upper bound. Four of those covers ordinary sampling fluctuation.
    const double se_call = *priced_call.value().standard_error;
    const double se_put = *priced_put.value().standard_error;
    EXPECT_NEAR(
        priced_call.value().value - priced_put.value().value, parity, 4.0 * (se_call + se_put));
}

TEST(HestonMonteCarloTest, IsReproducibleUnderTheSameSeed) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto model = benign();
    const auto first = HestonMonteCarloEngine::price(mk, call(), model, config(50, 20000, 7));
    const auto second = HestonMonteCarloEngine::price(mk, call(), model, config(50, 20000, 7));
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(first.value().value, second.value().value);
    EXPECT_EQ(*first.value().standard_error, *second.value().standard_error);
}

TEST(HestonMonteCarloTest, DifferentSeedsGiveDifferentEstimates) {
    const auto mk = market(100.0, 0.05, 0.0);
    const auto model = benign();
    const auto a = HestonMonteCarloEngine::price(mk, call(), model, config(50, 20000, 1));
    const auto b = HestonMonteCarloEngine::price(mk, call(), model, config(50, 20000, 2));
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_NE(a.value().value, b.value().value);
}

// A benign run leaves no non-finite paths, and the diagnostic says so.
TEST(HestonMonteCarloTest, ABenignRunHasNoNonFinitePaths) {
    const auto priced =
        HestonMonteCarloEngine::price(market(100.0, 0.05, 0.0), call(), benign(), config(100));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_EQ(int_diagnostic(priced.value(), "non_finite_paths"), 0);
}

// ---------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------

TEST(HestonMonteCarloTest, RefusesFewerThanTwoPaths) {
    const auto priced = HestonMonteCarloEngine::price(market(), call(), benign(), config(10, 1));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::InvalidArgument);
}

TEST(HestonMonteCarloTest, RefusesZeroMaturity) {
    const auto priced =
        HestonMonteCarloEngine::price(market(), call(100.0, 0.0), benign(), config(10));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
}

}  // namespace
}  // namespace diffusionworks
