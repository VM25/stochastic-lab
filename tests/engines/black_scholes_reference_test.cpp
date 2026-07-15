#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <string>

namespace diffusionworks {
namespace {

/// One published reference case.
///
/// VALIDATION-PLAN section 21 requires a reference-value source map. Rather than
/// keep that map in a separate document that can drift, each case carries its
/// citation, and the test name and failure message quote it.
///
/// `tolerance` is set to half a unit in the last published decimal place. A
/// source quoting 4.76 asserts only that the true value rounds to 4.76, so
/// demanding more than +/-0.005 would be testing the rounding rather than the
/// implementation. Tighter internal agreement is checked separately by the
/// invariant tests, which have no such quantisation.
struct ReferenceCase {
    const char* source;
    OptionType type;
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    double maturity;
    double expected;
    double tolerance;
};

double price_of(const ReferenceCase& c) {
    const auto market = MarketState::create(c.spot, c.rate, c.dividend_yield);
    EXPECT_TRUE(market.ok()) << market.error().describe();
    const auto option = EuropeanOption::create(c.type, c.strike, c.maturity);
    EXPECT_TRUE(option.ok()) << option.error().describe();
    const auto model = BlackScholesModel::create(c.volatility);
    EXPECT_TRUE(model.ok()) << model.error().describe();

    const auto result =
        BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
    EXPECT_TRUE(result.ok()) << result.error().describe();
    return result.value().value;
}

// ---------------------------------------------------------------------------
// Published prices
// ---------------------------------------------------------------------------

TEST(BlackScholesReferenceTest, MatchesPublishedPrices) {
    constexpr ReferenceCase cases[] = {
        // Hull, "Options, Futures, and Other Derivatives", 9th ed., Example 15.6.
        // S=42, K=40, r=10%, sigma=20%, T=0.5, no dividend. Call 4.76, put 0.81.
        {"Hull 9th ed. Example 15.6 (call)",
         OptionType::Call,
         42.0,
         40.0,
         0.10,
         0.0,
         0.20,
         0.5,
         4.76,
         0.005},
        {"Hull 9th ed. Example 15.6 (put)",
         OptionType::Put,
         42.0,
         40.0,
         0.10,
         0.0,
         0.20,
         0.5,
         0.81,
         0.005},

        // Haug, "The Complete Guide to Option Pricing Formulas", 2nd ed.,
        // section 1.1.1. S=60, K=65, T=0.25, r=8%, sigma=30%. Call 2.1334.
        {"Haug 2nd ed. 1.1.1 (call)",
         OptionType::Call,
         60.0,
         65.0,
         0.08,
         0.0,
         0.30,
         0.25,
         2.1334,
         0.00005},

        // Haug, 2nd ed., generalised Black-Scholes with carry b = r - q.
        // S=100, K=95, T=0.5, r=10%, b=5% (so q=5%), sigma=20%. Put 2.4648.
        // This case exercises the dividend-yield term, which the two cases above
        // leave at zero.
        {"Haug 2nd ed. generalised BSM (put, q=5%)",
         OptionType::Put,
         100.0,
         95.0,
         0.10,
         0.05,
         0.20,
         0.5,
         2.4648,
         0.00005},
    };

    for (const auto& c : cases) {
        const double actual = price_of(c);
        EXPECT_NEAR(actual, c.expected, c.tolerance)
            << "reference: " << c.source << "\n  expected " << c.expected << " +/- " << c.tolerance
            << ", got " << actual;
    }
}

// ---------------------------------------------------------------------------
// Published Greeks
//
// Hull, 9th ed., Chapter 19 develops every Greek from one running example:
// S=49, K=50, r=5%, sigma=20%, T=20 weeks = 0.3846 years, no dividend.
// Using a single parameter set across all five is deliberate on Hull's part and
// useful here: the Greeks must be mutually consistent at that point, not merely
// individually plausible.
// ---------------------------------------------------------------------------

class HullChapter19Test : public ::testing::Test {
protected:
    void SetUp() override {
        market_ = MarketState::create(49.0, 0.05, 0.0).value();
        option_ = EuropeanOption::create(OptionType::Call, 50.0, 0.3846).value();
        model_ = BlackScholesModel::create(0.20).value();

        const auto computed = BlackScholesAnalyticEngine::greeks(*market_, *option_, *model_);
        ASSERT_TRUE(computed.ok()) << computed.error().describe();
        greeks_ = computed.value();
    }

    std::optional<MarketState> market_;
    std::optional<EuropeanOption> option_;
    std::optional<BlackScholesModel> model_;
    Greeks greeks_;
};

TEST_F(HullChapter19Test, Delta) {
    // Hull section 19.4: delta = 0.522.
    EXPECT_NEAR(greeks_.delta, 0.522, 0.0005);
}

TEST_F(HullChapter19Test, Gamma) {
    // Hull section 19.6: gamma = 0.066.
    EXPECT_NEAR(greeks_.gamma, 0.066, 0.0005);
}

TEST_F(HullChapter19Test, Vega) {
    // Hull section 19.7: vega = 12.1, per unit of volatility (a move from 0.20
    // to 1.20). Hull states the per-1% figure as 0.121; this engine reports the
    // derivative itself, so the two differ by the documented factor of 100.
    EXPECT_NEAR(greeks_.vega, 12.1, 0.05);
}

TEST_F(HullChapter19Test, Theta) {
    // Hull section 19.5: theta = -4.31 per year.
    EXPECT_NEAR(greeks_.theta, -4.31, 0.005);
}

TEST_F(HullChapter19Test, Rho) {
    // Hull section 19.8: rho = 8.91, per unit of rate.
    EXPECT_NEAR(greeks_.rho, 8.91, 0.005);
}

// Hull quotes N(d1) = 0.5216 for this example. d1 is the input every Greek above
// shares, so pinning it localises a failure: if d1 is right and a Greek is
// wrong, the defect is in that Greek's formula, not in the setup.
TEST_F(HullChapter19Test, IntermediateTermsMatchPublishedValues) {
    const BlackScholesTerms t = BlackScholesAnalyticEngine::terms(*market_, *option_, *model_);

    EXPECT_FALSE(t.degenerate);
    EXPECT_NEAR(t.d1, 0.0542, 0.0001);
    EXPECT_NEAR(t.d2, -0.0698, 0.0001);
}

}  // namespace
}  // namespace diffusionworks
