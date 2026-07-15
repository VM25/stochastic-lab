#include <diffusionworks/engines/black_scholes_analytic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// External cross-validation against QuantLib.
//
// Role in the validation set (VALIDATION-PLAN section 17, ADR-003)
// ----------------------------------------------------------------
// QuantLib is a reference only. It never appears in the engine, the CLI, or any
// published result; it is linked into this test target alone, which builds only
// when DW_ENABLE_QUANTLIB=ON. Nothing in the project depends on it being
// installed.
//
// It contributes what the other oracles cannot: an independent *production*
// implementation, written by different people against the same mathematics. The
// published Hull and Haug values are human-checked but few and coarsely rounded;
// the mpmath fixture is exact but shares this project's reading of the
// conventions. If DiffusionWorks and QuantLib have both misread the same
// convention, only a disagreement here would reveal it.
//
// Conventions must match or the comparison is meaningless
// -------------------------------------------------------
// QuantLib works in dates, not year fractions. Maturities are therefore
// specified in whole days and converted with Actual/365 Fixed, which is exactly
// the day counter handed to QuantLib -- so both sides see the identical T rather
// than two values that differ in the twelfth digit. A NullCalendar avoids
// holiday adjustments moving the exercise date.
//
// A disagreement here is investigated, not silenced (VALIDATION-PLAN section 17):
// QuantLib does not automatically override this project's mathematics, but it
// does not lose automatically either.
// ---------------------------------------------------------------------------

constexpr QuantLib::Natural kDaysPerYear = 365;

/// Maturities are given in days so both engines can agree on T exactly.
struct Scenario {
    std::string name;
    double spot;
    double strike;
    double rate;
    double dividend_yield;
    double volatility;
    QuantLib::Natural days;
};

std::vector<Scenario> scenarios() {
    return {
        {"at_the_money", 100.0, 100.0, 0.05, 0.00, 0.20, 365},
        {"in_the_money", 130.0, 100.0, 0.05, 0.00, 0.20, 365},
        {"out_of_the_money", 70.0, 100.0, 0.05, 0.00, 0.20, 365},
        {"with_dividend", 100.0, 100.0, 0.05, 0.03, 0.20, 365},
        {"high_dividend", 100.0, 100.0, 0.02, 0.08, 0.25, 730},
        {"low_volatility", 100.0, 100.0, 0.05, 0.00, 0.05, 365},
        {"high_volatility", 100.0, 100.0, 0.05, 0.00, 0.60, 365},
        {"short_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 30},
        {"long_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 3650},
        {"negative_rate", 100.0, 100.0, -0.01, 0.00, 0.20, 365},
        {"deep_out_of_the_money", 100.0, 200.0, 0.00, 0.00, 0.20, 365},
        {"deep_in_the_money", 200.0, 100.0, 0.05, 0.00, 0.20, 365},
    };
}

[[nodiscard]] double year_fraction(QuantLib::Natural days) {
    return static_cast<double>(days) / static_cast<double>(kDaysPerYear);
}

/// What QuantLib reports for one scenario.
struct QuantLibResult {
    double price{};
    double delta{};
    double gamma{};
    double vega{};
    double theta{};
    double rho{};
};

[[nodiscard]] QuantLibResult price_with_quantlib(const Scenario& s, QuantLib::Option::Type type) {
    using namespace QuantLib;

    const Date today(15, July, 2026);
    Settings::instance().evaluationDate() = today;

    const DayCounter day_counter = Actual365Fixed();
    const Calendar calendar = NullCalendar();
    const Date exercise_date = today + Period(static_cast<Integer>(s.days), Days);

    const Handle<Quote> spot(ext::make_shared<SimpleQuote>(s.spot));
    const Handle<YieldTermStructure> dividend_curve(
        ext::make_shared<FlatForward>(today, s.dividend_yield, day_counter));
    const Handle<YieldTermStructure> rate_curve(
        ext::make_shared<FlatForward>(today, s.rate, day_counter));
    const Handle<BlackVolTermStructure> vol_surface(
        ext::make_shared<BlackConstantVol>(today, calendar, s.volatility, day_counter));

    const auto process =
        ext::make_shared<BlackScholesMertonProcess>(spot, dividend_curve, rate_curve, vol_surface);

    const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, s.strike);
    const auto exercise = ext::make_shared<EuropeanExercise>(exercise_date);

    VanillaOption option(payoff, exercise);
    option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

    QuantLibResult result;
    result.price = option.NPV();
    result.delta = option.delta();
    result.gamma = option.gamma();
    result.vega = option.vega();
    result.theta = option.theta();
    result.rho = option.rho();
    return result;
}

[[nodiscard]] PricingResult price_with_diffusionworks(const Scenario& s, OptionType type) {
    const auto market = MarketState::create(s.spot, s.rate, s.dividend_yield);
    EXPECT_TRUE(market.ok()) << market.error().describe();
    const auto option = EuropeanOption::create(type, s.strike, year_fraction(s.days));
    EXPECT_TRUE(option.ok()) << option.error().describe();
    const auto model = BlackScholesModel::create(s.volatility);
    EXPECT_TRUE(model.ok()) << model.error().describe();

    auto result = BlackScholesAnalyticEngine::price(market.value(), option.value(), model.value());
    EXPECT_TRUE(result.ok()) << result.error().describe();

    auto greeks = BlackScholesAnalyticEngine::greeks(market.value(), option.value(), model.value());
    EXPECT_TRUE(greeks.ok()) << greeks.error().describe();
    PricingResult priced = std::move(result).value();
    priced.greeks = std::move(greeks).value();
    return priced;
}

void expect_close(double actual,
                  double expected,
                  double rel_tolerance,
                  double abs_tolerance,
                  const std::string& what) {
    const double allowed = abs_tolerance + rel_tolerance * std::abs(expected);
    EXPECT_NEAR(actual, expected, allowed)
        << what << "\n  DiffusionWorks : " << actual << "\n  QuantLib       : " << expected
        << "\n  allowed        : " << allowed;
}

// Both engines evaluate the same closed form in double precision, so they should
// agree to within accumulated rounding -- far tighter than any modelling
// tolerance. 1e-12 relative is ~1e4 times double's resolution, leaving room for
// the two implementations to order their operations differently.
TEST(QuantLibCrossValidationTest, PricesAgree) {
    for (const auto& s : scenarios()) {
        for (const auto& [dw_type, ql_type] : {std::pair{OptionType::Call, QuantLib::Option::Call},
                                               std::pair{OptionType::Put, QuantLib::Option::Put}}) {
            const double ours = price_with_diffusionworks(s, dw_type).value;
            const double theirs = price_with_quantlib(s, ql_type).price;

            expect_close(ours,
                         theirs,
                         1e-12,
                         1e-14,
                         s.name + " " + std::string(to_string(dw_type)) + " price");
        }
    }
}

TEST(QuantLibCrossValidationTest, DeltaAndGammaAgree) {
    for (const auto& s : scenarios()) {
        for (const auto& [dw_type, ql_type] : {std::pair{OptionType::Call, QuantLib::Option::Call},
                                               std::pair{OptionType::Put, QuantLib::Option::Put}}) {
            const Greeks ours = *price_with_diffusionworks(s, dw_type).greeks;
            const QuantLibResult theirs = price_with_quantlib(s, ql_type);

            expect_close(*ours.delta,
                         theirs.delta,
                         1e-12,
                         1e-14,
                         s.name + " " + std::string(to_string(dw_type)) + " delta");
            expect_close(*ours.gamma,
                         theirs.gamma,
                         1e-12,
                         1e-15,
                         s.name + " " + std::string(to_string(dw_type)) + " gamma");
        }
    }
}

// Vega and rho need unit conversion, which is exactly the kind of mismatch this
// test exists to catch. QuantLib reports both per unit of the underlying
// variable, matching this project; had either scaled by 1/100, the disagreement
// would be a factor of 100 and unmissable.
TEST(QuantLibCrossValidationTest, VegaAndRhoAgreeInTheSameUnits) {
    for (const auto& s : scenarios()) {
        for (const auto& [dw_type, ql_type] : {std::pair{OptionType::Call, QuantLib::Option::Call},
                                               std::pair{OptionType::Put, QuantLib::Option::Put}}) {
            const Greeks ours = *price_with_diffusionworks(s, dw_type).greeks;
            const QuantLibResult theirs = price_with_quantlib(s, ql_type);

            expect_close(*ours.vega,
                         theirs.vega,
                         1e-12,
                         1e-13,
                         s.name + " " + std::string(to_string(dw_type)) + " vega");
            expect_close(*ours.rho,
                         theirs.rho,
                         1e-12,
                         1e-13,
                         s.name + " " + std::string(to_string(dw_type)) + " rho");
        }
    }
}

// QuantLib's theta is per year of calendar time, the same convention this project
// uses, so no rescaling is applied. A sign or per-day convention difference would
// show up immediately.
TEST(QuantLibCrossValidationTest, ThetaAgreesInTheSameConvention) {
    for (const auto& s : scenarios()) {
        for (const auto& [dw_type, ql_type] : {std::pair{OptionType::Call, QuantLib::Option::Call},
                                               std::pair{OptionType::Put, QuantLib::Option::Put}}) {
            const Greeks ours = *price_with_diffusionworks(s, dw_type).greeks;
            const QuantLibResult theirs = price_with_quantlib(s, ql_type);

            expect_close(*ours.theta,
                         theirs.theta,
                         1e-10,
                         1e-12,
                         s.name + " " + std::string(to_string(dw_type)) + " theta");
        }
    }
}

// Guards the comparison itself. If the two engines were handed different
// maturities, every test above would be comparing different options while
// appearing to pass at a loose tolerance.
TEST(QuantLibCrossValidationTest, BothEnginesSeeTheSameMaturity) {
    using namespace QuantLib;

    const Date today(15, July, 2026);
    Settings::instance().evaluationDate() = today;
    const DayCounter day_counter = Actual365Fixed();

    for (const auto& s : scenarios()) {
        const Date exercise_date = today + Period(static_cast<Integer>(s.days), Days);
        const double quantlib_maturity = day_counter.yearFraction(today, exercise_date);

        EXPECT_DOUBLE_EQ(quantlib_maturity, year_fraction(s.days))
            << s.name << ": the two engines disagree about T, so the comparison is void";
    }
}

}  // namespace
}  // namespace diffusionworks
