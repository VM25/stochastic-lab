#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/barrier_pde_engine.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <ql/exercise.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>
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

// ---------------------------------------------------------------------------
// Barrier options
//
// The value this adds over the mpmath fixture: Reiner-Rubinstein is a set of
// formulae with branches, and a transcription that picks the wrong branch is
// still a formula that produces plausible numbers. QuantLib's implementation was
// written independently from the same literature, so agreement is evidence that
// both read the branch structure the same way -- and the *conventions* the same
// way, which is the part a high-precision oracle cannot check.
//
// The convention at issue here is monitoring. QuantLib's AnalyticBarrierEngine
// prices continuous monitoring, and so does BarrierAnalyticEngine, so the two are
// comparable. Handing it a discretely monitored contract would compare different
// things.
// ---------------------------------------------------------------------------

TEST(QuantLibCrossValidation, BarrierCallsAgreeWithQuantLib) {
    struct BarrierScenario {
        std::string name;
        double spot;
        double strike;
        double barrier;
        double rate;
        double dividend_yield;
        double volatility;
        QuantLib::Natural days;
    };

    // Both directions and both branches of the strike-versus-barrier split. The
    // branch matters: B and D replace A and C when the barrier rather than the
    // strike bounds the payoff region, and a scenario set that stayed on one side
    // of K = B would exercise half the formula while looking thorough.
    const std::vector<BarrierScenario> cases{
        {"down_out_atm", 100.0, 100.0, 90.0, 0.05, 0.00, 0.20, 365},
        {"down_out_near_barrier", 92.0, 100.0, 90.0, 0.05, 0.00, 0.20, 365},
        {"down_out_far_barrier", 100.0, 100.0, 70.0, 0.05, 0.00, 0.20, 365},
        {"down_out_with_dividend", 100.0, 100.0, 80.0, 0.05, 0.03, 0.25, 365},
        {"down_out_long_dated", 110.0, 100.0, 85.0, 0.03, 0.00, 0.20, 730},
        {"down_out_high_vol", 100.0, 100.0, 75.0, 0.05, 0.00, 0.45, 365},
        // Barrier above strike: the B - D branch for a down barrier.
        {"down_barrier_above_strike", 120.0, 100.0, 110.0, 0.05, 0.00, 0.20, 365},
        {"down_barrier_above_strike_dividend", 130.0, 90.0, 105.0, 0.04, 0.02, 0.30, 365},
    };

    const std::vector<BarrierScenario> up_cases{
        {"up_out_atm", 100.0, 100.0, 120.0, 0.05, 0.00, 0.20, 365},
        {"up_out_near_barrier", 108.0, 100.0, 110.0, 0.05, 0.00, 0.20, 365},
        {"up_out_far_barrier", 100.0, 100.0, 200.0, 0.05, 0.00, 0.20, 365},
        {"up_out_with_dividend", 100.0, 100.0, 130.0, 0.05, 0.03, 0.25, 365},
        {"up_out_long_dated", 90.0, 100.0, 125.0, 0.03, 0.00, 0.20, 730},
        {"up_out_high_vol", 100.0, 100.0, 140.0, 0.05, 0.00, 0.45, 365},
        // Strike above barrier: the branch where the up-and-out is exactly zero.
        {"up_barrier_below_strike", 80.0, 100.0, 90.0, 0.05, 0.00, 0.20, 365},
        {"up_barrier_below_strike_high_vol", 70.0, 120.0, 95.0, 0.05, 0.00, 0.40, 365},
    };

    // Counted, because a comparison loop that silently runs zero scenarios passes.
    // Phase 6 shipped exactly that hole once; the count is the cheapest way to make
    // "the branch was exercised" a thing the test asserts rather than assumes.
    int compared = 0;

    const auto check = [&compared](const BarrierScenario& c, BarrierType barrier_type) {
        {
            ++compared;
            const double maturity = static_cast<double>(c.days) / static_cast<double>(kDaysPerYear);

            const auto market = MarketState::create(c.spot, c.rate, c.dividend_yield).value();
            const auto model = BlackScholesModel::create(c.volatility).value();
            const auto option = BarrierOption::create(OptionType::Call,
                                                      barrier_type,
                                                      c.strike,
                                                      c.barrier,
                                                      maturity,
                                                      MonitoringConvention::Continuous,
                                                      std::nullopt)
                                    .value();

            const auto ours = BarrierAnalyticEngine::price(market, option, model);
            ASSERT_TRUE(ours.ok()) << c.name << ": " << ours.error().describe();

            // --- QuantLib ---
            const QuantLib::Date today(1, QuantLib::January, 2026);
            QuantLib::Settings::instance().evaluationDate() = today;
            const QuantLib::DayCounter day_counter = QuantLib::Actual365Fixed();
            const QuantLib::Calendar calendar = QuantLib::NullCalendar();
            const QuantLib::Date expiry = today + c.days;

            const auto spot_quote = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(c.spot);
            const QuantLib::Handle<QuantLib::YieldTermStructure> rate_curve(
                QuantLib::ext::make_shared<QuantLib::FlatForward>(today, c.rate, day_counter));
            const QuantLib::Handle<QuantLib::YieldTermStructure> dividend_curve(
                QuantLib::ext::make_shared<QuantLib::FlatForward>(
                    today, c.dividend_yield, day_counter));
            const QuantLib::Handle<QuantLib::BlackVolTermStructure> vol_surface(
                QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
                    today, calendar, c.volatility, day_counter));

            const auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
                QuantLib::Handle<QuantLib::Quote>(spot_quote),
                dividend_curve,
                rate_curve,
                vol_surface);

            const auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
                QuantLib::Option::Call, c.strike);
            const auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(expiry);

            const QuantLib::Barrier::Type ql_barrier = [barrier_type] {
                switch (barrier_type) {
                    case BarrierType::DownAndOut:
                        return QuantLib::Barrier::DownOut;
                    case BarrierType::DownAndIn:
                        return QuantLib::Barrier::DownIn;
                    case BarrierType::UpAndOut:
                        return QuantLib::Barrier::UpOut;
                    case BarrierType::UpAndIn:
                        return QuantLib::Barrier::UpIn;
                }
                return QuantLib::Barrier::DownOut;
            }();

            QuantLib::BarrierOption ql_option(ql_barrier,
                                              c.barrier,
                                              0.0,  // no rebate
                                              payoff,
                                              exercise);
            ql_option.setPricingEngine(
                QuantLib::ext::make_shared<QuantLib::AnalyticBarrierEngine>(process));

            const double theirs = ql_option.NPV();

            // Absolute and relative together: a near-barrier knock-out is worth
            // almost nothing, where a relative tolerance is meaningless, while an
            // in-the-money one is worth tens, where an absolute one is too loose.
            EXPECT_NEAR(ours.value().value, theirs, 1e-9 + 1e-10 * std::abs(theirs))
                << c.name << " / " << to_string(barrier_type) << ": ours " << ours.value().value
                << " vs QuantLib " << theirs;
        }
    };

    for (const auto& c : cases) {
        for (const auto barrier_type : {BarrierType::DownAndOut, BarrierType::DownAndIn}) {
            check(c, barrier_type);
        }
    }
    for (const auto& c : up_cases) {
        for (const auto barrier_type : {BarrierType::UpAndOut, BarrierType::UpAndIn}) {
            check(c, barrier_type);
        }
    }

    EXPECT_EQ(compared, static_cast<int>(2 * (cases.size() + up_cases.size())));
}

// The PDE barrier engine, against QuantLib rather than against this project's own
// analytic engine. barrier_pde_test.cpp already checks convergence to the native
// closed form; this closes the loop against a third-party solver that shares no
// code with either, so an error common to both native engines cannot hide here.
//
// A convergence tolerance, not a 1e-9 identity: the PDE carries grid, time, and
// truncation error that the closed forms do not, so the test asserts the finest
// grid lands close and refining got it there rather than demanding exactness.
TEST(QuantLibCrossValidation, BarrierPdeConvergesToQuantLib) {
    const QuantLib::Date today(1, QuantLib::January, 2026);
    const QuantLib::DayCounter day_counter = QuantLib::Actual365Fixed();
    const QuantLib::Calendar calendar = QuantLib::NullCalendar();

    struct Case {
        BarrierType type;
        QuantLib::Barrier::Type ql_type;
        double barrier;
    };

    const std::vector<Case> cases{
        {BarrierType::DownAndOut, QuantLib::Barrier::DownOut, 90.0},
        {BarrierType::UpAndOut, QuantLib::Barrier::UpOut, 120.0},
    };

    const double spot = 100.0;
    const double strike = 100.0;
    const double rate = 0.05;
    const double dividend = 0.0;
    const double volatility = 0.2;
    const QuantLib::Natural days = 365;
    const double maturity = static_cast<double>(days) / static_cast<double>(kDaysPerYear);

    int compared = 0;
    for (const auto& c : cases) {
        QuantLib::Settings::instance().evaluationDate() = today;
        const auto spot_quote = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot);
        const QuantLib::Handle<QuantLib::YieldTermStructure> rate_curve(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(today, rate, day_counter));
        const QuantLib::Handle<QuantLib::YieldTermStructure> dividend_curve(
            QuantLib::ext::make_shared<QuantLib::FlatForward>(today, dividend, day_counter));
        const QuantLib::Handle<QuantLib::BlackVolTermStructure> vol_surface(
            QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(
                today, calendar, volatility, day_counter));
        const auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
            QuantLib::Handle<QuantLib::Quote>(spot_quote), dividend_curve, rate_curve, vol_surface);
        const auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(
            QuantLib::Option::Call, strike);
        const auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(today + days);
        QuantLib::BarrierOption ql_option(c.ql_type, c.barrier, 0.0, payoff, exercise);
        ql_option.setPricingEngine(
            QuantLib::ext::make_shared<QuantLib::AnalyticBarrierEngine>(process));
        const double theirs = ql_option.NPV();

        const auto mk = MarketState::create(spot, rate, dividend).value();
        const auto md = BlackScholesModel::create(volatility).value();
        const auto option = BarrierOption::create(OptionType::Call,
                                                  c.type,
                                                  strike,
                                                  c.barrier,
                                                  maturity,
                                                  MonitoringConvention::Continuous,
                                                  std::nullopt)
                                .value();

        double previous_error = std::numeric_limits<double>::infinity();
        double finest_error = 0.0;
        for (const std::int64_t nodes : {201, 401, 801, 1601}) {
            PdeConfig config;
            config.asset_nodes = nodes;
            config.time_steps = nodes;
            config.scheme = PdeScheme::CrankNicolson;
            config.rannacher = RannacherSteps{2};
            const auto priced = BarrierPdeEngine::price(mk, option, md, config);
            ASSERT_TRUE(priced.ok())
                << to_string(c.type) << " nodes=" << nodes << ": " << priced.error().describe();
            const double error = std::abs(priced.value().value - theirs);
            EXPECT_LT(error, previous_error)
                << to_string(c.type) << ": error did not shrink at nodes=" << nodes;
            previous_error = error;
            finest_error = error;
        }
        EXPECT_LT(finest_error, 3e-4)
            << to_string(c.type) << ": PDE did not converge to QuantLib within tolerance";
        ++compared;
    }
    EXPECT_EQ(compared, static_cast<int>(cases.size()));
}

}  // namespace
}  // namespace diffusionworks
