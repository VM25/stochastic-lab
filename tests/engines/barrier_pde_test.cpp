#include <diffusionworks/core/error.hpp>
#include <diffusionworks/engines/barrier_analytic.hpp>
#include <diffusionworks/engines/barrier_pde_engine.hpp>
#include <diffusionworks/engines/black_scholes_analytic.hpp>
#include <diffusionworks/engines/finite_difference_engine.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace diffusionworks {
namespace {

MarketState market(double spot = 100.0) {
    return MarketState::create(spot, 0.05, 0.0).value();
}

BlackScholesModel model(double volatility = 0.2) {
    return BlackScholesModel::create(volatility).value();
}

BarrierOption barrier(BarrierType type, double strike, double level, double maturity = 1.0) {
    return BarrierOption::create(OptionType::Call,
                                 type,
                                 strike,
                                 level,
                                 maturity,
                                 MonitoringConvention::Continuous,
                                 std::nullopt)
        .value();
}

/// The continuously monitored analytic price -- the independent oracle. It agrees
/// with mpmath to 1e-15 and QuantLib to 1e-9, so the PDE's job is to converge to it.
double analytic(const MarketState& mk,
                const BlackScholesModel& md,
                BarrierType type,
                double strike,
                double level,
                double maturity = 1.0) {
    return BarrierAnalyticEngine::price(mk, barrier(type, strike, level, maturity), md)
        .value()
        .value;
}

PdeConfig accurate(std::int64_t nodes, std::int64_t steps) {
    PdeConfig config;
    config.asset_nodes = nodes;
    config.time_steps = steps;
    config.scheme = PdeScheme::CrankNicolson;
    // Rannacher smoothing: the barrier and the payoff kink are both non-smooth
    // features, and Crank-Nicolson oscillates them over the first steps unless the
    // highest modes are damped first.
    config.rannacher = RannacherSteps{2};
    return config;
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

// ---------------------------------------------------------------------------
// Convergence to the analytic reference
//
// The headline: the PDE prices the *continuous* barrier and converges to the
// closed form as the grid refines. This is the PDE arm's whole claim, and it is
// checked against an oracle that shares no algebra with the finite-difference
// solver.
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, DownAndOutConvergesToTheAnalyticPrice) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    double previous_error = std::numeric_limits<double>::infinity();
    for (const std::int64_t nodes : {201, 401, 801, 1601}) {
        const auto priced = BarrierPdeEngine::price(
            mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(nodes, nodes));
        ASSERT_TRUE(priced.ok()) << "nodes=" << nodes << ": " << priced.error().describe();

        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, previous_error) << "error did not shrink at nodes=" << nodes;
        previous_error = error;
    }
    // The finest error, from the measurement: ~1.2e-4 at 1601 nodes.
    EXPECT_LT(previous_error, 3e-4);
}

TEST(BarrierPdeTest, UpAndOutConvergesToTheAnalyticPrice) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::UpAndOut, 100.0, 120.0);

    double previous_error = std::numeric_limits<double>::infinity();
    for (const std::int64_t nodes : {201, 401, 801, 1601}) {
        const auto priced = BarrierPdeEngine::price(
            mk, barrier(BarrierType::UpAndOut, 100.0, 120.0), md, accurate(nodes, nodes));
        ASSERT_TRUE(priced.ok()) << "nodes=" << nodes << ": " << priced.error().describe();

        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, previous_error) << "error did not shrink at nodes=" << nodes;
        previous_error = error;
    }
    EXPECT_LT(previous_error, 1e-5);
}

// The convergence is second order in dS. Not asserted as a fitted slope here -- that
// is EXP-07's PDE arm -- but the halving of error per doubling of resolution is the
// signature, and a first-order bug (a misplaced boundary, a wrong index) would break
// it conspicuously. A ratio floor of 3.0 leaves room for the sub-dominant time and
// truncation terms while still failing an order-one regression.
TEST(BarrierPdeTest, DownAndOutConvergesAtSecondOrder) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    std::vector<double> errors;
    for (const std::int64_t nodes : {201, 401, 801, 1601}) {
        const auto priced = BarrierPdeEngine::price(
            mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(nodes, nodes));
        ASSERT_TRUE(priced.ok());
        errors.push_back(std::abs(priced.value().value - reference));
    }
    for (std::size_t i = 1; i < errors.size(); ++i) {
        EXPECT_GT(errors[i - 1] / errors[i], 3.0)
            << "error ratio between levels " << i - 1 << " and " << i
            << " is below the second-order signature";
    }
}

// ---------------------------------------------------------------------------
// Separate grid and time convergence
//
// A price converging is not the same as each discretisation converging. Refining
// only space (with time already fine) must reduce the error toward a floor, and
// likewise for time. Conflating them can hide a scheme whose two errors happen to
// cancel at one resolution.
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, RefiningSpaceAloneReducesTheError) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    double previous_error = std::numeric_limits<double>::infinity();
    for (const std::int64_t nodes : {101, 201, 401, 801}) {
        // Time held fine so the space error dominates.
        const auto priced = BarrierPdeEngine::price(
            mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(nodes, 4000));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, previous_error) << "space refinement did not help at nodes=" << nodes;
        previous_error = error;
    }
}

TEST(BarrierPdeTest, RefiningTimeAloneReducesTheError) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    double previous_error = std::numeric_limits<double>::infinity();
    for (const std::int64_t steps : {25, 50, 100, 200}) {
        // Space held fine so the time error dominates.
        const auto priced = BarrierPdeEngine::price(
            mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(2001, steps));
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, previous_error) << "time refinement did not help at steps=" << steps;
        previous_error = error;
    }
}

// ---------------------------------------------------------------------------
// S_max truncation, for down barriers
//
// The upper boundary is asymptotic, so a nearer S_max is a cruder approximation.
// Widening it -- while holding dS fixed by scaling the node count -- must reduce the
// error. The up-and-out has no such term: its domain top is the barrier, an exact
// boundary, not a truncation.
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, WideningSmaxReducesDownBarrierTruncationError) {
    const auto mk = market();
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    double previous_error = std::numeric_limits<double>::infinity();
    // dS held constant at 0.25 (node count grows with S_max) and time held fine, so
    // the truncation term is what moves. The near multiples are where it is visible:
    // ~0.48 at 1.15x strike, ~0.02 at 1.3x, and by 2x it has fallen into the
    // grid/time floor. A wider net (2x..6x) would sit entirely in that plateau and
    // measure only noise, which is why the earlier version of this test could not see
    // the effect it claimed to.
    for (const double s_max_multiple : {1.15, 1.3, 1.5, 2.0}) {
        const double s_max = s_max_multiple * 100.0;
        // std::llround already returns long long (== int64_t here), so no cast.
        const std::int64_t nodes = std::llround(s_max / 0.25) + 1;
        PdeConfig config = accurate(nodes, 8000);
        config.s_max = s_max;
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, config);
        ASSERT_TRUE(priced.ok()) << "s_max=" << s_max << ": " << priced.error().describe();
        const double error = std::abs(priced.value().value - reference);
        EXPECT_LT(error, previous_error) << "widening S_max did not help at " << s_max;
        previous_error = error;
    }
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

// A knock-out can only ever pay what its vanilla pays, on a subset of paths, so its
// price cannot exceed the vanilla's. Compared against the vanilla priced by the
// *same PDE engine* on the same grid, so this is a pathwise property of the solver
// rather than a comparison across methods.
TEST(BarrierPdeTest, BarrierPriceNeverExceedsTheVanilla) {
    const auto mk = market();
    const auto md = model();

    PdeConfig config;
    config.asset_nodes = 801;
    config.time_steps = 400;
    config.scheme = PdeScheme::CrankNicolson;
    config.rannacher = RannacherSteps{2};

    const double vanilla =
        FiniteDifferenceEngine::price(
            mk, EuropeanOption::create(OptionType::Call, 100.0, 1.0).value(), md, config)
            .value()
            .value;

    for (const double level : {70.0, 80.0, 90.0, 95.0}) {
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(BarrierType::DownAndOut, 100.0, level), md, config);
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        EXPECT_LE(priced.value().value, vanilla + 1e-9) << "down B=" << level;
        EXPECT_GE(priced.value().value, -1e-9) << "down B=" << level;
    }
    for (const double level : {110.0, 120.0, 140.0}) {
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(BarrierType::UpAndOut, 100.0, level), md, config);
        ASSERT_TRUE(priced.ok()) << priced.error().describe();
        EXPECT_LE(priced.value().value, vanilla + 1e-9) << "up B=" << level;
        EXPECT_GE(priced.value().value, -1e-9) << "up B=" << level;
    }
}

// The solution stays non-negative everywhere, not just at the spot. A knock-out
// price is non-negative, so a negative grid value is oscillation. The diagnostic
// records the most negative value seen at any step; it must be zero.
TEST(BarrierPdeTest, TheSolutionIsNonNegativeEverywhere) {
    const auto mk = market();
    const auto md = model();
    for (const auto type : {BarrierType::DownAndOut, BarrierType::UpAndOut}) {
        const double level = type == BarrierType::DownAndOut ? 90.0 : 120.0;
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(type, 100.0, level), md, accurate(801, 400));
        ASSERT_TRUE(priced.ok());
        EXPECT_GE(diagnostic(priced.value(), "most_negative_value"), 0.0) << to_string(type);
    }
}

// A down-and-out is worth more the further the barrier sits below the spot, because
// fewer paths reach it. Monotone by construction of the contract; a sign error in
// the boundary or the reflected term would break it.
TEST(BarrierPdeTest, DownAndOutRisesAsTheBarrierRecedes) {
    const auto mk = market();
    const auto md = model();
    const auto config = accurate(801, 400);

    double previous = -1.0;
    for (const double level : {95.0, 90.0, 80.0, 70.0, 50.0}) {
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(BarrierType::DownAndOut, 100.0, level), md, config);
        ASSERT_TRUE(priced.ok());
        EXPECT_GT(priced.value().value, previous) << "at B=" << level;
        previous = priced.value().value;
    }
}

// ---------------------------------------------------------------------------
// Immediate knockout and worthless cases
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, AnAlreadyBreachedKnockOutIsExactlyWorthless) {
    const auto md = model();

    // Down-and-out, spot at or below the barrier.
    for (const double spot : {90.0, 85.0, 50.0}) {
        const auto priced = BarrierPdeEngine::price(
            market(spot), barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(401, 200));
        ASSERT_TRUE(priced.ok()) << "spot=" << spot << ": " << priced.error().describe();
        EXPECT_EQ(priced.value().value, 0.0) << "spot=" << spot;
        EXPECT_TRUE(priced.value().has_warnings());
    }

    // Up-and-out, spot at or above the barrier -- the case where the breached spot
    // sits above the grid, so it must be handled before interpolation.
    for (const double spot : {120.0, 130.0}) {
        const auto priced = BarrierPdeEngine::price(
            market(spot), barrier(BarrierType::UpAndOut, 100.0, 120.0), md, accurate(401, 200));
        ASSERT_TRUE(priced.ok()) << "spot=" << spot << ": " << priced.error().describe();
        EXPECT_EQ(priced.value().value, 0.0) << "spot=" << spot;
        EXPECT_TRUE(priced.value().has_warnings());
    }
}

// An up-and-out call struck at or above its barrier cannot pay: any path finishing
// above K >= B has crossed B and knocked out. Exactly zero, and it agrees with the
// analytic engine's exact zero.
TEST(BarrierPdeTest, UpAndOutStruckAboveTheBarrierIsWorthless) {
    const auto mk = market(80.0);
    const auto md = model();
    const auto priced = BarrierPdeEngine::price(
        mk, barrier(BarrierType::UpAndOut, 100.0, 90.0), md, accurate(401, 200));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_EQ(priced.value().value, 0.0);
    EXPECT_EQ(analytic(mk, md, BarrierType::UpAndOut, 100.0, 90.0), 0.0);
}

// ---------------------------------------------------------------------------
// Harder regimes: near-barrier spot, short maturity, high volatility
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, ConvergesWithTheSpotNearTheBarrier) {
    // A spot just above a down barrier: the value rises steeply from zero, which is
    // where interpolation and the boundary are most strained.
    const auto mk = market(91.0);
    const auto md = model();
    const double reference = analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0);

    const auto priced = BarrierPdeEngine::price(
        mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, accurate(1601, 800));
    ASSERT_TRUE(priced.ok()) << priced.error().describe();
    EXPECT_NEAR(priced.value().value, reference, 5e-3);
}

TEST(BarrierPdeTest, ConvergesAtShortMaturityAndHighVolatility) {
    const auto mk = market();
    for (const double maturity : {0.1, 0.25}) {
        for (const double volatility : {0.4, 0.8}) {
            const auto md = model(volatility);
            const double reference =
                analytic(mk, md, BarrierType::DownAndOut, 100.0, 90.0, maturity);
            const auto priced =
                BarrierPdeEngine::price(mk,
                                        barrier(BarrierType::DownAndOut, 100.0, 90.0, maturity),
                                        md,
                                        accurate(1601, 800));
            ASSERT_TRUE(priced.ok())
                << "T=" << maturity << " sigma=" << volatility << ": " << priced.error().describe();
            EXPECT_NEAR(priced.value().value, reference, 5e-3)
                << "T=" << maturity << " sigma=" << volatility;
        }
    }
}

// ---------------------------------------------------------------------------
// The up-and-out barrier must sit exactly on the top node
//
// Regression for a real defect: the first up-and-out grid aligned the strike and
// left the barrier at the nearest multiple of the spacing, a rounding error below B.
// The absorbing boundary then priced a barrier that was not the one asked for, and
// nearest_index(B) fell off the grid at some resolutions.
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, UpAndOutBarrierSitsExactlyOnTheTopNode) {
    const auto mk = market();
    const auto md = model();
    for (const std::int64_t nodes : {101, 201, 401, 801}) {
        const auto solution = BarrierPdeEngine::solve(
            mk, barrier(BarrierType::UpAndOut, 100.0, 120.0), md, accurate(nodes, nodes));
        ASSERT_TRUE(solution.ok()) << "nodes=" << nodes << ": " << solution.error().describe();
        // The top node is the barrier, exactly.
        EXPECT_EQ(solution.value().grid.s_max(), 120.0) << "nodes=" << nodes;
        // And the value there is the absorbing zero.
        EXPECT_EQ(solution.value().values.back(), 0.0) << "nodes=" << nodes;
    }
}

// ---------------------------------------------------------------------------
// What the engine refuses
// ---------------------------------------------------------------------------

TEST(BarrierPdeTest, RefusesDiscreteAndBridgeMonitoring) {
    const auto mk = market();
    const auto md = model();
    for (const auto convention :
         {MonitoringConvention::Discrete, MonitoringConvention::BrownianBridge}) {
        const auto option =
            BarrierOption::create(
                OptionType::Call, BarrierType::DownAndOut, 100.0, 90.0, 1.0, convention, 50)
                .value();
        const auto priced = BarrierPdeEngine::price(mk, option, md, accurate(201, 100));
        ASSERT_FALSE(priced.ok()) << to_string(convention);
        EXPECT_EQ(priced.error().code, ErrorCode::UnsupportedCombination);
    }
}

TEST(BarrierPdeTest, RefusesKnockInsAndPuts) {
    const auto mk = market();
    const auto md = model();

    for (const auto type : {BarrierType::DownAndIn, BarrierType::UpAndIn}) {
        const auto priced =
            BarrierPdeEngine::price(mk, barrier(type, 100.0, 90.0), md, accurate(201, 100));
        ASSERT_FALSE(priced.ok()) << to_string(type);
        EXPECT_EQ(priced.error().code, ErrorCode::NotImplemented);
    }

    const auto put = BarrierOption::create(OptionType::Put,
                                           BarrierType::DownAndOut,
                                           100.0,
                                           90.0,
                                           1.0,
                                           MonitoringConvention::Continuous,
                                           std::nullopt)
                         .value();
    const auto priced = BarrierPdeEngine::price(mk, put, md, accurate(201, 100));
    ASSERT_FALSE(priced.ok());
    EXPECT_EQ(priced.error().code, ErrorCode::NotImplemented);
}

// The explicit scheme beyond its stability bound must warn or fail, never return a
// finite number as though it were trustworthy. The same guard the vanilla engine
// has, exercised through the barrier engine.
TEST(BarrierPdeTest, ExplicitInstabilityIsSurfaced) {
    const auto mk = market();
    const auto md = model();
    PdeConfig config;
    config.asset_nodes = 201;
    config.time_steps = 50;  // far too few for stability at this dS
    config.scheme = PdeScheme::Explicit;

    const auto priced =
        BarrierPdeEngine::price(mk, barrier(BarrierType::DownAndOut, 100.0, 90.0), md, config);
    // Either it diverged to non-finite (a failure) or it stayed finite and warned.
    if (priced.ok()) {
        EXPECT_TRUE(priced.value().has_warnings())
            << "an unstable explicit run returned a finite value with no warning";
    } else {
        EXPECT_EQ(priced.error().code, ErrorCode::NonFiniteValue);
    }
}

}  // namespace
}  // namespace diffusionworks
