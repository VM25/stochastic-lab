#include <diffusionworks/core/error.hpp>
#include <diffusionworks/pde/asset_grid.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// The uniform grid
// ---------------------------------------------------------------------------

TEST(AssetGridTest, SpansZeroToSmaxInclusive) {
    const auto grid = AssetGrid::uniform(400.0, 5);
    ASSERT_TRUE(grid.ok()) << grid.error().describe();

    EXPECT_EQ(grid.value().nodes(), 5);
    EXPECT_DOUBLE_EQ(grid.value().spacing(), 100.0);

    // The endpoints must be exact, not nearly exact. The lower boundary condition
    // is written at S = 0 and the upper at S = S_max; applying either at
    // almost-there would be applying it in the wrong place.
    EXPECT_DOUBLE_EQ(grid.value().at(0), 0.0);
    EXPECT_DOUBLE_EQ(grid.value().at(4), 400.0);
    EXPECT_DOUBLE_EQ(grid.value().at(2), 200.0);
}

// The endpoints stay exact on a spacing that does not divide evenly in binary --
// where i*dS accumulates a rounding error and lands an ulp away from s_max.
TEST(AssetGridTest, EndpointsAreExactOnAnAwkwardSpacing) {
    for (const std::int64_t nodes : {7, 13, 101, 1001, 4097}) {
        const auto grid = AssetGrid::uniform(300.0, nodes);
        ASSERT_TRUE(grid.ok()) << "nodes = " << nodes;

        EXPECT_DOUBLE_EQ(grid.value().at(0), 0.0) << "nodes = " << nodes;
        EXPECT_DOUBLE_EQ(grid.value().at(nodes - 1), 300.0)
            << "nodes = " << nodes << ": the top node must be exactly s_max";
    }
}

TEST(AssetGridTest, NodesAreMonotoneAndEvenlySpaced) {
    const auto grid = AssetGrid::uniform(250.0, 51);
    ASSERT_TRUE(grid.ok());

    const double spacing = grid.value().spacing();
    for (std::int64_t i = 1; i < grid.value().nodes(); ++i) {
        const double step = grid.value().at(i) - grid.value().at(i - 1);
        EXPECT_GT(step, 0.0) << "at node " << i;
        EXPECT_NEAR(step, spacing, 1e-12 * spacing) << "at node " << i;
    }
}

TEST(AssetGridTest, RejectsADegenerateGrid) {
    EXPECT_FALSE(AssetGrid::uniform(0.0, 10).ok());
    EXPECT_FALSE(AssetGrid::uniform(-100.0, 10).ok());
    EXPECT_FALSE(AssetGrid::uniform(std::numeric_limits<double>::quiet_NaN(), 10).ok());
    EXPECT_FALSE(AssetGrid::uniform(std::numeric_limits<double>::infinity(), 10).ok());

    // Two nodes are only the boundaries: there is nothing between them to step.
    EXPECT_FALSE(AssetGrid::uniform(100.0, 2).ok());
    EXPECT_FALSE(AssetGrid::uniform(100.0, 0).ok());
    EXPECT_FALSE(AssetGrid::uniform(100.0, -5).ok());
    EXPECT_TRUE(AssetGrid::uniform(100.0, 3).ok());
}

// ---------------------------------------------------------------------------
// Strike alignment
//
// The European payoff is not differentiable at S = K. Aligning the strike to a
// node removes one source of error -- the terminal condition is sampled at the
// kink rather than at two nodes straddling it -- and stops the kink's offset from
// the nearest node drifting as the grid refines.
//
// These tests pin the *alignment*, which is an exact geometric property. What that
// alignment buys in convergence order is a separate, empirical question, and it is
// EXP-06's to answer rather than this file's to assert.
// ---------------------------------------------------------------------------

TEST(AssetGridTest, PlacesTheStrikeExactlyOnANode) {
    for (const double strike : {80.0, 100.0, 100.5, 133.7}) {
        for (const std::int64_t nodes : {51, 101, 201, 401}) {
            const auto grid = AssetGrid::with_strike_on_node(400.0, nodes, strike);
            ASSERT_TRUE(grid.ok())
                << "strike " << strike << ", nodes " << nodes << ": " << grid.error().describe();

            const auto index = grid.value().nearest_index(strike);
            ASSERT_TRUE(index.has_value());

            // Exactly, not nearly. The whole point is that the kink sits on a node
            // rather than a rounding error away from one.
            EXPECT_DOUBLE_EQ(grid.value().at(*index), strike)
                << "strike " << strike << ", nodes " << nodes;
        }
    }
}

TEST(AssetGridTest, StrikeAlignedGridStillEndsExactlyAtItsOwnSmax) {
    const auto grid = AssetGrid::with_strike_on_node(400.0, 101, 100.0);
    ASSERT_TRUE(grid.ok());

    EXPECT_DOUBLE_EQ(grid.value().at(0), 0.0);
    EXPECT_DOUBLE_EQ(grid.value().at(grid.value().nodes() - 1), grid.value().s_max());
}

// The aligned grid reports the s_max it actually built, which may differ from the
// request. Reported rather than silently substituted: the boundary condition is
// applied at the top node, and a caller comparing against their requested value
// should see the difference rather than assume it away.
TEST(AssetGridTest, ReportsTheAlignedSmaxRatherThanTheRequestedOne) {
    // 137 does not divide 400 evenly, so alignment must move s_max.
    const auto grid = AssetGrid::with_strike_on_node(400.0, 60, 137.0);
    ASSERT_TRUE(grid.ok()) << grid.error().describe();

    // Whatever s_max became, the top node is exactly it, and the spacing divides it.
    EXPECT_DOUBLE_EQ(grid.value().at(grid.value().nodes() - 1), grid.value().s_max());
    const double intervals = grid.value().s_max() / grid.value().spacing();
    EXPECT_NEAR(intervals, std::round(intervals), 1e-9)
        << "s_max must be an exact multiple of the spacing";

    // And it stays near the request rather than wandering off.
    EXPECT_GT(grid.value().s_max(), 380.0);
    EXPECT_LT(grid.value().s_max(), 420.0);
}

TEST(AssetGridTest, RejectsAStrikeOutsideTheGrid) {
    // A strike at or beyond s_max means the kink is never resolved.
    EXPECT_FALSE(AssetGrid::with_strike_on_node(100.0, 51, 100.0).ok());
    EXPECT_FALSE(AssetGrid::with_strike_on_node(100.0, 51, 150.0).ok());
    EXPECT_FALSE(AssetGrid::with_strike_on_node(400.0, 51, 0.0).ok());
    EXPECT_FALSE(AssetGrid::with_strike_on_node(400.0, 51, -10.0).ok());
}

// Too coarse to place the strike anywhere but the S=0 boundary: refused rather
// than silently snapped there, which would put the kink at the wrong price.
TEST(AssetGridTest, RejectsAGridTooCoarseToAlignTheStrike) {
    // s_max 400 over 3 nodes is a spacing of 200; a strike of 10 rounds to node 0.
    const auto grid = AssetGrid::with_strike_on_node(400.0, 3, 10.0);
    ASSERT_FALSE(grid.ok());
    EXPECT_EQ(grid.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

TEST(AssetGridTest, FindsTheNearestNode) {
    const auto grid = AssetGrid::uniform(100.0, 11);  // spacing 10
    ASSERT_TRUE(grid.ok());

    EXPECT_EQ(grid.value().nearest_index(0.0), 0);
    EXPECT_EQ(grid.value().nearest_index(50.0), 5);
    EXPECT_EQ(grid.value().nearest_index(100.0), 10);
    EXPECT_EQ(grid.value().nearest_index(52.0), 5);
    EXPECT_EQ(grid.value().nearest_index(58.0), 6);
}

// Absent rather than clamped. A caller asking about a spot beyond S_max has a
// modelling problem, and answering about S_max instead would conceal it behind a
// plausible price.
TEST(AssetGridTest, ReportsNoNodeForASpotOutsideTheGrid) {
    const auto grid = AssetGrid::uniform(100.0, 11);
    ASSERT_TRUE(grid.ok());

    EXPECT_FALSE(grid.value().nearest_index(-1.0).has_value());
    EXPECT_FALSE(grid.value().nearest_index(100.1).has_value());
    EXPECT_FALSE(grid.value().nearest_index(std::numeric_limits<double>::quiet_NaN()).has_value());
    EXPECT_FALSE(grid.value().nearest_index(std::numeric_limits<double>::infinity()).has_value());
}

// ---------------------------------------------------------------------------
// Barrier alignment
//
// The same arithmetic as strike alignment, and a different obligation. A strike
// half a cell off its node is a slightly misplaced kink; a *barrier* half a cell
// off its node is a Dirichlet boundary in the wrong place, which prices a
// different contract. So these tests assert exactness rather than closeness.
// ---------------------------------------------------------------------------

TEST(AssetGridTest, PlacesTheBarrierExactlyOnANode) {
    for (const double barrier : {50.0, 90.0, 95.0, 110.0, 137.0}) {
        for (const std::int64_t nodes : {51, 101, 201, 401}) {
            const auto grid = AssetGrid::with_barrier_on_node(400.0, nodes, barrier);
            ASSERT_TRUE(grid.ok())
                << "B=" << barrier << " nodes=" << nodes << ": " << grid.error().describe();

            const auto index = grid.value().nearest_index(barrier);
            ASSERT_TRUE(index.has_value());
            // Exactly, not nearly. A barrier a rounding error from its node is a
            // boundary condition applied at the wrong price.
            EXPECT_EQ(grid.value().at(*index), barrier) << "B=" << barrier << " nodes=" << nodes;
        }
    }
}

// The two aligners share their arithmetic, so aligning to the same number by either
// route must give the same grid. If they ever diverge, one of them has been changed
// without the other.
TEST(AssetGridTest, BarrierAndStrikeAlignmentAgreeOnTheSameLevel) {
    for (const double level : {70.0, 100.0, 133.0}) {
        const auto by_strike = AssetGrid::with_strike_on_node(400.0, 101, level);
        const auto by_barrier = AssetGrid::with_barrier_on_node(400.0, 101, level);
        ASSERT_TRUE(by_strike.ok());
        ASSERT_TRUE(by_barrier.ok());

        EXPECT_EQ(by_barrier.value().nodes(), by_strike.value().nodes()) << "level=" << level;
        EXPECT_EQ(by_barrier.value().spacing(), by_strike.value().spacing()) << "level=" << level;
        EXPECT_EQ(by_barrier.value().s_max(), by_strike.value().s_max()) << "level=" << level;
    }
}

// The barrier must lie inside the domain, or the boundary it defines is not on the
// grid at all and the solve would silently be a vanilla one.
TEST(AssetGridTest, RefusesABarrierOutsideTheDomain) {
    const auto beyond = AssetGrid::with_barrier_on_node(100.0, 101, 150.0);
    ASSERT_FALSE(beyond.ok());
    EXPECT_EQ(beyond.error().code, ErrorCode::InvalidArgument);
    // The message must name the barrier, not the strike: they are different
    // alignments with different consequences and a reader needs to know which failed.
    EXPECT_NE(beyond.error().describe().find("barrier"), std::string::npos)
        << beyond.error().describe();

    for (const double bad : {0.0, -10.0}) {
        EXPECT_FALSE(AssetGrid::with_barrier_on_node(400.0, 101, bad).ok()) << "B=" << bad;
    }
}

// A grid too coarse to resolve the barrier would place it on the S=0 boundary,
// where it would silently become the existing lower boundary rather than a barrier.
TEST(AssetGridTest, RefusesAGridTooCoarseToResolveTheBarrier) {
    const auto coarse = AssetGrid::with_barrier_on_node(10000.0, 3, 1.0);
    ASSERT_FALSE(coarse.ok());
    EXPECT_EQ(coarse.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks
