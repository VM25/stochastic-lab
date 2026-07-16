#include <diffusionworks/core/error.hpp>
#include <diffusionworks/pde/black_scholes_operator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

struct Setup {
    MarketState market;
    BlackScholesModel model;
    AssetGrid grid;
    OperatorCoefficients coefficients;
};

Setup make(double rate, double dividend, double volatility, double s_max, std::int64_t nodes) {
    auto market = MarketState::create(100.0, rate, dividend).value();
    auto model = BlackScholesModel::create(volatility).value();
    auto grid = AssetGrid::uniform(s_max, nodes).value();
    auto coefficients = black_scholes_coefficients(market, model, grid);
    EXPECT_TRUE(coefficients.ok()) << coefficients.error().describe();
    return Setup{market, model, grid, coefficients.value()};
}

// ---------------------------------------------------------------------------
// The coefficients against the PDE they discretise
//
// These values were derived independently -- symbolically, by sympy, from the
// Black-Scholes PDE with central differences -- rather than read back from the
// implementation:
//
//     a_i = (1/2) sigma^2 i^2 - (1/2) (r-q) i
//     b_i = -sigma^2 i^2 - r
//     c_i = (1/2) sigma^2 i^2 + (1/2) (r-q) i
// ---------------------------------------------------------------------------

TEST(BlackScholesOperatorTest, MatchesTheCoefficientsDerivedFromThePde) {
    const double rate = 0.05;
    const double dividend = 0.02;
    const double volatility = 0.3;
    const auto s = make(rate, dividend, volatility, 400.0, 21);

    const double variance = volatility * volatility;
    const double carry = rate - dividend;

    for (std::int64_t i = 1; i < s.grid.nodes() - 1; ++i) {
        const auto index = static_cast<double>(i);
        const auto k = static_cast<std::size_t>(i);

        EXPECT_NEAR(
            s.coefficients.a[k], 0.5 * variance * index * index - 0.5 * carry * index, 1e-12)
            << "a at node " << i;
        EXPECT_NEAR(s.coefficients.b[k], -variance * index * index - rate, 1e-12)
            << "b at node " << i;
        EXPECT_NEAR(
            s.coefficients.c[k], 0.5 * variance * index * index + 0.5 * carry * index, 1e-12)
            << "c at node " << i;
    }
}

// The structural identity: L applied to a constant V must give -rV, because a
// constant has no derivatives and only the -rV term survives. So a+b+c = -r
// exactly, at every interior node.
//
// This catches a whole class of error a spot-check of individual coefficients
// would miss -- any mistake that redistributes weight between the three
// diagonals while leaving each looking plausible.
TEST(BlackScholesOperatorTest, RowsSumToMinusTheRate) {
    for (const double rate : {0.0, 0.02, 0.05, 0.10}) {
        for (const double dividend : {0.0, 0.03, 0.07}) {
            for (const double volatility : {0.05, 0.2, 0.6}) {
                const auto s = make(rate, dividend, volatility, 400.0, 41);
                for (std::int64_t i = 1; i < s.grid.nodes() - 1; ++i) {
                    const auto k = static_cast<std::size_t>(i);
                    const double sum =
                        s.coefficients.a[k] + s.coefficients.b[k] + s.coefficients.c[k];
                    EXPECT_NEAR(sum, -rate, 1e-10) << "node " << i << " with r=" << rate
                                                   << " q=" << dividend << " sigma=" << volatility;
                }
            }
        }
    }
}

// The boundaries carry conditions, not operator rows, so their coefficients stay
// zero. An index therefore means the same thing here as in the grid, which is what
// lets the assembly loops be written without an offset.
TEST(BlackScholesOperatorTest, BoundaryNodesCarryNoOperatorRow) {
    const auto s = make(0.05, 0.0, 0.3, 400.0, 21);
    const auto last = static_cast<std::size_t>(s.grid.nodes() - 1);

    EXPECT_DOUBLE_EQ(s.coefficients.a[0], 0.0);
    EXPECT_DOUBLE_EQ(s.coefficients.b[0], 0.0);
    EXPECT_DOUBLE_EQ(s.coefficients.c[0], 0.0);
    EXPECT_DOUBLE_EQ(s.coefficients.a[last], 0.0);
    EXPECT_DOUBLE_EQ(s.coefficients.b[last], 0.0);
    EXPECT_DOUBLE_EQ(s.coefficients.c[last], 0.0);
}

// The coefficients depend on the node index, not on dS. Two grids with different
// spacing but the same node count produce identical operators, because
// sigma^2 S_i^2 / dS^2 = sigma^2 i^2 exactly. That is the property that lets the
// implementation avoid dividing by dS^2 at all.
TEST(BlackScholesOperatorTest, CoefficientsDependOnTheIndexNotTheSpacing) {
    const auto coarse = make(0.05, 0.02, 0.3, 200.0, 21);
    const auto fine = make(0.05, 0.02, 0.3, 800.0, 21);

    for (std::size_t i = 0; i < 21; ++i) {
        EXPECT_DOUBLE_EQ(coarse.coefficients.a[i], fine.coefficients.a[i]) << "a at " << i;
        EXPECT_DOUBLE_EQ(coarse.coefficients.b[i], fine.coefficients.b[i]) << "b at " << i;
        EXPECT_DOUBLE_EQ(coarse.coefficients.c[i], fine.coefficients.c[i]) << "c at " << i;
    }
}

// ---------------------------------------------------------------------------
// The M-matrix sign structure
//
// What it buys is narrow and worth stating: non-negative off-diagonals make the
// implicit operator's inverse entrywise non-negative, so a non-negative payoff
// stays non-negative at every step for any dtau. It is not a statement about
// accuracy -- a monotone scheme can be monotone and wrong.
// ---------------------------------------------------------------------------

TEST(BlackScholesOperatorTest, SignStructureHoldsUnderOrdinaryParameters) {
    const auto s = make(0.05, 0.0, 0.3, 400.0, 101);
    const auto diagnostics = diagnose_operator(s.market, s.model, s.grid, s.coefficients);

    EXPECT_TRUE(diagnostics.sign_structure_holds())
        << "sigma=0.3, r-q=0.05: the Peclet threshold is r-q/sigma^2 = 0.56, below node 1";
    EXPECT_NEAR(diagnostics.peclet_threshold_index, 0.05 / 0.09, 1e-12);
}

// The cell-Peclet failure, demonstrated rather than asserted. With small sigma and
// large carry the threshold (r-q)/sigma^2 rises above the low node indices, and
// a_i goes negative there.
TEST(BlackScholesOperatorTest, DetectsWhereConvectionOverwhelmsDiffusion) {
    // sigma = 0.05, r-q = 0.15: threshold = 0.15/0.0025 = 60. Nodes 1..59 fail.
    const auto s = make(0.15, 0.0, 0.05, 400.0, 101);
    const auto diagnostics = diagnose_operator(s.market, s.model, s.grid, s.coefficients);

    EXPECT_FALSE(diagnostics.sign_structure_holds());
    EXPECT_NEAR(diagnostics.peclet_threshold_index, 60.0, 1e-9);

    // Exactly the nodes below the threshold, and no others.
    ASSERT_FALSE(diagnostics.negative_sub_diagonal_nodes.empty());
    EXPECT_EQ(diagnostics.negative_sub_diagonal_nodes.front(), 1);
    EXPECT_EQ(diagnostics.negative_sub_diagonal_nodes.back(), 59);

    for (const std::int64_t node : diagnostics.negative_sub_diagonal_nodes) {
        EXPECT_LT(s.coefficients.a[static_cast<std::size_t>(node)], 0.0)
            << "node " << node << " was reported as failing but its a is non-negative";
    }
}

// The mirror case: a sufficiently negative carry drives c_i negative instead.
TEST(BlackScholesOperatorTest, DetectsTheSuperDiagonalFailureUnderNegativeCarry) {
    // r - q = -0.15, sigma = 0.05: c_i < 0 for i < 60.
    const auto s = make(0.0, 0.15, 0.05, 400.0, 101);
    const auto diagnostics = diagnose_operator(s.market, s.model, s.grid, s.coefficients);

    EXPECT_FALSE(diagnostics.sign_structure_holds());
    ASSERT_FALSE(diagnostics.negative_super_diagonal_nodes.empty());
    EXPECT_TRUE(diagnostics.negative_sub_diagonal_nodes.empty())
        << "with negative carry the sub-diagonal gains, it does not lose";
}

// Zero volatility leaves pure convection: no diffusion at any node, so the
// threshold is infinite rather than a division by zero.
TEST(BlackScholesOperatorTest, ReportsAnInfiniteThresholdWithoutDiffusion) {
    const auto s = make(0.05, 0.0, 0.0, 400.0, 21);
    const auto diagnostics = diagnose_operator(s.market, s.model, s.grid, s.coefficients);

    EXPECT_TRUE(std::isinf(diagnostics.peclet_threshold_index));
    EXPECT_FALSE(diagnostics.sign_structure_holds())
        << "with no diffusion every interior node is pure convection";
}

// The threshold is reported even when nothing fails, so a caller can see the
// margin rather than only whether the line was crossed.
TEST(BlackScholesOperatorTest, ReportsTheThresholdEvenWhenNoNodeFails) {
    const auto s = make(0.05, 0.0, 0.3, 400.0, 101);
    const auto diagnostics = diagnose_operator(s.market, s.model, s.grid, s.coefficients);

    EXPECT_TRUE(diagnostics.sign_structure_holds());
    EXPECT_GT(diagnostics.peclet_threshold_index, 0.0)
        << "the margin is visible even on a healthy grid";
}

// ---------------------------------------------------------------------------
// The explicit stability limit
// ---------------------------------------------------------------------------

// Read from the assembled coefficients, and agreeing with the von Neumann bound
// 1/(sigma^2 N^2 + r) for this discretisation. The agreement is the check: two
// routes to the same number, one from the matrix and one from the analysis.
TEST(BlackScholesOperatorTest, StabilityLimitMatchesTheVonNeumannBound) {
    for (const std::int64_t nodes : {21, 51, 101}) {
        for (const double volatility : {0.1, 0.3, 0.6}) {
            const double rate = 0.05;
            const auto s = make(rate, 0.0, volatility, 400.0, nodes);

            const auto limit = explicit_stability_limit(s.coefficients);
            ASSERT_TRUE(limit.ok()) << limit.error().describe();

            // The largest interior index is nodes-2.
            const auto top = static_cast<double>(nodes - 2);
            const double expected = 1.0 / (volatility * volatility * top * top + rate);

            EXPECT_NEAR(limit.value(), expected, 1e-12 * expected)
                << "nodes " << nodes << ", sigma " << volatility;
        }
    }
}

// The limit tightens as dS^2, which is the practical cost of the explicit scheme:
// doubling the spatial resolution quarters the affordable time step. Measured
// rather than asserted from the formula.
TEST(BlackScholesOperatorTest, StabilityLimitTightensQuadraticallyWithRefinement) {
    const auto coarse = make(0.05, 0.0, 0.3, 400.0, 51);
    const auto fine = make(0.05, 0.0, 0.3, 400.0, 101);

    const double coarse_limit = explicit_stability_limit(coarse.coefficients).value();
    const double fine_limit = explicit_stability_limit(fine.coefficients).value();

    // Roughly 4x, since the node count roughly doubled and the limit goes as 1/N^2.
    const double ratio = coarse_limit / fine_limit;
    EXPECT_GT(ratio, 3.5);
    EXPECT_LT(ratio, 4.5);
}

// Pure convection with no discounting: nothing to amplify, so the limit is
// unbounded. A true statement about a degenerate operator, reported rather than
// refused.
TEST(BlackScholesOperatorTest, ReportsAnUnboundedLimitForADegenerateOperator) {
    const auto s = make(0.0, 0.0, 0.0, 400.0, 21);
    const auto limit = explicit_stability_limit(s.coefficients);

    ASSERT_TRUE(limit.ok());
    EXPECT_TRUE(std::isinf(limit.value()));
}

TEST(BlackScholesOperatorTest, RejectsAnOperatorWithNoInteriorRow) {
    OperatorCoefficients tiny;
    tiny.a = {0.0, 0.0};
    tiny.b = {0.0, 0.0};
    tiny.c = {0.0, 0.0};

    EXPECT_FALSE(explicit_stability_limit(tiny).ok());
}

}  // namespace
}  // namespace diffusionworks
