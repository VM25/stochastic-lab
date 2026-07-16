#include <diffusionworks/core/error.hpp>
#include <diffusionworks/numerics/tridiagonal.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace diffusionworks {
namespace {

/// Builds a system from its diagonals and a *known* solution, by computing the
/// right-hand side that solution implies.
///
/// This is the trick that makes the tests independent. Rather than solving a
/// system and checking the answer looks reasonable, we choose the answer first,
/// derive the b it must satisfy, and require the solver to recover it. The
/// expected result is then known exactly and owes nothing to the code under test.
TridiagonalSystem system_with_solution(std::vector<double> lower,
                                       std::vector<double> diagonal,
                                       std::vector<double> upper,
                                       const std::vector<double>& solution) {
    const std::size_t n = diagonal.size();
    std::vector<double> rhs(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = diagonal[i] * solution[i];
        if (i > 0) {
            rhs[i] += lower[i] * solution[i - 1];
        }
        if (i + 1 < n) {
            rhs[i] += upper[i] * solution[i + 1];
        }
    }
    return TridiagonalSystem{.lower = std::move(lower),
                             .diagonal = std::move(diagonal),
                             .upper = std::move(upper),
                             .rhs = std::move(rhs)};
}

// ---------------------------------------------------------------------------
// Exactness
// ---------------------------------------------------------------------------

// A 1x1 system is a division. If this is wrong, nothing else can be right.
TEST(TridiagonalTest, SolvesASingleRow) {
    const TridiagonalSystem system{
        .lower = {0.0}, .diagonal = {4.0}, .upper = {0.0}, .rhs = {12.0}};

    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok()) << solution.error().describe();
    ASSERT_EQ(solution.value().values.size(), 1U);
    EXPECT_DOUBLE_EQ(solution.value().values[0], 3.0);
}

// A hand-solvable 3x3, worked out independently:
//
//     [ 2 1 0 ] [x0]   [ 5]
//     [ 1 3 1 ] [x1] = [10]
//     [ 0 1 2 ] [x2]   [ 9]
//
// From row 0: x0 = (5 - x1)/2. From row 2: x2 = (9 - x1)/2. Substituting into
// row 1: (5-x1)/2 + 3*x1 + (9-x1)/2 = 10  =>  7 + 2*x1 = 10  =>  x1 = 1.5,
// so x0 = 1.75 and x2 = 3.75.
TEST(TridiagonalTest, MatchesAHandSolvedSystem) {
    const TridiagonalSystem system{.lower = {0.0, 1.0, 1.0},
                                   .diagonal = {2.0, 3.0, 2.0},
                                   .upper = {1.0, 1.0, 0.0},
                                   .rhs = {5.0, 10.0, 9.0}};

    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok()) << solution.error().describe();

    EXPECT_NEAR(solution.value().values[0], 1.75, 1e-14);
    EXPECT_NEAR(solution.value().values[1], 1.5, 1e-14);
    EXPECT_NEAR(solution.value().values[2], 3.75, 1e-14);
}

// The identity matrix must return the right-hand side untouched. A solver that
// scales, offsets, or reverses its output fails here and nowhere subtler.
TEST(TridiagonalTest, IdentityReturnsTheRightHandSide) {
    const std::size_t n = 7;
    TridiagonalSystem system{.lower = std::vector<double>(n, 0.0),
                             .diagonal = std::vector<double>(n, 1.0),
                             .upper = std::vector<double>(n, 0.0),
                             .rhs = {}};
    for (std::size_t i = 0; i < n; ++i) {
        system.rhs.push_back(static_cast<double>(i) * 1.5 - 2.0);
    }

    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(solution.value().values[i], system.rhs[i]) << "at index " << i;
    }
}

// Recovering a solution chosen in advance, on a system large enough that an
// index error anywhere would surface.
TEST(TridiagonalTest, RecoversAKnownSolution) {
    const std::size_t n = 200;
    std::vector<double> lower(n, -1.0);
    std::vector<double> diagonal(n, 4.0);
    std::vector<double> upper(n, -1.0);
    lower[0] = 0.0;
    upper[n - 1] = 0.0;

    std::vector<double> expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Something with structure, so a solver that returns a constant or a
        // shifted copy cannot pass.
        expected[i] = std::sin(0.1 * static_cast<double>(i)) * 10.0 + static_cast<double>(i) * 0.01;
    }

    const auto system = system_with_solution(lower, diagonal, upper, expected);
    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok()) << solution.error().describe();

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(solution.value().values[i], expected[i], 1e-12) << "at index " << i;
    }
}

// The asymmetric case. A symmetric system hides a transposed lower/upper: swapping
// them changes nothing when they are equal. This one is only solvable correctly if
// each diagonal is used in its own place.
TEST(TridiagonalTest, DistinguishesTheLowerFromTheUpperDiagonal) {
    // Deliberately asymmetric: lower and upper differ in every row.
    const std::vector<double> lower{0.0, -1.0, -2.0, -3.0};
    const std::vector<double> diagonal{10.0, 12.0, 14.0, 16.0};
    const std::vector<double> upper{-5.0, -6.0, -7.0, 0.0};
    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};

    const auto system = system_with_solution(lower, diagonal, upper, expected);
    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok()) << solution.error().describe();

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(solution.value().values[i], expected[i], 1e-13) << "at index " << i;
    }

    // And the transposed system must give a *different* answer, or the test above
    // proves nothing about which diagonal went where.
    const auto transposed = system_with_solution(upper, diagonal, lower, expected);
    ASSERT_NE(transposed.rhs, system.rhs)
        << "the asymmetric system and its transpose share a right-hand side, so this test cannot "
           "detect a swapped diagonal";
}

// ---------------------------------------------------------------------------
// The residual: an independent check on the solve
// ---------------------------------------------------------------------------

TEST(TridiagonalTest, ResidualIsTinyForASolvedSystem) {
    const std::size_t n = 500;
    std::vector<double> lower(n, -1.0);
    std::vector<double> diagonal(n, 2.5);
    std::vector<double> upper(n, -1.0);
    lower[0] = 0.0;
    upper[n - 1] = 0.0;

    std::vector<double> expected(n);
    std::mt19937_64 rng(20260716);
    std::uniform_real_distribution<double> draw(-5.0, 5.0);
    for (double& v : expected) {
        v = draw(rng);
    }

    const auto system = system_with_solution(lower, diagonal, upper, expected);
    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok());

    const auto residual = tridiagonal_residual(system, solution.value().values);
    ASSERT_TRUE(residual.ok()) << residual.error().describe();
    EXPECT_LT(residual.value(), 1e-11) << "||Ax - b||_inf = " << residual.value();
}

// The residual must be able to *fail*. A check that passes on a wrong answer is
// not a check, so it is shown rejecting one.
TEST(TridiagonalTest, ResidualDetectsAWrongSolution) {
    const TridiagonalSystem system{.lower = {0.0, 1.0, 1.0},
                                   .diagonal = {2.0, 3.0, 2.0},
                                   .upper = {1.0, 1.0, 0.0},
                                   .rhs = {5.0, 10.0, 9.0}};

    const std::vector<double> correct{1.75, 1.5, 3.75};
    const std::vector<double> wrong{1.75, 1.6, 3.75};

    EXPECT_LT(tridiagonal_residual(system, correct).value(), 1e-13);
    EXPECT_GT(tridiagonal_residual(system, wrong).value(), 0.1);
}

// ---------------------------------------------------------------------------
// Diagonal dominance: the algorithm's precondition
// ---------------------------------------------------------------------------

TEST(TridiagonalTest, RecognisesDiagonalDominance) {
    // Strictly dominant: 4 > 1 + 1.
    const TridiagonalSystem dominant{.lower = {0.0, -1.0, -1.0},
                                     .diagonal = {4.0, 4.0, 4.0},
                                     .upper = {-1.0, -1.0, 0.0},
                                     .rhs = {1.0, 1.0, 1.0}};
    EXPECT_TRUE(is_diagonally_dominant(dominant));

    // Row 1 fails: 1 < 1 + 1.
    const TridiagonalSystem not_dominant{.lower = {0.0, -1.0, -1.0},
                                         .diagonal = {4.0, 1.0, 4.0},
                                         .upper = {-1.0, -1.0, 0.0},
                                         .rhs = {1.0, 1.0, 1.0}};
    EXPECT_FALSE(is_diagonally_dominant(not_dominant));
}

// Weak dominance everywhere with no strict row is not enough for the guarantee, so
// it is not reported as dominant.
TEST(TridiagonalTest, WeakDominanceWithoutAStrictRowIsNotDominant) {
    const TridiagonalSystem weak{.lower = {0.0, -1.0, -2.0},
                                 .diagonal = {1.0, 2.0, 2.0},
                                 .upper = {-1.0, -1.0, 0.0},
                                 .rhs = {1.0, 1.0, 1.0}};
    // Row 0: |1| == |0| + |-1|. Row 1: |2| == |-1| + |-1|. Row 2: |2| == |-2| + |0|.
    // Weak in every row, strict in none.
    EXPECT_FALSE(is_diagonally_dominant(weak));
}

// The corner entries are not part of the matrix and must not count toward
// dominance. If they did, a perfectly good system with junk in lower[0] would be
// rejected -- and callers would learn to zero it, hiding real errors.
TEST(TridiagonalTest, IgnoresTheUnusedCornerEntries) {
    TridiagonalSystem system{.lower = {999.0, -1.0, -1.0},
                             .diagonal = {4.0, 4.0, 4.0},
                             .upper = {-1.0, -1.0, 999.0},
                             .rhs = {1.0, 1.0, 1.0}};
    EXPECT_TRUE(is_diagonally_dominant(system))
        << "lower[0] and upper[n-1] multiply nothing and must not count";

    // And the junk must not reach the answer either.
    const auto with_junk = solve_tridiagonal(system);
    ASSERT_TRUE(with_junk.ok()) << with_junk.error().describe();

    system.lower[0] = 0.0;
    system.upper[2] = 0.0;
    const auto without_junk = solve_tridiagonal(system);
    ASSERT_TRUE(without_junk.ok());

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(with_junk.value().values[i], without_junk.value().values[i])
            << "at index " << i;
    }
}

// Regression against a mathematical overstatement.
//
// An earlier version of this solver refused every non-dominant system by default,
// on the stated grounds that the Thomas algorithm "requires" diagonal dominance.
// It does not. Dominance is *sufficient* for stability -- it rules out a vanishing
// pivot and bounds the growth factor -- but it is not *necessary*, and refusing on
// its absence rejects systems the algorithm solves perfectly well.
//
// This is the counterexample. The matrix is symmetric positive definite and not
// diagonally dominant (row 1: |2| < |1.5| + |1.5|), and Thomas solves it to
// machine precision. Any future reintroduction of a dominance gate fails here.
TEST(TridiagonalTest, SolvesANonDominantSystemThatIsNonethelessWellBehaved) {
    // [ 2.0  1.5  0  ]
    // [ 1.5  2.0  1.5]
    // [ 0    1.5  2.0]
    //
    // Row 1 is not dominant: |2.0| < |1.5| + |1.5|. The matrix is symmetric and
    // non-singular, and the elimination's pivots stay far from zero, so Thomas
    // solves it to machine precision -- which is precisely the point. Dominance
    // would have been enough to promise that in advance; its absence promises
    // nothing either way, and here the answer is exact.
    const std::vector<double> lower{0.0, 1.5, 1.5};
    const std::vector<double> diagonal{2.0, 2.0, 2.0};
    const std::vector<double> upper{1.5, 1.5, 0.0};
    const std::vector<double> expected{1.0, -2.0, 3.0};

    const auto system = system_with_solution(lower, diagonal, upper, expected);
    ASSERT_FALSE(is_diagonally_dominant(system)) << "the premise of this test is a non-dominant "
                                                    "matrix";

    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok())
        << "a non-dominant system must not be refused: dominance is sufficient for stability, "
           "not necessary. "
        << solution.error().describe();

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(solution.value().values[i], expected[i], 1e-13) << "at index " << i;
    }

    // The guarantee is absent and the solver says so, rather than pretending
    // either way.
    EXPECT_FALSE(solution.value().diagnostics.diagonally_dominant);
    EXPECT_GT(solution.value().diagnostics.smallest_pivot_ratio, 1e-3)
        << "no pivot came close to vanishing, which is why this solve is trustworthy";
}

// A larger non-dominant system, to show the first is not a three-row accident.
TEST(TridiagonalTest, SolvesALargerNonDominantSystem) {
    const std::size_t n = 100;
    std::vector<double> lower(n, 0.9);
    std::vector<double> diagonal(n, 1.0);
    std::vector<double> upper(n, 0.9);
    lower[0] = 0.0;
    upper[n - 1] = 0.0;

    std::vector<double> expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = std::cos(0.05 * static_cast<double>(i));
    }

    const auto system = system_with_solution(lower, diagonal, upper, expected);
    ASSERT_FALSE(is_diagonally_dominant(system)) << "row scale 1.0 < 0.9 + 0.9";

    const auto solution = solve_tridiagonal(system);
    ASSERT_TRUE(solution.ok()) << solution.error().describe();
    EXPECT_FALSE(solution.value().diagnostics.diagonally_dominant);

    const auto residual = tridiagonal_residual(system, solution.value().values);
    ASSERT_TRUE(residual.ok());
    EXPECT_LT(residual.value(), 1e-9) << "residual " << residual.value();
}

// The diagnostics are the point: a caller who needs the guarantee can ask for it,
// and one who merely needs an answer can check what actually happened.
TEST(TridiagonalTest, ReportsDominanceAndPivotHealthAsDiagnostics) {
    const TridiagonalSystem dominant{.lower = {0.0, -1.0, -1.0},
                                     .diagonal = {4.0, 4.0, 4.0},
                                     .upper = {-1.0, -1.0, 0.0},
                                     .rhs = {1.0, 1.0, 1.0}};

    const auto solution = solve_tridiagonal(dominant);
    ASSERT_TRUE(solution.ok());
    EXPECT_TRUE(solution.value().diagnostics.diagonally_dominant);
    // Strongly dominant: pivots stay near the diagonal, well clear of zero.
    EXPECT_GT(solution.value().diagnostics.smallest_pivot_ratio, 0.5);
    EXPECT_TRUE(std::isfinite(solution.value().diagnostics.smallest_pivot_ratio));
}

// ---------------------------------------------------------------------------
// Failure modes
// ---------------------------------------------------------------------------

// A vanishing pivot must be reported, not divided by. Dividing produces an
// infinity and then a NaN; a *nearly* vanishing pivot produces a finite, enormous,
// entirely wrong answer that nothing downstream would question.
TEST(TridiagonalTest, ReportsAVanishingPivot) {
    // Row 0's diagonal is zero: the elimination cannot start.
    const TridiagonalSystem zero_pivot{
        .lower = {0.0, 1.0}, .diagonal = {0.0, 1.0}, .upper = {1.0, 0.0}, .rhs = {1.0, 1.0}};

    const auto solution = solve_tridiagonal(zero_pivot);
    ASSERT_FALSE(solution.ok());
    EXPECT_EQ(solution.error().code, ErrorCode::NonFiniteValue);
    EXPECT_NE(solution.error().describe().find("row 0"), std::string::npos)
        << solution.error().describe();
}

// A pivot that vanishes mid-elimination, rather than at the start. The matrix is
// singular but no single coefficient looks wrong.
TEST(TridiagonalTest, ReportsAPivotThatVanishesDuringElimination) {
    // [1 1] [x0]   [1]
    // [1 1] [x1] = [2]   -- singular, and inconsistent.
    const TridiagonalSystem singular{
        .lower = {0.0, 1.0}, .diagonal = {1.0, 1.0}, .upper = {1.0, 0.0}, .rhs = {1.0, 2.0}};

    const auto solution = solve_tridiagonal(singular);
    ASSERT_FALSE(solution.ok());
    EXPECT_EQ(solution.error().code, ErrorCode::NonFiniteValue);
    EXPECT_NE(solution.error().describe().find("row 1"), std::string::npos)
        << solution.error().describe();
}

TEST(TridiagonalTest, RejectsMismatchedDiagonalLengths) {
    const TridiagonalSystem ragged{
        .lower = {0.0, 1.0}, .diagonal = {2.0, 2.0, 2.0}, .upper = {1.0, 0.0}, .rhs = {1.0, 1.0}};

    const auto solution = solve_tridiagonal(ragged);
    ASSERT_FALSE(solution.ok());
    EXPECT_EQ(solution.error().code, ErrorCode::InvalidArgument);
}

TEST(TridiagonalTest, RejectsAnEmptySystem) {
    EXPECT_FALSE(solve_tridiagonal(TridiagonalSystem{}).ok());
}

TEST(TridiagonalTest, RejectsNonFiniteCoefficients) {
    for (const double bad :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        TridiagonalSystem system{
            .lower = {0.0, 1.0}, .diagonal = {4.0, 4.0}, .upper = {1.0, 0.0}, .rhs = {1.0, 1.0}};
        system.diagonal[1] = bad;

        const auto solution = solve_tridiagonal(system);
        ASSERT_FALSE(solution.ok());
        EXPECT_EQ(solution.error().code, ErrorCode::NonFiniteValue);
    }
}

TEST(TridiagonalTest, ResidualRejectsAMismatchedSolutionLength) {
    const TridiagonalSystem system{
        .lower = {0.0, 1.0}, .diagonal = {4.0, 4.0}, .upper = {1.0, 0.0}, .rhs = {1.0, 1.0}};
    const std::vector<double> too_short{1.0};

    EXPECT_FALSE(tridiagonal_residual(system, too_short).ok());
}

// ---------------------------------------------------------------------------
// Conditioning
// ---------------------------------------------------------------------------

// The system the PDE schemes actually produce: strongly dominant, hundreds of
// rows. The residual must stay at rounding level rather than growing with n.
TEST(TridiagonalTest, StaysAccurateOnALargeDominantSystem) {
    for (const std::size_t n : {50U, 500U, 5000U}) {
        std::vector<double> lower(n, -1.0);
        std::vector<double> diagonal(n, 2.0 + 1e-6);  // barely dominant
        std::vector<double> upper(n, -1.0);
        lower[0] = 0.0;
        upper[n - 1] = 0.0;

        std::vector<double> expected(n);
        for (std::size_t i = 0; i < n; ++i) {
            expected[i] = 1.0 + std::cos(0.01 * static_cast<double>(i));
        }

        const auto system = system_with_solution(lower, diagonal, upper, expected);
        const auto solution = solve_tridiagonal(system);
        ASSERT_TRUE(solution.ok()) << "n = " << n << ": " << solution.error().describe();

        const auto residual = tridiagonal_residual(system, solution.value().values);
        ASSERT_TRUE(residual.ok());
        EXPECT_LT(residual.value(), 1e-10) << "n = " << n;

        // The residual being small does not by itself mean the answer is right --
        // a badly conditioned system can have a small residual and a large error.
        // Checked against the known solution too.
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(solution.value().values[i] - expected[i]));
        }
        EXPECT_LT(worst, 1e-8) << "n = " << n << ": worst componentwise error";
    }
}

}  // namespace
}  // namespace diffusionworks
