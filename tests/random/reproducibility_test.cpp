#include <diffusionworks/random/random_stream.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Reproducibility, stated at the strength it actually holds.
//
// "Reproducible" is not one property, and claiming it uniformly would overstate
// the guarantee. Three tiers exist and are tested differently:
//
//   tier 1  Philox counter output -- integer arithmetic only. Bit-exact on every
//           conforming platform. No floating point is involved at all.
//
//   tier 2  uniform_from_bits -- bit-exact, and provably so: the shift is
//           integer, the 52-bit integer converts to double exactly, +0.5 is
//           exact in that binade, and *2^-52 is a power of two. Nothing rounds.
//
//   tier 3  inverse_norm_cdf and everything downstream. The central branch
//           (~85% of draws) uses only + - * /, each correctly rounded by
//           IEEE-754, and with contraction disabled it is bit-exact across
//           conforming platforms. The tail branches (~15%) call std::log, whose
//           accuracy IEEE-754 does not specify: glibc, macOS libm and musl may
//           each return a different last bit. Those are bit-reproducible on a
//           fixed platform and toolchain, and agree to ~1 ulp across them.
//
// So tiers 1 and 2 are asserted bit-exactly and tier 3 with a tolerance.
// Demanding bit equality of tier 3 would give a test that passes on the machine
// it was written on and fails elsewhere for no defensible reason -- and the
// project would then be claiming a guarantee it had not earned.
//
// See docs/FLOATING-POINT-POLICY.md.
// ---------------------------------------------------------------------------

class ReproducibilityTierTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::filesystem::path path =
            std::filesystem::path(DW_TEST_DATA_DIR) / "references" / "random_stream_golden.json";

        std::ifstream stream(path);
        ASSERT_TRUE(stream.is_open())
            << "golden fixture not found: " << path
            << "\nregenerate with: python3 python/generate_random_stream_golden.py > "
               "data/references/random_stream_golden.json";

        ASSERT_NO_THROW(document_ = nlohmann::json::parse(stream));
        ASSERT_FALSE(document_["cases"].empty());
    }

    static RandomStream stream_for(const nlohmann::json& c) {
        return RandomStream(std::stoull(c["master_seed"].get<std::string>()),
                            static_cast<StreamPurpose>(c["purpose"].get<std::uint64_t>()),
                            std::stoull(c["path"].get<std::string>()));
    }

    static nlohmann::json document_;
};

nlohmann::json ReproducibilityTierTest::document_;

// The fixture must state which guarantee each field carries, or a later reader
// cannot tell a deliberate tolerance from a sloppy one.
TEST_F(ReproducibilityTierTest, FixtureDeclaresItsTiers) {
    ASSERT_TRUE(document_.contains("reproducibility_tiers"));
    const nlohmann::json& tiers = document_["reproducibility_tiers"];

    EXPECT_EQ(tiers["raw_words"]["guarantee"], "bit_exact_everywhere");
    EXPECT_EQ(tiers["uniforms_hex"]["guarantee"], "bit_exact_everywhere");
    EXPECT_EQ(tiers["normals"]["guarantee"], "tolerance_only_across_platforms");

    for (const std::string field : {"raw_words", "uniforms_hex", "normals"}) {
        EXPECT_FALSE(tiers[field]["reason"].get<std::string>().empty())
            << field << " declares a guarantee without saying why";
    }
}

// Tier 1: integer output, bit-exact everywhere.
TEST_F(ReproducibilityTierTest, PhiloxWordsAreBitExact) {
    for (const auto& c : document_["cases"]) {
        const std::uint64_t seed = std::stoull(c["master_seed"].get<std::string>());
        const std::uint64_t purpose = c["purpose"].get<std::uint64_t>();
        const std::uint64_t path = std::stoull(c["path"].get<std::string>());

        for (std::size_t position = 0; position < c["raw_words"].size(); ++position) {
            const std::uint64_t block = position / StreamLimits::kWordsPerBlock;
            const std::uint64_t word = position % StreamLimits::kWordsPerBlock;

            const PhiloxCounter output =
                Philox4x64::generate(PhiloxCounter{block, path, 0, 0}, PhiloxKey{seed, purpose});
            const auto expected =
                std::stoull(c["raw_words"][position].get<std::string>(), nullptr, 16);

            EXPECT_EQ(output[word], expected)
                << c["name"].get<std::string>() << " word " << position
                << " is not bit-exact, which integer arithmetic cannot excuse";
        }
    }
}

// Tier 2: the uniform transform is exact, so bit equality is the right assertion.
TEST_F(ReproducibilityTierTest, UniformsAreBitExact) {
    for (const auto& c : document_["cases"]) {
        RandomStream stream = stream_for(c);

        for (std::size_t position = 0; position < c["uniforms_hex"].size(); ++position) {
            const double actual = stream.next_uniform();
            const double expected = std::stod(c["uniforms_hex"][position].get<std::string>());

            // Bit equality, not EXPECT_DOUBLE_EQ: every step of the transform is
            // exact, so any difference is a defect rather than rounding.
            EXPECT_EQ(actual, expected) << c["name"].get<std::string>() << " uniform " << position
                                        << " is not bit-exact\n  actual   " << std::hexfloat
                                        << actual << "\n  expected " << expected;
        }
    }
}

// Tier 3: compared with a tolerance, because std::log's accuracy is not
// specified. The references are the true quantiles at 50 digits, so this
// validates AS 241 rather than merely recording its output.
TEST_F(ReproducibilityTierTest, NormalsMatchTheTrueQuantilesWithinTolerance) {
    for (const auto& c : document_["cases"]) {
        RandomStream stream = stream_for(c);

        for (std::size_t position = 0; position < c["normals"].size(); ++position) {
            const double actual = stream.next_normal();
            const double expected = std::stod(c["normals"][position].get<std::string>());

            const double allowed = 1e-15 + 1e-14 * std::abs(expected);
            EXPECT_NEAR(actual, expected, allowed)
                << c["name"].get<std::string>() << " normal " << position;
        }
    }
}

// Within one build, everything is bit-reproducible. This is the guarantee that
// actually matters for a stored result: rerunning it here must reproduce it
// exactly, tail branches included.
TEST_F(ReproducibilityTierTest, EverythingIsBitReproducibleWithinOneBuild) {
    for (const auto& c : document_["cases"]) {
        RandomStream first = stream_for(c);
        RandomStream second = stream_for(c);

        for (int i = 0; i < 64; ++i) {
            EXPECT_EQ(first.next_normal(), second.next_normal())
                << c["name"].get<std::string>() << " draw " << i;
        }
    }
}

// The coordinates a stored result records must address the same draws when
// rebuilt from them, including at the extreme end of the space.
TEST_F(ReproducibilityTierTest, MaximumCoordinatesAreAddressable) {
    bool found = false;
    for (const auto& c : document_["cases"]) {
        if (c["name"].get<std::string>() != "max_seed_max_path") {
            continue;
        }
        found = true;

        RandomStream stream(
            StreamLimits::kMaxMasterSeed, StreamPurpose::AssetShock, StreamLimits::kMaxPath);
        for (std::size_t position = 0; position < c["uniforms_hex"].size(); ++position) {
            EXPECT_EQ(stream.next_uniform(),
                      std::stod(c["uniforms_hex"][position].get<std::string>()))
                << "the largest coordinates do not reproduce at position " << position;
        }
    }
    EXPECT_TRUE(found) << "the fixture does not pin the edge of the coordinate space";
}

// ---------------------------------------------------------------------------
// Coordinate-space integrity
// ---------------------------------------------------------------------------

// The purpose values are frozen: changing one silently changes every stored
// result that used it while leaving the code looking correct. Pinned by literal
// so a renumbering cannot pass review unnoticed.
TEST(StreamCoordinatesTest, PurposeIdentifiersAreFrozen) {
    EXPECT_EQ(static_cast<std::uint64_t>(StreamPurpose::AssetShock), 0U);
    EXPECT_EQ(static_cast<std::uint64_t>(StreamPurpose::VarianceShock), 1U);
    EXPECT_EQ(static_cast<std::uint64_t>(StreamPurpose::Diagnostic), 1000U);
}

// The whole 64-bit space is usable for every coordinate, and nothing narrows.
TEST(StreamCoordinatesTest, LimitsSpanTheFullWordAndDoNotOverflow) {
    EXPECT_EQ(StreamLimits::kMaxMasterSeed, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(StreamLimits::kMaxPath, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(StreamLimits::kMaxPosition, std::numeric_limits<std::uint64_t>::max());

    // position / 4 must fit the counter word. It does, with two bits to spare.
    constexpr std::uint64_t max_block = StreamLimits::kMaxPosition / StreamLimits::kWordsPerBlock;
    EXPECT_LT(max_block, std::uint64_t{1} << 62U);
    EXPECT_LE(max_block, std::numeric_limits<std::uint64_t>::max());
}

// Injectivity, tested rather than argued.
//
// Philox is a bijection on (counter, key), so distinct coordinates give distinct
// draws provided the map onto (counter, key, lane) is itself injective. If it
// were not, two different paths would silently draw identical numbers while
// appearing independent -- and every error estimate built on them would be wrong
// in a way no moment test could reveal.
TEST(StreamCoordinatesTest, DistinctCoordinatesGiveDistinctDraws) {
    std::set<double> seen;

    for (const std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{12345}}) {
        for (const StreamPurpose purpose :
             {StreamPurpose::AssetShock, StreamPurpose::VarianceShock, StreamPurpose::Diagnostic}) {
            for (std::uint64_t path = 0; path < 8; ++path) {
                RandomStream stream(seed, purpose, path);
                for (std::uint64_t position = 0; position < 8; ++position) {
                    const double value = stream.next_uniform();
                    EXPECT_TRUE(seen.insert(value).second)
                        << "two distinct coordinate tuples produced the same draw: seed=" << seed
                        << " purpose=" << static_cast<std::uint64_t>(purpose) << " path=" << path
                        << " position=" << position;
                }
            }
        }
    }

    // 3 seeds x 3 purposes x 8 paths x 8 positions.
    EXPECT_EQ(seen.size(), 3U * 3U * 8U * 8U);
}

// Neighbouring coordinates are the dangerous case: seeds, paths and positions are
// all consecutive integers in practice, so a weakly mixed word would correlate
// exactly the tuples that occur together.
TEST(StreamCoordinatesTest, NeighbouringCoordinatesDoNotAlias) {
    const auto first_draw = [](std::uint64_t seed, StreamPurpose purpose, std::uint64_t path) {
        return RandomStream(seed, purpose, path).next_uniform();
    };

    const double base = first_draw(1000, StreamPurpose::AssetShock, 100);

    EXPECT_NE(base, first_draw(1001, StreamPurpose::AssetShock, 100)) << "adjacent seeds alias";
    EXPECT_NE(base, first_draw(1000, StreamPurpose::AssetShock, 101)) << "adjacent paths alias";
    EXPECT_NE(base, first_draw(1000, StreamPurpose::VarianceShock, 100))
        << "adjacent purposes alias";

    // The subtle one: a stream addressed by (seed, path) must not coincide with
    // (path, seed). A symmetric mixing would alias the diagonal.
    EXPECT_NE(first_draw(7, StreamPurpose::AssetShock, 9),
              first_draw(9, StreamPurpose::AssetShock, 7))
        << "the coordinate map is symmetric in seed and path, so transposed tuples alias";
}

// Seeking must not lose a lane. Positions straddling a block boundary are where
// an off-by-one in the block/lane split would show, and it would leave the
// moments untouched.
TEST(StreamCoordinatesTest, LaneSelectionIsExactAcrossBlockBoundaries) {
    RandomStream reference(555, StreamPurpose::AssetShock, 0);
    std::vector<double> sequential;
    sequential.reserve(64);
    for (int i = 0; i < 64; ++i) {
        sequential.push_back(reference.next_uniform());
    }

    for (std::uint64_t position = 0; position < 64; ++position) {
        RandomStream stream(555, StreamPurpose::AssetShock, 0);
        stream.seek(position);
        EXPECT_EQ(stream.next_uniform(), sequential[position])
            << "lane " << position % StreamLimits::kWordsPerBlock << " of block "
            << position / StreamLimits::kWordsPerBlock << " disagrees with sequential reading";
    }
}

// A position at the top of the space must still address the right lane; the
// block index there is 2^62 - 1.
TEST(StreamCoordinatesTest, ExtremePositionsAreAddressable) {
    RandomStream stream(1, StreamPurpose::AssetShock, 0);
    stream.seek(StreamLimits::kMaxPosition - 4);

    for (int i = 0; i < 4; ++i) {
        const double u = stream.next_uniform();
        EXPECT_GT(u, 0.0);
        EXPECT_LT(u, 1.0);
        EXPECT_TRUE(std::isfinite(inverse_norm_cdf(u)));
    }
}

}  // namespace
}  // namespace diffusionworks
