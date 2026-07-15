#include <diffusionworks/random/philox.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Bit-exact validation of Philox4x64-10.
//
// A random number generator cannot be validated by looking at its output. Any
// plausible stream of bits looks like any other, which is exactly the situation
// this project treats as untrustworthy: a subtly wrong implementation -- a
// transposed counter word, a key bumped one round early -- still produces
// uniform-looking noise that passes every statistical test the sample size can
// support, while silently correlating streams that are assumed independent.
//
// The only meaningful check is bit-exact reproduction of a reference computed
// elsewhere. data/references/philox_kat.json is anchored to the published
// Random123 vectors and independently confirmed against numpy.random.Philox; see
// python/generate_philox_kat.py.
// ---------------------------------------------------------------------------

class PhiloxKatTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::filesystem::path path =
            std::filesystem::path(DW_TEST_DATA_DIR) / "references" / "philox_kat.json";

        std::ifstream stream(path);
        ASSERT_TRUE(stream.is_open())
            << "KAT fixture not found: " << path
            << "\nregenerate with: python3 python/generate_philox_kat.py > "
               "data/references/philox_kat.json";

        ASSERT_NO_THROW(document_ = nlohmann::json::parse(stream));
    }

    static std::uint64_t from_hex(const nlohmann::json& value) {
        return std::stoull(value.get<std::string>(), nullptr, 16);
    }

    static nlohmann::json document_;
};

nlohmann::json PhiloxKatTest::document_;

TEST_F(PhiloxKatTest, ReproducesEveryKnownAnswerVector) {
    ASSERT_FALSE(document_["cases"].empty());

    for (const auto& c : document_["cases"]) {
        PhiloxCounter counter{};
        PhiloxKey key{};
        PhiloxCounter expected{};

        for (std::size_t i = 0; i < 4; ++i) {
            counter[i] = from_hex(c["counter"][i]);
            expected[i] = from_hex(c["expected"][i]);
        }
        for (std::size_t i = 0; i < 2; ++i) {
            key[i] = from_hex(c["key"][i]);
        }

        const PhiloxCounter actual = Philox4x64::generate(counter, key);

        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(actual[i], expected[i])
                << "case '" << c["name"].get<std::string>() << "' ("
                << c["source"].get<std::string>() << ") word " << i << "\n  expected 0x" << std::hex
                << expected[i] << "\n  actual   0x" << actual[i];
        }
    }
}

// The fixture is only worth anything if it is pinned to the literature rather
// than to whatever this project happened to compute.
TEST_F(PhiloxKatTest, FixtureIsAnchoredToThePublishedVectors) {
    ASSERT_TRUE(document_.contains("provenance"));
    EXPECT_TRUE(document_["provenance"]["verified_against_published_kat"].get<bool>());

    // At least one case must come from Random123 itself.
    bool has_published = false;
    for (const auto& c : document_["cases"]) {
        if (c["source"].get<std::string>().find("Random123") != std::string::npos) {
            has_published = true;
        }
    }
    EXPECT_TRUE(has_published) << "the fixture cites no published vector";

    ASSERT_TRUE(document_.contains("algorithm"));
    EXPECT_EQ(document_["algorithm"]["rounds"].get<int>(), Philox4x64::kRounds);
}

// The constants are the algorithm. A typo in one would still yield noise that
// looks perfectly random, so they are pinned against the paper via the fixture.
TEST_F(PhiloxKatTest, ConstantsMatchThePublishedAlgorithm) {
    const nlohmann::json& constants = document_["algorithm"]["constants"];

    EXPECT_EQ(from_hex(constants["M0"]), Philox4x64::kMultiplier0);
    EXPECT_EQ(from_hex(constants["M1"]), Philox4x64::kMultiplier1);
    EXPECT_EQ(from_hex(constants["W0"]), Philox4x64::kWeyl0);
    EXPECT_EQ(from_hex(constants["W1"]), Philox4x64::kWeyl1);
}

// ---------------------------------------------------------------------------
// Structural properties
// ---------------------------------------------------------------------------

TEST(PhiloxTest, IsDeterministic) {
    const PhiloxCounter counter{7, 42, 0, 0};
    const PhiloxKey key{12345, 6789};

    const PhiloxCounter first = Philox4x64::generate(counter, key);
    for (int repeat = 0; repeat < 10; ++repeat) {
        EXPECT_EQ(Philox4x64::generate(counter, key), first);
    }
}

// Every counter word must reach the output, or part of the addressing space is
// silently aliased: two different paths would draw identical numbers while
// appearing independent.
TEST(PhiloxTest, EveryCounterWordChangesTheOutput) {
    const PhiloxKey key{0, 0};
    const PhiloxCounter base = Philox4x64::generate(PhiloxCounter{0, 0, 0, 0}, key);

    for (std::size_t word = 0; word < 4; ++word) {
        PhiloxCounter perturbed{0, 0, 0, 0};
        perturbed[word] = 1;
        EXPECT_NE(Philox4x64::generate(perturbed, key), base)
            << "counter word " << word << " does not affect the output";
    }
}

TEST(PhiloxTest, EveryKeyWordChangesTheOutput) {
    const PhiloxCounter counter{0, 0, 0, 0};
    const PhiloxCounter base = Philox4x64::generate(counter, PhiloxKey{0, 0});

    for (std::size_t word = 0; word < 2; ++word) {
        PhiloxKey perturbed{0, 0};
        perturbed[word] = 1;
        EXPECT_NE(Philox4x64::generate(counter, perturbed), base)
            << "key word " << word << " does not affect the output";
    }
}

// Adjacent counters must produce unrelated output. This is the property the
// whole design rests on: path indices are consecutive integers, so if nearby
// counters gave nearby output, neighbouring paths would be correlated and every
// error estimate built on them would be wrong.
TEST(PhiloxTest, AdjacentCountersProduceUnrelatedOutput) {
    const PhiloxKey key{0, 0};

    std::set<std::uint64_t> seen;
    int total = 0;

    for (std::uint64_t index = 0; index < 256; ++index) {
        const PhiloxCounter output = Philox4x64::generate(PhiloxCounter{index, 0, 0, 0}, key);
        for (const std::uint64_t word : output) {
            seen.insert(word);
            ++total;
        }
    }

    // 1024 draws from a 64-bit space: a collision would be astronomically
    // unlikely, so any repeat means the counter is not being mixed.
    EXPECT_EQ(seen.size(), static_cast<std::size_t>(total))
        << "consecutive counters produced repeated words";
}

// Bijectivity over the counter, checked on a sample: distinct counters must give
// distinct output. Philox is a bijection by construction, and a collision would
// mean the implementation has broken it.
TEST(PhiloxTest, DistinctCountersGiveDistinctOutput) {
    const PhiloxKey key{0xDEADBEEF, 0xCAFEBABE};
    std::set<PhiloxCounter> outputs;

    for (std::uint64_t block = 0; block < 32; ++block) {
        for (std::uint64_t path = 0; path < 32; ++path) {
            const PhiloxCounter output =
                Philox4x64::generate(PhiloxCounter{block, path, 0, 0}, key);
            EXPECT_TRUE(outputs.insert(output).second)
                << "collision at block=" << block << " path=" << path;
        }
    }

    EXPECT_EQ(outputs.size(), 32U * 32U);
}

// A weak generator often shows a stuck or biased bit, which no moment test at
// realistic sample sizes would catch. Each bit position should be set about half
// the time.
TEST(PhiloxTest, EveryBitPositionVaries) {
    const PhiloxKey key{1, 2};
    std::array<int, 64> ones{};
    int words = 0;

    for (std::uint64_t block = 0; block < 512; ++block) {
        const PhiloxCounter output = Philox4x64::generate(PhiloxCounter{block, 0, 0, 0}, key);
        for (const std::uint64_t word : output) {
            ++words;
            for (int bit = 0; bit < 64; ++bit) {
                if (((word >> bit) & 1U) != 0U) {
                    ++ones[static_cast<std::size_t>(bit)];
                }
            }
        }
    }

    // 2048 words. A fair bit has standard deviation sqrt(2048*0.25) ~ 22.6, so
    // 5 sigma is ~113. The bound is loose on purpose: this is looking for a
    // stuck or systematically biased bit, not testing uniformity, which the KAT
    // vectors already settle.
    const double expected = 0.5 * words;
    for (int bit = 0; bit < 64; ++bit) {
        EXPECT_NEAR(static_cast<double>(ones[static_cast<std::size_t>(bit)]), expected, 120.0)
            << "bit " << bit << " is biased: " << ones[static_cast<std::size_t>(bit)] << " of "
            << words;
    }
}

}  // namespace
}  // namespace diffusionworks
