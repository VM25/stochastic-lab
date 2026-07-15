#pragma once

#include <array>
#include <cstdint>

namespace diffusionworks {

/// 256-bit Philox counter, as four 64-bit words.
using PhiloxCounter = std::array<std::uint64_t, 4>;

/// 128-bit Philox key, as two 64-bit words.
using PhiloxKey = std::array<std::uint64_t, 2>;

namespace detail {

/// Full 128-bit product of two 64-bit words.
///
/// Philox's mixing depends on the *high* half of the product, which ordinary
/// 64-bit multiplication discards, so this cannot be written with `a * b`.
inline void
mulhilo64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) noexcept {
#if defined(__SIZEOF_INT128__)
    const __uint128_t product = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    hi = static_cast<std::uint64_t>(product >> 64U);
    lo = static_cast<std::uint64_t>(product);
#else
    // Portable schoolbook fallback for toolchains without __int128. Splits each
    // operand into 32-bit halves and reassembles the 128-bit product.
    constexpr std::uint64_t kLowMask = 0xFFFFFFFFULL;
    const std::uint64_t a_lo = a & kLowMask;
    const std::uint64_t a_hi = a >> 32U;
    const std::uint64_t b_lo = b & kLowMask;
    const std::uint64_t b_hi = b >> 32U;

    const std::uint64_t lo_lo = a_lo * b_lo;
    const std::uint64_t hi_lo = a_hi * b_lo;
    const std::uint64_t lo_hi = a_lo * b_hi;
    const std::uint64_t hi_hi = a_hi * b_hi;

    const std::uint64_t cross = (lo_lo >> 32U) + (hi_lo & kLowMask) + lo_hi;

    hi = (hi_lo >> 32U) + (cross >> 32U) + hi_hi;
    lo = (cross << 32U) | (lo_lo & kLowMask);
#endif
}

}  // namespace detail

/// Philox4x64-10: a counter-based pseudorandom bijection.
///
/// Maps a 256-bit counter and a 128-bit key to 256 bits of output. From
/// Salmon, Moraes, Dror and Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3"
/// (SC11), section 4.3; the constants below are that paper's.
///
/// Why counter-based rather than a conventional stateful generator
/// ---------------------------------------------------------------
/// A stateful generator produces a sequence, so a value's identity depends on
/// how many draws preceded it -- which, under parallel execution, depends on how
/// work was scheduled. Reproducing a result then requires reproducing the
/// schedule, and per-thread streams have to be seeded so as not to overlap,
/// which is assumed more often than it is proved.
///
/// Philox instead makes each output a pure function of its coordinates. There is
/// no sequence and no state to advance, so:
///
///   - a path's randomness is identical at any thread count, making Phase 12's
///     one-thread equivalence a property of the design rather than a claim to be
///     defended;
///   - no shared mutable RNG can exist, because there is nothing to share
///     (ADR-010, ADR-011);
///   - common random numbers and antithetic pairing are exact identities: reuse
///     the coordinates, or map u to 1-u;
///   - stream independence rests on the bijection itself rather than on a
///     seeding heuristic.
///
/// The cost is roughly ten multiply-xor rounds per four values, which is
/// competitive with Mersenne Twister and does not touch memory.
///
/// This implementation is validated bit-for-bit against the published Random123
/// known-answer vectors and, independently, against numpy.random.Philox. See
/// data/references/philox_kat.json.
class Philox4x64 {
public:
    /// Round count. Ten is the standard parameterisation; Random123 reports that
    /// the generator passes BigCrush from seven rounds, so ten carries margin.
    static constexpr int kRounds = 10;

    // Multipliers and Weyl increments from Random123 section 4.3. The Weyl
    // constants are the fractional parts of the golden ratio and sqrt(3)-1.
    static constexpr std::uint64_t kMultiplier0 = 0xD2E7470EE14C6C93ULL;
    static constexpr std::uint64_t kMultiplier1 = 0xCA5A826395121157ULL;
    static constexpr std::uint64_t kWeyl0 = 0x9E3779B97F4A7C15ULL;
    static constexpr std::uint64_t kWeyl1 = 0xBB67AE8584CAA73BULL;

    /// Applies the 10-round bijection.
    ///
    /// Deterministic and stateless: the same counter and key always yield the
    /// same output, on any platform and at any thread count.
    [[nodiscard]] static PhiloxCounter generate(const PhiloxCounter& counter,
                                                const PhiloxKey& key) noexcept {
        PhiloxCounter state = counter;
        PhiloxKey round_key = key;

        for (int round = 0; round < kRounds; ++round) {
            // The key is bumped between rounds, not before the first one.
            if (round > 0) {
                round_key[0] += kWeyl0;
                round_key[1] += kWeyl1;
            }
            state = single_round(state, round_key);
        }

        return state;
    }

private:
    [[nodiscard]] static PhiloxCounter single_round(const PhiloxCounter& counter,
                                                    const PhiloxKey& key) noexcept {
        std::uint64_t hi0 = 0;
        std::uint64_t lo0 = 0;
        std::uint64_t hi1 = 0;
        std::uint64_t lo1 = 0;
        detail::mulhilo64(kMultiplier0, counter[0], hi0, lo0);
        detail::mulhilo64(kMultiplier1, counter[2], hi1, lo1);

        return PhiloxCounter{hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0};
    }
};

}  // namespace diffusionworks
