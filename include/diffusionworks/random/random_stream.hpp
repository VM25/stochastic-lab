#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/numerics/normal.hpp>
#include <diffusionworks/random/philox.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace diffusionworks {

/// Which family of shocks a draw belongs to.
///
/// Two streams must never share coordinates, or their draws would be identical
/// rather than independent. Under Heston, for instance, the asset and variance
/// shocks are correlated by construction; if they also shared a stream they
/// would be the *same* number, and the correlation would be a fiction.
///
/// The numeric values are FROZEN. They are part of the reproducibility contract:
/// changing one silently changes every stored result that used it, while leaving
/// the code looking correct. New purposes take new values; existing ones are
/// never renumbered, and gaps are never reused. A test pins them.
enum class StreamPurpose : std::uint64_t {
    /// Shocks driving the asset price.
    AssetShock = 0,

    /// Shocks driving the variance process (Heston).
    VarianceShock = 1,

    /// Draws used by tests and diagnostics, kept away from pricing streams.
    Diagnostic = 1000,
};

/// The coordinate space, and why it cannot collide.
///
/// A draw is identified by four coordinates and a lane:
///
///     value = f(master_seed, purpose, path, position)
///
/// mapped onto Philox as
///
///     key     = (master_seed, purpose)
///     counter = (position / 4, path, 0, 0)
///     lane    = position mod 4
///
/// Collision-freedom is structural rather than statistical. Philox is a bijection
/// on (counter, key), so distinct (counter, key) pairs give distinct 256-bit
/// outputs; and the map above is injective, since master_seed, purpose, path and
/// position / 4 are each recoverable from the pair, with the lane selecting a
/// word inside that block. Two different coordinate tuples therefore cannot
/// address the same value -- no birthday bound, no seeding heuristic, no
/// assumption of independence between separately seeded generators.
///
/// Every coordinate is a full uint64 and none is narrowed, so no cast can
/// truncate one. The limits below are exhaustive rather than advisory:
struct StreamLimits {
    /// Largest master seed. The whole 64-bit space is usable.
    static constexpr std::uint64_t kMaxMasterSeed = std::numeric_limits<std::uint64_t>::max();

    /// Largest purpose identifier.
    static constexpr std::uint64_t kMaxPurpose = std::numeric_limits<std::uint64_t>::max();

    /// Largest path index. 1.8e19 paths; a run at 1e9 paths per second would need
    /// ~585 years to exhaust it.
    static constexpr std::uint64_t kMaxPath = std::numeric_limits<std::uint64_t>::max();

    /// Largest position within a path, i.e. the deepest time step or dimension.
    ///
    /// position / 4 must not overflow the counter word, and it cannot: dividing a
    /// uint64 by 4 yields at most 2^62 - 1, comfortably inside counter[0]. The
    /// binding limit is position itself.
    static constexpr std::uint64_t kMaxPosition = std::numeric_limits<std::uint64_t>::max();

    /// Philox emits four 64-bit words per block; `position mod 4` selects the lane.
    static constexpr std::uint64_t kWordsPerBlock = 4;
};

/// Maps a 64-bit word to a uniform in the open interval (0, 1).
///
/// Both endpoints must be excluded: inverse_norm_cdf sends 0 to -infinity and 1
/// to +infinity, and a single such draw would poison a path with a non-finite
/// state. Excluding them by construction, rather than by a rejection test, is
/// what preserves the one-uniform-per-draw correspondence that common random
/// numbers depend on.
///
/// Takes the top 52 bits and centres the value in its bucket, giving
/// [2^-53, 1 - 2^-53].
///
/// Why 52 bits and not 53
/// ----------------------
/// The obvious `(bits >> 11) + 0.5) * 2^-53` uses a double's full 53-bit
/// mantissa and is wrong. For all-ones input the integer part is 2^53 - 1, which
/// lies in a binade where the spacing of doubles is exactly 1, so 2^53 - 1 + 0.5
/// is not representable. It rounds *up* to 2^53, and the product is exactly 1.0
/// -- the very endpoint the centring was meant to exclude. The bug fires once in
/// 2^53 draws: rare enough to survive any test that merely samples, and fatal
/// when it lands.
///
/// Shifting by 12 puts the integer part at most 2^52 - 1, which sits in a binade
/// with spacing 0.5, so the half-integer is exact and the product is
/// 1 - 2^-53 < 1. The cost is one bit of resolution: 2^52 distinct uniforms
/// rather than 2^53, which is immaterial next to the sampling error of any Monte
/// Carlo estimate.
[[nodiscard]] inline double uniform_from_bits(std::uint64_t bits) noexcept {
    return (static_cast<double>(bits >> 12U) + 0.5) * 0x1.0p-52;
}

/// A deterministic stream of draws belonging to one path.
///
/// Every value is a pure function of four coordinates -- master seed, purpose,
/// path index, and position within the path:
///
///     value = f(master_seed, purpose, path, position)
///
/// so nothing about scheduling can perturb it. Path 500 draws the same numbers
/// whether the run uses one thread or sixty-four, and whether it is the first
/// path evaluated or the last.
///
/// This is a value type holding only its coordinates and a small buffer. Each
/// path constructs its own, so there is no shared mutable generator to
/// synchronise or to seed (ADR-010, ADR-011). Copying one is meaningful: the
/// copy continues from the same position and yields the same draws.
///
/// Not thread-safe, and deliberately so: a stream is owned by the path it
/// belongs to. Sharing one between threads is a design error, not a locking
/// problem.
class RandomStream {
public:
    /// Binds a stream to its coordinates. No draws are made until one is asked
    /// for.
    RandomStream(std::uint64_t master_seed, StreamPurpose purpose, std::uint64_t path) noexcept
        : key_{master_seed, static_cast<std::uint64_t>(purpose)}, path_(path) {}

    /// Next uniform in (0, 1).
    [[nodiscard]] double next_uniform() noexcept { return uniform_from_bits(next_bits()); }

    /// Next standard normal, by inverting the CDF of the next uniform.
    ///
    /// Exactly one uniform per normal, which is what fixes a draw's coordinates
    /// and makes common random numbers exact across runs.
    [[nodiscard]] double next_normal() noexcept { return inverse_norm_cdf(next_uniform()); }

    /// The antithetic partner of the next normal: -z, obtained from 1-u.
    ///
    /// Exact rather than approximate. Because inverse_norm_cdf is monotone and
    /// odd about u = 0.5, inverting 1-u gives precisely the negation of
    /// inverting u, so an antithetic pair is a genuine reflection of the same
    /// draw and not merely another sample.
    [[nodiscard]] double next_antithetic_normal() noexcept {
        return inverse_norm_cdf(1.0 - next_uniform());
    }

    /// Position of the next draw within this path.
    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }

    /// Rewinds to the start of the path.
    void reset() noexcept { seek(0); }

    /// Jumps to an arbitrary position in O(1).
    ///
    /// Free here, where a stateful generator would need jump-ahead machinery:
    /// position is a coordinate, not a count of work done.
    void seek(std::uint64_t position) noexcept {
        position_ = position;
        buffer_valid_ = false;
    }

    [[nodiscard]] std::uint64_t master_seed() const noexcept { return key_[0]; }

    [[nodiscard]] std::uint64_t purpose() const noexcept { return key_[1]; }

    [[nodiscard]] std::uint64_t path() const noexcept { return path_; }

private:
    [[nodiscard]] std::uint64_t next_bits() noexcept {
        // Philox emits four words per block, so consecutive draws within a block
        // are served from the buffer and only every fourth costs a round of
        // mixing.
        const std::uint64_t block = position_ / StreamLimits::kWordsPerBlock;
        const std::uint64_t word = position_ % StreamLimits::kWordsPerBlock;

        if (!buffer_valid_ || block != buffered_block_) {
            // counter[0] indexes blocks along the path and counter[1] the path
            // itself. The remaining words stay zero: Philox is a bijection over
            // the whole 256-bit counter, so restricting to a sub-lattice costs
            // no independence, and leaving them free lets a later phase widen
            // the addressing without disturbing existing streams.
            buffer_ = Philox4x64::generate(PhiloxCounter{block, path_, 0, 0}, key_);
            buffered_block_ = block;
            buffer_valid_ = true;
        }

        ++position_;
        return buffer_[static_cast<std::size_t>(word)];
    }

    PhiloxKey key_;
    std::uint64_t path_;
    std::uint64_t position_{0};

    PhiloxCounter buffer_{};
    std::uint64_t buffered_block_{0};
    bool buffer_valid_{false};
};

/// A pair of correlated standard normals.
struct CorrelatedNormals {
    double first{};
    double second{};
};

/// A validated correlation coefficient with its Cholesky complement precomputed.
///
/// Exists because correlate() sits in the innermost path loop, where validating
/// on every call would be both wasteful and, worse, untrustworthy: an unchecked
/// correlate() given rho = 2 would compute sqrt(1 - 4), clamp the negative
/// argument to zero, and silently return a perfectly plausible pair carrying a
/// correlation of 1 rather than the 2 that was asked for. That is precisely the
/// class of failure this project refuses -- a wrong number that looks right.
///
/// Following the project's domain-type pattern (ADR-006), validation happens once
/// at construction and the hot loop consumes a value that cannot be invalid.
/// Precomputing sqrt(1 - rho^2) is a consequence rather than the motive: it also
/// removes a square root from every draw.
class CorrelationCoefficient {
public:
    /// Validates and constructs. Requires rho in [-1, 1] and finite.
    ///
    /// The interval is closed: perfect correlation is degenerate but meaningful
    /// (the two shocks coincide), and rejecting it would rule out a limit the
    /// validation plan requires testing.
    [[nodiscard]] static Result<CorrelationCoefficient> create(double rho);

    [[nodiscard]] double value() const noexcept { return rho_; }

    /// \f$ \sqrt{1-\rho^2} \f$, the off-diagonal Cholesky factor.
    [[nodiscard]] double complement() const noexcept { return complement_; }

private:
    CorrelationCoefficient(double rho, double complement) noexcept
        : rho_(rho), complement_(complement) {}

    double rho_;
    double complement_;
};

/// Combines two independent standard normals into a correlated pair.
///
/// \f[ Z_1 = Z_1, \qquad Z_2' = \rho Z_1 + \sqrt{1-\rho^2}\, Z_2 \f]
///
/// This is MATHEMATICAL-SPEC section 8, and is the Cholesky factor of a 2x2
/// correlation matrix written out. The first component passes through unchanged,
/// which matters for reproducibility: switching a model from independent to
/// correlated shocks leaves the asset's own draws untouched, so the two runs
/// remain comparable.
///
/// The two normals must come from different streams. Drawing both from one would
/// make z2 the draw after z1 rather than an independent one -- still
/// reproducible, and still wrong.
///
/// Unchecked by design: the coefficient carries its own validity, so there is
/// nothing left to check here.
[[nodiscard]] inline CorrelatedNormals
correlate(double z1, double z2, const CorrelationCoefficient& rho) noexcept {
    return CorrelatedNormals{z1, rho.value() * z1 + rho.complement() * z2};
}

}  // namespace diffusionworks
