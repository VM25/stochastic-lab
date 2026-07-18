#include <diffusionworks/concurrency/parallel_reduce.hpp>
#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace diffusionworks {
namespace {

// ---------------------------------------------------------------------------
// Partitioning
// ---------------------------------------------------------------------------

// The blocks tile [0, count) exactly, in order, with no gaps or overlaps, and sizes
// differ by at most one -- the deterministic split ADR-011 reproducibility rests on.
TEST(ParallelReduceTest, PartitionTilesTheRangeExactly) {
    for (const std::int64_t count : {1, 2, 7, 100, 1000}) {
        for (const int blocks : {1, 2, 3, 8, 64}) {
            const auto parts = partition_indices(count, blocks);
            ASSERT_FALSE(parts.empty());
            EXPECT_EQ(parts.front().begin, 0);
            EXPECT_EQ(parts.back().end, count);
            std::int64_t covered = 0;
            std::int64_t previous_end = 0;
            std::int64_t min_size = count;
            std::int64_t max_size = 0;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                EXPECT_EQ(parts[i].begin, previous_end) << "gap or overlap at block " << i;
                EXPECT_LE(parts[i].begin, parts[i].end);
                const std::int64_t size = parts[i].end - parts[i].begin;
                covered += size;
                previous_end = parts[i].end;
                min_size = std::min(min_size, size);
                max_size = std::max(max_size, size);
            }
            EXPECT_EQ(covered, count);
            // Never more blocks than indices, and near-even.
            EXPECT_LE(static_cast<std::int64_t>(parts.size()), count);
            EXPECT_LE(max_size - min_size, 1);
        }
    }
}

TEST(ParallelReduceTest, EffectiveWorkerCountIsClamped) {
    EXPECT_EQ(effective_worker_count(1, 1000), 1);
    EXPECT_EQ(effective_worker_count(8, 1000), 8);
    EXPECT_EQ(effective_worker_count(0, 1000), 1);   // zero -> one
    EXPECT_EQ(effective_worker_count(-4, 1000), 1);  // negative -> one
    EXPECT_EQ(effective_worker_count(100, 10), 10);  // never more than the indices
    EXPECT_EQ(effective_worker_count(4, 0), 1);
}

// ---------------------------------------------------------------------------
// The reduction
// ---------------------------------------------------------------------------

struct Sum {
    std::int64_t value{};
};

// The reduced result is the same at every thread count -- here exactly, because
// integer addition is associative. This is the reproducibility contract.
TEST(ParallelReduceTest, IsInvariantToThreadCount) {
    constexpr std::int64_t n = 10000;
    const std::int64_t expected = n * (n - 1) / 2;
    for (const int threads : {1, 2, 3, 7, 16, 64}) {
        const auto result = parallel_reduce<Sum>(
            n,
            threads,
            [] { return Sum{}; },
            [](std::int64_t i, Sum& local) {
                local.value += i;
                return Status::success();
            },
            [](Sum& into, const Sum& from) { into.value += from.value; });
        ASSERT_TRUE(result.ok()) << "threads=" << threads << ": " << result.error().describe();
        EXPECT_EQ(result.value().value, expected) << "threads=" << threads;
    }
}

// Every index is visited exactly once, no matter the thread count -- checked with an
// atomic tally independent of the reduction.
TEST(ParallelReduceTest, VisitsEveryIndexExactlyOnce) {
    constexpr std::int64_t n = 5000;
    for (const int threads : {1, 4, 9}) {
        std::vector<std::atomic<int>> visits(static_cast<std::size_t>(n));
        const auto result = parallel_reduce<Sum>(
            n,
            threads,
            [] { return Sum{}; },
            [&](std::int64_t i, Sum& local) {
                visits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
                local.value += 1;
                return Status::success();
            },
            [](Sum& into, const Sum& from) { into.value += from.value; });
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result.value().value, n);
        for (std::int64_t i = 0; i < n; ++i) {
            EXPECT_EQ(visits[static_cast<std::size_t>(i)].load(), 1) << "index " << i;
        }
    }
}

// A failure in one block is reported, and the same error is returned no matter which
// thread hit it first: the first failure in block order.
TEST(ParallelReduceTest, ReportsTheFirstErrorInBlockOrder) {
    constexpr std::int64_t n = 1000;
    for (const int threads : {1, 4, 8}) {
        const auto result = parallel_reduce<Sum>(
            n,
            threads,
            [] { return Sum{}; },
            [](std::int64_t i, Sum& local) -> Status {
                if (i == 500) {
                    return Status::failure(ErrorCode::PathFailure, "index 500 failed", "test");
                }
                local.value += i;
                return Status::success();
            },
            [](Sum& into, const Sum& from) { into.value += from.value; });
        ASSERT_FALSE(result.ok()) << "threads=" << threads;
        EXPECT_EQ(result.error().code, ErrorCode::PathFailure);
        EXPECT_NE(result.error().message.find("500"), std::string::npos);
    }
}

}  // namespace
}  // namespace diffusionworks
