#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace diffusionworks {

/// A half-open block of work indices assigned to one worker.
struct WorkBlock {
    std::int64_t begin{};
    std::int64_t end{};
};

/// Splits `[0, count)` into `blocks` contiguous, near-even partitions.
///
/// Deterministic: for a given (count, blocks) the same index always lands in the same
/// block, so a worker sees the same paths at any thread count -- the property ADR-011's
/// reproducibility rests on. The first `count % blocks` partitions carry one extra
/// index, so the sizes differ by at most one and the split does not depend on order.
[[nodiscard]] std::vector<WorkBlock> partition_indices(std::int64_t count, int blocks);

/// Resolves a requested worker count to an effective one in `[1, count]`.
///
/// There is never a reason to run more workers than there are indices, and a request
/// of zero or fewer is treated as one. The upper end is the caller's responsibility to
/// keep sane (a configuration validates it); this only prevents empty blocks.
[[nodiscard]] int effective_worker_count(int requested, std::int64_t count);

/// Deterministic parallel map-reduce over `[0, count)`.
///
/// Partitions the range into `threads` contiguous blocks (`partition_indices`), runs
/// each block on its own `std::jthread` with its own thread-local accumulator built by
/// `make_local`, applies `body(index, local)` to each index in the block *in order*,
/// and -- after all workers join -- reduces the locals into one **in block order** with
/// `reduce(into, from)`. Nothing is shared and mutated across workers: each writes only
/// its own accumulator and its own error slot, so there are no locks and no data races.
///
/// Determinism (ADR-011)
/// ---------------------
/// At one thread the result is bit-identical to a sequential run: one block, one
/// accumulator, the same order of operations. At a fixed thread count the partition and
/// the reduction order are fixed, so the result is reproducible. Across thread counts
/// the answer changes only by the floating-point reassociation of an exact,
/// order-defined merge (e.g. `OnlineMoments::merge`) -- a documented effect, not a race.
///
/// Errors
/// ------
/// `body` returns a `Status`; a block stops at its first failure. After the join, the
/// first error **in block order** is returned, so the reported failure does not depend
/// on which thread happened to finish first.
template<typename Local, typename MakeLocal, typename Body, typename Reduce>
[[nodiscard]] Result<Local>
parallel_reduce(std::int64_t count, int threads, MakeLocal make_local, Body body, Reduce reduce) {
    const int worker_count = effective_worker_count(threads, count);
    const std::vector<WorkBlock> blocks = partition_indices(count, worker_count);

    // Constructed before any worker starts, then each worker mutates only its own slot.
    std::vector<Local> locals;
    locals.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        locals.push_back(make_local());
    }
    std::vector<std::optional<Error>> errors(blocks.size());

    {
        std::vector<std::jthread> workers;
        workers.reserve(blocks.size());
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            workers.emplace_back([&, b] {
                for (std::int64_t index = blocks[b].begin; index < blocks[b].end; ++index) {
                    const Status status = body(index, locals[b]);
                    if (!status) {
                        errors[b] = status.error();
                        return;
                    }
                }
            });
        }
        // The jthreads join here as `workers` is destroyed.
    }

    // The first error in block order, so the reported failure is deterministic.
    for (const std::optional<Error>& error : errors) {
        if (error.has_value()) {
            return Result<Local>::failure(*error);
        }
    }

    Local merged = std::move(locals.front());
    for (std::size_t b = 1; b < locals.size(); ++b) {
        reduce(merged, locals[b]);
    }
    return Result<Local>::success(std::move(merged));
}

}  // namespace diffusionworks
