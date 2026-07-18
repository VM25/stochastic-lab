#include <diffusionworks/concurrency/parallel_reduce.hpp>

#include <algorithm>

namespace diffusionworks {

int effective_worker_count(int requested, std::int64_t count) {
    if (count <= 0) {
        return 1;
    }
    const int wanted = std::max(requested, 1);
    // Never more workers than indices: a worker with an empty block does nothing but
    // cost a thread.
    const auto capped = std::min(static_cast<std::int64_t>(wanted), count);
    return static_cast<int>(capped);
}

std::vector<WorkBlock> partition_indices(std::int64_t count, int blocks) {
    std::vector<WorkBlock> partitions;
    if (count <= 0 || blocks <= 0) {
        return partitions;
    }
    const auto block_count = std::min(static_cast<std::int64_t>(blocks), count);
    partitions.reserve(static_cast<std::size_t>(block_count));

    // Near-even split: the first `remainder` blocks carry one extra index. This depends
    // only on (count, blocks), so the assignment is identical on every run.
    const std::int64_t base = count / block_count;
    const std::int64_t remainder = count % block_count;
    std::int64_t begin = 0;
    for (std::int64_t b = 0; b < block_count; ++b) {
        const std::int64_t size = base + (b < remainder ? 1 : 0);
        partitions.push_back(WorkBlock{.begin = begin, .end = begin + size});
        begin += size;
    }
    return partitions;
}

}  // namespace diffusionworks
