# Multithreading Methodology

How the Monte Carlo path loop is parallelised, and what that does and does not change
about the answer. The facility is `parallel_reduce` (`concurrency/parallel_reduce.hpp`);
the design authority is `ARCHITECTURE-DECISIONS.MD` ADR-011.

## 1. The design: partition, thread-local, reduce in order

A Monte Carlo run over `N` paths is embarrassingly parallel, because every path is a
pure function of `(seed, path_index)` through the counter-based generator: there is **no
shared mutable random-number state** to contend on. The parallelisation is therefore:

1. **Deterministic partition.** `[0, N)` is split into `T` contiguous, near-even blocks
   (`partition_indices`). For a given `(N, T)` the same index always lands in the same
   block, so a worker sees the same paths every run.
2. **Thread-local accumulation.** Each block runs on its own `std::jthread` with its own
   `OnlineMoments` accumulator, diagnostics, and path buffers -- allocated once per
   worker, never per path (ADR-013) -- so no two workers write the same memory.
3. **Order-defined reduction.** After the workers join, the thread-local accumulators
   are merged into one **in block order** with `OnlineMoments::merge` (ADR-011).

There are no locks in the path loop and no shared accumulator: each worker writes only
its own block's state and its own error slot. ThreadSanitizer confirms this is
race-free.

## 2. Reproducibility, exactly stated

- **One thread is the sequential reference.** At `threads = 1` there is one block and
  one accumulator, and the operations are exactly those of the single-threaded engine:
  the result is **bit-for-bit identical**.
- **A fixed thread count is reproducible.** The partition and the reduction order are
  fixed functions of `(N, T)`, so the same configuration produces the **same bits**
  every run. (The tests pin this with an exact equality, not a tolerance.)
- **Different thread counts agree up to reassociation.** `OnlineMoments::merge`
  (Chan-Golub-LeVeque) is algebraically exact for any split, but floating-point addition
  is not associative, so combining `T` partial accumulators is not bit-identical to a
  single sequential pass. The difference is the reassociation of an exact reduction --
  **a documented numerical effect, not a race**. In practice it is a few parts in
  `10^-10` on a `2*10^5`-path European price, far below the Monte Carlo standard error
  (order `10^-2`), so it never affects a reported figure or its confidence interval. The
  regression tests assert cross-thread agreement to `1e-9`.

This is the distinction the exit gate asks to be documented: changing the thread count
can change the last few bits of the estimate, and that is expected and bounded; it can
never change the answer beyond the reassociation of a sum that was always going to carry
rounding.

## 3. Thread-count configuration

`MonteCarloConfig::threads` selects the worker count; the CLI `--threads` flag feeds it.
It is validated to `[1, 1024]`. A request above the path count is clamped down
(`effective_worker_count`) -- there is no point in a worker with an empty block. Analytic
pricing has nothing to parallelise, so a thread count supplied there is reported as
having had no effect rather than silently ignored.

## 4. Scope and limitations

- This applies to the Monte Carlo path loop. The parallel facility is general
  (`parallel_reduce<Local>`), so the other simulation engines adopt the same pattern;
  each is validated the same way (one-thread equivalence, fixed-count reproducibility,
  cross-count agreement, ThreadSanitizer).
- The control-variate pilot for the Asian engine is a small, sequential pre-pass; only
  the main sample loop is parallelised, because the pilot's `beta` must be a single
  constant shared by every path.
- Speed-up is not claimed here -- that is Phase 13's benchmark. This phase's contract is
  correctness and reproducibility under threading, not throughput.
