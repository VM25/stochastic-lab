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
  `10^-13` on the mean and `10^-7` on the standard error, far below the Monte Carlo
  standard error (order `10^-2`), so it never affects a reported figure or its
  confidence interval.

  The regression tests assert this with a **scale-aware** tolerance
  (`tests/support/thread_agreement.hpp`), not a universal absolute constant: the bound
  is relative to the magnitude of the quantity, because reassociation scales with it. A
  mean-like estimator (a price, a Greek that ranges from `~0.02` gamma to `~40` vega) is
  checked to `1e-9` relative; a standard error, whose variance accumulation carries more
  cancellation, to `1e-5` relative -- both far above the reassociation actually observed
  and far below the relative move a real data race would cause (`~10^-2` or more), so the
  tolerance cannot mask a genuine disagreement while meaning the same thing across
  engines whose figures span several orders of magnitude. Quantities that reduce
  *exactly* -- warnings, failure statuses, path counts, knockout counts, and the Heston
  variance diagnostics (integer sums and a `min`) -- are asserted bit-identical, not to a
  tolerance.

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

## 4. Every Monte Carlo engine, and the care each one needed

The facility is general (`parallel_reduce<Local>`), and **all five Monte Carlo path
loops now use it**: the European and arithmetic-Asian GBM engines, the Heston
full-truncation Euler engine, the Monte Carlo Greek estimators, and the barrier
engine. Each is validated the same way -- one-thread equivalence, fixed-count
reproducibility, cross-count agreement, and ThreadSanitizer -- and each carried one
subtlety worth stating, because the reduction has to preserve more than a mean.

- **European.** The base case: one payoff per path (or one antithetic pair, drawn
  inside a single path index so partitioning keeps the pair together), reduced through
  `OnlineMoments::merge`.
- **Arithmetic Asian (control variate).** The control-variate pilot is a small
  *sequential* pre-pass over path indices `[N, N + pilot)`, disjoint from the
  production indices `[0, N)`. Only the main sample loop is parallelised, because the
  pilot's `beta` and the control's expectation must be single constants shared by every
  path. The pilot's non-positive-state excursions are folded into the reduced
  main-sample count by an integer add, so the reported diagnostic matches the
  sequential engine and no worker can lose them. Because the pilot never runs in
  parallel, its statistics -- `beta`, the control expectation, the pilot correlation --
  are **bit-identical at every thread count**.
- **Heston.** The negative-variance event count and the non-finite-path count are
  integer sums; the minimum pre-truncation variance is a `min`, seeded to `v0` in every
  worker exactly as the sequential run seeds it. All three are **exact, order-independent
  reductions**, so they are identical at any thread count and no worker's diagnostics are
  lost. A non-finite path stays a diagnostic rather than an error -- the worker counts it
  and still succeeds -- and the price is blocked afterwards on the *reduced* count, so the
  failure status is the same however the paths partition.
- **Greeks.** Each path's whole common-random-number estimator -- for the
  finite-difference Greeks, the up and down (and, for gamma, mid) re-prices against that
  path's single shared draw -- is computed inside one `contribution` call for a single
  index and added to one accumulator. Partitioning by index therefore keeps each
  estimator contribution **within one worker**; the shared draw is never split across
  threads, which cross-count agreement to `1e-9` confirms (a split would decorrelate the
  paired re-prices and move the estimate far more).
- **Barrier.** The early-knockout `break` lives entirely inside one path's own
  monitoring loop, and the asset shocks and bridge uniforms are keyed by
  `(seed, purpose, index)`. So which worker owns a path cannot change the coordinates it
  draws, the order it draws them, or when it stops: the parallel run tests every interval
  against exactly the bridge uniform the sequential run did. The knock counts, which the
  early break drives, are exact integer reductions and so are **bit-identical** at every
  thread count.

The pattern in every case: `OnlineMoments::merge` for the estimator (reassociated across
thread counts), and integer sums or `min` for the diagnostics (exact at every thread
count). The counter-based generator is what makes it safe -- every path is a pure
function of `(seed, index)`, so there is no shared mutable random state to contend on.

Speed-up is not claimed here -- that is Phase 13's benchmark. This phase's contract is
correctness and reproducibility under threading, not throughput.
