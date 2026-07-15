# Benchmark suite.
#
# BENCHMARK-PLAN section 1 requires correctness before measurement, so this
# directory stays empty until the engine it would measure exists and is
# validated. The suites land in Phase 13 (BUILD-PLAN), once Monte Carlo, PDE,
# Greek, and calibration paths are validated and there is something meaningful
# and correct to time.
#
# The `benchmark` preset configures and builds cleanly against this file; it
# simply produces no benchmark targets yet. A placeholder benchmark timing
# nothing of consequence would be worse than none: it would imply performance
# evidence that does not exist.
