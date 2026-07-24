# Statistical Confidence Coverage Methodology

How EXP-14 checks whether the Monte Carlo confidence intervals cover the true value as often
as they claim. The experiment is defined in `docs/EXPERIMENT-CATALOG.MD` (EXP-14).

## 1. The question

A Monte Carlo run reports a value and a confidence interval, and the interval carries a
nominal level -- 95%. That claim is testable: if the interval is well calibrated, then across
many independent runs the true value should fall inside it 95% of the time. Coverage below the
nominal level means the interval is too narrow (or mis-centred) and the reported uncertainty
is optimistic. EXP-14 measures the actual coverage and compares it to the claim.

## 2. Procedure

For each cell -- a (strike, sample size) pair -- the experiment runs `trial_count` independent
simulations, each with its own seed, and each producing a 95% interval on the sample mean. It
counts how many of those intervals contain the exact analytic Black-Scholes price, and reports
that fraction as the observed coverage. Because coverage is itself an estimate over a finite
number of trials, it carries a standard error `sqrt(p(1-p)/trials)`, and a cell is called
**defensible** when its observed coverage is within a few of those standard errors of the
nominal level -- statistically consistent with it, not necessarily equal.

The Monte Carlo uses a fixed per-trial seed schedule, so the coverage is deterministic and
reproducible.

## 3. The two axes that break coverage

The sweep varies the two things that make the interval's normal approximation fail:

- **Sample size.** The interval is a Student-t interval on the sample mean, which is
  approximately normal only once the central limit theorem has taken hold. At a small sample
  it may not have.
- **Payoff skewness.** A deep-out-of-the-money call pays on a small fraction of paths, so its
  discounted payoff is severely right-skewed. The experiment reports the exact payoff skewness,
  computed by quadrature over the log-normal terminal law (no simulation) -- an exact property
  of the payoff, not an estimate -- and it grows sharply with moneyness.

The two interact: a skewed payoff needs a larger sample before its sample mean is normal, so
the coverage degrades most at a small sample on a deeply out-of-the-money option.

## 4. Status

The status describes **what was measured**, not whether the sweep ran.

- **`fail`** -- under-coverage at the largest sample, where the central limit theorem should
  hold. That would be a real methodology failure.
- **`warning`** -- a resolved under-coverage anywhere. This is the current result: the
  reported 95% interval is not worth 95% in the small-sample skewed corner, so a reader
  quoting it there would be quoting something this experiment disproved.
- **`inconclusive`** -- nothing under-covered, but nothing stressed the interval either
  (maximum payoff skewness below `kStressSkewness = 5`). A sweep of mild payoffs cannot
  license "the intervals are calibrated".
- **`pass`** -- the sweep reached a genuinely skewed payoff and nothing under-covered.

An earlier revision had this inverted: it made *observing* the degradation the `pass`
condition and "no degradation anywhere" `inconclusive`. That let the experiment earn a pass
for successfully finding its own target defect, which describes the harness rather than the
finding. The guard against a too-easy sweep was a real concern and is kept — it is now the
`inconclusive` branch, gated on payoff skewness rather than on having found a failure.

The intervals themselves are not revised: they are correct where they are used at production
path counts, and the small-sample under-coverage is a documented property of the normal
approximation, not a bug in the estimator.

## 5. Limitations

- Coverage is estimated from a finite number of trials and carries its own standard error;
  "defensible" means statistically consistent with nominal, not exactly equal.
- Only the European call under Black-Scholes is used, where an exact reference exists. The
  finding is a property of the interval and the payoff, but the specific path counts at which
  under-coverage bites are for this instrument and these parameters.
- The interval under test is the Student-t interval on the sample mean. A different
  construction (bootstrap, skew-adjusted) would move the small-sample boundary; this experiment
  characterises the interval the engine actually reports, not the best possible one.
