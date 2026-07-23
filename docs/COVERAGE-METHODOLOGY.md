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

`pass` requires two things. First, the intervals must be **defensible where the central limit
theorem should hold** -- at the largest sample, across every moneyness. A large-sample
under-coverage would be a real methodology failure and fails the record. Second, the sweep
must actually **reach the regime where coverage degrades** -- some small-sample skewed cell
must under-cover -- otherwise the experiment has not exercised the defect it exists to detect,
and the result is `inconclusive` rather than a clean pass. The intervals are not revised: they
are correct where they are used at production path counts, and the small-sample under-coverage
is a documented property of the normal approximation, not a bug.

## 5. Limitations

- Coverage is estimated from a finite number of trials and carries its own standard error;
  "defensible" means statistically consistent with nominal, not exactly equal.
- Only the European call under Black-Scholes is used, where an exact reference exists. The
  finding is a property of the interval and the payoff, but the specific path counts at which
  under-coverage bites are for this instrument and these parameters.
- The interval under test is the Student-t interval on the sample mean. A different
  construction (bootstrap, skew-adjusted) would move the small-sample boundary; this experiment
  characterises the interval the engine actually reports, not the best possible one.
