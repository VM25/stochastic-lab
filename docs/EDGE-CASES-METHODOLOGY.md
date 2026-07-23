# Numerical Edge Cases Methodology

How EXP-15 checks that the system fails safely or stays correct under limiting and difficult
inputs. The experiment is defined in `docs/EXPERIMENT-CATALOG.MD` (EXP-15).

## 1. The contract

Every edge case must do exactly one of two things: **produce the correct value**, or **refuse
with an explicit error**. A NaN, an infinity, a silently wrong number, or a silent fallback is
a failure. The catalog is emphatic that any unresolved edge case blocks completion, so the
record passes only if every case resolves and no non-finite value escapes.

The cases are curated to the catalog's named limits, and each is classified and checked:

| Category | What it checks | How it passes |
| --- | --- | --- |
| `limiting_behavior` | maturity to zero, volatility to zero (and exactly zero) | price matches the theoretical limit (intrinsic, or discounted forward intrinsic) within tolerance |
| `extreme_valid` | deep in/out of the money, extreme strike and maturity, small non-degenerate Heston variance | price is finite and inside the no-arbitrage bounds |
| `already_breached` | a down-and-out below its barrier, an up-and-out above it | price is (essentially) zero -- correctly knocked out |
| `degenerate_refusal` | Heston correlation at ±1, near-zero variance | the engine refuses with an explicit `ConvergenceFailure` |
| `invalid_input_rejected` | negative volatility, maturity, or strike; correlation outside [-1, 1] | rejected at construction, never reaching an engine |

## 2. What the cases establish

- **Limits match theory.** As the maturity goes to zero the Black-Scholes call approaches its
  intrinsic value; as the volatility goes to zero it approaches the discounted forward
  intrinsic. Exactly-zero maturity and exactly-zero volatility are handled as the limits they
  are, not as errors.
- **Extremes stay bounded and finite.** Deep in and out of the money, and at extreme but valid
  strikes and maturities, the price stays inside `[max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}]`
  and finite.
- **An already-breached barrier is worth zero.** The barrier is constructed without the spot,
  so the knock-out is detected at pricing; a down-and-out below its barrier or an up-and-out
  above it prices to zero, not to a non-finite value.
- **Degenerate regions fail safely.** The Heston characteristic-function integral does not
  converge as the correlation reaches ±1 or the variance approaches zero, and there the engine
  refuses with an explicit status rather than returning a plausible wrong price. A failure that
  stops is safe; a confident wrong answer is not.
- **Invalid inputs never reach an engine.** A negative volatility, maturity, or strike, or a
  correlation outside `[-1, 1]`, is rejected at construction.

## 3. Status

`pass` requires that every case resolve -- a correct limit, a bounded price, a zero, a safe
refusal, or a rejected construction -- and that no non-finite value escape anywhere. A single
unresolved case, or one escaped NaN or infinity, fails the record.

## 4. Limitations

- The Heston refusal at the correlation boundary and near-zero variance is a property of this
  engine's Gauss-Legendre integration, which does not converge there -- not a statement that
  those prices do not exist. A different quadrature might price a wider region. What is
  established is that the engine refuses explicitly rather than returning a wrong number.
- The limiting values are checked at a small but non-zero maturity and volatility (and at
  exactly zero where the engine accepts it). "Approaches the limit" is verified at these points,
  not proved as a limit; the tolerance is set so the discretization there is well inside it.
- The edge cases are a fixed, curated list covering the catalog's named limits. They are not
  exhaustive over the parameter space; a case not listed is not tested here.
