import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { EquationBlock, InlineMath } from "@/components/Equation";
import { RelatedExperiments } from "@/components/RelatedExperiments";

export const metadata: Metadata = {
  title: "Numerical methods",
  description:
    "Simulation and discretisation, variance reduction, finite differences, Greek estimators, characteristic-function pricing, and calibration.",
};

export default function MethodsPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Stage 02 — numerical method"
        title="Numerical methods"
        lede="Every price here is produced by more than one method, so the methods can be checked against each other. The programme measures the rate at which each converges, and refuses to fit where a rate is not resolved."
      />

      <Section idx="02.1" title="Monte Carlo and its error" id="monte-carlo">
        <p>
          A Monte Carlo price is the discounted sample mean of the payoff over{" "}
          <InlineMath tex="N" /> paths, reported with the standard error that quantifies how
          much it can be trusted:
        </p>
        <EquationBlock
          tex="\hat{V}_N = e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} g(S^{(i)}), \qquad \mathrm{SE} = \frac{s}{\sqrt{N}}"
          describe="The estimator is the discounted average of the payoff g over N sampled terminal prices, and its standard error is the sample standard deviation divided by the square root of N."
        />
        <p>
          The error decays as <InlineMath tex="N^{-1/2}" />, and EXP-01 confirms that rate
          across every scenario. Randomness is a counter-based Philox stream: a path&apos;s
          draws depend only on its index and the master seed, never on scheduling, which is
          what lets a threaded run reproduce a serial one bit for bit.
        </p>
        <RelatedExperiments ids={["EXP-01", "EXP-04"]} />
      </Section>

      <Section idx="02.2" title="Discretisation: Euler–Maruyama and Milstein" id="discretisation">
        <p>Where the transition is not sampled exactly, the SDE is discretised. Euler–Maruyama takes</p>
        <EquationBlock
          tex="S_{t+\Delta t} = S_t + (r - q)\,S_t\,\Delta t + \sigma\,S_t\,\Delta W"
          describe="The next price equals the current price plus the drift term times the time step plus sigma times the price times the Brownian increment."
        />
        <p>and Milstein adds the second-order correction</p>
        <EquationBlock
          tex="+\; \tfrac{1}{2}\sigma^2 S_t\big((\Delta W)^2 - \Delta t\big)"
          describe="Plus one half sigma squared times the price times the Brownian increment squared minus the time step."
        />
        <p>
          Euler–Maruyama is strong order <InlineMath tex="\tfrac{1}{2}" /> and Milstein strong
          order 1; both are weak order 1. EXP-02 and EXP-03 measure these orders, and publish
          both a full-range and an asymptotic-window fit where the two disagree, because a
          theoretical order is an asymptotic statement.
        </p>
        <RelatedExperiments ids={["EXP-02", "EXP-03"]} />
      </Section>

      <Section idx="02.3" title="Variance reduction" id="variance-reduction">
        <p>
          Antithetic sampling, a geometric-Asian control variate, and their combination reduce
          variance without bias. The engine ranks them by <em>work-normalised</em> efficiency —
          reciprocal variance times deterministic work units — rather than by wall-clock time,
          so the ranking is a property of the estimators and not of the machine.
        </p>
        <RelatedExperiments ids={["EXP-05"]} />
      </Section>

      <Section idx="02.4" title="Finite differences" id="finite-differences">
        <p>The Black–Scholes price also solves a backward parabolic PDE:</p>
        <EquationBlock
          tex="\frac{\partial V}{\partial t} + \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q)S\frac{\partial V}{\partial S} - rV = 0"
          describe="The time derivative of the value plus one half sigma squared S squared times the second spatial derivative plus the drift times the first spatial derivative minus r times the value equals zero."
        />
        <p>
          The engine discretises it with explicit, fully implicit, and Crank–Nicolson time
          stepping, plus an optional Rannacher start-up. Spatial and temporal orders are
          measured <em>separately</em>, each with the other&apos;s error driven down and the
          premise then verified — a joint sweep measures a mixture, not an order. Barrier
          contracts are solved with an absorbing boundary at the barrier.
        </p>
        <RelatedExperiments ids={["EXP-06", "EXP-07"]} />
      </Section>

      <Section idx="02.5" title="Greeks" id="greeks">
        <p>
          Four estimators of each sensitivity: analytic, central finite differences under
          common random numbers, pathwise, and likelihood-ratio. No single one wins
          everywhere, and the engine reports the trade-off per regime rather than declaring a
          winner. EXP-08 corrects a textbook heuristic about how the finite-difference
          variance behaves under common random numbers.
        </p>
        <RelatedExperiments ids={["EXP-08"]} />
      </Section>

      <Section idx="02.6" title="Heston pricing and calibration" id="calibration">
        <p>
          Heston prices come from integrating the characteristic function; the engine checks
          the function&apos;s own invariants — <InlineMath tex="\varphi(0)=1" />, conjugate
          symmetry, and the martingale identity{" "}
          <InlineMath tex="\varphi_2(-i) = S_0 e^{(r-q)T}" /> — before trusting a price.
          Calibration minimises a weighted objective by Levenberg–Marquardt from multiple
          starts, reporting convergence, fit, parameter dispersion, and penalty reliance
          separately.
        </p>
        <RelatedExperiments ids={["EXP-09", "EXP-11", "EXP-12"]} />
      </Section>
    </div>
  );
}
