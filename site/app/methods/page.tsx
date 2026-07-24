import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { EquationBlock, InlineMath } from "@/components/Equation";
import { RelatedExperiments } from "@/components/RelatedExperiments";

export const metadata: Metadata = {
  title: "Numerical methods",
  description:
    "How a model becomes a number: simulation, variance reduction, finite differences, sensitivities, and calibration.",
};

export default function MethodsPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Numerical methods"
        title="Turning a model into a number"
        lede="A model is a set of equations; a price is a single number. Several different methods bridge that gap, each with its own error and its own blind spots. The project prices the same contracts several ways precisely so the methods can be checked against one another."
      />

      <Section idx="A" title="Simulation" id="monte-carlo">
        <p>
          The most direct method simulates thousands of possible price paths, values the option on
          each, and averages. Because it is an average of random samples, it comes with a built-in
          measure of its own reliability — a standard error that shrinks as more paths are added:
        </p>
        <EquationBlock
          tex="\text{price} \approx e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \text{payoff}\big(S^{(i)}\big), \qquad \text{error} \sim \frac{1}{\sqrt{N}}"
          describe="The price is approximated by the discounted average payoff over N simulated paths, and the statistical error shrinks in proportion to one over the square root of N."
        />
        <p>
          That <InlineMath tex="1/\sqrt{N}" /> rate is a promise the theory makes, and the first
          study confirms the method keeps it: to halve the error you need four times the paths. It
          is reliable and general, but slow to reach high precision, which motivates the sharper
          methods that follow.
        </p>
        <RelatedExperiments ids={["EXP-01", "EXP-04"]} />
      </Section>

      <Section idx="B" title="Discretising the path" id="discretisation">
        <p>
          A simulated path is built one small time step at a time, and there is more than one recipe
          for taking a step. The simplest, Euler–Maruyama, moves the price by its drift and a random
          shock; a refinement, Milstein, adds a correction term that makes each path more faithful
          to the true dynamics. How quickly the error falls as the steps shrink — the{" "}
          <em>order of convergence</em> — is a precise theoretical claim, and two studies measure it
          directly, publishing both an honest full-range estimate and the sharper value that holds
          once the steps are fine enough.
        </p>
        <RelatedExperiments ids={["EXP-02", "EXP-03"]} />
      </Section>

      <Section idx="C" title="Doing more with fewer paths" id="variance-reduction">
        <p>
          Simulation error can be cut without simply adding paths, by using clever sampling tricks:
          pairing each random path with its mirror image, or exploiting a related contract whose
          answer is known exactly. The project measures how much each trick is worth in a way that
          reflects the extra work it costs, so the comparison reflects the method rather than the
          machine it happened to run on.
        </p>
        <RelatedExperiments ids={["EXP-05"]} />
      </Section>

      <Section idx="D" title="Solving the pricing equation directly" id="finite-differences">
        <p>
          A price can also be found without simulation at all. Under Black–Scholes it satisfies a
          differential equation, and solving that equation on a grid gives the price directly. The
          project measures how the grid method converges as the grid is refined, and shows a
          well-known instability in one popular scheme — and the standard fix for it — measured
          rather than merely asserted.
        </p>
        <RelatedExperiments ids={["EXP-06", "EXP-07"]} />
      </Section>

      <Section idx="E" title="Sensitivities" id="greeks">
        <p>
          Traders care not only about a price but about how it moves as the market moves — the{" "}
          &ldquo;Greeks.&rdquo; There are several ways to estimate them, from simple bumping of the
          inputs to more refined formulas, and no single method is best everywhere. One study
          compares them and overturns a textbook rule of thumb about how noisy the simple method
          really is.
        </p>
        <RelatedExperiments ids={["EXP-08"]} />
      </Section>

      <Section idx="F" title="Calibration" id="calibration">
        <p>
          Finally, a model is only useful once its parameters are set to match the market. Calibration
          searches for the parameter values that best reproduce a set of observed option prices. The
          project runs this both on a controlled test with a known answer and on a real market — and
          reports not just how well the fit matches, but whether the fit actually determines the
          parameters, which turns out to be a subtler question.
        </p>
        <RelatedExperiments ids={["EXP-11", "EXP-12"]} />
      </Section>
    </div>
  );
}
