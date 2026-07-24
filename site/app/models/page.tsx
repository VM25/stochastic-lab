import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { EquationBlock, InlineMath } from "@/components/Equation";
import { RelatedExperiments } from "@/components/RelatedExperiments";

export const metadata: Metadata = {
  title: "Models & instruments",
  description:
    "Black–Scholes–Merton and Heston, their parameters and assumptions, the contracts priced, and the regions where each model fails.",
};

export default function ModelsPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Stage 01 — model"
        title="Models & instruments"
        lede={
          <>
            Two models under the risk-neutral measure, a handful of contracts, and — stated
            plainly — the regions where each model stops being trustworthy. The full
            specification is the engine&apos;s <code>MATHEMATICAL-SPEC</code>.
          </>
        }
      />

      <Section idx="01.1" title="Market and conventions" id="market">
        <p>
          A single-asset market carries spot <InlineMath tex="S_0 > 0" />, a continuously
          compounded risk-free rate <InlineMath tex="r" />, and a continuous dividend yield{" "}
          <InlineMath tex="q" />. Time is in years and all pricing is under the risk-neutral
          measure <InlineMath tex="\mathbb{Q}" />; the discount factor is{" "}
          <InlineMath tex="e^{-rT}" /> and the forward is{" "}
          <InlineMath tex="F = S_0 e^{(r-q)T}" />.
        </p>
      </Section>

      <Section idx="01.2" title="Black–Scholes–Merton" id="black-scholes">
        <p>Under the risk-neutral measure the spot follows a geometric Brownian motion:</p>
        <EquationBlock
          tex="dS_t = (r - q)\,S_t\,dt + \sigma\,S_t\,dW_t"
          describe="The change in the spot price equals the risk-neutral drift r minus q times the spot, plus sigma times the spot times the increment of a Brownian motion."
        />
        <p>The European call has the closed form</p>
        <EquationBlock
          tex="C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2), \quad d_{1,2} = \frac{\ln(S_0/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}"
          describe="The call price equals the discounted spot times the normal CDF of d1 minus the discounted strike times the normal CDF of d2, where d1 and d2 depend on log-moneyness, drift, and volatility."
        />
        <p>with the put following by put–call parity. The parameters are:</p>
        <dl className="defs">
          <dt>
            <InlineMath tex="\sigma" />
          </dt>
          <dd>annualised volatility, the single free parameter of the model</dd>
          <dt>
            <InlineMath tex="K" />
          </dt>
          <dd>strike</dd>
          <dt>
            <InlineMath tex="T" />
          </dt>
          <dd>maturity, in years</dd>
          <dt>
            <InlineMath tex="N(\cdot)" />
          </dt>
          <dd>the standard normal cumulative distribution function</dd>
        </dl>
        <p>
          <strong>Assumptions.</strong> Constant volatility, frictionless continuous trading,
          a single deterministic rate, and log-normal terminal prices. The closed form is the
          reference the simulated and PDE engines are scored against.
        </p>
      </Section>

      <Section idx="01.3" title="Heston" id="heston">
        <p>
          The Heston model replaces the constant volatility with a stochastic variance that
          mean-reverts and is correlated with the spot:
        </p>
        <EquationBlock
          tex="\begin{aligned} dS_t &= (r - q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \\ dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v \\ d\langle W^S, W^v\rangle_t &= \rho\,dt \end{aligned}"
          describe="The spot follows a diffusion with instantaneous variance v. The variance follows a Cox-Ingersoll-Ross process reverting to theta at rate kappa with vol-of-variance xi, and the two Brownian motions are correlated with coefficient rho."
        />
        <dl className="defs">
          <dt>
            <InlineMath tex="v_0" />
          </dt>
          <dd>initial variance</dd>
          <dt>
            <InlineMath tex="\kappa" />
          </dt>
          <dd>mean-reversion rate of the variance</dd>
          <dt>
            <InlineMath tex="\theta" />
          </dt>
          <dd>long-run variance level</dd>
          <dt>
            <InlineMath tex="\xi" />
          </dt>
          <dd>volatility of variance</dd>
          <dt>
            <InlineMath tex="\rho" />
          </dt>
          <dd>spot–variance correlation</dd>
        </dl>
        <p>
          Pricing uses the characteristic function in its numerically stable
          &ldquo;little-trap&rdquo; form, integrated by Gauss–Legendre quadrature. The engine
          validates the function directly, not only through the prices it produces.
        </p>
        <RelatedExperiments ids={["EXP-09", "EXP-10"]} />
      </Section>

      <Section idx="01.4" title="Failure regions" id="failure-regions">
        <p>
          A model is only as trustworthy as its worst regime, so the engine names them rather
          than avoiding them.
        </p>
        <p>
          <strong>The Feller condition.</strong> The variance process can reach zero unless
        </p>
        <EquationBlock
          tex="2\kappa\theta \geq \xi^2"
          describe="Twice kappa times theta is greater than or equal to xi squared."
        />
        <p>
          A violation is treated as a <em>regime to be priced and measured</em>, not an invalid
          input: violating parameter sets are priced, their discretisation bias measured, and
          their uncertainty attached. EXP-10 quantifies exactly this.
        </p>
        <p>
          <strong>Weak identification.</strong> A short-dated option surface carries little
          information about the long-run variance level, so <InlineMath tex="\kappa" /> and{" "}
          <InlineMath tex="\theta" /> can trade off against each other while the fit stays
          good. EXP-12 measures this on a real surface, and it is the sharpest limitation in
          the whole programme.
        </p>
        <RelatedExperiments ids={["EXP-10", "EXP-12"]} />
      </Section>

      <Section idx="01.5" title="Instruments" id="instruments">
        <p>The contracts priced across the programme:</p>
        <dl className="defs">
          <dt>European</dt>
          <dd>call and put — the closed-form baseline</dd>
          <dt>Arithmetic Asian</dt>
          <dd>call and put, with the geometric Asian (which has a closed form) as control variate</dd>
          <dt>Barrier</dt>
          <dd>
            down-and-out and up-and-out calls, under continuous, discrete, and Brownian-bridge
            monitoring
          </dd>
        </dl>
        <p>
          The monitoring convention is part of the contract, not a solver setting. A discretely
          monitored barrier is a different instrument from a continuously monitored one, and
          EXP-07 measures how different.
        </p>
        <RelatedExperiments ids={["EXP-05", "EXP-07"]} />
      </Section>
    </div>
  );
}
