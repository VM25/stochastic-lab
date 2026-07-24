import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { EquationBlock, InlineMath } from "@/components/Equation";
import { RelatedExperiments } from "@/components/RelatedExperiments";

export const metadata: Metadata = {
  title: "Stochastic models",
  description:
    "Black–Scholes and Heston: the equations that turn assumptions about randomness into a price, and the regions where each stops being reliable.",
};

export default function ModelsPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Stochastic models"
        title="Two ways to model a random price"
        lede="A pricing model is a story about how a price moves at random through time, written as an equation. Two such stories dominate practice — one simple, one richer — and the difference between them is where the interesting behaviour lives."
      />

      <Section idx="A" title="The setting" id="setting">
        <p>
          Every price starts from the same footing: a current price <InlineMath tex="S_0" />, an
          interest rate <InlineMath tex="r" />, and a dividend yield <InlineMath tex="q" />. Pricing
          then asks what a contract is worth today given how the price might evolve, discounting
          future payoffs back to the present. The models differ in one thing only: how they
          describe that evolution.
        </p>
      </Section>

      <Section idx="B" title="Black–Scholes: constant volatility" id="black-scholes">
        <p>
          The classical model assumes the price drifts upward at the interest rate while jiggling
          with a constant volatility <InlineMath tex="\sigma" />:
        </p>
        <EquationBlock
          tex="dS_t = (r - q)\,S_t\,dt + \sigma\,S_t\,dW_t"
          describe="The change in the price equals the drift, the interest rate minus the dividend yield, times the price times the time step, plus a constant volatility sigma times the price times the increment of a random Brownian motion."
        />
        <p>
          Its great virtue is a closed-form answer: the price of a European option is a simple
          formula, no simulation required. That makes it the yardstick — the exact value every
          numerical method on this site is checked against. Its great flaw is the assumption of
          constant volatility, which real markets contradict: options at different strikes imply
          different volatilities, the well-known &ldquo;smile&rdquo; the model cannot reproduce.
        </p>
      </Section>

      <Section idx="C" title="Heston: volatility that moves" id="heston">
        <p>
          The Heston model fixes that by letting the volatility itself be random. The variance{" "}
          <InlineMath tex="v_t" /> follows its own equation, pulled back toward a long-run level and
          correlated with the price:
        </p>
        <EquationBlock
          tex="\begin{aligned} dS_t &= (r - q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \\ dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v \end{aligned}"
          describe="The price follows a diffusion whose volatility is the square root of a variance v. The variance mean-reverts toward a long-run level theta at speed kappa, with its own randomness scaled by the volatility-of-variance xi."
        />
        <p>Its five parameters each control a recognisable feature of the market:</p>
        <dl className="defs">
          <dt>
            <InlineMath tex="v_0" />
          </dt>
          <dd>where volatility starts today</dd>
          <dt>
            <InlineMath tex="\kappa" />
          </dt>
          <dd>how quickly volatility is pulled back to its average</dd>
          <dt>
            <InlineMath tex="\theta" />
          </dt>
          <dd>the long-run average level of variance</dd>
          <dt>
            <InlineMath tex="\xi" />
          </dt>
          <dd>how much the volatility itself fluctuates</dd>
          <dt>
            <InlineMath tex="\rho" />
          </dt>
          <dd>how price moves and volatility moves are correlated — usually negative</dd>
        </dl>
        <p>
          With these, Heston can reproduce the volatility smile. The price is a shade harder to
          compute — there is no elementary formula — but a well-understood integral gives it
          accurately, and the project checks that integral against its own defining properties
          before trusting it.
        </p>
        <RelatedExperiments ids={["EXP-09", "EXP-10"]} />
      </Section>

      <Section idx="D" title="Where each model breaks" id="failure">
        <p>
          A model is only as good as its worst regime, and the project names those rather than
          avoiding them.
        </p>
        <p>
          <strong>Volatility hitting zero.</strong> In the Heston model the variance can fall all
          the way to zero unless its parameters satisfy a balance condition,
        </p>
        <EquationBlock
          tex="2\kappa\theta \geq \xi^2"
          describe="Twice kappa times theta is at least xi squared."
        />
        <p>
          When that balance is broken, the variance repeatedly touches zero, and a careless
          simulation produces nonsense. The project treats such a regime as something to price
          carefully and measure — not as a forbidden input — and one study is devoted to getting it
          right.
        </p>
        <p>
          <strong>Too little information.</strong> Even when the model is well behaved, a set of
          option prices may not be enough to pin its parameters down. A market with only a few
          maturities tells you little about the long-run behaviour of volatility, so very different
          parameter sets can fit equally well. The calibration study measures this directly on a
          real market, and it is the sharpest limitation the project uncovers.
        </p>
        <RelatedExperiments ids={["EXP-10", "EXP-12"]} />
      </Section>
    </div>
  );
}
