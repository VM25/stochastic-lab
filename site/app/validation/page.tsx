import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { RelatedExperiments } from "@/components/RelatedExperiments";
import { Figure } from "@/components/Figure";
import { FIGURE_COPY } from "@/lib/figure-copy";

export const metadata: Metadata = {
  title: "Validation",
  description:
    "How the engine is checked: separated evidence, no unresolved fits, both convergence fits published, and failures preserved.",
};

export default function ValidationPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Stage 04 — validation"
        title="Validation"
        lede="A pricing engine that only checks itself proves nothing. This one is scored against published values, independent methods, analytic invariants, and its own statistical claims — and it refuses to fit what it cannot resolve."
      />

      <Section idx="04.1" title="Four principles" id="principles">
        <ol>
          <li>
            <strong>Evidence is separated by kind.</strong> External references, analytic
            invariants, and internal numerical checks are reported in distinct categories. They
            are not all independent oracles, and no single oracle is load-bearing.
          </li>
          <li>
            <strong>Nothing unresolved is fitted.</strong> A quantity is fitted only once it
            clears a stated multiple of its own standard error. A power-law fit will return a
            confident interval for pure noise, so the refusal is mechanical, not discretionary.
          </li>
          <li>
            <strong>Both fits are published where they disagree.</strong> A theoretical order
            is asymptotic, so a full-range fit over coarse levels is a faithful description of a
            curve that is not a straight line — reported alongside the asymptotic-window fit.
          </li>
          <li>
            <strong>Failures are preserved.</strong> A disappointing outcome is a result. It is
            kept with its evidence, not re-run until it improves or smoothed into a pass.
          </li>
        </ol>
      </Section>

      <Section idx="04.2" title="References and cross-method agreement" id="references">
        <p>
          European prices are checked across analytic, crude and antithetic Monte Carlo, and
          Crank–Nicolson finite differences; Heston Monte Carlo against the characteristic
          function. Methods are compared in units of their own combined standard error, so a
          disagreement is judged against sampling noise rather than an absolute tolerance. The
          characteristic function is validated directly against a published value and four
          independently generated high-precision references.
        </p>
        <Figure
          file="exp13_agreement.png"
          alt={FIGURE_COPY["exp13_agreement.png"].alt}
          caption={FIGURE_COPY["exp13_agreement.png"].caption}
        />
        <RelatedExperiments ids={["EXP-13", "EXP-09"]} />
      </Section>

      <Section idx="04.3" title="Financial invariants" id="invariants">
        <p>
          Put–call parity, the no-arbitrage price bounds, monotonicity in the inputs, and the
          barrier in–out relationship are checked as identities the prices must satisfy
          regardless of method. The martingale identity{" "}
          <span className="numeric">φ₂(−i) = S₀e^(r−q)T</span> is enforced on the Heston
          characteristic function across every regime.
        </p>
        <RelatedExperiments ids={["EXP-09", "EXP-15"]} />
      </Section>

      <Section idx="04.4" title="Confidence coverage" id="coverage">
        <p>
          A reported 95% interval is only worth trusting if it covers at the rate it claims.
          EXP-14 measures that directly, and finds that at a small sample with a severely skewed
          payoff the interval under-covers — a warning, because a reader quoting the interval in
          that regime would be quoting something the experiment disproved. More paths restore
          it.
        </p>
        <Figure
          file="exp14_coverage.png"
          alt={FIGURE_COPY["exp14_coverage.png"].alt}
          caption={FIGURE_COPY["exp14_coverage.png"].caption}
        />
        <RelatedExperiments ids={["EXP-14"]} />
      </Section>

      <Section idx="04.5" title="PDE, Heston, and edge cases" id="other">
        <p>
          The finite-difference engine is validated by its convergence order — a first-order
          drop would betray a misplaced boundary, so the order is a standing regression, not a
          one-time check. The Heston discretisation is validated against the semi-analytic
          reference. Every numerical edge case must either produce the correct limiting value or
          refuse explicitly; no non-finite value may escape.
        </p>
        <RelatedExperiments ids={["EXP-06", "EXP-10", "EXP-15"]} />
      </Section>
    </div>
  );
}
