import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { RelatedExperiments } from "@/components/RelatedExperiments";
import { Figure } from "@/components/Figure";
import { FIGURE_COPY } from "@/lib/figure-copy";

export const metadata: Metadata = {
  title: "Validation",
  description:
    "How every result is checked — against exact formulas, against independent methods, and against the model's own defining properties.",
};

export default function ValidationPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Validation"
        title="How the results are checked"
        lede="A method that only checks itself proves nothing. Every number here is confirmed against something independent — an exact formula, a rival method, or a mathematical identity the answer must satisfy — and the project refuses to report anything it cannot resolve above the noise."
      />

      <Section idx="A" title="The discipline" id="principles">
        <ol>
          <li>
            <strong>Independent checks, kept separate.</strong> A published reference value, a rival
            method, and a mathematical identity are different kinds of evidence, and the project
            keeps them apart rather than pretending they all confirm each other.
          </li>
          <li>
            <strong>Nothing unresolved is reported as fact.</strong> If an effect cannot be
            distinguished from simulation noise, it is reported as unresolved — never fitted to a
            confident-looking number, because a fitting procedure will happily draw a trend through
            pure randomness.
          </li>
          <li>
            <strong>Honest about early behaviour.</strong> A convergence rate is a statement about
            the fine-grid limit, so where a coarse-grid estimate disagrees with the limit, both are
            shown and the difference is explained.
          </li>
          <li>
            <strong>Failures are kept.</strong> A disappointing result is a result. It stays,
            explained, rather than being re-run until it looks better.
          </li>
        </ol>
      </Section>

      <Section idx="B" title="Independent methods agree" id="agreement">
        <p>
          The strongest check is redundancy: price the same contract by an exact formula, by
          simulation, and by solving the pricing equation, and confirm they agree — not to an
          arbitrary tolerance, but to within the simulation&apos;s own margin of error. The Heston
          model&apos;s price is checked the same way, against a published reference and several
          independently computed high-precision values.
        </p>
        <Figure
          file="exp13_agreement.png"
          alt={FIGURE_COPY["exp13_agreement.png"].alt}
          caption={FIGURE_COPY["exp13_agreement.png"].caption}
        />
        <RelatedExperiments ids={["EXP-13", "EXP-09"]} />
      </Section>

      <Section idx="C" title="Prices must obey the rules" id="invariants">
        <p>
          Some things must be true of any correct price, regardless of method: a call and a put at
          the same strike are linked by a fixed relationship, prices must stay within no-arbitrage
          bounds, and a more valuable position cannot be worth less. These identities are checked
          as a matter of course, and the Heston pricing formula is held to the mathematical
          properties that define it.
        </p>
        <RelatedExperiments ids={["EXP-09", "EXP-15"]} />
      </Section>

      <Section idx="D" title="Confidence intervals that mean what they say" id="coverage">
        <p>
          When simulation reports a 95% confidence interval, that claim can itself be tested: run
          the experiment many times and count how often the interval really contains the true value.
          The project finds that the interval is trustworthy in normal conditions and{" "}
          <em>not</em> trustworthy for a rare, extreme payoff with too few paths — a caveat worth
          knowing before quoting the number.
        </p>
        <Figure
          file="exp14_coverage.png"
          alt={FIGURE_COPY["exp14_coverage.png"].alt}
          caption={FIGURE_COPY["exp14_coverage.png"].caption}
        />
        <RelatedExperiments ids={["EXP-14"]} />
      </Section>

      <Section idx="E" title="Behaviour at the edges" id="edges">
        <p>
          Finally, the methods are pushed to their limits: a maturity shrinking to zero, a barrier
          already breached, a correlation at its extreme. Each such case must either return the
          correct limiting value or refuse cleanly — never quietly return a plausible but wrong
          number. Refusing, here, is the correct behaviour.
        </p>
        <RelatedExperiments ids={["EXP-06", "EXP-10", "EXP-15"]} />
      </Section>
    </div>
  );
}
