import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { InlineMath } from "@/components/Equation";
import { PayoffDiagram } from "@/components/PayoffDiagram";
import { RelatedExperiments } from "@/components/RelatedExperiments";
import styles from "./instruments.module.css";

export const metadata: Metadata = {
  title: "Derivative structures",
  description:
    "The contracts priced in the studies — European, Asian, and barrier options — and the geometry of their payoffs.",
};

export default function InstrumentsPage() {
  return (
    <div className="page">
      <PageHeader
        stage="Derivative structures"
        title="The contracts, and the shape of their payoffs"
        lede="An option is defined by what it pays at expiry. That payoff — a simple kink for a plain call, a knock-out cliff for a barrier — is what makes one contract easy to price and another a genuine test of a method."
      />

      <Section idx="A" title="European options" id="european">
        <p>
          The simplest contract: the right to buy (a call) or sell (a put) at a fixed strike{" "}
          <InlineMath tex="K" /> on a single expiry date. The payoff depends only on the final
          price, so a call is worth <InlineMath tex="\max(S_T - K, 0)" /> and a put{" "}
          <InlineMath tex="\max(K - S_T, 0)" />. These have a closed-form price under
          Black–Scholes, which makes them the reference every other method is measured against.
        </p>
        <div className={styles.diagrams}>
          <PayoffDiagram kind="call" />
          <PayoffDiagram kind="put" />
        </div>
      </Section>

      <Section idx="B" title="Asian options" id="asian">
        <p>
          An Asian option pays on the <em>average</em> price over the life of the contract rather
          than the final price alone. Averaging smooths out a last-minute spike, which makes these
          harder to manipulate — and, because the arithmetic average has no simple formula, a good
          test of simulation. The project uses the closely related geometric-average option, which
          does have a formula, as a reference to sharpen the estimate.
        </p>
        <RelatedExperiments ids={["EXP-05"]} />
      </Section>

      <Section idx="C" title="Barrier options" id="barrier">
        <p>
          A barrier option switches on or off if the price touches a set level. An{" "}
          <em>up-and-out</em> call, for example, pays like an ordinary call — unless the price ever
          reaches the barrier, at which point it is worthless for good. That cliff makes the
          contract cheaper, and it makes the price acutely sensitive to how often the barrier is
          checked, which is the subject of one of the studies.
        </p>
        <div className={styles.diagrams}>
          <PayoffDiagram kind="up-and-out" />
        </div>
        <p>
          Whether the barrier is monitored continuously, daily, or weekly is part of the contract,
          not a modelling choice — and a daily-monitored barrier is a genuinely different
          instrument from a continuously monitored one. The barrier study measures exactly how
          different.
        </p>
        <RelatedExperiments ids={["EXP-07"]} />
      </Section>
    </div>
  );
}
