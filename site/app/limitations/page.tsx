import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { StatusBadge } from "@/components/StatusBadge";
import { ORDERED_RECORDS, STUDY_IDS } from "@/lib/records.generated";
import styles from "./limitations.module.css";

export const metadata: Metadata = {
  title: "Limitations & findings",
  description:
    "The five flagged studies, what each caveat means for a reader, and the regions where each method stops being reliable.",
};

// Plain-English framing for each flagged study — what the caveat means and why it is a
// caveat rather than a clean pass. Keyed by the internal study id, not shown to readers.
const WARNING_MEANING: Record<string, string> = {
  "EXP-02":
    "The convergence rate looks slightly too slow if you measure it on coarse steps, and reaches its textbook value only once the steps are fine enough. Both estimates are shown, so the rate is never quoted before it has genuinely settled.",
  "EXP-07":
    "Two caveats a clean pass would hide. For one barrier the bias was too small to measure reliably, so the study declines to fit a number to noise rather than publish a shaky one. And a standard textbook correction, accurate for one type of barrier, turns out to be measurably wrong for the other — a useful thing to know before relying on it.",
  "EXP-10":
    "The pricing scheme handles the difficult, near-degenerate regime and its error shrinks in the expected way there. In the easy regime the error is already so small it cannot be measured, so the convergence rate is confirmed for one regime and left open for the other rather than assumed to hold for both.",
  "EXP-12":
    "Every fit to the real market matches the observed prices well, yet the fitted parameters disagree by more than their own size from one fit to the next. A good fit is not a determined model, so the study reports fit quality and parameter certainty as two separate findings.",
  "EXP-14":
    "The reported 95% confidence interval is trustworthy in normal conditions and unreliable for a rare, extreme payoff when too few paths are used — where it covers the true value far less often than it claims. More paths restore it; the caveat is about knowing when it applies.",
};

const FAILURE_REGIONS = [
  {
    region: "Deep out-of-the-money, short maturity",
    detail:
      "The hardest region for any simulation-based estimate. The option pays on very few paths, so every sampled quantity is noisy, and it is where a confidence interval is least reliable.",
  },
  {
    region: "Barriers close to the current price",
    detail:
      "A knock-out barrier just above the spot makes the contract nearly worthless and acutely sensitive to how it is monitored, and the usual closed-form correction does not apply there.",
  },
  {
    region: "Long-run volatility from a short-dated market",
    detail:
      "A market with only a few maturities cannot separate how fast volatility reverts from the level it reverts to; those parameters are simply not determined by the data.",
  },
  {
    region: "A kinked payoff on a coarse grid",
    detail:
      "One popular grid-based scheme oscillates on coarse time steps near an option's payoff kink. A standard smoothing step fixes it, at a cost that is measured rather than assumed.",
  },
];

export default function LimitationsPage() {
  const warnings = ORDERED_RECORDS.filter((r) => r.status === "warning");

  return (
    <div className="page">
      <PageHeader
        stage="Limitations & findings"
        title="Where the methods fall short"
        lede="Five of the fifteen studies end in a caveat rather than a clean result, and these are the most useful findings in the project. Each is a place where the obvious reading of a number would be wrong — and none is a malfunction, only an honest limit worth stating out loud."
      />

      <Section idx="A" title="The five flagged studies" id="warnings">
        <ul className={styles.warnings}>
          {warnings.map((r) => {
            const position = (STUDY_IDS as readonly string[]).indexOf(r.id) + 1;
            return (
              <li key={r.slug} className={styles.warning}>
                <div className={styles.warningHead}>
                  <Link href={`/studies/${r.slug}/`} className={styles.warningTitle}>
                    <span className={styles.warningId}>Study {String(position).padStart(2, "0")}</span>
                    <span>{r.name}</span>
                  </Link>
                  <StatusBadge status={r.status} />
                </div>
                <p className={styles.warningQ}>{r.question}</p>
                <p className={styles.warningMeaning}>{WARNING_MEANING[r.id]}</p>
                <Link href={`/studies/${r.slug}/`} className={styles.warningMore}>
                  Read the full study →
                </Link>
              </li>
            );
          })}
        </ul>
      </Section>

      <Section idx="B" title="Known trouble spots" id="regions">
        <p>
          Independent of the individual studies, four regions are where a given method should not be
          trusted without care:
        </p>
        <dl className={styles.regions}>
          {FAILURE_REGIONS.map((f) => (
            <div key={f.region} className={styles.region}>
              <dt>{f.region}</dt>
              <dd>{f.detail}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section idx="C" title="What this says about the project" id="conclusion">
        <p>
          None of these is hidden, and none is a defect in the work — each was found by looking for
          it and is reported in full. The value of the project is not that the methods never fall
          short, but that it says clearly where they do. A number you can trust is one that comes
          with an honest account of when it can be trusted.
        </p>
      </Section>
    </div>
  );
}
