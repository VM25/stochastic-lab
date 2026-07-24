import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { StatusBadge } from "@/components/StatusBadge";
import { ORDERED_RECORDS } from "@/lib/records.generated";
import styles from "./limitations.module.css";

export const metadata: Metadata = {
  title: "Limitations & failure regions",
  description:
    "The five warning experiments, what each caveat means, and the regions where each method stops working.",
};

// Editorial framing for each warning — what the caveat means and why it is a warning
// rather than a pass. Number-free; the specific figures live on each experiment's own
// page, rendered from its record.
const WARNING_MEANING: Record<string, string> = {
  "EXP-02":
    "The full-range convergence slopes fall short of theory while the asymptotic-window slopes cover it, because the error is only leading-order a power law. Both fits are published; the local orders climb toward theory as the grid refines, which is what distinguishes this from a wrong order.",
  "EXP-07":
    "Two caveats a bare pass would hide. One barrier arm's bias never cleared its own noise, so the experiment refuses to fit it rather than publish a fitted order for a curve of noise. And the closed-form continuity correction, reliable for down-barriers, is measurably wrong for up-barriers — kept in the record as a reference that fails there, not removed.",
  "EXP-10":
    "Full truncation prices every regime without a single non-finite path, and its bias decays measurably where it can be resolved — but that is only one of the two regimes. In the benign regime the bias never clears the noise, so no decay order is fitted, and the scheme's order is established for one regime and not the other.",
  "EXP-12":
    "Every calibration scenario converges and fits the real surface well, and the parameters still disagree by more than an order of magnitude on the pair governing the long run. A good fit is not an identified parameter set, so the record reports fit quality and parameter dispersion as separate findings.",
  "EXP-14":
    "The nominal 95% confidence interval covers at its claimed rate where the central limit theorem holds, and under-covers at a small sample with a severely skewed payoff. The degradation is expected and explained, and it is still a warning: quoting the interval in that regime would quote something the experiment disproved.",
};

const FAILURE_REGIONS = [
  {
    region: "Deep out-of-the-money, short maturity",
    detail:
      "The worst region for every sampled estimator. Few paths pay, so relative noise is largest; the likelihood-ratio Greek variance is worst here, and it is where the confidence interval under-covers.",
  },
  {
    region: "Up-and-out barriers near the spot",
    detail:
      "The continuity correction is not usable, and the contract's tiny price makes the relative monitoring bias enormous.",
  },
  {
    region: "Long-run Heston parameters from a short-dated surface",
    detail:
      "Mean reversion and the long-run variance level are not identified: they trade off along a valley the data does not resolve.",
  },
  {
    region: "Plain Crank–Nicolson on a kinked payoff at coarse steps",
    detail:
      "The scheme oscillates. Use the Rannacher start-up, whose cost and benefit are both measured in EXP-06.",
  },
];

export default function LimitationsPage() {
  const warnings = ORDERED_RECORDS.filter((r) => r.status === "warning");

  return (
    <div className="page">
      <PageHeader
        stage="Stage 07 — the limits"
        title="Limitations & failure regions"
        lede="The five warnings are the most useful results in the programme, because each is a place where the obvious reading of a number would be wrong. None is a defect in the engine; each is a fact about the numerics that a project reporting only its successes would not have found."
      />

      <Section idx="07.1" title="The five warnings" id="warnings">
        <ul className={styles.warnings}>
          {warnings.map((r) => (
            <li key={r.id} className={styles.warning}>
              <div className={styles.warningHead}>
                <Link href={`/experiments/${r.id.toLowerCase()}/`} className={styles.warningTitle}>
                  <span className={styles.warningId}>{r.id}</span>
                  <span>{r.name}</span>
                </Link>
                <StatusBadge status={r.status} />
              </div>
              <p className={styles.warningQ}>{r.question}</p>
              <p className={styles.warningMeaning}>{WARNING_MEANING[r.id]}</p>
              <Link href={`/experiments/${r.id.toLowerCase()}/`} className={styles.warningMore}>
                Full record & figures →
              </Link>
            </li>
          ))}
        </ul>
      </Section>

      <Section idx="07.2" title="Known failure regions" id="regions">
        <p>
          Independent of the experiment statuses, four regions are where a given method should
          not be trusted without care:
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

      <Section idx="07.3" title="What none of this is" id="not">
        <p>
          A warning is not a malfunction, and none of these regions is hidden. Every one is
          measured, published, and kept in a committed record — which is the whole point of the
          programme. The engine&apos;s value is not that it never falls short, but that it says
          exactly where it does.
        </p>
      </Section>
    </div>
  );
}
