import Link from "next/link";
import { Section } from "@/components/Section";
import { ExperimentLattice } from "@/components/ExperimentLattice";
import { SPINE } from "@/lib/nav";
import { STATUS_COUNTS, GENERATOR_COMMIT, ORDERED_RECORDS } from "@/lib/records.generated";
import styles from "./page.module.css";

const total = ORDERED_RECORDS.length;

export default function OverviewPage() {
  return (
    <div className="page">
      <header className={styles.hero}>
        <p className="eyebrow">A C++ stochastic derivatives modeling and validation engine</p>
        <h1 className={styles.title}>
          DiffusionWorks<span className={styles.titleSub}>Stochastic Derivatives Numerics</span>
        </h1>
        <p className={styles.thesis}>
          A pricing engine is easy to write and hard to trust. This one is built to be
          checked: fifteen numerical experiments measure what the code actually does, every
          number traces to a committed artifact, and the places where the numerics fall short
          are measured and kept rather than smoothed away.
        </p>

        <dl className={styles.ledger} aria-label="Experiment reconciliation">
          <div className={styles.ledgerItem}>
            <dt>Experiments</dt>
            <dd className="numeric">{total}</dd>
          </div>
          <div className={styles.ledgerItem} data-status="pass">
            <dt>Pass</dt>
            <dd className="numeric">{STATUS_COUNTS.pass ?? 0}</dd>
          </div>
          <div className={styles.ledgerItem} data-status="warning">
            <dt>Warning</dt>
            <dd className="numeric">{STATUS_COUNTS.warning ?? 0}</dd>
          </div>
          <div className={styles.ledgerItem} data-status="fail">
            <dt>Fail</dt>
            <dd className="numeric">{STATUS_COUNTS.fail ?? 0}</dd>
          </div>
        </dl>
        <p className={styles.ledgerNote}>
          The five warnings are the substance. Each experiment ran to completion and answered
          its question; a warning marks a caveat that changes what may be quoted — not a
          malfunction.
        </p>
      </header>

      <Section idx="Q" title="The research question" id="question">
        <p>
          Can a from-scratch C++ derivatives engine be made auditable end to end — its
          convergence rates, its cross-method agreement, its calibration, and its failure
          regions all reproducible from committed evidence rather than asserted? The answer is
          the fifteen experiments below, and the honest reconciliation of their results.
        </p>
      </Section>

      <Section idx="03" title="The evidence lattice" id="lattice">
        <p>
          Every mandatory experiment, with its status shown by glyph and word as well as
          colour. Select one to read its question, method, reference, result, uncertainty,
          interpretation, limitations, and the exact command that regenerates it.
        </p>
        <ExperimentLattice />
        <p className={styles.latticeFoot}>
          <Link href="/experiments/">Open the experiment program →</Link>
        </p>
      </Section>

      <Section idx="→" title="From model to evidence" id="pipeline">
        <p>
          The engine, and this site, are organised as one pipeline. Each stage is a page.
        </p>
        <ol className={styles.spine} aria-label="The computation pipeline">
          {SPINE.map((stage, i) => (
            <li key={stage} className={styles.spineNode}>
              <span className={styles.spineIdx} aria-hidden="true">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className={styles.spineLabel}>{stage}</span>
            </li>
          ))}
        </ol>
      </Section>

      <Section idx="↳" title="Where to go next" id="links">
        <ul className={styles.linkGrid}>
          <li>
            <Link href="/models/">Models &amp; instruments</Link>
            <span>Black–Scholes, Heston, and the contracts, with their failure regions.</span>
          </li>
          <li>
            <Link href="/methods/">Numerical methods</Link>
            <span>Simulation, variance reduction, PDE, Greeks, calibration.</span>
          </li>
          <li>
            <Link href="/validation/">Validation</Link>
            <span>References, financial invariants, and confidence coverage.</span>
          </li>
          <li>
            <Link href="/calibration/">Calibration &amp; market surface</Link>
            <span>A real surface every scenario fits and none identifies.</span>
          </li>
          <li>
            <Link href="/architecture/">Architecture &amp; reproducibility</Link>
            <span>How a displayed number traces back to a git commit.</span>
          </li>
          <li>
            <Link href="/limitations/">Limitations &amp; failure regions</Link>
            <span>The five warnings, and where each method stops working.</span>
          </li>
        </ul>
      </Section>

      <p className={styles.provenance}>
        Artifacts generated at commit <code>{GENERATOR_COMMIT}</code> from a clean tree.
      </p>
    </div>
  );
}
