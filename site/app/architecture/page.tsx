import type { Metadata } from "next";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { GENERATOR_COMMIT, MANIFEST, ORDERED_RECORDS } from "@/lib/records.generated";
import styles from "./architecture.module.css";

export const metadata: Metadata = {
  title: "Architecture & reproducibility",
  description:
    "The layered C++ core, the validation boundary, the Python and QuantLib boundaries, and how a displayed number traces back to a git commit.",
};

const LAYERS = [
  { name: "Configuration", note: "JSON, schema-versioned; a malformed config fails, never falls back to defaults" },
  { name: "CLI command", note: "price · simulate · greeks · validate · experiment · calibrate" },
  { name: "Instrument + MarketState + Model", note: "value types, validated at construction" },
  { name: "Pricing / Simulation / Calibration engine", note: "no I/O, no CLI, no plotting" },
  { name: "Statistics + Validation + Diagnostics", note: "standard errors, invariants, refusals" },
  { name: "JSON record / CSV summary / console", note: "the artifacts this site reads" },
];

export default function ArchitecturePage() {
  const figureCount = ORDERED_RECORDS.reduce((n, r) => n + r.figures.length, 0);

  return (
    <div className="page">
      <PageHeader
        stage="Stage 06 — architecture"
        title="Architecture & reproducibility"
        lede="The numerical core is independent of the interface, the plotting, and the validation adapters, so a result is a property of the engine and not of how it was invoked. Every published number traces back to a single git commit."
      />

      <Section idx="06.1" title="The layered core" id="layers">
        <p>
          The engine is one pipeline. Each layer depends only on those above it; the numerical
          core has no dependency on the CLI, on plotting, on QuantLib, or on this site.
        </p>
        <ol className={styles.stack} aria-label="Architecture layers, top to bottom">
          {LAYERS.map((layer, i) => (
            <li key={layer.name} className={styles.layer}>
              <span className={styles.layerIdx} aria-hidden="true">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className={styles.layerBody}>
                <span className={styles.layerName}>{layer.name}</span>
                <span className={styles.layerNote}>{layer.note}</span>
              </span>
            </li>
          ))}
        </ol>
        <p>
          Errors are values: every fallible operation returns a <code>Result&lt;T&gt;</code>,
          and no engine throws across an API boundary. An engine that cannot produce a
          trustworthy number <strong>refuses</strong> and says why, rather than returning a
          plausible one.
        </p>
      </Section>

      <Section idx="06.2" title="Concurrency, without a performance claim" id="concurrency">
        <p>
          Multithreading exists for one reason: to prove that threads do not change an answer.
          Randomness is a counter-based Philox stream keyed by path index, so a reduction is
          deterministic regardless of how work is scheduled, and a threaded run reproduces a
          serial one bit for bit. Race-safety is checked under ThreadSanitizer in CI. No part of
          the project measures or claims throughput.
        </p>
      </Section>

      <Section idx="06.3" title="The Python and QuantLib boundaries" id="boundaries">
        <p>
          Python computes nothing that appears in a record: it plots, aggregates, and checks.
          Every figure on this site is rendered from a committed record by a reporting-only
          script that fits nothing. QuantLib appears only as an external validation oracle in a
          single test target, never in a pricing path — a clean checkout builds and tests fully
          without it.
        </p>
      </Section>

      <Section idx="06.4" title="How a number traces to a commit" id="reproducibility">
        <p>
          Every record embeds the compiler, flags, git commit, and hardware that produced it —
          because the project builds with <code>-ffp-contract=off</code>, a slope produced under
          different flags is a different result. The whole artifact set was generated in one
          pass, from a clean tree, at a single commit.
        </p>
        <dl className={styles.facts}>
          <div>
            <dt>Generator commit</dt>
            <dd className="numeric">{GENERATOR_COMMIT}</dd>
          </div>
          <div>
            <dt>Tree at generation</dt>
            <dd>{MANIFEST.generator_tree_clean ? "clean" : "dirty"}</dd>
          </div>
          <div>
            <dt>Experiments</dt>
            <dd className="numeric">{ORDERED_RECORDS.length}</dd>
          </div>
          <div>
            <dt>Figures</dt>
            <dd className="numeric">{figureCount}</dd>
          </div>
        </dl>
        <p>
          The full inventory, with a SHA-256 for every record, CSV, and figure, is in the
          manifest. Its <code>artifact_commit</code> is null by construction — a manifest cannot
          name the commit that adds it.
        </p>
        <p>
          <a href="/artifacts/MANIFEST.json" download>
            Download MANIFEST.json
          </a>
        </p>
      </Section>
    </div>
  );
}
