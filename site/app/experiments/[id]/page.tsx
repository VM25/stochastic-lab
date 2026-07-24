import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { EXPERIMENT_IDS, RECORDS } from "@/lib/records.generated";
import { FIGURE_COPY } from "@/lib/figure-copy";
import { StatusBadge } from "@/components/StatusBadge";
import { Figure } from "@/components/Figure";
import { Section } from "@/components/Section";
import { SummaryTable } from "@/components/SummaryTable";
import { CopyField } from "@/components/CopyField";
import { RecordProvenance } from "@/components/RecordProvenance";
import styles from "./experiment.module.css";

export const dynamicParams = false;

export function generateStaticParams() {
  return EXPERIMENT_IDS.map((id) => ({ id: id.toLowerCase() }));
}

function recordFor(param: string) {
  return RECORDS[param.toUpperCase()];
}

export function generateMetadata({ params }: { params: { id: string } }): Metadata {
  const record = recordFor(params.id);
  if (!record) return {};
  return {
    title: `${record.id} — ${record.name}`,
    description: record.question,
  };
}

function renderConfigValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((v) => renderConfigValue(v)).join(", ")}]`;
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export default function ExperimentPage({ params }: { params: { id: string } }) {
  const record = recordFor(params.id);
  if (!record) notFound();

  const paragraphs = record.interpretation.split("\n\n").filter(Boolean);
  const configEntries = Object.entries(record.configuration);
  const index = (EXPERIMENT_IDS as readonly string[]).indexOf(record.id);
  const prev = index > 0 ? RECORDS[EXPERIMENT_IDS[index - 1]] : null;
  const next = index < EXPERIMENT_IDS.length - 1 ? RECORDS[EXPERIMENT_IDS[index + 1]] : null;

  return (
    <article className="page">
      <p className={styles.trail}>
        <Link href="/experiments/">Experiment program</Link> / {record.id}
      </p>

      <header className={styles.head}>
        <div className={styles.headTop}>
          <span className="eyebrow">{record.id}</span>
          <StatusBadge status={record.status} />
        </div>
        <h1>{record.name}</h1>
        {/* 1. Question */}
        <p className={styles.question}>{record.question}</p>
      </header>

      {/* 2-6. Figures, then the engine's own interpretation (result, error, cost). */}
      <Section idx="fig" title="Figures" id="figures">
        {record.figures.map((file) => {
          const copy = FIGURE_COPY[file];
          return (
            <Figure
              key={file}
              file={file}
              alt={copy?.alt ?? record.name}
              caption={copy?.caption ?? record.name}
            />
          );
        })}
      </Section>

      <Section idx="int" title="Interpretation" id="interpretation">
        {paragraphs.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </Section>

      <Section idx="tab" title="Summary" id="summary">
        <SummaryTable table={record.table} caption={`${record.id} summary table (from ${record.id}.csv)`} />
      </Section>

      {/* 8. Limitations — populated on every record, including the passing ones. */}
      <Section idx="lim" title="Limitations" id="limitations">
        <ul className={styles.limitations}>
          {record.limitations.map((lim, i) => (
            <li key={i}>{lim}</li>
          ))}
        </ul>
      </Section>

      {/* 9. Reproduction and provenance. */}
      <Section idx="cfg" title="Configuration" id="configuration">
        <dl className="defs">
          {configEntries.map(([key, value]) => (
            <div key={key} style={{ display: "contents" }}>
              <dt>{key}</dt>
              <dd className="numeric">{renderConfigValue(value)}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section idx="rep" title="Reproduction & provenance" id="reproduction">
        <CopyField label="Regenerate this record" value={record.reproduction_command} />
        <RecordProvenance record={record} />
      </Section>

      <nav className={styles.pager} aria-label="Adjacent experiments">
        {prev ? (
          <Link href={`/experiments/${prev.id.toLowerCase()}/`} className={styles.pagerLink}>
            <span className={styles.pagerDir}>← {prev.id}</span>
            <span className={styles.pagerName}>{prev.name}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/experiments/${next.id.toLowerCase()}/`} className={`${styles.pagerLink} ${styles.pagerNext}`}>
            <span className={styles.pagerDir}>{next.id} →</span>
            <span className={styles.pagerName}>{next.name}</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
