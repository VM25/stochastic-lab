import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { ORDERED_RECORDS, RECORD_BY_SLUG, STUDY_IDS } from "@/lib/records.generated";
import { FIGURE_COPY } from "@/lib/figure-copy";
import { StatusBadge } from "@/components/StatusBadge";
import { Figure } from "@/components/Figure";
import { Section } from "@/components/Section";
import styles from "./study.module.css";

export const dynamicParams = false;

export function generateStaticParams() {
  return ORDERED_RECORDS.map((r) => ({ slug: r.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }): Metadata {
  const record = RECORD_BY_SLUG[params.slug];
  if (!record) return {};
  return { title: record.name, description: record.question };
}

export default function StudyPage({ params }: { params: { slug: string } }) {
  const record = RECORD_BY_SLUG[params.slug];
  if (!record) notFound();

  const paragraphs = record.interpretation.split("\n\n").filter(Boolean);
  const position = (STUDY_IDS as readonly string[]).indexOf(record.id);
  const prev = position > 0 ? ORDERED_RECORDS[position - 1] : null;
  const next = position < ORDERED_RECORDS.length - 1 ? ORDERED_RECORDS[position + 1] : null;
  const lead = paragraphs[0];
  const rest = paragraphs.slice(1);

  return (
    <article className="page">
      <p className={styles.trail}>
        <Link href="/studies/">Research studies</Link> / Study {String(position + 1).padStart(2, "0")}
      </p>

      <header className={styles.head}>
        <div className={styles.headTop}>
          <span className="eyebrow">Study {String(position + 1).padStart(2, "0")}</span>
          <StatusBadge status={record.status} />
        </div>
        <h1>{record.name}</h1>
        <p className={styles.question}>
          <span className={styles.qlabel}>The question</span>
          {record.question}
        </p>
      </header>

      <Section idx="→" title="What the study found" id="finding">
        {lead && <p className={styles.lead}>{lead}</p>}
        {rest.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </Section>

      <Section idx="◇" title="Evidence" id="evidence">
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

      <Section idx="!" title="What this does not establish" id="limitations">
        <ul className={styles.limitations}>
          {record.limitations.map((lim, i) => (
            <li key={i}>{lim}</li>
          ))}
        </ul>
      </Section>

      <nav className={styles.pager} aria-label="Adjacent studies">
        {prev ? (
          <Link href={`/studies/${prev.slug}/`} className={styles.pagerLink}>
            <span className={styles.pagerDir}>← Previous</span>
            <span className={styles.pagerName}>{prev.name}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/studies/${next.slug}/`} className={`${styles.pagerLink} ${styles.pagerNext}`}>
            <span className={styles.pagerDir}>Next →</span>
            <span className={styles.pagerName}>{next.name}</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
