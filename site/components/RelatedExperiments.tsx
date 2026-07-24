import Link from "next/link";
import { RECORDS, STUDY_IDS } from "@/lib/records.generated";
import { StatusBadge } from "./StatusBadge";
import styles from "./RelatedExperiments.module.css";

// Links to the studies that supply the evidence for a section. Each carries its
// finding status, so a reader on a narrative page can see at a glance whether the
// backing study was clean or flagged before opening it.

export function RelatedExperiments({ ids }: { ids: string[] }) {
  return (
    <aside className={styles.wrap} aria-label="Related studies">
      <p className={styles.label}>Related studies</p>
      <ul className={styles.list}>
        {ids.map((id) => {
          const r = RECORDS[id];
          if (!r) return null;
          const position = (STUDY_IDS as readonly string[]).indexOf(id) + 1;
          return (
            <li key={r.slug}>
              <Link href={`/studies/${r.slug}/`} className={styles.item}>
                <span className={styles.id}>{String(position).padStart(2, "0")}</span>
                <span className={styles.name}>{r.name}</span>
                <StatusBadge status={r.status} />
              </Link>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
