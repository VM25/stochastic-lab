import Link from "next/link";
import { RECORDS } from "@/lib/records.generated";
import { StatusBadge } from "./StatusBadge";
import styles from "./RelatedExperiments.module.css";

// A row of links to the experiments that supply the evidence for a section. Each
// carries its live status from the record, so a reader on a prose page can see at a
// glance whether the backing experiment passed or warned before opening it.

export function RelatedExperiments({ ids }: { ids: string[] }) {
  return (
    <aside className={styles.wrap} aria-label="Backing experiments">
      <p className={styles.label}>Evidence</p>
      <ul className={styles.list}>
        {ids.map((id) => {
          const r = RECORDS[id];
          if (!r) return null;
          return (
            <li key={id}>
              <Link href={`/experiments/${id.toLowerCase()}/`} className={styles.item}>
                <span className={styles.id}>{r.id}</span>
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
