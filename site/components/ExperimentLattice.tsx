import Link from "next/link";
import { ORDERED_RECORDS } from "@/lib/records.generated";
import { STATUS_GLYPH } from "./StatusBadge";
import styles from "./ExperimentLattice.module.css";

// The fifteen studies as a lattice of nodes. Each carries its number in the sequence,
// a status glyph, and its subject. The five flagged findings are as visible as the
// clean ones — the honesty of the programme, made structural.

export function ExperimentLattice() {
  return (
    <ul className={styles.lattice} aria-label="The fifteen studies">
      {ORDERED_RECORDS.map((r, i) => {
        const num = String(i + 1).padStart(2, "0");
        return (
          <li key={r.slug}>
            <Link href={`/studies/${r.slug}/`} className={styles.node} data-status={r.status}>
              <span className={styles.top}>
                <span className={styles.num}>{num}</span>
                <span className={styles.glyph} aria-hidden="true" data-status={r.status}>
                  {STATUS_GLYPH[r.status]}
                </span>
              </span>
              <span className={styles.name}>{r.name}</span>
              <span className={styles.status} data-status={r.status}>
                {r.status}
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
