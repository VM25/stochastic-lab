import Link from "next/link";
import { ORDERED_RECORDS } from "@/lib/records.generated";
import { STATUS_GLYPH } from "./StatusBadge";
import styles from "./ExperimentLattice.module.css";

// The signature: the fifteen experiments as a computational lattice. Each node is a
// checkpoint carrying its number, its status glyph, and its subject. The warnings
// are as visible as the passes — the honesty of the reconciliation, made structural.

export function ExperimentLattice() {
  return (
    <ul className={styles.lattice} aria-label="The fifteen experiments">
      {ORDERED_RECORDS.map((r) => {
        const num = r.id.replace("EXP-", "");
        return (
          <li key={r.id}>
            <Link href={`/experiments/${r.id.toLowerCase()}/`} className={styles.node} data-status={r.status}>
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
