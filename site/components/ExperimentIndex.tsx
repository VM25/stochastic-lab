"use client";

import Link from "next/link";
import { useState } from "react";
import type { ExperimentRecord, ExperimentStatus } from "@/lib/records.generated";
import { StatusBadge } from "./StatusBadge";
import styles from "./ExperimentIndex.module.css";

type Filter = "all" | ExperimentStatus;

// A filterable index of all fifteen experiments. The filter is a set of real toggle
// buttons (keyboard-operable, aria-pressed), and the count updates so a reader always
// knows how many are shown. Never hides information behind hover.

export function ExperimentIndex({ records }: { records: ExperimentRecord[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const shown = filter === "all" ? records : records.filter((r) => r.status === filter);

  const filters: { key: Filter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "pass", label: "Pass" },
    { key: "warning", label: "Warning" },
  ];

  return (
    <div>
      <div className={styles.controls} role="group" aria-label="Filter experiments by status">
        {filters.map((f) => (
          <button
            key={f.key}
            type="button"
            className={styles.filter}
            data-active={filter === f.key}
            aria-pressed={filter === f.key}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
        <span className={styles.count} aria-live="polite">
          {shown.length} of {records.length}
        </span>
      </div>

      <ol className={styles.list}>
        {shown.map((r) => (
          <li key={r.id}>
            <Link href={`/experiments/${r.id.toLowerCase()}/`} className={styles.row} data-status={r.status}>
              <span className={styles.rowId}>{r.id}</span>
              <span className={styles.rowMain}>
                <span className={styles.rowName}>{r.name}</span>
                <span className={styles.rowQ}>{r.question}</span>
              </span>
              <span className={styles.rowStatus}>
                <StatusBadge status={r.status} />
              </span>
            </Link>
          </li>
        ))}
      </ol>
    </div>
  );
}
