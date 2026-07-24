"use client";

import Link from "next/link";
import { useState } from "react";
import type { StudyRecord, StudyStatus } from "@/lib/records.generated";
import { StatusBadge } from "./StatusBadge";
import styles from "./ExperimentIndex.module.css";

type Filter = "all" | StudyStatus;

// A filterable index of all fifteen studies. The filter is a set of real toggle
// buttons (keyboard-operable, aria-pressed), and the count updates so a reader always
// knows how many are shown. Never hides information behind hover.

export function ExperimentIndex({ records }: { records: StudyRecord[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const shown = filter === "all" ? records : records.filter((r) => r.status === filter);

  const filters: { key: Filter; label: string }[] = [
    { key: "all", label: "All studies" },
    { key: "pass", label: "Clean result" },
    { key: "warning", label: "Flagged caveat" },
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
        {shown.map((r) => {
          const position = records.indexOf(r) + 1;
          return (
            <li key={r.id}>
              <Link href={`/studies/${r.slug}/`} className={styles.row} data-status={r.status}>
                <span className={styles.rowId}>{String(position).padStart(2, "0")}</span>
                <span className={styles.rowMain}>
                  <span className={styles.rowName}>{r.name}</span>
                  <span className={styles.rowQ}>{r.question}</span>
                </span>
                <span className={styles.rowStatus}>
                  <StatusBadge status={r.status} />
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
