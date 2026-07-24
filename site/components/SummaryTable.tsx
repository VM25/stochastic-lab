import type { SummaryTable as SummaryTableData } from "@/lib/records.generated";
import styles from "./SummaryTable.module.css";

// The record's own summary table, rendered as an accessible HTML table with column
// headers. Numeric-looking cells are set in the tabular monospace so columns align.
// Values are rendered verbatim; nothing is reformatted or recomputed.

const NUMERIC = /^-?\d/;

export function SummaryTable({ table, caption }: { table: SummaryTableData; caption: string }) {
  return (
    <div className={styles.scroll}>
      <table className={styles.table}>
        <caption className={styles.caption}>{caption}</caption>
        <thead>
          <tr>
            {table.headers.map((h) => (
              <th scope="col" key={h}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => {
                const text = String(cell);
                const numeric = NUMERIC.test(text);
                return (
                  <td key={j} className={numeric ? styles.num : undefined} data-label={table.headers[j]}>
                    {text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
