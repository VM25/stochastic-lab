import type { ExperimentStatus } from "@/lib/records.generated";
import styles from "./StatusBadge.module.css";

// Status is never conveyed by colour alone (DESIGN-SPEC accessibility): each state
// carries a distinct glyph and the word itself. The glyphs are geometric, not
// emoji, so they render identically everywhere.
const GLYPH: Record<ExperimentStatus, string> = {
  pass: "▲", // filled triangle up
  warning: "◆", // filled diamond
  fail: "■", // filled square
  inconclusive: "○", // hollow circle
};

const LABEL: Record<ExperimentStatus, string> = {
  pass: "pass",
  warning: "warning",
  fail: "fail",
  inconclusive: "inconclusive",
};

export function StatusBadge({ status }: { status: ExperimentStatus }) {
  return (
    <span className={styles.badge} data-status={status}>
      <span className={styles.glyph} aria-hidden="true">
        {GLYPH[status]}
      </span>
      <span className={styles.label}>{LABEL[status]}</span>
    </span>
  );
}

export { GLYPH as STATUS_GLYPH };
