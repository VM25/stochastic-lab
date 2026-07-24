import type { StudyStatus } from "@/lib/records.generated";
import styles from "./StatusBadge.module.css";

// Status is never conveyed by colour alone (DESIGN-SPEC accessibility): each state
// carries a distinct glyph and the word itself. The glyphs are geometric, not
// emoji, so they render identically everywhere.
const GLYPH: Record<StudyStatus, string> = {
  pass: "▲", // filled triangle up
  warning: "◆", // filled diamond
  fail: "■", // filled square
  inconclusive: "○", // hollow circle
};

// Research verdicts, not test statuses. A clean study reached a firm result; a flagged
// one carries a caveat a reader should know before quoting it.
const LABEL: Record<StudyStatus, string> = {
  pass: "clean",
  warning: "flagged",
  fail: "unresolved",
  inconclusive: "inconclusive",
};

export function StatusBadge({ status }: { status: StudyStatus }) {
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
