"use client";

import { useState } from "react";
import styles from "./CopyField.module.css";

// A copyable command or value. The text is always visible (never hover-only), and the
// copy button degrades gracefully: if the clipboard API is unavailable the field
// still shows and can be selected manually.

export function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={styles.field}>
      <span className={styles.label}>{label}</span>
      <div className={styles.row}>
        <code className={styles.value}>{value}</code>
        <button type="button" className={styles.button} onClick={copy} aria-label={`Copy ${label}`}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </div>
  );
}
