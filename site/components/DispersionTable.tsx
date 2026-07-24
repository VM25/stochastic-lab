import { RECORDS } from "@/lib/records.generated";
import styles from "./DispersionTable.module.css";

// The EXP-12 parameter dispersion, read from the record. Each row's min, max, and
// mean come straight from results.parameter_dispersion; the relative-dispersion
// column is stddev / |mean|, a display derivation from those same record fields. No
// number here is authored by hand.

interface Spread {
  min: number;
  max: number;
  mean: number;
  stddev: number;
}

const ROWS: { key: string; label: string; identified: boolean }[] = [
  { key: "correlation", label: "Correlation ρ", identified: true },
  { key: "initial_variance", label: "Initial variance v₀", identified: true },
  { key: "vol_of_variance", label: "Vol-of-variance ξ", identified: true },
  { key: "long_run_variance", label: "Long-run variance θ", identified: false },
  { key: "mean_reversion", label: "Mean reversion κ", identified: false },
];

function fmt(x: number): string {
  const abs = Math.abs(x);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return x.toExponential(2);
  return x.toFixed(abs < 1 ? 3 : 2);
}

export function DispersionTable() {
  const dispersion = RECORDS["EXP-12"].results.parameter_dispersion as Record<string, Spread>;
  return (
    <div className={styles.scroll}>
      <table className={styles.table}>
        <caption className={styles.caption}>
          Calibrated Heston parameters across seven fitting scenarios
        </caption>
        <thead>
          <tr>
            <th scope="col">Parameter</th>
            <th scope="col">Min</th>
            <th scope="col">Max</th>
            <th scope="col">Relative dispersion</th>
            <th scope="col">Identified?</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ key, label, identified }) => {
            const s = dispersion[key];
            const relative = Math.abs(s.stddev / s.mean);
            return (
              <tr key={key} data-identified={identified}>
                <th scope="row">{label}</th>
                <td className={styles.num}>{fmt(s.min)}</td>
                <td className={styles.num}>{fmt(s.max)}</td>
                <td className={styles.num}>{(relative * 100).toFixed(0)}%</td>
                <td className={styles.verdict} data-identified={identified}>
                  {identified ? "determined" : "not identified"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
