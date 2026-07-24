import styles from "./PayoffDiagram.module.css";

// The payoff of a contract as a function of the final price — the geometry that
// defines what the option is worth at expiry. Drawn as static SVG from the payoff
// formula itself, so the shape is the mathematics, not an illustration.

type Kind = "call" | "put" | "up-and-out";

const W = 260;
const H = 160;
const PAD = 28;
const K = 100; // strike, at the middle of the axis
const B = 130; // barrier, for the knock-out
const S_MAX = 170;
const S_MIN = 40;

function x(s: number): number {
  return PAD + ((s - S_MIN) / (S_MAX - S_MIN)) * (W - 2 * PAD);
}
function y(payoff: number): number {
  const max = 70;
  return H - PAD - (Math.min(payoff, max) / max) * (H - 2 * PAD);
}

function payoffPath(kind: Kind): { d: string; extras: React.ReactNode } {
  const pts: [number, number][] = [];
  for (let s = S_MIN; s <= S_MAX; s += 1) {
    let v = 0;
    if (kind === "call") v = Math.max(s - K, 0);
    else if (kind === "put") v = Math.max(K - s, 0);
    else v = s >= B ? 0 : Math.max(s - K, 0); // up-and-out call: worthless once S reaches the barrier
    pts.push([x(s), y(v)]);
  }
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const extras =
    kind === "up-and-out" ? (
      <line className={styles.barrier} x1={x(B)} y1={PAD - 6} x2={x(B)} y2={H - PAD} />
    ) : null;
  return { d, extras };
}

const LABEL: Record<Kind, string> = {
  call: "European call",
  put: "European put",
  "up-and-out": "Up-and-out call",
};

export function PayoffDiagram({ kind }: { kind: Kind }) {
  const { d, extras } = payoffPath(kind);
  return (
    <figure className={styles.fig}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Payoff diagram for a ${LABEL[kind].toLowerCase()}`}>
        {/* axes */}
        <line className={styles.axis} x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} />
        <line className={styles.axis} x1={PAD} y1={PAD - 6} x2={PAD} y2={H - PAD} />
        {/* strike marker */}
        <line className={styles.strike} x1={x(K)} y1={H - PAD} x2={x(K)} y2={H - PAD + 5} />
        <text className={styles.tick} x={x(K)} y={H - PAD + 15} textAnchor="middle">
          K
        </text>
        {extras}
        <path className={styles.payoff} d={d} />
      </svg>
      <figcaption>{LABEL[kind]}</figcaption>
    </figure>
  );
}
