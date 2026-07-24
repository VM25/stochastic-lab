import styles from "./StochasticPaths.module.css";

// The first thing a visitor sees: a real diffusion. These are genuine geometric
// Brownian motion sample paths — the process at the heart of the project — simulated
// deterministically at build time and drawn as static SVG, with the terminal
// distribution they induce sketched on the right. Not decoration: the same dynamics
// every model on the site prices against.

// A small deterministic PRNG so the picture is identical on every build.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number): number {
  // Box–Muller.
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const WIDTH = 1000;
const HEIGHT = 440;
const PLOT_W = 830; // leave room on the right for the terminal distribution
const PATHS = 30;
const STEPS = 140;
const S0 = 100;
const MU = 0.05;
const SIGMA = 0.32;
const T = 1;

export function StochasticPaths() {
  const rng = mulberry32(20260724);
  const dt = T / STEPS;
  const drift = (MU - 0.5 * SIGMA * SIGMA) * dt;
  const vol = SIGMA * Math.sqrt(dt);

  const paths: number[][] = [];
  for (let p = 0; p < PATHS; p++) {
    const s: number[] = [S0];
    for (let i = 1; i <= STEPS; i++) {
      s.push(s[i - 1] * Math.exp(drift + vol * gauss(rng)));
    }
    paths.push(s);
  }

  const terminals = paths.map((s) => s[STEPS]);
  const lo = Math.min(S0 * 0.45, ...paths.flat());
  const hi = Math.max(S0 * 1.55, ...paths.flat());

  const xOf = (i: number) => (i / STEPS) * PLOT_W;
  const yOf = (price: number) => HEIGHT - ((price - lo) / (hi - lo)) * HEIGHT;

  const polylines = paths.map((s) => s.map((price, i) => `${xOf(i).toFixed(1)},${yOf(price).toFixed(1)}`).join(" "));

  // The terminal density as a small horizontal histogram on the right.
  const BINS = 26;
  const counts = new Array(BINS).fill(0);
  for (const v of terminals) {
    const b = Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * BINS)));
    counts[b] += 1;
  }
  const maxCount = Math.max(...counts, 1);
  const binH = HEIGHT / BINS;

  const startY = yOf(S0);

  return (
    <div className={styles.wrap} aria-hidden="true">
      <svg
        className={styles.svg}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="xMidYMid slice"
        role="presentation"
      >
        {/* the strike / start reference */}
        <line
          x1="0"
          y1={startY}
          x2={PLOT_W}
          y2={startY}
          className={styles.reference}
        />
        {/* the sample paths */}
        <g className={styles.paths}>
          {polylines.map((pts, i) => (
            <polyline
              key={i}
              points={pts}
              className={styles.path}
              style={{ animationDelay: `${(i * 60).toFixed(0)}ms` }}
            />
          ))}
        </g>
        {/* the terminal distribution the paths induce */}
        <g className={styles.hist}>
          {counts.map((c, b) => {
            const w = (c / maxCount) * (WIDTH - PLOT_W - 12);
            const y = HEIGHT - (b + 1) * binH;
            return <rect key={b} x={PLOT_W + 8} y={y + 1} width={w.toFixed(1)} height={(binH - 2).toFixed(1)} />;
          })}
        </g>
      </svg>
    </div>
  );
}
