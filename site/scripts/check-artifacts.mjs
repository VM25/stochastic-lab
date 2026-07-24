// Two guarantees for a public research site:
//
// 1. No result number is hand-authored. Every number the site shows comes from the
//    generated data module (built from the study records). This greps the hand-written
//    source for result-shaped numeric literals and fails on any outside a small
//    allowlist of fixed conventions, so a number can never drift from its record by
//    being typed into prose.
//
// 2. Every rendered figure is presented and shipped.
//
// The data files behind the site are NOT exposed — that is enforced separately in
// check-public.mjs, which also forbids developer-facing language.

import fs from "node:fs";
import path from "node:path";
import { SITE, OUT, REPO, ensureExport, htmlFiles, attrValues } from "./_html.mjs";

ensureExport();

const failures = [];

// ---- 1. Every rendered figure is presented and shipped -----------------------------
// The committed figures are copied under clean, descriptive names; there must be one
// per committed figure, and every one must be shown on some page.
const committedCount = fs
  .readdirSync(path.join(REPO, "docs", "figures"))
  .filter((f) => f.endsWith(".png")).length;
const exportedFigures = fs.existsSync(path.join(OUT, "figures"))
  ? fs.readdirSync(path.join(OUT, "figures")).filter((f) => f.endsWith(".png")).sort()
  : [];
if (exportedFigures.length !== committedCount) {
  failures.push(`${exportedFigures.length} figures in the export, ${committedCount} committed`);
}
if (exportedFigures.some((f) => /^exp\d{2}_/.test(f) || f.includes("_"))) {
  failures.push("a figure is served under its internal name (should be a clean, hyphenated name)");
}

const referencedFigures = new Set();
for (const file of htmlFiles()) {
  const html = fs.readFileSync(file, "utf8");
  for (const src of attrValues(html, "src")) {
    const m = src.match(/\/figures\/([a-z0-9-]+\.png)/);
    if (m) referencedFigures.add(m[1]);
  }
}
for (const fig of exportedFigures) {
  if (!referencedFigures.has(fig)) failures.push(`figure ${fig} is rendered nowhere on the site`);
}

// ---- 2. No hand-authored result numbers --------------------------------------------
const HANDWRITTEN = [];
const walkSrc = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (["node_modules", ".next", "out"].includes(entry.name)) continue;
      walkSrc(full);
    } else if (/\.(tsx?|mjs)$/.test(entry.name)) {
      if (full.endsWith("records.generated.ts")) continue;
      if (full.includes(`${path.sep}scripts${path.sep}`)) continue;
      HANDWRITTEN.push(full);
    }
  }
};
walkSrc(path.join(SITE, "app"));
walkSrc(path.join(SITE, "components"));
walkSrc(path.join(SITE, "lib"));

// A measured result looks like a decimal with two or more fraction digits or a
// percentage. Deterministic path/payoff visuals legitimately use geometry constants,
// so the two SVG components that draw them are exempt from this scan.
const RESULT_LITERAL = /\b\d{1,4}\.\d{2,}\b|\b\d{1,3}%/g;
const ALLOW = new Set(["95", "14.2", "0.16", "2.35"]);
const GEOMETRY_EXEMPT = new Set(["StochasticPaths.tsx", "PayoffDiagram.tsx"]);

for (const file of HANDWRITTEN) {
  if (GEOMETRY_EXEMPT.has(path.basename(file))) continue;
  const text = fs.readFileSync(file, "utf8");
  text.split("\n").forEach((line, i) => {
    const matches = line.match(RESULT_LITERAL);
    if (!matches) return;
    for (const raw of matches) {
      if (ALLOW.has(raw.replace("%", ""))) continue;
      failures.push(
        `${path.relative(SITE, file)}:${i + 1}: hand-authored result-shaped number "${raw}" — ` +
          "render it from a record instead"
      );
    }
  });
}

for (const f of failures) console.error(`FAIL: ${f}`);
console.log(
  `\n${exportedFigures.length} figures, ${referencedFigures.size} rendered; ` +
    `${HANDWRITTEN.length} source files scanned for hand-authored numbers`
);
if (failures.length) {
  console.error(`ARTIFACT CHECK FAILED with ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("ARTIFACT CHECK PASSED: figures presented, no result number hand-authored");
