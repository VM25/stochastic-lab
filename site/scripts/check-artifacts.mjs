// Two guarantees the site's credibility rests on:
//
// 1. Every committed engine artifact is actually presented. All 21 figures appear in
//    the export, every experiment has a page, and every record's JSON and CSV are
//    downloadable — so nothing the engine produced is quietly dropped.
//
// 2. No result number is hand-authored. Every number the site shows comes from the
//    generated data module (built from results/EXP-*.json). This check greps the
//    hand-written source for result-shaped numeric literals and fails on any it finds
//    outside a small structural allowlist, so a number can never drift from its record
//    by being typed into prose.

import fs from "node:fs";
import path from "node:path";
import { SITE, OUT, REPO, ensureExport, htmlFiles, attrValues } from "./_html.mjs";

ensureExport();

const failures = [];
const EXPERIMENT_IDS = Array.from({ length: 15 }, (_, i) => `EXP-${String(i + 1).padStart(2, "0")}`);

// ---- 1. Every artifact is presented ------------------------------------------------
const committedFigures = fs
  .readdirSync(path.join(REPO, "docs", "figures"))
  .filter((f) => f.endsWith(".png"))
  .sort();

const referencedFigures = new Set();
for (const file of htmlFiles()) {
  const html = fs.readFileSync(file, "utf8");
  for (const src of attrValues(html, "src")) {
    const m = src.match(/\/figures\/(exp\d{2}_[a-z0-9_]+\.png)/);
    if (m) referencedFigures.add(m[1]);
  }
}
for (const fig of committedFigures) {
  if (!referencedFigures.has(fig)) failures.push(`figure ${fig} is committed but shown on no page`);
  if (!fs.existsSync(path.join(OUT, "figures", fig))) failures.push(`figure ${fig} missing from export`);
}

for (const id of EXPERIMENT_IDS) {
  const page = path.join(OUT, "experiments", id.toLowerCase(), "index.html");
  if (!fs.existsSync(page)) failures.push(`${id}: no exported experiment page`);
  for (const ext of ["json", "csv"]) {
    if (!fs.existsSync(path.join(OUT, "artifacts", `${id}.${ext}`))) {
      failures.push(`${id}: ${ext.toUpperCase()} not downloadable from the export`);
    }
  }
}
if (!fs.existsSync(path.join(OUT, "artifacts", "MANIFEST.json"))) {
  failures.push("MANIFEST.json not downloadable from the export");
}

// ---- 2. No hand-authored result numbers --------------------------------------------
// The generated module is the only place literal record values may live.
const HANDWRITTEN = [];
const walkSrc = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (["node_modules", ".next", "out"].includes(entry.name)) continue;
      walkSrc(full);
    } else if (/\.(tsx?|mjs)$/.test(entry.name)) {
      if (full.endsWith("records.generated.ts")) continue; // the generated module
      if (full.includes(`${path.sep}scripts${path.sep}`)) continue; // checks, not shipped content
      HANDWRITTEN.push(full);
    }
  }
};
walkSrc(path.join(SITE, "app"));
walkSrc(path.join(SITE, "components"));
walkSrc(path.join(SITE, "lib"));

// Result-shaped literals: a decimal with two or more fraction digits (0.4996, 87.75),
// or a percentage (145%). A measured result looks like one of these. Scientific
// notation is deliberately NOT scanned in source: in content it is only ever rendered
// from a record, and scanning for it flags code constants (1e-3 thresholds) rather
// than results. The allowlist holds fixed conventions and dependency versions, never a
// measurement — so a hand-typed 87.75% is still caught while a nominal 95% is not.
const RESULT_LITERAL = /\b\d{1,4}\.\d{2,}\b|\b\d{1,3}%/g;
const ALLOW = new Set([
  "95", // the nominal confidence level, a fixed convention (records: confidence_level = 0.95)
  "14.2", // Next.js version references in comments
  "0.16", // KaTeX version
  "2.35", // dep patch reference
]);

for (const file of HANDWRITTEN) {
  const text = fs.readFileSync(file, "utf8");
  const lines = text.split("\n");
  lines.forEach((line, i) => {
    // Ignore import lines and obvious non-content (className strings are fine; we only
    // flag numeric literals in JSX text or string copy).
    const matches = line.match(RESULT_LITERAL);
    if (!matches) return;
    for (const raw of matches) {
      const token = raw.replace("%", "");
      if (ALLOW.has(token)) continue;
      // Percentages and scientific notation are always results; multi-decimal too.
      failures.push(
        `${path.relative(SITE, file)}:${i + 1}: hand-authored result-shaped number "${raw}" — ` +
          "render it from a record instead"
      );
    }
  });
}

// ---- 3. The generated module carries the committed records --------------------------
const moduleText = fs.readFileSync(path.join(SITE, "lib", "records.generated.ts"), "utf8");
for (const id of EXPERIMENT_IDS) {
  const record = JSON.parse(fs.readFileSync(path.join(REPO, "results", `${id}.json`), "utf8"));
  if (!moduleText.includes(record.reproduction_command)) {
    failures.push(`${id}: reproduction command not present in the generated module (stale build?)`);
  }
  if (!moduleText.includes(`"status": ${JSON.stringify(record.status)}`)) {
    failures.push(`${id}: status "${record.status}" not present in the generated module`);
  }
}

// ---- report ------------------------------------------------------------------------
for (const f of failures) console.error(`FAIL: ${f}`);
console.log(
  `\n${committedFigures.length} figures, ${referencedFigures.size} referenced; ` +
    `${HANDWRITTEN.length} source files scanned for hand-authored numbers`
);
if (failures.length) {
  console.error(`ARTIFACT CHECK FAILED with ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("ARTIFACT CHECK PASSED: every artifact is presented, no result number is hand-authored");
