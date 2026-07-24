// Static-export integrity. The export must be a complete set of files a plain file
// server can serve — one HTML page per route, the fallback pages present, no Next.js
// server-only output, and every referenced asset on disk.

import fs from "node:fs";
import path from "node:path";
import { OUT, ensureExport, htmlFiles } from "./_html.mjs";

ensureExport();

const failures = [];

const EXPECTED_ROUTES = [
  "index.html",
  "models/index.html",
  "methods/index.html",
  "experiments/index.html",
  "validation/index.html",
  "calibration/index.html",
  "architecture/index.html",
  "limitations/index.html",
  "404.html",
];
for (let i = 1; i <= 15; i++) {
  EXPECTED_ROUTES.push(`experiments/exp-${String(i).padStart(2, "0")}/index.html`);
}
for (const route of EXPECTED_ROUTES) {
  if (!fs.existsSync(path.join(OUT, route))) failures.push(`missing exported route: ${route}`);
}

// No server-only output should leak into a static export.
for (const forbidden of ["server", "standalone"]) {
  if (fs.existsSync(path.join(OUT, forbidden))) {
    failures.push(`export contains server-only directory: ${forbidden}/`);
  }
}

// Every page must have non-trivial content (a build that emits empty shells passes a
// route check but ships nothing).
for (const file of htmlFiles()) {
  const bytes = fs.statSync(file).size;
  if (bytes < 1000) failures.push(`${path.relative(OUT, file)} is suspiciously small (${bytes} bytes)`);
}

// The self-hosted fonts must be present (next/font emits them under _next/static/media).
const media = path.join(OUT, "_next", "static", "media");
if (!fs.existsSync(media) || fs.readdirSync(media).length === 0) {
  failures.push("no self-hosted font files under _next/static/media (fonts would 404)");
}

for (const f of failures) console.error(`FAIL: ${f}`);
console.log(`\n${htmlFiles().length} exported HTML pages, ${EXPECTED_ROUTES.length} routes expected`);
if (failures.length) {
  console.error(`EXPORT CHECK FAILED with ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("EXPORT CHECK PASSED: complete static export, no server-only output");
