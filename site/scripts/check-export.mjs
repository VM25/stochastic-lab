// Static-export integrity. The export must be a complete set of files a plain file
// server can serve — one page per section and study, the fallback and SEO files
// present, no server-only output, and every page non-trivial.

import fs from "node:fs";
import path from "node:path";
import { SITE, OUT, ensureExport, htmlFiles } from "./_html.mjs";

ensureExport();

const failures = [];

const EXPECTED_ROUTES = [
  "index.html",
  "models/index.html",
  "instruments/index.html",
  "methods/index.html",
  "studies/index.html",
  "calibration/index.html",
  "validation/index.html",
  "limitations/index.html",
  "404.html",
];

// The study slugs, read from the generated module so this check tracks the records.
const moduleText = fs.readFileSync(path.join(SITE, "lib", "records.generated.ts"), "utf8");
const slugs = [...moduleText.matchAll(/"slug":\s*"([a-z0-9-]+)"/g)].map((m) => m[1]);
if (slugs.length !== 15) failures.push(`expected 15 study slugs, found ${slugs.length}`);
for (const slug of slugs) EXPECTED_ROUTES.push(`studies/${slug}/index.html`);

for (const route of EXPECTED_ROUTES) {
  if (!fs.existsSync(path.join(OUT, route))) failures.push(`missing exported route: ${route}`);
}

// SEO files.
for (const seo of ["sitemap.xml", "robots.txt"]) {
  if (!fs.existsSync(path.join(OUT, seo))) failures.push(`missing ${seo}`);
}

// No server-only output should leak into a static export.
for (const forbidden of ["server", "standalone"]) {
  if (fs.existsSync(path.join(OUT, forbidden))) {
    failures.push(`export contains server-only directory: ${forbidden}/`);
  }
}

// Every page must have non-trivial content.
for (const file of htmlFiles()) {
  const bytes = fs.statSync(file).size;
  if (bytes < 1000) failures.push(`${path.relative(OUT, file)} is suspiciously small (${bytes} bytes)`);
}

// Self-hosted fonts must be present.
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
