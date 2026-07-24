// The site is a public research project, not a code showcase. This enforces the two
// halves of that:
//
// 1. No project artifact or repository file is downloadable. The export ships pages and
//    figures — no JSON, CSV, YAML, manifest, or checksum a visitor could fetch.
//
// 2. No developer-facing language in visible text. Terms that belong to how the code
//    was built — languages, tooling, version-control mechanics, file formats, internal
//    study identifiers — must not appear in what a reader sees.

import fs from "node:fs";
import path from "node:path";
import { OUT, ensureExport, htmlFiles, routeOf } from "./_html.mjs";

ensureExport();

const failures = [];

// ---- 1. No downloadable data files -------------------------------------------------
// Framework assets under _next/ are not project artifacts; everything else is
// visitor-facing. sitemap.xml and robots.txt are expected SEO files, not data.
const DATA_EXT = /\.(json|csv|ya?ml|toml|ndjson)$/i;
const ALLOWED_FILES = new Set(["sitemap.xml", "robots.txt"]);
const walk = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "_next") continue; // framework build assets
      walk(full);
    } else if (DATA_EXT.test(entry.name) && !ALLOWED_FILES.has(entry.name)) {
      failures.push(`downloadable data file exposed: /${path.relative(OUT, full)}`);
    }
  }
};
walk(OUT);
if (fs.existsSync(path.join(OUT, "artifacts"))) {
  failures.push("an /artifacts directory is exposed");
}

// ---- 2. No developer-facing language in visible text -------------------------------
// Match on the rendered text only (tags stripped). The list targets terms that are
// unambiguously about how the code was built, not the mathematics.
const FORBIDDEN = [
  /\bc\+\+\b/i,
  /\btypescript\b/i,
  /\bjavascript\b/i,
  /\bnext\.js\b/i,
  /\bnetlify\b/i,
  /\bgit\s+commit\b/i,
  /\bcommit\s+[0-9a-f]{7}\b/i,
  /\bsha-?256\b/i,
  /\bchecksum/i,
  /\bmanifest\b/i,
  /\bjson\b/i,
  /\bcsv\b/i,
  /\byaml\b/i,
  /\brepository\b/i,
  /\bcodebase\b/i,
  /\bstatic export\b/i,
  /\breproduction command\b/i,
  /\bseed policy\b/i,
  /\bcommand[- ]line\b/i,
  /\bgenerator commit\b/i,
  /\bnpm\b/i,
  /\bEXP-\d{2}\b/,
];

for (const file of htmlFiles()) {
  const html = fs.readFileSync(file, "utf8");
  // Strip scripts/styles and tags to get visible text.
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/gi, " ");
  for (const re of FORBIDDEN) {
    const m = text.match(re);
    if (m) failures.push(`${routeOf(file)}: developer-facing term in visible text: "${m[0]}"`);
  }
}

for (const f of failures) console.error(`FAIL: ${f}`);
console.log(`\nscanned ${htmlFiles().length} pages for exposed files and developer language`);
if (failures.length) {
  console.error(`PUBLIC-FACING CHECK FAILED with ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("PUBLIC-FACING CHECK PASSED: no exposed artifacts, no developer language");
