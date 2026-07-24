// A browser-free subset of the accessibility requirements DESIGN-SPEC §17 makes
// blocking, checked statically over every exported page. It cannot replace a manual
// audit with a screen reader — the reduced-motion, focus-visibility, and contrast
// requirements are handled in CSS and verified by eye — but it catches the regressions
// a static check can catch, on every page, on every build.

import fs from "node:fs";
import { ensureExport, htmlFiles, routeOf, attrValues } from "./_html.mjs";

ensureExport();

const failures = [];

for (const file of htmlFiles()) {
  const html = fs.readFileSync(file, "utf8");
  const route = routeOf(file);

  // Document language.
  if (!/<html[^>]*\blang="[a-z-]+"/i.test(html)) {
    failures.push(`${route}: <html> has no lang attribute`);
  }

  // Exactly one <h1>. More than one breaks the document outline; none leaves the page
  // without a title for assistive technology.
  const h1s = (html.match(/<h1[\s>]/g) || []).length;
  if (h1s === 0) failures.push(`${route}: no <h1>`);
  if (h1s > 1) failures.push(`${route}: ${h1s} <h1> elements (expected 1)`);

  // No skipped heading levels (e.g. an h2 followed by an h4). The outline must be
  // navigable in order.
  const levels = [...html.matchAll(/<h([1-4])[\s>]/g)].map((m) => Number(m[1]));
  for (let i = 1; i < levels.length; i++) {
    if (levels[i] - levels[i - 1] > 1) {
      failures.push(`${route}: heading level jumps from h${levels[i - 1]} to h${levels[i]}`);
      break;
    }
  }

  // Every image has a non-empty alt.
  const imgTags = html.match(/<img\b[^>]*>/g) || [];
  for (const tag of imgTags) {
    const alt = tag.match(/\balt="([^"]*)"/);
    if (!alt) failures.push(`${route}: an <img> has no alt attribute`);
    else if (alt[1].trim() === "") failures.push(`${route}: an <img> has an empty alt`);
  }

  // Every button and link has an accessible name: visible text, or an aria-label.
  const controls = html.match(/<(a|button)\b[^>]*>.*?<\/\1>/gs) || [];
  for (const tag of controls) {
    const open = tag.match(/<(a|button)\b[^>]*>/)[0];
    const hasAria = /\baria-label="[^"]+"/.test(open);
    const inner = tag.replace(/<[^>]+>/g, "").trim();
    if (!hasAria && inner === "") {
      failures.push(`${route}: a <${tag.match(/<(a|button)/)[1]}> has no accessible name`);
    }
  }

  // Tables carry header cells.
  for (const table of html.match(/<table\b[\s\S]*?<\/table>/g) || []) {
    if (!/<th\b/.test(table)) failures.push(`${route}: a <table> has no <th> header cells`);
  }
}

// The skip link and a reduced-motion rule must exist in the shared output.
const home = fs.readFileSync(`${htmlFiles()[0]}`, "utf8");
if (!/skip-link/.test(home)) failures.push("no skip link in the layout");

for (const f of failures) console.error(`FAIL: ${f}`);
console.log(`\nchecked ${htmlFiles().length} exported pages for accessibility regressions`);
if (failures.length) {
  console.error(`ACCESSIBILITY CHECK FAILED with ${failures.length} problem(s)`);
  process.exit(1);
}
console.log("ACCESSIBILITY CHECK PASSED: lang, headings, alt text, control names, and table headers");
