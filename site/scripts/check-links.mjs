// Every internal link and asset reference in the exported site must resolve to a file
// that exists. A dead link on a results site that claims full traceability is exactly
// the kind of defect a reader cannot be expected to find by clicking.

import fs from "node:fs";
import { ensureExport, htmlFiles, routeOf, attrValues, resolveInternal } from "./_html.mjs";

ensureExport();

const failures = [];
let checked = 0;

for (const file of htmlFiles()) {
  const html = fs.readFileSync(file, "utf8");
  const refs = [...attrValues(html, "href"), ...attrValues(html, "src")];
  for (const ref of refs) {
    if (!ref.startsWith("/")) continue; // external, anchor, mailto, or relative asset
    if (ref.startsWith("//")) continue; // protocol-relative external
    if (ref.startsWith("/_next/")) continue; // framework assets, emitted by the build
    checked += 1;
    if (resolveInternal(ref) === false) {
      failures.push(`${routeOf(file)} -> ${ref} resolves to nothing`);
    }
  }
}

for (const f of failures) console.error(`FAIL: ${f}`);
console.log(`\nchecked ${checked} internal references across the export`);
if (failures.length) {
  console.error(`LINK CHECK FAILED with ${failures.length} broken reference(s)`);
  process.exit(1);
}
console.log("LINK CHECK PASSED: every internal link and asset resolves");
