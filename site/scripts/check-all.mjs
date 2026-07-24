// Run every site check in sequence and summarise. Each check is also runnable on its
// own; this is the one CI invokes after the build.

import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

const CHECKS = [
  ["Static export integrity", "check-export.mjs"],
  ["Broken internal links", "check-links.mjs"],
  ["Artifact presentation & stale claims", "check-artifacts.mjs"],
  ["Accessibility regressions", "check-a11y.mjs"],
];

let failed = 0;
for (const [label, script] of CHECKS) {
  console.log(`\n=== ${label} ===`);
  const result = spawnSync(process.execPath, [path.join(here, script)], { stdio: "inherit" });
  if (result.status !== 0) failed += 1;
}

console.log("\n" + "=".repeat(48));
if (failed) {
  console.error(`${failed} of ${CHECKS.length} site checks FAILED`);
  process.exit(1);
}
console.log(`All ${CHECKS.length} site checks passed.`);
