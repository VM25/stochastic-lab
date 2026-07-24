// Small shared helpers for the check scripts. They read the exported HTML in out/
// with targeted regular expressions rather than a DOM library: the export has a known
// structure, and keeping the checks dependency-free keeps the audit surface small.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const SITE = path.resolve(here, "..");
export const OUT = path.join(SITE, "out");
export const REPO = path.resolve(SITE, "..");

export function ensureExport() {
  if (!fs.existsSync(OUT) || !fs.existsSync(path.join(OUT, "index.html"))) {
    console.error("No static export found. Run `npm run build` first.");
    process.exit(1);
  }
}

export function htmlFiles() {
  const out = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith(".html")) out.push(full);
    }
  };
  walk(OUT);
  return out;
}

// The route a given .html file serves, e.g. out/models/index.html -> /models/.
export function routeOf(file) {
  const rel = path.relative(OUT, file).replace(/index\.html$/, "").replace(/\\/g, "/");
  return "/" + rel;
}

export function attrValues(html, attr) {
  const re = new RegExp(`${attr}="([^"]*)"`, "g");
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) out.push(m[1]);
  return out;
}

// Resolve an internal absolute path (href/src beginning with /) to a file in out/.
export function resolveInternal(target) {
  const clean = target.split("#")[0].split("?")[0];
  if (!clean.startsWith("/")) return null;
  const candidates = [
    path.join(OUT, clean),
    path.join(OUT, clean, "index.html"),
    path.join(OUT, clean.replace(/\/$/, "") + ".html"),
  ];
  return candidates.find((c) => fs.existsSync(c)) ?? false;
}
