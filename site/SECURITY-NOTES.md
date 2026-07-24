# Dependency audit notes

`npm audit` reports advisories against `next` and its transitive `postcss`. They are
retained deliberately, and this records why.

This site is a **static export** (`output: "export"` in `next.config.mjs`). The build
emits plain HTML, CSS, and JS into `out/`, which Netlify serves as files. There is no
Next.js server in production: no Server Actions, no Image Optimization API (images are
`unoptimized`), no rewrites, no middleware, no React Server Component request handling.

Every retained advisory targets one of those server or build surfaces:

- **Next.js Image Optimizer `remotePatterns` DoS** — the optimizer is disabled.
- **RSC HTTP request deserialization DoS**, **HTTP request smuggling in rewrites**,
  **dev-server origin verification**, **Server Actions DoS**, **cache-key confusion** —
  all require the Next.js server, which the static export does not ship.
- **PostCSS XSS / file read** — PostCSS runs only at build time over CSS authored in
  this repository, not attacker-controlled input.

The advisories' remediation is `next@16`, a breaking major (App Router and build-API
changes). The project's fixed stack specifies Next.js, and upgrading a major to silence
findings the static-export threat model already excludes would add risk, not remove it.
The project pins the **latest patched release within the major it uses** (`next@14.2.x`)
and re-evaluates on each dependency bump.
