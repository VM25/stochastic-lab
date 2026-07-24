# Deployment

The site is deployed by **Netlify continuous deployment from Git**. This is the normal
and only expected workflow.

- **Project:** `diffusionworks` on Netlify → <https://diffusionworks.netlify.app>
- **Production branch:** `main`
- **Base directory:** `site`
- **Build command:** `npm run build`
- **Publish directory:** `out` (resolved relative to the base, i.e. `site/out`)
- **Node version:** pinned to `20` (in [`netlify.toml`](../netlify.toml) and
  [`site/.nvmrc`](.nvmrc), matching the CI build)

All of these are set in [`netlify.toml`](../netlify.toml) at the repository root, so the
build is reproducible without depending on anything typed into the dashboard.

## Normal workflow

Push to `main`. Netlify detects the push, runs the build, and publishes automatically.
There is nothing else to do — no local build, no manual upload.

Pull requests get **Deploy Previews** at a temporary URL so a change can be reviewed
before it is merged. Only `main` produces a production deploy; other branches do not get
their own public URLs.

## One-time connection (already configured, kept here for reference)

Connecting a Netlify project to a GitHub repository requires authorizing Netlify's GitHub
App on the repository, which is a browser step and cannot be scripted. It was done once,
in the Netlify UI, for the existing `diffusionworks` project:

1. Netlify → the **diffusionworks** project → **Project configuration → Build & deploy →
   Continuous deployment → Link repository**.
2. Choose **GitHub**, authorize the Netlify GitHub App for `VM25/stochastic-lab`.
3. Netlify reads the build settings from `netlify.toml`; confirm production branch `main`,
   base `site`, build `npm run build`, publish `out`.
4. Under **Branches and deploy contexts**, keep **Deploy only the production branch** and
   leave **Deploy Previews** enabled.

The project, its production domain, redirects, headers, and metadata are preserved by
this connection — it changes only how builds are triggered.

## Emergency fallback only

If Netlify's Git integration is unavailable and a fix must ship immediately, a build can
be published directly from a local machine:

```bash
cd site && npm ci && npm run build
cd .. && netlify deploy --prod --dir=site/out --no-build
```

This is a break-glass procedure, not the normal path. Anything shipped this way must be
followed by a push to `main` so the deployed state matches the repository and the next
automatic build reproduces it.
