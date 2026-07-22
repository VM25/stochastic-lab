# Provenance and privacy inspection

## Integrity

`SHA256SUMS` lists the SHA-256 checksum of every evidence file in this directory, taken
at archive time, before the files were committed. The files are verbatim copies of the
capture tooling's outputs; nothing was edited, truncated, or reformatted.

Verify with:

```
shasum -a 256 -c SHA256SUMS
```

## Privacy inspection (pre-publication)

The archive was inspected before committing for credentials, tokens, secrets, email
addresses, personalized hostnames, user filesystem paths, device serial numbers, and
window titles. Findings:

- **No credentials, tokens, secrets, or email addresses** appear in any file.
- **No user filesystem paths** appear in `waiter.log` (the orchestrator logs no paths).
- The rejected JSONs carry `host_name: "MacBookPro"` — Apple's generic model-name
  default, not a personalized hostname. The identical value is already published in the
  repository's committed experiment records (`results/EXP-01.json` `build_metadata`), so
  the archive discloses nothing new.
- The log names three processes observed as interference: `Brave Browser Helper
  (Renderer)`, `com.apple.WebKit.WebContent`, and `dw_bench_monte_carlo_scaling`. These
  are the evidentiary core — the recorded causes of the voided attempts — and redacting
  them would alter the evidence, so they are retained.

**Redaction verdict: none required.** No sanitized variant exists because no content was
removed; these files are the raw evidence.
