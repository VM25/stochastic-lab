#!/usr/bin/env python3
"""Check the documentation set for broken links, orphaned figures, and stale scope.

Checking only, never writing (ADR-002). Three properties, each of which has failed at
least once in this project's history and none of which a reader can be expected to
verify by hand:

1. **Links resolve.** Every relative Markdown link in `docs/` and `results/README.md`
   points at a file that exists. A methodology note that references a renamed document
   is worse than no reference: it tells a reader evidence exists and then hides it.

2. **No orphaned figures, no phantom figures.** Every figure committed under
   `docs/figures/` is referenced by at least one document, and every figure a document
   references is committed. A chart nobody points at is a chart nobody reads, and it
   quietly rots when the experiment behind it is regenerated. (Before the technical
   paper was written, 16 of 21 committed figures were referenced nowhere.)

3. **The removed scope stays removed.** The benchmarking and optimization phase was
   deleted from this project; prose describing DiffusionWorks as a performance engine,
   or claiming profiling or optimization work, is a leftover. "Benchmark" meaning a
   *published reference value* -- the Fang-Oosterlee COS price -- is legitimate and is
   not flagged, so the check looks for the performance senses specifically.

4. **The API is documented.** Every public header under `include/` carries `///`
   documentation. The annotated headers *are* this project's API reference -- Doxygen is
   a permitted dependency rather than a required one, and a renderer nobody runs would
   not make the API better documented than the declarations do. Enforcing coverage keeps
   that an actual property instead of a claim, and catches a new header added without it.

Usage:
    python3 python/check_docs.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
FIGURE_MENTION = re.compile(r"\b(exp\d{2}_[a-z0-9_]+\.png)\b")

# The performance senses of the removed scope. Deliberately not the bare word
# "benchmark", which legitimately names a published reference value throughout the
# Heston documentation.
STALE_SCOPE = [
    (re.compile(r"performance engine", re.I), "the removed 'Performance Engine' subtitle"),
    (re.compile(r"google\s*benchmark", re.I), "the removed Google Benchmark dependency"),
    (re.compile(r"\bprofiling\b", re.I), "profiling, which is out of scope"),
    (re.compile(r"performance[- ]benchmark", re.I), "performance benchmarking, which is out of scope"),
    (re.compile(r"\bthroughput\b", re.I), "throughput, a performance claim"),
    (re.compile(r"\bspeedup\b", re.I), "speedup, a performance claim"),
    (re.compile(r"scaling (?:experiment|study|baseline)", re.I), "the removed scaling experiments"),
]

# Documents whose job is to record that the scope was removed, and which therefore
# must be allowed to name it. Each is a deliberate historical record, not a leftover.
SCOPE_HISTORY_ALLOWED = {
    "docs/ARCHITECTURE-DECISIONS.MD",  # ADR-025/026, both marked Withdrawn
    "docs/PHASE-13-READINESS-AUDIT.md",  # records the scope-change increment
    "docs/VARIANCE-REDUCTION-METHODOLOGY.md",  # explains why efficiency is not wall-clock
    "docs/MULTITHREADING-METHODOLOGY.md",  # states the project performs no benchmarking
    "docs/BUILD-PLAN.MD",  # records the phase renumbering
    "docs/ROADMAP.MD",
}


def documents() -> list[pathlib.Path]:
    found = sorted(pathlib.Path("docs").glob("*.md")) + sorted(pathlib.Path("docs").glob("*.MD"))
    readme = pathlib.Path("results/README.md")
    if readme.exists():
        found.append(readme)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    args = parser.parse_args()

    docs = documents()
    failures: list[str] = []
    referenced: set[str] = set()

    for doc in docs:
        text = doc.read_text()

        for _, target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.split("#")[0]
            if not path:
                continue
            beside = (doc.parent / path).resolve()
            from_root = (pathlib.Path.cwd() / path).resolve()
            if not beside.exists() and not from_root.exists():
                failures.append(f"{doc}: link to '{target}' resolves to nothing")

        referenced.update(FIGURE_MENTION.findall(text))

        if str(doc) not in SCOPE_HISTORY_ALLOWED:
            for pattern, why in STALE_SCOPE:
                for match in pattern.finditer(text):
                    line = text[: match.start()].count("\n") + 1
                    failures.append(f"{doc}:{line}: '{match.group(0)}' -- {why}")

    committed = {p.name for p in args.figures.glob("*.png")}
    for orphan in sorted(committed - referenced):
        failures.append(f"figure {orphan} is committed but referenced by no document")
    for phantom in sorted(referenced - committed):
        failures.append(f"figure {phantom} is referenced but not committed")

    headers = sorted(pathlib.Path("include").rglob("*.hpp"))
    undocumented = [h for h in headers if not re.search(r"^///", h.read_text(), re.M)]
    for header in undocumented:
        failures.append(f"{header}: public header carries no /// documentation")

    for line in failures:
        print(f"FAIL: {line}")

    print(
        f"\n{len(docs)} documents, {len(committed)} figures "
        f"({len(referenced & committed)} referenced), "
        f"{len(headers) - len(undocumented)} of {len(headers)} public headers documented"
    )
    if failures:
        print(f"DOCUMENTATION CHECK FAILED with {len(failures)} problem(s)")
        return 1
    print(
        "DOCUMENTATION CHECK PASSED: links resolve, figures are referenced, "
        "the API is documented, scope is clean"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
