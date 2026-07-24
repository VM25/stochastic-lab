#!/usr/bin/env python3
"""Audit the experiment artifact set for completeness and emit results/MANIFEST.json.

Reporting and checking only, never generating (ADR-002). This script does not run an
experiment, does not fit anything, and does not decide a status: it reads what is
committed and verifies that the catalog's required artifacts are all present and
mutually consistent.

What it checks, per EXPERIMENT-CATALOG "Required Artifacts" and the final gate:

* every catalogued experiment has a committed JSON record and a CSV summary;
* every record carries the required fields, a non-empty limitations list, a stored
  configuration, the seed policy, and a reproduction command;
* every record was produced by the same generator commit, with a clean tree;
* every experiment has at least one committed figure and a methodology note (or an
  explicitly named shared one);
* every committed figure belongs to a catalogued experiment.

It then writes a manifest recording, for each experiment, the id, status, config,
record, figures, methodology, and a SHA-256 of every artifact, together with the
generator commit. The *artifact* commit -- the one that commits the manifest -- is by
construction not knowable from inside the manifest; it is whatever commit `git log`
reports for this file, which is why the generator commit is recorded explicitly here
rather than inferred later.

Usage:
    python3 python/generate_manifest.py --check          # audit only, no write
    python3 python/generate_manifest.py --out results/MANIFEST.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

# The catalogue, in one place: id -> (stored config, methodology note).
# EXP-11 and EXP-12 share a methodology note; that sharing is explicit here rather
# than left for a reader to infer from a missing file.
CATALOG: dict[str, tuple[str, str]] = {
    "EXP-01": ("configs/experiment/convergence.json", "docs/CONVERGENCE-METHODOLOGY.md"),
    "EXP-02": ("configs/experiment/convergence.json", "docs/CONVERGENCE-METHODOLOGY.md"),
    "EXP-03": ("configs/experiment/convergence.json", "docs/CONVERGENCE-METHODOLOGY.md"),
    "EXP-04": ("configs/experiment/convergence.json", "docs/CONVERGENCE-METHODOLOGY.md"),
    "EXP-05": (
        "configs/experiment/variance_reduction.json",
        "docs/VARIANCE-REDUCTION-METHODOLOGY.md",
    ),
    "EXP-06": ("configs/experiment/pde.json", "docs/PDE-CONVERGENCE-METHODOLOGY.md"),
    "EXP-07": ("configs/experiment/barrier.json", "docs/BARRIER-MONITORING-METHODOLOGY.md"),
    "EXP-08": ("configs/experiment/greeks.json", "docs/GREEKS-METHODOLOGY.md"),
    "EXP-09": (
        "configs/experiment/heston_cf_validation.json",
        "docs/HESTON-CF-VALIDATION-METHODOLOGY.md",
    ),
    "EXP-10": (
        "configs/experiment/heston_simulation.json",
        "docs/HESTON-SIMULATION-METHODOLOGY.md",
    ),
    "EXP-11": ("configs/experiment/calibration.json", "docs/CALIBRATION-METHODOLOGY.md"),
    "EXP-12": ("configs/experiment/calibration_stability.json", "docs/CALIBRATION-METHODOLOGY.md"),
    "EXP-13": ("configs/experiment/cross_method.json", "docs/CROSS-METHOD-METHODOLOGY.md"),
    "EXP-14": ("configs/experiment/coverage.json", "docs/COVERAGE-METHODOLOGY.md"),
    "EXP-15": ("configs/experiment/edge_cases.json", "docs/EDGE-CASES-METHODOLOGY.md"),
}

REQUIRED_FIELDS = [
    "id",
    "name",
    "question",
    "status",
    "interpretation",
    "limitations",
    "reproduction_command",
    "configuration",
    "results",
    "table",
    "runtime_seconds",
    "build_metadata",
]

# A stored configuration must pin the randomness for a Monte-Carlo experiment. The
# deterministic ones (PDE, characteristic function, edge cases) legitimately have no
# seed, so they are not required to carry one.
SEEDLESS = {"EXP-06", "EXP-09", "EXP-15"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_policy(record: dict) -> dict | None:
    """The seed fields the record's own stored configuration carries."""
    configuration = record.get("configuration", {})
    if not isinstance(configuration, dict):
        return None
    keys = [k for k in configuration if "seed" in k.lower()]
    return {k: configuration[k] for k in keys} or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/MANIFEST.json"))
    parser.add_argument("--check", action="store_true", help="audit only; do not write a manifest")
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--figures", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    args = parser.parse_args()

    failures: list[str] = []
    entries = []
    generator_commits: set[str] = set()
    dirty_records: list[str] = []

    committed_figures = sorted(p.name for p in args.figures.glob("*.png"))
    claimed_figures: set[str] = set()

    for experiment_id, (config_path, methodology) in CATALOG.items():
        record_path = args.results / f"{experiment_id}.json"
        csv_path = args.results / f"{experiment_id}.csv"

        if not record_path.exists():
            failures.append(f"{experiment_id}: no committed record at {record_path}")
            continue
        if not csv_path.exists():
            failures.append(f"{experiment_id}: no CSV summary at {csv_path}")
        if not pathlib.Path(config_path).exists():
            failures.append(f"{experiment_id}: stored config {config_path} is missing")
        if not pathlib.Path(methodology).exists():
            failures.append(f"{experiment_id}: methodology note {methodology} is missing")

        record = json.loads(record_path.read_text())

        for field in REQUIRED_FIELDS:
            if field not in record:
                failures.append(f"{experiment_id}: record is missing '{field}'")
        if not record.get("limitations"):
            failures.append(f"{experiment_id}: record lists no limitations")
        if not record.get("reproduction_command"):
            failures.append(f"{experiment_id}: record carries no reproduction command")
        if not record.get("configuration"):
            failures.append(f"{experiment_id}: record carries no stored configuration")

        seeds = seed_policy(record)
        if seeds is None and experiment_id not in SEEDLESS:
            failures.append(f"{experiment_id}: stored configuration pins no seed")

        build = record.get("build_metadata", {})
        commit = build.get("git_commit")
        if commit:
            generator_commits.add(commit)
        if build.get("git_dirty"):
            dirty_records.append(experiment_id)

        prefix = f"exp{experiment_id.split('-')[1]}_"
        figures = sorted(p.name for p in args.figures.glob(f"{prefix}*.png"))
        if not figures:
            failures.append(f"{experiment_id}: no committed figure")
        claimed_figures.update(figures)

        artifacts = [record_path, csv_path] + [args.figures / f for f in figures]
        entries.append(
            {
                "id": experiment_id,
                "name": record.get("name"),
                "status": record.get("status"),
                "runtime_seconds": record.get("runtime_seconds"),
                "config": config_path,
                "record_json": str(record_path),
                "record_csv": str(csv_path),
                "figures": [f"{args.figures}/{f}" for f in figures],
                "methodology": methodology,
                "seed_policy": seeds,
                "generator_commit": commit,
                "generator_tree_clean": not build.get("git_dirty", False),
                "checksums_sha256": {
                    str(p): sha256(p) for p in artifacts if p.exists()
                },
            }
        )

    orphans = sorted(set(committed_figures) - claimed_figures)
    for orphan in orphans:
        failures.append(f"figure {orphan} belongs to no catalogued experiment")

    if len(generator_commits) > 1:
        failures.append(
            "records were produced by more than one commit, so they are not one clean pass: "
            + ", ".join(sorted(c[:9] for c in generator_commits))
        )
    for experiment_id in dirty_records:
        failures.append(f"{experiment_id}: generated from a dirty tree (git_dirty is true)")

    for line in failures:
        print(f"FAIL: {line}")

    statuses: dict[str, int] = {}
    for entry in entries:
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
    print(
        f"\n{len(entries)} of {len(CATALOG)} experiments present; "
        f"statuses: {json.dumps(statuses, sort_keys=True)}"
    )
    print(f"figures: {len(committed_figures)} committed, {len(claimed_figures)} claimed")

    if failures:
        print(f"\nAUDIT FAILED with {len(failures)} problem(s)")
        return 1
    print("\nAUDIT PASSED: every catalogued experiment has record, CSV, figure, and methodology")

    if args.check:
        return 0

    generator = next(iter(generator_commits), None)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "generator_commit": generator,
        "generator_commit_short": generator[:7] if generator else None,
        "generator_tree_clean": not dirty_records,
        "artifact_commit_note": (
            "The artifact commit is the commit that adds this manifest, and so cannot be "
            "named inside it; `git log --follow results/MANIFEST.json` reports it. The "
            "generator_commit above is the source revision the records were produced from."
        ),
        "experiment_count": len(entries),
        "status_counts": statuses,
        "experiments": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
