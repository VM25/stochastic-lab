#!/usr/bin/env python3
"""Write each experiment's CSV summary from its own committed JSON record.

Reporting only, not analysis (ADR-002). This reads the `table` a record already
carries -- the summary the C++ engine built and published -- and serialises it. It
computes nothing.

Why this exists rather than `diffusionworks experiment --format csv`: that command
*re-runs the experiment*. A second run produces a second set of runtime measurements,
so the CSV beside a record described a different execution than the record did. The
committed pairs disagreed in exactly that way -- EXP-08 reported a mean runtime of
0.0242637 in its JSON and 0.00626071 in its CSV -- which makes the CSV not a summary
of the record it sits next to. Deriving it from the record makes the two provably the
same run, and makes the manifest's checksums cover a consistent pair.

The serialisation is byte-identical to the CLI's for every experiment whose table has
no runtime column, which is how the equivalence was checked.

Usage:
    python3 python/derive_csv.py results/EXP-07.json
    python3 python/derive_csv.py results/EXP-*.json          # in place, beside each record
    python3 python/derive_csv.py staging/*.json --outdir results
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys


def derive(record: dict) -> str:
    table = record.get("table")
    if not isinstance(table, dict) or "headers" not in table or "rows" not in table:
        raise ValueError(f"{record.get('id', '?')}: record carries no summary table")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table["headers"])
    for row in table["rows"]:
        writer.writerow(row)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=pathlib.Path, help="experiment JSON records")
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help="where to write the CSVs (default: beside each record)",
    )
    args = parser.parse_args()

    failed = False
    for path in args.records:
        record = json.loads(path.read_text())
        try:
            text = derive(record)
        except ValueError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            failed = True
            continue
        outdir = args.outdir or path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{path.stem}.csv"
        out.write_text(text)
        rows = len(record["table"]["rows"])
        print(f"wrote {out} ({rows} rows)")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
