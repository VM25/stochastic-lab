#!/usr/bin/env python3
"""Fail if a header names a standard type without including its header.

Why this exists
---------------
A header that relies on a transitive include compiles until the header it was
leaning on changes. Moving `ConfidenceInterval` out of `engines/pricing_result.hpp`
and into `core/interval.hpp` removed the only path by which
`statistics/multi_seed.hpp` was seeing `<optional>`. It kept compiling on macOS,
where libc++ happens to pull `<optional>` in through another route, and broke
every compiling job on CI, where libstdc++ does not:

    multi_seed.cpp:89: error: 'struct MultiSeedSummary' has no member named 'rmse'

The error names a member, not an include, and points at a file the change never
touched -- so it reads as a corrupted build rather than a missing header. That
distance between cause and symptom is what makes this worth a dedicated check.

Nine other headers had the same latent defect, surviving only because their
transitive paths happened to hold. They are fixed; this keeps them fixed.

The check is deliberately syntactic and deliberately incomplete. It does not
parse C++ -- it looks for a literal `std::` symbol outside comments and asks
whether the header that declares it is included. That cannot catch everything
(include-what-you-use is the tool that would), but it catches this class exactly,
runs in a second, and needs no compilation database.
"""

from __future__ import annotations

import pathlib
import re
import sys

# The standard header that declares each symbol. Only entries where the mapping is
# unambiguous: `std::size_t` is declared in several headers, but <cstddef> is the
# one that promises it, so naming it is never wrong.
REQUIRED = {
    "std::optional": "optional",
    "std::nullopt": "optional",
    "std::vector": "vector",
    "std::string": "string",
    "std::string_view": "string_view",
    "std::variant": "variant",
    "std::span": "span",
    "std::array": "array",
    "std::function": "functional",
    "std::uint8_t": "cstdint",
    "std::uint32_t": "cstdint",
    "std::uint64_t": "cstdint",
    "std::int8_t": "cstdint",
    "std::int32_t": "cstdint",
    "std::int64_t": "cstdint",
    "std::size_t": "cstddef",
    "std::ptrdiff_t": "cstddef",
}

# `std::string_view` contains `std::string`, and `std::uint64_t` does not contain
# `std::uint8_t` but `std::int64_t` is a substring of `std::uint64_t`. Longest
# first, and each match consumed, so a symbol is attributed to exactly one entry.
ORDERED = sorted(REQUIRED, key=len, reverse=True)


def strip_comments(text: str) -> str:
    """Remove comments so a symbol named in prose is not counted as a use."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def missing_includes(path: pathlib.Path) -> list[tuple[str, str]]:
    text = path.read_text()
    code = strip_comments(text)

    found: list[tuple[str, str]] = []
    for symbol in ORDERED:
        if symbol not in code:
            continue
        # Consume the match so a longer symbol does not also report its prefix.
        code = code.replace(symbol, "")
        header = REQUIRED[symbol]
        if f"#include <{header}>" not in text:
            found.append((symbol, header))
    return found


def main() -> int:
    roots = [pathlib.Path("include"), pathlib.Path("src")]
    headers = sorted(h for root in roots for h in root.rglob("*.hpp"))

    if not headers:
        print("no headers found; run from the repository root", file=sys.stderr)
        return 1

    failures = 0
    for header in headers:
        for symbol, required in missing_includes(header):
            print(f"{header}: names {symbol} but does not include <{required}>")
            failures += 1

    if failures:
        print("")
        print(f"{failures} header(s) rely on a transitive include.")
        print("A header must include what it names, or it compiles only until the")
        print("header it was leaning on changes -- and the resulting error points at")
        print("the wrong file.")
        return 1

    print(f"OK: all {len(headers)} headers include what they name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
