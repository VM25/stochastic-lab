#!/usr/bin/env python3
"""Generate known-answer test vectors for the Philox4x64-10 counter-based RNG.

External analysis, not part of the engine (ADR-002). Produces a fixed, versioned
fixture that the C++ test suite consumes.

Why a KAT fixture
-----------------
A random number generator cannot be validated by inspecting its output: any
plausible-looking stream of bits passes casual inspection, which is exactly the
failure mode this project distrusts. The only meaningful check is that the
implementation reproduces a bit-exact reference computed elsewhere.

Three independent anchors are used, and they must all agree:

  1. The published Random123 known-answer vectors (Salmon et al., "Parallel
     Random Numbers: As Easy as 1, 2, 3", SC11). These are hard-coded below and
     asserted, so the fixture is pinned to the literature rather than to this
     script.
  2. A pure-Python transcription of the algorithm from that paper.
  3. numpy.random.Philox, an independent implementation of the same generator.
     numpy increments its counter before emitting its first block, so its first
     four words correspond to counter = 1; that offset is verified rather than
     assumed.

If all three agree, the algorithm is understood correctly and the fixture is
trustworthy. The C++ implementation is then checked against it.

Usage:
    python3 python/generate_philox_kat.py > data/references/philox_kat.json
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

# Philox4x64 constants (Random123, section 4.3).
M0 = 0xD2E7470EE14C6C93
M1 = 0xCA5A826395121157
W0 = 0x9E3779B97F4A7C15  # golden ratio
W1 = 0xBB67AE8584CAA73B  # sqrt(3) - 1
MASK = (1 << 64) - 1
ROUNDS = 10


def mulhilo(a, b):
    product = a * b
    return (product >> 64) & MASK, product & MASK


def philox_round(ctr, key):
    hi0, lo0 = mulhilo(M0, ctr[0])
    hi1, lo1 = mulhilo(M1, ctr[2])
    return [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]


def philox4x64_10(ctr, key):
    ctr = list(ctr)
    key = list(key)
    for i in range(ROUNDS):
        if i > 0:
            key[0] = (key[0] + W0) & MASK
            key[1] = (key[1] + W1) & MASK
        ctr = philox_round(ctr, key)
    return ctr


# Published Random123 known-answer vectors for philox4x64 with 10 rounds.
# Format: (counter, key, expected output).
PUBLISHED_KAT = [
    (
        [0x0000000000000000] * 4,
        [0x0000000000000000] * 2,
        [0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B],
    ),
    (
        [0xFFFFFFFFFFFFFFFF] * 4,
        [0xFFFFFFFFFFFFFFFF] * 2,
        [0x87B092C3013FE90B, 0x438C3C67BE8D0224, 0x9CC7D7C69CD777B6, 0xA09CAEBF594F0BA0],
    ),
    (
        [0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89],
        [0x452821E638D01377, 0xBE5466CF34E90C6C],
        [0xA528F45403E61D95, 0x38C72DBD566E9788, 0xA5A1610E72FD18B5, 0x57BD43B5E52B7FE6],
    ),
]


def verify_against_published():
    """The transcription must reproduce the paper's vectors exactly."""
    for ctr, key, expected in PUBLISHED_KAT:
        actual = philox4x64_10(ctr, key)
        if actual != expected:
            sys.exit(
                "FATAL: transcription disagrees with the published Random123 KAT.\n"
                f"  counter  : {[hex(x) for x in ctr]}\n"
                f"  key      : {[hex(x) for x in key]}\n"
                f"  expected : {[hex(x) for x in expected]}\n"
                f"  actual   : {[hex(x) for x in actual]}"
            )


def verify_against_numpy():
    """An independent implementation must agree, or the algorithm is misread.

    Returns a note describing the check, or None when numpy is unavailable.
    """
    try:
        from numpy.random import Philox
    except ImportError:
        return None

    # numpy advances the counter before generating, so its first block is at
    # counter = 1 rather than 0. Verifying this rather than assuming it is the
    # point: a wrong offset would make the two agree on nothing and be blamed on
    # the algorithm.
    generator = Philox(key=0, counter=0)
    numpy_words = [int(x) for x in generator.random_raw(4)]
    ours = philox4x64_10([1, 0, 0, 0], [0, 0])

    if numpy_words != ours:
        sys.exit(
            "FATAL: numpy.random.Philox disagrees with the transcription.\n"
            f"  numpy : {[hex(x) for x in numpy_words]}\n"
            f"  ours  : {[hex(x) for x in ours]}"
        )

    import numpy

    return (
        f"numpy {numpy.__version__} random.Philox agrees at counter=1 "
        "(numpy advances its counter before emitting the first block)"
    )


# Additional vectors covering the counter and key words individually. The
# published set leaves the middle counter words at zero or all-ones together; a
# transposition of ctr[1] and ctr[3], or of the two key words, would survive it.
EXTRA_CASES = [
    ("counter_word_0", [1, 0, 0, 0], [0, 0]),
    ("counter_word_1", [0, 1, 0, 0], [0, 0]),
    ("counter_word_2", [0, 0, 1, 0], [0, 0]),
    ("counter_word_3", [0, 0, 0, 1], [0, 0]),
    ("key_word_0", [0, 0, 0, 0], [1, 0]),
    ("key_word_1", [0, 0, 0, 0], [0, 1]),
    ("sequential_counter_1", [1, 0, 0, 0], [0xDEADBEEF, 0]),
    ("sequential_counter_2", [2, 0, 0, 0], [0xDEADBEEF, 0]),
    ("large_path_index", [0, 1_000_000, 0, 0], [12345, 0]),
    ("high_bit_counter", [0x8000000000000000, 0, 0, 0], [0, 0]),
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def hex64(x):
    return f"{x:016x}"


def main():
    verify_against_published()
    numpy_note = verify_against_numpy()

    cases = []
    for index, (ctr, key, expected) in enumerate(PUBLISHED_KAT):
        cases.append(
            {
                "name": f"random123_published_{index}",
                "source": "Random123 kat_vectors, philox4x64 10 rounds",
                "counter": [hex64(x) for x in ctr],
                "key": [hex64(x) for x in key],
                "expected": [hex64(x) for x in expected],
            }
        )

    for name, ctr, key in EXTRA_CASES:
        cases.append(
            {
                "name": name,
                "source": "generated by python/generate_philox_kat.py",
                "counter": [hex64(x) for x in ctr],
                "key": [hex64(x) for x in key],
                "expected": [hex64(x) for x in philox4x64_10(ctr, key)],
            }
        )

    document = {
        "schema_version": 1,
        "description": (
            "Known-answer test vectors for Philox4x64-10, used to validate the DiffusionWorks "
            "counter-based random number generator bit-for-bit."
        ),
        "algorithm": {
            "name": "Philox4x64-10",
            "rounds": ROUNDS,
            "counter_bits": 256,
            "key_bits": 128,
            "reference": (
                "Salmon, Moraes, Dror, Shaw. 'Parallel Random Numbers: As Easy as 1, 2, 3'. "
                "SC11. Constants from section 4.3."
            ),
            "constants": {"M0": hex64(M0), "M1": hex64(M1), "W0": hex64(W0), "W1": hex64(W1)},
            "word_order": "counter[0..3] and key[0..1] are little-endian word indices, not bytes",
        },
        "provenance": {
            "generator": "python/generate_philox_kat.py",
            "verified_against_published_kat": True,
            "verified_against_numpy": numpy_note or "numpy unavailable; check skipped",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
        },
        "cases": cases,
    }

    json.dump(document, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
