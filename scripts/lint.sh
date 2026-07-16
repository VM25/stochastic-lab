#!/usr/bin/env bash
#
# Runs clang-format and clang-tidy over first-party sources, exactly as CI does.
#
# Usage:
#   scripts/lint.sh            # check only; nonzero exit on any finding
#   scripts/lint.sh --fix      # reformat in place, then check
#
# Both tools are pinned (see .github/workflows/ci.yml). Their output changes
# between major versions, so an unpinned tool would make "clean locally" and
# "clean in CI" mean different things.
#
#   pip install 'clang-format==20.1.7' 'clang-tidy==20.1.0'
#
# On macOS clang-tidy must be told where the SDK lives. Without -isysroot it
# fails to find <cstdint>, aborts the translation unit, and then reports nothing
# -- which looks exactly like success. That silent-pass mode is the reason this
# script exists rather than a bare clang-tidy invocation.
#
# A full pass takes several minutes and streams its output as it goes. Judge the
# result by the exit code, or by the "static analysis clean" line at the end --
# never by counting "error:" lines in a run that has not finished, which reports
# only the files reached so far.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILD_DIR=${BUILD_DIR:-build/debug}
FIX=0
[[ ${1:-} == "--fix" ]] && FIX=1

find_tool() {
    local name=$1
    if command -v "$name" > /dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    # pip --user installs land outside PATH on some systems.
    local candidate
    for candidate in "$HOME/Library/Python/"*/bin/"$name" "$HOME/.local/bin/$name"; do
        if [[ -x $candidate ]]; then
            echo "$candidate"
            return 0
        fi
    done
    echo "error: $name not found. Install with: pip install 'clang-format==20.1.7' 'clang-tidy==20.1.0'" >&2
    return 1
}

CLANG_FORMAT=$(find_tool clang-format)
CLANG_TIDY=$(find_tool clang-tidy)

# Read into an array with a while-loop rather than `mapfile`: macOS ships bash
# 3.2, which predates it, and this script must run where developers actually are.
SOURCES=()
while IFS= read -r file; do
    SOURCES+=("$file")
done < <(find include src tests benchmarks -type f \
    \( -name '*.hpp' -o -name '*.cpp' \) 2> /dev/null | sort)

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "No sources found."
    exit 0
fi

# --- Formatting -------------------------------------------------------------

if [[ $FIX -eq 1 ]]; then
    echo "==> clang-format --fix (${#SOURCES[@]} files)"
    "$CLANG_FORMAT" -i --style=file "${SOURCES[@]}"
fi

echo "==> clang-format --check (${#SOURCES[@]} files)"
"$CLANG_FORMAT" --dry-run --Werror --style=file "${SOURCES[@]}"
echo "    formatting clean"

# --- Include hygiene --------------------------------------------------------
#
# Cheap, and it catches a class of defect the compilers here cannot: a header
# leaning on a transitive include builds fine under libc++ and fails under
# libstdc++, with an error naming a struct member in a file the change never
# touched. Run before clang-tidy, which takes minutes.

echo "==> include hygiene"
python3 scripts/check_includes.py

# --- Static analysis --------------------------------------------------------

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "error: $BUILD_DIR/compile_commands.json not found. Run: cmake --preset debug" >&2
    exit 1
fi

TIDY_ARGS=(-p "$BUILD_DIR" --warnings-as-errors='*')

if [[ $(uname) == "Darwin" ]]; then
    SDK=$(xcrun --show-sdk-path)
    TIDY_ARGS+=(--extra-arg="-isysroot" --extra-arg="$SDK")
fi

# Only first-party translation units. Fetched dependencies share the compilation
# database but are not this project's code.
TU=()
while IFS= read -r file; do
    TU+=("$file")
done < <(find src -type f -name '*.cpp' | sort)

# Every translation unit is analysed, even after one fails, and the failures are
# summarised at the end.
#
# The obvious loop -- run clang-tidy and let `set -e` abort -- stops at the first
# bad file and never reports the rest. A full pass takes minutes, so that turns
# one fix-and-rerun into many, and, worse, it makes a partial result look like a
# complete one: a reader who sees five findings and fixes them has no reason to
# suspect the twenty behind them. That is precisely the failure this project
# exists to prevent, so the tooling must not commit it either.
echo "==> clang-tidy (${#TU[@]} translation units)"
FAILED=()
for tu in "${TU[@]}"; do
    echo "    $tu"
    if ! "$CLANG_TIDY" "${TIDY_ARGS[@]}" "$tu"; then
        FAILED+=("$tu")
    fi
done

if (( ${#FAILED[@]} > 0 )); then
    echo ""
    echo "==> static analysis FAILED in ${#FAILED[@]} of ${#TU[@]} translation units:"
    for tu in "${FAILED[@]}"; do
        echo "    $tu"
    done
    exit 1
fi

# The terminal marker. Grep for this rather than counting "error:" lines: the
# output streams, so a count taken mid-run reports only the files reached so far.
echo "    static analysis clean"
