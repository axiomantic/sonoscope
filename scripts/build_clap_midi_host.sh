#!/usr/bin/env bash
#
# build_clap_midi_host.sh — build the sonoscope v2 CLAP MIDI host.
# PIN-GATES FIRST: runs verify_clap_sdk.sh and
# REFUSES to compile on CLAP SDK drift (pins are law). Then compiles
# tools/clap_midi_host/clap_midi_host.c against the vendored, pinned CLAP 1.2.6
# headers (vendor/clap/include) with -Werror, producing build/clap_midi_host.
#
# The binary is a platform artifact rebuilt from pinned inputs — gitignored
# (.gitignore: build/), never committed.
#
# Usage:
#   scripts/build_clap_midi_host.sh
#
# Exit codes:
#   0  success (pin verified, host compiled to build/clap_midi_host)
#   1  pin drift (verify_clap_sdk.sh failed) or compile failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_ROOT/tools/clap_midi_host/clap_midi_host.c"
INC="$REPO_ROOT/vendor/clap/include"
OUT_DIR="$REPO_ROOT/build"
OUT="$OUT_DIR/clap_midi_host"

# --- pin gate: refuse to build on SDK drift (design §3 / §10) ----------------
echo "== pin gate: verify_clap_sdk.sh =="
"$SCRIPT_DIR/verify_clap_sdk.sh"

if [ ! -f "$SRC" ]; then
    echo "build_clap_midi_host: source not found: $SRC" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "== compile: clap_midi_host =="
# -Werror clean, C11, arm64 (macOS/arm64 is the runnable target; design §1).
# -ldl for dlopen/dlsym. Only the pinned headers are on the include path.
clang -std=c11 -Wall -Wextra -Werror -arch arm64 \
    -I"$INC" \
    "$SRC" \
    -ldl \
    -o "$OUT"

echo "build_clap_midi_host: OK -> $OUT"
