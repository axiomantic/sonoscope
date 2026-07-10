#!/usr/bin/env bash
#
# verify_clap_sdk.sh — recompute the vendored CLAP SDK tree hash and compare
# against the pin manifest. Exits 0
# iff the vendored header tree is present and its tree_sha256 matches; hard-fails
# NONZERO on ANY mismatch or missing tree. Drift is a hard fail (AGENTS.md "Pins
# are law"). This is the pin gate scripts/build_clap_midi_host.sh (P2) runs before
# compiling the C host, and that doctor/CI consume.
#
# Usage:
#   scripts/verify_clap_sdk.sh [MANIFEST_PATH]
#
# MANIFEST_PATH defaults to pins/clap_sdk.manifest.toml (resolved relative to this
# script). The optional argument lets callers (e.g. the tamper test) point
# verification at an alternate/tampered manifest copy without touching the repo
# manifest. A repo-relative [clap_sdk.vendored_tree].path is resolved against the
# repo root; an absolute path is honored as-is (so the tamper test can pin a tmp
# copy of the tree). Mirrors scripts/verify_surge_xt.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="${1:-$SCRIPT_DIR/../pins/clap_sdk.manifest.toml}"

if [ ! -f "$MANIFEST" ]; then
    echo "verify_clap_sdk: manifest not found: $MANIFEST" >&2
    exit 2
fi

# --- hashing helper (MUST match the manifest's documented algorithm) ---------

# Order-stable tree hash: hash each regular file, sort the "<hash>  <relpath>"
# lines under LC_ALL=C, then hash the concatenation. Robust to spaces via -print0.
tree_sha256() {
    local root="$1"
    ( cd "$root" && find . -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 ) \
        | shasum -a 256 | awk '{print $1}'
}

# --- manifest parsing (TOML subset -> tab-delimited records) -----------------
# Deliberately a narrow subset matched to this manifest's shape (mirrors the awk
# parser in verify_surge_xt.sh); it is NOT a general TOML parser.

parse_manifest() {
    awk '
        function unquote(s) {
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            # Quoted value: take up to the CLOSING quote, so a trailing inline
            # "# comment" is dropped while a "#" INSIDE the quotes is preserved.
            if (substr(s, 1, 1) == "\"") {
                rest = substr(s, 2)
                q = index(rest, "\"")
                if (q > 0) return substr(rest, 1, q - 1)
                return rest
            }
            # Bare value: strip a trailing inline comment (an unquoted "#").
            h = index(s, "#")
            if (h > 0) s = substr(s, 1, h - 1)
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            return s
        }
        /^[ \t]*#/  { next }
        /^[ \t]*$/  { next }
        /^\[/ {
            sec = $0
            gsub(/^[ \t]+|[ \t]+$/, "", sec)
            next
        }
        {
            eq = index($0, "=")
            if (eq == 0) next
            key = unquote(substr($0, 1, eq - 1))
            val = unquote(substr($0, eq + 1))
            if (sec == "[clap_sdk]" && key == "version")                        print "VERSION\t" val
            else if (sec == "[clap_sdk]" && key == "commit_sha")                print "COMMIT\t"  val
            else if (sec == "[clap_sdk]" && key == "source_repo")               print "REPO\t"    val
            else if (sec == "[clap_sdk.vendored_tree]" && key == "path")        print "PATH\t"    val
            else if (sec == "[clap_sdk.vendored_tree]" && key == "tree_sha256") print "TREE\t"    val
        }
    ' "$MANIFEST"
}

VERSION=""
COMMIT=""
REPO=""
TREE_PATH=""
TREE_HASH=""

while IFS="$(printf '\t')" read -r kind val; do
    case "$kind" in
        VERSION) VERSION="$val" ;;
        COMMIT)  COMMIT="$val" ;;
        REPO)    REPO="$val" ;;
        PATH)    TREE_PATH="$val" ;;
        TREE)    TREE_HASH="$val" ;;
    esac
done < <(parse_manifest)

# --- comparison --------------------------------------------------------------

fail=0
report_ok()   { echo "OK    $1"; }
report_fail() { echo "FAIL  $1" >&2 ; fail=1; }

if [ -z "$TREE_PATH" ]; then
    report_fail "vendored_tree: no path in manifest"
elif [ -z "$TREE_HASH" ]; then
    report_fail "vendored_tree: no tree_sha256 in manifest"
else
    # Resolve a repo-relative path against the repo root; honor an absolute path.
    case "$TREE_PATH" in
        /*) resolved="$TREE_PATH" ;;
        *)  resolved="$REPO_ROOT/$TREE_PATH" ;;
    esac
    if [ ! -d "$resolved" ]; then
        report_fail "vendored_tree: missing directory: $resolved"
    else
        got="$(tree_sha256 "$resolved")"
        if [ "$got" = "$TREE_HASH" ]; then
            report_ok "clap_sdk vendored_tree ($resolved)"
        else
            report_fail "clap_sdk vendored_tree hash mismatch: want $TREE_HASH got $got ($resolved)"
        fi
    fi
fi

echo "verify_clap_sdk: CLAP SDK $VERSION @ $COMMIT ($REPO)"

if [ "$fail" -ne 0 ]; then
    echo "verify_clap_sdk: FAILED (pin drift or missing vendored tree)" >&2
    exit 1
fi

echo "verify_clap_sdk: OK (vendored CLAP headers match the pin)"
exit 0
