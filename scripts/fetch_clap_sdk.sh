#!/usr/bin/env bash
#
# fetch_clap_sdk.sh — reproducible re-vendor of the pinned CLAP SDK headers into
# vendor/clap/include. For a clean-machine
# re-vendor: clone free-audio/clap at the pinned tag, HARD-VERIFY the checked-out
# commit equals the pinned commit_sha, copy include/clap into the repo, then run
# scripts/verify_clap_sdk.sh so the tree hash is confirmed. Mirrors
# scripts/fetch_qwen_model.sh (pins are law; no sha/commit is ever fabricated).
#
# Unlike the qwen fetch, the CLAP pins are FULLY resolved (public, header-only,
# small) so this script actually performs the fetch. It still REFUSES (exit 2) if
# the manifest's commit_sha/version are unresolved placeholders.
#
# Usage:
#   scripts/fetch_clap_sdk.sh [MANIFEST]
#
# Exit codes:
#   0  success (cloned at pinned commit, vendored, tree hash verified)
#   1  drift: checked-out commit != pinned commit, or verify_clap_sdk.sh fails
#   2  refusal: unresolved placeholder / usage error / manifest absent / no git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="${1:-$SCRIPT_DIR/../pins/clap_sdk.manifest.toml}"
TODO_SENTINEL="TODO(pin)"

if [ ! -f "$MANIFEST" ]; then
    echo "fetch_clap_sdk: manifest not found: $MANIFEST" >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    echo "fetch_clap_sdk: git not found on PATH" >&2
    exit 2
fi

# --- manifest parsing (same subset as verify_clap_sdk.sh) --------------------
parse_manifest() {
    awk '
        function unquote(s) {
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            if (substr(s, 1, 1) == "\"") {
                rest = substr(s, 2)
                q = index(rest, "\"")
                if (q > 0) return substr(rest, 1, q - 1)
                return rest
            }
            h = index(s, "#")
            if (h > 0) s = substr(s, 1, h - 1)
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            return s
        }
        /^[ \t]*#/  { next }
        /^[ \t]*$/  { next }
        /^\[/ { sec = $0; gsub(/^[ \t]+|[ \t]+$/, "", sec); next }
        {
            eq = index($0, "=")
            if (eq == 0) next
            key = unquote(substr($0, 1, eq - 1))
            val = unquote(substr($0, eq + 1))
            if (sec == "[clap_sdk]" && key == "version")                 print "VERSION\t" val
            else if (sec == "[clap_sdk]" && key == "commit_sha")         print "COMMIT\t"  val
            else if (sec == "[clap_sdk]" && key == "source_repo")        print "REPO\t"    val
            else if (sec == "[clap_sdk.vendored_tree]" && key == "path") print "PATH\t"    val
        }
    ' "$MANIFEST"
}

VERSION="" ; COMMIT="" ; REPO="" ; TREE_PATH=""
while IFS="$(printf '\t')" read -r kind val; do
    case "$kind" in
        VERSION) VERSION="$val" ;;
        COMMIT)  COMMIT="$val" ;;
        REPO)    REPO="$val" ;;
        PATH)    TREE_PATH="$val" ;;
    esac
done < <(parse_manifest)

# Refuse while any pin is unresolved (mirror install_surge_xt.sh / fetch_qwen_model.sh).
for name in VERSION COMMIT REPO TREE_PATH; do
    val="${!name}"
    if [ -z "$val" ] || [ "$val" = "$TODO_SENTINEL" ]; then
        echo "fetch_clap_sdk: refusing — [$name] unresolved in $MANIFEST" >&2
        echo "  Pins are law: no commit/version is fabricated (project policy)." >&2
        exit 2
    fi
done

case "$TREE_PATH" in
    /*) DEST="$TREE_PATH" ;;
    *)  DEST="$REPO_ROOT/$TREE_PATH" ;;
esac

# Clone into a fresh temp checkout at the pinned tag, then confirm the commit.
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "fetch_clap_sdk: cloning $REPO @ $VERSION"
if ! git clone --quiet --depth 1 --branch "$VERSION" "$REPO" "$WORK/clap-src"; then
    echo "fetch_clap_sdk: git clone failed" >&2
    exit 1
fi

got_commit="$(git -C "$WORK/clap-src" rev-parse HEAD)"
if [ "$got_commit" != "$COMMIT" ]; then
    echo "fetch_clap_sdk: commit drift — tag $VERSION is $got_commit, pinned $COMMIT" >&2
    exit 1
fi

SRC="$WORK/clap-src/include/clap"
if [ ! -f "$SRC/clap.h" ]; then
    echo "fetch_clap_sdk: expected header include/clap/clap.h missing in clone" >&2
    exit 1
fi

# Re-vendor: replace the include/clap subtree with the freshly cloned headers.
mkdir -p "$DEST/clap"
rm -rf "$DEST/clap"
mkdir -p "$DEST/clap"
cp -R "$SRC/." "$DEST/clap/"

echo "fetch_clap_sdk: vendored include/clap -> $DEST/clap"

# Confirm the vendored tree hash matches the pin (hard-fail on drift).
if ! bash "$SCRIPT_DIR/verify_clap_sdk.sh" "$MANIFEST"; then
    echo "fetch_clap_sdk: post-vendor verify FAILED" >&2
    exit 1
fi

echo "fetch_clap_sdk: done — CLAP SDK $VERSION @ $COMMIT vendored + verified"
exit 0
