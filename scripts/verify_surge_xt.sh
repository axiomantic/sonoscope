#!/usr/bin/env bash
#
# verify_surge_xt.sh — recompute the Surge XT pin hashes and compare against the
# manifest (Task A4, design §11.3). Exits 0 iff every pinned artifact is present
# and every hash matches; hard-fails NONZERO on ANY mismatch or missing artifact.
# Drift is a hard fail (AGENTS.md "Pins are law").
#
# Usage:
#   scripts/verify_surge_xt.sh [MANIFEST_PATH]
#
# MANIFEST_PATH defaults to pins/surge_xt.manifest.toml (resolved relative to
# this script). The optional argument lets callers (e.g. the tamper test) point
# verification at an alternate/tampered manifest copy without touching the repo
# manifest.
#
# Autonomy note (A4/M3): this script is the AUTONOMOUS half of A4 — it only
# reads plugin/factory files and hashes them. The install half
# (scripts/install_surge_xt.sh) needs `sudo` and is NON-autonomous.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-$SCRIPT_DIR/../pins/surge_xt.manifest.toml}"

if [ ! -f "$MANIFEST" ]; then
    echo "verify_surge_xt: manifest not found: $MANIFEST" >&2
    exit 2
fi

# --- hashing helpers (MUST match the manifest's documented algorithm) --------

# Order-stable tree hash: hash each regular file, sort the "<hash>  <relpath>"
# lines under LC_ALL=C, then hash the concatenation. Robust to spaces via -print0.
tree_sha256() {
    local root="$1"
    ( cd "$root" && find . -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 ) \
        | shasum -a 256 | awk '{print $1}'
}

file_sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
}

# --- manifest parsing (TOML subset -> tab-delimited records) -----------------

parse_manifest() {
    awk '
        function unquote(s) {
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            # Quoted value: take up to the CLOSING quote, so a trailing inline
            # "# comment" is dropped while a "#" INSIDE the quotes is preserved
            # (mirrors the parser in scripts/fetch_qwen_model.sh).
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
            if (sec == "[surge_xt]" && key == "release_version")            print "VERSION\t"      val
            else if (sec == "[surge_xt.vst3]" && key == "path")             print "VST3_PATH\t"    val
            else if (sec == "[surge_xt.vst3]" && key == "tree_sha256")      print "VST3_TREE\t"    val
            else if (sec == "[surge_xt.clap]" && key == "path")             print "CLAP_PATH\t"    val
            else if (sec == "[surge_xt.clap]" && key == "tree_sha256")      print "CLAP_TREE\t"    val
            else if (sec == "[surge_xt.factory_content]" && key == "path")  print "FACTORY_PATH\t" val
            else if (sec == "[surge_xt.factory_content.entries]")           print "FACTORY_ENTRY\t" key "\t" val
        }
    ' "$MANIFEST"
}

VERSION=""
VST3_PATH="" ; VST3_TREE=""
CLAP_PATH="" ; CLAP_TREE=""
FACTORY_PATH=""
FAC_NAMES=()
FAC_HASHES=()

while IFS="$(printf '\t')" read -r kind a b; do
    case "$kind" in
        VERSION)       VERSION="$a" ;;
        VST3_PATH)     VST3_PATH="$a" ;;
        VST3_TREE)     VST3_TREE="$a" ;;
        CLAP_PATH)     CLAP_PATH="$a" ;;
        CLAP_TREE)     CLAP_TREE="$a" ;;
        FACTORY_PATH)  FACTORY_PATH="$a" ;;
        FACTORY_ENTRY) FAC_NAMES+=("$a") ; FAC_HASHES+=("$b") ;;
    esac
done < <(parse_manifest)

# --- comparison --------------------------------------------------------------

fail=0

report_ok()   { echo "OK    $1"; }
report_fail() { echo "FAIL  $1" >&2 ; fail=1; }

check_tree() {
    local label="$1" path="$2" want="$3"
    if [ -z "$want" ]; then
        report_fail "$label: no pinned hash in manifest"
        return
    fi
    if [ ! -d "$path" ]; then
        report_fail "$label: missing directory: $path"
        return
    fi
    local got
    got="$(tree_sha256 "$path")"
    if [ "$got" = "$want" ]; then
        report_ok "$label ($path)"
    else
        report_fail "$label hash mismatch: want $want got $got ($path)"
    fi
}

echo "verify_surge_xt: Surge XT release $VERSION"
check_tree "VST3" "$VST3_PATH" "$VST3_TREE"
check_tree "CLAP" "$CLAP_PATH" "$CLAP_TREE"

if [ -z "$FACTORY_PATH" ]; then
    report_fail "factory_content: no path in manifest"
elif [ ! -d "$FACTORY_PATH" ]; then
    report_fail "factory_content: missing directory: $FACTORY_PATH"
else
    i=0
    while [ "$i" -lt "${#FAC_NAMES[@]}" ]; do
        name="${FAC_NAMES[$i]}"
        want="${FAC_HASHES[$i]}"
        entry="$FACTORY_PATH/$name"
        if [ -d "$entry" ]; then
            got="$(tree_sha256 "$entry")"
        elif [ -f "$entry" ]; then
            got="$(file_sha256 "$entry")"
        else
            report_fail "factory[$name]: missing entry: $entry"
            i=$((i + 1))
            continue
        fi
        if [ "$got" = "$want" ]; then
            report_ok "factory[$name]"
        else
            report_fail "factory[$name] hash mismatch: want $want got $got"
        fi
        i=$((i + 1))
    done

    # Completeness (MINOR-1): the manifest header asserts the pinned entries ARE
    # the COMPLETE top-level set. Enumerate the ACTUAL top-level entries and
    # hard-fail on any that is not pinned, so a NEW file/dir added at the
    # factory-content root is caught as drift (missing entries are already caught
    # by the loop above). -mindepth/-maxdepth 1 + -print0 keeps this robust to
    # spaces/newlines in entry names.
    while IFS= read -r -d '' actual; do
        name="${actual##*/}"
        pinned=0
        j=0
        while [ "$j" -lt "${#FAC_NAMES[@]}" ]; do
            if [ "$name" = "${FAC_NAMES[$j]}" ]; then
                pinned=1
                break
            fi
            j=$((j + 1))
        done
        if [ "$pinned" -eq 0 ]; then
            report_fail "factory: unpinned top-level entry present: $name"
        fi
    done < <(find "$FACTORY_PATH" -mindepth 1 -maxdepth 1 -print0)
fi

if [ "$fail" -ne 0 ]; then
    echo "verify_surge_xt: FAILED (pin drift or missing artifact)" >&2
    exit 1
fi

echo "verify_surge_xt: OK (all pins match)"
exit 0
