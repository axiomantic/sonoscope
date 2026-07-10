#!/usr/bin/env bash
#
# verify_a4_pkg.sh — recompute a Surge XT installer .pkg's sha256 and compare
# against the A4 pin manifest (Task A4, design §11.3 — M-install). Exits 0 iff
# the provided .pkg is present and its sha256 matches the pinned pkg_sha256;
# hard-fails NONZERO on ANY mismatch or missing artifact. Drift is a hard fail
# (AGENTS.md "Pins are law"). Mirrors scripts/verify_clap_sdk.sh and
# scripts/verify_surge_xt.sh.
#
# Usage:
#   scripts/verify_a4_pkg.sh PKG_PATH [MANIFEST_PATH]
#
# PKG_PATH: the .pkg to verify — a provided/downloaded official installer, or the
#   Homebrew Caskroom copy (/opt/homebrew/Caskroom/surge-xt/<ver>/...pkg). The
#   .pkg is NOT downloaded or installed by this script; it only reads and hashes
#   the file it is given (the download+`sudo installer` half is the non-autonomous
#   scripts/install_surge_xt.sh).
# MANIFEST_PATH: optional, defaults to pins/a4_surge_xt_pkg.manifest.toml
#   (resolved relative to this script). The optional argument lets callers (e.g.
#   the tamper test) point verification at an alternate/tampered manifest copy
#   without touching the repo manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
    echo "verify_a4_pkg: usage: verify_a4_pkg.sh PKG_PATH [MANIFEST_PATH]" >&2
    exit 2
fi

PKG_PATH="$1"
MANIFEST="${2:-$SCRIPT_DIR/../pins/a4_surge_xt_pkg.manifest.toml}"

if [ ! -f "$MANIFEST" ]; then
    echo "verify_a4_pkg: manifest not found: $MANIFEST" >&2
    exit 2
fi
if [ ! -f "$PKG_PATH" ]; then
    echo "verify_a4_pkg: .pkg not found: $PKG_PATH" >&2
    exit 2
fi

# --- hashing helper (MUST match the manifest's documented algorithm) ---------
file_sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
}

# --- manifest parsing (TOML subset -> tab-delimited records) -----------------
# Deliberately a narrow subset matched to this manifest's shape (mirrors the awk
# parser in verify_clap_sdk.sh / verify_surge_xt.sh); NOT a general TOML parser.
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
            if (sec == "[a4_pkg]" && key == "name")                 print "NAME\t"    val
            else if (sec == "[a4_pkg]" && key == "release_version") print "VERSION\t" val
            else if (sec == "[a4_pkg]" && key == "pkg_sha256")      print "SHA\t"     val
        }
    ' "$MANIFEST"
}

NAME=""
VERSION=""
SHA=""
while IFS="$(printf '\t')" read -r kind val; do
    case "$kind" in
        NAME)    NAME="$val" ;;
        VERSION) VERSION="$val" ;;
        SHA)     SHA="$val" ;;
    esac
done < <(parse_manifest)

# --- comparison --------------------------------------------------------------
fail=0
report_ok()   { echo "OK    $1"; }
report_fail() { echo "FAIL  $1" >&2 ; fail=1; }

if [ -z "$SHA" ]; then
    report_fail "a4_pkg: no pkg_sha256 in manifest"
else
    got="$(file_sha256 "$PKG_PATH")"
    if [ "$got" = "$SHA" ]; then
        report_ok "a4_pkg ${NAME:-.pkg} sha256 ($PKG_PATH)"
    else
        report_fail "a4_pkg sha256 mismatch: want $SHA got $got ($PKG_PATH)"
    fi
fi

echo "verify_a4_pkg: Surge XT installer .pkg $VERSION ($NAME)"

if [ "$fail" -ne 0 ]; then
    echo "verify_a4_pkg: FAILED (pin drift or missing artifact)" >&2
    exit 1
fi

echo "verify_a4_pkg: OK (.pkg matches the pin)"
exit 0
