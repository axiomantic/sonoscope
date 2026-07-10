#!/usr/bin/env bash
#
# fetch_qwen_model.sh — reproducible, pinned + checksummed acquisition of the
# Qwen2-Audio-7B-Instruct weights the optional `sonoscope[perception]` extra runs
# against (Task A5, reconciled by B3 — design §11.2 / §11.3 / §18.1 — M-install).
#
# ===========================================================================
# B3 RESOLVED (2026-07-05, verdict PASS). Runtime
# pivoted Nexa-SDK-GGUF -> transformers reference runtime. Acquisition is a
# huggingface_hub snapshot_download of the pinned HF repo @ exact `revision`
# (pins/qwen_model.manifest.toml), then a sha256-verify of each pinned file.
# The `download` mode still HARD-REFUSES (exit 2) while hf_repo/revision or any
# sha is unresolved. No URL/sha is ever fabricated (project policy: pins are law).
# The weights are ~16 GB — they live in cache_dir, OUTSIDE the repo.
# ===========================================================================
#
# Modes:
#   fetch_qwen_model.sh [download] [MANIFEST]
#       snapshot_download the pinned repo@revision into the cache dir, then verify
#       each pinned file's sha256, hard-failing NONZERO on mismatch. Refuses
#       (exit 2) while hf_repo/revision or any sha is an unresolved placeholder.
#
#   fetch_qwen_model.sh verify MANIFEST LOCAL_DIR
#       Standalone verification of ALREADY-PRESENT files in LOCAL_DIR against the
#       manifest shas — no network, no cache. This is the isolated sha256-compare
#       branch; tests/test_model_pin.py::test_fetch_detects_hash_mismatch drives
#       it against a tampered manifest to prove the compare genuinely catches
#       drift (RED-proving, green-mirage discipline). Exit 0 iff every listed
#       file is present and matches; NONZERO on any mismatch/missing file; exit 2
#       if a listed entry still has an empty (placeholder) sha (nothing to
#       verify against).
#
# Exit codes:
#   0  success (download+verify OK, or standalone verify all-match)
#   1  sha256 MISMATCH or missing artifact (hard drift fail)
#   2  refusal: unresolved TODO(pin) placeholder / usage error / manifest absent
#
# Drift is a hard fail (design §11.3, AGENTS.md "Pins are law").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MANIFEST="$SCRIPT_DIR/../pins/qwen_model.manifest.toml"
TODO_SENTINEL="TODO(pin)"

# --- hashing helper (MUST match the manifest's documented algorithm) ---------
file_sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
}

# --- single shared sha256 compare --------------------------------------------
# sha_matches FILE EXPECTED -> exit 0 iff sha256(FILE) equals EXPECTED, else 1.
# BOTH do_verify (the RED-tested path) and do_download share this one compare so
# the verified branch and the production download branch can never drift apart.
sha_matches() {
    local file="$1" expected="$2" got
    got="$(file_sha256 "$file")"
    [ "$got" = "$expected" ]
}

# --- manifest parsing (TOML subset -> tab-delimited records) -----------------
#
# Emits one record per (role, field): "<role>\t<field>\t<value>" for every key
# under a [qwen_model.files.<role>] table, plus "CACHE\t<dir>" for the cache dir.
# Deliberately a narrow subset matched to this manifest's shape (mirrors the
# awk parser in verify_surge_xt.sh); it is NOT a general TOML parser.
parse_manifest() {
    awk '
        function unquote(s) {
            gsub(/^[ \t]+|[ \t]+$/, "", s)
            # strip an inline comment that is OUTSIDE a quoted string
            if (substr(s, 1, 1) == "\"") {
                # quoted value: take up to the closing quote
                rest = substr(s, 2)
                q = index(rest, "\"")
                if (q > 0) return substr(rest, 1, q - 1)
                return rest
            }
            # bare value: drop trailing comment
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
            role = ""
            if (sec ~ /^\[qwen_model\.files\./) {
                role = sec
                sub(/^\[qwen_model\.files\./, "", role)
                sub(/\].*$/, "", role)
            }
            next
        }
        {
            eq = index($0, "=")
            if (eq == 0) next
            key = substr($0, 1, eq - 1)
            gsub(/^[ \t]+|[ \t]+$/, "", key)
            val = unquote(substr($0, eq + 1))
            if (sec == "[qwen_model]" && key == "cache_dir") { print "CACHE\t" val; next }
            if (sec == "[qwen_model]" && key == "hf_repo")   { print "REPO\t" val; next }
            if (sec == "[qwen_model]" && key == "revision")  { print "REV\t" val; next }
            if (role != "") print role "\t" key "\t" val
        }
    ' "$1"
}

# Read the parsed manifest into parallel role arrays.
ROLES=()
declare -a R_FILE R_URL R_SHA
CACHE_DIR=""
HF_REPO=""
REVISION=""

load_manifest() {
    local manifest="$1"
    if [ ! -f "$manifest" ]; then
        echo "fetch_qwen_model: manifest not found: $manifest" >&2
        exit 2
    fi
    local kind a b
    while IFS="$(printf '\t')" read -r kind a b; do
        case "$kind" in
            CACHE) CACHE_DIR="$a" ;;
            REPO)  HF_REPO="$a" ;;
            REV)   REVISION="$a" ;;
            *)
                # kind == role, a == field, b == value
                local idx="" i=0
                while [ "$i" -lt "${#ROLES[@]}" ]; do
                    if [ "${ROLES[$i]}" = "$kind" ]; then idx="$i"; break; fi
                    i=$((i + 1))
                done
                if [ -z "$idx" ]; then
                    idx="${#ROLES[@]}"
                    ROLES+=("$kind")
                    R_FILE[$idx]=""; R_URL[$idx]=""; R_SHA[$idx]=""
                fi
                case "$a" in
                    filename)     R_FILE[$idx]="$b" ;;
                    url)          R_URL[$idx]="$b" ;;
                    model_sha256) R_SHA[$idx]="$b" ;;
                esac
                ;;
        esac
    done < <(parse_manifest "$manifest")

    if [ "${#ROLES[@]}" -eq 0 ]; then
        echo "fetch_qwen_model: no [qwen_model.files.*] entries in manifest: $manifest" >&2
        exit 2
    fi
}

# --- standalone verify mode --------------------------------------------------
# Verify already-present files in LOCAL_DIR against the manifest shas. This is
# the isolated sha256-compare branch the RED-proving test exercises.
do_verify() {
    local manifest="$1" local_dir="$2"
    load_manifest "$manifest"

    local fail=0 i=0
    while [ "$i" -lt "${#ROLES[@]}" ]; do
        local role="${ROLES[$i]}" fname="${R_FILE[$i]}" want="${R_SHA[$i]}"
        if [ -z "$fname" ]; then
            echo "fetch_qwen_model: [$role] has no filename in manifest" >&2
            exit 2
        fi
        if [ -z "$want" ]; then
            echo "fetch_qwen_model: [$role] model_sha256 is empty (placeholder) — nothing to verify" >&2
            exit 2
        fi
        local path="$local_dir/$fname"
        if [ ! -f "$path" ]; then
            echo "FAIL  [$role] missing file: $path" >&2
            fail=1
            i=$((i + 1))
            continue
        fi
        if sha_matches "$path" "$want"; then
            echo "OK    [$role] $fname"
        else
            echo "FAIL  [$role] sha256 mismatch: want $want got $(file_sha256 "$path") ($fname)" >&2
            fail=1
        fi
        i=$((i + 1))
    done

    if [ "$fail" -ne 0 ]; then
        echo "fetch_qwen_model: VERIFY FAILED (pin drift or missing artifact)" >&2
        exit 1
    fi
    echo "fetch_qwen_model: verify OK (all model pins match)"
    exit 0
}

# --- download mode -----------------------------------------------------------
# B3 pivot: the model is the transformers-runtime HF repo Qwen2-Audio-7B-Instruct,
# pinned by exact `revision`. Acquisition is a huggingface_hub snapshot_download of
# that repo@revision into cache_dir (NOT per-file curl of GGUF URLs), then a
# sha256-verify of every pinned file via the SAME compare do_verify uses. No URL or
# sha is ever fabricated (pins are law); refuses while hf_repo/revision/shas are
# unresolved.
do_download() {
    local manifest="$1"
    load_manifest "$manifest"

    # Refuse while the repo/revision or any sha is unresolved (mirror install_surge_xt.sh).
    if [ -z "$HF_REPO" ] || [ "$HF_REPO" = "$TODO_SENTINEL" ] \
       || [ -z "$REVISION" ] || [ "$REVISION" = "$TODO_SENTINEL" ]; then
        echo "fetch_qwen_model: refusing — hf_repo/revision unresolved in $manifest" >&2
        echo "  These are intentionally not fabricated (project policy: pins are law)." >&2
        exit 2
    fi
    local i=0
    while [ "$i" -lt "${#ROLES[@]}" ]; do
        local role="${ROLES[$i]}" sha="${R_SHA[$i]}"
        if [ -z "$sha" ] || [ "$sha" = "$TODO_SENTINEL" ]; then
            echo "fetch_qwen_model: refusing — [$role] model_sha256 unresolved (placeholder)" >&2
            exit 2
        fi
        i=$((i + 1))
    done

    # Expand a leading ~ in the cache dir.
    case "$CACHE_DIR" in
        "~"/*) CACHE_DIR="$HOME/${CACHE_DIR#\~/}" ;;
        "~")   CACHE_DIR="$HOME" ;;
    esac
    if [ -z "$CACHE_DIR" ]; then
        echo "fetch_qwen_model: no cache_dir in manifest" >&2
        exit 2
    fi
    mkdir -p "$CACHE_DIR"

    echo "fetch_qwen_model: snapshot_download $HF_REPO @ $REVISION -> $CACHE_DIR"
    # snapshot_download places real files (local_dir) so the sha-verify below can
    # hash them directly. A download failure hard-fails NONZERO (no partial keep).
    # Finding 2 (cycle 2): the Python env is uv-managed (AGENTS.md); system
    # ``python3`` may lack ``huggingface_hub``. Run the download under ``uv run``
    # so it resolves the project interpreter + its pinned deps.
    if ! SONOSCOPE_REPO="$HF_REPO" SONOSCOPE_REV="$REVISION" SONOSCOPE_DEST="$CACHE_DIR" \
         uv run python -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["SONOSCOPE_REPO"], revision=os.environ["SONOSCOPE_REV"],
                  local_dir=os.environ["SONOSCOPE_DEST"])
'; then
        echo "fetch_qwen_model: snapshot_download failed" >&2
        exit 1
    fi

    # Verify every pinned file's sha256 (shared compare; hard-fail on drift).
    local fail=0
    i=0
    while [ "$i" -lt "${#ROLES[@]}" ]; do
        local role="${ROLES[$i]}" fname="${R_FILE[$i]}" want="${R_SHA[$i]}"
        local dest="$CACHE_DIR/$fname"
        if [ ! -f "$dest" ]; then
            echo "FAIL  [$role] missing after download: $dest" >&2; fail=1; i=$((i + 1)); continue
        fi
        if sha_matches "$dest" "$want"; then
            echo "OK    [$role] $fname"
        else
            echo "FAIL  [$role] sha256 mismatch: want $want got $(file_sha256 "$dest")" >&2; fail=1
        fi
        i=$((i + 1))
    done
    if [ "$fail" -ne 0 ]; then
        echo "fetch_qwen_model: DOWNLOAD VERIFY FAILED (pin drift)" >&2
        exit 1
    fi
    echo "fetch_qwen_model: done — repo@revision downloaded + all pins verified into $CACHE_DIR"
    exit 0
}

# --- dispatch ----------------------------------------------------------------
mode="${1:-download}"
case "$mode" in
    verify)
        if [ "$#" -ne 3 ]; then
            echo "usage: fetch_qwen_model.sh verify MANIFEST LOCAL_DIR" >&2
            exit 2
        fi
        do_verify "$2" "$3"
        ;;
    download)
        do_download "${2:-$DEFAULT_MANIFEST}"
        ;;
    *)
        # Treat a lone path argument as a manifest for download mode.
        if [ -f "$mode" ]; then
            do_download "$mode"
        else
            echo "usage: fetch_qwen_model.sh [download [MANIFEST]] | verify MANIFEST LOCAL_DIR" >&2
            exit 2
        fi
        ;;
esac
