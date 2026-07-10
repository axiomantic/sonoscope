#!/usr/bin/env bash
#
# install_surge_xt.sh — reproducible, pinned + checksummed Surge XT acquisition
# for a CLEAN macOS machine (by design — M-install).
#
# ===========================================================================
# NON-AUTONOMOUS — NOT RUN BY THE AGENT.
# ===========================================================================
# This script runs `sudo installer -pkg ... -target /`, which requires an
# interactive admin password and performs system-level side effects. Per
# AGENTS.md ("Surge XT install needs `sudo` ... non-autonomous — hand it to
# the operator"), the agent MUST NOT run this. On the machine this pin was
# authored on, the operator had ALREADY installed Surge XT 1.3.4 manually,
# so this script was NOT executed then — only verify_surge_xt.sh and the
# RED/GREEN pin tests ran. This script is the documented recipe to reproduce the
# pinned install on a fresh machine.
#
# Upstream shape (the bug this rewrite fixes): the official Surge Synth Team
# macOS artifact is a `.dmg`, and the `installer -pkg` archive lives INSIDE it.
# The prior version curled the URL straight to a `.pkg` and ran `installer` on
# it — wrong: the download is a disk image, not a flat package. Correct flow:
#
#   download DMG -> verify DMG sha256 (pins/a4_surge_xt_pkg.manifest.toml
#   [a4_pkg.upstream_dmg].sha256, hard-fail on mismatch) -> hdiutil attach ->
#   locate inner .pkg -> verify inner .pkg sha256 via verify_a4_pkg.sh (hard-fail
#   on mismatch) -> `sudo installer -pkg <inner.pkg> -target /` -> hdiutil detach
#   (always, via a cleanup trap) -> locate + hash installed artifacts.
#
# Both the DMG and the inner .pkg are checksum-verified against the pins BEFORE
# any install runs (pins-are-law wired into the install itself).
#
# Dry-run: set SURGE_INSTALL_DRYRUN=1 to print the resolved DMG url / mount /
# verify / installer commands and exit BEFORE any download, mount, or sudo — so
# the flow is inspectable without side effects.

set -euo pipefail

# --- pinned release ----------------------------------------------------------
# Version confirmed from the installed bundle's Info.plist (CFBundleShortVersionString)
# and the `org.surge-synth-team.surge-xt` pkg receipts (`pkgutil --pkgs`).
SURGE_VERSION="1.3.4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/../pins/surge_xt.manifest.toml"
A4_PKG_MANIFEST="$SCRIPT_DIR/../pins/a4_surge_xt_pkg.manifest.toml"
VERIFY_A4_PKG="$SCRIPT_DIR/verify_a4_pkg.sh"

# --- pinned upstream DMG (url + sha256 read from the authoritative A4 pin) ----
# The A4 manifest is the single source of truth for the DMG url and its sha256
# and for the inner .pkg sha256. Read them here rather than duplicating the
# literals, so this script can never drift from the pin.
read_a4_value() {
    # $1 = section header (e.g. "[a4_pkg.upstream_dmg]"), $2 = key name.
    awk -v want_sec="$1" -v want_key="$2" '
        function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
        /^[ \t]*#/ { next }
        /^[ \t]*$/ { next }
        /^\[/ { sec = trim($0); next }
        {
            eq = index($0, "=")
            if (eq == 0) next
            key = trim(substr($0, 1, eq - 1))
            val = trim(substr($0, eq + 1))
            # Strip surrounding double quotes if present.
            if (substr(val, 1, 1) == "\"") {
                rest = substr(val, 2)
                q = index(rest, "\"")
                if (q > 0) val = substr(rest, 1, q - 1)
            }
            if (sec == want_sec && key == want_key) { print val; exit }
        }
    ' "$A4_PKG_MANIFEST"
}

DMG_URL="$(read_a4_value "[a4_pkg.upstream_dmg]" "url")"
DMG_SHA256="$(read_a4_value "[a4_pkg.upstream_dmg]" "sha256")"
PKG_SHA256="$(read_a4_value "[a4_pkg]" "pkg_sha256")"

if [ -z "$DMG_URL" ] || [ -z "$DMG_SHA256" ] || [ -z "$PKG_SHA256" ]; then
    echo "install_surge_xt: could not read DMG url/sha256 or inner .pkg sha256" >&2
    echo "  from $A4_PKG_MANIFEST — refusing to run (pins are law)." >&2
    exit 2
fi

# --- dry-run: print the resolved plan and exit before any side effect --------
# Honors SURGE_INSTALL_DRYRUN=1. Prints exactly the commands that WOULD run so
# the mount -> verify -> install -> detach flow is inspectable without touching
# the system (no download, no mount, no sudo).
if [ "${SURGE_INSTALL_DRYRUN:-0}" = "1" ]; then
    echo "install_surge_xt: DRY RUN (SURGE_INSTALL_DRYRUN=1) — no side effects"
    echo "  version:        $SURGE_VERSION"
    echo "  DMG url:        $DMG_URL"
    echo "  DMG sha256:     $DMG_SHA256"
    echo "  inner .pkg sha: $PKG_SHA256"
    echo "  would run:"
    echo "    curl -fsSL \"$DMG_URL\" -o <workdir>/surge-xt-${SURGE_VERSION}.dmg"
    echo "    shasum -a 256 <dmg>            # compare == $DMG_SHA256 (hard-fail on mismatch)"
    echo "    hdiutil attach -nobrowse -readonly <dmg>   # mount"
    echo "    <locate *.pkg inside the mounted volume>"
    echo "    bash \"$VERIFY_A4_PKG\" <inner.pkg>          # compare == $PKG_SHA256 (hard-fail)"
    echo "    sudo installer -pkg <inner.pkg> -target /   # the actual install"
    echo "    hdiutil detach <mount>        # always, via cleanup trap"
    exit 0
fi

# --- workspace + cleanup trap ------------------------------------------------
# The trap ALWAYS detaches the mount (if any) and removes the temp dir, even on
# error/interrupt, so no dangling /Volumes mount or temp DMG is left behind.
WORKDIR="$(mktemp -d)"
DMG_PATH="$WORKDIR/surge-xt-${SURGE_VERSION}.dmg"
MOUNT_POINT=""

cleanup() {
    if [ -n "$MOUNT_POINT" ] && [ -d "$MOUNT_POINT" ]; then
        hdiutil detach "$MOUNT_POINT" -quiet || \
            echo "install_surge_xt: warning — failed to detach $MOUNT_POINT" >&2
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# --- 1. download the DMG -----------------------------------------------------
echo "install_surge_xt: downloading Surge XT ${SURGE_VERSION} DMG"
curl -fsSL "$DMG_URL" -o "$DMG_PATH"

# --- 2. verify the DMG sha256 (hard-fail on mismatch; pins are law) ----------
got_dmg="$(shasum -a 256 "$DMG_PATH" | awk '{print $1}')"
if [ "$got_dmg" != "$DMG_SHA256" ]; then
    echo "install_surge_xt: DMG sha256 MISMATCH — refusing to mount/install" >&2
    echo "  want $DMG_SHA256" >&2
    echo "  got  $got_dmg" >&2
    exit 1
fi
echo "install_surge_xt: DMG sha256 OK"

# --- 3. mount the DMG + locate the inner .pkg --------------------------------
# Mount read-only and headless. Parse the mounted volume path from hdiutil's
# plist output so it is robust to volume-name changes.
echo "install_surge_xt: mounting DMG"
MOUNT_POINT="$(
    hdiutil attach "$DMG_PATH" -nobrowse -readonly -plist \
        | grep -A1 '<key>mount-point</key>' \
        | grep '<string>' \
        | sed -E 's:.*<string>(.*)</string>.*:\1:' \
        | head -n1
)"
if [ -z "$MOUNT_POINT" ] || [ ! -d "$MOUNT_POINT" ]; then
    echo "install_surge_xt: failed to determine DMG mount point" >&2
    exit 1
fi
echo "install_surge_xt: mounted at $MOUNT_POINT"

INNER_PKG="$(find "$MOUNT_POINT" -maxdepth 2 -name '*.pkg' -type f | head -n1)"
if [ -z "$INNER_PKG" ] || [ ! -f "$INNER_PKG" ]; then
    echo "install_surge_xt: no .pkg found inside mounted DMG ($MOUNT_POINT)" >&2
    exit 1
fi
echo "install_surge_xt: inner .pkg — $INNER_PKG"

# --- 4. verify the inner .pkg sha256 against the A4 pin (hard-fail) ----------
# Reuse the dedicated verifier so the compare stays identical to the pin test.
echo "install_surge_xt: verifying inner .pkg against the A4 pin"
if ! bash "$VERIFY_A4_PKG" "$INNER_PKG"; then
    echo "install_surge_xt: inner .pkg FAILED pin verification — refusing to install" >&2
    exit 1
fi
echo "install_surge_xt: inner .pkg sha256 OK"

# --- 5. install (NON-AUTONOMOUS: interactive sudo, system side effects) ------
# This is the operator step. It writes the VST3, CLAP, AU, app, and the factory
# `resources` payload to system paths. Runs on the verified inner .pkg.
sudo installer -pkg "$INNER_PKG" -target /

# --- 6. detach the mount (the trap also guarantees this on any exit path) -----
hdiutil detach "$MOUNT_POINT" -quiet
MOUNT_POINT=""

# --- locate + hash installed artifacts ---------------------------------------
# System install paths (the per-user Plug-Ins dirs are the alternative override).
VST3="/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
CLAP="/Library/Audio/Plug-Ins/CLAP/Surge XT.clap"
FACTORY="/Library/Application Support/Surge XT"

for artifact in "$VST3" "$CLAP" "$FACTORY"; do
    if [ ! -e "$artifact" ]; then
        echo "install_surge_xt: expected artifact missing after install: $artifact" >&2
        exit 1
    fi
done

echo "install_surge_xt: installed. Now verifying against the pinned manifest."
# Re-verify against the manifest so a fresh install is proven pin-clean. If the
# manifest was authored on a different Surge build, update its hashes here.
"$SCRIPT_DIR/verify_surge_xt.sh" "$MANIFEST"

echo "install_surge_xt: done (Surge XT ${SURGE_VERSION})"
