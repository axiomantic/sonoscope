"""A4: prove the Surge XT pin verifier actually catches factory-content drift.

Pins are law (design §11.3, AGENTS.md "Pins are law"): ``verify_surge_xt.sh``
recomputes the pinned VST3 / CLAP / factory-content hashes and MUST hard-fail on
any mismatch. These tests exercise the real installed Surge XT on disk, so they
carry the ``integration`` marker and skip (with an explicit reason) when the
pinned artifacts are absent — per AGENTS.md they are DESELECTED by the default
``pytest -m "not integration"`` unit run and invoked explicitly on a provisioned
machine.

Green-mirage discipline (AGENTS.md): ``test_verify_detects_tampered_factory`` is
the RED-proving test — it first shows an untampered manifest copy verifies clean,
then mutates exactly ONE factory-content hash and asserts the verifier flips to a
NONZERO exit, proving the check catches a real (single-hash) drift rather than
only ever passing. ``test_verify_passes_clean_install`` is the paired GREEN.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_surge_xt.sh"
MANIFEST = REPO_ROOT / "pins" / "surge_xt.manifest.toml"
A4_PKG_MANIFEST = REPO_ROOT / "pins" / "a4_surge_xt_pkg.manifest.toml"

# 64-char lowercase hex (a sha256 digest), used to assert the pin is resolved.
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

# The factory sub-entry whose hash the RED test mutates. Directory entry in the
# real install, so the verifier recomputes a real tree hash to compare against.
_TAMPER_ENTRY = "wavetables"
_ZERO_HASH = "0" * 64


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def _surge_installed() -> bool:
    """True iff the pinned VST3 bundle and factory-content dir exist on disk."""
    surge = _load_manifest()["surge_xt"]
    vst3 = Path(surge["vst3"]["path"])
    factory = Path(surge["factory_content"]["path"])
    return vst3.is_dir() and factory.is_dir()


requires_surge = pytest.mark.skipif(
    not _surge_installed(),
    reason="Surge XT not installed at the pinned paths; integration artifact absent (run scripts/install_surge_xt.sh)",
)


def _run_verify(manifest_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(VERIFY_SCRIPT), str(manifest_path)],
        capture_output=True,
        text=True,
    )


def _sh_tree_sha256(root: Path) -> str:
    """Tree hash of ``root`` via the EXACT pipeline verify_surge_xt.sh uses.

    Shelling out (rather than reimplementing in Python) guarantees the generated
    sandbox manifest's hashes match what the verifier recomputes, so the only
    variable the RED test isolates is the extra unpinned top-level entry.
    """
    script = (
        '( cd "$1" && find . -type f -print0 | LC_ALL=C sort -z '
        "| xargs -0 shasum -a 256 ) | shasum -a 256 | awk '{print $1}'"
    )
    proc = subprocess.run(
        ["bash", "-c", script, "_", str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _sh_file_sha256(path: Path) -> str:
    """Plain file sha256 via the same pipeline verify_surge_xt.sh uses."""
    proc = subprocess.run(
        ["bash", "-c", "shasum -a 256 \"$1\" | awk '{print $1}'", "_", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _sandbox_manifest(factory_dir: Path) -> str:
    """Build a manifest pinning a sandbox factory dir's current top-level entries.

    The VST3/CLAP sections are reused verbatim from the real manifest (so those
    checks still verify against the real install under ``requires_surge``); only
    the ``factory_content`` path + entries are redirected at ``factory_dir`` with
    freshly computed, matching hashes. A directory entry gets a tree hash; a file
    entry gets a plain file hash — mirroring the manifest's documented split.
    """
    surge = _load_manifest()["surge_xt"]
    lines = [
        "[surge_xt]",
        f'release_version = "{surge["release_version"]}"',
        'installer_artifact_sha256 = ""',
        "",
        "[surge_xt.vst3]",
        f'path = "{surge["vst3"]["path"]}"',
        f'tree_sha256 = "{surge["vst3"]["tree_sha256"]}"',
        "",
        "[surge_xt.clap]",
        f'path = "{surge["clap"]["path"]}"',
        f'tree_sha256 = "{surge["clap"]["tree_sha256"]}"',
        "",
        "[surge_xt.factory_content]",
        f'path = "{factory_dir}"',
        "",
        "[surge_xt.factory_content.entries]",
    ]
    for entry in sorted(factory_dir.iterdir()):
        digest = _sh_tree_sha256(entry) if entry.is_dir() else _sh_file_sha256(entry)
        lines.append(f'"{entry.name}" = "{digest}"')
    return "\n".join(lines) + "\n"


def _mutate_one_factory_hash(manifest_text: str, entry: str) -> str:
    """Return the manifest text with exactly ONE factory entry's hash zeroed."""
    out_lines: list[str] = []
    mutated = 0
    for line in manifest_text.splitlines():
        if line.strip().startswith(f'"{entry}"') and "=" in line:
            out_lines.append(f'"{entry}" = "{_ZERO_HASH}"')
            mutated += 1
        else:
            out_lines.append(line)
    assert mutated == 1, f"expected exactly one {entry!r} entry, mutated {mutated}"
    return "\n".join(out_lines) + "\n"


def _build_sandbox_install(tmp_path: Path):
    """A self-contained VST3/CLAP/factory tree under a parent dir named ``sand#box``.

    The ``#`` in the parent name lets the manifest exercise a ``#`` INSIDE a quoted
    path value, which the parser must preserve (never treat as an inline comment).
    Returns the created paths so the caller can pin their real hashes.
    """
    root = tmp_path / "sand#box"
    root.mkdir()
    vst3 = root / "Surge XT.vst3"
    vst3.mkdir()
    (vst3 / "plugin.bin").write_text("vst3 bytes\n")
    clap = root / "Surge XT.clap"
    clap.mkdir()
    (clap / "plugin.bin").write_text("clap bytes\n")
    factory = root / "factory"
    factory.mkdir()
    wavetables = factory / "wavetables"
    wavetables.mkdir()
    (wavetables / "a.wt").write_text("wt bytes\n")
    patches = factory / "patches.txt"
    patches.write_text("patch bytes\n")
    return vst3, clap, factory, wavetables, patches


def test_verify_parses_inline_comments_and_hashed_paths(tmp_path):
    """RED-proving (Gemini finding): the parser strips a trailing inline ``# comment``
    on a key-value line while preserving a ``#`` inside a quoted value.

    A fully sandboxed install (no real Surge XT, so this is a default-gate test) is
    pinned with freshly computed matching hashes, then given (a) a factory-entry
    line with a trailing ``# comment`` and (b) plugin/factory paths that contain a
    ``#`` inside their quotes. Under the OLD ``unquote`` the trailing comment was
    appended to the entry hash -> mismatch -> NONZERO exit; the fixed parser drops
    the comment (and keeps the interior ``#``) so verification exits 0.
    """
    vst3, clap, factory, wavetables, patches = _build_sandbox_install(tmp_path)

    manifest_lines = [
        "[surge_xt]",
        'release_version = "1.3.4"  # pinned release (trailing inline comment)',
        "",
        "[surge_xt.vst3]",
        f'path = "{vst3}"',
        f'tree_sha256 = "{_sh_tree_sha256(vst3)}"',
        "",
        "[surge_xt.clap]",
        f'path = "{clap}"',
        f'tree_sha256 = "{_sh_tree_sha256(clap)}"',
        "",
        "[surge_xt.factory_content]",
        f'path = "{factory}"',
        "",
        "[surge_xt.factory_content.entries]",
        f'"wavetables" = "{_sh_tree_sha256(wavetables)}"  # dir entry (inline comment)',
        f'"patches.txt" = "{_sh_file_sha256(patches)}"',
    ]
    manifest_path = tmp_path / "sandbox.manifest.toml"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    result = _run_verify(manifest_path)
    assert result.returncode == 0, (
        f"inline-commented manifest with '#'-in-path values must verify clean.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_installer_artifact_sha256_is_resolved_and_matches_a4_pin():
    """RED->GREEN: the surge_xt pin's ``installer_artifact_sha256`` is now the real
    .pkg digest (previously ``""``) and equals the authoritative A4 .pkg pin.

    Before fix (a) this field was an empty string, so this test would fail on the
    non-empty/64-hex assertion (RED); after filling it with the real digest it is
    a lowercase 64-hex sha256 that must match ``a4_surge_xt_pkg.manifest.toml``'s
    ``[a4_pkg].pkg_sha256`` (GREEN). Pins-are-law: the two manifests reference the
    SAME installer .pkg, so their digests must not drift apart. Pure manifest
    parse — no Surge install needed, so this runs on the default (unit) gate.
    """
    surge_digest = _load_manifest()["surge_xt"]["installer_artifact_sha256"]
    assert _HEX64.match(surge_digest), (
        "installer_artifact_sha256 must be a resolved lowercase 64-hex sha256, "
        f"got {surge_digest!r}"
    )

    with A4_PKG_MANIFEST.open("rb") as fh:
        a4_digest = tomllib.load(fh)["a4_pkg"]["pkg_sha256"]
    assert surge_digest == a4_digest, (
        "surge_xt installer_artifact_sha256 must match the authoritative A4 "
        f".pkg pin: surge={surge_digest!r} a4={a4_digest!r}"
    )


@pytest.mark.integration
@requires_surge
def test_verify_passes_clean_install():
    """GREEN: the real install matches the pinned manifest -> exit 0."""
    result = _run_verify(MANIFEST)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.integration
@requires_surge
def test_verify_detects_tampered_factory(tmp_path):
    """RED-proving: mutate ONE factory-content hash -> verifier exits NONZERO.

    Runs against a temp-copied manifest so it never needs write access to the
    repo manifest. First proves the untampered copy verifies clean (exit 0), so
    the only variable is the single mutated hash.
    """
    original = MANIFEST.read_text()

    clean_copy = tmp_path / "clean.manifest.toml"
    clean_copy.write_text(original)
    baseline = _run_verify(clean_copy)
    assert baseline.returncode == 0, (
        f"untampered copy should verify clean.\nstdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )

    tampered = _mutate_one_factory_hash(original, _TAMPER_ENTRY)
    assert tampered != original
    tampered_copy = tmp_path / "tampered.manifest.toml"
    tampered_copy.write_text(tampered)

    result = _run_verify(tampered_copy)
    assert result.returncode != 0, (
        f"tampered factory hash must force NONZERO exit.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.integration
@requires_surge
def test_verify_detects_unpinned_toplevel_entry(tmp_path):
    """RED-proving (MINOR-1): a NEW unpinned top-level factory entry -> NONZERO.

    The manifest header asserts the pinned entries ARE the complete top-level
    set, so an extra file/dir at the factory-content root is drift. Uses a temp
    sandbox factory tree + a generated manifest (never the real /Library path):
    first proves the sandbox verifies clean, then adds one unpinned sibling and
    asserts the verifier flips to a NONZERO exit. Without the completeness check
    in verify_surge_xt.sh this stays at exit 0 (RED); with it, NONZERO (GREEN).
    """
    factory = tmp_path / "factory"
    factory.mkdir()
    # A pinned directory entry (tree hash) and a pinned file entry (file hash),
    # mirroring the real manifest's mixed dir/file top-level set.
    pinned_dir = factory / "pinned_dir"
    pinned_dir.mkdir()
    (pinned_dir / "a.txt").write_text("pinned content\n")
    (factory / "pinned_file.txt").write_text("pinned file content\n")

    manifest_path = tmp_path / "sandbox.manifest.toml"
    manifest_path.write_text(_sandbox_manifest(factory))

    baseline = _run_verify(manifest_path)
    assert baseline.returncode == 0, (
        f"sandbox with only pinned entries should verify clean.\nstdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )

    # Add a NEW top-level entry that is NOT in the pinned set -> undetected drift
    # unless verify enumerates the actual top-level entries.
    (factory / "EVIL.txt").write_text("unpinned intruder\n")

    result = _run_verify(manifest_path)
    assert result.returncode != 0, (
        f"unpinned top-level entry must force NONZERO exit.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
