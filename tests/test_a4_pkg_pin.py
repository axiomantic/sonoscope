"""A4: prove the Surge XT installer .pkg pin verifier actually catches drift.

Pins are law (design §11.3, AGENTS.md "Pins are law"): ``verify_a4_pkg.sh``
recomputes the pinned installer ``.pkg`` sha256 and MUST hard-fail on any
mismatch. Mirrors ``tests/test_clap_sdk_pin.py`` / ``tests/test_surge_pin.py``.

Green-mirage discipline (AGENTS.md): the mechanism tests do NOT need the real
~400 MB installer — a small synthetic ``.pkg`` stand-in proves the sha256
compare trips, exactly as the CLAP tamper test proves its tree-hash mechanism
with a byte flip. ``test_verify_detects_tampered_pkg`` first shows a matching
digest verifies clean (exit 0), then flips ONE byte and asserts a NONZERO exit;
``test_verify_detects_mismatched_pin`` keeps the file constant and pins a wrong
digest, proving the compare (not just file mutation) is what trips.
``test_verify_passes_real_pkg`` is the paired GREEN over the real pinned .pkg on
a provisioned machine (integration-marked; skips with an explicit reason when the
Caskroom copy is absent).
"""

import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_a4_pkg.sh"
MANIFEST = REPO_ROOT / "pins" / "a4_surge_xt_pkg.manifest.toml"

_ZERO_HASH = "0" * 64


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def _run_verify(pkg_path: Path, manifest_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(VERIFY_SCRIPT), str(pkg_path), str(manifest_path)],
        capture_output=True,
        text=True,
    )


def _sh_file_sha256(path: Path) -> str:
    """Plain file sha256 via the EXACT pipeline verify_a4_pkg.sh uses.

    Shelling out (rather than reimplementing in Python) guarantees the sandbox
    manifest's pinned digest matches what the verifier recomputes, so the only
    variable the RED test isolates is the single mutated byte.
    """
    proc = subprocess.run(
        ["bash", "-c", 'shasum -a 256 "$1" | awk \'{print $1}\'', "_", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _sandbox_manifest(pkg_name: str, pkg_sha256: str) -> str:
    """A manifest pinning ``pkg_name`` at ``pkg_sha256`` (mirrors the repo shape).

    Only the ``[a4_pkg]`` keys the verifier reads are emitted; the verifier hashes
    whatever PKG_PATH it is handed and compares against this ``pkg_sha256``.
    """
    return (
        "\n".join(
            [
                "[a4_pkg]",
                f'name = "{pkg_name}"',
                'release_version = "1.3.4"',
                f'pkg_sha256 = "{pkg_sha256}"',
            ]
        )
        + "\n"
    )


def test_verify_detects_tampered_pkg(tmp_path):
    """RED-proving: a byte-flipped .pkg -> NONZERO; matching digest -> exit 0.

    Uses a small synthetic .pkg stand-in (the real installer is not needed to
    prove the sha256 compare mechanism, mirroring the CLAP tamper test). First
    pins the synthetic file's real digest and proves verify exits 0 — so the only
    variable is the single mutated byte — then flips one byte and asserts verify
    flips to a NONZERO exit.
    """
    pkg = tmp_path / "surge-xt-macOS-1.3.4.pkg"
    pkg.write_bytes(b"synthetic pkg payload\n")

    clean_hash = _sh_file_sha256(pkg)
    manifest_path = tmp_path / "sandbox.manifest.toml"
    manifest_path.write_text(_sandbox_manifest(pkg.name, clean_hash))

    baseline = _run_verify(pkg, manifest_path)
    assert baseline.returncode == 0, (
        f"untampered synthetic .pkg should verify clean.\n"
        f"stdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )

    data = bytearray(pkg.read_bytes())
    assert data, "expected non-empty synthetic .pkg"
    data[0] ^= 0x01
    pkg.write_bytes(bytes(data))

    tampered = _run_verify(pkg, manifest_path)
    assert tampered.returncode != 0, (
        f"byte-flipped .pkg must force NONZERO exit.\n"
        f"stdout:\n{tampered.stdout}\nstderr:\n{tampered.stderr}"
    )


def test_verify_detects_mismatched_pin(tmp_path):
    """RED-proving: pin a wrong digest for an untouched file -> NONZERO.

    Complements the byte-flip test: keeps the file constant but pins a zeroed
    digest, proving the compare (not only file mutation) is what trips.
    """
    pkg = tmp_path / "surge-xt-macOS-1.3.4.pkg"
    pkg.write_bytes(b"synthetic pkg payload\n")

    manifest_path = tmp_path / "sandbox.manifest.toml"
    manifest_path.write_text(_sandbox_manifest(pkg.name, _ZERO_HASH))

    result = _run_verify(pkg, manifest_path)
    assert result.returncode != 0, (
        f"mismatched pinned digest must force NONZERO exit.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# --- integration: the real pinned .pkg on this machine (skip if absent) ------

#: A4-confirmed Homebrew Caskroom location of the pinned Surge XT installer .pkg.
_CASKROOM_PKG = Path(
    "/opt/homebrew/Caskroom/surge-xt/1.3.4/surge-xt-macOS-1.3.4.pkg"
)


requires_real_pkg = pytest.mark.skipif(
    not _CASKROOM_PKG.is_file(),
    reason=(
        f"Surge XT installer .pkg absent from {_CASKROOM_PKG} "
        "(integration artifact; provisioned via scripts/install_surge_xt.sh or "
        "`brew install --cask surge-xt`)"
    ),
)


@pytest.mark.integration
@requires_real_pkg
def test_verify_passes_real_pkg():
    """GREEN: the real pinned .pkg matches the repo manifest -> exit 0."""
    result = _run_verify(_CASKROOM_PKG, MANIFEST)
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
