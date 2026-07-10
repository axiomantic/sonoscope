"""Prove the CLAP SDK pin verifier actually catches vendored-header drift.

Pins are law (AGENTS.md "Pins are law"):
``verify_clap_sdk.sh`` recomputes the vendored ``vendor/clap/include`` tree hash
and MUST hard-fail on any mismatch. Because the CLAP headers are committed into
the repo (header-only, small), these tests need NO external artifact — they run
in the default ``pytest -m "not integration"`` gate (no marker).

Green-mirage discipline (AGENTS.md): ``test_verify_detects_tampered_header`` is
the RED-proving test — it first shows a tmp copy of the vendored tree verifies
clean against a matching manifest, then mutates exactly ONE byte of ONE header
and asserts the verifier flips to a NONZERO exit, proving the tree hash genuinely
catches drift rather than only ever passing. ``test_verify_passes_clean`` is the
paired GREEN over the real committed headers.
"""

import shutil
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_clap_sdk.sh"
MANIFEST = REPO_ROOT / "pins" / "clap_sdk.manifest.toml"
VENDORED_INCLUDE = REPO_ROOT / "vendor" / "clap" / "include"

# The single header the RED test mutates one byte of. entry.h is a stable,
# always-present entry point in the CLAP SDK (by design, alongside clap.h).
_TAMPER_HEADER = Path("clap") / "entry.h"


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def _run_verify(manifest_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(VERIFY_SCRIPT), str(manifest_path)],
        capture_output=True,
        text=True,
    )


def _sh_tree_sha256(root: Path) -> str:
    """Tree hash of ``root`` via the EXACT pipeline verify_clap_sdk.sh uses.

    Shelling out (rather than reimplementing in Python) guarantees the generated
    sandbox manifest's hash matches what the verifier recomputes, so the only
    variable the RED test isolates is the single mutated header byte.
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


def _sandbox_manifest(include_dir: Path, tree_hash: str) -> str:
    """A manifest pinning ``include_dir`` (absolute) at ``tree_hash``.

    Reuses the repo manifest's source_repo/version/commit verbatim (identity
    metadata) but redirects [clap_sdk.vendored_tree].path at the tmp copy with the
    supplied hash, so the verifier hashes the sandbox tree, not the real one.
    """
    clap = _load_manifest()["clap_sdk"]
    return (
        "\n".join(
            [
                "[clap_sdk]",
                f'source_repo = "{clap["source_repo"]}"',
                f'version = "{clap["version"]}"',
                f'commit_sha = "{clap["commit_sha"]}"',
                "",
                "[clap_sdk.vendored_tree]",
                f'path = "{include_dir}"',
                f'tree_sha256 = "{tree_hash}"',
            ]
        )
        + "\n"
    )


def test_verify_passes_clean():
    """GREEN: the real committed vendored headers match the pin -> exit 0."""
    result = _run_verify(MANIFEST)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_verify_detects_tampered_header(tmp_path):
    """RED-proving: mutate ONE byte of ONE vendored header -> verifier NONZERO.

    Copies the vendored tree to tmp, pins its freshly computed (matching) tree
    hash, and first proves the untampered copy verifies clean (exit 0) — so the
    only variable is the single mutated byte. Then flips one byte of entry.h and
    asserts the verifier exits NONZERO, proving the tree hash catches real drift.
    """
    sandbox_include = tmp_path / "include"
    shutil.copytree(VENDORED_INCLUDE, sandbox_include)

    clean_hash = _sh_tree_sha256(sandbox_include)
    manifest_path = tmp_path / "sandbox.manifest.toml"
    manifest_path.write_text(_sandbox_manifest(sandbox_include, clean_hash))

    baseline = _run_verify(manifest_path)
    assert baseline.returncode == 0, (
        f"untampered tmp copy should verify clean.\n"
        f"stdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )

    # Flip exactly one byte of one header (append a single space is a change too,
    # but mutate in place to keep the file set identical — pure content drift).
    victim = sandbox_include / _TAMPER_HEADER
    data = bytearray(victim.read_bytes())
    assert data, f"expected non-empty header: {victim}"
    data[0] ^= 0x01
    victim.write_bytes(bytes(data))

    tampered = _run_verify(manifest_path)
    assert tampered.returncode != 0, (
        f"tampered header byte must force NONZERO exit.\n"
        f"stdout:\n{tampered.stdout}\nstderr:\n{tampered.stderr}"
    )
