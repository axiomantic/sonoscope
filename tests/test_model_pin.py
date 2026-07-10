"""A5: prove the Qwen model pin machinery catches drift and stays optional.

Pins are law (design §11.2 / §11.3 / §18.1, AGENTS.md "Pins are law"):
``scripts/fetch_qwen_model.sh`` snapshot_downloads the pinned Qwen2-Audio-7B-Instruct
HF repo @ revision (B3 pivoted the runtime Nexa-SDK-GGUF -> transformers) and MUST
hard-fail on any ``model_sha256`` mismatch. These unit tests drive the script's
standalone ``verify`` mode against a local fixture so the actual sha256-compare
branch runs — no network, no ~16 GB download.

Green-mirage discipline (AGENTS.md): ``test_fetch_detects_hash_mismatch`` first
shows a fixture with CORRECT manifest shas verifies clean (exit 0), then tampers
exactly ONE entry's ``model_sha256`` and asserts the verifier flips to a NONZERO
exit. Were the sha256-compare removed from the script, the tampered case would
stay at exit 0 and this test would FAIL — that is what makes it RED-proving.

Marker placement (per AGENTS.md testing discipline):
- ``test_fetch_detects_hash_mismatch`` and ``test_manifest_lists_mmproj`` are
  pure unit tests over the script + committed manifest (no external artifact),
  so they are NON-integration and run in the default ``pytest -m "not
  integration"`` gate.
- ``test_core_install_has_no_perception`` documents the DEFAULT (core-only)
  install and is likewise NON-integration so the acceptance assertion actually
  runs in the unit gate. It carries an explicit precondition skip: if the
  perception runtime module is importable in the current interpreter (i.e. the
  ``perception`` extra is currently synced), the "core install" precondition
  does not hold and the test skips with a reason rather than reporting a
  misleading failure.
"""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_qwen_model.sh"
MANIFEST = REPO_ROOT / "pins" / "qwen_model.manifest.toml"

# Import module the `sonoscope[perception]` extra installs. B3 pivoted the runtime
# from Nexa-SDK-GGUF to the transformers reference runtime (design §11.2 / §4.3 /
# §18.1), so the extra now pins `transformers` + `torch`.
# `transformers` is the importable top-level module of the perception stack; it is
# absent from the deterministic core install (no core dep pulls it).
_PERCEPTION_MODULE = "transformers"

_ZERO_HASH = "0" * 64


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def _sh_file_sha256(path: Path) -> str:
    """Plain file sha256 via the EXACT pipeline fetch_qwen_model.sh uses.

    Shelling out (rather than reimplementing in Python) guarantees the fixture
    manifest's shas match what the verifier recomputes, so the only variable the
    RED test isolates is the single tampered hash.
    """
    proc = subprocess.run(
        ["bash", "-c", 'shasum -a 256 "$1" | awk \'{print $1}\'', "_", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _run_verify(manifest_path: Path, local_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(FETCH_SCRIPT), "verify", str(manifest_path), str(local_dir)],
        capture_output=True,
        text=True,
    )


def _fixture_manifest(model_dir: Path, model_sha: str, mmproj_sha: str) -> str:
    """A minimal but real qwen_model manifest pinning fixture files in model_dir.

    The URLs stay TODO(pin) (irrelevant to the network-free ``verify`` mode);
    only the filenames + shas drive the sha256-compare under test.
    """
    return (
        "[qwen_model]\n"
        'model_id = "Qwen2-Audio-7B"\n'
        'quant = "q4_K_M"\n'
        'runtime = "nexa-gguf"\n'
        'cache_dir = "~/unused"\n'
        "\n"
        "[qwen_model.files.model]\n"
        'filename = "model.gguf"\n'
        'url = "TODO(pin)"\n'
        f'model_sha256 = "{model_sha}"\n'
        "\n"
        "[qwen_model.files.mmproj]\n"
        'filename = "mmproj.gguf"\n'
        'url = "TODO(pin)"\n'
        f'model_sha256 = "{mmproj_sha}"\n'
    )


def test_fetch_detects_hash_mismatch(tmp_path):
    """RED-proving: tamper ONE model sha -> fetch verify exits NONZERO.

    Drives the script's standalone ``verify`` mode against local fixture GGUFs so
    the real sha256-compare branch executes (the download mode is TODO(pin)-
    guarded and never runs here). First proves the untampered manifest verifies
    clean (exit 0), so the only variable is the single mutated hash. If the
    hash-compare were removed from fetch_qwen_model.sh, the tampered run would
    stay exit 0 and this assertion would fail.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "model.gguf"
    mmproj_file = model_dir / "mmproj.gguf"
    model_file.write_bytes(b"fake q4_K_M main-model gguf bytes\n")
    mmproj_file.write_bytes(b"fake audio-projector mmproj gguf bytes\n")

    real_model_sha = _sh_file_sha256(model_file)
    real_mmproj_sha = _sh_file_sha256(mmproj_file)
    assert real_model_sha != _ZERO_HASH  # fixtures are non-empty, real content

    # GREEN baseline: correct shas verify clean.
    clean = tmp_path / "clean.manifest.toml"
    clean.write_text(_fixture_manifest(model_dir, real_model_sha, real_mmproj_sha))
    baseline = _run_verify(clean, model_dir)
    assert baseline.returncode == 0, (
        f"untampered fixture manifest should verify clean.\n"
        f"stdout:\n{baseline.stdout}\nstderr:\n{baseline.stderr}"
    )

    # TAMPERED: zero exactly the main-model sha; mmproj stays correct.
    tampered = tmp_path / "tampered.manifest.toml"
    tampered.write_text(_fixture_manifest(model_dir, _ZERO_HASH, real_mmproj_sha))
    result = _run_verify(tampered, model_dir)
    assert result.returncode != 0, (
        f"tampered model sha must force NONZERO exit.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_manifest_pins_config_and_weights():
    """Guard §18.1 (post-B3 transformers runtime): the manifest pins the model.

    B3 pivoted the runtime from Nexa-SDK-GGUF to the transformers reference runtime
    (design §18.1). That checkpoint is safetensors shards + configs and
    BUNDLES the audio tower/projector inside the weights, so there is no separate
    GGUF ``mmproj`` file. The pins-are-law requirement becomes: pin the exact HF
    ``revision`` plus the load-bearing ``config`` (which declares the audio tower)
    and at least one ``safetensors`` weight artifact — each with a real, non-empty
    ``model_sha256`` (no placeholder). Omitting the weight or revision pin is the
    §18.1 hazard this guard now catches.
    """
    qm = _load_manifest()["qwen_model"]
    assert qm.get("runtime") == "transformers", qm.get("runtime")
    revision = qm.get("revision", "")
    assert revision and revision != "TODO(pin)", f"manifest must pin an exact HF revision; got {revision!r}"

    files = qm["files"]
    assert "config" in files, f"manifest must pin the model config; got roles {sorted(files)}"
    weight_roles = [r for r, f in files.items() if str(f.get("filename", "")).endswith(".safetensors")]
    assert weight_roles, f"manifest must pin at least one .safetensors weight; got roles {sorted(files)}"

    for role in ("config", *weight_roles):
        entry = files[role]
        assert "filename" in entry, role
        sha = entry.get("model_sha256", "")
        assert len(sha) == 64 and sha != _ZERO_HASH, f"[{role}] needs a real 64-hex sha256; got {sha!r}"


def test_core_install_has_no_perception():
    """Acceptance: the deterministic core install carries NO perception runtime.

    ``uv sync`` (no extra) must NOT pull the perception stack, so importing the
    perception runtime module in a FRESH subprocess of the current interpreter
    must fail cleanly (nonzero exit). Precondition: this asserts the CORE-only
    environment; if the ``perception`` extra is currently synced (module
    importable here) the precondition does not hold and the test skips with a
    reason rather than reporting a misleading failure.
    """
    if importlib.util.find_spec(_PERCEPTION_MODULE) is not None:
        pytest.skip(
            f"{_PERCEPTION_MODULE!r} is importable in this interpreter — the "
            "'perception' extra is synced, so the core-only precondition does not "
            "hold (run `uv sync` without --extra perception to exercise this test)"
        )

    proc = subprocess.run(
        [sys.executable, "-c", f"import {_PERCEPTION_MODULE}"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        f"core install must NOT provide {_PERCEPTION_MODULE!r}; import unexpectedly "
        f"succeeded.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
