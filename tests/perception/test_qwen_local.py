"""Tests for QwenLocalAdapter (Task G1, design §4.3 / §7.1 / §10.2).

Runtime: the transformers reference runtime
(`Qwen2AudioForConditionalGeneration` + `AutoProcessor`, torch/MPS fp16). These
tests assert the adapter contract that earlier feasibility testing confirmed.

Marker placement (AGENTS.md testing discipline):
- The graceful-`unavailable`, singleton-load-once, cooperative-timeout,
  freetext-grounding, and pins-drift tests are NON-integration: they never load
  the ~16 GB weights (they use a bad model path or a monkeypatched loader/model
  stub) so they run in the default `pytest -m "not integration"` gate.
- `test_describe_real_model` is `@pytest.mark.integration`: it loads the real
  model (via the `qwen_model` conftest skip guard) and runs a real audio-in
  describe(). It skips with an explicit reason when the model / `perception`
  extra is absent (never a silent pass).
"""

from __future__ import annotations

import time
import tomllib
from pathlib import Path

import pytest

from sonoscope.perception import qwen_local
from sonoscope.perception.base import AdapterHealth, PerceptionAdapter
from sonoscope.perception.qwen_local import QwenLocalAdapter
from sonoscope.schema.models import PerceptionBlock

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "pins" / "qwen_model.manifest.toml"


class _FakeLoaded:
    """Stand-in for a loaded model whose run() returns fixed text and never
    trips the deadline (proves the ok-path + singleton caching without weights)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def run(self, wav_path: Path, deadline, max_new_tokens: int) -> str:
        return self._text


class _SlowLoaded:
    """Cooperative slow model: loops calling the per-token deadline until it
    fires (mirrors the B3 StoppingCriteria decode loop). Proves the 60 s timeout
    wrapper returns status='error' — it does NOT prove native interruptibility
    (that is empirically recorded in B3 / I5, not asserted here)."""

    def run(self, wav_path: Path, deadline, max_new_tokens: int) -> str:
        while not deadline():
            time.sleep(0.005)
        return "partial (interrupted)"


def test_adapter_satisfies_protocol():
    # C4 contract: the concrete adapter is a structural PerceptionAdapter.
    assert isinstance(QwenLocalAdapter(), PerceptionAdapter) is True


def test_unavailable_when_model_missing(monkeypatch, tmp_path):
    # §10.2 graceful degradation: a KNOWN-absent model/runtime yields a
    # status='unavailable' block with a non-null message and NEVER raises into
    # the caller. A valid-form but uncached repo id exercises the real detection
    # whether or not the perception extra is installed: without it the runtime
    # import fails (ImportError); with it, from_pretrained(local_files_only=True)
    # on an uncached id raises OSError. Both are the narrowed KNOWN-absent set
    # (MINOR-3) and map to status='unavailable'.
    monkeypatch.setattr(qwen_local, "_STATE", None)
    adapter = QwenLocalAdapter(model_ref="sonoscope-fake/no-such-model")

    block = adapter.describe(tmp_path / "x.wav")
    assert block.status == "unavailable"
    assert block.description is not None
    # Loop-safe: a second call also returns unavailable (failure is not cached
    # as a poisoned singleton) and the block validates against the C1 schema.
    block2 = adapter.describe(tmp_path / "y.wav")
    assert block2.status == "unavailable"
    assert PerceptionBlock.model_validate(block.model_dump()) == block


def test_singleton_loads_once(monkeypatch):
    # R7 / §7.1: the model loads exactly ONCE per process (in-process singleton),
    # never per describe() call. Asserted via a load counter around a stub loader.
    calls = {"n": 0}

    def fake_load(model_ref, revision, device_pref):
        calls["n"] += 1
        return _FakeLoaded("a stub description")

    monkeypatch.setattr(qwen_local, "_STATE", None)
    monkeypatch.setattr(qwen_local, "_load_model", fake_load)

    adapter = QwenLocalAdapter()
    b1 = adapter.describe(Path("/x.wav"))
    b2 = adapter.describe(Path("/y.wav"))

    assert calls["n"] == 1
    assert b1.status == "ok"
    assert b2.status == "ok"


def test_unexpected_load_error_propagates_as_error(monkeypatch, tmp_path):
    # MINOR-3: a genuine load fault (NOT known-absent) — e.g. MPS OOM, a corrupted
    # shard, or a torch runtime fault — must NOT be masked as status='unavailable'.
    # It propagates out of describe() (which catches only _ModelUnavailable) so the
    # F1 perception-never-fatal boundary records it as status='error' — the honest
    # classification for a real environment fault, distinct from "not installed".
    monkeypatch.setattr(qwen_local, "_STATE", None)

    def boom(model_ref, revision, device_pref):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(qwen_local, "_load_model", boom)

    adapter = QwenLocalAdapter()
    with pytest.raises(RuntimeError):
        adapter.describe(tmp_path / "x.wav")
    # Load-failure-not-cached: the singleton stays None so a later call re-attempts
    # rather than being poisoned by the transient fault.
    assert qwen_local._STATE is None


def test_describe_timeout_becomes_error(monkeypatch):
    # M2 / I5: a describe() that exceeds the hard timeout returns status='error'
    # (exact enum), loop-safe. Cooperative stub — see the _SlowLoaded docstring.
    monkeypatch.setattr(qwen_local, "_STATE", None)
    monkeypatch.setattr(qwen_local, "_load_model", lambda *a: _SlowLoaded())

    adapter = QwenLocalAdapter(timeout_s=0.05)
    block = adapter.describe(Path("/x.wav"))

    assert block.status == "error"


def test_freetext_grounding_labeled(monkeypatch):
    # §4.3 / §3.3: default grounding is 'advisory-freetext' and an ok block
    # carries the exact advisory disclaimer string.
    assert QwenLocalAdapter.grounding == "advisory-freetext"
    assert (
        qwen_local.DISCLAIMER
        == "Advisory only. Not ground truth. May be inaccurate or hallucinated."
    )

    monkeypatch.setattr(qwen_local, "_STATE", None)
    monkeypatch.setattr(qwen_local, "_load_model", lambda *a: _FakeLoaded("a bright tone"))
    block = QwenLocalAdapter().describe(Path("/x.wav"))

    assert block.status == "ok"
    assert block.grounding == "advisory-freetext"
    assert block.disclaimer == qwen_local.DISCLAIMER
    # An ok freetext block validates and round-trips against the C1 schema.
    assert PerceptionBlock.model_validate(block.model_dump()) == block


def test_pins_match_manifest():
    # Pins are law (AGENTS.md): the runtime constants re-exported in qwen_local
    # (so the adapter needs no packaged manifest file) MUST NOT drift from the
    # authoritative pins/qwen_model.manifest.toml. This is the drift guard.
    with MANIFEST.open("rb") as fh:
        manifest = tomllib.load(fh)["qwen_model"]

    assert qwen_local.HF_REPO == manifest["hf_repo"]
    assert qwen_local.MODEL_REVISION == manifest["revision"]
    assert qwen_local.MODEL_ID == manifest["model_id"]
    assert qwen_local.MODEL_RUNTIME == manifest["runtime"]
    assert qwen_local.MODEL_SHA256 == manifest["files"]["weights_index"]["model_sha256"]


@pytest.mark.integration
def test_describe_real_model(qwen_model, tmp_path):
    # The real proof: load the pinned Qwen2-Audio weights via transformers+MPS
    # and run a real audio-in describe() on a short committed fixture. Skips
    # (explicit reason) when the model/extra is absent.
    wav = REPO_ROOT / "corpus" / "qwen_probe" / "cutoff__cutoff_high.wav"
    assert wav.exists()

    adapter = QwenLocalAdapter()
    health = adapter.health()
    assert health == AdapterHealth(
        available=True,
        runtime="transformers",
        model_id="Qwen2-Audio-7B-Instruct",
        reason=None,
    )

    block = adapter.describe(wav)
    assert block.status == "ok"
    assert block.grounding == "advisory-freetext"
    assert block.disclaimer == qwen_local.DISCLAIMER
    assert block.description is not None
    assert len(block.description) > 0
    assert PerceptionBlock.model_validate(block.model_dump()) == block
