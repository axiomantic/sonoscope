"""Tests for the PerceptionAdapter interface + NullAdapter (Task C4, design 4.3).

Contract: perception is ALWAYS present via NullAdapter so the loop emits valid
JSON with no model installed. A NullAdapter describe() yields a schema-valid
perception block with status=="disabled"; health() reports available=False.
"""

from __future__ import annotations

from pathlib import Path

from sonoscope.perception.base import AdapterHealth, PerceptionAdapter
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.schema.models import PerceptionBlock


class _OkAdapter:
    """In-test adapter proving a healthy 'ok' block also validates + satisfies
    the Protocol (the non-Null path is exercisable through the same contract)."""

    id = "fake-ok"
    grounding = "advisory-freetext"

    def describe(self, wav_path: Path, deterministic=None) -> PerceptionBlock:
        return PerceptionBlock(
            status="ok",
            grounding="advisory-freetext",
            adapter={
                "id": "fake-ok",
                "model": "fake",
                "quant": "none",
                "runtime": "test",
                "model_sha256": "0" * 64,
            },
            description="a bright tone",
            grounding_map={"brightness": "deterministic.summary.spectral_centroid_hz"},
            disclaimer="Advisory only.",
        )

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            available=True, runtime="test", model_id="fake", reason=None
        )


def test_null_and_ok_satisfy_protocol():
    assert isinstance(NullAdapter(), PerceptionAdapter) is True
    assert isinstance(_OkAdapter(), PerceptionAdapter) is True


def test_null_returns_disabled(tmp_path):
    wav = tmp_path / "x.wav"
    block = NullAdapter().describe(wav, None)

    assert block.status == "disabled"
    # Exact-equality on the full dumped block (a disabled block carries no
    # adapter output per I7).
    assert block.model_dump() == {
        "status": "disabled",
        "grounding": "none",
        "adapter": None,
        "description": None,
        "structured": None,
        "grounding_map": None,
        "disclaimer": None,
    }
    # Validates against the C1 schema perception model (round-trips exactly).
    assert PerceptionBlock.model_validate(block.model_dump()) == block


def test_null_health_available_false():
    health = NullAdapter().health()
    assert health == AdapterHealth(
        available=False,
        runtime="none",
        model_id="none",
        reason="No perception model configured (NullAdapter).",
    )
    assert health.available is False
    assert health.reason is not None


def test_ok_adapter_output_validates():
    block = _OkAdapter().describe(Path("/x.wav"), None)
    assert block.status == "ok"
    assert PerceptionBlock.model_validate(block.model_dump()) == block
