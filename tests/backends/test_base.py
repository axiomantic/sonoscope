"""Tests for the RenderBackend interface + data objects (Task C3, design 4.1).

The load-bearing contract here is I1: ``RenderMeta`` carries ONLY the
backend-owned fields, a strict SUBSET of ``report.render`` (schema
``RenderBlock``). The orchestrator (F-layer) fills the residual fields
``{backend, backend_version, determinism}`` — the backend never does.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sonoscope.backends.base import (
    ParamInfo,
    PluginInfo,
    RawState,
    RenderBackend,
    RenderMeta,
    RenderRequest,
    RenderResult,
)
from sonoscope.schema.models import RenderBlock


class FakeBackend:
    """Trivial in-test backend proving the Protocol is satisfiable."""

    id = "fake"
    version = "0.0.1"

    def probe(self, plugin_path: Path) -> PluginInfo:
        return PluginInfo(
            name="Fake",
            format="vst3",
            params=[
                ParamInfo(
                    name="cutoff",
                    index=0,
                    kind="float",
                    num_steps=None,
                    default=0.5,
                )
            ],
            input_channels=2,
            output_channels=2,
            is_instrument=True,
            needs_gui=False,
            latency_samples=0,
        )

    def render(self, req: RenderRequest) -> RenderResult:
        meta = RenderMeta(
            sample_rate_hz=req.sample_rate_hz,
            block_size=req.block_size,
            channels=req.channels,
            duration_s=1.0,
            wav_subtype="PCM_F32",
            wav_sha256="0" * 64,
            render_wall_ms=7,
            warnings=[],
        )
        return RenderResult(
            wav_path=Path("/tmp/fake.wav"), render_meta=meta, warnings=[]
        )


def _make_request() -> RenderRequest:
    return RenderRequest(
        plugin_path=Path("/plugins/Fake.vst3"),
        plugin_format="vst3",
        stimulus=None,
        param_set=None,
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        raw_state=None,
        seed=None,
    )


def test_fake_backend_satisfies_protocol():
    backend = FakeBackend()
    assert isinstance(backend, RenderBackend) is True

    result = backend.render(_make_request())
    assert isinstance(result, RenderResult) is True
    assert result.render_meta == RenderMeta(
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        duration_s=1.0,
        wav_subtype="PCM_F32",
        wav_sha256="0" * 64,
        render_wall_ms=7,
        warnings=[],
    )


def test_render_meta_field_set_is_exact():
    # RenderMeta owns exactly the eight backend-populated fields (design 4.1).
    render_meta_fields = {f.name for f in dataclasses.fields(RenderMeta)}
    assert render_meta_fields == {
        "sample_rate_hz",
        "block_size",
        "channels",
        "duration_s",
        "wav_subtype",
        "wav_sha256",
        "render_wall_ms",
        "warnings",
    }


def test_render_meta_is_subset_of_render_block():
    # I1 subset contract: RenderMeta fields ⊆ RenderBlock fields (exact).
    render_meta_fields = {f.name for f in dataclasses.fields(RenderMeta)}
    render_block_fields = set(RenderBlock.model_fields)
    assert (render_meta_fields <= render_block_fields) is True
    # The residual is exactly the orchestrator-owned trio (design 4.1 / I1).
    residual = render_block_fields - render_meta_fields
    assert residual == {"backend", "backend_version", "determinism"}


def test_raw_state_reuses_schema_type():
    # No duplication (single source of truth): backends.RawState IS the schema
    # RawStateBlock, which maps 1:1 to report.input.raw_state (design 4.1).
    from sonoscope.schema.models import RawStateBlock

    assert RawState is RawStateBlock
    state = RawState(captured=False)
    assert state.model_dump() == {
        "captured": False,
        "plugin_binary_sha256": None,
        "blob_ref": None,
    }


def test_paraminfo_kind_accepts_only_valid_kinds():
    for kind in ("float", "bool", "stepped"):
        info = ParamInfo(
            name="p", index=0, kind=kind, num_steps=None, default=0.0
        )
        assert info.kind == kind

    with pytest.raises(ValueError):
        ParamInfo(name="p", index=0, kind="int", num_steps=None, default=0.0)
