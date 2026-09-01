"""Schema contract for the wav-analysis report kind (design section 10, D1).

WavAnalysisReport is a top-level JSON ARRAY (RootModel[list[WavChunkAnalysis]],
length >= 1). Each element is self-describing (schema_version + kind). Models forbid
extra keys (strict). These tests pin the additive contract and the SCHEMA_VERSION
bump 1.3.0 -> 1.4.0.

All assertions are exact-equality (Level 4+): complete expected objects/dicts are
constructed and compared with ``==``; no substring/truthiness checks on the value
under test. ``make_wav_chunk`` reuses the deterministic + descriptors fixtures from
the sibling schema tests so the chunk block is a real, fully-populated object.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from sonoscope.schema.generate import SCHEMA_KINDS, json_schema_for
from sonoscope.schema.models import (
    SCHEMA_VERSION,
    AnalyzedWindow,
    DescriptorsBlock,
    DeterministicBlock,
    InputProvenance,
    WavAnalysisReport,
    WavChunkAnalysis,
)
from tests.schema.test_descriptors_models import DESCRIPTORS
from tests.schema.test_models import REPORT


def test_schema_version_bumped_to_1_4_0():
    # Additive wav-analysis kind + input_provenance (MINOR bump; major stays 1).
    assert SCHEMA_VERSION == "1.5.0"


def _provenance(**overrides) -> InputProvenance:
    base = dict(
        original_sample_rate=44100,
        n_channels=1,
        source_subtype="PCM_16",
        resample_res_type="soxr_hq",
        soxr_version="1.1.0",
        analyzed_window=AnalyzedWindow(
            native_offset_samples=0,
            native_length_samples=44100,
            native_sample_rate=44100,
            analyzed_samples_48k=48000,
        ),
        max_chunk_seconds=600.0,
        chunk_index=0,
        n_chunks=1,
    )
    base.update(overrides)
    return InputProvenance(**base)


@pytest.fixture
def make_wav_chunk():
    """Factory building a valid, fully-populated WavChunkAnalysis.

    Reuses REPORT["deterministic"] and DESCRIPTORS (the sibling schema fixtures)
    so the deterministic + descriptors blocks are real objects, not stubs.
    """

    def _make(*, chunk_index: int, n_chunks: int) -> WavChunkAnalysis:
        return WavChunkAnalysis(
            generated_at="2026-07-09T12:00:00Z",
            sonoscope_version="0.1.0",
            input_provenance=_provenance(
                chunk_index=chunk_index, n_chunks=n_chunks
            ),
            deterministic=DeterministicBlock.model_validate(
                deepcopy(REPORT["deterministic"])
            ),
            descriptors=DescriptorsBlock.model_validate(deepcopy(DESCRIPTORS)),
        )

    return _make


def test_input_provenance_defaults_and_literals():
    p = _provenance()
    assert p.analysis_dtype == "float32"
    assert p.channel_reduction == "mean_spectral_max_peak"


def test_input_provenance_channel_reduction_is_frozen_literal():
    # The single allowed value; any other string is a hard ValidationError.
    with pytest.raises(ValidationError):
        _provenance(channel_reduction="max")


def test_input_provenance_48k_noop_allows_null_resample():
    p = _provenance(
        original_sample_rate=48000, resample_res_type=None, soxr_version=None
    )
    assert p.resample_res_type is None
    assert p.soxr_version is None


def test_wav_chunk_analysis_forbids_extra_keys(make_wav_chunk):
    # Start from a fully-valid chunk and add exactly one unexpected key, so the
    # only possible validation failure is the strict-extra contract (not a
    # missing-field error masking the removal of extra="forbid").
    valid = make_wav_chunk(chunk_index=0, n_chunks=1).model_dump()
    valid["unexpected_key"] = 1
    with pytest.raises(ValidationError) as exc:
        WavChunkAnalysis.model_validate(valid)
    errors = exc.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"


def test_wav_chunk_analysis_full_object_round_trips(make_wav_chunk):
    chunk = make_wav_chunk(chunk_index=0, n_chunks=1)
    reloaded = WavChunkAnalysis.model_validate_json(chunk.model_dump_json())
    assert reloaded == chunk
    # kind + schema_version defaults are stamped on every element.
    assert reloaded.kind == "wav-chunk-analysis"
    assert reloaded.schema_version == "1.5.0"


def test_wav_analysis_report_roundtrips_single_and_multi_chunk(make_wav_chunk):
    one_list = [make_wav_chunk(chunk_index=0, n_chunks=1)]
    one = WavAnalysisReport(one_list)
    reparsed = WavAnalysisReport.model_validate_json(one.model_dump_json())
    assert reparsed.root == one_list

    many_list = [make_wav_chunk(chunk_index=i, n_chunks=3) for i in range(3)]
    many = WavAnalysisReport(many_list)
    reparsed_many = WavAnalysisReport.model_validate_json(many.model_dump_json())
    assert reparsed_many.root == many_list
    assert [c.input_provenance.chunk_index for c in reparsed_many.root] == [0, 1, 2]


def test_wav_analysis_kind_registered_and_emits_array_schema():
    assert "wav-analysis" in SCHEMA_KINDS
    schema = json_schema_for("wav-analysis")
    # RootModel[list[...]] produces a top-level array schema.
    assert schema["type"] == "array"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    # The docstring's "Length >= 1" invariant must be reflected in the schema.
    assert schema["minItems"] == 1


def test_wav_analysis_report_rejects_empty_list():
    with pytest.raises(ValidationError) as exc_info:
        WavAnalysisReport.model_validate([])
    errors = exc_info.value.errors()
    assert any(e["type"] == "too_short" for e in errors)

    with pytest.raises(ValidationError):
        WavAnalysisReport([])
