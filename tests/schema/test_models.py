"""Schema contract tests (Task C1).

Exact-equality round-trips against hand-authored reference dicts mirroring
design section 3.2-3.6, plus RED-proving validators for perception nullability
and the exit-code contract.
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from sonoscope.schema.models import (
    SCHEMA_VERSION,
    AnalysisReport,
    DeterminismFloors,
    ErrorItem,
    ExitCode,
    FatalError,
    IterateDelta,
    PerceptionBlock,
)


# --- Reference fixtures (mirror design section 3) ---------------------------

FLOORS_OBJ = {
    "schema_version": "1.0.0",
    "kind": "determinism-floors",
    "generated_at": "2026-07-04T12:00:00Z",
    "binary_sha256": "bin-sha",
    "patch_class": "noise_free",
    "resolved_sha256": "resolved-sha",
    "stimulus_ref": "corpus/midi/c3_sustain_2s.mid",
    "repeats": 5,
    "is_bit_identical": True,
    "floors": {
        "deterministic.summary.spectral_centroid_hz": {
            "floor": 15.0,
            "unit": "hz",
            "method": "range",
            "repeats": 5,
            "timestamp": "2026-07-04T12:00:00Z",
            "binary_sha256": "bin-sha",
            "patch_class": "noise_free",
        }
    },
}

REPORT = {
    "schema_version": "1.0.0",
    "generated_at": "2026-07-04T12:00:00Z",
    "sonoscope_version": "0.1.0",
    "input": {
        "plugin": {
            "path": "/plugins/Surge XT.vst3",
            "format": "vst3",
            "binary_sha256": "bin-sha",
            "backend": "pedalboard-vst3",
        },
        "stimulus": {
            "kind": "midi",
            "ref": "corpus/midi/c3_sustain_2s.mid",
            "ref_sha256": "stim-sha",
            "sample_rate_hz": 48000,
            "duration_s": 2.0,
        },
        "param_set": {
            "ref": "specs/lowpass_open.json",
            "spec_sha256": "spec-sha",
            "resolved_sha256": "resolved-sha",
        },
        "raw_state": {
            "captured": False,
            "plugin_binary_sha256": None,
            "blob_ref": None,
        },
    },
    "render": {
        "sample_rate_hz": 48000,
        "block_size": 512,
        "channels": 2,
        "duration_s": 2.0,
        "wav_subtype": "PCM_F32",
        "backend": "pedalboard-vst3",
        "backend_version": "0.9.23",
        "wav_sha256": "wav-sha",
        "render_wall_ms": 640,
        "determinism": {
            "repeats": 5,
            "is_bit_identical": True,
            "patch_class": "noise_free",
            "noise_floor_measured": True,
            "floors_ref": "cache/determinism/bin-sha/noise_free.json",
            "floors": FLOORS_OBJ,
        },
        "warnings": [],
    },
    "deterministic": {
        "library": {
            "name": "librosa",
            "version": "0.10.2",
            "params_sha256": "params-sha",
        },
        "summary": {
            "duration_s": 2.0,
            "sample_rate_hz": 48000,
            "channels": 2,
            "rms_dbfs": -18.3,
            "peak_dbfs": -3.1,
            "crest_factor_db": 15.2,
            "dc_offset": 0.0002,
            "spectral_centroid_hz": 1830.5,
            "spectral_bandwidth_hz": 2200.0,
            "spectral_rolloff_hz": 4100.0,
            "spectral_flatness": 0.042,
            "zero_crossing_rate": 0.081,
            "onset_count": 1,
            "onset_rate_hz": 0.5,
            "tempo_bpm": None,
            "tempo_confidence": None,
            "mfcc_mean": [1.0, 2.0, 3.0],
            "mfcc_std": [0.1, 0.2, 0.3],
        },
        "integrity": {
            "is_silent": False,
            "all_channels_silent": False,
            "silence_threshold_dbfs": -80.0,
            "has_nan": False,
            "has_inf": False,
            "has_denormal": False,
            "clip_count": 0,
            "clip_fraction": 0.0,
            "dc_offset_exceeds": False,
            "dc_offset_threshold": 0.001,
        },
        "notes": [
            "tempo_bpm suppressed: onset_count < 4 (not enough onsets for reliable estimate)"
        ],
    },
    "tripwires": {
        "expected_audio": True,
        "results": [
            {"id": "silent-output", "verdict": "PASS", "detail": "rms_dbfs -18.3 > -80.0"},
            {"id": "nan-inf", "verdict": "PASS", "detail": None},
            {"id": "denormal", "verdict": "PASS", "detail": None},
            {"id": "clipping", "verdict": "PASS", "detail": "clip_fraction 0.0"},
        ],
        "overall": "PASS",
    },
    "perception": {
        "status": "ok",
        "grounding": "advisory-freetext",
        "adapter": {
            "id": "qwen2-audio-local",
            "model": "Qwen2-Audio-7B",
            "quant": "q4_K_M",
            "runtime": "nexa-gguf",
            "model_sha256": "model-sha",
        },
        "description": "A bright, sustained tone with moderate high-frequency content and no obvious noise.",
        "structured": {
            "brightness": "bright",
            "noisiness": "tonal",
            "dynamics": "steady",
        },
        "grounding_map": {
            "brightness": "deterministic.summary.spectral_centroid_hz",
            "noisiness": "deterministic.summary.spectral_flatness",
            "dynamics": "deterministic.summary.crest_factor_db",
        },
        "disclaimer": "Advisory only. Not ground truth. May be inaccurate or hallucinated.",
    },
    "descriptors": None,
    "descriptor_gate": None,
    "errors": [],
}

DISABLED_PERCEPTION = {
    "status": "disabled",
    "grounding": "none",
    "adapter": None,
    "description": None,
    "structured": None,
    "grounding_map": None,
    "disclaimer": None,
}

FATAL = {
    "schema_version": "1.0.0",
    "kind": "fatal-error",
    "generated_at": "2026-07-04T12:00:00Z",
    "sonoscope_version": "0.1.0",
    "error": {
        "code": "RENDER_SUBPROCESS_CRASH",
        "message": "Render subprocess exited with signal SIGSEGV after 1 retry.",
        "detail": {"signal": "SIGSEGV", "retries": 1},
        "severity": "fatal",
        "component": "render",
    },
}

ERROR_ITEM = {
    "code": "RENDER_BLOCK_SIZE_COERCED",
    "message": "Backend coerced block size 512 -> 480.",
    "detail": {"requested": 512, "actual": 480},
    "severity": "warning",
    "component": "render",
}


def _delta_report():
    candidate = deepcopy(REPORT)
    candidate["deterministic"]["summary"]["spectral_centroid_hz"] = 980.2
    return {
        "schema_version": "1.0.0",
        "kind": "iterate-delta",
        "baseline": deepcopy(REPORT),
        "candidate": candidate,
        "expectation": {
            "metric": "deterministic.summary.spectral_centroid_hz",
            "direction": "decrease",
            "min_effect": None,
        },
        "delta": {
            "metric": "deterministic.summary.spectral_centroid_hz",
            "baseline_value": 1830.5,
            "candidate_value": 980.2,
            "abs_delta": -850.3,
            "measured_floor": 15.0,
            "noise_threshold_multiplier": 3.0,
            "noise_threshold": 45.0,
            "significant": True,
            "matches_expectation": True,
        },
        "verdict": "PASS",
    }


# --- Round-trip exact-equality tests ---------------------------------------


def test_schema_version_constant():
    # Bumped 1.0.0 -> 1.1.0 (additive midi-analysis kind, S1), then
    # 1.1.0 -> 1.2.0 (additive descriptors block), then
    # 1.2.0 -> 1.3.0 (additive descriptor_gate field), then
    # 1.4.0 -> 1.5.0 (additive IntegrityBlock.all_channels_silent); major
    # stays 1 so check_schema_version is unaffected.
    assert SCHEMA_VERSION == "1.5.0"


def test_report_roundtrips_reference_json():
    assert AnalysisReport.model_validate(REPORT).model_dump(mode="json") == REPORT


def test_determinism_floors_roundtrips():
    assert DeterminismFloors.model_validate(FLOORS_OBJ).model_dump(mode="json") == FLOORS_OBJ


def test_iterate_delta_roundtrips():
    d = _delta_report()
    assert IterateDelta.model_validate(d).model_dump(mode="json") == d


def test_fatal_error_roundtrips():
    assert FatalError.model_validate(FATAL).model_dump(mode="json") == FATAL


def test_error_item_roundtrips():
    assert ErrorItem.model_validate(ERROR_ITEM).model_dump(mode="json") == ERROR_ITEM


# --- Strictness & enum contracts -------------------------------------------


def test_extra_field_rejected():
    d = deepcopy(REPORT)
    d["unexpected"] = 1
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(d)


def test_verdict_enum_rejects_unknown():
    d = deepcopy(REPORT)
    d["tripwires"]["overall"] = "MAYBE"
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(d)


def test_wav_subtype_rejects_non_pcm_f32():
    d = deepcopy(REPORT)
    d["render"]["wav_subtype"] = "PCM_16"
    with pytest.raises(ValidationError):
        AnalysisReport.model_validate(d)


def test_exit_codes_match_design():
    assert {c.name: c.value for c in ExitCode} == {
        "OK": 0,
        "USAGE": 1,
        "INPUT": 2,
        "RENDER": 3,
        "ANALYSIS": 4,
        "ENVIRONMENT": 5,
    }


# --- Perception nullability (I7): RED + GREEN ------------------------------


def test_disabled_perception_block_validates():
    block = PerceptionBlock.model_validate(DISABLED_PERCEPTION)
    assert block.model_dump(mode="json") == DISABLED_PERCEPTION


@pytest.mark.parametrize(
    "missing", ["adapter", "description", "grounding_map", "disclaimer"]
)
def test_ok_perception_requires_fields(missing):
    d = deepcopy(REPORT["perception"])
    d[missing] = None
    with pytest.raises(ValidationError):
        PerceptionBlock.model_validate(d)


def test_ok_perception_allows_absent_structured():
    d = deepcopy(REPORT["perception"])
    d["structured"] = None
    block = PerceptionBlock.model_validate(d)
    expected = deepcopy(REPORT["perception"])
    expected["structured"] = None
    assert block.model_dump(mode="json") == expected
