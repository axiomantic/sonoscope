"""Iterate engine tests (Task F3) — the significance gate.

Green-mirage discipline on the R2 noise-floor significance gate (design §3.4,
§8): the load-bearing test is ``test_noop_change_is_inconclusive`` — a sub-floor
change in the *expected* direction must be ``INCONCLUSIVE``, never a false PASS.
A significance-blind implementation (``verdict = PASS`` whenever the direction
sign matches, ignoring the measured floor) returns PASS here and is caught.

Fixtures are hand-authored :class:`AnalysisReport` / :class:`DeterminismFloors`
objects (F1 / F2 contract outputs) so the metric delta and the measured floor are
controlled exactly; assertions are exact-equality (exact verdict enums, exact
threshold arithmetic) per the plan's Level-4 standard.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import pytest

from sonoscope.errors import InputError
from sonoscope.iterate import (
    DEFAULT_NOISE_THRESHOLD_MULTIPLIER,
    _extract_metric,
    run_iterate,
)
from sonoscope.schema.models import AnalysisReport, DeterminismFloors, ExitCode

METRIC = "deterministic.summary.spectral_centroid_hz"


# --- reference fixtures (mirror design §3.2 / §3.5) -------------------------

_FLOORS_OBJ = {
    "schema_version": "1.0.0",
    "kind": "determinism-floors",
    "generated_at": "2026-07-04T12:00:00Z",
    "binary_sha256": "bin-sha",
    "patch_class": "noisy",
    "resolved_sha256": "resolved-sha",
    "stimulus_ref": "corpus/midi/c3_sustain_2s.mid",
    "repeats": 5,
    "is_bit_identical": False,
    "floors": {
        METRIC: {
            "floor": 15.0,
            "unit": "hz",
            "method": "range",
            "repeats": 5,
            "timestamp": "2026-07-04T12:00:00Z",
            "binary_sha256": "bin-sha",
            "patch_class": "noisy",
        }
    },
}

_REPORT = {
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
            "is_bit_identical": False,
            "patch_class": "noisy",
            "noise_floor_measured": True,
            "floors_ref": "cache/determinism/bin-sha/noisy.json",
            "floors": _FLOORS_OBJ,
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
            "spectral_centroid_hz": 1000.0,
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
            "silence_threshold_dbfs": -80.0,
            "has_nan": False,
            "has_inf": False,
            "has_denormal": False,
            "clip_count": 0,
            "clip_fraction": 0.0,
            "dc_offset_exceeds": False,
            "dc_offset_threshold": 0.001,
        },
        "notes": [],
    },
    "tripwires": {
        "expected_audio": True,
        "results": [
            {"id": "silent-output", "verdict": "PASS", "detail": None},
        ],
        "overall": "PASS",
    },
    "perception": {
        "status": "disabled",
        "grounding": "none",
        "adapter": None,
        "description": None,
        "structured": None,
        "grounding_map": None,
        "disclaimer": None,
    },
    "errors": [],
}


def _report(centroid: float) -> AnalysisReport:
    """A valid :class:`AnalysisReport` with a controlled centroid metric value."""
    data = deepcopy(_REPORT)
    data["deterministic"]["summary"]["spectral_centroid_hz"] = centroid
    return AnalysisReport.model_validate(data)


def _floors(
    floor: float,
    *,
    entry_repeats: int = 5,
    floors_repeats: int = 5,
) -> DeterminismFloors:
    """A :class:`DeterminismFloors` with a controlled floor + presence-stability.

    ``entry_repeats`` is how many of ``floors_repeats`` renders had the metric
    present; ``entry_repeats < floors_repeats`` marks an unstable (present<->absent)
    feature whose floor is unreliable (F2 handoff, presence-instability).
    """
    data = deepcopy(_FLOORS_OBJ)
    data["repeats"] = floors_repeats
    data["floors"][METRIC]["floor"] = floor
    data["floors"][METRIC]["repeats"] = entry_repeats
    return DeterminismFloors.model_validate(data)


# --- the four required acceptance tests (plan F3) ---------------------------


def test_noop_change_is_inconclusive():
    """RED-proving false-PASS guard: a SUB-FLOOR change IN the expected direction
    is INCONCLUSIVE, never PASS.

    baseline 1000 -> candidate 999 is a 1.0 Hz decrease; the measured floor is
    10.0 Hz so the noise threshold is 30.0 Hz. The change (1.0) is buried in the
    plugin's own noise (<= 30.0), so no claim can be made. A significance-blind
    implementation that returns PASS because "decrease matches decrease" is caught
    here — the delta magnitude never cleared the floor.
    """
    result = run_iterate(
        _report(1000.0),
        _report(999.0),
        _floors(10.0),
        metric=METRIC,
        direction="decrease",
    )
    assert result.verdict == "INCONCLUSIVE"
    assert result.delta.significant is False
    assert result.delta.abs_delta == -1.0


def test_supra_threshold_matching_is_pass():
    """A decrease well beyond the noise threshold + direction=decrease -> PASS."""
    result = run_iterate(
        _report(1000.0),
        _report(200.0),
        _floors(15.0),
        metric=METRIC,
        direction="decrease",
    )
    assert result.verdict == "PASS"
    assert result.delta.significant is True
    assert result.delta.matches_expectation is True
    assert result.delta.abs_delta == -800.0
    assert result.delta.baseline_value == 1000.0
    assert result.delta.candidate_value == 200.0


def test_supra_threshold_wrong_direction_is_fail():
    """A supra-threshold INCREASE when a decrease was expected -> FAIL."""
    result = run_iterate(
        _report(200.0),
        _report(1000.0),
        _floors(15.0),
        metric=METRIC,
        direction="decrease",
    )
    assert result.verdict == "FAIL"
    assert result.delta.significant is True
    assert result.delta.matches_expectation is False
    assert result.delta.abs_delta == 800.0


def test_noise_threshold_is_floor_times_multiplier():
    """The stored ``noise_threshold`` is EXACTLY ``measured_floor * 3.0``."""
    result = run_iterate(
        _report(1000.0),
        _report(200.0),
        _floors(15.0),
        metric=METRIC,
        direction="decrease",
    )
    assert result.delta.measured_floor == 15.0
    assert result.delta.noise_threshold_multiplier == 3.0
    assert DEFAULT_NOISE_THRESHOLD_MULTIPLIER == 3.0
    assert result.delta.noise_threshold == 15.0 * 3.0
    assert (
        result.delta.noise_threshold
        == result.delta.measured_floor * result.delta.noise_threshold_multiplier
    )


# --- significance-gate coverage beyond the required four --------------------


def test_presence_instability_floor_is_inconclusive():
    """A supra-threshold change whose metric floor is UNRELIABLE -> INCONCLUSIVE.

    F2 records a per-feature ``repeats`` = how many of N renders had the feature
    present. When that shrinks below the full N (an unstable present<->absent
    feature), the measured floor is not trustworthy — a small/zero floor must not
    be over-trusted into a confident "significant". Here the raw magnitude (800)
    clears the naive threshold (45), but ``entry_repeats=2 < floors_repeats=5`` so
    the verdict is INCONCLUSIVE, not PASS.
    """
    result = run_iterate(
        _report(1000.0),
        _report(200.0),
        _floors(15.0, entry_repeats=2, floors_repeats=5),
        metric=METRIC,
        direction="decrease",
    )
    assert result.verdict == "INCONCLUSIVE"
    assert result.delta.significant is False


def test_min_effect_above_delta_is_inconclusive():
    """``min_effect`` raises the effective floor without changing stored fields.

    A 100 Hz decrease clears the noise threshold (45) but not the caller's
    asserted minimum effect size (200), so the verdict is INCONCLUSIVE. The stored
    ``noise_threshold`` still records ``measured_floor * multiplier`` (45), and the
    expectation echoes ``min_effect``.
    """
    result = run_iterate(
        _report(1000.0),
        _report(900.0),
        _floors(15.0),
        metric=METRIC,
        direction="decrease",
        min_effect=200.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert result.delta.significant is False
    assert result.delta.noise_threshold == 45.0
    assert result.expectation.min_effect == 200.0


# --- Fix 1: direction="stable" verdict + consistency (design §3.4, reading A) --


def test_stable_unchanged_metric_is_pass():
    """RED-proving the stable-direction fix: a SUB-FLOOR change + direction=stable
    is PASS (the metric stayed put), never INCONCLUSIVE.

    baseline 1000 -> candidate 999 is a 1.0 Hz change; the measured floor is 10.0
    Hz so the noise threshold is 30.0 Hz. The change (1.0) is below the floor, so
    the metric did NOT meaningfully move — a ``stable`` assertion PASSES. Under the
    OLD short-circuit-to-INCONCLUSIVE ``_verdict`` (``not significant`` returned
    INCONCLUSIVE for every direction) this returned INCONCLUSIVE, so this test is
    the RED proof of the reading-A fix. ``matches_expectation`` (``not significant``
    = True) is now CONSISTENT with the PASS verdict, not the prior self-contradiction.
    """
    result = run_iterate(
        _report(1000.0),
        _report(999.0),
        _floors(10.0),
        metric=METRIC,
        direction="stable",
    )
    assert result.verdict == "PASS"
    assert result.delta.significant is False
    assert result.delta.matches_expectation is True
    assert result.delta.abs_delta == -1.0


def test_stable_significant_change_is_fail():
    """A SUPRA-threshold change + direction=stable is FAIL — the metric was asserted
    to stay put but it moved beyond the noise floor.

    ``matches_expectation`` (``not significant`` = False) is consistent with FAIL.
    """
    result = run_iterate(
        _report(1000.0),
        _report(200.0),
        _floors(15.0),
        metric=METRIC,
        direction="stable",
    )
    assert result.verdict == "FAIL"
    assert result.delta.significant is True
    assert result.delta.matches_expectation is False
    assert result.delta.abs_delta == -800.0


# --- Fix 2: close direction + error-path green-mirage gaps -------------------


def test_increase_direction_matching_is_pass():
    """A supra-threshold INCREASE + direction=increase -> PASS."""
    result = run_iterate(
        _report(200.0),
        _report(1000.0),
        _floors(15.0),
        metric=METRIC,
        direction="increase",
    )
    assert result.verdict == "PASS"
    assert result.delta.significant is True
    assert result.delta.matches_expectation is True
    assert result.delta.abs_delta == 800.0


def test_change_direction_significant_is_pass():
    """A supra-threshold change + direction=change -> PASS regardless of sign.

    ``change`` asserts only that the metric moved beyond the floor, either way.
    Here the candidate is LOWER (a decrease) yet direction=change still PASSes.
    """
    result = run_iterate(
        _report(1000.0),
        _report(200.0),
        _floors(15.0),
        metric=METRIC,
        direction="change",
    )
    assert result.verdict == "PASS"
    assert result.delta.significant is True
    assert result.delta.matches_expectation is True
    assert result.delta.abs_delta == -800.0


def test_unknown_metric_path_is_input_error():
    """An unknown dotted metric path -> InputError ITERATE_METRIC_NOT_FOUND (exit 2)."""
    with pytest.raises(InputError) as exc_info:
        run_iterate(
            _report(1000.0),
            _report(200.0),
            _floors(15.0),
            metric="deterministic.summary.no_such_feature",
            direction="decrease",
        )
    assert exc_info.value.code == "ITERATE_METRIC_NOT_FOUND"
    assert exc_info.value.exit_code == ExitCode.INPUT
    assert exc_info.value.exit_code == 2


def test_suppressed_metric_is_input_error():
    """A suppressed/None metric (tempo_bpm) -> InputError ITERATE_METRIC_UNAVAILABLE
    (exit 2) — no delta can be computed."""
    with pytest.raises(InputError) as exc_info:
        run_iterate(
            _report(1000.0),
            _report(200.0),
            _floors(15.0),
            metric="deterministic.summary.tempo_bpm",
            direction="change",
        )
    assert exc_info.value.code == "ITERATE_METRIC_UNAVAILABLE"
    assert exc_info.value.exit_code == ExitCode.INPUT
    assert exc_info.value.exit_code == 2


def test_missing_floor_is_input_error():
    """A metric with no FloorEntry -> InputError ITERATE_FLOOR_MISSING (exit 2).

    ``spectral_bandwidth_hz`` is a valid numeric report feature, but only the
    centroid metric has a measured floor in the fixture, so the floor lookup fails.
    """
    with pytest.raises(InputError) as exc_info:
        run_iterate(
            _report(1000.0),
            _report(200.0),
            _floors(15.0),
            metric="deterministic.summary.spectral_bandwidth_hz",
            direction="decrease",
        )
    assert exc_info.value.code == "ITERATE_FLOOR_MISSING"
    assert exc_info.value.exit_code == ExitCode.INPUT
    assert exc_info.value.exit_code == 2


# --- dict-field traversal in the metric extractor (Gemini review finding) ----


def _report_with_dict_floor_key(key: str, floor_value: float) -> AnalysisReport:
    """A report whose ``render.determinism.floors.floors`` dict carries ``key``.

    ``floors`` is a ``dict[str, FloorEntry]`` (design §3.5), so a metric path that
    walks THROUGH the dict (``render.determinism.floors.floors.<key>.floor``)
    exercises the dict-descent branch of ``_extract_metric``. A simple (dotless)
    key is used so a single path segment addresses it unambiguously.
    """
    data = deepcopy(_REPORT)
    entry = deepcopy(_FLOORS_OBJ["floors"][METRIC])
    entry["floor"] = floor_value
    data["render"]["determinism"]["floors"]["floors"][key] = entry
    return AnalysisReport.model_validate(data)


def test_metric_path_traverses_dict():
    """RED-proving: a metric path resolving THROUGH a dict field descends to the
    numeric leaf. Without the dict branch the walk hits the else -> InputError, so
    the extractor could never reach a dict-nested value."""
    report = _report_with_dict_floor_key("mymetric", 42.0)
    value = _extract_metric(
        report, "render.determinism.floors.floors.mymetric.floor"
    )
    assert value == 42.0


def test_metric_path_unknown_dict_key_is_input_error():
    """A missing dict key is the SAME typed ITERATE_METRIC_NOT_FOUND InputError
    (exit 2) as a missing model field — never a silent fallback."""
    report = _report_with_dict_floor_key("mymetric", 42.0)
    with pytest.raises(InputError) as exc_info:
        _extract_metric(
            report, "render.determinism.floors.floors.no_such_key.floor"
        )
    assert exc_info.value.code == "ITERATE_METRIC_NOT_FOUND"
    assert exc_info.value.exit_code == ExitCode.INPUT
    assert exc_info.value.exit_code == 2


# --- Fix 3: guard non-positive multiplier -----------------------------------


def test_nonpositive_multiplier_is_input_error():
    """A non-positive ``noise_threshold_multiplier`` -> InputError
    ITERATE_MULTIPLIER_INVALID (exit 2) — mirrors determinism.py's ``repeats < 2``
    guard; a <= 0 multiplier would collapse/invert the significance threshold."""
    with pytest.raises(InputError) as exc_info:
        run_iterate(
            _report(1000.0),
            _report(200.0),
            _floors(15.0),
            metric=METRIC,
            direction="decrease",
            noise_threshold_multiplier=0.0,
        )
    assert exc_info.value.code == "ITERATE_MULTIPLIER_INVALID"
    assert exc_info.value.exit_code == ExitCode.INPUT
    assert exc_info.value.exit_code == 2
