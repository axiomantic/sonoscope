"""Task 4: ``iterate.diff_descriptor_terms`` + the ``iterate-descriptors`` CLI.

The pure ``diff_descriptor_terms`` reports the descriptor-term REGRESSION signal
(present/absent set changes + direction changes over gate-eligible measured terms)
plus a SEPARATE tolerance-banded value-drift sub-list. The ``iterate-descriptors``
subcommand loads two on-disk ``AnalysisReport`` JSONs (via the NEW ``_load_report``
helper), runs the diff, and prints a single-line strict-JSON object to stdout
(non-finite drift values sanitized to JSON ``null``).

All assertions are exact-equality (full ``DescriptorTermDiff`` objects / exact
``.code`` + ``detail`` / exact exit codes / byte-exact stdout) per AGENTS.md.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from sonoscope import cli
from sonoscope.iterate import (
    DescriptorTermDiff,
    ValueDrift,
    diff_descriptor_terms,
)
from sonoscope.schema import ExitCode
from sonoscope.schema.models import (
    AnalysisReport,
    AnalyzedWindow,
    DescriptorsBlock,
    DescriptorsLibrary,
    DeterministicBlock,
    InputProvenance,
    MeasuredDescriptor,
    WavAnalysisReport,
    WavChunkAnalysis,
)


# --- fixtures ---------------------------------------------------------------


def _block(measured: list[MeasuredDescriptor]) -> DescriptorsBlock:
    """F7 fixture helper: a valid ``DescriptorsBlock`` (summary + library required)."""
    return DescriptorsBlock(
        measured=list(measured),
        hybrid=[],
        advisory=[],
        summary="",
        library=DescriptorsLibrary(thresholds_sha256="x", deriver_version="test"),
    )


#: A known-valid ``AnalysisReport`` payload (mirrors the reference report used across
#: the suite); the ``descriptors`` block is injected per-test.
_BASE_REPORT = {
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
            "floors": {
                "schema_version": "1.0.0",
                "kind": "determinism-floors",
                "generated_at": "2026-07-04T12:00:00Z",
                "binary_sha256": "bin-sha",
                "patch_class": "noisy",
                "resolved_sha256": "resolved-sha",
                "stimulus_ref": "corpus/midi/c3_sustain_2s.mid",
                "repeats": 5,
                "is_bit_identical": False,
                "floors": {},
            },
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
        "results": [{"id": "silent-output", "verdict": "PASS", "detail": None}],
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


def _report(block: DescriptorsBlock | None) -> AnalysisReport:
    """A valid ``AnalysisReport`` with (or without) a descriptors block."""
    data = deepcopy(_BASE_REPORT)
    data["descriptors"] = block.model_dump() if block is not None else None
    return AnalysisReport.model_validate(data)


def _write_report(tmp_path, name: str, block: DescriptorsBlock | None) -> str:
    p = tmp_path / name
    p.write_text(_report(block).model_dump_json(), encoding="utf-8")
    return str(p)


# --- Step 4.1: diff_descriptor_terms exact equality --------------------------


def test_diff_added_removed_direction_and_drift():
    baseline = _block(
        [
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="high"),
            MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high"),
        ]
    )
    candidate = _block(
        [
            MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high"),
            MeasuredDescriptor(term="dense", value=9.0, metric="o", direction="high"),
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="low"),
        ]
    )
    assert diff_descriptor_terms(
        baseline, candidate, value_tolerance=0.5
    ) == DescriptorTermDiff(
        added=["dense"],
        removed=[],
        direction_changed=["bright"],
        value_drift=[
            ValueDrift(term="loud", baseline_value=-12.0, candidate_value=-9.5)
        ],
    )


def test_diff_value_drift_within_tolerance_not_flagged():
    baseline = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high")]
    )
    assert diff_descriptor_terms(
        baseline, candidate, value_tolerance=5.0
    ) == DescriptorTermDiff(
        added=[], removed=[], direction_changed=[], value_drift=[]
    )


def test_diff_value_drift_exactly_at_tolerance_not_flagged():
    # S1: _value_drifted uses strict ``abs(delta) > tolerance``; a delta whose
    # magnitude EXACTLY equals value_tolerance is at the boundary and NOT flagged.
    # abs(-9.5 - -12.0) == 2.5 == value_tolerance -> not > tolerance -> no drift.
    baseline = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high")]
    )
    assert diff_descriptor_terms(
        baseline, candidate, value_tolerance=2.5
    ) == DescriptorTermDiff(
        added=[], removed=[], direction_changed=[], value_drift=[]
    )


# Defense-in-depth: the PURE library function must reject a bad
# ``value_tolerance`` directly, independent of the CLI guard. A negative
# tolerance makes ``abs(delta) > tolerance`` always true (false-positive drift);
# a non-finite tolerance makes it always false (zero drift). Direct callers
# bypass the CLI's UsageError guard, so the function raises ValueError itself.


def test_diff_negative_value_tolerance_raises_valueerror():
    baseline = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high")]
    )
    with pytest.raises(ValueError) as excinfo:
        diff_descriptor_terms(baseline, candidate, value_tolerance=-1.0)
    assert str(excinfo.value) == (
        "value_tolerance must be a finite, non-negative number; got -1.0"
    )


def test_diff_nan_value_tolerance_raises_valueerror():
    baseline = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high")]
    )
    with pytest.raises(ValueError) as excinfo:
        diff_descriptor_terms(baseline, candidate, value_tolerance=math.nan)
    assert str(excinfo.value) == (
        "value_tolerance must be a finite, non-negative number; got nan"
    )


def test_diff_inf_value_tolerance_raises_valueerror():
    baseline = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high")]
    )
    with pytest.raises(ValueError) as excinfo:
        diff_descriptor_terms(baseline, candidate, value_tolerance=math.inf)
    assert str(excinfo.value) == (
        "value_tolerance must be a finite, non-negative number; got inf"
    )


def test_diff_removed_term():
    baseline = _block(
        [
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="high"),
            MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high"),
        ]
    )
    candidate = _block(
        [MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high")]
    )
    assert diff_descriptor_terms(baseline, candidate) == DescriptorTermDiff(
        added=[], removed=["bright"], direction_changed=[], value_drift=[]
    )


def test_diff_ignores_estimated_rows():
    baseline = _block(
        [
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="high"),
            MeasuredDescriptor(
                term="tempo-audio",
                value=120.0,
                metric="tempo_bpm",
                direction="value",
                estimated=True,
            ),
        ]
    )
    candidate = _block(
        [
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="high"),
            MeasuredDescriptor(
                term="tempo-audio",
                value=999.0,
                metric="tempo_bpm",
                direction="value",
                estimated=True,
            ),
        ]
    )
    # Estimated rows are filtered on both sides, so the 999.0 vs 120.0 tempo churn
    # is invisible: no added/removed/direction/drift entries.
    assert diff_descriptor_terms(baseline, candidate) == DescriptorTermDiff(
        added=[], removed=[], direction_changed=[], value_drift=[]
    )


# --- Step 4.2: sorted-output determinism -------------------------------------


def test_diff_output_is_sorted_regardless_of_input_order():
    baseline = _block(
        [
            MeasuredDescriptor(term="spare", value=1.0, metric="o", direction="low"),
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="high"),
            MeasuredDescriptor(term="loud", value=-12.0, metric="r", direction="high"),
        ]
    )
    candidate = _block(
        [
            MeasuredDescriptor(term="loud", value=-9.5, metric="r", direction="high"),
            MeasuredDescriptor(term="dark", value=200.0, metric="c", direction="low"),
            MeasuredDescriptor(term="bright", value=3000.0, metric="c", direction="low"),
            MeasuredDescriptor(term="dense", value=9.0, metric="o", direction="high"),
        ]
    )
    # added   = {dark, dense}         (spare removed)
    # removed = {spare}
    # both    = {bright, loud}; bright high->low changed
    # drift   = loud (2.5 > 0.0); sorted by term
    assert diff_descriptor_terms(baseline, candidate) == DescriptorTermDiff(
        added=["dark", "dense"],
        removed=["spare"],
        direction_changed=["bright"],
        value_drift=[
            ValueDrift(term="loud", baseline_value=-12.0, candidate_value=-9.5)
        ],
    )


# --- Step 4.3: non-finite drift serialization (strict JSON) ------------------


def test_serialize_non_finite_baseline_value_is_null():
    diff = DescriptorTermDiff(
        added=[],
        removed=[],
        direction_changed=[],
        value_drift=[
            ValueDrift(term="loud", baseline_value=float("nan"), candidate_value=-9.5)
        ],
    )
    line = cli._descriptor_diff_json_line(diff)
    assert line == (
        '{"added":[],"removed":[],"direction_changed":[],'
        '"value_drift":[{"term":"loud","baseline_value":null,'
        '"candidate_value":-9.5}]}'
    )
    assert json.loads(line) == {
        "added": [],
        "removed": [],
        "direction_changed": [],
        "value_drift": [
            {"term": "loud", "baseline_value": None, "candidate_value": -9.5}
        ],
    }


def test_serialize_non_finite_candidate_value_is_null():
    diff = DescriptorTermDiff(
        added=[],
        removed=[],
        direction_changed=[],
        value_drift=[
            ValueDrift(term="loud", baseline_value=-12.0, candidate_value=float("inf"))
        ],
    )
    line = cli._descriptor_diff_json_line(diff)
    assert line == (
        '{"added":[],"removed":[],"direction_changed":[],'
        '"value_drift":[{"term":"loud","baseline_value":-12.0,'
        '"candidate_value":null}]}'
    )
    assert json.loads(line) == {
        "added": [],
        "removed": [],
        "direction_changed": [],
        "value_drift": [
            {"term": "loud", "baseline_value": -12.0, "candidate_value": None}
        ],
    }


# --- Step 4.3b: report-load failures (exact .code + detail, BOTH sides) ------


def test_unreadable_baseline(tmp_path, capsys):
    good = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            str(tmp_path / "nope.json"),
            "--candidate",
            good,
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: embeds the platform OSError string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "unreadable",
            "side": "baseline",
            "path": str(tmp_path / "nope.json"),
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_unreadable_candidate(tmp_path, capsys):
    good = _write_report(tmp_path, "baseline.json", _bright_block())
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            good,
            "--candidate",
            str(tmp_path / "nope.json"),
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: embeds the platform OSError string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "unreadable",
            "side": "candidate",
            "path": str(tmp_path / "nope.json"),
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_unparseable_baseline(tmp_path, capsys):
    bad = tmp_path / "baseline.json"
    bad.write_text("{not json", encoding="utf-8")
    good = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        ["iterate-descriptors", "--baseline", str(bad), "--candidate", good]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: embeds the pydantic ValidationError string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "invalid_report",
            "side": "baseline",
            "path": str(bad),
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_unparseable_candidate(tmp_path, capsys):
    good = _write_report(tmp_path, "baseline.json", _bright_block())
    bad = tmp_path / "candidate.json"
    bad.write_text("{not json", encoding="utf-8")
    code = cli.main(
        ["iterate-descriptors", "--baseline", good, "--candidate", str(bad)]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: embeds the pydantic ValidationError string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "invalid_report",
            "side": "candidate",
            "path": str(bad),
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_valid_json_not_a_report_baseline(tmp_path, capsys):
    bad = tmp_path / "baseline.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    good = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        ["iterate-descriptors", "--baseline", str(bad), "--candidate", good]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: pydantic-generated (missing-required-fields) string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "invalid_report",
            "side": "baseline",
            "path": str(bad),
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_valid_json_not_a_report_candidate(tmp_path, capsys):
    good = _write_report(tmp_path, "baseline.json", _bright_block())
    bad = tmp_path / "candidate.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    code = cli.main(
        ["iterate-descriptors", "--baseline", good, "--candidate", str(bad)]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    error = dict(payload["error"])
    del error["message"]  # excluded: pydantic-generated (missing-required-fields) string
    assert error == {
        "code": "DESCRIPTORS_REPORT_INVALID",
        "detail": {
            "reason": "invalid_report",
            "side": "candidate",
            "path": str(bad),
        },
        "severity": "fatal",
        "component": "analyze",
    }


# --- Step 4.3c: missing --baseline/--candidate flag -> typed USAGE error ------
#
# ``--baseline`` / ``--candidate`` are argparse ``default=None`` (not ``required``),
# so an omitted flag reaches ``_load_report`` as ``path=None``. Without the guard,
# ``Path(None)`` raises a bare ``TypeError`` that escapes to the generic handler and
# is misreported as ``INTERNAL_ERROR`` (exit 1) — a sonoscope-bug shape. The guard
# turns it into a clean typed ``USAGE_MISSING_REPORT`` (exit 1, USAGE, cli). Exit
# 1 alone is a green mirage here (INTERNAL_ERROR is ALSO exit 1), so these tests
# pin the exact ``.code``/``.component`` and assert it is NOT the internal fatal.


def test_missing_baseline_flag_is_usage_error(tmp_path, capsys):
    good = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(["iterate-descriptors", "--candidate", good])
    captured = capsys.readouterr()
    assert code == int(ExitCode.USAGE)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_MISSING_REPORT",
        "message": "--baseline is required for iterate-descriptors",
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }
    assert payload["error"]["code"] != "INTERNAL_ERROR"


def test_missing_candidate_flag_is_usage_error(tmp_path, capsys):
    good = _write_report(tmp_path, "baseline.json", _bright_block())
    code = cli.main(["iterate-descriptors", "--baseline", good])
    captured = capsys.readouterr()
    assert code == int(ExitCode.USAGE)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_MISSING_REPORT",
        "message": "--candidate is required for iterate-descriptors",
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }
    assert payload["error"]["code"] != "INTERNAL_ERROR"


# --- Step 4.4: CLI I1 (missing block, both sides) + I2 (end-to-end GREEN) -----


def _bright_block() -> DescriptorsBlock:
    return _block(
        [
            MeasuredDescriptor(
                term="bright",
                value=3200.0,
                metric="spectral_centroid_hz",
                direction="high",
            )
        ]
    )


def test_missing_descriptors_block_baseline(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", None)
    candidate = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "DESCRIPTORS_NO_BLOCK",
        "message": "iterate-descriptors baseline report has no descriptors block",
        "detail": {"reason": "no_descriptors_block", "side": "baseline"},
        "severity": "fatal",
        "component": "analyze",
    }


def test_missing_descriptors_block_candidate(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", _bright_block())
    candidate = _write_report(tmp_path, "candidate.json", None)
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "DESCRIPTORS_NO_BLOCK",
        "message": "iterate-descriptors candidate report has no descriptors block",
        "detail": {"reason": "no_descriptors_block", "side": "candidate"},
        "severity": "fatal",
        "component": "analyze",
    }


def test_iterate_descriptors_end_to_end(tmp_path, capsys):
    baseline = _write_report(
        tmp_path,
        "baseline.json",
        _block(
            [
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="high"
                ),
                MeasuredDescriptor(
                    term="loud", value=-12.0, metric="r", direction="high"
                ),
            ]
        ),
    )
    candidate = _write_report(
        tmp_path,
        "candidate.json",
        _block(
            [
                MeasuredDescriptor(
                    term="loud", value=-9.5, metric="r", direction="high"
                ),
                MeasuredDescriptor(
                    term="dense", value=9.0, metric="o", direction="high"
                ),
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="low"
                ),
            ]
        ),
    )
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.OK)
    assert captured.out == (
        '{"added":["dense"],"removed":[],"direction_changed":["bright"],'
        '"value_drift":[{"term":"loud","baseline_value":-12.0,'
        '"candidate_value":-9.5}]}\n'
    )


# --- PR review: --value-tolerance must be finite + non-negative ---------------
#
# ``--value-tolerance`` is argparse ``type=float, default=0.0`` and flows into
# ``diff_descriptor_terms`` -> ``_value_drifted`` as ``abs(delta) > value_tolerance``.
# A bad flag silently produces WRONG drift output:
#   * negative -> ``abs(delta) > -1`` is ALWAYS true  -> every shared term wrongly
#     reported as drift.
#   * nan/inf  -> ``abs(delta) > nan/inf`` is ALWAYS false -> drift NEVER reported.
# The CLI must reject a non-finite / negative tolerance loud at the boundary
# (exit 1 USAGE, ``USAGE_INVALID_VALUE_TOLERANCE``, component "cli") BEFORE any
# report load/diff. Exit 1 alone is a green mirage (INTERNAL_ERROR is also exit 1),
# so pin the exact ``.code``/``.component``/message and the full error envelope.


def test_negative_value_tolerance_is_usage_error(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", _bright_block())
    candidate = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "-1.0",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.USAGE)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_INVALID_VALUE_TOLERANCE",
        "message": (
            "iterate-descriptors --value-tolerance must be a finite, "
            "non-negative number; got -1.0"
        ),
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }
    assert payload["error"]["code"] != "INTERNAL_ERROR"


def test_nan_value_tolerance_is_usage_error(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", _bright_block())
    candidate = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "nan",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.USAGE)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_INVALID_VALUE_TOLERANCE",
        "message": (
            "iterate-descriptors --value-tolerance must be a finite, "
            "non-negative number; got nan"
        ),
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }
    assert payload["error"]["code"] != "INTERNAL_ERROR"


def test_inf_value_tolerance_is_usage_error(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", _bright_block())
    candidate = _write_report(tmp_path, "candidate.json", _bright_block())
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "inf",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.USAGE)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_INVALID_VALUE_TOLERANCE",
        "message": (
            "iterate-descriptors --value-tolerance must be a finite, "
            "non-negative number; got inf"
        ),
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }
    assert payload["error"]["code"] != "INTERNAL_ERROR"


def test_positive_value_tolerance_suppresses_drift(tmp_path, capsys):
    """GREEN: a valid positive tolerance is accepted AND affects the diff.

    Same reports as the end-to-end test, but ``--value-tolerance 5.0`` bands the
    ``loud`` delta (|-9.5 - -12.0| = 2.5) below tolerance, so it is NOT flagged.
    """
    baseline = _write_report(
        tmp_path,
        "baseline.json",
        _block(
            [
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="high"
                ),
                MeasuredDescriptor(
                    term="loud", value=-12.0, metric="r", direction="high"
                ),
            ]
        ),
    )
    candidate = _write_report(
        tmp_path,
        "candidate.json",
        _block(
            [
                MeasuredDescriptor(
                    term="loud", value=-9.5, metric="r", direction="high"
                ),
                MeasuredDescriptor(
                    term="dense", value=9.0, metric="o", direction="high"
                ),
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="low"
                ),
            ]
        ),
    )
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "5.0",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.OK)
    assert captured.out == (
        '{"added":["dense"],"removed":[],"direction_changed":["bright"],'
        '"value_drift":[]}\n'
    )


# --- Item 4: wav-path (JSON ARRAY) reports are diffed CHUNK-WISE -------------
#
# ``analyze --wav`` emits a WavAnalysisReport (a JSON ARRAY of WavChunkAnalysis,
# one entry per chunk), not a single AnalysisReport. iterate-descriptors accepts
# both shapes: two arrays are compared chunk i vs chunk i and emit the distinct
# ``{"chunk_count":N,"chunks":[...]}`` object; a chunk-count mismatch and a mixed
# array/single pair are typed InputErrors (exit 2), never a partial comparison.


def _wav_chunk(block: DescriptorsBlock, *, chunk_index: int, n_chunks: int):
    """One ``WavChunkAnalysis`` carrying ``block`` (descriptors is REQUIRED there)."""
    return WavChunkAnalysis(
        generated_at="2026-07-09T12:00:00Z",
        sonoscope_version="0.1.0",
        input_provenance=InputProvenance(
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
            chunk_index=chunk_index,
            n_chunks=n_chunks,
        ),
        deterministic=DeterministicBlock.model_validate(
            deepcopy(_BASE_REPORT["deterministic"])
        ),
        descriptors=block,
    )


def _write_wav_report(tmp_path, name: str, blocks: list[DescriptorsBlock]) -> str:
    report = WavAnalysisReport(
        [
            _wav_chunk(b, chunk_index=i, n_chunks=len(blocks))
            for i, b in enumerate(blocks)
        ]
    )
    p = tmp_path / name
    p.write_text(report.model_dump_json(), encoding="utf-8")
    return str(p)


def _bright_loud(bright_direction: str, loud_value: float) -> DescriptorsBlock:
    return _block(
        [
            MeasuredDescriptor(
                term="bright", value=3000.0, metric="c", direction=bright_direction
            ),
            MeasuredDescriptor(
                term="loud", value=loud_value, metric="r", direction="high"
            ),
        ]
    )


def test_wav_array_reports_diff_chunkwise(tmp_path, capsys):
    """RED before the fix: two wav-path reports exit 2 (DESCRIPTORS_REPORT_INVALID).

    After the fix they exit 0 and emit the chunk-wise diff object.
    """
    baseline = _write_wav_report(
        tmp_path,
        "baseline.json",
        [_bright_loud("high", -12.0), _bright_loud("high", -20.0)],
    )
    candidate = _write_wav_report(
        tmp_path,
        "candidate.json",
        [
            _block(
                [
                    MeasuredDescriptor(
                        term="bright", value=3000.0, metric="c", direction="low"
                    ),
                    MeasuredDescriptor(
                        term="loud", value=-9.5, metric="r", direction="high"
                    ),
                    MeasuredDescriptor(
                        term="dense", value=9.0, metric="o", direction="high"
                    ),
                ]
            ),
            _bright_loud("high", -20.0),
        ],
    )
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "0.5",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.OK)
    assert json.loads(captured.out) == {
        "chunk_count": 2,
        "chunks": [
            {
                "chunk_index": 0,
                "added": ["dense"],
                "removed": [],
                "direction_changed": ["bright"],
                "value_drift": [
                    {
                        "term": "loud",
                        "baseline_value": -12.0,
                        "candidate_value": -9.5,
                    }
                ],
            },
            {
                "chunk_index": 1,
                "added": [],
                "removed": [],
                "direction_changed": [],
                "value_drift": [],
            },
        ],
    }


def test_wav_array_reports_chunkwise_line_is_byte_exact(tmp_path, capsys):
    """The chunk-wise line is single-line compact strict JSON (byte-exact)."""
    baseline = _write_wav_report(tmp_path, "baseline.json", [_bright_loud("high", -12.0)])
    candidate = _write_wav_report(tmp_path, "candidate.json", [_bright_loud("low", -12.0)])
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.OK)
    assert captured.out == (
        '{"chunk_count":1,"chunks":[{"chunk_index":0,"added":[],"removed":[],'
        '"direction_changed":["bright"],"value_drift":[]}]}\n'
    )


def test_wav_array_chunk_count_mismatch_is_input_error(tmp_path, capsys):
    baseline = _write_wav_report(
        tmp_path,
        "baseline.json",
        [_bright_loud("high", -12.0), _bright_loud("high", -20.0)],
    )
    candidate = _write_wav_report(tmp_path, "candidate.json", [_bright_loud("high", -12.0)])
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "DESCRIPTORS_CHUNK_COUNT_MISMATCH",
        "message": (
            "iterate-descriptors requires equal chunk counts to diff wav-path "
            "reports chunk-wise; baseline has 2 chunks, candidate has 1"
        ),
        "detail": {
            "reason": "chunk_count_mismatch",
            "baseline_chunks": 2,
            "candidate_chunks": 1,
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_mixed_array_baseline_single_candidate_is_input_error(tmp_path, capsys):
    baseline = _write_wav_report(tmp_path, "baseline.json", [_bright_loud("high", -12.0)])
    candidate = _write_report(tmp_path, "candidate.json", _bright_loud("high", -12.0))
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "DESCRIPTORS_REPORT_SHAPE_MISMATCH",
        "message": (
            "iterate-descriptors cannot compare a wav-path chunk array against a "
            "single analysis report; baseline is wav-chunk-array, candidate is "
            "analysis-report. Produce both reports from the same analyze path."
        ),
        "detail": {
            "reason": "shape_mismatch",
            "baseline_shape": "wav-chunk-array",
            "candidate_shape": "analysis-report",
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_mixed_single_baseline_array_candidate_is_input_error(tmp_path, capsys):
    baseline = _write_report(tmp_path, "baseline.json", _bright_loud("high", -12.0))
    candidate = _write_wav_report(tmp_path, "candidate.json", [_bright_loud("high", -12.0)])
    code = cli.main(
        ["iterate-descriptors", "--baseline", baseline, "--candidate", candidate]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.INPUT)
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "DESCRIPTORS_REPORT_SHAPE_MISMATCH",
        "message": (
            "iterate-descriptors cannot compare a wav-path chunk array against a "
            "single analysis report; baseline is analysis-report, candidate is "
            "wav-chunk-array. Produce both reports from the same analyze path."
        ),
        "detail": {
            "reason": "shape_mismatch",
            "baseline_shape": "analysis-report",
            "candidate_shape": "wav-chunk-array",
        },
        "severity": "fatal",
        "component": "analyze",
    }


def test_single_report_pair_output_unchanged_by_array_support(tmp_path, capsys):
    """GREEN (no regression): the two-single-report path stays byte-identical."""
    baseline = _write_report(
        tmp_path,
        "baseline.json",
        _block(
            [
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="high"
                ),
                MeasuredDescriptor(
                    term="loud", value=-12.0, metric="r", direction="high"
                ),
            ]
        ),
    )
    candidate = _write_report(
        tmp_path,
        "candidate.json",
        _block(
            [
                MeasuredDescriptor(
                    term="loud", value=-9.5, metric="r", direction="high"
                ),
                MeasuredDescriptor(
                    term="dense", value=9.0, metric="o", direction="high"
                ),
                MeasuredDescriptor(
                    term="bright", value=3000.0, metric="c", direction="low"
                ),
            ]
        ),
    )
    code = cli.main(
        [
            "iterate-descriptors",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--value-tolerance",
            "0.5",
        ]
    )
    captured = capsys.readouterr()
    assert code == int(ExitCode.OK)
    assert captured.out == (
        '{"added":["dense"],"removed":[],"direction_changed":["bright"],'
        '"value_drift":[{"term":"loud","baseline_value":-12.0,'
        '"candidate_value":-9.5}]}\n'
    )
