"""midi-analysis schema contract tests (Task S1).

Green-mirage discipline: a fully-populated reference report round-trips by exact
equality; strictness (extra="forbid"), the exact enum literals, the range
validation, and the input-source discriminator are each RED-proven. Also guards
the SCHEMA_VERSION bump, the "midi" Component addition, and the generate/registry
wiring for the new report kind (design section 5).
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from sonoscope.schema.generate import DRAFT_2020_12_URI, json_schema_for
from sonoscope.schema.models import (
    SCHEMA_VERSION,
    ErrorItem,
    MidiAnalysisReport,
    MidiCaptureMeta,
    MidiEvent,
    MidiInputBlock,
)

# --- Reference fixture (mirror design section 5) ----------------------------

_ON_A = {"t_samples": 0, "t_ticks": 0, "type": "note_on",
         "channel": 0, "note": 36, "velocity": 100}
_OFF_A = {"t_samples": 3000, "t_ticks": 120, "type": "note_off",
          "channel": 0, "note": 36, "velocity": 0}
_ON_A_ACTUAL = {"t_samples": 2, "t_ticks": 0, "type": "note_on",
                "channel": 0, "note": 36, "velocity": 100}
_ON_B_EXPECTED = {"t_samples": 6000, "t_ticks": 240, "type": "note_on",
                  "channel": 1, "note": 72, "velocity": 100}
_ON_B_ACTUAL = {"t_samples": 6000, "t_ticks": 240, "type": "note_on",
                "channel": 1, "note": 72, "velocity": 90}

REPORT = {
    "schema_version": "1.1.0",
    "kind": "midi-analysis",
    "generated_at": "2026-07-06T12:00:00Z",
    "sonoscope_version": "0.1.0",
    "input": {
        "source": "plugin",
        "plugin": {
            "path": "/plugins/ReferenceSequencer.clap",
            "binary_sha256": "plugin-sha",
            "plugin_id": "com.example.reference-sequencer",
            "plugin_name": "Reference Sequencer",
        },
        "file": None,
        "transport": {
            "sample_rate": 48000,
            "block_size": 512,
            "tempo_bpm": 120.0,
            "start_position_beats": 0.0,
            "duration_beats": 1.0,
            "tsig_num": 4,
            "tsig_den": 4,
            "playing": True,
        },
        "expected": {
            "ref": "fixtures/midi/correct1.expected.json",
            "spec_sha256": "expected-sha",
            "event_count": 6,
        },
    },
    "midi": {
        "capture_meta": {
            "sample_rate": 48000,
            "block_size": 512,
            "duration_samples": 24000,
            "ppq": 960,
            "tempo_bpm": 120.0,
            "start_position_beats": 0.0,
            "duration_beats": 1.0,
            "tsig_num": 4,
            "tsig_den": 4,
            "plugin_id": "com.example.reference-sequencer",
            "plugin_name": "Reference Sequencer",
            "source": "plugin",
            "binary_sha256": "plugin-sha",
            "events_sha256": "events-sha",
            "block_size_invariant": True,
            "timing_tolerance_samples": 1,
        },
        "events": [_ON_A, _OFF_A, _ON_B_ACTUAL],
        "expected_vs_actual": {
            "matched": 1,
            "missing": [_OFF_A],
            "extra": [_ON_B_ACTUAL],
            "mistimed": [
                {
                    "expected": _ON_A,
                    "actual": _ON_A_ACTUAL,
                    "delta_samples": 2,
                    "delta_ticks": 0,
                }
            ],
            "wrong_field": [
                {
                    "expected": _ON_B_EXPECTED,
                    "actual": _ON_B_ACTUAL,
                    "field": "velocity",
                }
            ],
        },
        "integrity": {
            "every_note_on_has_off": False,
            "stuck_notes": [_ON_B_ACTUAL],
            "dangling_offs": [],
        },
        "verdict": "RED",
        "reasons": ["stuck-note", "wrong-field", "timing"],
    },
    "descriptors": None,
    "errors": [
        {
            "code": "MIDI_CAPTURE_NORMALIZED_RUNNING_STATUS",
            "message": "0x90 velocity 0 normalized to note_off.",
            "detail": {"count": 1},
            "severity": "warning",
            "component": "midi",
        }
    ],
}


# --- Round-trip exact-equality (green mirage) -------------------------------


def test_midi_analysis_report_validates():
    model = MidiAnalysisReport.model_validate(REPORT)
    # JSON round-trip through the serializer reproduces the reference exactly.
    reloaded = MidiAnalysisReport.model_validate_json(model.model_dump_json())
    assert reloaded.model_dump(mode="json") == REPORT


def test_midi_report_without_expected_validates():
    # expected_vs_actual is present only when an expected list was supplied.
    d = deepcopy(REPORT)
    d["midi"]["expected_vs_actual"] = None
    d["input"]["expected"] = None
    assert MidiAnalysisReport.model_validate(d).model_dump(mode="json") == d


# --- Strictness & enum contracts (RED) --------------------------------------


def test_midi_block_extra_forbidden():
    d = deepcopy(REPORT)
    d["midi"]["unexpected"] = 1
    with pytest.raises(ValidationError):
        MidiAnalysisReport.model_validate(d)


def test_midi_verdict_literal():
    d = deepcopy(REPORT)
    d["midi"]["verdict"] = "MAYBE"
    with pytest.raises(ValidationError):
        MidiAnalysisReport.model_validate(d)


def test_midi_event_type_literal():
    d = deepcopy(REPORT)
    d["midi"]["events"][0]["type"] = "aftertouch"
    with pytest.raises(ValidationError):
        MidiAnalysisReport.model_validate(d)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("channel", 16), ("channel", -1), ("note", 128), ("velocity", 200),
        # Lower-bound rejections: note/velocity/t_samples/t_ticks ge=0 were
        # enforced but untested (review gap) — lock the floor.
        ("note", -1), ("velocity", -1), ("t_samples", -1), ("t_ticks", -1),
    ],
)
def test_midi_event_range_validated(field, bad):
    d = deepcopy(_ON_A)
    d[field] = bad
    with pytest.raises(ValidationError):
        MidiEvent.model_validate(d)


# --- capture-meta positive-value contracts (RED) ----------------------------
# Divisor / positive fields in the tick math must reject 0 or negative values,
# so a malformed capture is a hard ValidationError, never silent drift into a
# divide-by-zero downstream (design section 5 capture provenance).


@pytest.mark.parametrize(
    "field,bad",
    [
        ("sample_rate", 0),      # gt=0: divisor in tick math
        ("ppq", 0),              # gt=0: divisor
        ("tsig_num", 0),         # gt=0
        ("tsig_den", 0),         # gt=0: divisor
        ("tempo_bpm", 0.0),      # gt=0: divisor in downstream beat math
        ("block_size", -1),      # ge=0: negative is invalid (0 is a sentinel)
        ("duration_samples", -1),  # ge=0: negative is invalid
    ],
)
def test_midi_capture_meta_positive_validated(field, bad):
    d = deepcopy(REPORT["midi"]["capture_meta"])
    d[field] = bad
    with pytest.raises(ValidationError):
        MidiCaptureMeta.model_validate(d)


def test_midi_capture_meta_block_size_zero_accepted():
    # C1 constraint: the file-source path sets block_size=0 as a deliberate
    # "not-applicable" sentinel (a static .mid file has no processing block
    # size). block_size MUST allow 0 — this locks it against a future tighten
    # to gt=0 that would break the file source.
    d = deepcopy(REPORT["midi"]["capture_meta"])
    d["block_size"] = 0
    assert MidiCaptureMeta.model_validate(d).block_size == 0


def test_midi_input_source_consistency():
    # A plugin source with no plugin ref is a hard error, not silent drift.
    d = deepcopy(REPORT["input"])
    d["plugin"] = None
    with pytest.raises(ValidationError):
        MidiInputBlock.model_validate(d)


def test_midi_input_file_source_consistency():
    # The inverse discriminator side: a file source must carry a file ref and
    # no stray plugin ref. Proves both sides of the discriminated union, not
    # just the plugin side (review gap).
    d = deepcopy(REPORT["input"])
    d["source"] = "file"
    d["plugin"] = None
    d["file"] = {"path": "/fixtures/correct1.mid", "file_sha256": "file-sha"}
    # Consistent file source validates.
    assert MidiInputBlock.model_validate(d).source == "file"
    # A stray plugin ref on a file source is rejected.
    stray = deepcopy(d)
    stray["plugin"] = {
        "path": "/plugins/ReferenceSequencer.clap",
        "binary_sha256": "plugin-sha",
        "plugin_id": "com.example.reference-sequencer",
        "plugin_name": "Reference Sequencer",
    }
    with pytest.raises(ValidationError):
        MidiInputBlock.model_validate(stray)
    # A file source missing its file ref is rejected.
    missing = deepcopy(d)
    missing["file"] = None
    with pytest.raises(ValidationError):
        MidiInputBlock.model_validate(missing)


# --- Version, Component, and registry wiring --------------------------------


def test_schema_version_bumped():
    assert SCHEMA_VERSION == "1.5.0"


def test_component_includes_midi():
    item = ErrorItem.model_validate(
        {
            "code": "MIDI_X",
            "message": "m",
            "detail": None,
            "severity": "error",
            "component": "midi",
        }
    )
    assert item.component == "midi"


def test_component_rejects_unknown():
    with pytest.raises(ValidationError):
        ErrorItem.model_validate(
            {
                "code": "X",
                "message": "m",
                "detail": None,
                "severity": "error",
                "component": "bogus",
            }
        )


def test_generate_midi_analysis_kind():
    schema = json_schema_for("midi-analysis")
    assert schema["$schema"] == DRAFT_2020_12_URI
    assert schema["title"] == "MidiAnalysisReport"
    assert set(schema["properties"].keys()) == set(
        MidiAnalysisReport.model_fields.keys()
    )
    assert schema["properties"]["schema_version"]["default"] == "1.5.0"
