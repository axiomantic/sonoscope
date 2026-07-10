"""``analyze_midi`` green-mirage tests (NON-integration).

These tests exercise the full
``analyze_midi`` pipeline WITHOUT a real plugin/host by injecting a FAKE
:class:`~sonoscope.backends.midi_capture.MidiCaptureBackend` that returns a canned
event list. They cover both input sources, the slice, the per-source ``offvel0``
default, a RED-proving broken capture, and error propagation. The DEMO golden and
the CORRECT#1 expected fixture (``specs/refseq_demo_correct1.json``) drive the
schema-valid PASS path.
"""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from sonoscope.backends.midi_capture import (
    MidiCaptureRequest,
    MidiCaptureResult,
)
from sonoscope.errors import MidiCaptureError
from sonoscope.features.midi_tripwires import OFFVEL0_RED_REASON, STUCK_NOTE_ID
from sonoscope.midi_orchestrator import (
    MidiFileSource,
    MidiSlice,
    analyze_midi,
)
from sonoscope.schema import MidiAnalysisReport, MidiCaptureMeta, MidiEvent

_REPO_ROOT = Path(__file__).resolve().parents[1]
#: The CORRECT#1 golden expected fixture (the canonical DEMO event list).
CORRECT1_GOLDEN = _REPO_ROOT / "specs" / "refseq_demo_correct1.json"
#: A clean corpus ``.mid`` (4 note pairs, explicit note_off) for the file source.
CORPUS_MID = _REPO_ROOT / "corpus" / "midi" / "phrase_4note.mid"


# --- the DEMO golden events (the canonical 8-event CORRECT#1 list) ----------

#: The exact CORRECT#1 capture: DEMO @120bpm / start0 / 1 beat / 48 kHz, in the
#: canonical order (ascending t_samples; note_off before note_on at a coincident
#: sample). Mirrors ``specs/refseq_demo_correct1.json``.
DEMO_GOLDEN_EVENTS: list[MidiEvent] = [
    MidiEvent(t_samples=0, t_ticks=0, type="note_on", channel=0, note=36, velocity=100),
    MidiEvent(t_samples=6000, t_ticks=240, type="note_off", channel=0, note=36, velocity=0),
    MidiEvent(t_samples=6000, t_ticks=240, type="note_on", channel=1, note=72, velocity=100),
    MidiEvent(t_samples=9000, t_ticks=360, type="note_off", channel=1, note=72, velocity=0),
    MidiEvent(t_samples=12000, t_ticks=480, type="note_on", channel=0, note=36, velocity=100),
    MidiEvent(t_samples=18000, t_ticks=720, type="note_off", channel=0, note=36, velocity=0),
    MidiEvent(t_samples=18000, t_ticks=720, type="note_on", channel=1, note=72, velocity=100),
    MidiEvent(t_samples=21000, t_ticks=840, type="note_off", channel=1, note=72, velocity=0),
]


def _demo_request(plugin_path: str = "/fake/ReferenceSequencer.clap") -> MidiCaptureRequest:
    """The DEMO transport/render request (@120 / start0 / 1 beat / 48 kHz)."""
    return MidiCaptureRequest(
        plugin_path=Path(plugin_path),
        tempo_bpm=120.0,
        start_position_beats=0.0,
        duration_beats=1.0,
        tsig_num=4,
        tsig_den=4,
        sample_rate=48000,
        block_size=512,
        plugin_id="com.example.reference-sequencer",
    )


class _FakeMidiCaptureBackend:
    """A FAKE B1 backend: returns canned events (or raises), no plugin/host.

    Satisfies the ``analyze_midi`` capture Protocol. It builds a plausible
    ``source="plugin"`` :class:`MidiCaptureMeta` from the request so the assembled
    report is real; the events are whatever the test canned.
    """

    def __init__(
        self,
        events: list[MidiEvent],
        *,
        raises: MidiCaptureError | None = None,
    ) -> None:
        self._events = events
        self._raises = raises

    def capture(self, req: MidiCaptureRequest) -> MidiCaptureResult:
        if self._raises is not None:
            raise self._raises
        duration_samples = round(
            req.duration_beats * (60.0 / req.tempo_bpm) * req.sample_rate
        )
        meta = MidiCaptureMeta(
            sample_rate=req.sample_rate,
            block_size=req.block_size,
            duration_samples=duration_samples,
            tempo_bpm=req.tempo_bpm,
            start_position_beats=req.start_position_beats,
            duration_beats=req.duration_beats,
            tsig_num=req.tsig_num,
            tsig_den=req.tsig_den,
            plugin_id=req.plugin_id,
            plugin_name="Reference Sequencer",
            source="plugin",
            binary_sha256="fake-binary-sha256",
            events_sha256="fake-events-sha256",
        )
        return MidiCaptureResult(events=list(self._events), meta=meta)


# --- tests -------------------------------------------------------------------


def test_report_validates_against_schema() -> None:
    """Fake plugin capture of the golden vs the CORRECT#1 golden -> PASS report.

    A full :class:`MidiAnalysisReport` that round-trips against the S1 model:
    verdict PASS, matched==8, integrity clean.
    """
    backend = _FakeMidiCaptureBackend(DEMO_GOLDEN_EVENTS)
    report = analyze_midi(
        _demo_request(), backend=backend, expected=CORRECT1_GOLDEN
    )

    # Round-trips against the S1 model (schema-valid).
    reloaded = MidiAnalysisReport.model_validate_json(report.model_dump_json())
    assert reloaded == report

    assert report.kind == "midi-analysis"
    assert report.input.source == "plugin"
    assert report.input.plugin is not None
    assert report.input.file is None
    assert report.input.expected is not None
    assert report.input.expected.event_count == 8
    assert report.midi.capture_meta.source == "plugin"
    assert len(report.midi.events) == 8
    assert report.midi.verdict == "PASS"
    assert report.midi.reasons == []
    assert report.midi.expected_vs_actual is not None
    assert report.midi.expected_vs_actual.matched == 8
    assert report.midi.expected_vs_actual.missing == []
    assert report.midi.expected_vs_actual.extra == []
    assert report.midi.expected_vs_actual.mistimed == []
    assert report.midi.expected_vs_actual.wrong_field == []
    assert report.midi.integrity.every_note_on_has_off is True
    assert report.midi.integrity.stuck_notes == []
    assert report.midi.integrity.dangling_offs == []


def test_broken_capture_is_red() -> None:
    """Fake capture that DROPS a note_off vs the golden -> RED, stuck-note.

    End-to-end RED proof: the missing note_off leaves note_on ch0 n36 @0 with no
    matching off in the window -> the #1 stuck-note firewall trips.
    """
    mutated = [
        e
        for e in DEMO_GOLDEN_EVENTS
        # Drop the FIRST note_off (ch0 n36 @6000), leaving on@0 unclosed.
        if not (e.type == "note_off" and e.channel == 0 and e.note == 36 and e.t_samples == 6000)
    ]
    backend = _FakeMidiCaptureBackend(mutated)
    report = analyze_midi(
        _demo_request(), backend=backend, expected=CORRECT1_GOLDEN
    )

    assert report.midi.verdict == "RED"
    # Round-trips even in the RED state (a valid, gate-able report).
    assert MidiAnalysisReport.model_validate_json(report.model_dump_json()) == report
    # stuck-note is the #1 priority wire -> first reason.
    assert report.midi.reasons[0].startswith(f"{STUCK_NOTE_ID}:")
    assert report.midi.integrity.every_note_on_has_off is False
    stuck = report.midi.integrity.stuck_notes
    assert len(stuck) == 1
    assert (stuck[0].channel, stuck[0].note, stuck[0].t_samples) == (0, 36, 0)


def test_file_source_analysis() -> None:
    """File source (a corpus ``.mid``), no expected -> valid report, no diff."""
    report = analyze_midi(
        MidiFileSource(path=CORPUS_MID, sample_rate=48000)
    )

    assert MidiAnalysisReport.model_validate_json(report.model_dump_json()) == report
    assert report.input.source == "file"
    assert report.input.file is not None
    assert report.input.file.path == str(CORPUS_MID)
    assert report.input.file.file_sha256 is not None
    assert report.input.plugin is None
    assert report.input.expected is None
    assert report.midi.capture_meta.source == "file"
    # No capture repeats for a static file (by design).
    assert report.midi.capture_meta.events_sha256 is None
    assert report.midi.capture_meta.block_size_invariant is None
    assert len(report.midi.events) == 8  # 4 clean note pairs
    assert report.midi.expected_vs_actual is None  # no expected supplied
    assert report.midi.integrity.every_note_on_has_off is True
    assert report.midi.verdict == "PASS"


def test_slice_applied() -> None:
    """A slice restricts analysis to the in-window events (half-open + rebase).

    The window ``[6000, 12000)`` samples keeps only the middle three events
    (dropping the @0 and the @12000+ events); rebasing shifts them to a 0-origin.
    """
    backend = _FakeMidiCaptureBackend(DEMO_GOLDEN_EVENTS)
    report = analyze_midi(
        _demo_request(),
        backend=backend,
        slice_spec=MidiSlice(offset=6000, length=6000, unit="samples"),
    )

    got = [
        (e.type, e.channel, e.note, e.t_samples) for e in report.midi.events
    ]
    assert got == [
        ("note_off", 0, 36, 0),  # was @6000, rebased to 0
        ("note_on", 1, 72, 0),  # was @6000, rebased to 0
        ("note_off", 1, 72, 3000),  # was @9000, rebased to 3000
    ]


def test_slice_and_expected_same_frame() -> None:
    """Slice + expected must be compared in the SAME (windowed+rebased) frame.

    The expected spec is authored in FULL-capture coordinates (here the SAME
    golden 8 events the FAKE backend returns). Applying the window
    ``[6000, 12000)`` (rebase) to the CAPTURED events windows+rebases them to
    three 0-origin events; the fix applies the IDENTICAL slice to the EXPECTED
    events before ``evaluate_midi`` so both sides share one frame. The in-window
    captured vs in-window expected diff must therefore be a clean match: verdict
    PASS, ``matched == 3`` (the in-window count), missing/extra empty.

    On the pre-fix code the expected list is NOT sliced, so three rebased
    captured events are diffed against the full 8 absolute-time expected events
    -> spurious ``missing``/``mistimed`` and a RED verdict (the C1 bug).
    """
    backend = _FakeMidiCaptureBackend(DEMO_GOLDEN_EVENTS)
    report = analyze_midi(
        _demo_request(),
        backend=backend,
        expected=CORRECT1_GOLDEN,  # the SAME 8 events, authored in FULL coords
        slice_spec=MidiSlice(offset=6000, length=6000, unit="samples"),
    )

    # midi.events are the analyzed (rebased) window: three 0-origin events.
    assert len(report.midi.events) == 3
    # The expected ref event_count describes the FULL authored spec (provenance).
    assert report.input.expected is not None
    assert report.input.expected.event_count == 8
    # Both sides sliced into the same frame -> a clean in-window match.
    assert report.midi.expected_vs_actual is not None
    assert report.midi.expected_vs_actual.matched == 3
    assert report.midi.expected_vs_actual.missing == []
    assert report.midi.expected_vs_actual.extra == []
    assert report.midi.expected_vs_actual.mistimed == []
    assert report.midi.expected_vs_actual.wrong_field == []
    assert report.midi.verdict == "PASS"


def test_offvel0_default_red_for_plugin() -> None:
    """Plugin source: a ``note_on`` vel-0 trips the offvel0 contract by default."""
    events = [
        MidiEvent(t_samples=0, t_ticks=0, type="note_on", channel=0, note=36, velocity=100),
        # A 0x90 vel-0 note-off (contract violation for Reference Sequencer).
        MidiEvent(t_samples=6000, t_ticks=240, type="note_on", channel=0, note=36, velocity=0),
    ]
    backend = _FakeMidiCaptureBackend(events)
    report = analyze_midi(_demo_request(), backend=backend)  # no policy override

    assert report.midi.verdict == "RED"
    assert OFFVEL0_RED_REASON in report.midi.reasons


def test_offvel0_default_normalize_for_file(tmp_path: Path) -> None:
    """File source: a ``0x90 vel0`` note-off is normalized by default (not RED)."""
    mid = mido.MidiFile(ticks_per_beat=960)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", channel=0, note=60, velocity=100, time=0))
    # A note_on velocity 0 = the standard running-status note-off convention.
    track.append(mido.Message("note_on", channel=0, note=60, velocity=0, time=480))
    mid.tracks.append(track)
    mid_path = tmp_path / "vel0_off.mid"
    mid.save(str(mid_path))

    report = analyze_midi(
        MidiFileSource(path=mid_path, sample_rate=48000, tempo_bpm=120.0)
    )  # no policy override -> file default "normalize"

    assert report.midi.verdict == "PASS"
    assert report.midi.integrity.every_note_on_has_off is True
    # The normalize path records an informational note (not a RED reason).
    assert any(r.startswith("offvel0-normalized") for r in report.midi.reasons)


def test_capture_error_propagates() -> None:
    """A backend capture failure surfaces as MidiCaptureError, never swallowed."""
    err = MidiCaptureError(
        "MIDI_CAPTURE_SUBPROCESS_CRASH",
        "clap_midi_host exited with signal SIGSEGV.",
        component="midi",
    )
    backend = _FakeMidiCaptureBackend(DEMO_GOLDEN_EVENTS, raises=err)

    with pytest.raises(MidiCaptureError) as excinfo:
        analyze_midi(_demo_request(), backend=backend)
    assert excinfo.value is err
