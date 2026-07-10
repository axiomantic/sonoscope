"""C2 analyze_midi descriptors attachment (design §7). Mirrors the plugin-free
tests/test_midi_orchestrator.py::test_file_source_analysis baseline."""
from pathlib import Path

from sonoscope.midi_orchestrator import MidiFileSource, analyze_midi

# repo root: tests/descriptors/<this file> -> parents[2]. (Baseline defines
# CORPUS_MID = _REPO_ROOT / "corpus" / "midi" / "phrase_4note.mid" with
# _REPO_ROOT = parents[1] from tests/. Inlined here to avoid an unprecedented
# `from tests...` cross-module import — confirm the path resolves.)
_CORPUS_MID = Path(__file__).resolve().parents[2] / "corpus" / "midi" / "phrase_4note.mid"


def test_file_source_descriptors_attached():
    report = analyze_midi(MidiFileSource(path=_CORPUS_MID, sample_rate=48000))
    # C2 attachment: exactly the six terms in frozen order. (Accessing
    # .measured on a missing descriptors block would raise AttributeError,
    # so no `is not None` guard is needed — and it is forbidden anyway.)
    assert [m.term for m in report.descriptors.measured] == [
        "note-density", "register", "pitch-range",
        "polyphony", "velocity-dynamics", "ioi",
    ]
    # Pin the exact derived values to guard the analyze_midi ->
    # resolve_window_samples -> derive_midi_descriptors wiring end-to-end as a
    # REGRESSION GUARD. The term-list assertion above survives a wiring
    # regression (wrong window_samples / wrong event list still yields six
    # ordered terms), so without value pins the wiring is untested. NOTE: for a
    # .mid file source, capture_meta.duration_samples = time-of-last-event, so
    # note-density here is file-source-caveated per design §7.1 (this is a
    # regression pin, not a semantic density claim).
    values = {m.term: m.value for m in report.descriptors.measured}
    assert values == {
        "note-density": 2.0,
        "register": 53.75,
        "pitch-range": 12.0,
        "polyphony": 1.0,
        "velocity-dynamics": 0.0,
        "ioi": 0.5,
    }
    assert report.descriptors.library.deriver_version == "midi-1.0.0"
    # Non-regression: fields the deriver must NOT touch stay exactly equal to
    # the test_file_source_analysis baseline.
    assert len(report.midi.events) == 8
    assert report.midi.verdict == "PASS"
    assert report.midi.integrity.every_note_on_has_off is True
    assert report.midi.expected_vs_actual is None
