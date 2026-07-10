"""C2 MIDI readout render branches (design §8) + C1-goldens-unchanged regression."""
from sonoscope.descriptors.summary import _render_readout, render_summary
from sonoscope.schema.models import MeasuredDescriptor


def _r(term, value):
    return MeasuredDescriptor(
        term=term, value=value, metric="x", direction="value",
        threshold=None, estimated=False, confidence=None,
    )


def test_render_each_midi_term():
    assert _render_readout(_r("note-density", 2.0)) == "2.00 notes/s"
    assert _render_readout(_r("register", 54.0)) == "note 54.0"
    assert _render_readout(_r("pitch-range", 36.0)) == "36 st"
    assert _render_readout(_r("polyphony", 2.0)) == "2 voices"
    assert _render_readout(_r("velocity-dynamics", 12.5)) == "velocity std 12.5"
    assert _render_readout(_r("ioi", 0.125)) == "0.125s IOI"


def test_full_empty_summary_string():
    rows = [
        _r("note-density", 0.0), _r("register", 0.0), _r("pitch-range", 0.0),
        _r("polyphony", 0.0), _r("velocity-dynamics", 0.0), _r("ioi", 0.0),
    ]
    assert render_summary(rows, [], []) == (
        "measured: 0.00 notes/s, note 0.0, 0 st, 0 voices, "
        "velocity std 0.0, 0.000s IOI"
    )


def test_audio_readout_branches_unchanged():
    # C1 regression: audio value-readout terms render byte-identically.
    # tempo-audio renders via builtin round(): f"{round(descriptor.value)} BPM"
    # (summary.py L23) — NOT a f"{v:.0f}" format spec — so 120.4 -> "120 BPM".
    assert _render_readout(_r("tempo-audio", 120.4)) == "120 BPM"
    assert _render_readout(_r("rhythmic-density", 3.2)) == "3.2 onsets/s"
