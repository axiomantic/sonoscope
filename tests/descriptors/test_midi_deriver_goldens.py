"""Populated-block golden (by design) — the exact block handed downstream."""
from sonoscope.descriptors.midi_deriver import derive_midi_descriptors
from sonoscope.schema.models import MidiCaptureMeta, MidiEvent


# INLINE builders (matching repo convention: tests/descriptors/test_deriver_goldens.py
# DUPLICATES its inline builders rather than cross-importing from another test
# module — cross-test-module imports like `from tests.descriptors.test_midi_deriver
# import _ev, _meta` are unprecedented here and may not resolve as a package).
def _ev(t_samples, typ, note, velocity, channel=0):
    return MidiEvent(
        t_samples=t_samples, t_ticks=0, type=typ,
        channel=channel, note=note, velocity=velocity,
    )


def _meta(sample_rate=48000):
    return MidiCaptureMeta(
        sample_rate=sample_rate, block_size=512, duration_samples=0,
        tempo_bpm=120.0, start_position_beats=0.0, duration_beats=0.0,
        tsig_num=4, tsig_den=4, source="file",
    )


def _fixture_11_1_events():
    return [
        _ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 90), _ev(0, "note_on", 67, 100),
        _ev(24000, "note_off", 60, 0), _ev(24000, "note_off", 64, 0),
        _ev(24000, "note_off", 67, 0),
        _ev(48000, "note_on", 72, 40), _ev(72000, "note_off", 72, 0),
        _ev(72000, "note_on", 48, 110), _ev(95000, "note_off", 48, 0),
    ]


def test_populated_block_matches_design_11_1():
    block = derive_midi_descriptors(
        _fixture_11_1_events(), _meta(sample_rate=48000), window_samples=96000
    )
    values = {m.term: m.value for m in block.measured}
    assert values == {
        "note-density": 1.5,
        "register": 62.2,
        "pitch-range": 24.0,
        "polyphony": 3.0,
        "velocity-dynamics": 24.166091947189145,
        "ioi": 0.75,
    }
    assert block.summary == (
        "measured: 1.50 notes/s, note 62.2, 24 st, 3 voices, "
        "velocity std 24.2, 0.750s IOI"
    )
    assert block.library.thresholds_sha256 == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )
    assert block.library.deriver_version == "midi-1.0.0"
