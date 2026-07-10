"""Emit the two sample descriptors blocks (by design)."""
from __future__ import annotations

from pathlib import Path

from sonoscope.descriptors.midi_deriver import derive_midi_descriptors
from sonoscope.schema.models import MidiCaptureMeta, MidiEvent


def _ev(t, typ, note, vel, ch=0):
    return MidiEvent(t_samples=t, t_ticks=0, type=typ, channel=ch, note=note, velocity=vel)


def _meta():
    return MidiCaptureMeta(
        sample_rate=48000, block_size=512, duration_samples=96000,
        tempo_bpm=120.0, start_position_beats=0.0, duration_beats=0.0,
        tsig_num=4, tsig_den=4, source="file",
    )  # required (no-default) fields per models.py L459-470


_POPULATED = [
    _ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 90), _ev(0, "note_on", 67, 100),
    _ev(24000, "note_off", 60, 0), _ev(24000, "note_off", 64, 0),
    _ev(24000, "note_off", 67, 0),
    _ev(48000, "note_on", 72, 40), _ev(72000, "note_off", 72, 0),
    _ev(72000, "note_on", 48, 110), _ev(95000, "note_off", 48, 0),
]


def main() -> None:
    out_dir = Path("docs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    populated = derive_midi_descriptors(_POPULATED, _meta(), window_samples=96000)
    empty = derive_midi_descriptors([], _meta(), window_samples=96000)
    (out_dir / "midi-descriptors-populated.json").write_text(
        populated.model_dump_json(indent=2) + "\n"
    )
    (out_dir / "midi-descriptors-empty.json").write_text(
        empty.model_dump_json(indent=2) + "\n"
    )
    print("wrote docs/samples/midi-descriptors-{populated,empty}.json")


if __name__ == "__main__":
    main()
