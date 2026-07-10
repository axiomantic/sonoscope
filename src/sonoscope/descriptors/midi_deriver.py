"""Pure MIDI descriptor deriver (Cycle 2).

Reduces a MidiBlock event list into exactly six threshold-free MEASURED
value-readouts. No I/O, no clock, no RNG: identical inputs -> byte-identical
DescriptorsBlock. See the ratified C2 MIDI descriptor design (development
artifact, not in-repo).
"""
from __future__ import annotations

import statistics

from sonoscope.descriptors.summary import render_summary
from sonoscope.descriptors.thresholds import thresholds_sha256
from sonoscope.descriptors.vocab import MIDI_TERM_ORDER
from sonoscope.schema.models import (
    DescriptorsBlock,
    DescriptorsLibrary,
    MeasuredDescriptor,
    MidiCaptureMeta,
    MidiEvent,
)

MIDI_DERIVER_VERSION: str = "midi-1.0.0"
MIDI_DERIVER_THRESHOLDS: dict[str, int | float | str | bool] = {}

# term -> frozen metric string (ap11). Emission order comes from MIDI_TERM_ORDER.
_MIDI_METRICS: dict[str, str] = {
    "note-density": "notes_per_second",
    "register": "mean_note",
    "pitch-range": "note_span_semitones",
    "polyphony": "max_concurrent_notes",
    "velocity-dynamics": "velocity_std",
    "ioi": "median_ioi_seconds",
}


def _normalize_offvel0(events: list[MidiEvent]) -> list[MidiEvent]:
    """Reclassify note_on velocity==0 as note_off (note_on velocity-0 = note_off convention).
    Per-event transform, never a re-sort (by design)."""
    out: list[MidiEvent] = []
    for e in events:
        if e.type == "note_on" and e.velocity == 0:
            out.append(e.model_copy(update={"type": "note_off"}))
        else:
            out.append(e)
    return out


def _sweep_key(indexed: tuple[int, MidiEvent]) -> tuple[int, int, int]:
    """Total order for both the seed pairing and the concurrency sweep (by design):
    by t_samples, note_off BEFORE note_on at a coincident t, original index as
    the final deterministic tie-break. Sharing ONE key across the seed and the
    sweep is what makes them agree by construction."""
    idx, e = indexed
    return (e.t_samples, 0 if e.type == "note_off" else 1, idx)


def _dangling_off_seed(events: list[MidiEvent]) -> int:
    """Count note_off events whose matching note_on is absent (voices already
    sounding at window start), via per-(channel,note) LIFO pairing under the
    SAME off-before-on order as the sweep (`_sweep_key`, by design).

    Coincident off-before-on convention: a note_off at t closes any voice open
    just before t; an unmatched off implies a voice sounding since window start.
    This is deliberately the INVERSE of E1's on-before-off integrity pairing —
    E1 optimizes for clean zero-length-note pairing, whereas here the seed MUST
    match the sweep's traversal so `running` can never go negative (see
    `_max_concurrent`)."""
    open_notes: dict[tuple[int, int], list[MidiEvent]] = {}
    dangling = 0
    for _idx, e in sorted(enumerate(events), key=_sweep_key):
        key = (e.channel, e.note)
        if e.type == "note_on":
            open_notes.setdefault(key, []).append(e)
        elif e.type == "note_off":
            stack = open_notes.get(key)
            if stack:
                stack.pop()
            else:
                dangling += 1
    return dangling


def _max_concurrent(events: list[MidiEvent]) -> int:
    """Sweep-line peak concurrency (by design). Seeded with the count of dangling
    note_offs (pre-window voices) computed under the SAME `_sweep_key`
    off-before-on order as the sweep, so seed and sweep agree by construction.

    With that consistency, `running` is provably non-negative throughout:
    running = seed + (ons seen) - (offs seen) = (currently-open voices >= 0) +
    (seed - dangling offs seen so far >= 0) >= 0. No negative-dip self-correction
    is needed — a coincident dangling-off + same-(channel,note) retrigger is now
    counted by the seed rather than salvaged mid-sweep."""
    seed = _dangling_off_seed(events)

    running = seed
    peak = seed
    for _idx, e in sorted(enumerate(events), key=_sweep_key):
        if e.type == "note_off":
            running -= 1
        elif e.type == "note_on":
            running += 1
        if running > peak:
            peak = running
    return peak


def derive_midi_descriptors(
    events: list[MidiEvent],
    capture_meta: MidiCaptureMeta,
    window_samples: int,
) -> DescriptorsBlock:
    """Pure MIDI deriver -> DescriptorsBlock with exactly 6 measured
    value-readouts in MIDI_TERM_ORDER. 0.0 sentinels for empty/degenerate."""
    sr = capture_meta.sample_rate  # gt=0, model-guaranteed
    normalized = _normalize_offvel0(events)  # note_on vel==0 -> note_off (by design)
    note_ons = [e for e in normalized if e.type == "note_on"]  # vel>0 post-normalize
    onsets = sorted({e.t_samples for e in note_ons})  # unique onsets (by design)

    window_seconds = window_samples / sr if window_samples > 0 else 0.0
    density = len(onsets) / window_seconds if window_seconds > 0 else 0.0

    register = sum(e.note for e in note_ons) / len(note_ons) if note_ons else 0.0

    if note_ons:
        _notes = [e.note for e in note_ons]
        pitch_range = float(max(_notes) - min(_notes))
    else:
        pitch_range = 0.0

    polyphony = float(_max_concurrent(normalized))

    _vels = [e.velocity for e in note_ons]
    velocity_std = statistics.pstdev(_vels) if _vels else 0.0

    if len(onsets) < 2:
        median_ioi = 0.0
    else:
        _deltas = [(onsets[i + 1] - onsets[i]) / sr for i in range(len(onsets) - 1)]
        median_ioi = statistics.median(_deltas)

    values: dict[str, float] = {term: 0.0 for term in MIDI_TERM_ORDER}
    values["note-density"] = density
    values["register"] = register
    values["pitch-range"] = pitch_range
    values["polyphony"] = polyphony
    values["velocity-dynamics"] = velocity_std
    values["ioi"] = median_ioi

    measured = [
        MeasuredDescriptor(
            term=term,
            value=float(values[term]),
            metric=_MIDI_METRICS[term],
            direction="value",
            threshold=None,
            estimated=False,
            confidence=None,
        )
        for term in MIDI_TERM_ORDER
    ]
    library = DescriptorsLibrary(
        thresholds_sha256=thresholds_sha256(MIDI_DERIVER_THRESHOLDS),
        deriver_version=MIDI_DERIVER_VERSION,
    )
    return DescriptorsBlock(
        measured=measured,
        hybrid=[],
        advisory=[],
        summary=render_summary(measured, [], []),
        library=library,
    )
