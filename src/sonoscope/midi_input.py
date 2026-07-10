"""E2 input loaders + slice utility for the ``analyze-midi`` path.

Three PURE functions that feed the E1 comparator (``features/midi_tripwires``)
from the two v2 MIDI sources, plus the event-stream slice the operator folded
into the plan (``analyze-midi`` accepts EITHER ``--plugin`` capture (B1) OR
``--file X.mid``, with an optional ``--offset/--length`` window):

- :func:`load_expected_events` — parse an expected-event GOLDEN (a JSON list
  of events) into a canonical ``list[MidiEvent]`` (``MidiExpectedSpec``,
  operator-simplified to a bare list).
- :func:`load_midi_file` — decode a standalone ``.mid`` via ``mido`` into a
  canonical ``list[MidiEvent]`` (the v2 file-source addition, by design).
- :func:`apply_slice` — select a half-open ``[offset, offset+length)`` window in
  samples/seconds/beats/ticks and (by default) rebase the slice to 0.

Design invariants honored (by design):

- **Timing model matches B1.** ``t_ticks`` is @960 PPQ via the SAME
  :func:`~sonoscope.backends.midi_capture._ticks_at_960` formula (imported, not
  re-derived, so the axis can never drift). The canonical order matches B1/E1:
  ascending ``t_samples``; at a coincident sample ``note_off`` sorts BEFORE
  ``note_on``.
- **Faithful decode.** A ``.mid`` ``note_on`` with velocity 0 is a common
  note-off convention, but the LOADER stays faithful (like B1): it decodes as
  ``note_on`` velocity 0. Whether that is a contract violation is the E1
  comparator's decision (``offvel0_as_0x90_policy``), not the loader's.
- **Never a silent skip / silent default.** A malformed golden, an
  unreadable/corrupt ``.mid``, or a negative slice bound raises a typed
  :class:`~sonoscope.errors.InputError` (component ``"midi"``, exit 2), never a
  swallowed exception or a fabricated event.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Optional, Union

import mido
from pydantic import ValidationError

from sonoscope.backends.midi_capture import PPQ, _canonical_sort, _ticks_at_960
from sonoscope.errors import InputError
from sonoscope.schema import MidiEvent

_COMPONENT = "midi"

# --- Typed error codes (component == "midi"; all map to InputError -> exit 2) --
#: A ``--expected`` golden is not a list, is missing a required field, or carries
#: an out-of-range value (loader "typed InputError on invalid").
MIDI_EXPECTED_SPEC_INVALID = "MIDI_EXPECTED_SPEC_INVALID"
#: A ``--file`` ``.mid`` is missing/unreadable/corrupt (by design, file source).
MIDI_FILE_INVALID = "MIDI_FILE_INVALID"
#: An ``--offset/--length`` slice bound is negative (a nonsensical window).
MIDI_SLICE_INVALID = "MIDI_SLICE_INVALID"

#: Default file tempo when a ``.mid`` carries no ``set_tempo`` meta and the caller
#: supplies no override (MIDI standard default = 120 BPM = 500000 us/beat).
_DEFAULT_TEMPO_US_PER_BEAT = 500_000

#: The slice unit domain (design: samples is the canonical timing axis; the other
#: three are conveniences converted to samples for the window comparison).
SliceUnit = Literal["samples", "seconds", "beats", "ticks"]


# --- 1. expected-event golden loader -----------------------------------------


def load_expected_events(
    source: Union[str, Path, list[Any]],
    *,
    sample_rate: Optional[int] = None,
    tempo_bpm: Optional[float] = None,
) -> list[MidiEvent]:
    """Load an expected-event golden into a canonical ``list[MidiEvent]``.

    ``source`` is EITHER a path to a JSON file OR an already-deserialized object.
    The content MUST be a list of event objects, each::

        {"t_samples": int?, "t_ticks": int?, "type": "note_on"|"note_off",
         "channel": int, "note": int, "velocity": int}

    ``t_samples`` and/or ``t_ticks`` (at least one) is required per event. The E1
    comparator gates on ``t_samples``, so the real goldens (``MidiExpectedSpec``)
    carry BOTH axes and are used verbatim. When exactly one
    axis is given, the other is DERIVED via the B1 960-PPQ formula — which needs
    ``sample_rate`` and ``tempo_bpm``; a one-sided event without those params is a
    typed error (we never fabricate a timing axis the comparator reads).

    Every field range is validated by constructing a :class:`MidiEvent` (channel
    0-15, note/velocity 0-127); any violation, a missing field, a non-list
    top-level, or an unreadable file raises :class:`InputError`
    (``MIDI_EXPECTED_SPEC_INVALID``, component ``"midi"``, exit 2) — NEVER a
    silent skip. The result is canonical-ordered (ascending ``t_samples``;
    ``note_off`` before ``note_on`` at a coincident sample) to match B1/E1.
    """
    raw = _load_expected_source(source)
    if not isinstance(raw, list):
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            "expected-event golden must be a JSON list of events, "
            f"got {type(raw).__name__}",
            detail={"reason": "not_a_list", "got_type": type(raw).__name__},
            component=_COMPONENT,
        )

    events = [
        _expected_event_from_obj(obj, i, sample_rate, tempo_bpm)
        for i, obj in enumerate(raw)
    ]
    return _canonical_sort(events)


def _load_expected_source(source: Union[str, Path, list[Any]]) -> Any:
    """Return the deserialized golden: read+parse a path, else pass through."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InputError(
                MIDI_EXPECTED_SPEC_INVALID,
                f"expected-event golden not readable at {path}",
                detail={"reason": "unreadable", "path": str(path)},
                component=_COMPONENT,
            ) from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(
                MIDI_EXPECTED_SPEC_INVALID,
                f"expected-event golden at {path} is not valid JSON: {exc}",
                detail={"reason": "bad_json", "path": str(path)},
                component=_COMPONENT,
            ) from exc
    return source


def _expected_event_from_obj(
    obj: Any,
    index: int,
    sample_rate: Optional[int],
    tempo_bpm: Optional[float],
) -> MidiEvent:
    """Validate one golden event object into a :class:`MidiEvent`."""
    if not isinstance(obj, dict):
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            f"expected event #{index} is not an object",
            detail={"reason": "event_not_object", "index": index},
            component=_COMPONENT,
        )

    t_samples, t_ticks = _resolve_axes(obj, index, sample_rate, tempo_bpm)

    # Required non-timing fields; a missing key is a hard error (never defaulted).
    missing = [k for k in ("type", "channel", "note", "velocity") if k not in obj]
    if missing:
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            f"expected event #{index} missing field(s): {', '.join(missing)}",
            detail={"reason": "missing_field", "index": index, "missing": missing},
            component=_COMPONENT,
        )

    # Range/enum validation lives in the MidiEvent model; a violation (bad channel,
    # note>127, unknown type, ...) surfaces as a typed InputError, not a raw
    # pydantic ValidationError.
    try:
        return MidiEvent(
            t_samples=t_samples,
            t_ticks=t_ticks,
            type=obj["type"],
            channel=obj["channel"],
            note=obj["note"],
            velocity=obj["velocity"],
        )
    except ValidationError as exc:
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            f"expected event #{index} has invalid field(s): {exc}",
            detail={"reason": "invalid_field", "index": index},
            component=_COMPONENT,
        ) from exc


def _resolve_axes(
    obj: dict[str, Any],
    index: int,
    sample_rate: Optional[int],
    tempo_bpm: Optional[float],
) -> tuple[int, int]:
    """Resolve ``(t_samples, t_ticks)`` for one golden event.

    Both given -> used verbatim (the authoritative golden). Exactly one given ->
    the other is derived via the 960-PPQ formula, which requires ``sample_rate``
    and ``tempo_bpm``. Neither given, or one given without conversion params, is a
    typed error — the comparator's timing axis is never fabricated.
    """
    has_s = "t_samples" in obj
    has_t = "t_ticks" in obj
    if not has_s and not has_t:
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            f"expected event #{index} has neither t_samples nor t_ticks",
            detail={"reason": "no_timing_axis", "index": index},
            component=_COMPONENT,
        )
    if has_s and has_t:
        return obj["t_samples"], obj["t_ticks"]

    if sample_rate is None or tempo_bpm is None:
        raise InputError(
            MIDI_EXPECTED_SPEC_INVALID,
            f"expected event #{index} gives only "
            f"{'t_samples' if has_s else 't_ticks'}; supply sample_rate and "
            "tempo_bpm so the other timing axis can be derived",
            detail={"reason": "one_axis_no_params", "index": index},
            component=_COMPONENT,
        )
    if has_s:
        # t_samples given -> derive t_ticks@960 (B1's exact formula).
        return obj["t_samples"], _ticks_at_960(obj["t_samples"], sample_rate, tempo_bpm)
    # t_ticks given -> invert the formula to the sample axis (the gated one).
    return _samples_from_ticks(obj["t_ticks"], sample_rate, tempo_bpm), obj["t_ticks"]


# --- 2. standalone .mid loader -----------------------------------------------


def load_midi_file(
    path: Union[str, Path],
    *,
    sample_rate: int,
    tempo_bpm: Optional[float] = None,
) -> list[MidiEvent]:
    """Decode a standalone ``.mid`` into a canonical ``list[MidiEvent]``.

    Merges all tracks (``mido.merge_tracks``) and walks the merged stream
    accumulating BOTH absolute ticks and absolute seconds, so each note event
    carries:

    - ``t_ticks`` @960 PPQ = the file's absolute ticks RESCALED from the file's
      own ``ticks_per_beat`` (a tempo-independent musical-time conversion), and
    - ``t_samples`` = absolute seconds x ``sample_rate``, where seconds accrue via
      ``mido.tick2second`` under the tempo in effect for each delta.

    Tempo source: ``tempo_bpm`` (if given) OVERRIDES the file and is held constant
    (file ``set_tempo`` metas are ignored). Otherwise the file's ``set_tempo``
    metas drive a running tempo; a file with none defaults to 120 BPM (the MIDI
    standard default; documented here since a bare ``list[MidiEvent]`` has no
    channel to surface a runtime note).

    Decode is FAITHFUL (like B1): ``note_off`` -> ``note_off``; ``note_on`` ->
    ``note_on`` INCLUDING velocity 0 (the common ``note_on`` vel-0 note-off
    convention is NOT normalized here — that is the E1 comparator's policy).
    Non-note messages are skipped. A missing/unreadable/corrupt file raises
    :class:`InputError` (``MIDI_FILE_INVALID``, component ``"midi"``, exit 2),
    never a raw ``mido``/OS exception. The result is canonical-ordered to match
    B1/E1.
    """
    mid = _open_midi_file(path)
    ticks_per_beat = mid.ticks_per_beat
    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        # SMPTE-format division (negative ticks_per_beat) is out of scope; a
        # non-PPQ file is a typed error, not a divide-by-zero crash.
        raise InputError(
            MIDI_FILE_INVALID,
            f"{path} uses a non-PPQ ticks_per_beat ({ticks_per_beat!r}); "
            "only PPQ-timed .mid files are supported",
            detail={"reason": "non_ppq", "ticks_per_beat": ticks_per_beat},
            component=_COMPONENT,
        )

    override = tempo_bpm is not None
    current_tempo = (
        mido.bpm2tempo(tempo_bpm) if override else _DEFAULT_TEMPO_US_PER_BEAT
    )

    abs_ticks = 0
    abs_seconds = 0.0
    events: list[MidiEvent] = []
    try:
        for msg in mido.merge_tracks(mid.tracks):
            # Advance time by THIS delta under the currently-effective tempo BEFORE
            # applying a tempo change (a set_tempo's own delta runs at the prior
            # tempo).
            abs_ticks += msg.time
            abs_seconds += mido.tick2second(msg.time, ticks_per_beat, current_tempo)

            if msg.type == "set_tempo" and not override:
                current_tempo = msg.tempo
                continue

            if msg.type == "note_on":
                event_type = "note_on"  # faithful: vel-0 stays note_on
            elif msg.type == "note_off":
                event_type = "note_off"
            else:
                continue  # non-note message: skipped (documented)

            events.append(
                MidiEvent(
                    t_samples=round(abs_seconds * sample_rate),
                    t_ticks=round(abs_ticks * PPQ / ticks_per_beat),
                    type=event_type,
                    channel=msg.channel,
                    note=msg.note,
                    velocity=msg.velocity,
                )
            )
    except (EOFError, OSError, ValueError) as exc:
        # A lazily-surfaced parse fault mid-walk is still a corrupt-file error.
        raise InputError(
            MIDI_FILE_INVALID,
            f"{path} is a corrupt/unparseable .mid: {exc}",
            detail={"reason": "corrupt", "path": str(path)},
            component=_COMPONENT,
        ) from exc

    return _canonical_sort(events)


def _open_midi_file(path: Union[str, Path]) -> mido.MidiFile:
    """Open a ``.mid`` via ``mido``; map any failure to a typed InputError."""
    try:
        return mido.MidiFile(str(path))
    except (EOFError, OSError, ValueError) as exc:
        raise InputError(
            MIDI_FILE_INVALID,
            f".mid file not readable/parseable at {path}: {exc}",
            detail={"reason": "open_failed", "path": str(path)},
            component=_COMPONENT,
        ) from exc


# --- 3. slice utility --------------------------------------------------------


def apply_slice(
    events: list[MidiEvent],
    *,
    offset: float,
    length: Optional[float],
    unit: SliceUnit,
    sample_rate: int,
    tempo_bpm: float,
    rebase: bool = True,
) -> list[MidiEvent]:
    """Select the half-open window ``[offset, offset+length)`` on an event stream.

    The bound is expressed in ``unit`` and converted to SAMPLES (the canonical
    timing axis) for the comparison: ``samples`` as-is; ``seconds`` x
    ``sample_rate``; ``beats`` -> samples via ``tempo_bpm`` + ``sample_rate``;
    ``ticks`` @960 -> samples via the inverse 960-PPQ formula. An event is kept
    when ``offset_samples <= t_samples < end_samples`` — strictly HALF-OPEN, so an
    event AT ``end_samples`` is excluded (matches B1's window convention).
    ``length is None`` selects to the end of the stream.

    ``rebase`` (default ``True``) shifts the kept slice to its own 0-origin:
    ``offset_samples`` is subtracted from each ``t_samples`` and ``t_ticks`` is
    recomputed @960 from the shifted sample position, so the slice is analyzed as
    a self-contained span (the operator's intent — analyze a sub-window as if it
    were the whole capture). ``rebase=False`` keeps absolute times.

    A negative OR non-finite (NaN/inf) ``offset`` or ``length`` raises
    :class:`InputError` (``MIDI_SLICE_INVALID``, component ``"midi"``, exit 2).
    """
    # Fix 2: reject non-finite bounds BEFORE the sign check — ``NaN < 0`` and
    # ``inf < 0`` are both False, so a NaN/inf would slip past the sign guard and
    # blow up later in ``round()`` as a raw ValueError/OverflowError. Guard here so
    # the failure stays the typed MIDI_SLICE_INVALID.
    if not math.isfinite(offset):
        raise InputError(
            MIDI_SLICE_INVALID,
            f"slice offset must be finite, got {offset}",
            detail={"reason": "non_finite_offset", "offset": offset},
            component=_COMPONENT,
        )
    if offset < 0:
        raise InputError(
            MIDI_SLICE_INVALID,
            f"slice offset must be >= 0, got {offset}",
            detail={"reason": "negative_offset", "offset": offset},
            component=_COMPONENT,
        )
    if length is not None and not math.isfinite(length):
        raise InputError(
            MIDI_SLICE_INVALID,
            f"slice length must be finite, got {length}",
            detail={"reason": "non_finite_length", "length": length},
            component=_COMPONENT,
        )
    if length is not None and length < 0:
        raise InputError(
            MIDI_SLICE_INVALID,
            f"slice length must be >= 0, got {length}",
            detail={"reason": "negative_length", "length": length},
            component=_COMPONENT,
        )

    offset_samples = _unit_to_samples(offset, unit, sample_rate, tempo_bpm)
    end_samples: Optional[int] = None
    if length is not None:
        end_samples = offset_samples + _unit_to_samples(
            length, unit, sample_rate, tempo_bpm
        )

    kept = [
        e
        for e in events
        if e.t_samples >= offset_samples
        and (end_samples is None or e.t_samples < end_samples)
    ]

    if not rebase:
        return _canonical_sort(kept)

    rebased = [
        e.model_copy(
            update={
                "t_samples": e.t_samples - offset_samples,
                "t_ticks": _ticks_at_960(
                    e.t_samples - offset_samples, sample_rate, tempo_bpm
                ),
            }
        )
        for e in kept
    ]
    return _canonical_sort(rebased)


def _unit_to_samples(
    value: float, unit: SliceUnit, sample_rate: int, tempo_bpm: float
) -> int:
    """Convert a slice bound in ``unit`` to an integer sample count."""
    if unit == "samples":
        return round(value)
    if unit == "seconds":
        return round(value * sample_rate)
    if unit == "beats":
        # beats -> seconds (60/tempo per beat) -> samples.
        return round(value * (60.0 / tempo_bpm) * sample_rate)
    # ticks @960 -> samples (inverse of _ticks_at_960).
    return _samples_from_ticks(value, sample_rate, tempo_bpm)


def resolve_window_samples(
    offset: Optional[float],
    length: Optional[float],
    unit: Optional[SliceUnit],
    *,
    sample_rate: int,
    tempo_bpm: float,
    full_duration_samples: int,
) -> int:
    """Analysis-window length in samples for the descriptor density axis (by design).

    No slice (offset is None) -> the full-capture duration.
    Sliced, length set        -> the slice length in samples.
    Sliced, length None       -> offset to the full-capture end (full - offset).

    Reuses :func:`_unit_to_samples` (same module) so the density axis cannot
    drift from the slice axis, and takes PRIMITIVE args so this module never
    back-imports ``MidiSlice`` (dependency direction stays
    ``midi_orchestrator -> midi_input``).
    """
    if offset is None:
        return full_duration_samples
    if unit is None:
        raise ValueError("unit must be provided when offset is set")
    if length is not None:
        return _unit_to_samples(length, unit, sample_rate, tempo_bpm)
    offset_samples = _unit_to_samples(offset, unit, sample_rate, tempo_bpm)
    return max(0, full_duration_samples - offset_samples)


# --- shared helpers ----------------------------------------------------------


def _samples_from_ticks(t_ticks: float, sample_rate: int, tempo_bpm: float) -> int:
    """Inverse of :func:`_ticks_at_960`: ``t_ticks`` @960 PPQ -> ``t_samples``."""
    return round(t_ticks / PPQ * (60.0 / tempo_bpm) * sample_rate)
