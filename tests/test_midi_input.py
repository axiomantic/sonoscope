"""E2 unit tests — expected-golden + .mid loaders and the slice utility.

Green-mirage discipline: exact-equality assertions on the produced
``MidiEvent`` fields, and every error path RED-proving (a typed
:class:`InputError` with the exact code/exit/component — NEVER a silent skip or
a fabricated default). Corpus ``.mid`` content is asserted against the known
generator constants (scripts/generate_corpus.py: 480 PPQ, 120 BPM, C3 = note
48, velocity 100, 2 s sustain @ 48 kHz).
"""

from __future__ import annotations

from pathlib import Path

import mido
import pytest
from pydantic import BaseModel, ValidationError

from sonoscope.errors import InputError
from sonoscope.midi_input import (
    MIDI_EXPECTED_SPEC_INVALID,
    MIDI_FILE_INVALID,
    MIDI_SLICE_INVALID,
    apply_slice,
    load_expected_events,
    load_midi_file,
)
from sonoscope.schema import ExitCode, MidiEvent

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_MIDI = _REPO_ROOT / "corpus" / "midi"

_SR = 48_000
_TEMPO = 120.0


def _ev(
    t_samples: int,
    t_ticks: int,
    *,
    type: str = "note_on",  # noqa: A002 - mirrors the schema field name
    channel: int = 0,
    note: int = 60,
    velocity: int = 100,
) -> MidiEvent:
    return MidiEvent(
        t_samples=t_samples,
        t_ticks=t_ticks,
        type=type,
        channel=channel,
        note=note,
        velocity=velocity,
    )


# --- 1. load_expected_events -------------------------------------------------


def test_load_expected_valid() -> None:
    """A both-axes golden loads verbatim into exact MidiEvents (canonical order)."""
    spec = [
        {
            "t_samples": 0,
            "t_ticks": 0,
            "type": "note_on",
            "channel": 0,
            "note": 48,
            "velocity": 100,
        },
        {
            "t_samples": 96_000,
            "t_ticks": 3_840,
            "type": "note_off",
            "channel": 0,
            "note": 48,
            "velocity": 0,
        },
    ]

    events = load_expected_events(spec)

    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=48, velocity=100),
        _ev(96_000, 3_840, type="note_off", channel=0, note=48, velocity=0),
    ]


def test_load_expected_from_json_file(tmp_path: Path) -> None:
    """A path source is read + parsed (not only in-memory lists)."""
    import json

    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            [
                {
                    "t_samples": 5,
                    "t_ticks": 1,
                    "type": "note_on",
                    "channel": 3,
                    "note": 64,
                    "velocity": 90,
                }
            ]
        ),
        encoding="utf-8",
    )

    events = load_expected_events(path)

    assert events == [_ev(5, 1, type="note_on", channel=3, note=64, velocity=90)]


def test_load_expected_tick_only_ok() -> None:
    """A tick-only golden is allowed; t_samples is derived from sample_rate+tempo."""
    spec = [
        {
            "t_ticks": 3_840,
            "type": "note_off",
            "channel": 0,
            "note": 48,
            "velocity": 0,
        }
    ]

    events = load_expected_events(spec, sample_rate=_SR, tempo_bpm=_TEMPO)

    # 3840 ticks @960 = 4 beats; @120 BPM = 2.0 s; x 48000 = 96000 samples.
    assert events == [_ev(96_000, 3_840, type="note_off", channel=0, note=48, velocity=0)]


def test_load_expected_samples_only_derives_ticks() -> None:
    """A samples-only golden derives t_ticks via the B1 960-PPQ formula."""
    spec = [
        {
            "t_samples": 24_000,
            "type": "note_on",
            "channel": 1,
            "note": 60,
            "velocity": 100,
        }
    ]

    events = load_expected_events(spec, sample_rate=_SR, tempo_bpm=_TEMPO)

    # 24000 samples @48k = 0.5 s @120 BPM = 1 beat = 960 ticks @960 PPQ.
    assert events == [_ev(24_000, 960, type="note_on", channel=1, note=60, velocity=100)]


def test_load_expected_malformed_is_input_error() -> None:
    """Malformed goldens RED-prove a typed InputError (never a silent skip)."""
    cases = [
        # not a list
        ({"nope": True}, "not_a_list"),
        # missing a required field (velocity)
        (
            [{"t_samples": 0, "t_ticks": 0, "type": "note_on", "channel": 0, "note": 48}],
            "missing_field",
        ),
        # out-of-range note (>127) -> model validation surfaces as typed error
        (
            [
                {
                    "t_samples": 0,
                    "t_ticks": 0,
                    "type": "note_on",
                    "channel": 0,
                    "note": 200,
                    "velocity": 100,
                }
            ],
            "invalid_field",
        ),
        # neither timing axis present
        (
            [{"type": "note_on", "channel": 0, "note": 48, "velocity": 100}],
            "no_timing_axis",
        ),
        # one axis but no conversion params -> never fabricate the gated axis
        (
            [
                {
                    "t_ticks": 960,
                    "type": "note_on",
                    "channel": 0,
                    "note": 48,
                    "velocity": 100,
                }
            ],
            "one_axis_no_params",
        ),
    ]
    for source, reason in cases:
        with pytest.raises(InputError) as exc_info:
            load_expected_events(source)
        err = exc_info.value
        assert err.code == MIDI_EXPECTED_SPEC_INVALID
        assert err.exit_code == ExitCode.INPUT  # exit 2
        assert err.component == "midi"
        assert err.detail is not None and err.detail["reason"] == reason


# --- 2. load_midi_file -------------------------------------------------------


def test_load_midi_file() -> None:
    """The C3-sustain corpus .mid decodes to the exact note_on/note_off pair."""
    events = load_midi_file(_CORPUS_MIDI / "c3_sustain_2s.mid", sample_rate=_SR)

    # note_on C3(48) v100 @0; note_off C3(48) v0 ~2 s later (2.0 s x 48k = 96000
    # samples; 4 beats @960 PPQ = 3840 ticks).
    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=48, velocity=100),
        _ev(96_000, 3_840, type="note_off", channel=0, note=48, velocity=0),
    ]


def test_load_midi_file_phrase() -> None:
    """The 4-note phrase decodes to 8 events at the 0.5 s (24000-sample) grid."""
    events = load_midi_file(_CORPUS_MIDI / "phrase_4note.mid", sample_rate=_SR)

    # C3 E3 G3 C4, each on for 0.5 s back-to-back (step = 480 file ticks = 0.5 s).
    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=48, velocity=100),
        _ev(24_000, 960, type="note_off", channel=0, note=48, velocity=0),
        _ev(24_000, 960, type="note_on", channel=0, note=52, velocity=100),
        _ev(48_000, 1_920, type="note_off", channel=0, note=52, velocity=0),
        _ev(48_000, 1_920, type="note_on", channel=0, note=55, velocity=100),
        _ev(72_000, 2_880, type="note_off", channel=0, note=55, velocity=0),
        _ev(72_000, 2_880, type="note_on", channel=0, note=60, velocity=100),
        _ev(96_000, 3_840, type="note_off", channel=0, note=60, velocity=0),
    ]


def test_load_midi_file_missing_is_input_error() -> None:
    """A bad path RED-proves MIDI_FILE_INVALID (exit 2), not a raw OSError."""
    with pytest.raises(InputError) as exc_info:
        load_midi_file("/no/such/file.mid", sample_rate=_SR)
    err = exc_info.value
    assert err.code == MIDI_FILE_INVALID
    assert err.exit_code == ExitCode.INPUT  # exit 2
    assert err.component == "midi"


def test_midi_file_note_on_vel0_stays_note_on(tmp_path: Path) -> None:
    """A note_on velocity-0 in a .mid decodes FAITHFULLY as note_on, not note_off."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    # The classic running-status note-off: note_on with velocity 0.
    track.append(mido.Message("note_on", note=60, velocity=0, time=480))
    path = tmp_path / "vel0.mid"
    mid.save(str(path))

    events = load_midi_file(path, sample_rate=_SR)

    # Default 120 BPM (no set_tempo): 480 file ticks = 0.5 s = 24000 samples;
    # 480/480 * 960 = 960 ticks @960 PPQ. The vel-0 event stays note_on.
    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=60, velocity=100),
        _ev(24_000, 960, type="note_on", channel=0, note=60, velocity=0),
    ]


def test_midi_file_validationerror_is_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pydantic ``ValidationError` raised while building a MidiEvent mid-walk
    surfaces as the typed MIDI_FILE_INVALID (exit 2), never a raw ValidationError.

    NOTE (env-dependent): in pydantic 2.13.4 ``ValidationError`` subclasses
    ``ValueError``, so the loader's existing ``except (EOFError, OSError,
    ValueError)`` already catches it — this test LOCKS that typed-wrap behavior
    (an unproven path) rather than proving a bug fix. mido clamps note/velocity to
    0-127 and channel to 0-15, so a genuine ValidationError cannot be forced
    through real .mid data; we monkeypatch the loader's ``MidiEvent`` symbol to
    raise a real pydantic ValidationError for the first note.
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    path = tmp_path / "ve.mid"
    mid.save(str(path))

    import sonoscope.midi_input as mi

    class _Probe(BaseModel):
        v: int

    def _raise(**_kwargs: object) -> MidiEvent:  # matches MidiEvent call shape
        _Probe(v="not-an-int")  # genuine pydantic ValidationError
        raise AssertionError("unreachable")  # pragma: no cover

    monkeypatch.setattr(mi, "MidiEvent", _raise)

    with pytest.raises(InputError) as exc_info:
        mi.load_midi_file(path, sample_rate=_SR)
    err = exc_info.value
    assert err.code == MIDI_FILE_INVALID
    assert err.exit_code == ExitCode.INPUT  # exit 2
    assert err.component == "midi"
    assert err.detail is not None and err.detail["reason"] == "corrupt"

    # The forced fault is genuinely a pydantic ValidationError (not a plain
    # ValueError we conflated), so the wrap covers the real symbol.
    with pytest.raises(ValidationError):
        _Probe(v="not-an-int")


def _build_two_tempo_mid(path: Path) -> None:
    """A single-track .mid: 120 BPM (file default) then a set_tempo to 60 BPM.

    Layout (ticks_per_beat=480), deltas resolved by the loader's running tempo:

      note_on  note=60 v100 time=0    -> abs 0 ticks   / 0.0 s
      set_tempo 60 BPM     time=480    -> +480 ticks at PRIOR 120 BPM = +0.5 s
      note_off note=60 v0  time=0      -> abs 480 ticks / 0.5 s
      note_on  note=64 v100 time=480   -> +480 ticks at NEW   60 BPM = +1.0 s
      note_off note=64 v0  time=480    -> +480 ticks at       60 BPM = +1.0 s
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=480))
    track.append(mido.Message("note_off", note=60, velocity=0, time=0))
    track.append(mido.Message("note_on", note=64, velocity=100, time=480))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    mid.save(str(path))


def test_load_midi_file_two_tempo_accrual(tmp_path: Path) -> None:
    """A set_tempo change accrues at the PRIOR tempo up to the change, the NEW
    tempo after — proving the running-tempo accrual is correct, not just present.

    Hand-computed @48 kHz (ticks_per_beat=480, PPQ=960):
      - note_on  note60 : abs 0.0 s      -> t_samples 0,      t_ticks 0
      - note_off note60 : abs 0.5 s      -> t_samples 24000,  t_ticks 960
          (the 480-tick delta ran at the PRIOR 120 BPM: 0.5 s)
      - note_on  note64 : abs 1.5 s      -> t_samples 72000,  t_ticks 1920
          (the next 480-tick delta ran at the NEW 60 BPM: +1.0 s, NOT +0.5 s)
      - note_off note64 : abs 2.5 s      -> t_samples 120000, t_ticks 2880
    """
    path = tmp_path / "two_tempo.mid"
    _build_two_tempo_mid(path)

    events = load_midi_file(path, sample_rate=_SR)

    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=60, velocity=100),
        _ev(24_000, 960, type="note_off", channel=0, note=60, velocity=0),
        _ev(72_000, 1_920, type="note_on", channel=0, note=64, velocity=100),
        _ev(120_000, 2_880, type="note_off", channel=0, note=64, velocity=0),
    ]


def test_load_midi_file_tempo_override_ignores_set_tempo(tmp_path: Path) -> None:
    """An explicit ``tempo_bpm`` overrides the file's set_tempo metas (held
    constant); the 60 BPM meta is ignored.

    Same file as the two-tempo test, forced to a constant 120 BPM @48 kHz:
      - note_on  note60 : 0.0 s -> 0
      - note_off note60 : 0.5 s -> 24000
      - note_on  note64 : 1.0 s -> 48000  (NOT 72000 — the 60 BPM meta is ignored)
      - note_off note64 : 1.5 s -> 72000
    """
    path = tmp_path / "two_tempo_override.mid"
    _build_two_tempo_mid(path)

    events = load_midi_file(path, sample_rate=_SR, tempo_bpm=120.0)

    assert events == [
        _ev(0, 0, type="note_on", channel=0, note=60, velocity=100),
        _ev(24_000, 960, type="note_off", channel=0, note=60, velocity=0),
        _ev(48_000, 1_920, type="note_on", channel=0, note=64, velocity=100),
        _ev(72_000, 2_880, type="note_off", channel=0, note=64, velocity=0),
    ]


# --- 3. apply_slice ----------------------------------------------------------


def test_apply_slice_window() -> None:
    """Half-open [offset, offset+length) in samples; event at end excluded."""
    events = [
        _ev(0, 0),
        _ev(100, 4),
        _ev(200, 8),
        _ev(300, 12),
    ]

    kept = apply_slice(
        events,
        offset=100,
        length=200,  # window [100, 300)
        unit="samples",
        sample_rate=_SR,
        tempo_bpm=_TEMPO,
        rebase=False,
    )

    # 100 (== offset) kept; 200 kept; 0 below; 300 (== end) EXCLUDED (half-open).
    assert [e.t_samples for e in kept] == [100, 200]


def test_apply_slice_rebase() -> None:
    """rebase subtracts offset_samples and recomputes t_ticks @960 from 0."""
    events = [_ev(100, 4), _ev(250, 10)]

    kept = apply_slice(
        events,
        offset=100,
        length=None,  # to end
        unit="samples",
        sample_rate=_SR,
        tempo_bpm=_TEMPO,
        rebase=True,
    )

    # 100 -> 0 (t_ticks 0); 250 -> 150 samples -> 150/48000 * 2 * 960 = 6 ticks.
    assert kept == [
        _ev(0, 0),
        _ev(150, 6),
    ]


def test_apply_slice_beats_unit() -> None:
    """A beats slice converts to samples via tempo + sample_rate."""
    events = [
        _ev(0, 0),
        _ev(24_000, 960),
        _ev(47_999, 1_919),
        _ev(48_000, 1_920),
        _ev(50_000, 2_000),
    ]

    kept = apply_slice(
        events,
        offset=1.0,  # 1 beat @120 BPM = 0.5 s = 24000 samples
        length=1.0,  # 1 beat -> window [24000, 48000)
        unit="beats",
        sample_rate=_SR,
        tempo_bpm=_TEMPO,
        rebase=False,
    )

    # [24000, 48000): 24000 kept, 47999 kept, 48000 excluded (end), 50000 above.
    assert [e.t_samples for e in kept] == [24_000, 47_999]


def test_apply_slice_negative_is_input_error() -> None:
    """A negative offset OR length RED-proves MIDI_SLICE_INVALID (exit 2)."""
    events = [_ev(0, 0)]

    for kwargs in (
        {"offset": -1.0, "length": 100.0},
        {"offset": 0.0, "length": -5.0},
    ):
        with pytest.raises(InputError) as exc_info:
            apply_slice(
                events,
                unit="samples",
                sample_rate=_SR,
                tempo_bpm=_TEMPO,
                **kwargs,  # type: ignore[arg-type]
            )
        err = exc_info.value
        assert err.code == MIDI_SLICE_INVALID
        assert err.exit_code == ExitCode.INPUT  # exit 2
        assert err.component == "midi"


def test_apply_slice_non_finite_is_input_error() -> None:
    """A non-finite (NaN/inf) offset OR length RED-proves MIDI_SLICE_INVALID
    (exit 2), never a raw ``round(nan)`` ValueError / ``round(inf)`` OverflowError.
    """
    events = [_ev(0, 0)]

    cases = [
        ({"offset": float("nan"), "length": 100.0}, "non_finite_offset"),
        ({"offset": float("inf"), "length": 100.0}, "non_finite_offset"),
        ({"offset": 0.0, "length": float("inf")}, "non_finite_length"),
        ({"offset": 0.0, "length": float("nan")}, "non_finite_length"),
    ]
    for kwargs, reason in cases:
        with pytest.raises(InputError) as exc_info:
            apply_slice(
                events,
                unit="samples",
                sample_rate=_SR,
                tempo_bpm=_TEMPO,
                **kwargs,  # type: ignore[arg-type]
            )
        err = exc_info.value
        assert err.code == MIDI_SLICE_INVALID
        assert err.exit_code == ExitCode.INPUT  # exit 2
        assert err.component == "midi"
        assert err.detail is not None and err.detail["reason"] == reason


def test_apply_slice_ticks_unit() -> None:
    """A ticks slice converts @960 PPQ to samples (half-open) and rebases.

    offset=960 ticks and length=960 ticks @120 BPM, 48 kHz: each 960 ticks @960
    = 1 beat = 0.5 s = 24000 samples, so the window is [24000, 48000).
    """
    events = [
        _ev(0, 0),
        _ev(24_000, 960),
        _ev(36_000, 1_440),
        _ev(48_000, 1_920),
    ]

    kept = apply_slice(
        events,
        offset=960.0,  # 960 ticks @960 = 24000 samples
        length=960.0,  # -> window [24000, 48000)
        unit="ticks",
        sample_rate=_SR,
        tempo_bpm=_TEMPO,
        rebase=True,
    )

    # 24000 kept, 36000 kept, 48000 excluded (== end). Rebased to 0-origin:
    #   24000 -> 0      (t_ticks 0)
    #   36000 -> 12000  (12000/48000 * 2 * 960 = 480 ticks @960)
    assert kept == [
        _ev(0, 0),
        _ev(12_000, 480),
    ]


def test_apply_slice_seconds_unit() -> None:
    """A seconds slice converts to samples via sample_rate (half-open window).

    offset=0.5 s and length=0.5 s @48 kHz -> window [24000, 48000).
    """
    events = [
        _ev(0, 0),
        _ev(24_000, 960),
        _ev(47_999, 1_919),
        _ev(48_000, 1_920),
        _ev(50_000, 2_000),
    ]

    kept = apply_slice(
        events,
        offset=0.5,  # 0.5 s = 24000 samples
        length=0.5,  # -> window [24000, 48000)
        unit="seconds",
        sample_rate=_SR,
        tempo_bpm=_TEMPO,
        rebase=False,
    )

    # [24000, 48000): 24000 kept, 47999 kept, 48000 excluded (end), 50000 above.
    assert [e.t_samples for e in kept] == [24_000, 47_999]
