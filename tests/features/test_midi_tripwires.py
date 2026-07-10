"""``evaluate_midi`` tests — the MIDI comparator green-mirage suite.

Green-mirage discipline: every tripwire that can fire ``RED`` ships a RED case
built from a *real* fault (a dropped note_off, a single wrong field, a beyond-
tolerance time shift, an on-before-off at a coincident sample, a 0x90 vel-0
under the explicit-0x80 policy) paired with the GREEN CORRECT#1 pass. Verdicts
are asserted with **exact enum equality** and reasons with **exact-equality**
(no substring) so a constant ``"PASS"`` / ``"RED"`` stub cannot pass and each
RED test FAILS if its wire were removed.

Ground truth = the CORRECT#1 golden (by design): the 8-event two-note
ostinato at 120 BPM / 48 kHz. ``t_ticks`` is derived from ``t_samples`` at 960
PPQ (``ticks = round(t_samples / 48000 * (120/60) * 960) = round(t_samples/25)``).
"""

from __future__ import annotations

import pytest

from sonoscope.features.midi_tripwires import (
    MISSING_OR_EXTRA_ID,
    MISTIMED_ID,
    OFFVEL0_RED_REASON,
    ORDERING_ID,
    STUCK_NOTE_ID,
    WRONG_FIELD_ID,
    evaluate_midi,
)
from sonoscope.schema import MidiEvent

# --- CORRECT#1 golden builders (120 BPM / 48 kHz; ticks = t_samples / 25) -----

_SR = 48000
_BPM = 120.0


def _ticks(t_samples: int) -> int:
    """``t_ticks`` @960 PPQ from ``t_samples`` (the decode rule, by design)."""
    return round(t_samples / _SR * (_BPM / 60.0) * 960)


def _on(t_samples: int, channel: int, note: int, velocity: int = 100) -> MidiEvent:
    return MidiEvent(
        t_samples=t_samples,
        t_ticks=_ticks(t_samples),
        type="note_on",
        channel=channel,
        note=note,
        velocity=velocity,
    )


def _off(t_samples: int, channel: int, note: int, velocity: int = 0) -> MidiEvent:
    return MidiEvent(
        t_samples=t_samples,
        t_ticks=_ticks(t_samples),
        type="note_off",
        channel=channel,
        note=note,
        velocity=velocity,
    )


def _correct_one() -> list[MidiEvent]:
    """The CORRECT#1 golden: 8 events, canonical order (off-before-on at coincidence).

    on ch0 n36 v100 @0; off ch0 n36 @6000 + on ch1 n72 v100 @6000; off ch1 n72
    @9000; on ch0 n36 @12000; off ch0 n36 @18000 + on ch1 n72 @18000; off ch1
    n72 @21000.
    """
    return [
        _on(0, 0, 36),
        _off(6000, 0, 36),
        _on(6000, 1, 72),
        _off(9000, 1, 72),
        _on(12000, 0, 36),
        _off(18000, 0, 36),
        _on(18000, 1, 72),
        _off(21000, 1, 72),
    ]


# --- GREEN: the correct pattern diffed against itself -------------------------


def test_correct_pattern_is_pass() -> None:
    """CORRECT#1 vs itself → PASS, matched=8, clean integrity, empty diff/reasons."""
    events = _correct_one()
    result = evaluate_midi(events, _correct_one())

    assert result.verdict == "PASS"
    assert result.reasons == []
    # Integrity clean: every note_on paired.
    assert result.integrity.every_note_on_has_off is True
    assert result.integrity.stuck_notes == []
    assert result.integrity.dangling_offs == []
    # Diff: all 8 matched, nothing else.
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 8
    assert diff.missing == []
    assert diff.extra == []
    assert diff.mistimed == []
    assert diff.wrong_field == []


# --- RED #1: stuck-note (THE M5 firewall check) -------------------------------


def test_stuck_note_is_red() -> None:
    """Drop a note_off → the orphan note_on is stuck → RED naming (channel,note).

    RED-proving: with no expected list supplied, the ONLY wire that can fire is
    stuck-note; remove it and the verdict would be PASS.
    """
    events = _correct_one()
    # Drop off ch0 n36 @6000: on ch0 n36 @0 now has no matching note_off.
    events = [e for e in events if not (e.type == "note_off" and e.channel == 0 and e.note == 36 and e.t_samples == 6000)]

    result = evaluate_midi(events)  # no expected: isolates the integrity wire

    assert result.verdict == "RED"
    assert result.integrity.stuck_notes != []
    stuck = result.integrity.stuck_notes[0]
    assert (stuck.channel, stuck.note, stuck.t_samples) == (0, 36, 0)
    assert result.integrity.every_note_on_has_off is False
    assert result.reasons == [f"{STUCK_NOTE_ID}: ch0 n36@0"]


def test_overlapping_stuck_note_pairs_lifo() -> None:
    """Two overlapping same-(ch,note) note_ons + ONE note_off → the EARLIER on is stuck.

    Boundary: the ONE case where LIFO vs FIFO diverges. Integrity pairs a note_off
    to the MOST-RECENT unmatched note_on of that (channel, note) (`stack.pop()`),
    so the note_off @2000 pairs to the LATER on @1000, leaving the EARLIER on @0
    stuck. A silent switch to FIFO would leave the count at 1 but NAME @1000 —
    asserting the named event locks the LIFO behavior. No expected list is
    supplied, so stuck-note is the sole wire that can fire (RED-proving).
    """
    # Same (channel, note); two overlapping ons, a single closing off.
    captured = [_on(0, 0, 36), _on(1000, 0, 36), _off(2000, 0, 36)]

    result = evaluate_midi(captured)  # integrity-only: isolates the stuck-note wire

    assert result.verdict == "RED"
    # Exactly ONE note remains unpaired (leftover-count is 1 under LIFO or FIFO)...
    assert len(result.integrity.stuck_notes) == 1
    # ...but LIFO names the EARLIER on @0 (FIFO would name the later on @1000).
    stuck = result.integrity.stuck_notes[0]
    assert (stuck.channel, stuck.note, stuck.t_samples) == (0, 36, 0)
    assert result.integrity.every_note_on_has_off is False
    assert result.integrity.dangling_offs == []
    assert result.reasons == [f"{STUCK_NOTE_ID}: ch0 n36@0"]


# --- priority ordering of reasons[] (multiple wires firing at once) -----------


def test_reasons_priority_order_stuck_note_before_missing() -> None:
    """TWO wires fire at once → reasons[] is emitted in priority order (by design).

    Drop `off ch0 n36 @6000` from the capture while diffing against the full
    CORRECT#1 expected list. That single drop trips TWO wires: the orphaned
    `on ch0 n36 @0` is a stuck note (#1, integrity — the off@18000 pairs LIFO to
    on@12000), and the dropped note_off is `missing` vs expected (#2,
    missing-or-extra). `_build_reasons` appends wires in priority order, so
    stuck-note MUST precede missing-or-extra. The comparator matches on reasons[], so this
    exact ordering is contract — full-string equality locks the emission order.
    """
    expected = _correct_one()
    captured = [
        e
        for e in _correct_one()
        if not (e.type == "note_off" and e.channel == 0 and e.note == 36 and e.t_samples == 6000)
    ]

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    # BOTH wires fire: stuck-note (integrity) AND missing-or-extra (diff).
    assert result.integrity.stuck_notes != []
    diff = result.expected_vs_actual
    assert diff is not None
    assert [(e.channel, e.note, e.t_samples) for e in diff.missing] == [(0, 36, 6000)]
    assert diff.extra == []
    # Priority order (by design): stuck-note (#1) BEFORE missing-or-extra (#2).
    assert result.reasons == [
        f"{STUCK_NOTE_ID}: ch0 n36@0",
        f"{MISSING_OR_EXTRA_ID}: 1 missing, 0 extra",
    ]


# --- RED #2: missing / extra --------------------------------------------------


def test_missing_event_is_red() -> None:
    """Drop a note_on from the capture → it is `missing` vs expected → RED.

    Dropping a note_ON (not off) leaves a dangling_off (not a stuck note), so the
    ONLY RED wire is missing-or-extra. RED-proving: remove that wire → PASS.
    """
    expected = _correct_one()
    captured = [e for e in _correct_one() if not (e.type == "note_on" and e.channel == 1 and e.note == 72 and e.t_samples == 18000)]

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 7
    assert [(e.channel, e.note, e.t_samples) for e in diff.missing] == [(1, 72, 18000)]
    assert diff.extra == []
    assert result.integrity.stuck_notes == []  # a dropped ON dangles the off, not stuck
    assert result.reasons == [f"{MISSING_OR_EXTRA_ID}: 1 missing, 0 extra"]


def test_extra_event_is_red() -> None:
    """Add a spurious (paired) note the expected list lacks → `extra` → RED."""
    expected = _correct_one()
    captured = _correct_one() + [_on(3000, 2, 50), _off(4000, 2, 50)]

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 8
    assert diff.missing == []
    assert [(e.channel, e.note, e.t_samples, e.type) for e in diff.extra] == [
        (2, 50, 3000, "note_on"),
        (2, 50, 4000, "note_off"),
    ]
    assert result.reasons == [f"{MISSING_OR_EXTRA_ID}: 0 missing, 2 extra"]


# --- RED #3: wrong-field (EXACT, one of channel/note/velocity/type) -----------


def _first_on_ch0_n36_at0_replaced(replacement: MidiEvent) -> list[MidiEvent]:
    """CORRECT#1 with the leading `on ch0 n36 v100 @0` swapped for `replacement`."""
    out = _correct_one()
    out[0] = replacement
    return out


def test_wrong_channel_is_red() -> None:
    """First note-pair off ONLY in channel → wrong_field(channel), clean integrity.

    The whole on/off pair is moved to channel 1 so captured integrity stays clean
    (no stuck/dangling): wrong-field is thus the SOLE firing wire — RED-proving.
    A per-endpoint change would instead orphan a note_on into a stuck note.
    """
    expected = _correct_one()
    captured = _correct_one()
    captured[0] = _on(0, 1, 36)  # on ch0->ch1 n36 @0
    captured[1] = _off(6000, 1, 36)  # off ch0->ch1 n36 @6000

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    assert result.integrity.stuck_notes == []
    assert result.integrity.dangling_offs == []
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 6
    assert diff.missing == []  # aligned as wrong-field, NOT missing+extra
    assert diff.extra == []
    assert [w.field for w in diff.wrong_field] == ["channel", "channel"]
    assert result.reasons == [f"{WRONG_FIELD_ID}: channel@0; channel@6000"]


def test_wrong_note_is_red() -> None:
    """First note-pair off ONLY in note → wrong_field(note), clean integrity."""
    expected = _correct_one()
    captured = _correct_one()
    captured[0] = _on(0, 0, 37)  # n36->n37 @0
    captured[1] = _off(6000, 0, 37)  # n36->n37 @6000

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    assert result.integrity.stuck_notes == []
    assert result.integrity.dangling_offs == []
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 6
    assert [w.field for w in diff.wrong_field] == ["note", "note"]
    assert diff.missing == []
    assert diff.extra == []
    assert result.reasons == [f"{WRONG_FIELD_ID}: note@0; note@6000"]


def test_wrong_velocity_is_red() -> None:
    """One event off ONLY in velocity → wrong_field(field=velocity)."""
    expected = _correct_one()
    captured = _first_on_ch0_n36_at0_replaced(_on(0, 0, 36, velocity=99))  # 100 -> 99

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    diff = result.expected_vs_actual
    assert diff is not None
    assert [w.field for w in diff.wrong_field] == ["velocity"]
    assert diff.missing == []
    assert diff.extra == []
    assert result.reasons == [f"{WRONG_FIELD_ID}: velocity@0"]


def test_wrong_type_is_red() -> None:
    """One event off ONLY in type (on→off) → wrong_field(field=type), not missing+extra."""
    expected = _correct_one()
    captured = _first_on_ch0_n36_at0_replaced(_off(0, 0, 36, velocity=100))  # on -> off

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    diff = result.expected_vs_actual
    assert diff is not None
    assert [w.field for w in diff.wrong_field] == ["type"]
    assert diff.missing == []
    assert diff.extra == []
    assert result.reasons == [f"{WRONG_FIELD_ID}: type@0"]


def test_two_field_diff_is_missing_plus_extra_not_wrong_field() -> None:
    """A residual off in TWO fields (channel AND note) → missing+extra, NOT wrong_field.

    Boundary: Stage-2 (`_align_wrong_field`) aligns ONLY single-field-off residuals
    (`len(diffs) == 1`). The leading pair is moved in BOTH channel and note (timing
    unchanged), so each residual expected/actual pair differs in two exact fields —
    never a wrong-field candidate. It must therefore fall through to `missing` (the
    expected pair) + `extra` (the captured pair), with `wrong_field` empty, proving
    a 2-field residual is NOT mislabeled as a single wrong-field. The whole pair is
    moved together so captured integrity stays clean (RED comes via missing-or-extra).
    """
    expected = _correct_one()
    captured = _correct_one()
    captured[0] = _on(0, 1, 37)  # ch0->ch1 AND n36->n37 @0 (two fields; timing equal)
    captured[1] = _off(6000, 1, 37)  # ch0->ch1 AND n36->n37 @6000

    result = evaluate_midi(captured, expected)

    assert result.verdict == "RED"
    # Clean integrity: the moved pair still pairs internally (no stuck/dangling).
    assert result.integrity.stuck_notes == []
    assert result.integrity.dangling_offs == []
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 6  # the six untouched events still match exactly
    # The 2-field residual is NOT aligned as wrong-field...
    assert diff.wrong_field == []
    # ...it falls through to missing (expected) + extra (captured).
    assert [(e.channel, e.note, e.t_samples) for e in diff.missing] == [
        (0, 36, 0),
        (0, 36, 6000),
    ]
    assert [(e.channel, e.note, e.t_samples) for e in diff.extra] == [
        (1, 37, 0),
        (1, 37, 6000),
    ]
    assert result.reasons == [f"{MISSING_OR_EXTRA_ID}: 2 missing, 2 extra"]


# --- RED #4: mistimed (the ±SAMPLE timing gate) -------------------------------


def _off_ch0_n36_at6000_shifted(new_t: int) -> list[MidiEvent]:
    """CORRECT#1 with `off ch0 n36 @6000` moved to `new_t` (same ch/note/type/vel)."""
    out = _correct_one()
    for i, e in enumerate(out):
        if e.type == "note_off" and e.channel == 0 and e.note == 36 and e.t_samples == 6000:
            out[i] = _off(new_t, 0, 36)
    return out


def test_mistimed_beyond_tolerance_is_red() -> None:
    """Shift one event by 2 samples (> ±1 tolerance) → mistimed → RED."""
    expected = _correct_one()
    captured = _off_ch0_n36_at6000_shifted(6002)  # delta +2 samples

    result = evaluate_midi(captured, expected)  # default tolerance = 1 sample

    assert result.verdict == "RED"
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 7
    assert len(diff.mistimed) == 1
    mist = diff.mistimed[0]
    assert mist.delta_samples == 2
    assert mist.expected.t_samples == 6000
    assert mist.actual.t_samples == 6002
    assert result.reasons == [f"{MISTIMED_ID}: @6000 delta 2 samples"]


def test_within_tolerance_is_pass() -> None:
    """Shift one event by exactly 1 sample (== tolerance) → still matched → PASS."""
    expected = _correct_one()
    captured = _off_ch0_n36_at6000_shifted(6001)  # delta +1 sample, within ±1

    result = evaluate_midi(captured, expected)

    assert result.verdict == "PASS"
    diff = result.expected_vs_actual
    assert diff is not None
    assert diff.matched == 8  # the ±1-sample near-match still counts as matched
    assert diff.mistimed == []
    assert result.reasons == []


# --- RED #5: ordering (on before its own off at a coincident timestamp) -------


def test_ordering_on_before_own_off_is_red() -> None:
    """note_on emitted before its own note_off at the SAME sample → RED.

    RED-proving: integrity PAIRS the coincident on/off (not stuck), and no
    expected list is supplied, so ordering is the only wire that can fire.
    """
    # Deliberately on-BEFORE-off at t=1000 for the same (channel, note).
    captured = [_on(1000, 0, 36), _off(1000, 0, 36)]

    result = evaluate_midi(captured)

    assert result.verdict == "RED"
    assert result.integrity.stuck_notes == []  # the pair is matched, not stuck
    assert result.integrity.every_note_on_has_off is True
    assert result.reasons == [f"{ORDERING_ID}: ch0 n36@1000 note_on before note_off"]


# --- offvel0 contract (0x90 vel-0 under the explicit-0x80 policy) -------------


def test_offvel0_note_on_is_red_default() -> None:
    """A note_on velocity-0 under the default "red" policy → the contract RED.

    RED-proving: the vel-0 note_on is PAIRED (a later note_off), so integrity is
    clean and no expected list is supplied — the offvel0 contract is the only
    wire that can fire; remove it and the verdict would be PASS.
    """
    captured = [_on(0, 0, 36, velocity=0), _off(6000, 0, 36)]

    result = evaluate_midi(captured)  # default policy = "red"

    assert result.verdict == "RED"
    assert result.integrity.stuck_notes == []
    assert result.reasons == [OFFVEL0_RED_REASON]


def test_offvel0_normalize_policy() -> None:
    """Under "normalize", a note_on vel-0 is folded to note_off → not RED, noted."""
    captured = [_on(0, 0, 36, velocity=0), _off(6000, 0, 36)]

    result = evaluate_midi(captured, offvel0_as_0x90_policy="normalize")

    assert result.verdict == "PASS"
    # Treated as a note_off: no stuck note, no offvel0-contract RED.
    assert result.integrity.stuck_notes == []
    assert OFFVEL0_RED_REASON not in result.reasons
    # The fold is recorded as an informational note (by design "note it").
    assert result.reasons == [
        "offvel0-normalized: ch0 n36@0 note_on(vel0) treated as note_off"
    ]


# --- dangling-off detection (integrity; not itself a RED wire) ----------------


def test_dangling_off_detected() -> None:
    """A lone note_off (no preceding note_on) → `dangling_offs`, verdict PASS.

    A dangling-off is reported in integrity but is intentionally NOT a RED trigger:
    `_has_red` never inspects `dangling_offs`. A windowed / half-open capture whose
    opening note_on fell outside the slice must not false-RED — this locks that
    half-open-slice-boundary semantics so a future regression cannot make windowed
    captures RED. Integrity-only (no expected list): the only wire that could fire
    is stuck-note, and it must stay silent (dangling-off ≠ stuck-note).
    """
    captured = [_off(1000, 0, 36)]

    result = evaluate_midi(captured)

    assert result.integrity.dangling_offs != []
    dangling = result.integrity.dangling_offs[0]
    assert (dangling.channel, dangling.note, dangling.t_samples) == (0, 36, 1000)
    assert result.integrity.every_note_on_has_off is False
    assert result.integrity.stuck_notes == []
    # Dangling-off alone is NOT RED (by design / `_has_red`): verdict PASS, and no
    # stuck-note reason emitted.
    assert result.verdict == "PASS"
    assert result.reasons == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
