"""E1 ``evaluate_midi`` — the MIDI comparator + tripwire engine (by design).

`src/sonoscope/features/midi_tripwires.py`, the MIDI analogue of
`features/tripwires.py`: a **pure function** of the captured ground-truth event
list (plus an optional expected list), exact-enum verdicts, no re-capture. The
orchestrator (C1) assembles the full :class:`~sonoscope.schema.MidiBlock`; this
module produces the *parts* it computes — the integrity block, the
expected-vs-actual diff (when an expected list is supplied), the ``PASS``/``RED``
verdict, and the priority-ordered ``reasons[]`` (each a firing tripwire).

Design invariants honored here:

- **Ground truth = the event list.** Every verdict is a pure function of the
  already-decoded events; no randomness, exact enum equality.
- **Timing gate is SAMPLES (by design), NOT ticks.** ``t_samples`` is the
  canonical timing axis and the ``timing_tolerance_samples`` gate is ± that many
  samples. ``t_ticks`` is carried through :class:`MistimedEvent.delta_ticks` for
  human readability only; it never gates a verdict.
- **Five tripwires in priority order** (by design): ``stuck-note`` (#1, the M5
  firewall), ``missing-or-extra``, ``wrong-field`` (EXACT), ``mistimed``,
  ``ordering``. Plus an ``offvel0-contract`` check for the explicit-0x80 policy.
- **overall roll-up.** ``verdict = RED`` if ANY tripwire is RED, else ``PASS``.
  ``reasons[]`` lists each firing wire in priority order. ``ERROR`` is reserved
  for the fatal envelope and is never produced here.

Faithful-decode boundary (by design): B1 decodes a ``0x90`` with velocity 0
FAITHFULLY as ``note_on`` velocity 0; flagging that contract violation is THIS
comparator's job, gated by ``offvel0_as_0x90_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sonoscope.schema import (
    ExpectedVsActual,
    MidiEvent,
    MidiField,
    MidiIntegrity,
    MidiVerdict,
    MistimedEvent,
    WrongFieldEvent,
)

# --- Verdict constants (schema MidiVerdict values; SSOT for comparisons) ------
PASS: MidiVerdict = "PASS"
RED: MidiVerdict = "RED"

# --- Tripwire ids (priority order; stable strings) -----------------------------
STUCK_NOTE_ID = "stuck-note"
MISSING_OR_EXTRA_ID = "missing-or-extra"
WRONG_FIELD_ID = "wrong-field"
MISTIMED_ID = "mistimed"
ORDERING_ID = "ordering"
OFFVEL0_ID = "offvel0-contract"

#: The exact contract-violation reason for a note-off encoded as ``0x90`` vel-0
#: under the default (Reference Sequencer) ``"red"`` policy (open question, by design).
OFFVEL0_RED_REASON = "note-off-as-0x90-vel0 violates explicit-0x80 contract"

#: The four fields compared with ZERO tolerance for the wrong-field tripwire
#: ("EXACT", by design). ``t_samples``/``t_ticks`` are the timing axis and are NOT
#: in this set — a same-field pair off in time is ``mistimed``, not wrong-field.
_EXACT_FIELDS: tuple[MidiField, ...] = ("channel", "note", "type", "velocity")

# Policy alias for the 0x90-vel0 handling (default matches the Reference Sequencer contract).
Offvel0Policy = Literal["red", "normalize"]


@dataclass(frozen=True)
class MidiEvaluation:
    """The pure-function output parts the orchestrator folds into a MidiBlock.

    ``evaluate_midi`` returns this; C1 assembles the full
    :class:`~sonoscope.schema.MidiBlock` (adding ``capture_meta`` and ``events``).
    ``expected_vs_actual`` is ``None`` iff no expected list was supplied.
    """

    integrity: MidiIntegrity
    expected_vs_actual: Optional[ExpectedVsActual]
    verdict: MidiVerdict
    reasons: list[str]


def evaluate_midi(
    captured: list[MidiEvent],
    expected: Optional[list[MidiEvent]] = None,
    *,
    timing_tolerance_samples: int = 1,
    offvel0_as_0x90_policy: Offvel0Policy = "red",
) -> MidiEvaluation:
    """Evaluate the MIDI ground truth into integrity + diff + verdict.

    Parameters
    ----------
    captured:
        The canonical decoded event list from B1 (ground truth). Its *given*
        order is authoritative for the ``ordering`` tripwire (a synth sees events
        in emission order); integrity pairing and the diff are order-independent
        (they sort internally by time).
    expected:
        The optional expected event list. When ``None`` the expected-vs-actual
        diff is not produced and tripwires 2-4 (missing-or-extra, wrong-field,
        mistimed) do not run; integrity, ordering and the offvel0 contract always
        run.
    timing_tolerance_samples:
        The ± timing gate in SAMPLES (by design). A near-match
        with ``|actual - expected| <= tolerance`` counts as matched; beyond it is
        ``mistimed``.
    offvel0_as_0x90_policy:
        ``"red"`` (default, Reference Sequencer): a ``note_on`` velocity-0 event is a
        contract violation -> RED. ``"normalize"``: treat it as a ``note_off``
        for integrity/diff/ordering and record an informational note (for future
        non-Reference Sequencer sources that legitimately emit 0x90 vel-0 as note-off).
    """
    # --- offvel0 policy: detect the 0x90-vel0 events and normalize if asked ---
    # A note_on with velocity 0 is B1's faithful decode of a 0x90 vel-0. Under
    # "normalize" it acts as a note_off for all downstream analysis; under "red"
    # it stays a note_on (and trips the offvel0 contract wire below).
    offvel0_events = [
        e for e in captured if e.type == "note_on" and e.velocity == 0
    ]
    if offvel0_as_0x90_policy == "normalize":
        effective = [
            e.model_copy(update={"type": "note_off"})
            if (e.type == "note_on" and e.velocity == 0)
            else e
            for e in captured
        ]
    else:
        effective = list(captured)

    # --- integrity (always; no expected needed) ------------------------------
    integrity = _compute_integrity(effective)

    # --- expected-vs-actual diff (only when an expected list is supplied) -----
    expected_vs_actual: Optional[ExpectedVsActual] = None
    if expected is not None:
        expected_vs_actual = _compute_diff(
            effective, expected, timing_tolerance_samples
        )

    # --- ordering (always; keys on the GIVEN emission order) -----------------
    ordering_violations = _detect_ordering(effective)

    # --- roll-up: verdict + priority-ordered reasons -------------------------
    reasons = _build_reasons(
        integrity=integrity,
        diff=expected_vs_actual,
        ordering_violations=ordering_violations,
        offvel0_events=offvel0_events,
        offvel0_policy=offvel0_as_0x90_policy,
    )
    verdict: MidiVerdict = RED if _has_red(
        integrity=integrity,
        diff=expected_vs_actual,
        ordering_violations=ordering_violations,
        offvel0_events=offvel0_events,
        offvel0_policy=offvel0_as_0x90_policy,
    ) else PASS

    return MidiEvaluation(
        integrity=integrity,
        expected_vs_actual=expected_vs_actual,
        verdict=verdict,
        reasons=reasons,
    )


# --- integrity / note pairing ------------------------------------------------


def _pairing_key(event: MidiEvent) -> tuple[int, int]:
    """Sort key for integrity pairing: by time, note_ON BEFORE note_off.

    Deliberately the INVERSE of B1's canonical (off-before-on) order: at a
    coincident ``t_samples`` a note_on must be *available* before its note_off so
    a zero-length note pairs cleanly rather than false-tripping stuck/dangling.
    The bad emission order is caught by the separate ``ordering`` tripwire, not
    by integrity (by design).
    """
    return (event.t_samples, 0 if event.type == "note_on" else 1)


def _compute_integrity(events: list[MidiEvent]) -> MidiIntegrity:
    """Pair note_on/note_off by ``(channel, note)`` and roll up integrity.

    Matches a note_off to the MOST RECENT unmatched note_on of the same
    ``(channel, note)`` (a per-key LIFO stack over time order). Leftover note_ons
    -> ``stuck_notes`` (the #1 M5-firewall check); a note_off with no open note_on
    -> ``dangling_offs``. ``every_note_on_has_off`` is the clean-integrity roll-up.
    """
    open_notes: dict[tuple[int, int], list[MidiEvent]] = {}
    dangling_offs: list[MidiEvent] = []

    for event in sorted(events, key=_pairing_key):
        key = (event.channel, event.note)
        if event.type == "note_on":
            open_notes.setdefault(key, []).append(event)
        else:  # note_off
            stack = open_notes.get(key)
            if stack:
                stack.pop()  # match the most recent unmatched note_on (LIFO)
            else:
                dangling_offs.append(event)

    stuck_notes = [e for stack in open_notes.values() for e in stack]
    # Report in canonical (time-ordered) form so the block is deterministic.
    stuck_notes.sort(key=lambda e: (e.t_samples, e.channel, e.note))
    dangling_offs.sort(key=lambda e: (e.t_samples, e.channel, e.note))

    return MidiIntegrity(
        every_note_on_has_off=not stuck_notes and not dangling_offs,
        stuck_notes=stuck_notes,
        dangling_offs=dangling_offs,
    )


# --- expected-vs-actual diff -------------------------------------------------


def _exact_bucket(event: MidiEvent) -> tuple[int, int, str, int]:
    """Bucket key for stage-1 exact matching: (channel, note, type, velocity)."""
    return (event.channel, event.note, event.type, event.velocity)


def _diff_fields(a: MidiEvent, b: MidiEvent) -> list[MidiField]:
    """The subset of ``_EXACT_FIELDS`` on which two events differ (EXACT)."""
    return [f for f in _EXACT_FIELDS if getattr(a, f) != getattr(b, f)]


def _compute_diff(
    captured: list[MidiEvent],
    expected: list[MidiEvent],
    tolerance_samples: int,
) -> ExpectedVsActual:
    """Match captured<->expected and classify into the five diff buckets.

    Deterministic, order-independent (both inputs are consumed via time-sorted
    buckets, not their given order). Two stages:

    STAGE 1 — exact matches. Bucket both lists by ``(channel, note, type,
    velocity)``; within a bucket pair by NEAREST time (greedily, smallest
    ``|dt|`` first — robust to unequal counts, where a naive index-zip would
    mis-pair the survivors). A pair within ``tolerance_samples`` -> ``matched``;
    a pair beyond tolerance (same all four fields, off ONLY in time) ->
    ``mistimed``. Unpaired events (unequal counts) fall through to the residual.

    STAGE 2 — wrong-field on the stage-1 residual. A residual expected/actual
    pair that is a near-match in time (``|dt| <= tolerance``) AND differs in
    EXACTLY ONE of channel/note/type/velocity -> ``wrong_field`` (EXACT, zero
    tolerance on the field). This is what disambiguates a wrong-channel event
    from a spurious ``missing`` + ``extra`` pair: it is aligned to its
    nearest-time single-field-off counterpart before either side can be declared
    missing/extra. Candidates are consumed greedily in a fully-ordered priority
    (smallest ``|dt|`` first, then expected time/index, then actual time/index)
    so the result never depends on input order. Whatever remains unpaired ->
    ``missing`` (expected) / ``extra`` (actual).
    """
    matched_count = 0
    mistimed: list[MistimedEvent] = []
    residual_exp: list[MidiEvent] = []
    residual_act: list[MidiEvent] = []

    # STAGE 1 — exact-field buckets, paired by ascending time.
    exp_buckets: dict[tuple[int, int, str, int], list[MidiEvent]] = {}
    act_buckets: dict[tuple[int, int, str, int], list[MidiEvent]] = {}
    for e in expected:
        exp_buckets.setdefault(_exact_bucket(e), []).append(e)
    for a in captured:
        act_buckets.setdefault(_exact_bucket(a), []).append(a)

    for bucket in sorted(set(exp_buckets) | set(act_buckets)):
        exp_sorted = sorted(
            exp_buckets.get(bucket, []), key=lambda e: (e.t_samples, e.t_ticks)
        )
        act_sorted = sorted(
            act_buckets.get(bucket, []), key=lambda e: (e.t_samples, e.t_ticks)
        )
        # Greedy nearest-time pairing (smallest |dt| first) within the bucket.
        candidates = sorted(
            (
                (abs(a.t_samples - e.t_samples), e.t_samples, ei, a.t_samples, ai)
                for ei, e in enumerate(exp_sorted)
                for ai, a in enumerate(act_sorted)
            )
        )
        used_e: set[int] = set()
        used_a: set[int] = set()
        for _dt, _et, ei, _at, ai in candidates:
            if ei in used_e or ai in used_a:
                continue
            used_e.add(ei)
            used_a.add(ai)
            e, a = exp_sorted[ei], act_sorted[ai]
            if abs(a.t_samples - e.t_samples) <= tolerance_samples:
                matched_count += 1
            else:
                # Same (channel, note, type, velocity); off ONLY in time.
                mistimed.append(
                    MistimedEvent(
                        expected=e,
                        actual=a,
                        delta_samples=a.t_samples - e.t_samples,
                        delta_ticks=a.t_ticks - e.t_ticks,
                    )
                )
        residual_exp.extend(e for ei, e in enumerate(exp_sorted) if ei not in used_e)
        residual_act.extend(a for ai, a in enumerate(act_sorted) if ai not in used_a)

    # STAGE 2 — wrong-field alignment over the stage-1 residual.
    wrong_field, missing, extra = _align_wrong_field(
        residual_exp, residual_act, tolerance_samples
    )

    mistimed.sort(key=lambda m: (m.expected.t_samples, m.expected.channel, m.expected.note))
    wrong_field.sort(
        key=lambda w: (w.expected.t_samples, w.expected.channel, w.expected.note)
    )
    missing.sort(key=lambda e: (e.t_samples, e.channel, e.note))
    extra.sort(key=lambda e: (e.t_samples, e.channel, e.note))

    return ExpectedVsActual(
        matched=matched_count,
        missing=missing,
        extra=extra,
        mistimed=mistimed,
        wrong_field=wrong_field,
    )


def _align_wrong_field(
    residual_exp: list[MidiEvent],
    residual_act: list[MidiEvent],
    tolerance_samples: int,
) -> tuple[list[WrongFieldEvent], list[MidiEvent], list[MidiEvent]]:
    """Greedily pair near-time single-field-off residuals into wrong_field.

    Returns ``(wrong_field, missing, extra)``. A candidate pair is a residual
    expected/actual within ``tolerance_samples`` in time that differs in EXACTLY
    one exact field. Candidates are consumed in a fully-ordered priority so the
    assignment is deterministic and order-independent; unpaired residuals fall
    through to ``missing`` (expected side) / ``extra`` (actual side).
    """
    candidates: list[tuple[int, int, int, int, int, MidiField]] = []
    for ei, e in enumerate(residual_exp):
        for ai, a in enumerate(residual_act):
            if abs(a.t_samples - e.t_samples) > tolerance_samples:
                continue
            diffs = _diff_fields(e, a)
            if len(diffs) == 1:
                candidates.append(
                    (abs(a.t_samples - e.t_samples), e.t_samples, ei, ai, a.t_samples, diffs[0])
                )
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[4], c[3]))

    used_exp: set[int] = set()
    used_act: set[int] = set()
    wrong_field: list[WrongFieldEvent] = []
    for _dt, _et, ei, ai, _at, field in candidates:
        if ei in used_exp or ai in used_act:
            continue
        used_exp.add(ei)
        used_act.add(ai)
        wrong_field.append(
            WrongFieldEvent(
                expected=residual_exp[ei], actual=residual_act[ai], field=field
            )
        )

    missing = [e for ei, e in enumerate(residual_exp) if ei not in used_exp]
    extra = [a for ai, a in enumerate(residual_act) if ai not in used_act]
    return wrong_field, missing, extra


# --- ordering tripwire -------------------------------------------------------


def _detect_ordering(events: list[MidiEvent]) -> list[tuple[int, int, int]]:
    """Find note_on-before-its-own-note_off at a coincident timestamp.

    A RED condition: for the SAME ``(channel, note)`` at the SAME ``t_samples``, a
    ``note_on`` appears earlier in the GIVEN emission order than a ``note_off``.
    (B1's canonical order emits off-before-on at a coincident sample, so a healthy
    capture never trips this; a file source or a buggy capture can.) Returns the
    distinct ``(channel, note, t_samples)`` violations, time-sorted.
    """
    violations: set[tuple[int, int, int]] = set()
    for i, on in enumerate(events):
        if on.type != "note_on":
            continue
        for off in events[i + 1:]:
            if (
                off.type == "note_off"
                and off.channel == on.channel
                and off.note == on.note
                and off.t_samples == on.t_samples
            ):
                violations.add((on.channel, on.note, on.t_samples))
    return sorted(violations, key=lambda v: (v[2], v[0], v[1]))


# --- verdict + reasons roll-up -----------------------------------------------


def _offvel0_is_red(events: list[MidiEvent], policy: Offvel0Policy) -> bool:
    """The offvel0 contract fires RED only under ``"red"`` policy with a hit."""
    return policy == "red" and bool(events)


def _has_red(
    *,
    integrity: MidiIntegrity,
    diff: Optional[ExpectedVsActual],
    ordering_violations: list[tuple[int, int, int]],
    offvel0_events: list[MidiEvent],
    offvel0_policy: Offvel0Policy,
) -> bool:
    """``True`` iff any tripwire fires RED (the ``overall`` roll-up)."""
    if integrity.stuck_notes:
        return True
    if diff is not None and (diff.missing or diff.extra):
        return True
    if diff is not None and diff.wrong_field:
        return True
    if diff is not None and diff.mistimed:
        return True
    if ordering_violations:
        return True
    if _offvel0_is_red(offvel0_events, offvel0_policy):
        return True
    return False


def _build_reasons(
    *,
    integrity: MidiIntegrity,
    diff: Optional[ExpectedVsActual],
    ordering_violations: list[tuple[int, int, int]],
    offvel0_events: list[MidiEvent],
    offvel0_policy: Offvel0Policy,
) -> list[str]:
    """Build ``reasons[]`` — each firing wire in priority order.

    Priority: stuck-note, missing-or-extra, wrong-field, mistimed, ordering, then
    the offvel0 contract. Under ``"normalize"`` policy an offvel0 hit adds a
    single INFORMATIONAL note (not a RED wire) so the report records that a 0x90
    vel-0 was folded to note_off.
    """
    reasons: list[str] = []

    # 1. stuck-note (#1, the M5 firewall) — names each (channel, note).
    if integrity.stuck_notes:
        detail = "; ".join(
            f"ch{e.channel} n{e.note}@{e.t_samples}" for e in integrity.stuck_notes
        )
        reasons.append(f"{STUCK_NOTE_ID}: {detail}")

    # 2. missing-or-extra vs the expected list.
    if diff is not None and (diff.missing or diff.extra):
        reasons.append(
            f"{MISSING_OR_EXTRA_ID}: {len(diff.missing)} missing, "
            f"{len(diff.extra)} extra"
        )

    # 3. wrong-field (EXACT) — names the field and the expected time.
    if diff is not None and diff.wrong_field:
        detail = "; ".join(
            f"{w.field}@{w.expected.t_samples}" for w in diff.wrong_field
        )
        reasons.append(f"{WRONG_FIELD_ID}: {detail}")

    # 4. mistimed (timing gate is SAMPLES).
    if diff is not None and diff.mistimed:
        detail = "; ".join(
            f"@{m.expected.t_samples} delta {m.delta_samples} samples"
            for m in diff.mistimed
        )
        reasons.append(f"{MISTIMED_ID}: {detail}")

    # 5. ordering — note_on before its own note_off at a coincident timestamp.
    if ordering_violations:
        detail = "; ".join(
            f"ch{c} n{n}@{t} note_on before note_off"
            for (c, n, t) in ordering_violations
        )
        reasons.append(f"{ORDERING_ID}: {detail}")

    # offvel0 contract — RED under "red", an informational note under "normalize".
    if offvel0_events:
        if offvel0_policy == "red":
            reasons.append(OFFVEL0_RED_REASON)
        else:
            detail = "; ".join(
                f"ch{e.channel} n{e.note}@{e.t_samples} note_on(vel0) treated as note_off"
                for e in offvel0_events
            )
            reasons.append(f"offvel0-normalized: {detail}")

    return reasons
