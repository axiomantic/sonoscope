"""LIVE ReferenceSequencer.clap arbitrary-pattern integration goldens.

The arbitrary-pattern counterpart of the single-DEMO ``test_midi_capture_integration.py``:
for the FIRST time this exercises the plugin **state-load** path — each of
the four M4-complete fixtures is loaded into the REAL ``ReferenceSequencer.clap`` via CLAP
``load_state`` (``MidiCaptureRequest.state_b64``) BEFORE the transport is driven,
then the emitted note-out is captured over the half-open window
``[start, start+duration)`` and diffed against the committed goldens.

The four patterns (``tests/backends/fixtures/refseq-t3/`` — see ``PROVENANCE.md``,
delivered from the external fixtures):

- ``two_ring_basic``  — 2 rings, ch9, 4/4, block 512, 8 events.
- ``multi_ring_poly`` — 3 rings (ch 9/0/1), coincident off/on, 4/4, block 480, 20 events.
- ``free_ring_bars``  — 2 rings, 3/4 meter, 6-beat window, block 256, 16 events.
- ``fast_ring_offset``— 2 rings, mid-timeline start (beat 4), 4/4, block 512, 20 events.

For EACH pattern (by design):

1. **Golden PASS** — the real capture analyzed against ``<name>.correct.json``
   yields ``verdict == "PASS"``, ``matched ==`` the pattern's event count, a clean
   integrity block, and empty missing/extra/mistimed/wrong_field. This validates
   the state-load + capture + half-open-window round-trip: the real emission
   equals the scheduler-produced golden byte-for-byte.
2. **Dropped note-off** (``<name>.broken-droppedoff.json``) — RED. NOTE the
   classification: this is a ``missing-or-extra`` RED (exactly one ``extra``
   captured ``note_off``), **not** a ``stuck-note`` integrity RED. The stuck-note
   firewall runs on the captured ground truth (correct and fully paired); a
   note-off dropped from the *expected* spec surfaces as an extra captured
   note_off, not an integrity violation. (the fixture README labels this fault
   "stuck-note" after its real-capture semantics; through the sonoscope
   capture-vs-expected path it is missing-or-extra. Verified live, 017c9d4f.)
3. **Wrong channel** (``<name>.broken-wrongchannel.json``) — RED via a single
   ``wrong_field`` finding on ``channel`` (NOT a spurious missing+extra pair).
4. **Mistimed** (``<name>.broken-mistimed.json``) — RED via a single ``mistimed``
   finding whose ``delta_samples`` magnitude is 50 (> the ±1-sample tolerance).

Whole module is ``@pytest.mark.integration`` so the default ``pytest -m "not
integration"`` run deselects it. The ``clap_midi_host_path`` + ``refseq_clap_path``
fixtures (``tests/conftest.py``) skip with an EXPLICIT reason when the built host
or the real plugin is absent (AGENTS.md testing discipline: loud skip, never a
silent pass). Each pattern is captured ONCE at module scope (each capture spawns
the host subprocess); the four per-pattern diffs replay that single real capture
through ``analyze_midi`` via an injected replay backend, so the state-load +
capture is REAL while the host is spawned only once per pattern.

CROSS-REPO NOTE: the real ``ReferenceSequencer.clap`` and the fixtures are consumed
READ-ONLY; this test writes nothing into any external plugin project's repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pytest

from sonoscope.backends.midi_capture import (
    MidiCaptureBackend,
    MidiCaptureRequest,
    MidiCaptureResult,
)
from sonoscope.features.midi_tripwires import (
    MISSING_OR_EXTRA_ID,
    MISTIMED_ID,
    WRONG_FIELD_ID,
)
from sonoscope.midi_input import load_expected_events
from sonoscope.midi_orchestrator import analyze_midi

pytestmark = pytest.mark.integration

#: The fixture directory (pinned goldens; see PROVENANCE.md).
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "refseq-t3"

#: The four arbitrary patterns exercised (by design).
_PATTERNS = (
    "two_ring_basic",
    "multi_ring_poly",
    "free_ring_bars",
    "fast_ring_offset",
)

#: The mistimed fault's injected shift magnitude in SAMPLES (+50 is injected on
#: one expected event; the capture is on-time, so |actual - expected| == 50,
#: which exceeds the ±1-sample timing tolerance -> a mistimed finding).
_MISTIMED_SHIFT_SAMPLES = 50


def _request_for(name: str, plugin_binary: Path) -> MidiCaptureRequest:
    """Build the capture request for pattern ``name`` (transport + state_b64).

    Reads ``<name>.transport.json`` for the driving transport and
    ``<name>.state.b64`` for the SQBT v2 ``clap.state`` blob; the blob is carried
    on ``MidiCaptureRequest.state_b64`` so the C host loads it via CLAP
    ``load_state`` BEFORE driving the transport (the T3 state-load path).
    """
    transport = json.loads((_FIXTURE_DIR / f"{name}.transport.json").read_text())
    state_b64 = (_FIXTURE_DIR / f"{name}.state.b64").read_text().strip()
    return MidiCaptureRequest(
        plugin_path=plugin_binary,
        tempo_bpm=transport["tempo_bpm"],
        start_position_beats=transport["start_position_beats"],
        duration_beats=transport["duration_beats"],
        tsig_num=transport["tsig_num"],
        tsig_den=transport["tsig_den"],
        sample_rate=transport["sample_rate"],
        block_size=transport["block_size"],
        state_b64=state_b64,
    )


class _ReplayBackend:
    """Return a pre-captured REAL :class:`MidiCaptureResult` (capture once, diff many).

    The state-load + capture happens ONCE per pattern against the real host/plugin
    (module scope); this replays that exact real result into ``analyze_midi`` so
    the four per-pattern expected-diffs share the one real capture without
    re-spawning the host. It is NOT a synthetic FAKE — the events it returns are
    the genuine plugin emission.
    """

    def __init__(self, result: MidiCaptureResult) -> None:
        self._result = result

    def capture(self, req: MidiCaptureRequest) -> MidiCaptureResult:
        return self._result


class _Pattern(NamedTuple):
    """A single pattern's captured-once request + real capture + golden count."""

    req: MidiCaptureRequest
    result: MidiCaptureResult
    #: The correct.json event count (via E2 load_expected_events; no magic number).
    expected_count: int


@pytest.fixture(scope="module")
def t3_captures(
    clap_midi_host_path: Path, refseq_clap_path: Path
) -> dict[str, _Pattern]:
    """Capture all four patterns from the REAL ReferenceSequencer.clap ONCE (by design).

    For each pattern: load its SQBT state into the plugin (via ``state_b64`` ->
    CLAP ``load_state``), drive the transport, and capture the note-out. Each
    capture spawns the host subprocess, so they run once at module scope and the
    acceptance assertions replay the cached real result. Skips (via the fixtures)
    with an explicit reason when the built host or the plugin is absent.

    The golden event count is loaded from ``<name>.correct.json`` via
    :func:`load_expected_events` (the goldens carry both time axes, so no
    ``sample_rate``/``tempo_bpm`` derivation is needed) — the ``matched``
    assertion checks against the real spec length, not a hardcoded number.
    """
    backend = MidiCaptureBackend(host_path=clap_midi_host_path)
    captures: dict[str, _Pattern] = {}
    for name in _PATTERNS:
        req = _request_for(name, refseq_clap_path)
        result = backend.capture(req)
        correct = load_expected_events(_FIXTURE_DIR / f"{name}.correct.json")
        captures[name] = _Pattern(
            req=req, result=result, expected_count=len(correct)
        )
    return captures


@pytest.mark.parametrize("name", _PATTERNS)
def test_pattern_capture_matches_correct_golden(
    t3_captures: dict[str, _Pattern], name: str
) -> None:
    """Acceptance 1: the real capture PASSES against ``<name>.correct.json``.

    Validates the state-load + capture + half-open-window round-trip end to end:
    the plugin's emission for the loaded pattern equals the scheduler-produced
    golden EXACTLY — ``verdict == "PASS"``, all golden events matched, clean
    integrity, and empty missing/extra/mistimed/wrong_field (AGENTS.md Level 4+
    exact equality; the golden is ground truth).
    """
    pat = t3_captures[name]
    report = analyze_midi(
        pat.req,
        backend=_ReplayBackend(pat.result),
        expected=_FIXTURE_DIR / f"{name}.correct.json",
    )
    midi = report.midi
    assert midi.verdict == "PASS"
    assert midi.reasons == []
    eva = midi.expected_vs_actual
    assert eva is not None
    assert eva.matched == pat.expected_count
    assert len(midi.events) == pat.expected_count
    assert eva.missing == []
    assert eva.extra == []
    assert eva.mistimed == []
    assert eva.wrong_field == []
    assert midi.integrity.every_note_on_has_off is True
    assert midi.integrity.stuck_notes == []
    assert midi.integrity.dangling_offs == []


@pytest.mark.parametrize("name", _PATTERNS)
def test_pattern_dropped_note_off_trips_red(
    t3_captures: dict[str, _Pattern], name: str
) -> None:
    """Acceptance 2: a dropped note-off in the expected spec trips RED.

    The capture is correct (fully paired); the ``broken-droppedoff`` expected has
    one ``note_off`` removed, so the captured note_off has no expected counterpart
    -> exactly one ``extra`` (a ``note_off``) and a ``missing-or-extra`` RED.

    CLASSIFICATION: this is ``missing-or-extra``, NOT ``stuck-note``. The
    stuck-note firewall pairs the CAPTURED ground truth (correct here, so no stuck
    note); the dropped-off lives in the EXPECTED spec and therefore surfaces as an
    extra captured note_off. The fixture README names the fault "stuck-note"
    after its real-capture semantics; through the sonoscope capture-vs-expected
    path it is a missing-or-extra RED (verified live against build 017c9d4f).
    """
    pat = t3_captures[name]
    report = analyze_midi(
        pat.req,
        backend=_ReplayBackend(pat.result),
        expected=_FIXTURE_DIR / f"{name}.broken-droppedoff.json",
    )
    midi = report.midi
    assert midi.verdict == "RED"
    # A missing-or-extra tripwire fired (the dropped-off is caught).
    assert any(r.startswith(MISSING_OR_EXTRA_ID) for r in midi.reasons)
    eva = midi.expected_vs_actual
    assert eva is not None
    # Exactly one EXTRA captured event, and it is the un-dropped note_off.
    assert eva.missing == []
    assert len(eva.extra) == 1
    assert eva.extra[0].type == "note_off"
    assert eva.matched == pat.expected_count - 1
    # The capture itself is clean: the fault is a spec defect, not a stuck note.
    assert midi.integrity.stuck_notes == []


@pytest.mark.parametrize("name", _PATTERNS)
def test_pattern_wrong_channel_trips_red(
    t3_captures: dict[str, _Pattern], name: str
) -> None:
    """Acceptance 3: a wrong channel trips RED via a single wrong_field finding.

    The ``broken-wrongchannel`` expected has ONE event's ``channel`` changed. The
    comparator aligns it to its nearest-time single-field-off counterpart, so the
    result is exactly one ``wrong_field`` finding on ``channel`` — NOT a spurious
    ``missing`` + ``extra`` pair.
    """
    pat = t3_captures[name]
    report = analyze_midi(
        pat.req,
        backend=_ReplayBackend(pat.result),
        expected=_FIXTURE_DIR / f"{name}.broken-wrongchannel.json",
    )
    midi = report.midi
    assert midi.verdict == "RED"
    assert any(r.startswith(WRONG_FIELD_ID) for r in midi.reasons)
    eva = midi.expected_vs_actual
    assert eva is not None
    assert len(eva.wrong_field) == 1
    assert eva.wrong_field[0].field == "channel"
    # Disambiguated to wrong-field, not split into missing+extra.
    assert eva.missing == []
    assert eva.extra == []
    assert eva.matched == pat.expected_count - 1


@pytest.mark.parametrize("name", _PATTERNS)
def test_pattern_mistimed_trips_red(
    t3_captures: dict[str, _Pattern], name: str
) -> None:
    """Acceptance 4: a mistimed event trips RED via a single mistimed finding.

    The ``broken-mistimed`` expected has ONE event's ``t_samples`` shifted +50.
    The capture is on-time, so ``|actual - expected| == 50`` samples, which
    exceeds the ±1-sample timing tolerance -> exactly one ``mistimed`` finding
    (same channel/note/type/velocity, off only in time).
    """
    pat = t3_captures[name]
    report = analyze_midi(
        pat.req,
        backend=_ReplayBackend(pat.result),
        expected=_FIXTURE_DIR / f"{name}.broken-mistimed.json",
    )
    midi = report.midi
    assert midi.verdict == "RED"
    assert any(r.startswith(MISTIMED_ID) for r in midi.reasons)
    eva = midi.expected_vs_actual
    assert eva is not None
    assert len(eva.mistimed) == 1
    assert abs(eva.mistimed[0].delta_samples) == _MISTIMED_SHIFT_SAMPLES
    assert eva.missing == []
    assert eva.extra == []
    assert eva.matched == pat.expected_count - 1
