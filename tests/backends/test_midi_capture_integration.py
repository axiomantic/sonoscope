"""LIVE ReferenceSequencer.clap integration golden + determinism test.

The live counterpart of the FAKE-host ``test_midi_capture.py`` unit suite: this
module captures the DEMO pattern from the REAL ``ReferenceSequencer.clap`` through the built
C CLAP host and asserts the following acceptance criteria (by design):

1. **Golden PASS** — the DEMO capture (120 bpm / start 0 / 1 beat / 4-4 / 48 kHz)
   analyzed against the committed CORRECT#1 golden
   (``specs/refseq_demo_correct1.json``) yields ``verdict == "PASS"``,
   ``matched == 8``, and a clean integrity block.
2. **Byte-identical repeat** — two captures at the SAME block size produce an
   IDENTICAL decoded event list (exact equality on every field) AND an identical
   ``events_sha256`` (the byte-identity key).
3. **Block-size invariance** — the SAME (pattern, tempo, start, duration) captured
   under TWO different ``block_size`` values produces an IDENTICAL decoded event
   list (the half-open window + scheduler determinism, by design).

Whole module is ``@pytest.mark.integration`` so the default ``pytest -m "not
integration"`` run deselects it. The ``clap_midi_host_path`` + ``refseq_clap_path``
fixtures (``tests/conftest.py``) skip with an EXPLICIT reason when the built host
or the real plugin is absent (AGENTS.md testing discipline: loud skip, never a
silent pass). Capture is crash-isolated in the spawned host, so the three captures
run ONCE at module scope and the acceptance assertions read the cached result.

LIVE-GREEN STATUS (both root causes fixed): all three checks pass live. Two
issues previously drove them ``xfail(strict=True)`` and are now resolved:
(A) a HOST capture-boundary artifact — the C host TRUNCATED its final render
block to end exactly at the window, so for block sizes leaving a partial final
block (512, 128) the beat-1 loop downbeat whose true position is total_frames
(24000) was clamped inward to 23999, landing a spurious 9th event inside the
half-open [0,24000) window (stuck-note trips) and making 480 (which divides
24000 evenly) disagree with 512/128. Fixed HOST-side: the host now renders FULL
blocks one block PAST the window so a full block always spans total_frames, then
filters the capture back to the true [0, total_frames) window before serialize —
the boundary downbeat lands at its true 24000 and is excluded for ALL block
sizes. (B) a Reference Sequencer scheduler ±1-sample block-size-dependent offset, fixed
upstream by a round-to-nearest change (036af12). With both fixes the DEMO
capture is 8 events, byte-identical across 128/480/512 (true block-invariance),
and PASSES the CORRECT#1 golden. This is now a clean live-green gate.

CROSS-REPO NOTE: the real ``ReferenceSequencer.clap`` is consumed READ-ONLY (dlopened, never
modified/rebuilt); this test writes nothing into any external plugin project's repo.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

from sonoscope.backends.midi_capture import (
    MidiCaptureBackend,
    MidiCaptureRequest,
    MidiCaptureResult,
)
from sonoscope.midi_orchestrator import analyze_midi
from sonoscope.schema import MidiAnalysisReport

# Repo root: tests/backends/test_midi_capture_integration.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: The committed CORRECT#1 golden: the 8-event DEMO expectation the live
#: capture is diffed against for the golden-PASS acceptance.
_GOLDEN = _REPO_ROOT / "specs" / "refseq_demo_correct1.json"

# The DEMO golden transport/render parameters (by design): 1 beat @120 BPM /
# 48 kHz = 24000 samples of half-open capture window.
_TEMPO_BPM = 120.0
_START_BEATS = 0.0
_DURATION_BEATS = 1.0
_TSIG_NUM = 4
_TSIG_DEN = 4
_SAMPLE_RATE = 48000

#: The golden/repeat block size and the DIFFERENT block used for the
#: block-invariance acceptance (by design: the decoded list must be identical
#: across block sizes — the half-open window + scheduler determinism).
_BLOCK_SIZE = 512
_BLOCK_SIZE_ALT = 128

#: The number of events in the CORRECT#1 golden (by design acceptance: all matched).
_GOLDEN_MATCHED = 8


def _demo_request(plugin_binary: Path, *, block_size: int) -> MidiCaptureRequest:
    """The DEMO CORRECT#1 capture request at ``block_size`` (by design)."""
    return MidiCaptureRequest(
        plugin_path=plugin_binary,
        tempo_bpm=_TEMPO_BPM,
        start_position_beats=_START_BEATS,
        duration_beats=_DURATION_BEATS,
        tsig_num=_TSIG_NUM,
        tsig_den=_TSIG_DEN,
        sample_rate=_SAMPLE_RATE,
        block_size=block_size,
    )


class _LiveDemo(NamedTuple):
    """The once-per-module live captures the T1 acceptance assertions read."""

    #: analyze_midi against the CORRECT#1 golden (assertion 1: golden PASS).
    report: MidiAnalysisReport
    #: Two captures at the SAME block (assertion 2: byte-identical).
    capture_a: MidiCaptureResult
    capture_b: MidiCaptureResult
    #: A capture at a DIFFERENT block (assertion 3: block-size invariance).
    capture_alt: MidiCaptureResult


@pytest.fixture(scope="module")
def live_demo(
    clap_midi_host_path: Path, refseq_clap_path: Path
) -> _LiveDemo:
    """Capture the DEMO pattern from the REAL ReferenceSequencer.clap ONCE (by design).

    Runs the three live captures the acceptance needs — the golden-diff
    ``analyze_midi`` capture, a same-block repeat, and a different-block capture —
    against the built C host and the real plugin. Each capture spawns the host
    subprocess, so they are computed once at module scope and the acceptance
    assertions read the cached result. Skips (via the fixtures) with an explicit
    reason when the built host or the plugin is absent.
    """
    backend = MidiCaptureBackend(host_path=clap_midi_host_path)
    req = _demo_request(refseq_clap_path, block_size=_BLOCK_SIZE)

    report = analyze_midi(req, backend=backend, expected=_GOLDEN)
    capture_a = backend.capture(req)
    capture_b = backend.capture(req)
    capture_alt = backend.capture(replace(req, block_size=_BLOCK_SIZE_ALT))
    return _LiveDemo(
        report=report,
        capture_a=capture_a,
        capture_b=capture_b,
        capture_alt=capture_alt,
    )


@pytest.mark.integration
def test_demo_capture_matches_correct1_golden(live_demo: _LiveDemo) -> None:
    """Assertion 1: the live DEMO capture PASSES against the CORRECT#1 golden.

    ``analyze_midi`` captures the real plugin's DEMO pattern and diffs it against
    the committed 8-event golden. The verdict is exactly ``"PASS"``, all 8 golden
    events are matched, and the integrity block is clean (every note_on has its
    note_off; no stuck notes or dangling offs). Exact-equality (AGENTS.md Level 4+).
    """
    report = live_demo.report
    midi = report.midi
    # Exact PASS verdict + empty reasons (a truthful clean capture).
    assert midi.verdict == "PASS"
    assert midi.reasons == []
    # The expected-vs-actual diff: all 8 golden events matched, nothing missing,
    # extra, mistimed, or wrong-field (EXACT — the golden is the ground truth).
    eva = midi.expected_vs_actual
    assert eva is not None
    assert eva.matched == _GOLDEN_MATCHED
    assert eva.missing == []
    assert eva.extra == []
    assert eva.mistimed == []
    assert eva.wrong_field == []
    # Integrity clean: every note_on paired, no firewall (stuck) or dangling offs.
    assert midi.integrity.every_note_on_has_off is True
    assert midi.integrity.stuck_notes == []
    assert midi.integrity.dangling_offs == []


@pytest.mark.integration
def test_demo_capture_is_byte_identical_across_repeats(
    live_demo: _LiveDemo,
) -> None:
    """Assertion 2: two same-block captures are BYTE-IDENTICAL.

    The same request captured twice produces an IDENTICAL decoded event list
    (exact equality on every field of every event) AND an identical
    ``events_sha256`` — the byte-identity determinism key. A non-deterministic
    capture would diverge here (by design).
    """
    capture_a = live_demo.capture_a
    capture_b = live_demo.capture_b
    # Exact event-list equality (pydantic MidiEvent compares by all fields).
    assert capture_a.events == capture_b.events
    # And the byte-identity hash agrees (the recorded determinism key).
    assert capture_a.meta.events_sha256 == capture_b.meta.events_sha256


@pytest.mark.integration
def test_demo_capture_is_block_size_invariant(live_demo: _LiveDemo) -> None:
    """Assertion 3: the decoded list is IDENTICAL across two block sizes.

    The SAME (pattern, tempo, start, duration) captured under two DIFFERENT
    ``block_size`` values produces an IDENTICAL decoded event list — the half-open
    window + scheduler determinism, by design. Block size is a
    render-buffering detail that MUST NOT change the captured MIDI.
    """
    assert live_demo.capture_a.events == live_demo.capture_alt.events
