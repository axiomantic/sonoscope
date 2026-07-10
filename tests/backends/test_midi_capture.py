"""Tests for the ``MidiCaptureBackend`` (by design).

NON-integration: no real ``clap_midi_host`` binary and no ReferenceSequencer.clap. Every
test spawns a **FAKE** host — a tiny module-level python script the backend runs
by path — that drains stdin and echoes a canned JSON response (its stdout body
and exit code come from env vars the ``fake_host`` fixture sets). This exercises
the real ``subprocess`` spawn + the full decode/window/order/error pipeline
without the plugin, so the green-mirage suite is buildable now.

The fake source is a MODULE-LEVEL string (spawn-safe: it is written to a file and
executed as a standalone process, so it needs no import of this test module).

Coverage:
- ``test_decode_note_on_off`` — 0x90→note_on, 0x80→note_off; channel/note/vel.
- ``test_0x90_vel0_decodes_as_note_on_not_normalized`` — faithful, NO normalize.
- ``test_ticks_computed_at_960ppq`` — exact t_ticks from a known t_samples/sr/tempo.
- ``test_window_excludes_end_boundary_event`` — half-open ``[0, dur)``.
- ``test_coincident_ordering_off_before_on`` — off sorts before on at same sample.
- ``test_host_error_outcome_becomes_midi_capture_error`` — mapped, not silent empty.
- ``test_host_crash_becomes_midi_capture_error`` — nonzero/garbage/empty → mapped.
- ``test_playing_false_empty_is_valid_success`` — empty capture IS a valid success.
"""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest

from sonoscope.backends.midi_capture import (
    MIDI_CAPTURE_HOST_ERROR,
    MIDI_CAPTURE_SUBPROCESS_CRASH,
    MidiCaptureBackend,
    MidiCaptureRequest,
)
from sonoscope.errors import MidiCaptureError
from sonoscope.schema import ExitCode

# --- MIDI status bytes (channel in the low nibble) --------------------------
NOTE_ON = 0x90
NOTE_OFF = 0x80

# 1 beat @120 BPM / 48 kHz = 0.5 s = 24000 samples (the DEMO capture window).
_SR = 48000
_TEMPO = 120.0
_DUR_SAMPLES = 24000

# --- FAKE host: a standalone script the backend spawns by path --------------
# It drains stdin, writes the file named by SONOSCOPE_FAKE_HOST_STDOUT_FILE to
# stdout, and exits SONOSCOPE_FAKE_HOST_EXIT. Kept dependency-free + top-level so
# it runs as its own process with no import of this test module.
_FAKE_HOST_SOURCE = """\
#!/usr/bin/env python3
import os
import sys

sys.stdin.buffer.read()  # drain the request so the backend's write completes
stdout_file = os.environ.get("SONOSCOPE_FAKE_HOST_STDOUT_FILE", "")
if stdout_file:
    with open(stdout_file, "rb") as handle:
        sys.stdout.buffer.write(handle.read())
    sys.stdout.buffer.flush()
sys.exit(int(os.environ.get("SONOSCOPE_FAKE_HOST_EXIT", "0")))
"""


@pytest.fixture
def fake_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a factory building a backend wired to a configurable FAKE host.

    ``make(stdout_obj=..., stdout_text=..., exit_code=0)`` writes the canned
    response to a file, points the fake at it via env, and returns a
    ``MidiCaptureBackend`` whose ``host_path`` is the fake script.
    """
    script = tmp_path / "fake_clap_midi_host"
    script.write_text(_FAKE_HOST_SOURCE)
    script.chmod(0o755)

    def make(
        stdout_obj: Optional[dict[str, Any]] = None,
        *,
        stdout_text: Optional[str] = None,
        exit_code: int = 0,
    ) -> MidiCaptureBackend:
        if stdout_text is None:
            stdout_text = json.dumps(stdout_obj)
        response = tmp_path / "response.json"
        response.write_text(stdout_text)
        monkeypatch.setenv("SONOSCOPE_FAKE_HOST_STDOUT_FILE", str(response))
        monkeypatch.setenv("SONOSCOPE_FAKE_HOST_EXIT", str(exit_code))
        return MidiCaptureBackend(host_path=script)

    return make


def _request(**overrides: Any) -> MidiCaptureRequest:
    base: dict[str, Any] = dict(
        plugin_path=Path("/nonexistent/ReferenceSequencer.clap"),
        tempo_bpm=_TEMPO,
        start_position_beats=0.0,
        duration_beats=1.0,
        tsig_num=4,
        tsig_den=4,
        sample_rate=_SR,
        block_size=512,
    )
    base.update(overrides)
    return MidiCaptureRequest(**base)


def _success(events: list[dict[str, Any]], *, duration_samples: int = _DUR_SAMPLES) -> dict[str, Any]:
    return {
        "outcome": "success",
        "events": events,
        "meta": {
            "sample_rate": _SR,
            "block_size": 512,
            "duration_samples": duration_samples,
            "plugin_id": "com.example.reference-sequencer",
            "plugin_name": "Reference Sequencer",
            "n_events": len(events),
        },
    }


def _ev(t_samples: int, status: int, d1: int, d2: int) -> dict[str, Any]:
    return {"t_samples": t_samples, "midi": [status, d1, d2]}


# --- decode ------------------------------------------------------------------


def test_decode_note_on_off(fake_host) -> None:
    backend = fake_host(
        _success(
            [
                _ev(0, NOTE_ON | 1, 60, 100),
                _ev(100, NOTE_OFF | 2, 62, 40),
            ]
        )
    )

    result = backend.capture(_request())

    assert len(result.events) == 2
    on = result.events[0]
    assert on.type == "note_on"
    assert on.channel == 1
    assert on.note == 60
    assert on.velocity == 100
    assert on.t_samples == 0
    off = result.events[1]
    assert off.type == "note_off"
    assert off.channel == 2
    assert off.note == 62
    assert off.velocity == 40
    assert off.t_samples == 100
    assert result.meta.source == "plugin"
    assert result.meta.plugin_id == "com.example.reference-sequencer"
    assert result.meta.plugin_name == "Reference Sequencer"
    assert result.meta.events_sha256 is not None


def test_0x90_vel0_decodes_as_note_on_not_normalized(fake_host) -> None:
    # RED-proving the decoder does NOT normalize a 0x90 vel0 to note_off. The E1
    # comparator flags that contract violation later; the decoder stays faithful.
    backend = fake_host(_success([_ev(0, NOTE_ON, 64, 0)]))

    result = backend.capture(_request())

    assert len(result.events) == 1
    assert result.events[0].type == "note_on"
    assert result.events[0].velocity == 0


def test_ticks_computed_at_960ppq(fake_host) -> None:
    # t_samples=12000, sr=48000, tempo=120 -> 0.25 s -> 0.5 beat -> 480 ticks.
    backend = fake_host(_success([_ev(12000, NOTE_ON, 36, 100)]))

    result = backend.capture(_request())

    assert result.events[0].t_ticks == 480


# --- half-open window --------------------------------------------------------


def test_window_excludes_end_boundary_event(fake_host) -> None:
    # RED-proving the half-open rule: an event AT t_samples == duration_samples
    # (the trailing window-end downbeat) is EXCLUDED; the one at dur-1 is kept.
    backend = fake_host(
        _success(
            [
                _ev(_DUR_SAMPLES - 1, NOTE_ON, 36, 100),  # kept
                _ev(_DUR_SAMPLES, NOTE_ON, 48, 100),  # excluded (boundary)
            ],
            duration_samples=_DUR_SAMPLES,
        )
    )

    result = backend.capture(_request())

    assert len(result.events) == 1
    assert result.events[0].note == 36
    assert result.events[0].t_samples == _DUR_SAMPLES - 1


# --- canonical ordering ------------------------------------------------------


def test_coincident_ordering_off_before_on(fake_host) -> None:
    # Host emits an on FIRST, then an off, both at the same t_samples. Canonical
    # order must reorder so note_off sorts BEFORE note_on (by contract).
    backend = fake_host(
        _success(
            [
                _ev(6000, NOTE_ON, 36, 100),
                _ev(6000, NOTE_OFF, 36, 0),
            ]
        )
    )

    result = backend.capture(_request())

    assert [e.type for e in result.events] == ["note_off", "note_on"]
    assert result.events[0].t_samples == result.events[1].t_samples == 6000


# --- crash / error mapping (mirrors subprocess_render) -----------------------


def test_host_error_outcome_becomes_midi_capture_error(fake_host) -> None:
    # A host outcome=error (comes WITH a nonzero exit, by design) -> mapped error,
    # not a silent empty capture.
    backend = fake_host(
        {"outcome": "error", "error": {"code": "dlopen_failed", "message": "boom"}},
        exit_code=1,
    )

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_HOST_ERROR
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail == {"host_code": "dlopen_failed"}


@pytest.mark.parametrize(
    "stdout_text, exit_code",
    [
        ("not json at all {{{", 3),  # nonzero exit + garbage stdout
        ("", 0),  # clean exit but NO output: must NOT be a silent empty success
    ],
)
def test_host_crash_becomes_midi_capture_error(
    fake_host, stdout_text: str, exit_code: int
) -> None:
    backend = fake_host(stdout_text=stdout_text, exit_code=exit_code)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER


def test_playing_false_empty_is_valid_success(fake_host) -> None:
    # A genuinely empty capture (e.g. playing:false) is a VALID success, NOT an
    # error: outcome=success + events:[] + exit 0.
    backend = fake_host(_success([]))

    result = backend.capture(_request(playing=False))

    assert result.events == []
    assert result.meta.source == "plugin"
    assert result.meta.events_sha256 is not None


# --- malformed event (bad shape / out-of-range byte -> crash, never a drop) ---
# Forces MidiCaptureBackend._decode_one's three malformed-event exits
# (not-an-object, bad t_samples/midi shape, out-of-range ValidationError). Each
# is a corrupted host handoff mapped to MIDI_CAPTURE_SUBPROCESS_CRASH
# (detail reason "malformed_event") -- NEVER a raw KeyError/ValidationError and
# NEVER a silent drop (contrast the deliberate non-note SKIP below).
@pytest.mark.parametrize(
    "bad_event, why_fragment",
    [
        # out-of-range data byte: note 200 > 127 -> MidiEvent ValidationError branch.
        (_ev(0, NOTE_ON, 200, 100), "out-of-range field"),
        # velocity 128 > 127 -> same ValidationError branch, different field.
        (_ev(0, NOTE_ON, 60, 128), "out-of-range field"),
        # wrong-length midi triple (2 bytes, not 3) -> shape-check branch.
        ({"t_samples": 0, "midi": [NOTE_ON, 60]}, "bad t_samples/midi shape"),
        # non-int data byte in the triple -> shape-check branch.
        ({"t_samples": 0, "midi": [NOTE_ON, 60, "loud"]}, "bad t_samples/midi shape"),
        # missing t_samples (None is not an int) -> shape-check branch.
        ({"midi": [NOTE_ON, 60, 100]}, "bad t_samples/midi shape"),
        # event is not an object at all -> not-a-dict branch.
        ([NOTE_ON, 60, 100], "event is not an object"),
    ],
)
def test_malformed_event_becomes_crash(fake_host, bad_event, why_fragment) -> None:
    backend = fake_host(_success([bad_event]))

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "malformed_event"
    assert why_fragment in err.detail["why"]


# --- malformed success payload (missing events/meta / non-numeric meta) -------
# Forces MidiCaptureBackend._extract_success's corrupted-handoff exits. A success
# object the parent cannot rebuild maps to MIDI_CAPTURE_SUBPROCESS_CRASH
# (detail reason "malformed_success") -- never a raw KeyError, never a silent green.
def _success_missing(key: str) -> dict[str, Any]:
    obj = _success([_ev(0, NOTE_ON, 60, 100)])
    del obj[key]
    return obj


def _success_meta_nonnumeric(field: str) -> dict[str, Any]:
    obj = _success([_ev(0, NOTE_ON, 60, 100)])
    obj["meta"][field] = "not-a-number"
    return obj


@pytest.mark.parametrize(
    "bad_success, why_fragment",
    [
        (_success_missing("events"), "missing events/meta"),
        (_success_missing("meta"), "missing events/meta"),
        (_success_meta_nonnumeric("sample_rate"), "meta.sample_rate"),
        (_success_meta_nonnumeric("block_size"), "meta.block_size"),
        (_success_meta_nonnumeric("duration_samples"), "meta.duration_samples"),
    ],
)
def test_malformed_success_becomes_crash(fake_host, bad_success, why_fragment) -> None:
    backend = fake_host(bad_success)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "malformed_success"
    assert why_fragment in err.detail["why"]


def test_meta_bool_is_not_numeric(fake_host) -> None:
    # A JSON bool is an int subclass in Python; the guard explicitly rejects bool
    # so ``sample_rate: true`` is a malformed handoff, not a "numeric" 1.
    obj = _success([_ev(0, NOTE_ON, 60, 100)])
    obj["meta"]["sample_rate"] = True

    backend = fake_host(obj)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    assert excinfo.value.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert excinfo.value.detail is not None
    assert excinfo.value.detail["reason"] == "malformed_success"


# --- subprocess-level faults (timeout / missing binary / signalled) ----------


def test_timeout_becomes_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # A host that never returns must time out into MIDI_CAPTURE_SUBPROCESS_CRASH
    # (detail reason "timeout"), not hang. Injected by monkeypatching the real
    # subprocess.run to raise TimeoutExpired -- no wall-clock wait in the test.
    def _raise_timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["clap_midi_host"], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    backend = MidiCaptureBackend(host_path=Path("/unused/clap_midi_host"), timeout_s=0.5)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "timeout"
    assert err.detail["timeout_s"] == 0.5


def test_missing_host_binary_becomes_crash(tmp_path: Path) -> None:
    # The host binary is absent (never built): subprocess.run raises a real
    # FileNotFoundError, mapped to MIDI_CAPTURE_SUBPROCESS_CRASH (reason "not_found").
    missing = tmp_path / "clap_midi_host_does_not_exist"
    backend = MidiCaptureBackend(host_path=missing)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "not_found"
    assert err.detail["host_path"] == str(missing)


# A FAKE host that drains stdin then kills itself with SIGTERM, so subprocess.run
# reports a NEGATIVE returncode (-15). Exercises the real signalled-exit path
# through _crash_error, including the signal.Signals name lookup.
_FAKE_HOST_SIGNAL_SOURCE = """\
#!/usr/bin/env python3
import os
import signal
import sys

sys.stdin.buffer.read()  # drain the request so the backend's write completes
os.kill(os.getpid(), signal.SIGTERM)
"""


def test_signalled_exit_becomes_crash(tmp_path: Path) -> None:
    script = tmp_path / "fake_signal_host"
    script.write_text(_FAKE_HOST_SIGNAL_SOURCE)
    script.chmod(0o755)
    backend = MidiCaptureBackend(host_path=script)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "signalled"
    assert err.detail["exitcode"] == -int(signal.SIGTERM)
    assert err.detail["signal"] == "SIGTERM"


# --- non-note status: deliberate SKIP (never a silent WRONG decode) ----------


def test_non_note_status_is_skipped_not_decoded(fake_host) -> None:
    # A raw event whose status is NOT a note-on/note-off (0xB0 control-change /
    # 0xF8 clock) is intentionally DROPPED by the decoder (returns None), NOT
    # mis-decoded into a note event. The genuine note in the same payload survives,
    # proving the skip is targeted, not a swallow-everything.
    backend = fake_host(
        _success(
            [
                _ev(0, 0xB0, 7, 100),  # CC #7 (volume): non-note -> skipped
                _ev(100, NOTE_ON, 60, 90),  # real note -> kept
                _ev(200, 0xF8, 0, 0),  # MIDI clock: non-note -> skipped
            ]
        )
    )

    result = backend.capture(_request())

    assert len(result.events) == 1
    kept = result.events[0]
    assert kept.type == "note_on"
    assert kept.note == 60
    assert kept.velocity == 90
    assert kept.t_samples == 100


# --- success outcome on a NONZERO exit: a crash, not a fake success ----------


def test_success_outcome_on_nonzero_exit_is_crash(fake_host) -> None:
    # A well-formed outcome=success payload that arrives with a NONZERO exit code
    # is NOT trusted: _spawn requires BOTH outcome=success AND returncode==0, so
    # this falls through to _crash_error (reason "nonzero_exit"), never a fake green.
    backend = fake_host(_success([_ev(0, NOTE_ON, 60, 100)]), exit_code=1)

    with pytest.raises(MidiCaptureError) as excinfo:
        backend.capture(_request())

    err = excinfo.value
    assert err.code == MIDI_CAPTURE_SUBPROCESS_CRASH
    assert err.component == "midi"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail is not None
    assert err.detail["reason"] == "nonzero_exit"
    assert err.detail["exitcode"] == 1
