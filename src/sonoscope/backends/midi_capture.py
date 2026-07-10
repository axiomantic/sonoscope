"""B1 ``MidiCaptureBackend`` — spawn the C CLAP MIDI host, decode, canonicalize.

Task B1 (by design, "Python ``MidiCaptureBackend``").

The backend builds the C-host request JSON from a :class:`MidiCaptureRequest`,
spawns ``build/clap_midi_host`` as a **subprocess** (a foreign binary, not a
pickled Python target), feeds the request on stdin, and reads the single JSON
object on stdout. It MIRRORS ``subprocess_render.py``'s crash-isolation +
outcome-discrimination discipline: the ONLY path that returns a result is a
clean ``outcome:"success"`` object from an exit-0 child; a host ``outcome:"error"``,
a nonzero/signalled exit, or missing/unparseable stdout maps to a structured
:class:`MidiCaptureError` — **never a silent empty capture**. (A genuinely empty
capture — e.g. ``playing:false`` — is a valid ``outcome:"success"`` with
``events:[]`` and is returned as such.)

All *semantic* decoding lives here (the C host stays semantically dumb, by design):
raw ``[status, d1, d2]`` → a typed :class:`~sonoscope.schema.MidiEvent`, PPQ
ticks, the half-open capture window, and the canonical ordering. Decoding is
FAITHFUL: a ``0x90`` with velocity 0 decodes as ``note_on`` velocity 0 (it is
NOT normalized to ``note_off`` — flagging that contract violation is the E1
comparator's job, not the decoder's).
"""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from sonoscope.backends.pedalboard_vst3 import binary_sha256
from sonoscope.errors import MidiCaptureError
from sonoscope.schema import MidiCaptureMeta, MidiEvent

_COMPONENT = "midi"

#: Repo root (src/sonoscope/backends/midi_capture.py -> parents[3]) for the
#: repo-relative default host binary path (mirrors ``corpus._REPO_ROOT``).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default C-host binary; built (gitignored) by ``scripts/build_clap_midi_host.sh``.
#: Injectable via the backend constructor so tests can point at a FAKE host.
DEFAULT_HOST_PATH = _REPO_ROOT / "build" / "clap_midi_host"

#: Pulses-per-quarter-note for the ``t_ticks`` decode (by design).
PPQ = 960

#: ``outcome`` discriminator values (mirror the C host stdout contract).
OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"

# Error codes (component == "midi").
#: The C host reported a clean ``outcome:"error"`` (dlopen/activate/process fault).
MIDI_CAPTURE_HOST_ERROR = "MIDI_CAPTURE_HOST_ERROR"
#: Hard crash: nonzero/signalled exit, missing/unparseable stdout, unknown/absent
#: outcome, a success payload the parent cannot rebuild, or a malformed event.
MIDI_CAPTURE_SUBPROCESS_CRASH = "MIDI_CAPTURE_SUBPROCESS_CRASH"

#: MIDI status high-nibble (the low nibble carries the channel).
_STATUS_NOTE_OFF = 0x80
_STATUS_NOTE_ON = 0x90

#: Cap on stderr captured into a crash-error detail (keep the envelope bounded).
_STDERR_DETAIL_CAP = 2000

#: Default subprocess timeout (s). A capture is a short, bounded render.
_DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class MidiCaptureRequest:
    """A single capture request (the C-host input, in Python terms)."""

    plugin_path: Path
    tempo_bpm: float
    start_position_beats: float
    duration_beats: float
    tsig_num: int
    tsig_den: int
    sample_rate: int
    block_size: int
    plugin_id: Optional[str] = None
    playing: bool = True
    state_b64: Optional[str] = None


@dataclass
class MidiCaptureResult:
    """The backend's structured result: canonical events + populated meta."""

    events: list[MidiEvent]
    meta: MidiCaptureMeta


class MidiCaptureBackend:
    """Spawn the C CLAP MIDI host and turn one capture into typed events + meta.

    ``host_path`` is injectable (tests point it at a FAKE host script); it
    defaults to the repo-relative :data:`DEFAULT_HOST_PATH`.
    """

    def __init__(
        self,
        host_path: Path | str = DEFAULT_HOST_PATH,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.host_path = Path(host_path)
        self.timeout_s = timeout_s

    # --- public API ----------------------------------------------------------

    def capture(self, req: MidiCaptureRequest) -> MidiCaptureResult:
        """Run one capture: spawn → discriminate → decode → window → order → meta."""
        raw_events, host_meta = self._spawn(self._build_input(req))

        sample_rate = int(host_meta["sample_rate"])
        duration_samples = int(host_meta["duration_samples"])

        # Decode raw -> typed (faithful; by design), then apply the half-open window and
        # the canonical order. Order matters: decode assigns t_ticks, the window
        # filters on t_samples, the sort is the canonical contract order.
        decoded = self._decode_events(
            raw_events, sample_rate=sample_rate, tempo_bpm=req.tempo_bpm
        )
        windowed = self._apply_window(decoded, duration_samples)
        ordered = self._canonical_sort(windowed)

        return MidiCaptureResult(
            events=ordered, meta=self._build_meta(req, host_meta, ordered)
        )

    # --- request construction ------------------------------------------------

    @staticmethod
    def _build_input(req: MidiCaptureRequest) -> dict[str, Any]:
        """Build the C-host stdin object from the request."""
        payload: dict[str, Any] = {
            "plugin_path": str(req.plugin_path),
            "render": {
                "sample_rate": req.sample_rate,
                "block_size": req.block_size,
            },
            "transport": {
                "tempo_bpm": req.tempo_bpm,
                "start_position_beats": req.start_position_beats,
                "duration_beats": req.duration_beats,
                "tsig_num": req.tsig_num,
                "tsig_den": req.tsig_den,
                "playing": req.playing,
            },
        }
        # Optional fields are omitted (not null) so the host sees exactly the
        # keys it treats as present.
        if req.plugin_id is not None:
            payload["plugin_id"] = req.plugin_id
        if req.state_b64 is not None:
            payload["state_b64"] = req.state_b64
        return payload

    # --- spawn + outcome discrimination (mirrors subprocess_render.py) --------

    def _spawn(self, payload: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Spawn the host, feed stdin, and discriminate the stdout contract.

        Returns ``(raw_events, host_meta)`` ONLY on a clean ``outcome:"success"``
        from an exit-0 child. Every other path raises a mapped
        :class:`MidiCaptureError`; a silent empty success is therefore impossible.
        """
        input_bytes = json.dumps(payload).encode("utf-8")
        try:
            proc = subprocess.run(
                [str(self.host_path)],
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise MidiCaptureError(
                MIDI_CAPTURE_SUBPROCESS_CRASH,
                f"clap_midi_host binary not found at {self.host_path}",
                detail={"host_path": str(self.host_path), "reason": "not_found"},
                component=_COMPONENT,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MidiCaptureError(
                MIDI_CAPTURE_SUBPROCESS_CRASH,
                f"clap_midi_host timed out after {self.timeout_s}s",
                detail={"reason": "timeout", "timeout_s": self.timeout_s},
                component=_COMPONENT,
            ) from exc

        stdout_text = proc.stdout.decode("utf-8", errors="replace").strip()
        parsed = self._parse_stdout(stdout_text)

        if parsed is not None:
            outcome = parsed.get("outcome")
            # A clean host-reported error (it comes WITH a nonzero exit by design):
            # map it regardless of exit code so its code/message are preserved.
            if outcome == OUTCOME_ERROR:
                raise self._host_error(parsed.get("error"))
            # The ONLY success path: an ``outcome:"success"`` object AND exit 0.
            if outcome == OUTCOME_SUCCESS and proc.returncode == 0:
                return self._extract_success(parsed)

        # Everything else — nonzero/signalled exit without an error object, empty
        # or unparseable stdout, an unknown/absent outcome, or a success outcome
        # on a nonzero exit — is a hard crash, NEVER a silent empty capture.
        raise self._crash_error(proc)

    @staticmethod
    def _parse_stdout(stdout_text: str) -> Optional[dict[str, Any]]:
        """Parse the host's single JSON object; ``None`` if empty/unparseable."""
        if not stdout_text:
            return None
        try:
            obj = json.loads(stdout_text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def _extract_success(
        self, parsed: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """Pull ``events`` + ``meta`` from a success object; crash if malformed.

        A success payload the parent cannot use (missing ``events``/``meta`` or a
        meta lacking the numeric render fields) is a corrupted handoff mapped to a
        crash error (mirrors ``subprocess_render._malformed_success_error``) —
        never a raw ``KeyError`` and never a silent green.
        """
        events = parsed.get("events")
        meta = parsed.get("meta")
        if not isinstance(events, list) or not isinstance(meta, dict):
            raise self._malformed_success_error("missing events/meta")
        for key in ("sample_rate", "block_size", "duration_samples"):
            value = meta.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise self._malformed_success_error(f"meta.{key} missing or non-numeric")
        return events, meta

    # --- decode (raw -> typed MidiEvent) -------------------------------------

    def _decode_events(
        self, raw_events: list[Any], *, sample_rate: int, tempo_bpm: float
    ) -> list[MidiEvent]:
        """Decode each raw host event; skip non-note statuses (documented)."""
        decoded: list[MidiEvent] = []
        for raw in raw_events:
            event = self._decode_one(
                raw, sample_rate=sample_rate, tempo_bpm=tempo_bpm
            )
            if event is not None:
                decoded.append(event)
        return decoded

    def _decode_one(
        self, raw: Any, *, sample_rate: int, tempo_bpm: float
    ) -> Optional[MidiEvent]:
        """Decode one ``{t_samples, midi:[b0,b1,b2]}`` to a typed event.

        - ``0x90 | ch`` → ``note_on``; ``0x80 | ch`` → ``note_off``
          (channel = ``b0 & 0x0F``, note = ``b1``, velocity = ``b2``).
        - A ``0x90`` with velocity 0 decodes FAITHFULLY as ``note_on`` velocity 0
          (NO normalization to note_off — the E1 comparator flags that, not us).
        - Any other (non-note) status is skipped and returns ``None``: the host
          only emits note-output events, but a future dialect/CC could appear and
          must not crash the decode.
        """
        if not isinstance(raw, dict):
            raise self._malformed_event_error(raw, "event is not an object")
        t_samples = raw.get("t_samples")
        midi = raw.get("midi")
        if (
            not isinstance(t_samples, int)
            or isinstance(t_samples, bool)
            or not isinstance(midi, list)
            or len(midi) != 3
            or not all(isinstance(b, int) and not isinstance(b, bool) for b in midi)
        ):
            raise self._malformed_event_error(raw, "bad t_samples/midi shape")

        b0, b1, b2 = int(midi[0]), int(midi[1]), int(midi[2])
        status = b0 & 0xF0
        channel = b0 & 0x0F
        if status == _STATUS_NOTE_ON:
            event_type = "note_on"
        elif status == _STATUS_NOTE_OFF:
            event_type = "note_off"
        else:
            return None  # non-note status: not a MIDI-emission event, skip

        t_ticks = _ticks_at_960(t_samples, sample_rate, tempo_bpm)
        try:
            return MidiEvent(
                t_samples=t_samples,
                t_ticks=t_ticks,
                type=event_type,
                channel=channel,
                note=b1,
                velocity=b2,
            )
        except ValidationError as exc:
            # An out-of-range note/velocity byte is a malformed capture, mapped to
            # a crash error rather than an escaping pydantic ValidationError.
            raise self._malformed_event_error(raw, f"out-of-range field: {exc}") from exc

    # --- window + canonical order --------------------------------------------

    @staticmethod
    def _apply_window(
        events: list[MidiEvent], duration_samples: int
    ) -> list[MidiEvent]:
        """Keep only events in the half-open window ``[0, duration_samples)`` (by design).

        The capture window is strictly half-open: an event AT the window-end
        boundary (``t_samples == duration_samples`` — the trailing downbeat) is
        EXCLUDED, so a 1-beat capture is block-size-invariant and matches the
        golden. ``start_sample`` is 0 in the host's absolute frame.
        """
        return [event for event in events if event.t_samples < duration_samples]

    @staticmethod
    def _canonical_sort(events: list[MidiEvent]) -> list[MidiEvent]:
        """Thin static wrapper delegating to the single module-level
        ``_canonical_sort`` (the one logic source). A real static method
        (not a post-class rebind) so static type analysis resolves it; existing
        ``self._canonical_sort(...)`` / ``MidiCaptureBackend._canonical_sort(...)``
        call sites keep working."""
        return _canonical_sort(events)

    # --- meta ----------------------------------------------------------------

    def _build_meta(
        self,
        req: MidiCaptureRequest,
        host_meta: dict[str, Any],
        events: list[MidiEvent],
    ) -> MidiCaptureMeta:
        """Populate the capture provenance meta (by design)."""
        return MidiCaptureMeta(
            sample_rate=int(host_meta["sample_rate"]),
            block_size=int(host_meta["block_size"]),
            duration_samples=int(host_meta["duration_samples"]),
            tempo_bpm=req.tempo_bpm,
            start_position_beats=req.start_position_beats,
            duration_beats=req.duration_beats,
            tsig_num=req.tsig_num,
            tsig_den=req.tsig_den,
            plugin_id=_as_optional_str(host_meta.get("plugin_id")) or req.plugin_id,
            plugin_name=_as_optional_str(host_meta.get("plugin_name")),
            source="plugin",
            binary_sha256=_maybe_binary_sha256(req.plugin_path),
            events_sha256=_events_sha256(events),
            # block_size_invariant is B2's determinism concern; left None here.
        )

    # --- error mapping (mirrors subprocess_render.py's crash family) ----------

    @staticmethod
    def _host_error(error: Any) -> MidiCaptureError:
        """Map an ``outcome:"error"`` payload to a :class:`MidiCaptureError`."""
        if not isinstance(error, dict) or not error.get("code") or not error.get("message"):
            # A malformed error payload must still never be a silent success.
            return MidiCaptureError(
                MIDI_CAPTURE_HOST_ERROR,
                "clap_midi_host reported outcome=error with a malformed payload",
                detail={"raw": error},
                component=_COMPONENT,
            )
        return MidiCaptureError(
            MIDI_CAPTURE_HOST_ERROR,
            str(error["message"]),
            detail={"host_code": str(error["code"])},
            component=_COMPONENT,
        )

    def _crash_error(self, proc: subprocess.CompletedProcess[bytes]) -> MidiCaptureError:
        """Build the structured hard-crash error (crash-isolation, by design)."""
        exitcode = proc.returncode
        signalled = exitcode < 0
        signal_name: Optional[str] = None
        if signalled:
            try:
                signal_name = signal.Signals(-exitcode).name
            except ValueError:
                signal_name = f"SIG{-exitcode}"
            desc = f"exited with signal {signal_name}"
            reason = "signalled"
        elif exitcode != 0:
            desc = f"exited with code {exitcode}"
            reason = "nonzero_exit"
        else:
            # Exit 0 but no usable success object: missing/unparseable stdout or an
            # unknown outcome. A clean exit is NOT a licence for a silent empty.
            desc = "exited 0 without a success/error outcome"
            reason = "missing_or_unknown_outcome"

        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
        return MidiCaptureError(
            MIDI_CAPTURE_SUBPROCESS_CRASH,
            f"clap_midi_host {desc}.",
            detail={
                "exitcode": exitcode,
                "signal": signal_name,
                "reason": reason,
                "stderr": stderr_text[:_STDERR_DETAIL_CAP],
            },
            component=_COMPONENT,
        )

    @staticmethod
    def _malformed_success_error(why: str) -> MidiCaptureError:
        """Map a garbled ``outcome:"success"`` payload to a crash error."""
        return MidiCaptureError(
            MIDI_CAPTURE_SUBPROCESS_CRASH,
            "clap_midi_host reported outcome=success with a malformed payload",
            detail={"reason": "malformed_success", "why": why},
            component=_COMPONENT,
        )

    @staticmethod
    def _malformed_event_error(raw: Any, why: str) -> MidiCaptureError:
        """Map a malformed raw event to a crash error (never a silent drop)."""
        return MidiCaptureError(
            MIDI_CAPTURE_SUBPROCESS_CRASH,
            "clap_midi_host emitted a malformed MIDI event",
            detail={"reason": "malformed_event", "why": why, "raw": raw},
            component=_COMPONENT,
        )


# --- module-level helpers ----------------------------------------------------


def _canonical_sort(events: list[MidiEvent]) -> list[MidiEvent]:
    """Canonical order (by design): ascending ``t_samples``; at a coincident sample
    ``note_off`` sorts BEFORE ``note_on``. Stable/deterministic
    (Python's sort is stable, so equal keys keep capture order).

    This is the single shared definition; ``midi_input`` imports it and
    ``MidiCaptureBackend._canonical_sort`` is a thin static wrapper delegating here.
    """
    return sorted(
        events,
        key=lambda event: (event.t_samples, 0 if event.type == "note_off" else 1),
    )


def _ticks_at_960(t_samples: int, sample_rate: int, tempo_bpm: float) -> int:
    """``t_ticks`` @960 PPQ from ``t_samples`` (by design decode rule).

    ``t_ticks = round(t_samples / sample_rate * (tempo_bpm / 60) * 960)``.
    """
    return round(t_samples / sample_rate * (tempo_bpm / 60.0) * PPQ)


def _events_sha256(events: list[MidiEvent]) -> str:
    """Byte-identity hash over the canonical decoded list (by design).

    Hashes a compact, field-ordered JSON of the ordered events so an identical
    decoded list yields an identical digest (the B2 determinism key).
    """
    canonical = json.dumps(
        [
            [e.t_samples, e.t_ticks, e.type, e.channel, e.note, e.velocity]
            for e in events
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _maybe_binary_sha256(plugin_path: Path) -> Optional[str]:
    """Content hash of the ``.clap`` (tree hash for a bundle, else file hash).

    Reuses the ``pedalboard_vst3`` bundle/file tree-hash helper. Returns ``None``
    when the path is absent/unreadable (e.g. a FAKE-host unit test with no real
    plugin), so provenance hashing never turns a valid capture into an error.
    """
    path = Path(plugin_path)
    if not path.exists():
        return None
    try:
        return binary_sha256(path)
    except OSError:
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    """Coerce a host-meta string field to ``Optional[str]`` (None if absent)."""
    return value if isinstance(value, str) and value else None
