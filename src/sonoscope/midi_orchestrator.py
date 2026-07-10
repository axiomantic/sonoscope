"""C1 ``analyze_midi`` — the ``analyze-midi`` command's engine.

Task C1 (by design, Track C). The MIDI analogue of
``analysis_orchestrator.py`` (F1): compose the already-tested MIDI pieces into a
single ``analyze_midi(...) -> MidiAnalysisReport``. It ties together:

- **B1** :class:`~sonoscope.backends.midi_capture.MidiCaptureBackend` — the
  plugin-source capture (spawn the C CLAP host, decode, canonical order).
- **E2** :func:`~sonoscope.midi_input.load_midi_file` /
  :func:`~sonoscope.midi_input.load_expected_events` /
  :func:`~sonoscope.midi_input.apply_slice` — the file source, the expected
  golden, and the analysis-window slice.
- **E1** :func:`~sonoscope.features.midi_tripwires.evaluate_midi` — the pure
  comparator producing integrity + expected-vs-actual + verdict + reasons.
- **S1** the :class:`~sonoscope.schema.MidiAnalysisReport` model and its refs.

Two input SOURCES (by design):

- **plugin source** — a :class:`~sonoscope.backends.midi_capture.MidiCaptureRequest`
  (plugin_path + transport + render). Captured via an injectable backend (tests
  point it at a FAKE) into events + a plugin :class:`MidiCaptureMeta`
  (``source="plugin"``, ``binary_sha256`` of the ``.clap``).
- **file source** — a :class:`MidiFileSource` (a ``.mid`` path + sample_rate +
  optional tempo). Decoded via :func:`load_midi_file` into events + a synthesized
  ``source="file"`` :class:`MidiCaptureMeta`; the file content hash lives on the
  input block's :class:`MidiFileRef` (``file_sha256``).

Per-source ``offvel0`` default (open question, by design): a ``note_on`` with
velocity 0 is a note-off written as ``0x90 vel0``. The Reference Sequencer contract forbids it
(explicit ``0x80`` only), so the **plugin** default is ``"red"``; a generic
``.mid`` legitimately uses the convention, so the **file** default is
``"normalize"``. The caller may override either.

Deterministic-first, like F1: capture/load and ``evaluate_midi`` run BEFORE the
report is assembled, so a capture/load/spec failure surfaces as its already-typed
error (:class:`~sonoscope.errors.MidiCaptureError` / :class:`InputError`) and is
NEVER swallowed into a fabricated report.

CLI-wiring boundary: C1 builds only the engine; wiring the ``analyze-midi``
subcommand to it is owned by C2 (``cli.py`` is untouched here).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from sonoscope import __version__
from sonoscope.backends.midi_capture import (
    MidiCaptureBackend,
    MidiCaptureRequest,
    MidiCaptureResult,
)
from sonoscope.errors import InputError
from sonoscope.features.midi_tripwires import Offvel0Policy, evaluate_midi
from sonoscope.descriptors.midi_deriver import derive_midi_descriptors
from sonoscope.midi_input import (
    SliceUnit,
    apply_slice,
    load_expected_events,
    load_midi_file,
    resolve_window_samples,
)
from sonoscope.schema import (
    MidiAnalysisReport,
    MidiBlock,
    MidiCaptureMeta,
    MidiEvent,
    MidiExpectedRef,
    MidiFileRef,
    MidiInputBlock,
    MidiPluginRef,
    MidiTransportRef,
)

# --- per-source defaults (by design) ------------------------------------------

#: The plugin (Reference Sequencer) offvel0 default: a ``0x90 vel0`` note-off violates the
#: explicit-``0x80`` contract, so it is RED (open question, by design).
PLUGIN_OFFVEL0_POLICY: Offvel0Policy = "red"
#: The file (generic ``.mid``) offvel0 default: a ``0x90 vel0`` is the standard
#: MIDI note-off convention, so it is folded to ``note_off`` (informational note).
FILE_OFFVEL0_POLICY: Offvel0Policy = "normalize"

#: A ``.mid`` carries no ``set_tempo`` OR the caller gives no override -> the MIDI
#: standard default (120 BPM), used for the synthesized file meta/transport.
DEFAULT_FILE_TEMPO_BPM: float = 120.0
#: A file source has no block-based rendering; the required ``block_size`` field is
#: declared not-applicable with 0 (documented, never a fabricated block figure).
FILE_BLOCK_SIZE: int = 0
#: A bare ``.mid`` carries no authoritative time signature at load; the synthesized
#: transport uses the 4/4 standard default.
DEFAULT_TSIG_NUM: int = 4
DEFAULT_TSIG_DEN: int = 4

#: The inline (non-path) expected-source ref sentinel on :class:`MidiExpectedRef`.
INLINE_EXPECTED_REF: str = "inline"

#: A ``MidiFileSource.sample_rate`` <= 0 is a nonsensical timing axis: it silently
#: collapses every ``t_samples`` and ``duration_beats`` to 0 instead of failing. It
#: is a typed input error (component ``"midi"``, exit 2), never a silent zero span.
MIDI_SAMPLE_RATE_INVALID: str = "MIDI_SAMPLE_RATE_INVALID"


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` stamp (``generated_at``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _CaptureBackend(Protocol):
    """The injectable plugin-capture interface (real backend or a test FAKE)."""

    def capture(self, req: MidiCaptureRequest) -> MidiCaptureResult: ...


@dataclass(frozen=True)
class MidiFileSource:
    """A standalone ``.mid`` file source for :func:`analyze_midi` (by design).

    ``tempo_bpm`` (when given) overrides the file's own ``set_tempo`` metas for the
    sample-axis conversion and the synthesized meta/transport; when ``None`` the
    file tempo drives decode and the meta reports the 120 BPM standard default.
    """

    path: Union[str, Path]
    sample_rate: int
    tempo_bpm: Optional[float] = None


@dataclass(frozen=True)
class MidiSlice:
    """An optional analysis-window slice applied to the events before evaluation.

    A thin carrier for the E2 :func:`apply_slice` bound: the half-open
    ``[offset, offset+length)`` window in ``unit`` (``length is None`` -> to the
    end), with ``rebase`` (default ``True``) shifting the kept slice to a 0-origin.
    """

    offset: float
    length: Optional[float]
    unit: SliceUnit
    rebase: bool = True


@dataclass(frozen=True)
class _SourcePrep:
    """Internal: the per-source products the common tail folds into the report."""

    full_events: list[MidiEvent]
    capture_meta: MidiCaptureMeta
    plugin_ref: Optional[MidiPluginRef]
    file_ref: Optional[MidiFileRef]
    transport: MidiTransportRef
    sample_rate: int
    tempo_bpm: float
    default_policy: Offvel0Policy


def analyze_midi(
    source: Union[MidiCaptureRequest, MidiFileSource],
    *,
    backend: Optional[_CaptureBackend] = None,
    expected: Optional[Union[str, Path, list[Any]]] = None,
    slice_spec: Optional[MidiSlice] = None,
    timing_tolerance_samples: int = 1,
    offvel0_policy: Optional[Offvel0Policy] = None,
    generated_at: Optional[str] = None,
    sonoscope_version: str = __version__,
) -> MidiAnalysisReport:
    """Compose the full MIDI pipeline into a :class:`MidiAnalysisReport` (C1).

    Parameters
    ----------
    source:
        EITHER a :class:`~sonoscope.backends.midi_capture.MidiCaptureRequest`
        (plugin source: captured via ``backend``) OR a :class:`MidiFileSource` (a
        standalone ``.mid``, decoded via E2). The type discriminates the path.
    backend:
        The injectable plugin-capture backend (tests pass a FAKE). Only used for a
        plugin source; defaults to a real
        :class:`~sonoscope.backends.midi_capture.MidiCaptureBackend`.
    expected:
        Optional expected-event golden — a path/JSON-list (E2
        :func:`load_expected_events`). When given, the expected-vs-actual diff and
        tripwires 2-4 run. The golden is authored in FULL-capture coordinates: when
        ``slice_spec`` is also given, the SAME window+rebase is applied to the
        expected events before evaluation, so captured and expected are always
        compared in the SAME coordinate frame (see ``slice_spec``).
    slice_spec:
        Optional analysis-window slice (E2 :func:`apply_slice`) applied to the
        events BEFORE evaluation, so only in-window events are analyzed. When an
        ``expected`` golden is also supplied it is windowed+rebased with the
        IDENTICAL slice (both sides share one frame — composable: author one
        full-capture spec, slice-analyze any sub-window). This diverges provenance
        from the analyzed window: ``midi.capture_meta`` and ``input.transport``
        (and ``input.expected.event_count``) describe the FULL capture / FULL
        authored spec, while ``midi.events`` are the analyzed (possibly rebased)
        WINDOW.
    timing_tolerance_samples:
        The E1 timing gate in SAMPLES (by design; also recorded on the meta).
    offvel0_policy:
        Override the per-source ``0x90 vel0`` policy. Default: ``"red"`` for a
        plugin source (the Reference Sequencer explicit-``0x80`` contract), ``"normalize"`` for
        a file source (the standard note-off convention).

    A capture/load/spec failure surfaces as the already-typed error
    (:class:`~sonoscope.errors.MidiCaptureError` / :class:`InputError`) — this
    engine never swallows it into a fabricated report.
    """
    generated_at = generated_at or _now_iso()

    # --- resolve the source (GROUND TRUTH capture/load; never swallowed) ------
    if isinstance(source, MidiCaptureRequest):
        prep = _prepare_plugin_source(source, backend, timing_tolerance_samples)
        report_source = "plugin"
    elif isinstance(source, MidiFileSource):
        prep = _prepare_file_source(source, timing_tolerance_samples)
        report_source = "file"
    else:  # a caller passing neither source type is a programming error.
        raise TypeError(
            "analyze_midi source must be a MidiCaptureRequest (plugin) or "
            f"MidiFileSource (file), got {type(source).__name__}"
        )

    resolved_policy: Offvel0Policy = offvel0_policy or prep.default_policy

    # --- optional slice: window the events BEFORE evaluation (by design) ------
    if slice_spec is not None:
        analyzed = _apply_slice_spec(prep.full_events, slice_spec, prep)
    else:
        analyzed = prep.full_events

    # --- optional expected golden -> the expected list (by design) ------------
    # C1 fix: the golden is authored in FULL-capture coordinates. When a slice is
    # applied, the SAME window+rebase MUST be applied to the expected events so
    # ``evaluate_midi`` diffs captured-vs-expected in ONE frame (an unsliced
    # expected would be diffed against the rebased capture -> a silently wrong
    # verdict). ``expected_ref.event_count`` is built from the FULL loaded spec
    # (provenance describes the authored golden, not the analyzed window).
    expected_events: Optional[list[MidiEvent]] = None
    expected_ref: Optional[MidiExpectedRef] = None
    if expected is not None:
        full_expected = load_expected_events(
            expected, sample_rate=prep.sample_rate, tempo_bpm=prep.tempo_bpm
        )
        expected_ref = _expected_ref(expected, full_expected)
        if slice_spec is not None:
            expected_events = _apply_slice_spec(full_expected, slice_spec, prep)
        else:
            expected_events = full_expected

    # --- evaluate (pure E1 comparator) ----------------------------------------
    evaluation = evaluate_midi(
        analyzed,
        expected_events,
        timing_tolerance_samples=timing_tolerance_samples,
        offvel0_as_0x90_policy=resolved_policy,
    )

    # --- assemble the full versioned report (by design) -----------------------
    input_block = MidiInputBlock(
        source=report_source,
        plugin=prep.plugin_ref,
        file=prep.file_ref,
        transport=prep.transport,
        expected=expected_ref,
    )
    midi_block = MidiBlock(
        capture_meta=prep.capture_meta,
        events=analyzed,
        expected_vs_actual=evaluation.expected_vs_actual,
        integrity=evaluation.integrity,
        verdict=evaluation.verdict,
        reasons=evaluation.reasons,
    )

    # --- C2 descriptors: resolve the analysis window, then derive the block ----
    # The density axis (by design) uses the SAME unit->samples conversion as the slice
    # axis. Full-capture (no slice) resolves to the capture's duration_samples;
    # a slice unpacks to primitives so midi_input never back-imports MidiSlice.
    if slice_spec is None:
        window_samples = resolve_window_samples(
            None, None, None,
            sample_rate=prep.sample_rate,
            tempo_bpm=prep.tempo_bpm,
            full_duration_samples=prep.capture_meta.duration_samples,
        )
    else:
        window_samples = resolve_window_samples(
            slice_spec.offset, slice_spec.length, slice_spec.unit,
            sample_rate=prep.sample_rate,
            tempo_bpm=prep.tempo_bpm,
            full_duration_samples=prep.capture_meta.duration_samples,
        )
    descriptors_block = derive_midi_descriptors(
        analyzed, prep.capture_meta, window_samples
    )

    return MidiAnalysisReport(
        generated_at=generated_at,
        sonoscope_version=sonoscope_version,
        input=input_block,
        midi=midi_block,
        descriptors=descriptors_block,   # NEW (C2)
        errors=[],
    )


# --- slice application -------------------------------------------------------


def _apply_slice_spec(
    events: list[MidiEvent], slice_spec: MidiSlice, prep: _SourcePrep
) -> list[MidiEvent]:
    """Apply a :class:`MidiSlice` to an event list in the source's timing frame.

    The single window+rebase entry point (E2 :func:`apply_slice`), used for BOTH
    the captured events and — when an expected golden is supplied — the expected
    events, guaranteeing they are windowed with IDENTICAL bounds and end up in the
    SAME coordinate frame before :func:`evaluate_midi` (the C1 same-frame fix).
    """
    return apply_slice(
        events,
        offset=slice_spec.offset,
        length=slice_spec.length,
        unit=slice_spec.unit,
        sample_rate=prep.sample_rate,
        tempo_bpm=prep.tempo_bpm,
        rebase=slice_spec.rebase,
    )


# --- source preparation ------------------------------------------------------


def _prepare_plugin_source(
    req: MidiCaptureRequest,
    backend: Optional[_CaptureBackend],
    timing_tolerance_samples: int,
) -> _SourcePrep:
    """Capture the plugin source (B1) and build its refs/meta (never swallowed).

    ``backend.capture`` raises a typed :class:`MidiCaptureError` on any host
    failure; that error propagates unmodified. The backend's returned meta is the
    capture provenance (``source="plugin"``, ``binary_sha256``, ``events_sha256``);
    only the timing tolerance actually used is stamped onto it here.
    """
    capture = (backend or MidiCaptureBackend()).capture(req)
    capture_meta = capture.meta.model_copy(
        update={"timing_tolerance_samples": timing_tolerance_samples}
    )
    plugin_ref = MidiPluginRef(
        path=str(req.plugin_path),
        binary_sha256=capture_meta.binary_sha256,
        plugin_id=capture_meta.plugin_id,
        plugin_name=capture_meta.plugin_name,
    )
    transport = MidiTransportRef(
        sample_rate=req.sample_rate,
        block_size=req.block_size,
        tempo_bpm=req.tempo_bpm,
        start_position_beats=req.start_position_beats,
        duration_beats=req.duration_beats,
        tsig_num=req.tsig_num,
        tsig_den=req.tsig_den,
        playing=req.playing,
    )
    return _SourcePrep(
        full_events=capture.events,
        capture_meta=capture_meta,
        plugin_ref=plugin_ref,
        file_ref=None,
        transport=transport,
        sample_rate=req.sample_rate,
        tempo_bpm=req.tempo_bpm,
        default_policy=PLUGIN_OFFVEL0_POLICY,
    )


def _prepare_file_source(
    src: MidiFileSource, timing_tolerance_samples: int
) -> _SourcePrep:
    """Decode the ``.mid`` source (E2) and synthesize its refs/meta.

    :func:`load_midi_file` raises a typed :class:`InputError` on an
    unreadable/corrupt/non-PPQ file; that error propagates unmodified. The
    ``source="file"`` meta is synthesized: ``duration_samples`` is the span of the
    decoded events, the block size is not-applicable (0), and the tempo/tsig are
    the resolved override-or-standard defaults. Capture-determinism fields
    (``events_sha256``/``block_size_invariant``) stay ``None`` — a static file has
    no capture repeats. The file content hash lives on the :class:`MidiFileRef`.

    ``sample_rate`` is the canonical timing axis; a value <= 0 is rejected up front
    as a typed :class:`InputError` (never a silently-collapsed 0-sample duration).
    """
    if src.sample_rate <= 0:
        raise InputError(
            MIDI_SAMPLE_RATE_INVALID,
            f"MidiFileSource.sample_rate must be > 0, got {src.sample_rate}",
            detail={
                "reason": "non_positive_sample_rate",
                "sample_rate": src.sample_rate,
            },
            component="midi",
        )
    events = load_midi_file(
        src.path, sample_rate=src.sample_rate, tempo_bpm=src.tempo_bpm
    )
    tempo_bpm = (
        src.tempo_bpm if src.tempo_bpm is not None else DEFAULT_FILE_TEMPO_BPM
    )
    duration_samples = max((e.t_samples for e in events), default=0)
    duration_beats = (
        duration_samples / src.sample_rate * (tempo_bpm / 60.0)
        if src.sample_rate
        else 0.0
    )
    capture_meta = MidiCaptureMeta(
        sample_rate=src.sample_rate,
        block_size=FILE_BLOCK_SIZE,
        duration_samples=duration_samples,
        tempo_bpm=tempo_bpm,
        start_position_beats=0.0,
        duration_beats=duration_beats,
        tsig_num=DEFAULT_TSIG_NUM,
        tsig_den=DEFAULT_TSIG_DEN,
        source="file",
        timing_tolerance_samples=timing_tolerance_samples,
    )
    file_ref = MidiFileRef(
        path=str(src.path), file_sha256=_file_sha256(src.path)
    )
    transport = MidiTransportRef(
        sample_rate=src.sample_rate,
        block_size=FILE_BLOCK_SIZE,
        tempo_bpm=tempo_bpm,
        start_position_beats=0.0,
        duration_beats=duration_beats,
        tsig_num=DEFAULT_TSIG_NUM,
        tsig_den=DEFAULT_TSIG_DEN,
        playing=True,
    )
    return _SourcePrep(
        full_events=events,
        capture_meta=capture_meta,
        plugin_ref=None,
        file_ref=file_ref,
        transport=transport,
        sample_rate=src.sample_rate,
        tempo_bpm=tempo_bpm,
        default_policy=FILE_OFFVEL0_POLICY,
    )


# --- expected ref + hashing --------------------------------------------------


def _expected_ref(
    source: Union[str, Path, list[Any]], events: list[MidiEvent]
) -> MidiExpectedRef:
    """Build the expected-list provenance ref (by design).

    A path source records its path + file content hash; an inline list records the
    ``"inline"`` sentinel + no hash. ``event_count`` is the loaded length.
    """
    if isinstance(source, (str, Path)):
        return MidiExpectedRef(
            ref=str(source),
            spec_sha256=_file_sha256(source),
            event_count=len(events),
        )
    return MidiExpectedRef(
        ref=INLINE_EXPECTED_REF, spec_sha256=None, event_count=len(events)
    )


def _file_sha256(path: Union[str, Path]) -> Optional[str]:
    """SHA-256 of a file's bytes; ``None`` if unreadable.

    Provenance hashing is best-effort (mirrors B1's ``_maybe_binary_sha256``): a
    file that just loaded successfully hashes cleanly, but a hashing failure never
    turns a valid analysis into an error.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
