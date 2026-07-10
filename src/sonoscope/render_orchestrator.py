"""RenderOrchestrator — the ``render`` command's engine (Task E5, by design).

Wires the input-contract + render pipeline: it resolves the stimulus and the
param/MIDI spec, probes the backend (E3, cached), optionally re-injects a
hash-checked ``raw_state`` (E3), dispatches the crash-isolated subprocess render
(E4), and returns the rendered wav plus the populated ``report.render`` data.

Data-flow steps this implements (by design):

1. **Resolve stimulus + spec.** Verify the stimulus ``ref_sha256`` against the
   pinned corpus (E1) — a bad/missing/drifted ref is a hard INPUT error (exit 2),
   never a silent proceed to a default stimulus. Resolve the spec against
   the probed :class:`PluginInfo` (E2) → the resolved param vector + hashes.
2. **Render (subprocess).** ``2a`` probe the backend (cached by ``binary_sha256``
   inside the backend, E3); ``2b`` raw_state re-injection is validated in-backend
   (hash-stamp check, E3); ``2c`` ``backend.render`` runs in a fresh spawn
   child (E4) → wav + :class:`RenderMeta`.
3. **Return.** Assemble the :class:`RenderOutcome`: the wav + the orchestrator-
   owned ``report.render`` residuals (``backend`` ← ``backend.id``,
   ``backend_version`` ← ``backend.version``) + the resolved context. The
   ``render.determinism`` sub-block is DEFERRED to the F-layer (C3) and is
   left ``None`` here.

**Instrument-vs-effect routing (B1 C4, the orchestrator-side decision).** The
stimulus TYPE is selected from :attr:`PluginInfo.is_instrument`: an instrument
plugin gets a :class:`MidiStimulus` (from inline notes or a pinned corpus
``.mid``); an effect plugin gets an :class:`AudioStimulus` (from a pinned corpus
signal wav). The backend then dispatches its render overload on the concrete
stimulus type it receives (E3 ``_render_audio``).

**Seed handling (M2, carried from E3 review).** The C1 report schema exposes NO
dedicated seed slot — neither ``RenderBlock``/``RenderDeterminism`` nor
``InputBlock``/``ParamSetRef`` has a seed field, and adding one would break the
C1 contract tests. So E3's stopgap is RETAINED: the request seed is recorded by
the backend into ``RenderMeta.warnings`` under the ``seed-forwarded:`` prefix,
which flows verbatim into ``report.render.warnings`` here. The forwarded value is
also surfaced explicitly on :attr:`RenderOutcome.seed` for reproducibility.
F1's M7 warnings/errors aggregation should FILTER the ``seed-forwarded:``-prefixed
entry out of the user-facing ``report.render.warnings`` (it is a machine record,
not a coercion warning). This orchestrator does not filter it, so no information
is lost before F1 owns the aggregation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import mido
import numpy as np
import soundfile as sf

from sonoscope import corpus
from sonoscope.backends.base import (
    AudioStimulus,
    MidiStimulus,
    PluginInfo,
    RawState,
    RenderMeta,
    RenderRequest,
    RenderResult,
)
from sonoscope.backends.subprocess_render import render_in_subprocess
from sonoscope.corpus import CorpusItem
from sonoscope.errors import InputError
from sonoscope.resolver import ResolvedSpec, resolve
from sonoscope.spec import Note, Spec
from sonoscope.backends.base import RenderBackend

#: Stimulus/corpus INPUT failures are tagged ``component="corpus"`` (by design:
#: "missing/hash-mismatched corpus item" is the exit-2 INPUT class).
_COMPONENT = "corpus"

#: Spec stimulus refs are corpus-root-relative under this prefix (by design,
#: e.g. ``"corpus/midi/c3_sustain_2s.mid"``); corpus items key on the path with
#: the prefix stripped (``"midi/c3_sustain_2s.mid"``, see ``corpus.CorpusItem``).
_CORPUS_PREFIX = "corpus/"

#: Corpus item ``kind`` values (see ``scripts/generate_corpus.py`` catalog).
_KIND_MIDI = "midi"
_KIND_SIGNAL = "signal"

# Error codes (`error.code`; component == "corpus", exit 2).
STIMULUS_REF_UNKNOWN = "STIMULUS_REF_UNKNOWN"
STIMULUS_REF_MISSING = "STIMULUS_REF_MISSING"
STIMULUS_REF_HASH_MISMATCH = "STIMULUS_REF_HASH_MISMATCH"
STIMULUS_REF_ABSENT = "STIMULUS_REF_ABSENT"
STIMULUS_KIND_MISMATCH = "STIMULUS_KIND_MISMATCH"

_HASH_CHUNK = 1 << 20  # 1 MiB streaming read


@dataclass
class RenderOutcome:
    """E5 output: the rendered wav + the populated ``report.render`` data.

    Not a schema ``RenderBlock``: that model's ``determinism`` sub-block is a
    REQUIRED field the F-layer fills from the floors object (C3), so E5
    cannot construct a valid ``RenderBlock`` yet. This dataclass carries the
    backend-owned :class:`RenderMeta`, the orchestrator-owned ``report.render``
    residuals (``backend``/``backend_version``, C3), and the resolved context F1
    needs to assemble the full report — with ``determinism`` explicitly ``None``
    to mark it deferred.
    """

    #: The rendered wav on disk (E4 subprocess wrote it; parent reads it).
    wav_path: Path
    #: Backend-owned ``report.render.*`` fields (E3/C3 subset).
    render_meta: RenderMeta
    #: ``report.render.backend`` ← ``backend.id`` (C3, orchestrator-owned).
    backend: str
    #: ``report.render.backend_version`` ← ``backend.version`` (C3).
    backend_version: str
    #: The fully-resolved spec (resolved vector + ``resolved_sha256`` + derived
    #: ``expected_audio`` + ``patch_class`` + stimulus/render, E2).
    resolved: ResolvedSpec
    #: Verified pinned corpus ``ref_sha256`` (by design), or ``None`` for an
    #: inline-notes stimulus (no corpus item to pin).
    ref_sha256: Optional[str]
    #: The forwarded render seed (M8); recorded for reproducibility. No C1
    #: schema slot exists, so its user-facing record is the backend's
    #: ``seed-forwarded:`` warning (F1 filters it, see module docstring).
    seed: Optional[int]
    #: DEFERRED: the F-layer attaches ``render.determinism`` (C3).
    determinism: None = None


def _input_error(code: str, message: str, detail: dict) -> InputError:
    return InputError(code, message, detail=detail, component=_COMPONENT)


def _sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (mirrors ``corpus._sha256_file``)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_ref(
    ref: Optional[str], corpus_root: Path, manifest_path: Path
) -> tuple[Path, str, CorpusItem]:
    """Resolve + verify a stimulus ref against the pinned corpus (R4).

    Maps the corpus-relative ``ref`` to its :class:`CorpusItem`, confirms the file
    is present, and recomputes its sha256 against the pinned manifest value.
    Returns ``(file_path, ref_sha256, item)``. Any failure — an unknown ref, a
    missing file, or a hash drift — is a hard INPUT error (exit 2), never a silent
    proceed to a default/empty stimulus (pins-are-law).
    """
    if ref is None:
        raise _input_error(
            STIMULUS_REF_ABSENT,
            "stimulus requires a corpus ref for this plugin path, but none was "
            "provided (inline notes are only valid for an instrument plugin)",
            {},
        )
    rel = ref[len(_CORPUS_PREFIX):] if ref.startswith(_CORPUS_PREFIX) else ref
    items = {item.path: item for item in corpus.list_items(manifest_path)}
    item = items.get(rel)
    if item is None:
        raise _input_error(
            STIMULUS_REF_UNKNOWN,
            f"stimulus ref {ref!r} is not a pinned corpus item; refs are "
            "corpus-relative and must exist in corpus/manifest.toml",
            {"ref": ref, "resolved_path": rel},
        )
    file_path = Path(corpus_root) / item.path
    if not file_path.is_file():
        raise _input_error(
            STIMULUS_REF_MISSING,
            f"pinned corpus item {ref!r} is absent on disk at {file_path}",
            {"ref": ref, "path": str(file_path)},
        )
    actual = _sha256_file(file_path)
    if actual != item.sha256:
        raise _input_error(
            STIMULUS_REF_HASH_MISMATCH,
            f"corpus item {ref!r} bytes drifted from the pinned sha256 "
            "(pins are law; refusing to render an unverified stimulus)",
            {"ref": ref, "expected_sha256": item.sha256, "actual_sha256": actual},
        )
    return file_path, item.sha256, item


def _messages_from_notes(notes: Sequence[Note]) -> tuple[list[Any], float]:
    """Build absolute-time mido messages from inline spec notes (inline form).

    Each note yields a ``note_on`` at ``on`` and a ``note_off`` at ``off`` (both
    absolute seconds, as pedalboard's instrument overload expects, B1). The
    render duration is the latest note-off.
    """
    messages: list[Any] = []
    for note in notes:
        messages.append(
            mido.Message(
                "note_on", note=note.pitch, velocity=note.vel, time=float(note.on)
            )
        )
        messages.append(
            mido.Message(
                "note_off", note=note.pitch, velocity=0, time=float(note.off)
            )
        )
    messages.sort(key=lambda m: m.time)
    duration_s = max((float(note.off) for note in notes), default=0.0)
    return messages, duration_s


def _messages_from_midi_file(path: Path) -> tuple[list[Any], float]:
    """Load a pinned corpus ``.mid`` into absolute-time mido messages.

    ``mido.MidiFile`` iteration yields tempo-adjusted per-message DELTA seconds;
    accumulate to absolute time (what the instrument overload consumes, B1) and
    drop meta messages. Duration is the file's total playback length.
    """
    midi = mido.MidiFile(str(path))
    messages: list[Any] = []
    abs_time = 0.0
    for message in midi:
        abs_time += message.time
        if message.is_meta:
            continue
        messages.append(message.copy(time=abs_time))
    return messages, float(midi.length)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a pinned corpus signal wav as channel-major float32 (B1).

    soundfile reads frame-major ``(frames, channels)``; transpose to the
    channel-major ``(channels, frames)`` layout the effect overload expects.
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data.T, dtype=np.float32)
    return audio, int(sample_rate)


def build_stimulus(
    spec: Spec,
    plugin_info: PluginInfo,
    corpus_root: Path = corpus.DEFAULT_CORPUS_ROOT,
    manifest_path: Path = corpus.DEFAULT_MANIFEST,
) -> tuple[Any, Optional[str]]:
    """Build the runtime stimulus + return its verified ``ref_sha256``.

    Routes on :attr:`PluginInfo.is_instrument` (B1 C4, the orchestrator-side
    decision): an instrument plugin builds a :class:`MidiStimulus` (inline notes,
    or a pinned corpus ``.mid``); an effect plugin builds an
    :class:`AudioStimulus` from a pinned corpus signal wav. A stimulus whose
    corpus ``kind`` does not match the plugin nature (a signal fed to an
    instrument, or MIDI fed to an effect) is a hard INPUT error, never a silent
    coercion. ``ref_sha256`` is ``None`` for the inline-notes case (no pin).
    """
    stim = spec.stimulus
    if plugin_info.is_instrument:
        # Instrument path -> MidiStimulus. Inline notes take precedence over a
        # ref (the spec validator guarantees exactly one is set).
        if stim.notes is not None:
            messages, duration_s = _messages_from_notes(stim.notes)
            return MidiStimulus(messages=messages, duration_s=duration_s), None
        file_path, ref_sha256, item = _verify_ref(
            stim.ref, corpus_root, manifest_path
        )
        if item.kind != _KIND_MIDI:
            raise _input_error(
                STIMULUS_KIND_MISMATCH,
                f"instrument plugin {plugin_info.name!r} requires a MIDI stimulus, "
                f"but ref {stim.ref!r} is a {item.kind!r} corpus item",
                {"ref": stim.ref, "item_kind": item.kind, "is_instrument": True},
            )
        messages, duration_s = _messages_from_midi_file(file_path)
        return MidiStimulus(messages=messages, duration_s=duration_s), ref_sha256

    # Effect path -> AudioStimulus from a pinned corpus signal wav.
    if stim.notes is not None:
        raise _input_error(
            STIMULUS_KIND_MISMATCH,
            f"effect plugin {plugin_info.name!r} requires an audio stimulus, but "
            "the spec supplies inline MIDI notes",
            {"item_kind": _KIND_MIDI, "is_instrument": False},
        )
    file_path, ref_sha256, item = _verify_ref(stim.ref, corpus_root, manifest_path)
    if item.kind != _KIND_SIGNAL:
        raise _input_error(
            STIMULUS_KIND_MISMATCH,
            f"effect plugin {plugin_info.name!r} requires an audio stimulus, but "
            f"ref {stim.ref!r} is a {item.kind!r} corpus item",
            {"ref": stim.ref, "item_kind": item.kind, "is_instrument": False},
        )
    audio, sample_rate = _load_audio(file_path)
    return AudioStimulus(audio=audio, sample_rate_hz=sample_rate), ref_sha256


def render(
    spec: Spec,
    plugin_path: Path,
    backend: RenderBackend,
    *,
    plugin_format: Literal["vst3", "au"] = "vst3",
    raw_state: Optional[RawState] = None,
    corpus_root: Path = corpus.DEFAULT_CORPUS_ROOT,
    manifest_path: Path = corpus.DEFAULT_MANIFEST,
    sample_rate_hz: Optional[int] = None,
    block_size: Optional[int] = None,
    channels: Optional[int] = None,
    seed: Optional[int] = None,
) -> RenderOutcome:
    """Resolve → probe → subprocess-render → assemble ``report.render``.

    Probes the backend (cached by ``binary_sha256`` in the backend, E3), resolves
    the spec against the probe (E2), verifies + builds the stimulus (E1/B1 C4),
    then dispatches ``backend.render`` in a crash-isolated spawn child (E4). Any
    stimulus-ref failure is a hard INPUT error (exit 2) raised BEFORE dispatch, so
    a bad input never reaches a render. ``raw_state`` re-injection + hash-stamp
    validation happens inside the backend (E3). Returns the wav + the populated
    ``report.render`` data with ``determinism`` deferred to the F-layer.

    Render-param overrides (CLI ``--sample-rate``/``--block-size``/``--channels``/
    ``--seed``): a non-``None`` ``sample_rate_hz`` / ``block_size`` / ``channels`` /
    ``seed`` argument takes precedence over the spec's ``render.<field>`` when the
    :class:`RenderRequest` is built; ``None`` (the default) falls back to the spec
    value (unchanged behavior). This is the SINGLE place the override precedence is
    applied — the CLI render-family handlers forward their flags here. Rendering
    stays deterministic: identical inputs (including identical overrides) yield an
    identical request and therefore an identical render.
    """
    plugin_path = Path(plugin_path)

    # Probe (cached inside the backend by binary_sha256, E3).
    plugin_info = backend.probe(plugin_path)

    # Resolve the spec against the probe (E2). Validates every param
    # reference + the [0,1] domain; a failure is a hard INPUT error (exit 2).
    resolved = resolve(spec, plugin_info)

    # Verify the stimulus ref (ref_sha256) + build the routed stimulus
    # (MidiStimulus vs AudioStimulus on is_instrument, B1 C4). Fails BEFORE any
    # render dispatch on a bad/missing/drifted ref (no silent proceed).
    stimulus, ref_sha256 = build_stimulus(
        spec, plugin_info, corpus_root, manifest_path
    )

    # Override precedence (SINGLE application point): a non-None CLI override wins
    # over the spec's render.<field>; None falls back to the spec value.
    resolved_sample_rate_hz = (
        sample_rate_hz if sample_rate_hz is not None else spec.render.sample_rate_hz
    )
    resolved_block_size = (
        block_size if block_size is not None else spec.render.block_size
    )
    resolved_channels = channels if channels is not None else spec.render.channels
    resolved_seed = seed if seed is not None else spec.render.seed

    req = RenderRequest(
        plugin_path=plugin_path,
        plugin_format=plugin_format,
        stimulus=stimulus,
        param_set=resolved,
        sample_rate_hz=resolved_sample_rate_hz,
        block_size=resolved_block_size,
        channels=resolved_channels,
        raw_state=raw_state,
        # Seed forwarding: forwarded to the backend, recorded in RenderMeta
        # (no C1 schema slot exists; see module docstring seed handling).
        seed=resolved_seed,
    )

    # Crash-isolated subprocess render (E4). raw_state
    # hash-stamp validation happens inside backend.render (E3). The parent
    # reads the wav + RenderMeta; the child's exit frees the plugin/JUCE.
    result: RenderResult = render_in_subprocess(backend, req)

    # Assemble the populated report.render data; determinism deferred.
    return RenderOutcome(
        wav_path=result.wav_path,
        render_meta=result.render_meta,
        backend=backend.id,
        backend_version=backend.version,
        resolved=resolved,
        ref_sha256=ref_sha256,
        # Reflect the effective (override-or-spec) seed that was forwarded, so the
        # reproducibility record matches the request actually rendered.
        seed=resolved_seed,
        determinism=None,
    )
