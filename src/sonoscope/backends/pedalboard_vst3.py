"""PedalboardVST3Backend — the v1 render backend (Task E3, by design).

Implements the C3 :class:`~sonoscope.backends.base.RenderBackend` protocol against
pedalboard 0.9.23's **B1-confirmed** Surge XT API surface
(confirmed via prior spike investigation). Responsibilities:

- ``probe(plugin_path)`` -> :class:`PluginInfo` built from ``plugin.parameters``
  (named params, snake_case ``python_name`` keys), **cached by** ``binary_sha256``.
- Instrument path (``MidiStimulus``): ``plugin(midi_messages, duration,
  sample_rate, num_channels, buffer_size, reset)`` (B1).
- Effect path (``AudioStimulus``): ``plugin(audio, sample_rate, buffer_size,
  reset)`` (B1).
- Applies the E2 resolved param vector by ``python_name`` / index via
  ``param.raw_value`` (normalized 0..1) — never hardcodes names.
- ``raw_state`` **re-injection + hash-stamp validation only**: a state stamped to
  a different ``binary_sha256`` than the loaded binary is a hard error, never a
  silent proceed (by design). Interactive **capture** tooling is DEFERRED in v1 (B1's R3
  finding: Surge renders non-silent headless); see AGENTS.md.
- Seed forwarding (M8): the request seed is forwarded where honored (no
  plugin-honored path exists in v1) and RECORDED into ``RenderMeta.warnings`` for
  reproducibility (the schema ``RenderBlock`` exposes no dedicated seed field, and
  the C3 ``RenderMeta`` field set is a frozen subset that must not be extended).
- Writes a ``PCM_F32`` wav and returns :class:`RenderMeta` incl. ``wav_sha256``.

CAVEAT sources (B1): C1 forced-stereo, C2 python_name keys, C3 no ``default_value``
(derived from load-time ``raw_value``), C4 route via ``is_instrument``, C5
classify kind via ``type``/``valid_values``/``is_boolean``/``range`` not
``num_steps``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sonoscope.backends.base import (
    AudioStimulus,
    MidiStimulus,
    ParamInfo,
    ParamKind,
    PluginInfo,
    RawState,
    RenderMeta,
    RenderRequest,
    RenderResult,
)
from sonoscope.errors import RenderError, SonoscopeError
from sonoscope.wav_io import canonical_float_wav_bytes

_COMPONENT = "render"

#: Stable backend id -> ``report.render.backend`` (filled by the F-layer, C3).
BACKEND_ID = "pedalboard-vst3"

#: Schema ``RenderBlock.wav_subtype`` literal. The render WAV is written by the
#: canonical IEEE-float encoder (32-bit float LE, no PEAK chunk) — see
#: ``sonoscope.wav_io.canonical_float_wav_bytes``.
WAV_SUBTYPE = "PCM_F32"

# Error codes (`error.code`; component == "render").
RAW_STATE_STALE = "RAW_STATE_STALE"
RAW_STATE_UNSTAMPED = "RAW_STATE_UNSTAMPED"
RAW_STATE_BLOB_MISSING = "RAW_STATE_BLOB_MISSING"
PLUGIN_LOAD_FAILED = "PLUGIN_LOAD_FAILED"
PLUGIN_PROBE_FAILED = "PLUGIN_PROBE_FAILED"
STIMULUS_TYPE_UNKNOWN = "STIMULUS_TYPE_UNKNOWN"
PARAM_NOT_ON_PLUGIN = "PARAM_NOT_ON_PLUGIN"

# Minimal probe-render constants for channel-layout detection (no hardcoded bus).
_DETECT_SR = 48000.0
_DETECT_DURATION_S = 0.02
_DETECT_FRAMES = 64
_HASH_CHUNK = 1 << 20  # 1 MiB streaming read


# --- Stimulus contract the backend consumes ---------------------------------
# MidiStimulus / AudioStimulus are backend-agnostic runtime stimulus types and
# live in the contract layer (sonoscope.backends.base), imported above. E5
# (RenderOrchestrator) selects which stimulus to build from PluginInfo
# .is_instrument (by design / B1 C4: route on is_instrument, never on
# is_effect==False). The backend then dispatches the pedalboard overload by the
# concrete stimulus TYPE it receives.


def _render_error(code: str, message: str, detail: dict) -> RenderError:
    return RenderError(code, message, detail=detail, component=_COMPONENT)


# --- binary_sha256 (bundle tree hash) ---------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Order-stable bundle hash, mirroring ``scripts/verify_surge_xt.sh``.

    Algorithm (identical shape to A4's ``tree_sha256``): for every regular file
    under ``root``, emit ``"<file_sha256>  ./<relpath>\\n"``; sort those lines by
    the LC_ALL=C (bytewise) order of their paths; hash the concatenation. This
    makes ``binary_sha256`` deterministic across runs and independent of
    filesystem enumeration order.

    Traversal matches A4's ``find . -type f`` (no ``-L``): it does NOT descend
    into directory symlinks (``os.walk(..., followlinks=False)``) and counts only
    regular files (``is_file() and not is_symlink()``, so a symlink — type ``l``
    under ``find`` — is excluded). This keeps the hash byte-identical to A4 for
    bundles that contain directory symlinks (e.g. macOS ``Versions/Current`` or
    the clap-wrapper VST3), where ``rglob`` would have recursed THROUGH the link.
    """
    root = Path(root)
    rel_paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(dirpath) / filename
            # find -type f: regular files only (a symlink is type l, excluded).
            if path.is_file() and not path.is_symlink():
                rel_paths.append("./" + path.relative_to(root).as_posix())
    # LC_ALL=C sort == bytewise ordering of the path strings.
    rel_paths.sort(key=lambda rel: rel.encode("utf-8"))
    lines = "".join(
        f"{_file_sha256(root / rel)}  {rel}\n" for rel in rel_paths
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def binary_sha256(plugin_path: Path) -> str:
    """Content hash of the plugin binary: tree hash for a bundle, file hash else.

    A ``.vst3`` on macOS is a bundle directory; a single-file plugin hashes its
    bytes. This is the key both the probe cache and the ``raw_state`` stamp
    validation compare against (by design).
    """
    path = Path(plugin_path)
    if path.is_dir():
        return _tree_sha256(path)
    return _file_sha256(path)


# --- raw_state hash-stamp validation (M5) -----------------------------------


def validate_raw_state(
    raw_state: Optional[RawState], binary_hash: str
) -> bool:
    """Validate a captured ``raw_state`` against the loaded binary hash (by design).

    Returns ``True`` when a captured, correctly-stamped state must be re-injected;
    ``False`` when there is nothing to inject (absent or ``captured is False``).
    A captured state whose stamped ``plugin_binary_sha256`` does not match the
    loaded binary is STALE -> hard :class:`RenderError`, never a silent proceed to
    default state. A captured-but-unstamped state cannot be validated and is
    likewise a hard error.
    """
    if raw_state is None or not raw_state.captured:
        return False
    stamped = raw_state.plugin_binary_sha256
    if stamped is None:
        raise _render_error(
            RAW_STATE_UNSTAMPED,
            "raw_state.captured is True but plugin_binary_sha256 is unset; "
            "cannot validate against the loaded binary (refusing to proceed)",
            {"binary_sha256": binary_hash},
        )
    if stamped != binary_hash:
        raise _render_error(
            RAW_STATE_STALE,
            f"raw_state stamped to binary {stamped} but the loaded binary is "
            f"{binary_hash}; refusing to inject stale state (by design)",
            {"stamped_sha256": stamped, "loaded_sha256": binary_hash},
        )
    return True


# --- seed forwarding + recording (M8) ---------------------------------------


def seed_record_warnings(seed: Optional[int]) -> list[str]:
    """Record the forwarded render seed for reproducibility (by design).

    The seed is forwarded to the plugin where honored; v1's pedalboard path
    exposes no plugin-honored seed hook, so the value is only RECORDED. The schema
    ``RenderBlock`` has no dedicated seed field and the C3 ``RenderMeta`` field set
    is frozen, so the record lands in ``RenderMeta.warnings`` under a distinct
    ``seed-forwarded:`` prefix (not a coercion warning). Returns ``[]`` for no seed.
    """
    if seed is None:
        return []
    return [
        f"seed-forwarded: {seed} (backend has no plugin-honored seed path; "
        "recorded for reproducibility, measured floor is source of truth)"
    ]


# --- param-kind classification (B1 C5) --------------------------------------


def _classify(param: Any) -> tuple[ParamKind, Optional[int]]:
    """Classify a param's kind + num_steps via type/valid_values/is_boolean/range.

    Never trusts ``num_steps``/``is_discrete`` (B1 C5: INT_MAX / always-True on
    0.9.23). ``float`` = continuous; ``bool`` = boolean toggle; ``stepped`` =
    enumerated (string choices) or a quantized float with a real range step.
    """
    if bool(getattr(param, "is_boolean", False)):
        return "bool", None

    valid_values = getattr(param, "valid_values", None)
    n_values = len(valid_values) if valid_values else None

    if getattr(param, "type", None) is str:
        # Enumerated string param: choices live in valid_values.
        return "stepped", n_values

    # Continuous or quantized float: range == (min, max, step_or_None).
    param_range = getattr(param, "range", None)
    step = (
        param_range[2]
        if isinstance(param_range, (tuple, list)) and len(param_range) >= 3
        else None
    )
    if step is not None and n_values is not None and n_values >= 2:
        return "stepped", n_values
    return "float", None


# --- PluginInfo construction ------------------------------------------------


def _detect_channels(plugin: Any, is_instrument: bool) -> int:
    """Detect the plugin's output channel count via a minimal probe render.

    Uses pedalboard's DEFAULT channel layout (no hardcoded bus width): an empty
    instrument render or a 2-channel effect probe, reading ``out.shape[0]``.
    """
    if is_instrument:
        out = plugin(
            [], duration=_DETECT_DURATION_S, sample_rate=_DETECT_SR, reset=True
        )
    else:
        probe_in = np.zeros((2, _DETECT_FRAMES), dtype=np.float32)
        out = plugin(probe_in, sample_rate=_DETECT_SR, reset=True)
    return int(out.shape[0]) if getattr(out, "ndim", 1) == 2 else 1


def _build_plugin_info(plugin: Any) -> PluginInfo:
    """Build :class:`PluginInfo` from ``plugin.parameters`` (B1, C2, C3, C5)."""
    params = plugin.parameters  # ReadOnlyDictWrapper, keys = python_name (B1 C2)
    param_infos: list[ParamInfo] = []
    for index, python_name in enumerate(params.keys()):
        param = params[python_name]
        kind, num_steps = _classify(param)
        param_infos.append(
            ParamInfo(
                name=python_name,
                index=index,
                kind=kind,
                num_steps=num_steps,
                # B1 C3: no default_value attr on 0.9.23 -> derive the effective
                # init-patch default from the load-time normalized raw_value.
                default=float(param.raw_value),
            )
        )

    is_instrument = bool(plugin.is_instrument)  # B1 C4: route on this, not is_effect
    channels = _detect_channels(plugin, is_instrument)
    return PluginInfo(
        name=plugin.name,
        format="vst3",
        params=param_infos,
        # pedalboard exposes no per-bus channel counts; Surge is symmetric
        # 2-in/2-out and the detection render observes the natural layout.
        input_channels=channels,
        output_channels=channels,
        is_instrument=is_instrument,
        # B1 R3: Surge renders non-silent headless -> no GUI init required.
        needs_gui=False,
        latency_samples=int(plugin.reported_latency_samples),
    )


# --- resolved-param application ---------------------------------------------


def _iter_resolved_params(param_set: Any) -> list[Any]:
    """Normalize the request ``param_set`` to a list of resolved params.

    Accepts an E2 ``ResolvedSpec`` (``.params``), a bare sequence of
    ``ResolvedParam``-shaped objects (``.name``/``.index``/``.value``), or ``None``.
    """
    if param_set is None:
        return []
    params = getattr(param_set, "params", param_set)
    return list(params)


def _apply_params(plugin: Any, info: PluginInfo, param_set: Any) -> None:
    """Apply the resolved normalized vector by name/index via ``param.raw_value``.

    Names come from ``probe()`` (never hardcoded). ``ResolvedParam.value`` is the
    normalized [0,1] domain (E2), which is exactly ``raw_value``'s scale (B1).
    """
    params = plugin.parameters
    keys = set(params.keys())
    for resolved in _iter_resolved_params(param_set):
        name = getattr(resolved, "name", None)
        if name not in keys:
            # Fall back to the index -> python_name mapping from the probe.
            index = getattr(resolved, "index", None)
            if index is not None and 0 <= index < len(info.params):
                name = info.params[index].name
        if name not in keys:
            raise _render_error(
                PARAM_NOT_ON_PLUGIN,
                f"resolved param {getattr(resolved, 'name', None)!r} "
                f"(index {getattr(resolved, 'index', None)}) is not present on "
                f"plugin {info.name!r}",
                {
                    "name": getattr(resolved, "name", None),
                    "index": getattr(resolved, "index", None),
                    "plugin": info.name,
                },
            )
        params[name].raw_value = float(resolved.value)


def _reinject_raw_state(plugin: Any, raw_state: RawState) -> None:
    """Re-inject a validated ``raw_state`` blob into the plugin (by design)."""
    blob_ref = raw_state.blob_ref
    if not blob_ref or not Path(blob_ref).is_file():
        raise _render_error(
            RAW_STATE_BLOB_MISSING,
            f"raw_state validated but its blob_ref is absent: {blob_ref!r}",
            {"blob_ref": blob_ref},
        )
    plugin.raw_state = Path(blob_ref).read_bytes()


# --- backend ----------------------------------------------------------------


class PedalboardVST3Backend:
    """v1 RenderBackend: pedalboard -> VST3 (implements C3)."""

    id = BACKEND_ID

    def __init__(self, render_dir: Optional[Path] = None) -> None:
        # version -> report.render.backend_version (C3): the pedalboard host rev.
        self.version = _pedalboard_version()
        self._render_dir = Path(render_dir) if render_dir is not None else None
        # Instance-owned temp render dir for the render_dir-less path, created
        # lazily and REUSED across renders (Finding 1: never mkdtemp-per-render).
        # Cleaned by close()/the context manager. None until first use, and never
        # set when a caller-supplied render_dir owns the wav lifecycle instead.
        self._owned_temp_dir: Optional[str] = None
        # PluginInfo cache keyed by binary_sha256 (by design).
        self._probe_cache: dict[str, PluginInfo] = {}

    def close(self) -> None:
        """Remove the instance-owned temp render dir if one was lazily created.

        Finding 1: the render_dir-less path owns at most ONE temp dir (created by
        :meth:`_ensure_owned_temp_dir`), so a direct backend user has a bounded,
        explicitly-cleanable footprint. A caller-supplied ``render_dir`` is owned
        by the caller and is NEVER removed here. Idempotent.
        """
        if self._owned_temp_dir is not None:
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
            self._owned_temp_dir = None

    def __enter__(self) -> "PedalboardVST3Backend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- C3 protocol ---------------------------------------------------------

    def probe(self, plugin_path: Path) -> PluginInfo:
        """Introspect the plugin's params + I/O, cached by ``binary_sha256``."""
        path = Path(plugin_path)
        key = binary_sha256(path)
        cached = self._probe_cache.get(key)
        if cached is not None:
            return cached
        plugin = self._load(path)
        try:
            info = _build_plugin_info(plugin)
        except SonoscopeError:
            # Already a typed, exit-code-mapped error (e.g. a nested RenderError
            # from a helper): propagate UNCHANGED, never double-wrap it.
            raise
        except Exception as exc:  # noqa: BLE001 - map raw probe faults to RenderError
            # A RAW exception (channel detection / param parsing) would otherwise
            # escape probe() as an unmapped INTERNAL_ERROR (exit 1); map it to a
            # structured RenderError (component "render", exit RENDER=3).
            raise _render_error(
                PLUGIN_PROBE_FAILED,
                f"failed to probe plugin at {path}: {exc}",
                {"path": str(path)},
            ) from exc
        self._probe_cache[key] = info
        return info

    def render(self, req: RenderRequest) -> RenderResult:
        """Render one wav for the request; returns wav + :class:`RenderMeta`."""
        path = Path(req.plugin_path)

        # Validate raw_state BEFORE loading (a stale stamp hard-errors up front,
        # by design); binary_sha256 hashes the bundle on disk without a plugin load.
        binary_hash = binary_sha256(path)
        need_inject = validate_raw_state(req.raw_state, binary_hash)

        info = self.probe(path)
        plugin = self._load(path)
        if need_inject:
            _reinject_raw_state(plugin, req.raw_state)
        _apply_params(plugin, info, req.param_set)

        warnings: list[str] = []
        start = time.perf_counter()
        audio, coercion = _render_audio(plugin, info, req)
        render_wall_ms = int(round((time.perf_counter() - start) * 1000.0))
        warnings.extend(coercion)
        warnings.extend(seed_record_warnings(req.seed))

        duration_s = float(audio.shape[-1]) / float(req.sample_rate_hz)
        out_channels = int(audio.shape[0]) if audio.ndim == 2 else 1
        wav_path = self._write_wav(audio, req.sample_rate_hz)

        meta = RenderMeta(
            sample_rate_hz=req.sample_rate_hz,
            block_size=req.block_size,
            channels=out_channels,
            duration_s=duration_s,
            wav_subtype=WAV_SUBTYPE,
            wav_sha256=_file_sha256(wav_path),
            render_wall_ms=render_wall_ms,
            warnings=list(warnings),
        )
        return RenderResult(
            wav_path=wav_path, render_meta=meta, warnings=list(warnings)
        )

    # -- helpers -------------------------------------------------------------

    def _load(self, path: Path) -> Any:
        import pedalboard

        try:
            return pedalboard.load_plugin(str(path))
        except Exception as exc:  # noqa: BLE001 - re-raised as a mapped RenderError
            raise _render_error(
                PLUGIN_LOAD_FAILED,
                f"pedalboard failed to load plugin at {path}: {exc}",
                {"path": str(path)},
            ) from exc

    def _ensure_owned_temp_dir(self) -> Path:
        """Lazily create ONE reused temp dir for the render_dir-less path.

        Finding 1: with no caller ``render_dir`` the backend must still not
        mkdtemp a NEW directory on EVERY render (an unbounded disk leak on
        iterative/continuous QA runs — nothing ever deleted the per-render dirs).
        Instead it creates at most one instance-owned temp dir on first use and
        reuses it for all subsequent renders; :meth:`close` removes it.
        """
        if self._owned_temp_dir is None:
            self._owned_temp_dir = tempfile.mkdtemp(prefix="sonoscope-render-")
        return Path(self._owned_temp_dir)

    def _write_wav(self, audio: np.ndarray, sample_rate_hz: int) -> Path:
        out_dir = (
            self._render_dir
            if self._render_dir is not None
            else self._ensure_owned_temp_dir()
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / f"render-{uuid.uuid4().hex}.wav"

        # Route through the canonical IEEE-float encoder (no PEAK chunk) rather
        # than soundfile: libsndfile stamps a wall-clock timestamp into the PEAK
        # chunk of float WAVs, so two byte-identical-audio renders taken >~1 s
        # apart would differ by that field — silently breaking the byte-identity
        # that ``is_bit_identical`` depends on (determinism). The canonical
        # encoder makes the bytes a pure function of the samples. ``audio`` is
        # pedalboard's ``(channels, samples)``, which the encoder accepts directly.
        wav_path.write_bytes(
            canonical_float_wav_bytes(audio, int(sample_rate_hz))
        )
        return wav_path


def _render_audio(
    plugin: Any, info: PluginInfo, req: RenderRequest
) -> tuple[np.ndarray, list[str]]:
    """Dispatch the pedalboard overload by stimulus type; return (audio, warnings).

    ``MidiStimulus`` -> instrument overload (B1); ``AudioStimulus`` -> effect
    overload (B1). Emits a channel-coercion ``warnings`` entry when the request
    channel count differs from the plugin's rendered layout (B1 C1: Surge forces
    stereo).
    """
    stimulus = req.stimulus
    warnings: list[str] = []

    if isinstance(stimulus, MidiStimulus):
        out_channels = info.output_channels
        if req.channels != out_channels:
            warnings.append(
                f"channel-coercion: requested {req.channels} channel(s) but "
                f"plugin bus renders {out_channels}; coerced to {out_channels}"
            )
        audio = plugin(
            list(stimulus.messages),
            duration=float(stimulus.duration_s),
            sample_rate=float(req.sample_rate_hz),
            num_channels=out_channels,
            buffer_size=req.block_size,
            reset=True,
        )
        return audio, warnings

    if isinstance(stimulus, AudioStimulus):
        audio_in = np.asarray(stimulus.audio, dtype=np.float32)
        audio = plugin(
            audio_in,
            sample_rate=float(req.sample_rate_hz),
            buffer_size=req.block_size,
            reset=True,
        )
        rendered_channels = int(audio.shape[0]) if audio.ndim == 2 else 1
        if req.channels != rendered_channels:
            warnings.append(
                f"channel-coercion: requested {req.channels} channel(s) but "
                f"plugin bus rendered {rendered_channels}"
            )
        return audio, warnings

    raise _render_error(
        STIMULUS_TYPE_UNKNOWN,
        "req.stimulus must be a MidiStimulus (instrument path) or AudioStimulus "
        f"(effect path); got {type(stimulus).__name__}",
        {"stimulus_type": type(stimulus).__name__},
    )


def _pedalboard_version() -> str:
    import pedalboard

    return str(pedalboard.__version__)
