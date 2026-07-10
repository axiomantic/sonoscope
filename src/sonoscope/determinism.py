"""Determinism floors engine — the ``determinism`` command's engine (Task F2,
by design).

Renders one ``(plugin, stimulus, param-set)`` **N times** (default 5) and derives
a per-feature **nondeterminism floor**: for each ``deterministic.summary`` feature
the spread of its value across the N renders (``range`` = ``max - min``, default;
or ``variance`` = the sample standard deviation). The result is a C1
:class:`DeterminismFloors` object that is (a) cached at
``cache/determinism/<binary_sha256>/<patch_class>.json`` keyed by
``(binary_sha256, patch_class)`` and (b) echoed inline into every
``report.render.determinism.floors`` produced against that key. F3
(``iterate``) reads this object for its PASS / INCONCLUSIVE decision.

Design invariants honored here:

- **The floor is the source of truth (by design).** The engine never *assumes* a plugin
  is deterministic — it MEASURES the spread across N real renders and reports it.
- **Bit-identity is scoped honestly (by design).** ``is_bit_identical`` is ``True`` only
  when all N renders are byte-identical AND ``patch_class == "noise_free"``. A
  ``noisy`` patch_class WITHHOLDS the bit-repro claim even if the bytes happen to
  match (bit-identical output is asserted only for the ``noise_free`` regime).
- **D1 is the ground truth (Depends on D1).** Each render's features come from the
  real :func:`sonoscope.features.librosa_features.compute_summary`; the engine
  measures the spread of those ground-truth values.

**Interface / injection seam (Depends on E5).** :func:`measure_floors` takes an
injectable ``render_once`` callable that renders once and returns a wav path. The
engine calls it N times, reading each render's bytes + D1 features immediately.
H1 wires E5's :func:`sonoscope.render_orchestrator.render` (resolve → subprocess
render → wav) into this seam; a test injects a fake render. This keeps the engine
a pure, deterministically-testable floor computer without coupling it to E5's
subprocess/pickling machinery, while D1 stays a real dependency (the fake supplies
audio, not stubbed feature values).

**CLI-wiring boundary (I6).** F2 builds only the engine; wiring the
``determinism`` command to it is owned by H1 (C5 ships the stub). This module does
NOT touch ``cli.py``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from sonoscope.features.librosa_features import FROZEN_PARAMS, compute_summary
from sonoscope.schema.models import (
    DeterminismFloors,
    DeterministicSummary,
    FloorEntry,
    FloorMethod,
    PatchClass,
)

# --- constants (by design) ---------------------------------------------------

#: Default number of renders compared to measure the floor (by design).
DEFAULT_REPEATS: int = 5

#: Default floor method: ``range`` (``max - min``) is the conservative choice
#: (by design). ``variance`` reports the sample standard deviation instead.
DEFAULT_METHOD: FloorMethod = "range"

#: Cache root + subdir → ``cache/determinism/<binary_sha256>/<patch_class>.json``
#: (``floors_ref``). Tests pass ``cache_root=tmp_path`` so the repo
#: ``cache/`` dir is never polluted with test artifacts.
DEFAULT_CACHE_ROOT: Path = Path("cache")
_CACHE_SUBDIR: str = "determinism"

#: Feature-key prefix mirroring the design's floors example
#: (``deterministic.summary.spectral_centroid_hz``).
_FEATURE_PREFIX: str = "deterministic.summary."

#: Scalar ``DeterministicSummary`` features → unit (unit "matches
#: the feature"). Order mirrors the C1 ``DeterministicSummary`` declaration so the
#: floors map iterates in a stable, contract-aligned order. Every summary feature
#: gets an entry (the design's "one entry per deterministic.summary feature"); a
#: render-config field that cannot exhibit nondeterminism simply measures a
#: floor of ``0.0``. The two MFCC vectors are expanded per coefficient below.
_SCALAR_UNITS: dict[str, str] = {
    "duration_s": "s",
    "sample_rate_hz": "hz",
    "channels": "count",
    "rms_dbfs": "dbfs",
    "peak_dbfs": "dbfs",
    "crest_factor_db": "db",
    "dc_offset": "amplitude",
    "spectral_centroid_hz": "hz",
    "spectral_bandwidth_hz": "hz",
    "spectral_rolloff_hz": "hz",
    "spectral_flatness": "ratio",
    "zero_crossing_rate": "ratio",
    "onset_count": "count",
    "onset_rate_hz": "hz",
    "tempo_bpm": "bpm",
    "tempo_confidence": "ratio",
}

#: The MFCC vector features (list-valued) expanded to one floor per coefficient.
_MFCC_FEATURES: tuple[str, ...] = ("mfcc_mean", "mfcc_std")
_MFCC_UNIT: str = "mfcc"


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` stamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- feature specification (one entry per summary feature) --------------------


def _feature_specs() -> list[
    tuple[str, str, Callable[[DeterministicSummary], Optional[float]]]
]:
    """Ordered ``(feature_key, unit, extractor)`` list — one per summary feature.

    Scalar features extract their attribute directly; the two MFCC vectors are
    expanded to one entry per coefficient (``mfcc_mean.<i>``), giving a single
    ``float`` floor per feature dimension. Extractors return ``None`` when a
    feature is suppressed for that render (e.g. ``tempo_bpm`` under octave-error
    mitigation, D1) — the floor is then measured over the values that ARE present.
    """
    specs: list[
        tuple[str, str, Callable[[DeterministicSummary], Optional[float]]]
    ] = []
    for name, unit in _SCALAR_UNITS.items():
        specs.append(
            (
                _FEATURE_PREFIX + name,
                unit,
                lambda summary, _name=name: _as_optional_float(
                    getattr(summary, _name)
                ),
            )
        )
    n_mfcc = int(FROZEN_PARAMS["n_mfcc"])
    for name in _MFCC_FEATURES:
        for idx in range(n_mfcc):
            specs.append(
                (
                    f"{_FEATURE_PREFIX}{name}.{idx}",
                    _MFCC_UNIT,
                    lambda summary, _name=name, _idx=idx: _as_optional_float(
                        getattr(summary, _name)[_idx]
                    ),
                )
            )
    return specs


def _as_optional_float(value: object) -> Optional[float]:
    """Coerce a summary scalar to ``float``; pass ``None`` through unchanged."""
    if value is None:
        return None
    return float(value)


def feature_units() -> dict[str, str]:
    """The canonical ``feature_key -> unit`` map for every emitted floor.

    Exposed so callers/tests can assert every :class:`FloorEntry`'s unit matches
    its feature without duplicating the mapping.
    """
    return {key: unit for key, unit, _ in _feature_specs()}


# --- floor computation (by design) --------------------------------------------


def _compute_floor(values: list[float], method: FloorMethod) -> float:
    """Per-feature nondeterminism floor over the N (present) render values.

    ``range`` -> ``max - min`` (conservative default); ``variance`` -> the sample
    standard deviation (``ddof=1``). Fewer than two present values have no
    measurable spread, so the floor is ``0.0`` (never an undefined stddev).
    """
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if method == "range":
        return float(np.max(arr) - np.min(arr))
    # "variance" -> sample standard deviation across the N renders.
    return float(np.std(arr, ddof=1))


def _load_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a rendered wav as channel-major float32 ``(channels, frames)`` + sr.

    soundfile reads frame-major ``(frames, channels)``; transpose to the
    channel-major layout D1 expects (mirrors the render/analysis loaders).
    """
    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data.T, dtype=np.float32)
    return audio, int(sample_rate)


# --- public engine ----------------------------------------------------------


def measure_floors(
    render_once: Callable[[], Path],
    *,
    binary_sha256: str,
    patch_class: PatchClass,
    resolved_sha256: str,
    stimulus_ref: str,
    repeats: int = DEFAULT_REPEATS,
    method: FloorMethod = DEFAULT_METHOD,
    generated_at: Optional[str] = None,
) -> DeterminismFloors:
    """Render N times and derive the per-feature nondeterminism floor.

    ``render_once`` renders one ``(plugin, stimulus, param-set)`` and returns its
    wav path; it is called ``repeats`` times (default 5). Each render's bytes (for
    bit-identity) and D1 features (via :func:`compute_summary`, ground truth) are
    read immediately, before the next call may overwrite the path. Returns the C1
    :class:`DeterminismFloors` object; :func:`write_floors` persists it to the
    cache.

    ``is_bit_identical`` is ``True`` only when every render is byte-identical AND
    ``patch_class == "noise_free"`` — the ``noisy`` regime withholds the bit-repro
    claim even if the bytes match. ``method`` selects ``range`` (default) or
    ``variance`` (sample stddev). Measuring a floor is meaningless below two
    renders, so ``repeats < 2`` is a hard :class:`ValueError`; an unknown
    ``method`` likewise.
    """
    if repeats < 2:
        raise ValueError(
            f"repeats must be >= 2 to measure a nondeterminism floor; got {repeats}"
        )
    if method not in ("range", "variance"):
        raise ValueError(
            f"method must be 'range' or 'variance'; got {method!r}"
        )

    stamp = generated_at or _now_iso()

    # Render N times; capture each render's bytes + D1 features immediately so a
    # reused wav path (the fake/real single-file case) never loses a render.
    wav_bytes: list[bytes] = []
    summaries: list[DeterministicSummary] = []
    for _ in range(repeats):
        wav_path = Path(render_once())
        wav_bytes.append(wav_path.read_bytes())
        audio, sample_rate = _load_wav(wav_path)
        summaries.append(compute_summary(audio, sample_rate).summary)

    # Bit-identity: byte-identical across all N AND scoped to the noise_free
    # regime (a noisy patch_class never claims bit-identity, even if identical).
    all_byte_identical = all(b == wav_bytes[0] for b in wav_bytes)
    is_bit_identical = all_byte_identical and patch_class == "noise_free"

    # One FloorEntry per summary feature — the spread of its ground-truth
    # value across the N renders. A suppressed (None) feature contributes no
    # value; its per-feature ``repeats`` reflects the values actually compared.
    floors: dict[str, FloorEntry] = {}
    for key, unit, extract in _feature_specs():
        values = [
            v for v in (extract(summary) for summary in summaries) if v is not None
        ]
        floors[key] = FloorEntry(
            floor=_compute_floor(values, method),
            unit=unit,
            method=method,
            repeats=len(values),
            timestamp=stamp,
            binary_sha256=binary_sha256,
            patch_class=patch_class,
        )

    return DeterminismFloors(
        generated_at=stamp,
        binary_sha256=binary_sha256,
        patch_class=patch_class,
        resolved_sha256=resolved_sha256,
        stimulus_ref=stimulus_ref,
        repeats=repeats,
        is_bit_identical=is_bit_identical,
        floors=floors,
    )


# --- cache read / write (floors_ref) ------------------------------------------


def floors_cache_path(
    binary_sha256: str,
    patch_class: PatchClass,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    """The floors cache path for a key: ``<cache_root>/determinism/<binary_sha256>/
    <patch_class>.json`` (``floors_ref``, keyed by
    ``(binary_sha256, patch_class)``)."""
    return Path(cache_root) / _CACHE_SUBDIR / binary_sha256 / f"{patch_class}.json"


def write_floors(
    floors: DeterminismFloors, cache_root: Path = DEFAULT_CACHE_ROOT
) -> Path:
    """Persist a floors object to its ``(binary_sha256, patch_class)`` cache path.

    Serialised via the C1 model so the on-disk JSON is exactly the schema
    contract; parent dirs are created as needed. Returns the written path.

    The write is **atomic**: the JSON is written to a unique sibling temp file
    then :func:`os.replace`\\ d onto the target path (an atomic rename within the
    same directory on POSIX). A crash or concurrent run mid-write can therefore
    never leave a truncated JSON that would hard-fail :func:`read_floors` on
    every subsequent run — the cache file is only ever seen fully-written.
    """
    path = floors_cache_path(
        floors.binary_sha256, floors.patch_class, cache_root
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = floors.model_dump_json(indent=2)
    # Unique same-dir temp (pid + object id) so concurrent writers never clobber
    # each other's temp before the atomic rename onto ``path``.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{id(floors)}.tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        # Never leave the temp behind on failure (KeyboardInterrupt included).
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def read_floors(
    binary_sha256: str,
    patch_class: PatchClass,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Optional[DeterminismFloors]:
    """Load a cached floors object for a key, or ``None`` if none is cached.

    An absent cache is an honest miss (``None``), never a fabricated floor — the
    caller (F3/H1) then measures + writes one. A present cache round-trips exactly
    to the object that produced it (validated against the C1 model on load).
    """
    path = floors_cache_path(binary_sha256, patch_class, cache_root)
    if not path.is_file():
        return None
    return DeterminismFloors.model_validate_json(
        path.read_text(encoding="utf-8")
    )
