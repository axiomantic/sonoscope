"""Determinism floors engine tests (Task F2, design §3.5, §8).

Green-mirage discipline for the nondeterminism-floor engine's load-bearing
guarantees:

- **The floor catches real nondeterminism (RED).** A fake render injecting
  per-render amplitude jitter produces a **nonzero** floor for the jittered
  level features (``rms_dbfs`` / ``peak_dbfs``); a byte-deterministic fake
  produces an **exactly zero** floor for every feature (the paired GREEN — a
  nonzero floor genuinely signals nondeterminism, never a mirage).
- **Bit-identity is scoped honestly (§8).** ``is_bit_identical`` is ``True`` only
  when all N renders are byte-identical AND ``patch_class == "noise_free"``. A
  ``patch_class == "noisy"`` render WITHHOLDS the claim even when the bytes happen
  to be identical.
- **Cache round-trips exactly.** A floors object written to
  ``cache/determinism/<binary_sha256>/<patch_class>.json`` reloads to an object
  that is ``==`` the original (exact float / stamp preservation).
- **Every FloorEntry's unit matches its feature.**

Assertions are exact-equality (Level 4+): exact bools, exact ``0.0`` floors,
exact unit map, exact model round-trip. The engine runs the REAL D1
``compute_summary`` on each fake render (a logic-core test, not a stubbed floor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sonoscope.determinism import (
    DEFAULT_REPEATS,
    feature_units,
    floors_cache_path,
    measure_floors,
    read_floors,
    write_floors,
)
from sonoscope.schema.models import DeterminismFloors
from sonoscope.wav_io import canonical_float_wav_bytes

_SR = 48000  # frozen analysis rate — D1 requires the wav to be 48 kHz.
_BINARY_SHA256 = "b" * 64
_RESOLVED_SHA256 = "a" * 64
_STIMULUS_REF = "corpus/signals/logsweep_20-20k_2s.wav"

_RMS_KEY = "deterministic.summary.rms_dbfs"
_PEAK_KEY = "deterministic.summary.peak_dbfs"


class _FakeRender:
    """A fake single-render callable that writes a 48 kHz float32 tone.

    With ``jitter == 0.0`` every call writes byte-identical audio (the
    deterministic / bit-identical regime). With ``jitter > 0.0`` each call
    perturbs the tone amplitude from a seeded RNG, so the level features vary
    across renders (the noisy regime). Each call writes a UNIQUE
    ``render-<i>.wav`` under ``render_dir`` — mirroring the real backend, which
    names every render ``render-<uuid>.wav`` and never overwrites (so a
    deterministic render is genuinely byte-identical file-to-file).
    """

    def __init__(
        self, render_dir: Path, *, jitter: float = 0.0, seed: int = 0
    ) -> None:
        self._render_dir = render_dir
        self._render_dir.mkdir(parents=True, exist_ok=True)
        self._jitter = jitter
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self) -> Path:
        self.calls += 1
        n = int(_SR * 0.2)
        t = np.arange(n, dtype=np.float64) / _SR
        if self._jitter:
            amp = 0.25 + self._jitter * float(self._rng.standard_normal())
        else:
            amp = 0.25  # deterministic: identical audio -> identical bytes.
        tone = (amp * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        stereo = np.stack([tone, tone], axis=0)  # (channels, frames)
        path = self._render_dir / f"render-{self.calls}.wav"
        # Write via the SAME canonical encoder the real backend (E3) uses, so the
        # test faithfully mirrors production: a deterministic (jitter=0.0) render
        # yields byte-identical bytes regardless of wall-clock time. Plain
        # ``soundfile.write`` would stamp a PEAK-chunk timestamp and make renders
        # spanning a >~1 s boundary differ — a spurious, timing-dependent RED.
        path.write_bytes(canonical_float_wav_bytes(stereo, _SR))
        return path


def _measure(
    render: _FakeRender,
    *,
    patch_class: str,
    repeats: int = DEFAULT_REPEATS,
    method: str = "range",
    generated_at: str = "2026-07-04T12:00:00Z",
) -> DeterminismFloors:
    return measure_floors(
        render,
        binary_sha256=_BINARY_SHA256,
        patch_class=patch_class,  # type: ignore[arg-type]
        resolved_sha256=_RESOLVED_SHA256,
        stimulus_ref=_STIMULUS_REF,
        repeats=repeats,
        method=method,  # type: ignore[arg-type]
        generated_at=generated_at,
    )


# --- RED: the floor catches real nondeterminism -----------------------------


def test_floor_nonzero_on_noisy(tmp_path: Path) -> None:
    """A jittered fake render -> floor > 0 for the jittered level features, and
    an exactly-zero floor for a feature the amplitude jitter cannot move (the
    frozen ``sample_rate_hz``) — the RED-proving nondeterminism catch (§8)."""
    render = _FakeRender(tmp_path / "noisy", jitter=0.05, seed=1)
    floors = _measure(render, patch_class="noisy", repeats=5)

    assert render.calls == 5
    assert floors.floors[_RMS_KEY].floor > 0.0
    assert floors.floors[_PEAK_KEY].floor > 0.0
    # A pure amplitude jitter cannot change the frozen render rate: its floor is
    # exactly zero. This is the paired GREEN that keeps the nonzero floor above
    # from being a mirage.
    assert floors.floors["deterministic.summary.sample_rate_hz"].floor == 0.0


def test_deterministic_floor_is_zero(tmp_path: Path) -> None:
    """A byte-deterministic fake render -> EVERY feature floor is exactly 0.0
    (the GREEN half: no nondeterminism, no floor)."""
    render = _FakeRender(tmp_path / "det", jitter=0.0)
    floors = _measure(render, patch_class="noise_free", repeats=5)

    assert [entry.floor for entry in floors.floors.values()] == [
        0.0 for _ in floors.floors
    ]


# --- bit-identity scoped to noise_free (§8) ---------------------------------


def test_bit_identical_only_for_noise_free(tmp_path: Path) -> None:
    """Identical renders claim bit-identity ONLY under ``patch_class ==
    noise_free``; a ``noisy`` patch_class withholds the claim even though the
    bytes are identical (§8, is_bit_identical true only when noise_free)."""
    nf = _measure(
        _FakeRender(tmp_path / "nf", jitter=0.0),
        patch_class="noise_free",
        repeats=5,
    )
    noisy = _measure(
        _FakeRender(tmp_path / "noisy_det", jitter=0.0),
        patch_class="noisy",
        repeats=5,
    )

    assert nf.is_bit_identical is True
    assert noisy.is_bit_identical is False


def test_bit_identical_false_when_noise_free_bytes_differ(tmp_path: Path) -> None:
    """A ``noise_free`` patch_class whose renders are NOT byte-identical (jitter
    injected) withholds the bit-repro claim: ``is_bit_identical is False`` (§8,
    is_bit_identical true only when the bytes ACTUALLY match — the noise_free
    regime alone is insufficient). This is the RED half that would fail under a
    mutation dropping the ``all_byte_identical`` conjunct (leaving just
    ``is_bit_identical = patch_class == "noise_free"``)."""
    floors = _measure(
        _FakeRender(tmp_path / "nf_jitter", jitter=0.05, seed=4),
        patch_class="noise_free",
        repeats=5,
    )

    assert floors.is_bit_identical is False


# --- cache round-trip -------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    """A written floors cache reloads to an EXACTLY equal floors object, at the
    ``cache/determinism/<binary_sha256>/<patch_class>.json`` path (§3.5)."""
    floors = _measure(
        _FakeRender(tmp_path / "rt", jitter=0.05, seed=2),
        patch_class="noisy",
        repeats=4,
    )
    path = write_floors(floors, cache_root=tmp_path)

    assert path == floors_cache_path(
        _BINARY_SHA256, "noisy", cache_root=tmp_path
    )
    reloaded = read_floors(_BINARY_SHA256, "noisy", cache_root=tmp_path)
    assert reloaded == floors


def test_read_floors_absent_returns_none(tmp_path: Path) -> None:
    """A cache read for an unmeasured key returns ``None`` (never a fabricated
    floor) — the honest miss path."""
    assert read_floors(_BINARY_SHA256, "noise_free", cache_root=tmp_path) is None


# --- unit matches feature ---------------------------------------------------


def test_floor_unit_matches_feature(tmp_path: Path) -> None:
    """Each FloorEntry's unit matches its feature: the emitted floors map has
    exactly one entry per summary feature, and every entry's unit equals the
    canonical ``feature_units()`` value (§3.5)."""
    floors = _measure(
        _FakeRender(tmp_path / "units", jitter=0.01, seed=3),
        patch_class="noisy",
        repeats=3,
    )

    assert {key: entry.unit for key, entry in floors.floors.items()} == (
        feature_units()
    )
    # Spot-check the canonical units are the feature-appropriate ones.
    assert floors.floors["deterministic.summary.spectral_centroid_hz"].unit == (
        "hz"
    )
    assert floors.floors[_RMS_KEY].unit == "dbfs"
    assert floors.floors["deterministic.summary.spectral_flatness"].unit == (
        "ratio"
    )
    assert floors.floors["deterministic.summary.onset_count"].unit == "count"
    assert floors.floors["deterministic.summary.tempo_bpm"].unit == "bpm"
    assert floors.floors["deterministic.summary.mfcc_mean.0"].unit == "mfcc"
