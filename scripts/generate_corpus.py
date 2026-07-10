#!/usr/bin/env python3
"""Regenerate the pinned, checksummed stimulus corpus (Task E1, design §6.1, R4).

The corpus is **script-generated, not committed as opaque binaries**, so every
item is auditable and byte-for-byte regenerable (project policy: pins are law).
This script writes the deterministic signal WAVs + MIDI files under
``corpus/signals`` and ``corpus/midi`` and (re)writes ``corpus/manifest.toml``
recording each item's ``sha256`` + generator + params. ``sonoscope corpus
verify`` (engine: :mod:`sonoscope.corpus`) recomputes those hashes and hard-fails
on drift.

Determinism requirements honored here (design "Determinism requirement"):

- **Pinned RNG seed.** The only stochastic generator (``pink_noise``) draws from
  ``numpy.random.default_rng(SEED)`` with a fixed ``SEED``; numpy is pinned
  (1.26.4) so the stream is reproducible.
- **Fixed dtype / byte order.** All signals are float32, little-endian, written
  through a canonical IEEE-float WAV encoder (below) that emits **no PEAK chunk**
  — libsndfile's PEAK chunk embeds a wall-clock timestamp, which would break
  byte-identity across runs. The canonical encoder writes only ``fmt ``, ``fact``
  and ``data`` chunks, so identical samples yield identical bytes.
- **No timestamps.** MIDI is written via ``mido`` (standard MIDI files carry no
  wall-clock), and the WAV encoder carries none.

Run ``python scripts/generate_corpus.py`` to regenerate in place, or
``--out <dir>`` to generate into another tree (used by the deterministic-regen
tests). ``generate_all(out_dir)`` is importable for the same purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mido
import numpy as np

from sonoscope.wav_io import canonical_float_wav_bytes

# --- Pinned generation constants (recorded in the manifest header) -----------
# Fixed RNG seed for the single stochastic generator (pink_noise). Any change
# here changes pink_noise's bytes and therefore its manifest hash.
SEED: int = 0x12345678  # 305419896
SAMPLE_RATE_HZ: int = 48000
DURATION_S: float = 2.0
N_SAMPLES: int = int(round(DURATION_S * SAMPLE_RATE_HZ))  # 96000

# MIDI timing: 480 ticks/beat at 120 BPM (500000 us/beat) => 960 ticks/second.
MIDI_TICKS_PER_BEAT: int = 480
MIDI_TEMPO_US_PER_BEAT: int = 500000  # 120 BPM
_TICKS_PER_SECOND: int = MIDI_TICKS_PER_BEAT * 1_000_000 // MIDI_TEMPO_US_PER_BEAT
MIDI_VELOCITY: int = 100
# C4 = middle C = MIDI 60 (scientific pitch), so C3 = 48.
MIDI_NOTE_C3: int = 48

CORPUS_SCHEMA: str = "sonoscope-corpus/1"

# The canonical IEEE-float WAV encoder (deterministic, no PEAK chunk) now lives
# in ``sonoscope.wav_io`` so corpus generation, render output (E3), and the
# determinism engine's test fake all share one byte-identical encoder.


# --- Signal generators (48 kHz float32, DURATION_S seconds) ------------------


def _time_axis() -> np.ndarray:
    """Sample times ``t[n] = n / sr`` as float64 (precision for phase accuracy)."""
    return np.arange(N_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ


def gen_impulse() -> np.ndarray:
    """Unit impulse: sample 0 = 1.0, all others 0.0 (IR / latency probe)."""
    sig = np.zeros(N_SAMPLES, dtype=np.float32)
    sig[0] = np.float32(1.0)
    return sig


def gen_logsweep() -> np.ndarray:
    """Exponential (log) sine sweep 20 Hz -> 20 kHz over DURATION_S, amp 0.5."""
    f0, f1 = 20.0, 20000.0
    t = _time_axis()
    k = np.log(f1 / f0)
    phase = 2.0 * np.pi * f0 * DURATION_S / k * (np.exp(t / DURATION_S * k) - 1.0)
    return (0.5 * np.sin(phase)).astype(np.float32)


def gen_pink_noise() -> np.ndarray:
    """Pink (1/f) noise from a fixed-seed white draw, peak-normalized to 0.5.

    FFT method: shape a white Gaussian spectrum by ``1/sqrt(f)`` (bin 0 zeroed),
    inverse-transform, then peak-normalize. Deterministic given ``SEED`` and the
    pinned numpy version.
    """
    rng = np.random.default_rng(SEED)
    white = rng.standard_normal(N_SAMPLES)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(spectrum.size, dtype=np.float64)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0  # drop DC
    shaped = np.fft.irfft(spectrum * scale, n=N_SAMPLES)
    peak = float(np.max(np.abs(shaped)))
    if peak == 0.0:
        peak = 1.0
    return (0.5 * shaped / peak).astype(np.float32)


def gen_tone() -> np.ndarray:
    """Reference 1 kHz sine, amplitude 0.5, DURATION_S seconds."""
    t = _time_axis()
    return (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)


def gen_silence() -> np.ndarray:
    """All-zero signal (pairs with the silent-output tripwire semantics)."""
    return np.zeros(N_SAMPLES, dtype=np.float32)


# --- MIDI generators ---------------------------------------------------------


def _new_midi() -> tuple[mido.MidiFile, mido.MidiTrack]:
    midi = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=MIDI_TEMPO_US_PER_BEAT, time=0)
    )
    return midi, track


def gen_midi_c3_sustain() -> mido.MidiFile:
    """Single sustained C3 (MIDI 48) held for DURATION_S seconds."""
    midi, track = _new_midi()
    hold = int(round(DURATION_S)) * _TICKS_PER_SECOND
    track.append(
        mido.Message("note_on", note=MIDI_NOTE_C3, velocity=MIDI_VELOCITY, time=0)
    )
    track.append(
        mido.Message("note_off", note=MIDI_NOTE_C3, velocity=0, time=hold)
    )
    return midi


def gen_midi_phrase_4note() -> mido.MidiFile:
    """Four-note ascending phrase (C3 E3 G3 C4), each 0.5 s (onset/tempo probe)."""
    midi, track = _new_midi()
    notes = [
        MIDI_NOTE_C3,       # C3
        MIDI_NOTE_C3 + 4,   # E3
        MIDI_NOTE_C3 + 7,   # G3
        MIDI_NOTE_C3 + 12,  # C4
    ]
    step = _TICKS_PER_SECOND // 2  # 0.5 s per note
    for note in notes:
        track.append(
            mido.Message("note_on", note=note, velocity=MIDI_VELOCITY, time=0)
        )
        track.append(
            mido.Message("note_off", note=note, velocity=0, time=step)
        )
    return midi


def _midi_bytes(midi: mido.MidiFile) -> bytes:
    import io

    buf = io.BytesIO()
    midi.save(file=buf)
    return buf.getvalue()


# --- Item catalog (ordered; drives generation + manifest) --------------------


@dataclass(frozen=True)
class _ItemSpec:
    name: str
    path: str  # corpus-relative
    kind: str  # "signal" | "midi"
    generator: str
    builder: Callable[[], bytes]
    params: dict[str, Any]


def _signal_bytes(fn: Callable[[], np.ndarray]) -> Callable[[], bytes]:
    return lambda: canonical_float_wav_bytes(fn(), SAMPLE_RATE_HZ)


_SIGNAL_PARAMS_COMMON = {
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "duration_s": DURATION_S,
    "dtype": "float32",
    "wav_format": "ieee-float-le",
}

ITEM_SPECS: tuple[_ItemSpec, ...] = (
    _ItemSpec(
        name="impulse",
        path="signals/impulse_2s.wav",
        kind="signal",
        generator="gen_impulse",
        builder=_signal_bytes(gen_impulse),
        params={**_SIGNAL_PARAMS_COMMON, "amplitude": 1.0},
    ),
    _ItemSpec(
        name="sweep",
        path="signals/logsweep_20-20k_2s.wav",
        kind="signal",
        generator="gen_logsweep",
        builder=_signal_bytes(gen_logsweep),
        params={
            **_SIGNAL_PARAMS_COMMON,
            "amplitude": 0.5,
            "f0_hz": 20.0,
            "f1_hz": 20000.0,
            "shape": "log-sine",
        },
    ),
    _ItemSpec(
        name="pink_noise",
        path="signals/pink_noise_2s.wav",
        kind="signal",
        generator="gen_pink_noise",
        builder=_signal_bytes(gen_pink_noise),
        params={
            **_SIGNAL_PARAMS_COMMON,
            "amplitude": 0.5,
            "seed": SEED,
            "method": "fft-1/sqrt(f)",
        },
    ),
    _ItemSpec(
        name="tone",
        path="signals/tone_1k_2s.wav",
        kind="signal",
        generator="gen_tone",
        builder=_signal_bytes(gen_tone),
        params={**_SIGNAL_PARAMS_COMMON, "amplitude": 0.5, "frequency_hz": 1000.0},
    ),
    _ItemSpec(
        name="silence",
        path="signals/silence_2s.wav",
        kind="signal",
        generator="gen_silence",
        builder=_signal_bytes(gen_silence),
        params={**_SIGNAL_PARAMS_COMMON, "amplitude": 0.0},
    ),
    _ItemSpec(
        name="c3_sustain",
        path="midi/c3_sustain_2s.mid",
        kind="midi",
        generator="gen_midi_c3_sustain",
        builder=lambda: _midi_bytes(gen_midi_c3_sustain()),
        params={
            "ticks_per_beat": MIDI_TICKS_PER_BEAT,
            "tempo_us_per_beat": MIDI_TEMPO_US_PER_BEAT,
            "note": MIDI_NOTE_C3,
            "velocity": MIDI_VELOCITY,
            "duration_s": DURATION_S,
        },
    ),
    _ItemSpec(
        name="phrase_4note",
        path="midi/phrase_4note.mid",
        kind="midi",
        generator="gen_midi_phrase_4note",
        builder=lambda: _midi_bytes(gen_midi_phrase_4note()),
        params={
            "ticks_per_beat": MIDI_TICKS_PER_BEAT,
            "tempo_us_per_beat": MIDI_TEMPO_US_PER_BEAT,
            "notes": [
                MIDI_NOTE_C3,
                MIDI_NOTE_C3 + 4,
                MIDI_NOTE_C3 + 7,
                MIDI_NOTE_C3 + 12,
            ],
            "velocity": MIDI_VELOCITY,
            "note_duration_s": 0.5,
        },
    ),
)


@dataclass(frozen=True)
class GeneratedItem:
    name: str
    path: str  # corpus-relative
    kind: str
    generator: str
    sha256: str
    params: dict[str, Any]


def generate_all(out_dir: Path) -> list[GeneratedItem]:
    """Generate every corpus item under ``out_dir`` and return their metadata.

    Files are written to ``out_dir/<item.path>`` (parents created). The returned
    order matches :data:`ITEM_SPECS`. Deterministic: identical invocations
    produce byte-identical files (see module docstring).
    """
    out_dir = Path(out_dir)
    generated: list[GeneratedItem] = []
    for spec in ITEM_SPECS:
        payload = spec.builder()
        dest = out_dir / spec.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        generated.append(
            GeneratedItem(
                name=spec.name,
                path=spec.path,
                kind=spec.kind,
                generator=spec.generator,
                sha256=hashlib.sha256(payload).hexdigest(),
                params=dict(spec.params),
            )
        )
    return generated


# --- Manifest rendering (deterministic hand-written TOML) --------------------


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Repr keeps a trailing ``.0`` so floats stay floats on reload.
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"unsupported TOML scalar type: {type(value)!r}")


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return _toml_scalar(value)


def _toml_inline_table(params: dict[str, Any]) -> str:
    body = ", ".join(f"{k} = {_toml_value(v)}" for k, v in params.items())
    return "{ " + body + " }"


def render_manifest_text(items: list[GeneratedItem]) -> str:
    """Render the deterministic ``manifest.toml`` text for ``items``."""
    lines = [
        "# sonoscope stimulus corpus manifest (Task E1, design §6.1, R4).",
        "#",
        "# Pins-are-law: this file records the sha256 + generator + params of",
        "# every pinned stimulus item. `sonoscope corpus verify`",
        "# (sonoscope.corpus.verify) recomputes each hash and returns a FAILURE",
        "# result on drift. Items are script-generated (not opaque binaries):",
        "# regenerate byte-identically with `python scripts/generate_corpus.py`.",
        "#",
        "# All signals are 48 kHz float32 canonical IEEE-float WAV (no PEAK chunk,",
        "# no timestamps); the single stochastic generator (pink_noise) uses the",
        "# pinned RNG seed below. Do not hand-edit hashes — regenerate.",
        "",
        f'schema = "{CORPUS_SCHEMA}"',
        f"seed = {SEED}",
        f"sample_rate_hz = {SAMPLE_RATE_HZ}",
        "",
    ]
    for item in items:
        lines.append(f"[items.{item.name}]")
        lines.append(f'path = "{item.path}"')
        lines.append(f'kind = "{item.kind}"')
        lines.append(f'generator = "{item.generator}"')
        lines.append(f'sha256 = "{item.sha256}"')
        lines.append(f"params = {_toml_inline_table(item.params)}")
        lines.append("")
    return "\n".join(lines)


def write_manifest(items: list[GeneratedItem], manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(render_manifest_text(items), encoding="utf-8")


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_corpus = repo_root / "corpus"

    parser = argparse.ArgumentParser(
        description="Regenerate the pinned sonoscope stimulus corpus."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_corpus,
        help="corpus output directory (default: <repo>/corpus)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: <out>/manifest.toml)",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    manifest_path: Path = args.manifest or (out_dir / "manifest.toml")

    items = generate_all(out_dir)
    write_manifest(items, manifest_path)

    for item in items:
        print(f"{item.name:14s} {item.path:32s} {item.sha256}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
