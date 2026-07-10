"""Byte-deterministic IEEE-float WAV encoding — the single source of truth for
canonical WAV bytes across corpus generation, render output, and determinism.

libsndfile (via soundfile) writes a ``PEAK`` chunk into float WAVs whose header
embeds a **wall-clock timestamp**. Two renders of byte-identical audio taken more
than ~1 s apart therefore differ by that timestamp field, which silently breaks
byte-identity — the load-bearing signal behind ``is_bit_identical`` (determinism
§8) and behind the corpus's byte-for-byte regenerability (§6.1). soundfile 0.12.1
exposes no public API to suppress the PEAK chunk.

:func:`canonical_float_wav_bytes` sidesteps this entirely: it encodes 32-bit IEEE
little-endian float WAV emitting only the ``fmt ``/``fact``/``data`` chunks (no
PEAK, no timestamp), so the byte stream is a pure function of the samples. Every
producer of float WAV bytes in the codebase routes through it so bit-identity is
genuinely trustworthy.
"""

from __future__ import annotations

import struct

import numpy as np

_WAVE_FORMAT_IEEE_FLOAT = 3
_BITS_PER_SAMPLE = 32


def canonical_float_wav_bytes(audio: np.ndarray, sample_rate_hz: int) -> bytes:
    """Encode ``audio`` as a canonical 32-bit IEEE-float little-endian WAV.

    ``audio`` is ``(n_samples,)`` mono or ``(channels, n_samples)``. Emits only
    ``fmt ``/``fact``/``data`` chunks (no PEAK chunk), so the byte stream is a
    pure function of the samples — the basis for byte-identical regeneration.
    """
    arr = np.asarray(audio, dtype="<f4")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(
            f"audio must be 1-D or 2-D (channels, samples); got ndim={arr.ndim}"
        )
    n_channels, n_samples = arr.shape
    # Interleave to frame-major order (ch0[0], ch1[0], ch0[1], ...).
    interleaved = np.ascontiguousarray(arr.T, dtype="<f4").reshape(-1)
    data = interleaved.tobytes()

    byte_rate = sample_rate_hz * n_channels * _BITS_PER_SAMPLE // 8
    block_align = n_channels * _BITS_PER_SAMPLE // 8
    fmt_chunk = struct.pack(
        "<HHIIHH",
        _WAVE_FORMAT_IEEE_FLOAT,
        n_channels,
        sample_rate_hz,
        byte_rate,
        block_align,
        _BITS_PER_SAMPLE,
    )
    fact_chunk = struct.pack("<I", n_samples)

    riff_size = (
        4  # "WAVE"
        + (8 + len(fmt_chunk))
        + (8 + len(fact_chunk))
        + (8 + len(data))
    )

    parts = [
        b"RIFF",
        struct.pack("<I", riff_size),
        b"WAVE",
        b"fmt ",
        struct.pack("<I", len(fmt_chunk)),
        fmt_chunk,
        b"fact",
        struct.pack("<I", len(fact_chunk)),
        fact_chunk,
        b"data",
        struct.pack("<I", len(data)),
        data,
    ]
    return b"".join(parts)
