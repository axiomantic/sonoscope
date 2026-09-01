"""Deterministic integrity flags (Task D2, by design, M1/I5).

Pure function ``compute_integrity(audio) -> IntegrityBlock`` producing the
``deterministic.integrity`` booleans + counts of the analysis contract (schema
``IntegrityBlock``): silence, NaN, Inf, denormal, clipping, and DC-offset
tripwire inputs.

Design invariants honored here:

- **M1 thresholds (by design, "Integrity thresholds").** ``has_denormal``
  fires on any sample with ``0 < |x| < 1.1754944e-38`` (smallest normal
  float32); ``clip_count`` counts samples with ``|x| >= 1.0``;
  ``dc_offset_exceeds`` fires when ``|dc_offset| > 0.001`` (linear full scale);
  ``is_silent`` fires when RMS is at/below the frozen
  ``silence_threshold_dbfs = -80.0``.
- **Two silence predicates, one threshold.** ``is_silent`` is the OR-combined
  dead-channel tripwire (ANY channel at/below the cutoff). ``all_channels_silent``
  is the AND-combined whole-file predicate (EVERY channel at/below it) and is what
  gates descriptor interpretation — a stereo file with one dead channel is a
  broken channel, not a silent file, and must keep its descriptors. Both read the
  SAME ``SILENCE_THRESHOLD_DBFS``, so the two can never drift apart. A channel
  whose RMS is non-finite (NaN/Inf) is a fault, not silence, and is False under
  both.
- **Multi-channel combine (I5).** Every check runs per channel, then combines:
  boolean flags are the **OR** across channels; counts are the **SUM** across
  channels (``clip_fraction`` = summed ``clip_count`` / total samples over all
  channels). Audio is never downmixed to mono first.
- **Frozen threshold reuse (I4).** ``silence_threshold_dbfs`` is read from D1's
  hashed ``FROZEN_PARAMS`` so the silence cutoff has a single source of truth and
  any drift is caught by ``params_sha256``.

D2 does NOT edit ``features/__init__`` (D1 owns package exports, M1); it imports
the frozen param set by module path.
"""

from __future__ import annotations

import numpy as np

# Import the frozen silence threshold by module path (D2 must not edit
# ``features/__init__``; D1 owns that surface). Reusing the hashed value keeps
# a single source of truth for the -80 dBFS cutoff (I4).
from sonoscope.features.librosa_features import FROZEN_PARAMS
from sonoscope.schema.models import IntegrityBlock

# --- Frozen M1 thresholds (by design, "Integrity thresholds") ----------------
# ``is_silent`` cutoff, reused from D1's hashed FROZEN_PARAMS (I4).
SILENCE_THRESHOLD_DBFS: float = float(FROZEN_PARAMS["silence_threshold_dbfs"])
# Smallest NORMAL float32; anything in the open interval (0, this) is subnormal.
DENORMAL_MIN_NORMAL_FLOAT32: float = 1.1754944e-38
# DC-offset tripwire cutoff, linear full scale.
DC_OFFSET_THRESHOLD: float = 0.001

# Amplitude floor for dBFS conversion so all-zero silence yields a finite (very
# negative) dBFS instead of -inf. Matches D1's convention.
_AMPLITUDE_FLOOR: float = 1e-12


def _to_channels(audio: np.ndarray) -> np.ndarray:
    """Normalise input to ``(channels, n_samples)`` float32 (per-channel scan).

    1-D input is treated as single-channel. float32 mirrors the rendered
    ``PCM_F32`` buffer and D1's channel handling, preserving NaN/Inf/subnormal
    values exactly.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"audio must be 1-D (mono) or 2-D (channels, samples); got ndim={arr.ndim}"
    )


def _dbfs(linear: float) -> float:
    """Linear amplitude -> dBFS with a finite floor (no -inf)."""
    return 20.0 * float(np.log10(max(linear, _AMPLITUDE_FLOOR)))


def compute_integrity(audio: np.ndarray) -> IntegrityBlock:
    """Compute the ``deterministic.integrity`` block from a wav array (by design, M1).

    ``audio`` is ``(channels, n_samples)`` (or 1-D mono). Each check runs per
    channel and combines by OR (booleans) / SUM (counts) across channels (I5).
    Returns the C1 ``IntegrityBlock`` populated exactly.
    """
    channels = _to_channels(audio)
    n_channels, n_samples = channels.shape
    # Reject degenerate buffers on EITHER axis (MINOR-1). A ``(0, N)`` array
    # (zero channels) would otherwise slip past a samples-only guard and reach
    # ``clip_fraction = clip_count / total_samples`` with ``total_samples == 0``,
    # raising a raw ZeroDivisionError. Guarding both axes covers ``(0, N)``,
    # ``(N, 0)``, and 1-D empty inputs with a clean ValueError.
    if n_channels == 0 or n_samples == 0:
        raise ValueError(
            "audio has an empty channel or sample axis "
            f"(shape={channels.shape}); need at least one channel and one sample"
        )

    # Per-channel accumulators combined per I5 (OR for booleans, SUM for counts).
    has_nan = False
    has_inf = False
    has_denormal = False
    is_silent = False
    # AND-combined companion to ``is_silent``. Seeded True and ANDed per channel;
    # the degenerate zero-channel case cannot reach here (guarded above), so this
    # is never vacuously True.
    all_channels_silent = True
    dc_offset_exceeds = False
    clip_count = 0

    for c in range(n_channels):
        x = channels[c]
        absx = np.abs(x)

        # Fault flags (by design, M1). Comparisons against NaN are False, so a
        # NaN sample is not miscounted as a denormal or a clip; Inf is finite-
        # excluded from the denormal band but does satisfy |x| >= 1.0.
        has_nan |= bool(np.isnan(x).any())
        has_inf |= bool(np.isinf(x).any())
        has_denormal |= bool(
            np.any((absx > 0.0) & (absx < DENORMAL_MIN_NORMAL_FLOAT32))
        )
        clip_count += int(np.count_nonzero(absx >= 1.0))

        # Silence: RMS at/below -80 dBFS (by design). A non-finite RMS
        # (NaN/Inf present) is not silence.
        rms_lin = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        channel_silent = bool(
            np.isfinite(rms_lin) and _dbfs(rms_lin) <= SILENCE_THRESHOLD_DBFS
        )
        is_silent |= channel_silent
        all_channels_silent &= channel_silent

        # DC offset: channel mean magnitude past the linear-full-scale cutoff.
        dc = float(np.mean(x, dtype=np.float64))
        dc_offset_exceeds |= abs(dc) > DC_OFFSET_THRESHOLD

    total_samples = int(n_channels) * int(n_samples)
    clip_fraction = clip_count / total_samples

    return IntegrityBlock(
        is_silent=is_silent,
        all_channels_silent=all_channels_silent,
        silence_threshold_dbfs=SILENCE_THRESHOLD_DBFS,
        has_nan=has_nan,
        has_inf=has_inf,
        has_denormal=has_denormal,
        clip_count=int(clip_count),
        clip_fraction=float(clip_fraction),
        dc_offset_exceeds=dc_offset_exceeds,
        dc_offset_threshold=DC_OFFSET_THRESHOLD,
    )
