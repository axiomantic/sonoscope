"""Librosa deterministic summary-feature tests (Task D1, §4.2).

Green-mirage discipline: every acceptance check ships a fixture that provably
catches a real failure.

- ``test_centroid_tracks_brightness`` — a bright signal must report a higher
  spectral centroid than a dark one (fails if centroid is stubbed/constant).
- ``test_reproducible_bit_identical`` — two runs on the same wav produce an
  exactly-equal summary dict (bit-repro regime, I3).
- ``test_tempo_suppressed_low_onsets`` — a 1-onset signal suppresses
  ``tempo_bpm`` and emits the exact machine-readable note (octave-error
  mitigation; never a confident wrong BPM).
- ``test_params_sha256_changes_on_param_change`` — perturbing any frozen param
  changes the hash (I4: param drift is detectable).
- ``test_peak_is_max_across_channels`` — stereo peak is the max channel peak,
  not the mean (I5 reduction rule).

Fixtures are small synthetic numpy signals (sine tones, bursts) — no external
audio files, no randomness — so librosa output is deterministic.
"""

import numpy as np
import pytest

from sonoscope.features import compute_summary as pkg_compute_summary
from sonoscope.features import params_sha256 as pkg_params_sha256
from sonoscope.features.librosa_features import (
    FROZEN_PARAMS,
    NOTE_TEMPO_LOW_ONSETS,
    compute_summary,
    params_sha256,
)

SR = 48000


def _tone(freq: float, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    """Single-channel pure sine tone at ``freq`` Hz (shape ``(n_samples,)``)."""
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _one_onset(dur: float = 2.0) -> np.ndarray:
    """Silence with a single 50 ms tone burst at 0.5 s -> exactly one onset."""
    n = int(dur * SR)
    sig = np.zeros(n, dtype=np.float32)
    start = int(0.5 * SR)
    length = int(0.05 * SR)
    t = np.arange(length, dtype=np.float64) / SR
    sig[start : start + length] = (0.8 * np.sin(2.0 * np.pi * 440.0 * t)).astype(
        np.float32
    )
    return sig


# --- Acceptance: brightness -> centroid -------------------------------------


def test_centroid_tracks_brightness():
    dark = compute_summary(_tone(200.0), SR).summary.spectral_centroid_hz
    bright = compute_summary(_tone(8000.0), SR).summary.spectral_centroid_hz
    # Bright tone must exceed dark tone well beyond any framing artifact.
    assert bright > dark + 1000.0


# --- Acceptance: bit-identical reproducibility (I3) -------------------------


def test_reproducible_bit_identical():
    sig = _tone(440.0)
    first = compute_summary(sig, SR).summary.model_dump()
    second = compute_summary(sig.copy(), SR).summary.model_dump()
    assert first == second


# --- Acceptance: non-frozen sample rate is rejected (I4 reproducibility) -----


def test_non_frozen_sample_rate_rejected():
    # params_sha256 reports the frozen sr digest; computing features at any
    # other rate would silently drift from that hash. The guard hard-fails
    # (RED-proving: remove the guard and this raises no error).
    with pytest.raises(ValueError):
        compute_summary(_tone(440.0), sample_rate=44100)


# --- Acceptance: zero-channel input is rejected cleanly (Finding 2) ----------


def test_zero_channel_audio_rejected():
    # Finding 2 (Gemini review, final batch): zero-CHANNEL input (shape (0, N)) has
    # n_samples != 0 but n_channels == 0, so a samples-only guard is bypassed and
    # max(peak_lin_ch) later raises the opaque "max() arg is an empty sequence".
    # The two-dimension guard raises a clean ValueError instead.
    # RED-proving: against the single-condition (n_samples == 0) guard this raises
    # "max() arg is an empty sequence", not this clean message.
    with pytest.raises(
        ValueError, match="audio must have at least one channel and one sample"
    ):
        compute_summary(np.zeros((0, 100), dtype=np.float32), sample_rate=SR)


# --- Acceptance: tempo suppression (octave-error mitigation) ----------------


def test_tempo_suppressed_low_onsets():
    result = compute_summary(_one_onset(), SR)
    assert result.summary.tempo_bpm is None
    assert result.summary.tempo_confidence is None
    # Exact-equality on the notes list: only the low-onset note is present.
    assert result.notes == [NOTE_TEMPO_LOW_ONSETS]


# --- Acceptance: frozen-param hash detects drift (I4) -----------------------


@pytest.mark.parametrize(
    "key,new_value",
    [
        ("sr", 44100),
        ("n_fft", 4096),
        ("hop_length", 256),
        ("n_mfcc", 20),
        ("window", "hamming"),
        ("center", False),
        ("roll_percent", 0.95),
        ("silence_threshold_dbfs", -60.0),
    ],
)
def test_params_sha256_changes_on_param_change(key, new_value):
    base = params_sha256()
    perturbed = dict(FROZEN_PARAMS)
    perturbed[key] = new_value
    assert params_sha256(perturbed) != base
    # The unperturbed hash is stable across calls.
    assert params_sha256() == base


# --- Acceptance: peak = MAX across channels (I5) ----------------------------


def test_peak_is_max_across_channels():
    left = _tone(440.0, amp=0.1)  # quiet channel
    right = _tone(440.0, amp=0.9)  # loud channel
    stereo = np.stack([left, right])  # shape (2, n_samples)
    result = compute_summary(stereo, SR)

    loud_peak_db = 20.0 * np.log10(float(np.max(np.abs(right))))
    mean_of_channel_peaks_db = float(
        np.mean(
            [
                20.0 * np.log10(float(np.max(np.abs(left)))),
                20.0 * np.log10(float(np.max(np.abs(right)))),
            ]
        )
    )

    assert result.summary.channels == 2
    # peak_dbfs equals the loud channel's peak ...
    assert result.summary.peak_dbfs == pytest.approx(loud_peak_db, abs=1e-6)
    # ... and is provably NOT the mean of the two channel peaks.
    assert abs(result.summary.peak_dbfs - mean_of_channel_peaks_db) > 1.0


# --- Package-surface exports (D1 owns __init__) -----------------------------


def test_package_reexports_compute_and_hash():
    assert pkg_params_sha256() == params_sha256()
    result = pkg_compute_summary(_tone(440.0), SR)
    assert result.summary.channels == 1
