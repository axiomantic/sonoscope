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
        ("onset_delta", 0.5),
        ("onset_normalize", True),
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


# --- Onset-threshold + tempo-confidence fixtures ----------------------------
# Deterministic synthetic signals (fixed seeds, no external audio).


def _chord(dur: float = 2.0, amp: float = 0.3) -> np.ndarray:
    """Sustained C-major triad (C4/E4/G4) — one attack, no rhythm."""
    return sum(_tone(f, dur, amp) for f in (261.63, 329.63, 392.00)).astype(
        np.float32
    )


def _white_noise(dur: float = 2.0, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(dur * SR))).astype(np.float32)


def _drum_loop(bpm: float = 120.0, dur: float = 8.0, seed: int = 1) -> np.ndarray:
    """Kick on every beat, snare on 2 and 4, hat on every eighth."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm

    def add(at_s: float, seg: np.ndarray) -> None:
        i = int(at_s * SR)
        end = min(n, i + len(seg))
        if i < n:
            y[i:end] += seg[: end - i]

    def kick() -> np.ndarray:
        t = np.arange(int(0.15 * SR)) / SR
        f = 120.0 * np.exp(-t * 30.0) + 45.0
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18.0)

    def snare() -> np.ndarray:
        t = np.arange(int(0.12 * SR)) / SR
        return (
            rng.standard_normal(len(t)) * 0.6 + np.sin(2 * np.pi * 190 * t) * 0.4
        ) * np.exp(-t * 28.0)

    def hat() -> np.ndarray:
        t = np.arange(int(0.04 * SR)) / SR
        return rng.standard_normal(len(t)) * 0.25 * np.exp(-t * 120.0)

    for b in range(int(dur / beat)):
        add(b * beat, kick())
        if b % 4 in (1, 3):
            add(b * beat, snare())
    for e in range(int(dur / (beat / 2))):
        add(e * beat / 2, hat())
    return (y / np.max(np.abs(y)) * 0.8).astype(np.float32)


def _click_track(
    bpm: float = 120.0, dur: float = 8.0, jitter_ms: float = 0.0, seed: int = 7
) -> np.ndarray:
    """Isochronous 2 kHz clicks, optionally jittered by +/- ``jitter_ms``."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    t = np.arange(int(0.005 * SR)) / SR
    click = np.sin(2 * np.pi * 2000.0 * t) * np.exp(-t * 400.0)
    for b in range(int(dur / beat)):
        at = b * beat
        if jitter_ms:
            at += rng.uniform(-jitter_ms, jitter_ms) / 1000.0
        i = int(max(0.0, at) * SR)
        end = min(n, i + len(click))
        if i < n:
            y[i:end] += click[: end - i]
    return (y / np.max(np.abs(y)) * 0.8).astype(np.float32)


# --- RED: steady/noise material must report NO onsets -----------------------
# Before the onset_delta/onset_normalize change these yield 43, 39 and 43
# spurious onsets respectively (the normalised envelope rescales a flat signal's
# noise floor to [0,1], so the default peak-picker fires on it).


def test_steady_sine_has_no_onsets():
    assert compute_summary(_tone(440.0, dur=2.0), SR).summary.onset_count == 0


def test_sustained_chord_has_no_onsets():
    assert compute_summary(_chord(), SR).summary.onset_count == 0


def test_white_noise_reports_no_tempo():
    result = compute_summary(_white_noise(), SR)
    assert result.summary.onset_count == 0
    assert result.summary.tempo_bpm is None
    assert result.summary.tempo_confidence is None
    assert result.notes == [NOTE_TEMPO_LOW_ONSETS]


# --- GREEN: rhythmic material still detects onsets --------------------------


def test_drum_loop_onsets_detected():
    # 120 BPM over 8 s = 32 eighth-note hit positions; the detector resolves 31.
    assert compute_summary(_drum_loop(), SR).summary.onset_count == 31


# --- RED: tempo_confidence is a measured value, not a constant --------------
# Before the change every gate-passing signal reported exactly 1.0, so the
# jitter ordering below could not hold.


def test_click_track_confidence_is_measured():
    result = compute_summary(_click_track(), SR)
    assert result.notes == []
    assert result.summary.tempo_bpm == pytest.approx(119.68085106382979, abs=1e-6)
    assert result.summary.tempo_confidence == pytest.approx(
        0.904282192048749, abs=1e-6
    )


def test_confidence_decreases_with_timing_jitter():
    exact = compute_summary(_click_track(), SR).summary.tempo_confidence
    jittered = compute_summary(
        _click_track(jitter_ms=60.0), SR
    ).summary.tempo_confidence
    assert exact == pytest.approx(0.904282192048749, abs=1e-6)
    assert jittered == pytest.approx(0.30703175878763567, abs=1e-6)
    assert exact > jittered


# --- RED: confidence must describe the REPORTED BPM, not each channel's own --
# A stereo file whose channels disagree (60 BPM left, 160 BPM right) reports the
# MEAN of the two per-channel BPMs -- a tempo neither channel actually has.
# Measuring confidence at each channel's own beat_track BPM scores that
# artefact 0.8611558041261167, a confident wrong BPM. Measured at the reported
# BPM there is no periodicity in either channel, so it is exactly 0.0.


def _stereo_disagreeing() -> np.ndarray:
    left = _drum_loop(bpm=60.0, seed=1)
    right = _drum_loop(bpm=160.0, seed=2)
    n = min(len(left), len(right))
    return np.stack([left[:n], right[:n]])


def _stereo_coherent() -> np.ndarray:
    both = _drum_loop(bpm=120.0, seed=1)
    return np.stack([both, both])


def test_disagreeing_stereo_tempo_is_not_confident():
    result = compute_summary(_stereo_disagreeing(), SR)
    assert result.notes == []
    assert result.summary.tempo_bpm == pytest.approx(140.19756838905775, abs=1e-6)
    assert result.summary.tempo_confidence == pytest.approx(0.0, abs=1e-6)


def test_coherent_stereo_outscores_disagreeing_stereo():
    coherent = compute_summary(_stereo_coherent(), SR).summary
    disagreeing = compute_summary(_stereo_disagreeing(), SR).summary
    assert coherent.tempo_confidence == pytest.approx(0.8747715147681395, abs=1e-6)
    assert disagreeing.tempo_confidence == pytest.approx(0.0, abs=1e-6)
    assert coherent.tempo_confidence > disagreeing.tempo_confidence


# --- RED: a lag WINDOW makes an off-grid mean BPM look confident ----------
# 90 BPM left / 160 BPM right: the channels' own lags are 31 and 35, and the
# reported mean BPM's lag is the fractional 33. A +/-2 lag-bin window spans
# both channels' true lags and scores 0.84 for a BPM neither channel has.
# Interpolating at the exact fractional lag reports the real periodicity there.


def _stereo_disagreeing_90_160() -> np.ndarray:
    left = _drum_loop(bpm=90.0, seed=1)
    right = _drum_loop(bpm=160.0, seed=2)
    n = min(len(left), len(right))
    return np.stack([left[:n], right[:n]])


def test_off_grid_mean_bpm_is_not_confident():
    result = compute_summary(_stereo_disagreeing_90_160(), SR)
    assert result.summary.tempo_confidence == pytest.approx(0.039902305958740385, abs=1e-6)
