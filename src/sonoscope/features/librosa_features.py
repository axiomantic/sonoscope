"""Deterministic Librosa summary features (Task D1, by design).

Pure function ``compute_summary(audio, sample_rate) -> SummaryResult`` producing
the ``deterministic.summary`` scalars + MFCCs of the analysis contract (schema
``DeterministicSummary``), plus the tempo-suppression ``notes`` that belong on
the enclosing ``deterministic`` block.

Design invariants honored here:

- **Frozen, hashed param set (I4).** ``FROZEN_PARAMS`` is hashed into
  ``params_sha256`` so any param change is detectable in the report. ``sr`` is a
  *target* only — corpus audio is already 48 kHz and is **not resampled**.
- **Multi-channel reduction (I5).** Audio is analysed per channel (never a
  silent mono downmix). Spectral / scalar summaries reduce by **mean** across
  channels; ``peak_dbfs`` reduces by **max** (a true peak, never an average).
- **Octave-error mitigation.** ``tempo_bpm`` is emitted only with enough onsets
  AND a plausible BPM; otherwise it is ``None`` with a machine-readable note.
  ``tempo_confidence`` is emitted only alongside a ``tempo_bpm``, and is a
  measured 0-1 periodicity strength — the normalised autocorrelation of the
  onset-strength envelope at the lag implied by ``tempo_bpm`` — not a gate-pass
  marker. Sustained and noise material therefore scores near 0, exact-periodic
  material near 1, and confidence falls monotonically with timing jitter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import librosa
import numpy as np

from sonoscope.schema.models import DeterministicSummary

# --- Frozen parameter set (by design, I4) ------------------------------------
# Hashed into params_sha256; any change here changes the report's hash.
FROZEN_PARAMS: dict[str, int | float | str | bool] = {
    "sr": 48000,  # target rate; corpus is already 48 kHz — NO resample
    "n_fft": 2048,
    "hop_length": 512,
    "n_mfcc": 13,
    "window": "hann",
    "center": True,
    "roll_percent": 0.85,
    "silence_threshold_dbfs": -80.0,
    # Absolute peak-picking threshold on the UN-normalised onset envelope.
    # ``onset_strength`` uses ``power_to_db(ref=np.max)``, so the raw envelope is
    # amplitude-invariant and an absolute delta discriminates signal STRUCTURE,
    # not level. Normalising first (the librosa default) rescales a sustained
    # tone's noise floor to [0,1] and makes any delta fire on it.
    "onset_delta": 10.0,
    "onset_normalize": False,
}

# --- Octave-error-mitigation gates + machine-readable notes -----------------
# Minimum onsets for a trustworthy tempo estimate. The note string is exact and
# stable (consumed as a machine-readable marker; asserted verbatim in tests and
# mirrored by the C1 schema reference fixture).
TEMPO_MIN_ONSETS: int = 4
TEMPO_PLAUSIBLE_BPM: tuple[float, float] = (40.0, 300.0)
NOTE_TEMPO_LOW_ONSETS: str = (
    "tempo_bpm suppressed: onset_count < 4 "
    "(not enough onsets for reliable estimate)"
)
NOTE_TEMPO_IMPLAUSIBLE: str = (
    "tempo_bpm suppressed: estimate outside plausible range [40.0, 300.0] BPM"
)

# Amplitude floor for dBFS conversion, so silence yields a finite (very negative)
# value instead of ``-inf`` (which is not JSON-representable). Deterministic.
_AMPLITUDE_FLOOR: float = 1e-12


@dataclass(frozen=True)
class SummaryResult:
    """D1 output: the summary block plus tempo-suppression notes.

    ``notes`` are appended to the enclosing ``deterministic.notes`` list by the
    caller assembling the full ``DeterministicBlock`` (D2/D3 territory).
    """

    summary: DeterministicSummary
    notes: list[str]


def params_sha256(params: Mapping[str, object] = FROZEN_PARAMS) -> str:
    """SHA-256 of the frozen param set (I4).

    Serialised canonically (sorted keys, compact separators) so the digest is
    stable across runs and changes iff a param value changes.
    """
    payload = json.dumps(
        dict(params), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_channels(audio: np.ndarray) -> np.ndarray:
    """Normalise input to shape ``(channels, n_samples)`` float32 (librosa
    per-channel convention). 1-D input is treated as single-channel."""
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


def _tempo_confidence(
    onset_env: np.ndarray, bpm: float, sample_rate: int, hop: int
) -> float:
    """Periodicity strength at ``bpm``, in [0, 1] (by design).

    Normalised autocorrelation of the mean-removed onset-strength envelope,
    linearly interpolated at the exact fractional lag implied by ``bpm``.
    Returns 0.0 when no periodicity is measurable.
    """
    if bpm is None or bpm <= 0.0:
        return 0.0
    env = onset_env - onset_env.mean()
    if not np.any(env):
        return 0.0
    ac = librosa.autocorrelate(env, max_size=len(env))
    if ac[0] <= 0.0:
        return 0.0
    ac = ac / ac[0]
    lag = (60.0 / bpm) * sample_rate / hop
    if lag < 1.0 or lag >= len(ac) - 1:
        return 0.0
    i = int(np.floor(lag))
    frac = lag - i
    return float(np.clip(ac[i] * (1.0 - frac) + ac[i + 1] * frac, 0.0, 1.0))


def compute_summary(audio: np.ndarray, sample_rate: int) -> SummaryResult:
    """Compute ``deterministic.summary`` from a wav array (by design).

    ``audio`` is ``(channels, n_samples)`` (or 1-D mono). ``sample_rate`` is the
    native wav rate (used as-is; no resample). Returns a ``SummaryResult`` whose
    ``summary`` populates the C1 ``DeterministicSummary`` contract exactly.
    """
    # Reproducibility contract (I4): features are computed at the caller's
    # ``sample_rate`` but ``params_sha256`` reports the FROZEN sr digest. A
    # non-frozen rate would silently drift features from the claimed hash, so
    # hard-fail rather than emit a report that contradicts its own params hash.
    frozen_sr = FROZEN_PARAMS["sr"]
    if sample_rate != frozen_sr:
        raise ValueError(
            "sample_rate must equal the frozen params sr "
            f"({frozen_sr} Hz); got {sample_rate} Hz. Corpus audio is already "
            "48 kHz and is not resampled — features computed at a non-frozen "
            "rate would contradict the reported params_sha256."
        )

    channels = _to_channels(audio)
    n_channels, n_samples = channels.shape
    # Finding 2 (Gemini review, final batch): guard BOTH dimensions. Zero-channel
    # input (shape ``(0, N)``) has ``n_samples != 0`` but ``n_channels == 0``, so a
    # samples-only guard is bypassed and ``max(peak_lin_ch)`` later raises the
    # opaque ``max() arg is an empty sequence``. Fail cleanly up front instead.
    if n_channels == 0 or n_samples == 0:
        raise ValueError("audio must have at least one channel and one sample")

    n_fft = FROZEN_PARAMS["n_fft"]
    hop = FROZEN_PARAMS["hop_length"]
    n_mfcc = FROZEN_PARAMS["n_mfcc"]
    window = FROZEN_PARAMS["window"]
    center = FROZEN_PARAMS["center"]
    roll_percent = FROZEN_PARAMS["roll_percent"]
    onset_delta = FROZEN_PARAMS["onset_delta"]
    onset_normalize = FROZEN_PARAMS["onset_normalize"]

    duration_s = n_samples / float(sample_rate)

    # Per-channel accumulators (I5: reduce spectral/scalar by mean).
    rms_dbfs_ch: list[float] = []
    crest_db_ch: list[float] = []
    peak_lin_ch: list[float] = []
    centroid_ch: list[float] = []
    bandwidth_ch: list[float] = []
    rolloff_ch: list[float] = []
    flatness_ch: list[float] = []
    zcr_ch: list[float] = []
    mfcc_mean_ch: list[np.ndarray] = []
    mfcc_std_ch: list[np.ndarray] = []
    onset_counts: list[int] = []
    tempo_ch: list[float] = []
    onset_env_ch: list[np.ndarray] = []

    for c in range(n_channels):
        y = np.ascontiguousarray(channels[c], dtype=np.float32)

        # Level features.
        rms_lin = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        peak_lin = float(np.max(np.abs(y)))
        rms_db = _dbfs(rms_lin)
        peak_db = _dbfs(peak_lin)
        rms_dbfs_ch.append(rms_db)
        peak_lin_ch.append(peak_lin)
        crest_db_ch.append(peak_db - rms_db)

        # Spectral / timbral features (per-frame -> mean over frames).
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop, window=window,
            center=center,
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop, window=window,
            center=center,
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop, window=window,
            center=center, roll_percent=roll_percent,
        )
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop, window=window, center=center,
        )
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=n_fft, hop_length=hop, center=center,
        )
        mfcc = librosa.feature.mfcc(
            y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
            window=window, center=center,
        )

        centroid_ch.append(float(np.nan_to_num(np.mean(centroid))))
        bandwidth_ch.append(float(np.nan_to_num(np.mean(bandwidth))))
        rolloff_ch.append(float(np.nan_to_num(np.mean(rolloff))))
        flatness_ch.append(float(np.nan_to_num(np.mean(flatness))))
        zcr_ch.append(float(np.nan_to_num(np.mean(zcr))))
        mfcc_mean_ch.append(np.nan_to_num(np.mean(mfcc, axis=1)))
        mfcc_std_ch.append(np.nan_to_num(np.std(mfcc, axis=1)))

        # Rhythm features. The onset envelope is computed once and reused for
        # both detection and the tempo confidence (verified bit-identical to
        # letting onset_detect derive it from ``y``).
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sample_rate, hop_length=hop,
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sample_rate, hop_length=hop,
            units="frames", delta=onset_delta, normalize=onset_normalize,
        )
        onset_counts.append(int(len(onsets)))
        tempo = librosa.beat.beat_track(
            y=y, sr=sample_rate, hop_length=hop,
        )[0]
        bpm = float(np.atleast_1d(tempo)[0])
        tempo_ch.append(bpm)
        onset_env_ch.append(onset_env)

    # --- Reductions across channels (I5) ------------------------------------
    peak_dbfs = _dbfs(max(peak_lin_ch))  # max across channels (true peak)
    onset_count = int(round(float(np.mean(onset_counts))))
    onset_rate_hz = onset_count / duration_s if duration_s > 0.0 else 0.0
    tempo_candidate = float(np.mean(tempo_ch))

    # --- Octave-error mitigation (never a confident wrong BPM) --------------
    notes: list[str] = []
    tempo_bpm: float | None
    tempo_confidence: float | None
    lo, hi = TEMPO_PLAUSIBLE_BPM
    if onset_count < TEMPO_MIN_ONSETS:
        tempo_bpm = None
        tempo_confidence = None
        notes.append(NOTE_TEMPO_LOW_ONSETS)
    elif not (lo <= tempo_candidate <= hi):
        tempo_bpm = None
        tempo_confidence = None
        notes.append(NOTE_TEMPO_IMPLAUSIBLE)
    else:
        tempo_bpm = tempo_candidate
        # Passed both mitigation gates: measure the periodicity strength at the
        # REPORTED BPM -- not at each channel's own beat_track BPM, which for
        # disagreeing channels is a tempo the reported value never equals --
        # then reduce by mean across channels (I5). Suppression yields None.
        tempo_confidence = float(np.mean([
            _tempo_confidence(env, tempo_candidate, sample_rate, hop)
            for env in onset_env_ch
        ]))

    summary = DeterministicSummary(
        duration_s=float(duration_s),
        sample_rate_hz=int(sample_rate),
        channels=int(n_channels),
        rms_dbfs=float(np.mean(rms_dbfs_ch)),
        peak_dbfs=float(peak_dbfs),
        crest_factor_db=float(np.mean(crest_db_ch)),
        dc_offset=float(np.mean(channels, dtype=np.float64)),
        spectral_centroid_hz=float(np.mean(centroid_ch)),
        spectral_bandwidth_hz=float(np.mean(bandwidth_ch)),
        spectral_rolloff_hz=float(np.mean(rolloff_ch)),
        spectral_flatness=float(np.mean(flatness_ch)),
        zero_crossing_rate=float(np.mean(zcr_ch)),
        onset_count=int(onset_count),
        onset_rate_hz=float(onset_rate_hz),
        tempo_bpm=tempo_bpm,
        tempo_confidence=tempo_confidence,
        mfcc_mean=[float(x) for x in np.mean(mfcc_mean_ch, axis=0)],
        mfcc_std=[float(x) for x in np.mean(mfcc_std_ch, axis=0)],
    )
    return SummaryResult(summary=summary, notes=notes)
