"""Integrity-flag tests (Task D2, design §4.2 M1/I5).

Green-mirage discipline: every integrity check ships a RED fixture that provably
catches a real fault, paired with a GREEN clean case, asserted with exact
equality (exact booleans / exact counts) so a constant stub cannot pass.

- ``test_nan_detected`` / ``test_clean_no_nan`` — an injected NaN flips
  ``has_nan``; clean audio leaves it False.
- ``test_inf_detected`` / ``test_clean_no_inf`` — an injected +/-Inf flips
  ``has_inf``.
- ``test_denormal_detected`` / ``test_clean_no_denormal`` — a subnormal-float32
  magnitude (``0 < |x| < 1.1754944e-38``) flips ``has_denormal``.
- ``test_clipping_detected`` / ``test_no_clipping`` — a crafted fixture yields an
  EXACT ``clip_count`` (samples with ``|x| >= 1.0``) and matching
  ``clip_fraction``; the clean version reports zero.
- ``test_clip_count_sums_across_channels`` — proves counts SUM across channels
  (not per-channel), per the I5 combine rule.
- ``test_silence_detected`` / ``test_not_silent`` — all-zeros (RMS below
  -80 dBFS) sets ``is_silent``; a tone does not.
- ``test_dc_offset_exceeds`` / ``test_dc_offset_within_threshold`` — a biased
  signal (``|dc| > 0.001``) flips ``dc_offset_exceeds``.

Fixtures are small synthetic numpy signals — no external files, no randomness.
"""

import numpy as np
import pytest

# Import both modules by path (not via ``features/__init__``) so the D1/D2 dBFS
# SSOT lock (MINOR-2) compares the exact symbols each module owns.
from sonoscope.features import integrity, librosa_features
from sonoscope.features.integrity import (
    DC_OFFSET_THRESHOLD,
    DENORMAL_MIN_NORMAL_FLOAT32,
    SILENCE_THRESHOLD_DBFS,
    compute_integrity,
)


def _tone(freq: float = 440.0, dur: float = 0.1, amp: float = 0.5) -> np.ndarray:
    """Single-channel pure sine tone (shape ``(n_samples,)`` float32)."""
    sr = 48000
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# --- NaN ---------------------------------------------------------------------


def test_nan_detected():
    sig = _tone()
    sig[100] = np.float32(np.nan)
    assert compute_integrity(sig).has_nan is True


def test_clean_no_nan():
    assert compute_integrity(_tone()).has_nan is False


# --- Inf ---------------------------------------------------------------------


def test_inf_detected():
    sig = _tone()
    sig[50] = np.float32(np.inf)
    result = compute_integrity(sig)
    assert result.has_inf is True


def test_clean_no_inf():
    assert compute_integrity(_tone()).has_inf is False


# --- Denormal (subnormal float32) --------------------------------------------


def test_denormal_detected():
    sig = _tone()
    # A magnitude strictly inside (0, smallest-normal-float32): subnormal.
    sig[10] = np.float32(1e-40)
    result = compute_integrity(sig)
    assert result.has_denormal is True


def test_clean_no_denormal():
    # A tone at amplitude 0.5 has no values in the subnormal band.
    assert compute_integrity(_tone()).has_denormal is False


def test_denormal_boundary_smallest_normal_is_not_denormal():
    # The smallest NORMAL float32 is the exclusive upper bound; it is not a
    # denormal (guards the ``< threshold`` boundary, not ``<=``).
    sig = np.zeros(64, dtype=np.float32)
    sig[0] = np.float32(DENORMAL_MIN_NORMAL_FLOAT32)
    assert compute_integrity(sig).has_denormal is False


# --- Clipping ----------------------------------------------------------------

# Crafted mono fixture: samples with |x| >= 1.0 are the clips.
# 0.5(no) 1.0(yes) -1.5(yes) 0.9(no) 2.0(yes) -1.0(yes) 0.0(no) 0.99(no)
# 1.0(yes) 0.3(no)  -> exactly 5 clipped samples out of 10.
_CLIP_FIXTURE = np.array(
    [0.5, 1.0, -1.5, 0.9, 2.0, -1.0, 0.0, 0.99, 1.0, 0.3], dtype=np.float32
)
_CLIP_EXPECTED_COUNT = 5


def test_clipping_detected():
    result = compute_integrity(_CLIP_FIXTURE)
    assert result.clip_count == _CLIP_EXPECTED_COUNT
    assert result.clip_fraction == _CLIP_EXPECTED_COUNT / _CLIP_FIXTURE.size


def test_no_clipping():
    # Same fixture with every clipped value pulled inside full scale.
    clean = np.array(
        [0.5, 0.9, -0.5, 0.9, 0.2, -0.9, 0.0, 0.99, 0.9, 0.3], dtype=np.float32
    )
    result = compute_integrity(clean)
    assert result.clip_count == 0
    assert result.clip_fraction == 0.0


def test_clip_count_sums_across_channels():
    # Two identical channels -> counts SUM (I5), not per-channel or OR.
    stereo = np.stack([_CLIP_FIXTURE, _CLIP_FIXTURE])  # (2, 10)
    result = compute_integrity(stereo)
    assert result.clip_count == 2 * _CLIP_EXPECTED_COUNT
    assert result.clip_fraction == (2 * _CLIP_EXPECTED_COUNT) / stereo.size


# --- Silence -----------------------------------------------------------------


def test_silence_detected():
    silence = np.zeros(4800, dtype=np.float32)  # RMS below -80 dBFS
    assert compute_integrity(silence).is_silent is True


def test_not_silent():
    assert compute_integrity(_tone()).is_silent is False


# --- DC offset ---------------------------------------------------------------


def test_dc_offset_exceeds():
    # A tone biased by 0.01 (> 0.001 threshold) -> dc_offset_exceeds True.
    biased = _tone() + np.float32(0.01)
    assert compute_integrity(biased).dc_offset_exceeds is True


def test_dc_offset_within_threshold():
    # A zero-mean tone (bias 0.0) stays under the 0.001 threshold.
    assert compute_integrity(_tone()).dc_offset_exceeds is False


# --- Reported thresholds match the frozen contract ---------------------------


def test_reported_thresholds_are_frozen_values():
    result = compute_integrity(_tone())
    assert result.silence_threshold_dbfs == SILENCE_THRESHOLD_DBFS
    assert result.silence_threshold_dbfs == -80.0
    assert result.dc_offset_threshold == DC_OFFSET_THRESHOLD
    assert result.dc_offset_threshold == 0.001


# --- Empty-input guard (MINOR-1) ---------------------------------------------


def test_zero_channel_input_rejected():
    # A ``(0, 100)`` buffer (zero channels) has ``total_samples == 0``; without
    # the widened both-axes guard it slips past the samples-only check and hits
    # ``clip_count / total_samples`` -> raw ZeroDivisionError. Assert the clean
    # ValueError instead (RED-proving: reverting the guard raises ZeroDivision).
    with pytest.raises(ValueError):
        compute_integrity(np.zeros((0, 100), dtype=np.float32))


# --- dBFS SSOT lock: D2 must not desync from D1 (MINOR-2) ---------------------


def test_dbfs_floor_matches_d1():
    # ``_AMPLITUDE_FLOOR`` and the ``_dbfs`` formula are duplicated in D2 from
    # D1 and are NOT covered by ``params_sha256``; a future D1 edit could
    # silently desync D2. Lock both the floor constant and the conversion
    # formula so any drift trips this test (RED if either diverges).
    assert integrity._AMPLITUDE_FLOOR == librosa_features._AMPLITUDE_FLOOR
    # Lock the formula, not just the floor: both helpers must map a shared
    # sample value to the same dBFS.
    assert integrity._dbfs(0.5) == pytest.approx(
        librosa_features._dbfs(0.5), abs=1e-12
    )


# --- Interaction hazards + boundaries (MINOR-3, green-mirage RED proofs) ------


def test_nan_not_counted_as_denormal_or_clip():
    # A single NaN must flip ``has_nan`` only: comparisons against NaN are
    # False, so it is neither a denormal (``0 < |x| < min-normal``) nor a clip
    # (``|x| >= 1.0``). RED if the NaN guard/ordering regressed.
    sig = _tone()
    sig[100] = np.float32(np.nan)
    result = compute_integrity(sig)
    assert result.has_nan is True
    assert result.has_denormal is False
    assert result.clip_count == 0


def test_inf_is_clip_not_denormal_not_silent():
    # A single +Inf satisfies ``|x| >= 1.0`` (counts as one clip) but is finite-
    # excluded from the denormal band; its non-finite RMS must not read as
    # silence. RED if Inf were miscounted as denormal or skipped by the clip
    # test, or if the non-finite RMS were treated as silent.
    sig = _tone()
    sig[50] = np.float32(np.inf)
    result = compute_integrity(sig)
    assert result.has_inf is True
    assert result.has_denormal is False
    assert result.clip_count == 1
    assert result.is_silent is False


def test_isfinite_guard_nonfinite_channel_not_silent():
    # Contract lock: a buffer whose first channel is all-NaN (non-finite RMS)
    # and whose second channel is a normal tone must NOT read as silent.
    # HONESTY NOTE: this is a behavioral-contract lock, not a strict
    # guard-removal RED test. The ``np.isfinite(rms_lin)`` guard is behaviorally
    # redundant for the boolean result here: ``_dbfs(nan) -> nan`` and
    # ``_dbfs(inf) -> inf``, and both ``nan <= -80`` and ``inf <= -80`` are
    # already False, so the OR-combined ``is_silent`` stays False with or
    # without the guard. The guard remains as defensive intent; this test locks
    # the observable contract (non-finite channel is not silence).
    tone = _tone()
    nan_ch = np.full(tone.size, np.float32(np.nan), dtype=np.float32)
    stereo = np.stack([nan_ch, tone])  # (2, n)
    assert compute_integrity(stereo).is_silent is False


def test_dc_offset_exactly_at_threshold_not_exceeded():
    # Boundary proof for the strict ``>`` DC guard. NOTE: ``float32(0.001)``
    # promotes to 0.0010000000474974513 in float64 (rounds UP, > 0.001), so a
    # constant channel of exactly ``float32(0.001)`` would TRIP the guard. To
    # assert the true boundary without changing the production strict-``>``
    # semantics, use the largest float32 value that is provably <= 0.001
    # (``nextafter(0.001, 0)`` = 0.0009999999310821295). Its mean equals itself,
    # so ``|dc| <= 0.001`` holds and ``dc_offset_exceeds`` must be False.
    below = np.nextafter(np.float32(DC_OFFSET_THRESHOLD), np.float32(0.0))
    channel = np.full(64, below, dtype=np.float32)
    assert abs(float(np.mean(channel, dtype=np.float64))) <= DC_OFFSET_THRESHOLD
    assert compute_integrity(channel).dc_offset_exceeds is False


def test_is_silent_or_combines_across_channels():
    # One dead (all-zero) channel + one loud tone -> ``is_silent`` is True via
    # the I5 OR-combine. This is the intended dead-channel semantics that D3
    # consumes: any silent channel flags the block silent.
    tone = _tone(amp=0.5)
    dead = np.zeros(tone.size, dtype=np.float32)
    stereo = np.stack([dead, tone])  # (2, n)
    assert compute_integrity(stereo).is_silent is True


# --- Whole-file silence (AND-combined, distinct from the OR tripwire) --------
#
# ``is_silent`` OR-combines: it fires when ANY channel is dead, which is the
# right dead-channel tripwire but the wrong descriptor gate. ``all_channels_silent``
# AND-combines over the SAME ``SILENCE_THRESHOLD_DBFS``, so it means "the whole
# file is silent". The stereo case below is the one that separates them.


def test_all_channels_silent_mono_zeros():
    silence = np.zeros(4800, dtype=np.float32)
    result = compute_integrity(silence)
    assert result.is_silent is True
    assert result.all_channels_silent is True


def test_all_channels_silent_false_for_a_tone():
    result = compute_integrity(_tone())
    assert result.is_silent is False
    assert result.all_channels_silent is False


def test_all_channels_silent_and_combines_one_dead_channel():
    # THE distinguishing case: one dead channel + one full-scale tone. The OR
    # tripwire fires (a channel is broken); the whole-file predicate must NOT,
    # or a stereo file with one dead channel would lose every descriptor.
    tone = _tone(amp=1.0)
    dead = np.zeros(tone.size, dtype=np.float32)
    stereo = np.stack([dead, tone])  # (2, n)
    result = compute_integrity(stereo)
    assert result.is_silent is True
    assert result.all_channels_silent is False


def test_all_channels_silent_stereo_both_dead():
    dead = np.zeros(4800, dtype=np.float32)
    stereo = np.stack([dead, dead])  # (2, n)
    result = compute_integrity(stereo)
    assert result.is_silent is True
    assert result.all_channels_silent is True


def test_all_channels_silent_false_for_nan_buffer():
    # A NaN-filled buffer has a non-finite RMS. That is a fault, not silence:
    # the AND-combine must not report the file wholly silent.
    nan_buf = np.full(4800, np.float32(np.nan), dtype=np.float32)
    result = compute_integrity(nan_buf)
    assert result.has_nan is True
    assert result.all_channels_silent is False


def test_all_channels_silent_false_when_one_channel_is_nan_and_one_is_dead():
    # NaN channel + dead channel: the OR tripwire fires on the dead channel, but
    # the non-finite channel is not silence, so the whole-file predicate is False.
    dead = np.zeros(4800, dtype=np.float32)
    nan_ch = np.full(4800, np.float32(np.nan), dtype=np.float32)
    result = compute_integrity(np.stack([nan_ch, dead]))
    assert result.is_silent is True
    assert result.all_channels_silent is False


def test_all_channels_silent_reuses_the_frozen_silence_threshold():
    # Both predicates read the SAME frozen cutoff; a level just above it is not
    # silence under either. -79 dBFS constant channel: |x| = 10**(-79/20).
    amp = np.float32(10.0 ** (-79.0 / 20.0))
    channel = np.full(4800, amp, dtype=np.float32)
    result = compute_integrity(channel)
    assert result.silence_threshold_dbfs == SILENCE_THRESHOLD_DBFS
    assert result.is_silent is False
    assert result.all_channels_silent is False
