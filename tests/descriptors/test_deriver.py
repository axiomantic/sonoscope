"""Boundary RED/GREEN + gate + determinism tests for ``derive_descriptors``.

Every assertion is exact-equality only (design §11.1 green-mirage RED-must-trip).
RED cases assert the term is NOT present via exact-equality on an empty filtered
list; GREEN cases assert the emitted record equals a fully-built expected record.
Hybrid composites are recomputed in-test from the same ``norm`` + thresholds so
the expected ``anchor_value`` is bit-exact.

Boundary-operator asymmetry (design §5.3) is intentional: measured gated terms
fire strict ``>``/``<`` (at-threshold = no fire); hybrid feel-terms fire
inclusive ``>=`` (at-threshold = fire).
"""

from __future__ import annotations

from typing import Any

from sonoscope.descriptors.deriver import derive_descriptors, norm
from sonoscope.descriptors.thresholds import (
    DERIVER_THRESHOLDS as T,
)
from sonoscope.descriptors.thresholds import (
    DERIVER_VERSION,
    thresholds_sha256,
)
from sonoscope.schema.models import (
    DescriptorsLibrary,
    HybridDescriptor,
    MeasuredDescriptor,
)


def _summary(**overrides: Any):
    """Neutral baseline DeterministicSummary with keyword overrides.

    All metrics sit in their non-firing bands (no gated measured term fires,
    no hybrid composite reaches its fire threshold) except ``onset_rate_hz``
    which defaults to 4.0 (a mid rhythmic-density readout, always filtered out
    in single-term assertions).
    """
    from sonoscope.schema.models import DeterministicSummary

    base: dict[str, Any] = {
        "duration_s": 8.0,
        "sample_rate_hz": 48000,
        "channels": 2,
        "rms_dbfs": -25.0,
        "peak_dbfs": -10.0,
        "crest_factor_db": 10.0,
        "dc_offset": 0.0,
        "spectral_centroid_hz": 2000.0,
        "spectral_bandwidth_hz": 1000.0,
        "spectral_rolloff_hz": 4000.0,
        "spectral_flatness": 0.1,
        "zero_crossing_rate": 0.1,
        "onset_count": 0,
        "onset_rate_hz": 4.0,
        "tempo_bpm": None,
        "tempo_confidence": None,
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [0.0] * 13,
    }
    base.update(overrides)
    return DeterministicSummary(**base)


def _measured(block, term: str) -> list[MeasuredDescriptor]:
    return [m for m in block.measured if m.term == term]


def _hybrid(block, term: str) -> list[HybridDescriptor]:
    return [h for h in block.hybrid if h.term == term]


# mfcc coefficients 1..5 form a linear ramp centred on index 3, so their mean
# over the inclusive warm slice [mfcc_lo_idx, mfcc_hi_idx] equals the requested
# value exactly while every coefficient differs. Index 5 (the inclusive upper
# element) and the excluded index 0 therefore both differ from indices 1..4, so a
# slice off-by-one in the deriver (dropping index 5, or shifting the low index to
# 0) moves the computed mean and breaks the GREEN record-equality assertion.
# A uniform fill would leave the mean invariant under such a mutation (green mirage).
_MFCC_RAMP = 20.0
_MFCC_HEAD = 111.0  # excluded index-0 coefficient, distinct from the 1..5 band


def _mfcc(mean1to5: float) -> list[float]:
    """13-length mfcc_mean whose indices 1..5 mean == ``mean1to5`` (non-uniform).

    Indices 1..5 are a linear ramp centred on index 3 (``mean1to5 ± 2*ramp``): the
    ramp cancels over the inclusive slice so the mean is exactly ``mean1to5``, yet
    index 5 differs from indices 1..4 and index 0 (excluded) is distinct. Any
    off-by-one in the slice bounds therefore changes the computed mean.
    """
    m = [0.0] * 13
    m[0] = _MFCC_HEAD
    for i in range(1, 6):
        m[i] = mean1to5 + _MFCC_RAMP * (i - 3)
    return m


# ---------------------------------------------------------------------------
# In-test composite recomputation (mirrors deriver operation order exactly)
# ---------------------------------------------------------------------------


def _driving_score(s) -> float:
    no = norm(s.onset_rate_hz, T["driving.onset_lo"], T["driving.onset_hi"])
    nt = (
        norm(s.tempo_bpm, T["driving.tempo_lo"], T["driving.tempo_hi"])
        if s.tempo_bpm is not None
        else 0.0
    )
    nr = norm(s.rms_dbfs, T["driving.rms_lo"], T["driving.rms_hi"])
    return T["driving.w_onset"] * no + T["driving.w_tempo"] * nt + T["driving.w_rms"] * nr


def _punchy_score(s) -> float:
    nc = norm(s.crest_factor_db, T["punchy.crest_lo"], T["punchy.crest_hi"])
    no = norm(s.onset_rate_hz, T["punchy.onset_lo"], T["punchy.onset_hi"])
    return T["punchy.w_crest"] * nc + T["punchy.w_onset"] * no


def _warm_score(s) -> float:
    seg = s.mfcc_mean[T["warm.mfcc_lo_idx"] : T["warm.mfcc_hi_idx"] + 1]
    mlm = sum(seg) / len(seg)
    ncent = norm(s.spectral_centroid_hz, T["warm.centroid_lo"], T["warm.centroid_hi"])
    nmf = norm(mlm, T["warm.mfcc_lo"], T["warm.mfcc_hi"])
    return T["warm.w_centroid"] * (1 - ncent) + T["warm.w_mfcc"] * nmf


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


def test_norm_clamps_and_scales() -> None:
    assert norm(6.0, 0.0, 12.0) == 0.5
    assert norm(-5.0, 0.0, 12.0) == 0.0
    assert norm(99.0, 0.0, 12.0) == 1.0


# ---------------------------------------------------------------------------
# Gated single-metric terms — RED at/just-under, GREEN just-over
# ---------------------------------------------------------------------------


def test_bright_boundary() -> None:
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=2500.0)), "bright") == []
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=2499.0)), "bright") == []
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=2501.0)), "bright") == [
        MeasuredDescriptor(
            term="bright",
            value=2501.0,
            metric="spectral_centroid_hz",
            direction="high",
            threshold=2500.0,
        )
    ]


def test_dark_boundary() -> None:
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=800.0)), "dark") == []
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=801.0)), "dark") == []
    assert _measured(derive_descriptors(_summary(spectral_centroid_hz=799.0)), "dark") == [
        MeasuredDescriptor(
            term="dark",
            value=799.0,
            metric="spectral_centroid_hz",
            direction="low",
            threshold=800.0,
        )
    ]


def test_loud_boundary() -> None:
    assert _measured(derive_descriptors(_summary(rms_dbfs=-18.0)), "loud") == []
    assert _measured(derive_descriptors(_summary(rms_dbfs=-17.0)), "loud") == [
        MeasuredDescriptor(
            term="loud", value=-17.0, metric="rms_dbfs", direction="high", threshold=-18.0
        )
    ]


def test_quiet_boundary() -> None:
    assert _measured(derive_descriptors(_summary(rms_dbfs=-35.0)), "quiet") == []
    assert _measured(derive_descriptors(_summary(rms_dbfs=-36.0)), "quiet") == [
        MeasuredDescriptor(
            term="quiet", value=-36.0, metric="rms_dbfs", direction="low", threshold=-35.0
        )
    ]


def test_compressed_boundary() -> None:
    assert _measured(derive_descriptors(_summary(crest_factor_db=6.0)), "compressed") == []
    assert _measured(derive_descriptors(_summary(crest_factor_db=5.0)), "compressed") == [
        MeasuredDescriptor(
            term="compressed",
            value=5.0,
            metric="crest_factor_db",
            direction="low",
            threshold=6.0,
        )
    ]


def test_dynamic_boundary() -> None:
    assert _measured(derive_descriptors(_summary(crest_factor_db=15.0)), "dynamic") == []
    assert _measured(derive_descriptors(_summary(crest_factor_db=16.0)), "dynamic") == [
        MeasuredDescriptor(
            term="dynamic",
            value=16.0,
            metric="crest_factor_db",
            direction="high",
            threshold=15.0,
        )
    ]


def test_busy_boundary() -> None:
    assert _measured(derive_descriptors(_summary(onset_rate_hz=8.0)), "busy") == []
    assert _measured(derive_descriptors(_summary(onset_rate_hz=9.0)), "busy") == [
        MeasuredDescriptor(
            term="busy", value=9.0, metric="onset_rate_hz", direction="high", threshold=8.0
        )
    ]


def test_spare_boundary() -> None:
    assert _measured(derive_descriptors(_summary(onset_rate_hz=2.0)), "spare") == []
    assert _measured(derive_descriptors(_summary(onset_rate_hz=1.0)), "spare") == [
        MeasuredDescriptor(
            term="spare", value=1.0, metric="onset_rate_hz", direction="low", threshold=2.0
        )
    ]


# ---------------------------------------------------------------------------
# Gated multi-metric term — dense (AND). One RED per metric + one GREEN.
# ---------------------------------------------------------------------------


def test_dense_boundary() -> None:
    # onset at threshold (no fire), bandwidth over.
    assert (
        _measured(
            derive_descriptors(_summary(onset_rate_hz=8.0, spectral_bandwidth_hz=2001.0)),
            "dense",
        )
        == []
    )
    # onset over, bandwidth at threshold (no fire).
    assert (
        _measured(
            derive_descriptors(_summary(onset_rate_hz=8.5, spectral_bandwidth_hz=2000.0)),
            "dense",
        )
        == []
    )
    # both over -> fire.
    assert _measured(
        derive_descriptors(_summary(onset_rate_hz=8.5, spectral_bandwidth_hz=2001.0)),
        "dense",
    ) == [
        MeasuredDescriptor(
            term="dense",
            value=8.5,
            metric="onset_rate_hz+spectral_bandwidth_hz",
            direction="high",
            threshold=8.0,
        )
    ]


# ---------------------------------------------------------------------------
# Hybrid composites — inclusive >= fire (at-threshold fires)
# ---------------------------------------------------------------------------


def test_driving_boundary() -> None:
    red = _summary(onset_rate_hz=12.0, tempo_bpm=None, rms_dbfs=-25.0)
    assert _driving_score(red) < T["driving.fire"]
    assert _hybrid(derive_descriptors(red), "driving") == []

    at = _summary(onset_rate_hz=12.0, tempo_bpm=None, rms_dbfs=-23.0)
    score = _driving_score(at)
    assert score == T["driving.fire"]  # exactly at fire -> inclusive GREEN
    assert _hybrid(derive_descriptors(at), "driving") == [
        HybridDescriptor(
            term="driving",
            anchor_metric="driving_composite",
            anchor_value=score,
            direction="high",
            confidence=score,
        )
    ]

    over = _summary(onset_rate_hz=12.0, tempo_bpm=None, rms_dbfs=-22.0)
    score_over = _driving_score(over)
    assert score_over > T["driving.fire"]
    assert _hybrid(derive_descriptors(over), "driving") == [
        HybridDescriptor(
            term="driving",
            anchor_metric="driving_composite",
            anchor_value=score_over,
            direction="high",
            confidence=score_over,
        )
    ]


def test_punchy_boundary() -> None:
    red = _summary(crest_factor_db=17.0, onset_rate_hz=0.0)
    assert _punchy_score(red) < T["punchy.fire"]
    assert _hybrid(derive_descriptors(red), "punchy") == []

    at = _summary(crest_factor_db=18.0, onset_rate_hz=0.0)
    score = _punchy_score(at)
    assert score == T["punchy.fire"]
    assert _hybrid(derive_descriptors(at), "punchy") == [
        HybridDescriptor(
            term="punchy",
            anchor_metric="punchy_composite",
            anchor_value=score,
            direction="high",
            confidence=score,
        )
    ]

    over = _summary(crest_factor_db=18.0, onset_rate_hz=0.6)
    score_over = _punchy_score(over)
    assert score_over > T["punchy.fire"]
    assert _hybrid(derive_descriptors(over), "punchy") == [
        HybridDescriptor(
            term="punchy",
            anchor_metric="punchy_composite",
            anchor_value=score_over,
            direction="high",
            confidence=score_over,
        )
    ]


def test_warm_boundary() -> None:
    red = _summary(spectral_centroid_hz=600.0, mfcc_mean=_mfcc(-200.0))
    assert _warm_score(red) < T["warm.fire"]
    assert _hybrid(derive_descriptors(red), "warm") == []

    at = _summary(spectral_centroid_hz=500.0, mfcc_mean=_mfcc(-200.0))
    score = _warm_score(at)
    assert score == T["warm.fire"]
    assert _hybrid(derive_descriptors(at), "warm") == [
        HybridDescriptor(
            term="warm",
            anchor_metric="warm_composite",
            anchor_value=score,
            direction="high",
            confidence=score,
        )
    ]

    # Interior mfcc mean (0.0 -> nmf 0.5, not clamped) so an off-by-one slice moves
    # the mean and thus the composite: this is the assertion that catches a
    # mfcc_hi_idx/mfcc_lo_idx recalibration (green-mirage teeth).
    over = _summary(spectral_centroid_hz=500.0, mfcc_mean=_mfcc(0.0))
    score_over = _warm_score(over)
    assert score_over > T["warm.fire"]
    assert _hybrid(derive_descriptors(over), "warm") == [
        HybridDescriptor(
            term="warm",
            anchor_metric="warm_composite",
            anchor_value=score_over,
            direction="high",
            confidence=score_over,
        )
    ]


def test_warm_empty_mfcc_neutral_low_mid() -> None:
    # An empty mfcc_mean must NOT crash the deriver (ZeroDivisionError guard): the
    # missing low-mid contributes a NEUTRAL 0.0 to the mfcc term, while the centroid
    # term still fires warm. The emitted anchor_value equals the score computed with
    # mfcc_low_mid == 0.0, using the same norm + weights as the deriver.
    s = _summary(spectral_centroid_hz=500.0, mfcc_mean=[])
    mlm = 0.0
    ncent = norm(s.spectral_centroid_hz, T["warm.centroid_lo"], T["warm.centroid_hi"])
    nmf = norm(mlm, T["warm.mfcc_lo"], T["warm.mfcc_hi"])
    expected = T["warm.w_centroid"] * (1 - ncent) + T["warm.w_mfcc"] * nmf

    block = derive_descriptors(s)  # must not raise
    assert _hybrid(block, "warm") == [
        HybridDescriptor(
            term="warm",
            anchor_metric="warm_composite",
            anchor_value=expected,
            direction="high",
            confidence=expected,
        )
    ]


# ---------------------------------------------------------------------------
# tempo-audio gate boundaries (design §11.6)
# ---------------------------------------------------------------------------


def test_tempo_onset_count_gate() -> None:
    assert (
        _measured(
            derive_descriptors(_summary(onset_count=3, tempo_bpm=120.0, tempo_confidence=0.9)),
            "tempo-audio",
        )
        == []
    )
    assert _measured(
        derive_descriptors(_summary(onset_count=4, tempo_bpm=120.0, tempo_confidence=0.9)),
        "tempo-audio",
    ) == [
        MeasuredDescriptor(
            term="tempo-audio",
            value=120.0,
            metric="tempo_bpm",
            direction="value",
            threshold=None,
            estimated=True,
            confidence=0.9,
        )
    ]


def test_tempo_bpm_lower_bound() -> None:
    assert (
        _measured(
            derive_descriptors(_summary(onset_count=4, tempo_bpm=39.0, tempo_confidence=0.9)),
            "tempo-audio",
        )
        == []
    )
    assert _measured(
        derive_descriptors(_summary(onset_count=4, tempo_bpm=40.0, tempo_confidence=0.9)),
        "tempo-audio",
    ) == [
        MeasuredDescriptor(
            term="tempo-audio",
            value=40.0,
            metric="tempo_bpm",
            direction="value",
            threshold=None,
            estimated=True,
            confidence=0.9,
        )
    ]


def test_tempo_bpm_upper_bound() -> None:
    assert (
        _measured(
            derive_descriptors(_summary(onset_count=4, tempo_bpm=301.0, tempo_confidence=0.9)),
            "tempo-audio",
        )
        == []
    )
    assert _measured(
        derive_descriptors(_summary(onset_count=4, tempo_bpm=300.0, tempo_confidence=0.9)),
        "tempo-audio",
    ) == [
        MeasuredDescriptor(
            term="tempo-audio",
            value=300.0,
            metric="tempo_bpm",
            direction="value",
            threshold=None,
            estimated=True,
            confidence=0.9,
        )
    ]


def test_tempo_bpm_none_omitted() -> None:
    assert (
        _measured(
            derive_descriptors(_summary(onset_count=10, tempo_bpm=None)),
            "tempo-audio",
        )
        == []
    )


# ---------------------------------------------------------------------------
# Determinism + empty/degenerate + readout suppression + library stamp
# ---------------------------------------------------------------------------


def test_deriver_bit_identical() -> None:
    s = _summary(
        spectral_centroid_hz=3000.0,
        rms_dbfs=-10.0,
        crest_factor_db=20.0,
        onset_rate_hz=10.0,
        spectral_bandwidth_hz=2500.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        onset_count=8,
    )
    a = derive_descriptors(s)
    b = derive_descriptors(s)
    assert a.measured == b.measured
    assert a.hybrid == b.hybrid
    assert a.library == b.library


def test_empty_degenerate() -> None:
    # Fully degenerate summary: onset_rate_hz == 0.0 suppresses rhythmic-density,
    # tempo_bpm is None so tempo-audio is gate-omitted, spare requires positive
    # onset activity, and every other metric sits in its non-firing band.
    s = _summary(
        rms_dbfs=-25.0,
        spectral_centroid_hz=2000.0,
        crest_factor_db=10.0,
        onset_rate_hz=0.0,
        onset_count=0,
        tempo_bpm=None,
        spectral_bandwidth_hz=1000.0,
        mfcc_mean=[0.0] * 13,
    )
    block = derive_descriptors(s)
    assert block.measured == []
    assert block.hybrid == []
    assert block.advisory == []
    assert block.library.advisory_coverage is None
    assert block.library.advisory_dropped is None


def test_rhythmic_density_suppressed_boundary() -> None:
    assert (
        _measured(derive_descriptors(_summary(onset_rate_hz=0.0)), "rhythmic-density") == []
    )
    assert _measured(
        derive_descriptors(_summary(onset_rate_hz=0.5)), "rhythmic-density"
    ) == [
        MeasuredDescriptor(
            term="rhythmic-density",
            value=0.5,
            metric="onset_rate_hz",
            direction="value",
            threshold=None,
        )
    ]


def test_library_stamped() -> None:
    block = derive_descriptors(_summary())
    assert block.library == DescriptorsLibrary(
        thresholds_sha256=thresholds_sha256(),
        deriver_version=DERIVER_VERSION,
        advisory_coverage=None,
        advisory_dropped=None,
    )


def test_emission_order_measured_then_dense_then_readouts() -> None:
    # bright + dense + busy + rhythmic-density + tempo-audio all fire.
    s = _summary(
        spectral_centroid_hz=3000.0,
        onset_rate_hz=9.0,
        spectral_bandwidth_hz=2500.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        onset_count=8,
    )
    terms = [m.term for m in derive_descriptors(s).measured]
    assert terms == ["bright", "busy", "dense", "rhythmic-density", "tempo-audio"]


# --- Digital-silence gate (whole-file silence suppresses interpretation) ------
#
# The gate input is the integrity layer's whole-file silence predicate, passed in
# explicitly to keep the deriver pure and to keep ONE definition of silence. The
# RED case below is a digital-silence summary: before the gate existed it emitted
# the timbral adjectives "dark, quiet, compressed" plus the hybrid "warm"
# composed from the MFCCs of nothing.


def _digital_silence_summary():
    """Summary of an all-zeros buffer, as compute_summary actually reports it."""
    return _summary(
        rms_dbfs=-240.0,
        peak_dbfs=-240.0,
        crest_factor_db=0.0,
        dc_offset=0.0,
        spectral_centroid_hz=0.0,
        spectral_bandwidth_hz=0.0,
        spectral_rolloff_hz=0.0,
        spectral_flatness=0.0,
        zero_crossing_rate=0.0,
        onset_count=0,
        onset_rate_hz=0.0,
        tempo_bpm=None,
        tempo_confidence=None,
        mfcc_mean=[0.0] * 13,
        mfcc_std=[0.0] * 13,
    )


def test_silence_gate_emits_only_the_silent_term() -> None:
    # RED before the gate: measured == [dark, quiet, compressed] and
    # hybrid == [warm], summary "measured: dark, quiet, compressed, warm".
    block = derive_descriptors(_digital_silence_summary(), is_silent=True)
    assert block.measured == [
        MeasuredDescriptor(
            term="silent",
            value=-240.0,
            metric="rms_dbfs",
            direction="low",
            threshold=None,
            estimated=False,
            confidence=None,
        )
    ]
    assert block.hybrid == []
    assert block.advisory == []
    assert block.summary == "measured: silent"


def test_silence_gate_suppresses_hybrid_for_an_otherwise_firing_summary() -> None:
    # Proves the gate suppresses by the FLAG, not by the silent summary's values:
    # this summary fires driving + punchy + several measured terms on its own.
    s = _summary(
        onset_rate_hz=10.0,
        tempo_bpm=140.0,
        tempo_confidence=0.9,
        onset_count=20,
        rms_dbfs=-10.0,
        crest_factor_db=16.0,
        spectral_bandwidth_hz=2500.0,
    )
    assert [m.term for m in derive_descriptors(s).measured] == [
        "loud",
        "dynamic",
        "busy",
        "dense",
        "rhythmic-density",
        "tempo-audio",
    ]
    assert [h.term for h in derive_descriptors(s).hybrid] == ["driving", "punchy"]

    gated = derive_descriptors(s, is_silent=True)
    assert gated.measured == [
        MeasuredDescriptor(
            term="silent",
            value=-10.0,
            metric="rms_dbfs",
            direction="low",
            threshold=None,
            estimated=False,
            confidence=None,
        )
    ]
    assert gated.hybrid == []
    assert gated.summary == "measured: silent"


def test_silence_gate_does_not_fire_for_a_normal_signal() -> None:
    # GREEN: is_silent=False leaves a normal signal's descriptors untouched, and
    # the default (flag omitted) is identical to passing False.
    s = _summary(
        spectral_centroid_hz=3000.0,
        onset_rate_hz=9.0,
        spectral_bandwidth_hz=2500.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        onset_count=8,
    )
    assert derive_descriptors(s, is_silent=False) == derive_descriptors(s)
    block = derive_descriptors(s, is_silent=False)
    assert [m.term for m in block.measured] == [
        "bright",
        "busy",
        "dense",
        "rhythmic-density",
        "tempo-audio",
    ]
    assert [h.term for h in block.hybrid] == ["driving"]
    assert (
        block.summary
        == "measured: bright, busy, dense, driving, 9.0 onsets/s, 120 BPM"
    )


def test_silence_gate_still_stamps_the_library() -> None:
    block = derive_descriptors(_digital_silence_summary(), is_silent=True)
    assert block.library == DescriptorsLibrary(
        thresholds_sha256=thresholds_sha256(),
        deriver_version=DERIVER_VERSION,
        advisory_coverage=None,
        advisory_dropped=None,
    )
