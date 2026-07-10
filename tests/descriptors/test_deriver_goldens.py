"""Positive golden fixtures for ``derive_descriptors`` (design §11.3, DA #15).

A DISTINCT test class from the boundary RED/GREEN tests: realistic multi-metric
summaries prove the deriver *emits when it should* (the counterpart to the
"correctly silent" empty-degenerate test). Assertions are exact-equality on the
emitted record, filtered to the golden term.
"""

from __future__ import annotations

from typing import Any

from sonoscope.descriptors.deriver import derive_descriptors, norm
from sonoscope.descriptors.thresholds import DERIVER_THRESHOLDS as T
from sonoscope.schema.models import HybridDescriptor, MeasuredDescriptor


def _summary(**overrides: Any):
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


def test_golden_known_bright() -> None:
    s = _summary(spectral_centroid_hz=4000.0, rms_dbfs=-20.0, onset_rate_hz=3.0)
    assert _measured(derive_descriptors(s), "bright") == [
        MeasuredDescriptor(
            term="bright",
            value=4000.0,
            metric="spectral_centroid_hz",
            direction="high",
            threshold=2500.0,
        )
    ]


def test_golden_known_loud() -> None:
    s = _summary(rms_dbfs=-6.0, spectral_centroid_hz=1500.0, onset_rate_hz=3.0)
    assert _measured(derive_descriptors(s), "loud") == [
        MeasuredDescriptor(
            term="loud",
            value=-6.0,
            metric="rms_dbfs",
            direction="high",
            threshold=-18.0,
        )
    ]


def test_golden_known_driving() -> None:
    s = _summary(
        onset_rate_hz=10.0,
        tempo_bpm=140.0,
        tempo_confidence=0.9,
        onset_count=8,
        rms_dbfs=-10.0,
    )
    no = norm(s.onset_rate_hz, T["driving.onset_lo"], T["driving.onset_hi"])
    nt = norm(s.tempo_bpm, T["driving.tempo_lo"], T["driving.tempo_hi"])
    nr = norm(s.rms_dbfs, T["driving.rms_lo"], T["driving.rms_hi"])
    score = (
        T["driving.w_onset"] * no + T["driving.w_tempo"] * nt + T["driving.w_rms"] * nr
    )
    assert _hybrid(derive_descriptors(s), "driving") == [
        HybridDescriptor(
            term="driving",
            anchor_metric="driving_composite",
            anchor_value=score,
            direction="high",
            confidence=score,
        )
    ]
