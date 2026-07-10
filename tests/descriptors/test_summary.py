"""Exact-string renderer tests for ``render_summary`` (design §5.4).

The renderer is pure and deterministic; every assertion is exact string
equality. Ordering within the ``measured:`` clause is: gated measured terms,
then hybrid feel-terms, then readout terms (design example
``"measured: bright, driving, 128 BPM"``).
"""

from __future__ import annotations

from sonoscope.descriptors.summary import render_summary
from sonoscope.schema.models import (
    AdvisoryDescriptor,
    HybridDescriptor,
    MeasuredDescriptor,
)


def _bright() -> MeasuredDescriptor:
    return MeasuredDescriptor(
        term="bright",
        value=2501.0,
        metric="spectral_centroid_hz",
        direction="high",
        threshold=2500.0,
    )


def _driving() -> HybridDescriptor:
    return HybridDescriptor(
        term="driving",
        anchor_metric="driving_composite",
        anchor_value=0.7,
        direction="high",
        confidence=0.7,
    )


def _tempo(value: float = 128.0) -> MeasuredDescriptor:
    return MeasuredDescriptor(
        term="tempo-audio",
        value=value,
        metric="tempo_bpm",
        direction="value",
        threshold=None,
        estimated=True,
        confidence=0.9,
    )


def _rhythmic(value: float = 6.0) -> MeasuredDescriptor:
    return MeasuredDescriptor(
        term="rhythmic-density",
        value=value,
        metric="onset_rate_hz",
        direction="value",
        threshold=None,
    )


def _advisory(term: str) -> AdvisoryDescriptor:
    return AdvisoryDescriptor(term=term, source="lalm-mapped", confidence=0.6)


def test_measured_and_hybrid_only() -> None:
    assert render_summary([_bright()], [_driving()], []) == "measured: bright, driving"


def test_with_tempo_readout() -> None:
    assert (
        render_summary([_bright(), _tempo()], [_driving()], [])
        == "measured: bright, driving, 128 BPM"
    )


def test_with_rhythmic_density_readout() -> None:
    assert (
        render_summary([_bright(), _rhythmic(6.0)], [_driving()], [])
        == "measured: bright, driving, 6.0 onsets/s"
    )


def test_with_advisory_clause() -> None:
    advisory = [_advisory("cosmic"), _advisory("hypnotic")]
    assert (
        render_summary([_bright(), _tempo()], [_driving()], advisory)
        == "measured: bright, driving, 128 BPM; advisory: cosmic, hypnotic"
    )


def test_empty_advisory_clause_omitted() -> None:
    # advisory == [] => no "; advisory: ..." tail at all.
    assert render_summary([_bright()], [], []) == "measured: bright"


def test_both_empty_renders_none() -> None:
    assert render_summary([], [], []) == "measured: (none)"


def test_tempo_rounds_to_integer() -> None:
    assert render_summary([_tempo(128.4)], [], []) == "measured: 128 BPM"
