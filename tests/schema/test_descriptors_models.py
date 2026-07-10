"""Descriptor block schema contract tests (Task T1 / C1.1).

RED-must-trip exact-equality round-trips for the additive ``descriptors`` block
(measured / hybrid / advisory + library provenance) and the additive-optional
``descriptors`` field on ``AnalysisReport`` (wired) and ``MidiAnalysisReport``
(defined, unwired until C2).

All assertions are exact-equality (Level 4+): a complete expected object /
dict is constructed and compared with ``==``; no substring, truthiness, count,
or ``isinstance`` checks on the value under test.
"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from sonoscope.schema.models import (
    AdvisoryDescriptor,
    AnalysisReport,
    DescriptorsBlock,
    DescriptorsLibrary,
    HybridDescriptor,
    MeasuredDescriptor,
    MidiAnalysisReport,
)
from tests.schema.test_midi_models import REPORT as MIDI_REPORT
from tests.schema.test_models import REPORT as ANALYSIS_REPORT

# --- Reference descriptors block (mirror design section 3) ------------------
# One representative row per array + full library provenance, with EVERY field
# present so ``model_dump`` reproduces this dict exactly.

DESCRIPTORS = {
    "measured": [
        {
            "term": "bright",
            "value": 2501.0,
            "metric": "spectral_centroid_hz",
            "direction": "high",
            "threshold": 2500.0,
            "estimated": False,
            "confidence": None,
        },
        {
            "term": "tempo-audio",
            "value": 128.0,
            "metric": "tempo_bpm",
            "direction": "value",
            "threshold": None,
            "estimated": True,
            "confidence": 0.82,
        },
    ],
    "hybrid": [
        {
            "term": "driving",
            "anchor_metric": "driving_composite",
            "anchor_value": 0.72,
            "direction": "high",
            "confidence": 0.72,
        }
    ],
    "advisory": [
        {"term": "cosmic", "source": "lalm-mapped", "confidence": 0.6}
    ],
    "summary": "measured: bright, 128 BPM, driving; advisory: cosmic",
    "library": {
        "thresholds_sha256": "a" * 64,
        "deriver_version": "1.0.0",
        "advisory_coverage": 0.5,
        "advisory_dropped": 2,
    },
}


# --- Public-API surface (package-root export) -------------------------------


def test_descriptor_types_exported_from_package_root():
    from sonoscope import schema
    from sonoscope.schema import (
        AdvisoryDescriptor,
        AdvisorySource,
        DescriptorsBlock,
        DescriptorsLibrary,
        Direction,
        HybridDescriptor,
        MeasuredDescriptor,
    )
    from sonoscope.schema import models

    assert MeasuredDescriptor is models.MeasuredDescriptor
    assert HybridDescriptor is models.HybridDescriptor
    assert AdvisoryDescriptor is models.AdvisoryDescriptor
    assert DescriptorsLibrary is models.DescriptorsLibrary
    assert DescriptorsBlock is models.DescriptorsBlock
    assert Direction is models.Direction
    assert AdvisorySource is models.AdvisorySource
    # exported on the public surface
    for name in (
        "MeasuredDescriptor",
        "HybridDescriptor",
        "AdvisoryDescriptor",
        "DescriptorsLibrary",
        "DescriptorsBlock",
        "Direction",
        "AdvisorySource",
    ):
        assert name in schema.__all__


# --- Round-trip exact-equality ---------------------------------------------


def test_descriptors_block_round_trip():
    block = DescriptorsBlock.model_validate(DESCRIPTORS)
    # JSON round-trip reproduces both the model and the reference dict exactly.
    reloaded = DescriptorsBlock.model_validate_json(block.model_dump_json())
    assert reloaded == block
    assert reloaded.model_dump(mode="json") == DESCRIPTORS


def test_measured_readout_defaults():
    md = MeasuredDescriptor(
        term="tempo-audio", value=128.0, metric="tempo_bpm", direction="value"
    )
    assert md == MeasuredDescriptor(
        term="tempo-audio",
        value=128.0,
        metric="tempo_bpm",
        direction="value",
        threshold=None,
        estimated=False,
        confidence=None,
    )


# --- Strictness (_Strict / extra="forbid") ----------------------------------


def test_extra_field_forbidden():
    d = deepcopy(DESCRIPTORS)
    d["bogus"] = 1
    with pytest.raises(ValidationError):
        DescriptorsBlock.model_validate(d)


# --- Additive-optional field on the report kinds ----------------------------


def test_analysis_report_additive_optional():
    # Drop the key entirely to prove the additive-optional guarantee: a report
    # dict with NO descriptors key still validates and defaults to None.
    d = deepcopy(ANALYSIS_REPORT)
    d.pop("descriptors", None)
    report = AnalysisReport.model_validate(d)
    assert report.descriptors is None
    expected = deepcopy(d)
    expected["descriptors"] = None
    assert report.model_dump(mode="json") == expected


def test_analysis_report_carries_descriptors():
    d = deepcopy(ANALYSIS_REPORT)
    d["descriptors"] = deepcopy(DESCRIPTORS)
    report = AnalysisReport.model_validate(d)
    assert report.descriptors == DescriptorsBlock.model_validate(DESCRIPTORS)
    assert report.model_dump(mode="json") == d


def test_midi_report_descriptors_defaults_none():
    d = deepcopy(MIDI_REPORT)
    d.pop("descriptors", None)
    report = MidiAnalysisReport.model_validate(d)
    assert report.descriptors is None
    expected = deepcopy(d)
    expected["descriptors"] = None
    assert report.model_dump(mode="json") == expected
