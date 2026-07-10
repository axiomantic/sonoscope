"""Schema contract tests for the additive ``AnalysisReport.descriptor_gate`` field.

The descriptor-gate verdict must persist in the versioned report
schema by design (previously a CLI-only stderr signal). ``DescriptorGateResult`` is an
additive-optional block on ``AnalysisReport`` (Optional + None default =
backward compatible: a pre-1.3.0 report JSON with NO ``descriptor_gate`` key
still validates). The SCHEMA_VERSION bump 1.2.0 -> 1.3.0 (MINOR; major stays 1)
covers this additive field.

All assertions are exact-equality (Level 4+): a complete expected object / dict
is constructed and compared with ``==``; no substring, truthiness, or count
checks on the value under test.
"""

from __future__ import annotations

from copy import deepcopy

from sonoscope.schema.models import (
    SCHEMA_VERSION,
    AnalysisReport,
    DescriptorGateResult,
)
from tests.schema.test_models import REPORT


def test_schema_version_bumped_to_1_4_0():
    # Additive descriptor_gate field (1.3.0), then additive wav-analysis kind +
    # input_provenance (1.4.0); MINOR bumps, major stays 1 so
    # check_schema_version is unaffected.
    assert SCHEMA_VERSION == "1.4.0"


def test_descriptor_gate_result_round_trip():
    result = DescriptorGateResult(
        verdict="RED",
        reasons=["DESC_MISSING: dense", "DESC_UNEXPECTED: bright"],
        spec_sha256="a" * 64,
    )
    reloaded = DescriptorGateResult.model_validate_json(result.model_dump_json())
    assert reloaded == result
    assert reloaded.model_dump(mode="json") == {
        "verdict": "RED",
        "reasons": ["DESC_MISSING: dense", "DESC_UNEXPECTED: bright"],
        "spec_sha256": "a" * 64,
    }


def test_descriptor_gate_result_defaults():
    # reasons default_factory=list; spec_sha256 default None.
    result = DescriptorGateResult(verdict="PASS")
    assert result == DescriptorGateResult(verdict="PASS", reasons=[], spec_sha256=None)
    assert result.model_dump(mode="json") == {
        "verdict": "PASS",
        "reasons": [],
        "spec_sha256": None,
    }


def test_descriptor_gate_result_rejects_unknown_verdict():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DescriptorGateResult(verdict="MAYBE")


def test_descriptor_gate_result_rejects_extra_field():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DescriptorGateResult.model_validate(
            {"verdict": "PASS", "reasons": [], "spec_sha256": None, "bogus": 1}
        )


def test_analysis_report_descriptor_gate_defaults_none():
    # A report dict with NO descriptor_gate key validates and defaults to None.
    d = deepcopy(REPORT)
    d.pop("descriptor_gate", None)
    report = AnalysisReport.model_validate(d)
    assert report.descriptor_gate is None


def test_pre_1_4_0_report_without_descriptor_gate_still_validates():
    # Backward compatibility: a 1.2.0-shaped report JSON (no descriptor_gate key)
    # still validates; the additive field defaults None and dumps as None.
    d = deepcopy(REPORT)
    d["schema_version"] = "1.2.0"
    d.pop("descriptor_gate", None)
    report = AnalysisReport.model_validate(d)
    assert report.descriptor_gate is None
    expected = deepcopy(d)
    expected["descriptor_gate"] = None
    assert report.model_dump(mode="json") == expected


def test_descriptor_gate_serializes_when_set():
    d = deepcopy(REPORT)
    d.pop("descriptor_gate", None)
    report = AnalysisReport.model_validate(d)
    gate = DescriptorGateResult(
        verdict="RED", reasons=["DESC_MISSING: dense"], spec_sha256="b" * 64
    )
    report.descriptor_gate = gate
    dumped = report.model_dump(mode="json")
    assert dumped["descriptor_gate"] == {
        "verdict": "RED",
        "reasons": ["DESC_MISSING: dense"],
        "spec_sha256": "b" * 64,
    }
