"""Schema export engine tests (Task C2).

The Pydantic models are the single source of truth (design section 3.7); the
``schema`` command emits a draft-2020-12 JSON Schema *generated from* them. These
tests pin the dialect URI, the property/model correspondence (guarding against
schema/model divergence), the ``check_schema_version`` major guard, and the
wired CLI command.
"""

from __future__ import annotations

import json

import pytest

from sonoscope import cli
from sonoscope.errors import UsageError
from sonoscope.schema import ExitCode
from sonoscope.schema.generate import (
    DRAFT_2020_12_URI,
    SCHEMA_KINDS,
    check_schema_version,
    json_schema_for,
)
from sonoscope.schema.models import AnalysisReport
from tests.schema.test_models import REPORT

_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def test_dialect_uri_constant_exact():
    assert DRAFT_2020_12_URI == _2020_12


@pytest.mark.parametrize("kind", list(SCHEMA_KINDS))
def test_each_kind_declares_2020_12(kind):
    assert json_schema_for(kind)["$schema"] == _2020_12


def test_analysis_title_and_top_level_properties_match_model():
    schema = json_schema_for("analysis")
    assert schema["title"] == "AnalysisReport"
    assert set(schema["properties"].keys()) == set(AnalysisReport.model_fields.keys())


def test_analysis_schema_carries_schema_version_property():
    # check_schema_version consumers compare report.schema_version against
    # this schema; the property must be exported by the generator.
    schema = json_schema_for("analysis")
    # Reflects the SCHEMA_VERSION bumps 1.0.0 -> 1.1.0 (additive midi kind, S1),
    # 1.1.0 -> 1.2.0 (additive descriptors block), 1.2.0 -> 1.3.0 (additive
    # descriptor_gate field), and 1.3.0 -> 1.4.0 (additive wav-analysis kind +
    # input_provenance).
    assert schema["properties"]["schema_version"]["default"] == "1.5.0"


def test_unknown_kind_rejected():
    with pytest.raises(UsageError) as exc:
        json_schema_for("bogus")
    assert exc.value.exit_code == ExitCode.USAGE


def test_reference_report_round_trips_through_model():
    # Guard against schema/model divergence: the model that backs the exported
    # schema must still validate + reproduce a known-good report exactly.
    report = AnalysisReport.model_validate(REPORT)
    assert report.model_dump(mode="json") == REPORT


def test_matching_major_accepted():
    assert check_schema_version(1) is None


def test_unknown_major_rejected():
    with pytest.raises(UsageError) as exc:
        check_schema_version(2)
    assert exc.value.exit_code == ExitCode.USAGE


def test_cli_schema_command_prints_2020_12(capsys):
    code = cli.main(["schema"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["$schema"] == _2020_12
    assert payload["title"] == "AnalysisReport"


def test_cli_schema_command_writes_out_file(tmp_path):
    out = tmp_path / "analysis.schema.json"
    code = cli.main(["schema", "--kind", "analysis", "--out", str(out)])
    assert code == 0
    assert json.loads(out.read_text()) == json_schema_for("analysis")
