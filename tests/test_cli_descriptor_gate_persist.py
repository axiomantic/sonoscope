"""CLI persistence of the descriptor-gate verdict into the STDOUT report JSON.

The descriptor-gate verdict must persist in the versioned report
schema by design. ``_run_analyze`` now populates ``report.descriptor_gate`` (verdict +
reasons + the sha256 of the raw expectation-spec bytes) BEFORE printing the
report, so the field appears in the stdout report JSON — IN ADDITION to the
existing single-line stderr verdict and the exit-4 gate (both unchanged).

Unlike ``test_cli_descriptor_gate.py`` (which stubs the report with a fixed
``model_dump_json``), these tests drive a REAL :class:`AnalysisReport` so its
serializer actually renders the persisted ``descriptor_gate`` block, letting us
round-trip the exact :class:`DescriptorGateResult` back out of captured stdout.
The render seam (``analyze_plugin_spec``/backend/plugin/render-spec) is stubbed;
the descriptor comparator and the real spec-file read run for real.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from sonoscope import cli
from sonoscope.schema import ExitCode
from sonoscope.schema.models import AnalysisReport, DescriptorGateResult
from tests.schema.test_models import REPORT

#: A gate-eligible ``bright`` measured term so the comparator PASSes an
#: ``expect_present: ["bright"]`` and REDs an ``expect_present: ["dense"]``.
_DESCRIPTORS = {
    "measured": [
        {
            "term": "bright",
            "value": 3200.0,
            "metric": "spectral_centroid_hz",
            "direction": "high",
            "threshold": None,
            "estimated": False,
            "confidence": None,
        }
    ],
    "hybrid": [],
    "advisory": [],
    "summary": "",
    "library": {
        "thresholds_sha256": "x" * 64,
        "deriver_version": "test",
        "advisory_coverage": None,
        "advisory_dropped": None,
    },
}


def _real_report() -> AnalysisReport:
    """A genuine ``AnalysisReport`` carrying a gate-eligible descriptors block."""
    d = deepcopy(REPORT)
    d.pop("descriptor_gate", None)
    d["descriptors"] = deepcopy(_DESCRIPTORS)
    return AnalysisReport.model_validate(d)


def _install(monkeypatch, report):
    """Stub the render seam so ``analyze_plugin_spec`` returns ``report`` verbatim."""
    monkeypatch.setattr(cli, "analyze_plugin_spec", lambda *a, **k: report)
    monkeypatch.setattr(cli, "_backend", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_require_plugin", lambda plugin, *, component: Path(plugin))
    monkeypatch.setattr(
        cli, "_load_spec", lambda spec_path, *, component: (None, "spec-sha")
    )


def _write_spec(tmp_path: Path, payload: dict) -> tuple[str, bytes]:
    p = tmp_path / "expect.json"
    raw = json.dumps(payload).encode("utf-8")
    p.write_bytes(raw)
    return str(p), raw


def test_gated_run_persists_red_verdict_in_stdout(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _real_report())
    spec_path, spec_bytes = _write_spec(tmp_path, {"expect_present": ["dense"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec_path])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    payload = json.loads(captured.out)
    gate = DescriptorGateResult.model_validate(payload["descriptor_gate"])
    assert gate == DescriptorGateResult(
        verdict="RED",
        reasons=["DESC_MISSING: dense"],
        spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
    )
    # The additive schema field does NOT replace the stderr verdict signal.
    assert captured.err == '{"verdict":"RED","reasons":["DESC_MISSING: dense"]}\n'


def test_gated_run_persists_pass_verdict_in_stdout(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _real_report())
    spec_path, spec_bytes = _write_spec(tmp_path, {"expect_present": ["bright"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec_path])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    payload = json.loads(captured.out)
    gate = DescriptorGateResult.model_validate(payload["descriptor_gate"])
    assert gate == DescriptorGateResult(
        verdict="PASS",
        reasons=[],
        spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
    )
    assert captured.err == '{"verdict":"PASS","reasons":[]}\n'


def test_gated_fail_on_red_persists_gate_and_exits_analysis(tmp_path, monkeypatch, capsys):
    # The persisted field is additive: exit 4 on RED under --fail-on-red is unchanged.
    _install(monkeypatch, _real_report())
    spec_path, spec_bytes = _write_spec(tmp_path, {"expect_present": ["dense"]})

    code = cli.main(
        ["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec_path, "--fail-on-red"]
    )
    captured = capsys.readouterr()

    assert code == int(ExitCode.ANALYSIS)
    payload = json.loads(captured.out)
    gate = DescriptorGateResult.model_validate(payload["descriptor_gate"])
    assert gate == DescriptorGateResult(
        verdict="RED",
        reasons=["DESC_MISSING: dense"],
        spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
    )


def test_ungated_run_leaves_descriptor_gate_none(monkeypatch, capsys):
    _install(monkeypatch, _real_report())

    code = cli.main(["analyze", "--plugin", "p.vst3"])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    payload = json.loads(captured.out)
    assert payload["descriptor_gate"] is None
    assert captured.err == ""
