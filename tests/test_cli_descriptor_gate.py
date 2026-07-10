"""CLI wiring for the descriptor gate: ``analyze --expect-descriptors`` / ``--fail-on-red``.

These tests drive ``_run_analyze`` WITHOUT a real Surge render: ``analyze_plugin_spec``
is monkeypatched to return a hand-built report whose ``descriptors`` block is a real
:class:`DescriptorsBlock`, so the comparator (Task 1) runs against genuine data while the
render/backend/spec seams are stubbed out. The full render path (which needs Surge/plugin
infra unavailable in unit CI) is covered by the integration suite, not here.

The invariants under test (single-stdout-document ordering discipline — all
raising validation precedes the report print):

- stdout carries EXACTLY ONE JSON document on every path (the report, or — on any
  fatal — the fatal envelope, because ``_emit_fatal`` writes to stdout);
- the descriptor verdict is a single-line JSON object on stderr (byte-exact);
- exit code is ``ANALYSIS`` (4) iff ``--fail-on-red`` AND the verdict is RED, else OK (0);
- the cross-flag precondition and spec-load both fire BEFORE any render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonoscope import cli
from sonoscope.schema import ExitCode
from sonoscope.schema.models import (
    DescriptorsBlock,
    DescriptorsLibrary,
    MeasuredDescriptor,
)

#: The exact bytes the stubbed report serializes to. The gated path MUST print this
#: verbatim (plus the ``print`` newline) and nothing else to stdout — a second document
#: on stdout, or a different document, breaks the single-document invariant.
_REPORT_JSON = '{"kind":"analysis","descriptors":"<stub>"}'


class _FakeReport:
    """Minimal stand-in for an ``AnalysisReport``.

    ``_run_analyze`` reads only ``.descriptors`` (for the gate) and calls
    ``.model_dump_json()`` (for stdout), so a fake with those two members exercises the
    handler's ordering logic without constructing a full report. Mirrors the
    ``_FakeMidiReport`` pattern already used for ``analyze-midi`` in ``test_cli.py``.
    """

    def __init__(self, descriptors: DescriptorsBlock | None) -> None:
        self.descriptors = descriptors

    def model_dump_json(self) -> str:
        return _REPORT_JSON


def _block(measured: list[MeasuredDescriptor]) -> DescriptorsBlock:
    """F7 fixture helper: a valid ``DescriptorsBlock`` (summary + library required)."""
    return DescriptorsBlock(
        measured=list(measured),
        hybrid=[],
        advisory=[],
        summary="",
        library=DescriptorsLibrary(thresholds_sha256="x", deriver_version="test"),
    )


#: A block whose only gate-eligible term is ``bright`` (present, in-range).
def _bright_block() -> DescriptorsBlock:
    return _block(
        [
            MeasuredDescriptor(
                term="bright",
                value=3200.0,
                metric="spectral_centroid_hz",
                direction="high",
            )
        ]
    )


#: A block whose ``measured`` list carries the gate-eligible term ``bright`` TWICE.
#: ``evaluate_descriptors`` -> ``_extract_eligible`` rejects this as a malformed block
#: (duplicate_term) with a typed ``InputError`` (DESCRIPTORS_BLOCK_MALFORMED).
def _duplicate_bright_block() -> DescriptorsBlock:
    row = MeasuredDescriptor(
        term="bright",
        value=3200.0,
        metric="spectral_centroid_hz",
        direction="high",
    )
    return _block([row, row])


def _install(monkeypatch, report):
    """Stub the render seam; return the call-log list for ``analyze_plugin_spec``.

    An empty call log after a run PROVES the render never happened (used to assert the
    cross-flag precondition and spec-load fail BEFORE the render).
    """
    calls: list[tuple] = []

    def fake_analyze(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(cli, "analyze_plugin_spec", fake_analyze)
    monkeypatch.setattr(cli, "_backend", lambda *a, **k: None)
    monkeypatch.setattr(
        cli, "_require_plugin", lambda plugin, *, component: Path(plugin)
    )
    monkeypatch.setattr(
        cli, "_load_spec", lambda spec_path, *, component: (None, "spec-sha")
    )
    return calls


def _write_spec(tmp_path: Path, payload: dict) -> str:
    p = tmp_path / "expect.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


# --- Step 3.1: stdout byte-invariance ----------------------------------------


def test_stdout_byte_identical_with_and_without_spec(tmp_path, monkeypatch, capsys):
    """The gated (passing-spec) path prints the SAME single stdout document as legacy."""
    _install(monkeypatch, _FakeReport(_bright_block()))

    assert cli.main(["analyze", "--plugin", "p.vst3"]) == int(ExitCode.OK)
    stdout_without = capsys.readouterr().out

    spec = _write_spec(tmp_path, {"expect_present": ["bright"]})
    assert cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec]) == int(
        ExitCode.OK
    )
    stdout_with = capsys.readouterr().out

    assert stdout_without == _REPORT_JSON + "\n"
    assert stdout_with == _REPORT_JSON + "\n"
    assert stdout_with == stdout_without


# --- Step 3.2: byte-exact stderr verdict (PASS and RED) ----------------------


def test_stderr_verdict_red_is_byte_exact(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["dense"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    assert captured.out == _REPORT_JSON + "\n"
    assert captured.err == '{"verdict":"RED","reasons":["DESC_MISSING: dense"]}\n'


def test_stderr_verdict_pass_is_byte_exact(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["bright"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    assert captured.out == _REPORT_JSON + "\n"
    assert captured.err == '{"verdict":"PASS","reasons":[]}\n'


# --- Step 3.3: exit codes ----------------------------------------------------


def test_fail_on_red_with_red_exits_analysis(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["dense"]})

    code = cli.main(
        ["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec, "--fail-on-red"]
    )
    captured = capsys.readouterr()

    assert code == int(ExitCode.ANALYSIS)
    assert captured.out == _REPORT_JSON + "\n"
    assert captured.err == '{"verdict":"RED","reasons":["DESC_MISSING: dense"]}\n'


def test_fail_on_red_with_pass_exits_ok(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["bright"]})

    code = cli.main(
        ["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec, "--fail-on-red"]
    )
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    assert captured.out == _REPORT_JSON + "\n"
    assert captured.err == '{"verdict":"PASS","reasons":[]}\n'


def test_red_without_fail_on_red_exits_ok(tmp_path, monkeypatch, capsys):
    """A RED verdict without ``--fail-on-red`` still surfaces on stderr but exits 0."""
    _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["dense"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.OK)
    assert captured.out == _REPORT_JSON + "\n"
    assert captured.err == '{"verdict":"RED","reasons":["DESC_MISSING: dense"]}\n'


def test_fail_on_red_without_spec_is_usage_error(monkeypatch, capsys):
    """C4: ``--fail-on-red`` without ``--expect-descriptors`` → exit 1, before any render."""
    calls = _install(monkeypatch, _FakeReport(_bright_block()))

    code = cli.main(["analyze", "--plugin", "p.vst3", "--fail-on-red"])
    captured = capsys.readouterr()

    assert code == int(ExitCode.USAGE)
    assert calls == []  # precondition fired BEFORE the render
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "code": "USAGE_FAIL_ON_RED_REQUIRES_SPEC",
        "message": "analyze --fail-on-red requires --expect-descriptors",
        "detail": None,
        "severity": "fatal",
        "component": "cli",
    }


def test_missing_descriptors_block_is_input_error(tmp_path, monkeypatch, capsys):
    """C1: ``--expect-descriptors`` but ``report.descriptors is None`` → exit 2, no report on stdout."""
    _install(monkeypatch, _FakeReport(None))
    spec = _write_spec(tmp_path, {"expect_present": ["bright"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.INPUT)
    assert captured.err == ""
    payload = json.loads(captured.out)  # exactly one document on stdout
    assert payload["error"] == {
        "code": "DESCRIPTORS_NO_BLOCK",
        "message": "analyze --expect-descriptors requires a descriptors block in the report",
        "detail": {"reason": "no_descriptors_block"},
        "severity": "fatal",
        "component": "analyze",
    }


def test_bad_spec_exits_input_error_before_render(tmp_path, monkeypatch, capsys):
    """L-series + fail-fast: a malformed spec exits 2 WITHOUT rendering or printing a report."""
    calls = _install(monkeypatch, _FakeReport(_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["wobble"]})  # not gate-eligible

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.INPUT)
    assert calls == []  # spec load fired BEFORE the render
    assert captured.err == ""
    payload = json.loads(captured.out)  # exactly one document on stdout
    assert payload["error"] == {
        "code": "DESCRIPTORS_EXPECTED_SPEC_INVALID",
        "message": "expectation spec references ineligible term 'wobble' for block_kind 'audio'",
        "detail": {"reason": "unknown_term", "term": "wobble", "block_kind": "audio"},
        "severity": "fatal",
        "component": "analyze",
    }


def test_malformed_descriptors_block_is_input_error_before_print(
    tmp_path, monkeypatch, capsys
):
    """Evaluate-before-print: a malformed report ``descriptors`` block (duplicate
    gate-eligible term) makes ``evaluate_descriptors`` raise BEFORE the report print.

    Locks the ordering: because the raise precedes ``print(report...)``, stdout carries
    ONLY the fatal envelope (never the report body), and the verdict never reaches
    stderr. If the ``evaluate_descriptors`` call were moved AFTER the report print, the
    report body would already be on stdout — two documents, and ``json.loads`` would
    fail on the extra data (the full-envelope equality below asserts stdout is EXACTLY
    the fatal envelope, so a leaked report body cannot slip through).
    """
    _install(monkeypatch, _FakeReport(_duplicate_bright_block()))
    spec = _write_spec(tmp_path, {"expect_present": ["bright"]})

    code = cli.main(["analyze", "--plugin", "p.vst3", "--expect-descriptors", spec])
    captured = capsys.readouterr()

    assert code == int(ExitCode.INPUT)
    assert captured.err == ""  # the verdict never leaked to stderr
    payload = json.loads(captured.out)  # exactly one document on stdout
    assert payload["error"] == {
        "code": "DESCRIPTORS_BLOCK_MALFORMED",
        "message": "descriptors block has a duplicate gate-eligible term",
        "detail": {"reason": "duplicate_term", "term": "bright"},
        "severity": "fatal",
        "component": "analyze",
    }
