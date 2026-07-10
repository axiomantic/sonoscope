"""Dogfood — the silent-output test plugin's wrapped-VST3 clap-wrapper path +
silent-output RED (the green-mirage capstone).

This is the OTHER half of the v1 dogfood story from the Surge XT closed loop: it
proves the tool's *actual purpose* end to end against an external plugin project's
clap-wrapper VST3 — pedalboard LOADS the silent-output test plugin's wrapped ``.vst3``, the backend
INTROSPECTS its params, and the render->deterministic-ground-truth path RUNS — AND
it is the load-bearing green-mirage proof that the ``silent-output`` tripwire
actually catches a real failure. The silent-output test plugin writes ``0.0`` to
every output sample by construction, so a note-on (``expected_audio`` derived
``True`` from the ``kind="midi"`` stimulus) that produces silence is exactly the
"headless render silently produced nothing" failure the tripwire exists to catch.
The tripwire therefore fires ``RED`` on a GENUINELY silent render with audio
genuinely expected — not a rigged threshold.

THE CAPSTONE (by design): a ``RED`` verdict is a
healthy *finding*, not a crash. So the ``silent-output`` verdict is exactly
``"RED"`` and ``tripwires.overall`` is exactly ``"RED"``, YET the ``analyze``
process exits ``0`` — a truthful RED report was produced and printed. This is the
distinction between "the tool found a problem" (exit 0, RED report) and "the tool
itself failed" (nonzero exit, fatal envelope).

The 0-param caveat (a deliberate review decision): the walking-skeleton test CLAP returns
``nil`` for the params extension, so pedalboard enumerates **0 parameters**. The
acceptance sub-clause "introspects params (non-empty)" therefore CANNOT hold
against this plugin as built — that is the plugin's nature, not a tool defect.
Per that review decision the param assertion is RELAXED to "the
introspection accessor works and returns a VALID (possibly empty) params list";
the NON-EMPTY param assertion stays where it belongs — on the Surge XT
dogfood (``tests/dogfood/test_surge_xt.py``), where Surge's 775 named params
exercise the non-empty introspection path. This test proves the accessor + the
silent-output RED.

Everything here needs the wrapped-VST3 produced by the external plugin project's
wrapper build and so is
``@pytest.mark.integration``; the module fixture skips (with an explicit reason,
never a silent pass — AGENTS.md testing discipline) when the test bundle is
absent, and the whole module is deselected by the default ``pytest -m "not
integration"`` run.

CROSS-REPO NOTE: the ``.vst3`` lives in an external plugin project's repo and is
consumed READ-ONLY (loaded, never modified/rebuilt); this test writes nothing
into that repo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import pytest

from sonoscope import cli
from sonoscope.analysis_orchestrator import analyze_plugin_spec
from sonoscope.backends.base import PluginInfo
from sonoscope.backends.pedalboard_vst3 import PedalboardVST3Backend
from sonoscope.schema.models import AnalysisReport
from sonoscope.spec import Spec

# Repo root: tests/dogfood/test_dogfood_vst3.py -> parents[1].parent == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1].parent
_SPECS_DIR = _REPO_ROOT / "specs"
_SPIKE_SPEC = _SPECS_DIR / "dogfood_note.json"

# The silent-output test plugin's wrapped clap-wrapper VST3: the ``build/wrapper/assets/*.vst3`` bundle
# embeds the equivalent ``.clap`` (READ-ONLY consume). This is a DIFFERENT
# repo (an external plugin project); the test only loads it.
#: Env override for a non-default silent-output test plugin location, mirroring
#: ``SONOSCOPE_REFSEQ_CLAP`` in ``conftest.py``.
_SILENT_VST3_ENV = "SONOSCOPE_SILENT_VST3"
_SPIKE_VST3 = Path(os.environ.get(_SILENT_VST3_ENV) or "/path/to/SilentPlugin.vst3")

# The tripwire under test (features.tripwires.SILENT_OUTPUT_ID) — the guard
# against a headless render silently producing default/empty (silent) state.
_SILENT_OUTPUT_ID = "silent-output"

# Silence bound (by design): a render at/below the frozen -80 dBFS silence
# threshold is silent. The spike writes pure zeros, so its summary rms_dbfs sits
# at the deterministic amplitude floor (~-240 dBFS), far below this bound. This is
# the PRECONDITION that makes the silent-output RED legitimate (a genuinely silent
# render), not a threshold rigged to trip.
_SILENCE_BOUND_DBFS = -80.0


def _load_spec() -> Spec:
    """Load + validate the spike note-on spec exactly as the CLI's loader does."""
    return Spec.model_validate_json(_SPIKE_SPEC.read_bytes())


class _SpikeResult(NamedTuple):
    """The once-per-module measurements the acceptance assertions read."""

    report: AnalysisReport
    plugin_info: PluginInfo


@pytest.fixture(scope="module")
def spike(request: pytest.FixtureRequest) -> _SpikeResult:
    """Probe + analyze the spike wrapped-VST3 ONCE (by design).

    Runs the ``analyze_plugin_spec`` path (resolve -> subprocess render ->
    deterministic ground truth -> report) against the note-on spec, plus a
    ``backend.probe`` to exercise the tool-level param-introspection accessor. The
    render spawns an isolated subprocess, so the whole thing is computed once at
    module scope and the individual acceptance assertions read the cached result.
    Skips (explicit reason) when the spike bundle is not present on this machine.
    """
    if not _SPIKE_VST3.exists():
        pytest.skip(
            f"silent-output test plugin wrapped-VST3 not present at {_SPIKE_VST3} "
            "(integration artifact absent; produced by the external plugin "
            "project's wrapper build)"
        )

    backend = PedalboardVST3Backend()
    spec = _load_spec()

    # Tool-level param introspection (the accessor assertion, relaxed per the
    # 0-param caveat): probe() introspects the plugin's param surface at
    # runtime (names never hardcoded). For the spike this is a VALID empty list.
    plugin_info = backend.probe(_SPIKE_VST3)

    report = analyze_plugin_spec(
        spec,
        _SPIKE_VST3,
        backend,
        spec_sha256="dogfood-spike",
        spec_ref="specs/dogfood_note.json",
        perception_enabled=False,
    )
    return _SpikeResult(report=report, plugin_info=plugin_info)


def _silent_output_verdict(report: AnalysisReport) -> str:
    """Extract the ``silent-output`` tripwire verdict from the report block."""
    for result in report.tripwires.results:
        if result.id == _SILENT_OUTPUT_ID:
            return result.verdict
    raise AssertionError(
        f"no {_SILENT_OUTPUT_ID!r} tripwire in results "
        f"{[r.id for r in report.tripwires.results]}"
    )


@pytest.mark.integration
def test_spike_loads_introspects_and_renders(spike: _SpikeResult) -> None:
    """The clap-wrapper VST3 LOADS via pedalboard, INTROSPECTS params, RENDERS.

    Proves the tool's actual purpose end to end: pedalboard loaded the
    external plugin project's silent-output test plugin's clap-wrapper ``.vst3``, the backend
    introspected it (a valid :class:`PluginInfo`), and the render->deterministic
    path produced a valid :class:`AnalysisReport` (constructing it validates the
    full contract).

    The param-introspection assertion is RELAXED per the 0-param caveat: the
    walking-skeleton test CLAP exposes no params extension, so the accessor
    returns a VALID but EMPTY params
    list. This asserts the accessor WORKS and returns a valid (possibly empty)
    list — NOT that it is non-empty. The non-empty introspection path is
    proven on the Surge XT dogfood (775 named params), where params exist.
    """
    report = spike.report
    info = spike.plugin_info
    # Load + introspection produced a real PluginInfo for the clap-wrapper VST3.
    assert isinstance(info, PluginInfo)
    # The instrument classification the note-on render path depends on (the
    # spike declares CLAP_PLUGIN_FEATURE_INSTRUMENT).
    assert info.is_instrument is True
    # RELAXED accessor assertion (the 0-param caveat): the introspection accessor
    # returns a VALID list. For the spike that list is EMPTY (the CLAP
    # exposes 0 params) — asserting it is a valid empty list, NOT that it is
    # non-empty. Non-empty introspection is asserted on the Surge dogfood.
    assert isinstance(info.params, list)
    assert info.params == []
    # The full render->analyze path produced a valid, contract-complete report.
    assert isinstance(report, AnalysisReport)


@pytest.mark.integration
def test_spike_render_is_genuinely_silent(spike: _SpikeResult) -> None:
    """Precondition: the spike render is GENUINELY silent (rms_dbfs <= -80).

    This is what makes the silent-output RED legitimate rather than rigged: the
    render must actually be silent AND audio must actually be expected. The spike
    writes pure zeros by construction, so the deterministic ground-truth mean
    rms_dbfs sits at the amplitude floor (~-240 dBFS), far below the -80 dBFS
    silence bound. ``expected_audio`` is derived ``True`` from the ``kind="midi"``
    note-on stimulus (by design) — a note-on says "there should be sound".
    """
    report = spike.report
    # Genuinely silent output (the real precondition, not a threshold trick).
    assert report.deterministic.summary.rms_dbfs <= _SILENCE_BOUND_DBFS
    # Audio genuinely EXPECTED: a note-on (kind="midi") derives expected_audio True
    # (by design). Silence + audio-expected is exactly the failure the tripwire catches.
    assert report.tripwires.expected_audio is True


@pytest.mark.integration
def test_spike_silent_output_tripwire_is_red(spike: _SpikeResult) -> None:
    """THE CAPSTONE: the ``silent-output`` tripwire fires exactly ``RED``.

    The green-mirage proof: because the spike is silent by construction with audio
    expected, the silent-output tripwire genuinely catches the silent render and
    rolls the overall verdict to RED. Exact-equality on both verdicts (AGENTS.md
    Level 4+). A tripwire that only ever passes proves nothing; this proves it
    catches a real silent render.
    """
    report = spike.report
    # Exact-equality on the specific tripwire verdict AND the roll-up.
    assert _silent_output_verdict(report) == "RED"
    assert report.tripwires.overall == "RED"


@pytest.mark.integration
def test_spike_analyze_cli_exits_zero_on_red_finding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """THE CAPSTONE via the LITERAL ``analyze`` CLI surface: RED report, exit 0.

    The milestone DoD is phrased as a CLI command (``analyze --plugin <spike.vst3>
    --spec specs/dogfood_note.json``). This drives the REAL console-script entry
    point ``cli.main`` in-process with the exact documented flags and asserts the
    contract that closes the loop (by design): a RED tripwire verdict is a healthy
    FINDING, not a fatal error, so ``analyze`` prints the truthful RED report to
    stdout and EXITS 0 — proving the tool distinguishes "found a problem" (exit 0,
    RED report) from "the tool itself failed" (nonzero exit, fatal envelope).
    """
    # Same spike skip-guard as the module fixture: an absent integration artifact
    # is an explicit skip (with reason), never a silent pass.
    if not _SPIKE_VST3.exists():
        pytest.skip(
            f"silent-output test plugin wrapped-VST3 not present at {_SPIKE_VST3} "
            "(integration artifact absent; produced by the external plugin "
            "project's wrapper build)"
        )

    argv = [
        "analyze",
        "--plugin",
        str(_SPIKE_VST3),
        "--spec",
        str(_SPIKE_SPEC),
    ]

    # cli.main IS the ``sonoscope`` console-script entry point; it returns the
    # documented exit code and, for analyze, prints ONLY the AnalysisReport JSON.
    exit_code = cli.main(argv)
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    # THE CAPSTONE assertions (exact-equality, AGENTS.md Level 4+):
    # RED finding but exit 0 (a healthy finding, never a crash — by design).
    assert exit_code == 0
    silent_output = next(
        r for r in payload["tripwires"]["results"] if r["id"] == _SILENT_OUTPUT_ID
    )
    assert silent_output["verdict"] == "RED"
    assert payload["tripwires"]["overall"] == "RED"
    # The precondition that makes the RED legitimate, proved through the JSON too.
    assert payload["deterministic"]["summary"]["rms_dbfs"] <= _SILENCE_BOUND_DBFS
