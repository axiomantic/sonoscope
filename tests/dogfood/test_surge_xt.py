"""H2 dogfood — the Surge XT closed-loop milestone (design §14.1, §14.3).

This is the v1 success criterion made executable: the full
``(plugin, stimulus, param-set) -> wav -> deterministic analysis -> feedback``
loop run against a REAL plugin (Surge XT), plus the canonical ``iterate``
assertion that a defined parameter change (closing a lowpass filter) produced the
expected, *measurable* spectral change beyond the plugin's own measured
nondeterminism floor.

The two specs under ``specs/`` differ by EXACTLY one audible parameter — the
Scene A filter-1 cutoff (``a_filter_1_cutoff``, discovered via ``backend.probe``,
never hardcoded from documentation). Surge XT's factory init patch leaves
filter 1 bypassed (``a_filter_1_type == "Off"``), so a cutoff sweep on the raw
init patch is inaudible — the centroid does not move at all. Both specs therefore
also engage a 12 dB lowpass (``a_filter_1_type`` step 1 of 34, normalized
``1/33``) so the cutoff sits in the signal path; that single filter-type value is
identical across the two specs, leaving the cutoff (open ``0.9`` vs closed
``0.1``) as the only difference under test. This filter-routing requirement is a
real finding from the H2 probe, documented here so the specs are not mistaken for
over-specification.

Everything here needs the installed Surge XT VST3 and so is
``@pytest.mark.integration``; the module fixture skips (with an explicit reason,
never a silent pass — AGENTS.md testing discipline) when Surge is absent, and the
whole module is deselected by the default ``pytest -m "not integration"`` run.

Acceptance mirrors the impl plan Task H2 exactly:

1. ``analyze`` the open spec -> a valid :class:`AnalysisReport`, non-silent audio
   (``rms_dbfs > -80``), and ``tripwires.overall == "PASS"``.
2. THE HEADLINE: ``iterate`` open(baseline) vs closed(candidate) on
   ``deterministic.summary.spectral_centroid_hz`` with ``direction="decrease"``
   -> verdict exactly ``"PASS"`` AND ``delta.significant is True`` — the centroid
   drop clears the measured noise threshold. This is the closed-loop
   render->listen->feedback proof.
3. ``determinism`` on the noisy Surge patch -> a NONZERO floor (the same measured
   floor that feeds the iterate threshold in (2)).

Additionally (H1 MINOR-2): the §7.2 plugin-path latency targets in
``doctor.LATENCY_TARGETS_S`` — otherwise unexercised, since ``doctor``'s own
benchmark measures only the plugin-free path — are measured here against the real
render and deterministic-only-analyze wall times and reported vs target as a SOFT
criterion (over-target warns, matching ``doctor``, but never hard-fails the test;
slow hardware must not break the milestone).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import pytest

from sonoscope import cli, doctor
from sonoscope.analysis_orchestrator import analyze_plugin_spec
from sonoscope.backends.pedalboard_vst3 import PedalboardVST3Backend, binary_sha256
from sonoscope.determinism import DEFAULT_REPEATS, measure_floors
from sonoscope.iterate import run_iterate
from sonoscope.render_orchestrator import render
from sonoscope.schema.models import AnalysisReport, DeterminismFloors, IterateDelta
from sonoscope.spec import Spec

# Repo root: tests/dogfood/test_surge_xt.py -> parents[1].parent == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1].parent
_SPECS_DIR = _REPO_ROOT / "specs"

# A4-confirmed system install location for the pinned Surge XT VST3 bundle
# (mirrors tests/conftest.py::SURGE_XT_VST3). Kept local so this module-scoped
# fixture can skip without depending on the function-scoped ``surge_vst3_path``.
_SURGE_XT_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

# The metric under test: the deterministic ground-truth spectral centroid. A
# lowpass closing lowers it (less high-frequency energy passes -> darker).
_CENTROID_METRIC = "deterministic.summary.spectral_centroid_hz"

# Non-silence bound: the design's silent-output boundary is far below this; the
# open patch renders at ~-22 dBFS and the closed patch at ~-50 dBFS, both well
# above -80 dBFS (H2 acceptance 1). A silent render would sit at/below the
# silence threshold and fail this.
_NON_SILENT_DBFS = -80.0


class _LoopResult(NamedTuple):
    """The once-per-module closed-loop measurements the assertions read."""

    open_report: AnalysisReport
    closed_report: AnalysisReport
    floors: DeterminismFloors
    delta: IterateDelta
    render_2s_s: float
    deterministic_analyze_s: float


def _load_spec(name: str) -> Spec:
    """Load + validate a spec from ``specs/`` exactly as the CLI's loader does."""
    return Spec.model_validate_json((_SPECS_DIR / name).read_bytes())


@pytest.fixture(scope="module")
def loop() -> _LoopResult:
    """Run the full render->analyze->measure-floor->iterate loop against Surge ONCE.

    Every render spawns an isolated subprocess (~1 s), so the whole loop is
    computed once at module scope and the individual acceptance assertions read
    the cached result. Skips (explicit reason) when Surge XT is not installed.
    """
    if not _SURGE_XT_VST3.exists():
        pytest.skip(
            f"Surge XT not installed at {_SURGE_XT_VST3} "
            "(integration artifact absent; run scripts/install_surge_xt.sh)"
        )

    open_spec = _load_spec("surge_lowpass_open.json")
    closed_spec = _load_spec("surge_lowpass_closed.json")
    backend = PedalboardVST3Backend()

    # H1 MINOR-2 latency (§7.2): time a single 2 s render (render_2s) and a single
    # deterministic-only analyze (render + features == deterministic_analyze).
    # Both are warmed first: the first render in a process pays the Surge XT VST3
    # scan + JUCE init, and the first analyze pays ~2.2 s of librosa/scipy/sklearn
    # import + numba JIT. The targets measure the steady state, not that one-time
    # cost, so an unwarmed timing warns on every machine. Warm-up results are
    # discarded; the acceptance assertions below read the timed (warm) ones.
    render(open_spec, _SURGE_XT_VST3, backend)
    analyze_plugin_spec(
        open_spec,
        _SURGE_XT_VST3,
        backend,
        spec_sha256="dogfood-open-warmup",
        spec_ref="specs/surge_lowpass_open.json",
        perception_enabled=False,
    )

    t0 = time.perf_counter()
    seed_outcome = render(open_spec, _SURGE_XT_VST3, backend)
    render_2s_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    open_report = analyze_plugin_spec(
        open_spec,
        _SURGE_XT_VST3,
        backend,
        spec_sha256="dogfood-open",
        spec_ref="specs/surge_lowpass_open.json",
        perception_enabled=False,
    )
    deterministic_analyze_s = time.perf_counter() - t0

    closed_report = analyze_plugin_spec(
        closed_spec,
        _SURGE_XT_VST3,
        backend,
        spec_sha256="dogfood-closed",
        spec_ref="specs/surge_lowpass_closed.json",
        perception_enabled=False,
    )

    # H2 acceptance 3 / iterate input: measure the (binary, patch_class) floor on
    # the baseline (open) patch — the same floor that thresholds the iterate delta
    # below (this is exactly how cli._run_iterate feeds run_iterate).
    floors = measure_floors(
        lambda: render(open_spec, _SURGE_XT_VST3, backend).wav_path,
        binary_sha256=binary_sha256(_SURGE_XT_VST3),
        patch_class=open_spec.patch_class,
        resolved_sha256=seed_outcome.resolved.resolved_sha256,
        stimulus_ref=seed_outcome.resolved.stimulus.ref or "inline",
        repeats=DEFAULT_REPEATS,
    )

    # H2 acceptance 2 (THE HEADLINE): the closed-loop iterate verdict.
    delta = run_iterate(
        open_report,
        closed_report,
        floors,
        metric=_CENTROID_METRIC,
        direction="decrease",
    )

    return _LoopResult(
        open_report=open_report,
        closed_report=closed_report,
        floors=floors,
        delta=delta,
        render_2s_s=render_2s_s,
        deterministic_analyze_s=deterministic_analyze_s,
    )


@pytest.mark.integration
def test_open_render_is_valid_non_silent_and_tripwires_pass(loop: _LoopResult) -> None:
    """H2 acceptance 1: the open spec analyzes to a valid, non-silent, PASSing report.

    A valid :class:`AnalysisReport` (constructing it validates the full C1
    contract), audio comfortably above the silence bound, and an overall tripwire
    verdict of exactly ``PASS`` — the healthy end of the render->listen loop.
    """
    report = loop.open_report
    assert isinstance(report, AnalysisReport)
    # Non-silent: real audible output, not a silent/near-silent render (>-80 dBFS).
    assert report.deterministic.summary.rms_dbfs > _NON_SILENT_DBFS
    # No tripwire fired on the healthy open patch.
    assert report.tripwires.overall == "PASS"


@pytest.mark.integration
def test_closed_render_is_non_silent(loop: _LoopResult) -> None:
    """The closed (dark) patch is still audible — the lowpass darkens, not mutes.

    Guards the headline: a centroid "drop" caused by the render collapsing to
    silence would be a degenerate false proof. The closed patch must remain above
    the silence bound so the measured centroid drop reflects a real filter sweep.
    """
    assert loop.closed_report.deterministic.summary.rms_dbfs > _NON_SILENT_DBFS
    assert loop.closed_report.tripwires.overall == "PASS"


@pytest.mark.integration
def test_iterate_filter_sweep_pass(loop: _LoopResult) -> None:
    """H2 acceptance 2 (THE HEADLINE): closing the lowpass -> iterate verdict PASS.

    The canonical closed-loop render->listen->feedback proof: the spectral centroid
    genuinely drops when the filter closes, by a margin far beyond the plugin's
    measured nondeterminism floor, so the significance gate returns a true PASS —
    NOT a rigged threshold. ``abs_delta`` is the signed candidate-minus-baseline
    difference, so a decrease is negative and its magnitude clears the noise
    threshold by a wide margin.
    """
    delta = loop.delta.delta
    # Exact-equality on the verdict + significance (AGENTS.md Level 4+).
    assert loop.delta.verdict == "PASS"
    assert delta.significant is True
    # The change is a genuine DECREASE (signed abs_delta < 0)...
    assert delta.abs_delta < 0.0
    # ...and supra-threshold: its magnitude clears the measured noise threshold.
    # (Documents WHY it is significant; the significance flag above is the gate.)
    assert abs(delta.abs_delta) > delta.noise_threshold


@pytest.mark.integration
def test_iterate_closed_loop_via_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """H2 acceptance 2, proved through the LITERAL ``iterate`` CLI command surface.

    The milestone DoD is phrased as CLI commands (``iterate --plugin ... --baseline
    ... --candidate ... --metric ... --direction decrease``); the direct-engine test
    above proves ``run_iterate`` returns PASS, but not that the ``iterate`` argv path
    (argparse -> F1 analyze both specs -> F2 read/measure floor -> ``run_iterate`` ->
    ``IterateDelta`` JSON envelope -> design 3.6 exit code) produces it. This drives
    the REAL console-script entry point ``cli.main`` in-process with the exact
    documented flags and asserts the emitted JSON verdict + process exit code —
    closing the gap between the engine call and the shipped command. The direct
    tests above are retained; this is additive proof of the argv wiring.
    """
    # Same Surge skip-guard as the module ``loop`` fixture: an absent integration
    # artifact is an explicit skip (with reason), never a silent pass.
    if not _SURGE_XT_VST3.exists():
        pytest.skip(
            f"Surge XT not installed at {_SURGE_XT_VST3} "
            "(integration artifact absent; run scripts/install_surge_xt.sh)"
        )

    # The exact ``iterate`` command surface (cli.build_parser): open spec is the
    # baseline, closed spec the candidate, decreasing the deterministic centroid.
    argv = [
        "iterate",
        "--plugin",
        str(_SURGE_XT_VST3),
        "--baseline",
        str(_SPECS_DIR / "surge_lowpass_open.json"),
        "--candidate",
        str(_SPECS_DIR / "surge_lowpass_closed.json"),
        "--metric",
        _CENTROID_METRIC,
        "--direction",
        "decrease",
    ]

    # cli.main IS the ``sonoscope`` console-script entry point (pyproject
    # [project.scripts]); it returns the design 3.6 exit code (0 == OK) and, for
    # iterate, prints ONLY the IterateDelta JSON envelope to stdout on success.
    exit_code = cli.main(argv)
    stdout = capsys.readouterr().out

    # Exact-equality on exit code + verdict + significance (AGENTS.md Level 4+).
    assert exit_code == 0
    payload = json.loads(stdout)
    # THE HEADLINE, now proved through the JSON envelope the argv command emits.
    assert payload["verdict"] == "PASS"
    assert payload["delta"]["significant"] is True


@pytest.mark.integration
def test_determinism_floor_is_nonzero(loop: _LoopResult) -> None:
    """H2 acceptance 3: the noisy Surge patch yields a NONZERO centroid floor.

    Surge's noisy patch class is genuinely nondeterministic (noise oscillators /
    free-running LFOs / analog drift), so its measured per-feature floor is a
    real, positive number — the value that thresholds the iterate delta above. A
    zero floor would mean the threshold check was vacuous.
    """
    entry = loop.floors.floors[_CENTROID_METRIC]
    assert entry.floor > 0.0
    # The feature was present in every floor-measuring render (a reliable floor).
    assert entry.repeats == loop.floors.repeats


@pytest.mark.integration
def test_plugin_path_latency_reported_vs_targets(loop: _LoopResult) -> None:
    """H1 MINOR-2: measure the §7.2 plugin-path latency targets (SOFT criterion).

    ``doctor``'s own benchmark measures only the plugin-free path, leaving the
    ``render_2s`` and ``deterministic_analyze`` targets in ``LATENCY_TARGETS_S``
    otherwise unexercised. Here they are measured against the REAL Surge render and
    deterministic-only analyze. Per ``doctor``'s soft-criterion (I2), over-target
    is a non-fatal WARNING — never a hard failure (slow hardware must not break the
    milestone). The only hard assertions are structural: the measurements are real,
    finite, positive wall times.
    """
    measured = {
        "render_2s": loop.render_2s_s,
        "deterministic_analyze": loop.deterministic_analyze_s,
    }
    for metric, measured_s in measured.items():
        target_s = doctor.LATENCY_TARGETS_S[metric]
        # Structural (hard): a broken timer / no-op render would trip this.
        assert measured_s > 0.0
        status = "OK" if measured_s <= target_s else "OVER"
        print(
            f"[H2 latency] {metric}: measured={measured_s:.3f}s "
            f"target={target_s:.3f}s -> {status}"
        )
        if measured_s > target_s:
            # Soft criterion: report, do not fail (matches doctor's I2 behavior).
            warnings.warn(
                f"§7.2 latency over target: {metric} "
                f"measured={measured_s:.3f}s > target={target_s:.3f}s",
                stacklevel=2,
            )
