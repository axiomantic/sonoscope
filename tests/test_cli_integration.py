"""H1: CLI command wiring integration (design §4.5, §3.6/§3.7).

Exercises the real command dispatch that H1 wires in place of the C5 stubs:

- `schema --kind <k>` prints a valid draft-2020-12 JSON Schema for every kind.
- `analyze --wav` analyzes a standalone wav (a synthetic tmp wav here, no plugin
  or model required): it emits the wav-analysis JSON array and exits 0.
- `corpus verify` maps a corpus-integrity FAILURE to a nonzero exit (pins are
  law); a clean corpus exits 0.
- `probe` degrades gracefully when perception is unavailable: it reports the
  unavailable status and exits 0 (perception is advisory, never fatal, §10.2).

These are non-integration: they use argument-only paths (schema), the pure
usage-guard path (`analyze --wav`), and monkeypatched engines (corpus verify /
probe adapter) so no Surge XT, no model, and no plugin are required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from jsonschema import Draft202012Validator

from sonoscope import cli
from sonoscope.backends.base import RenderMeta
from sonoscope.corpus import ItemVerification, VerifyResult
from sonoscope.errors import RenderError
from sonoscope.perception.base import AdapterHealth
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.schema import ExitCode, IterateDirection
from sonoscope.schema.generate import DRAFT_2020_12_URI, SCHEMA_KINDS
from sonoscope.schema.models import AdapterInfo, PerceptionBlock


def test_schema_command_all_kinds(capsys):
    """Every `--kind` prints a valid draft-2020-12 JSON Schema (exit 0).

    MINOR-4: beyond the ``$schema`` label + json-parseability, each emitted schema
    is validated against the draft-2020-12 metaschema (``Draft202012Validator.
    check_schema``) so a malformed-but-parseable schema that merely stamps the
    right dialect URI cannot slip through as a green mirage.
    """
    for kind in SCHEMA_KINDS:
        code = cli.main(["schema", "--kind", kind])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["$schema"] == DRAFT_2020_12_URI
        # Structural validation against the 2020-12 metaschema (not just the label).
        Draft202012Validator.check_schema(payload)


def test_schema_out_trailing_sep_creates_dir_and_per_kind_file(tmp_path, capsys):
    """RED-proving (Gemini cycle 3): ``schema --kind <k> --out <dir>/`` with a
    trailing separator on a NON-EXISTENT path is a directory request, not a file
    named after the last segment. The dir is created and the schema is written to
    ``<kind>.schema.json`` inside it. RED against the pre-fix ``dest.is_dir()``-
    only detection, which (the dir not existing yet) wrote a FILE literally named
    ``newdir``."""
    kind = SCHEMA_KINDS[0]
    newdir = tmp_path / "newdir"
    out_arg = str(newdir) + os.sep  # trailing separator, dir does not exist yet

    code = cli.main(["schema", "--kind", kind, "--out", out_arg])
    assert code == int(ExitCode.OK)

    # The directory was created and holds the per-kind schema file (not a file
    # literally named "newdir").
    assert newdir.is_dir()
    assert not Path(str(newdir)).is_file()
    schema_file = newdir / f"{kind}.schema.json"
    assert schema_file.is_file()
    payload = json.loads(schema_file.read_text())
    assert payload["$schema"] == DRAFT_2020_12_URI


def test_determinism_repeats_below_two_is_usage_error(capsys):
    """MINOR-1: `determinism --repeats 1` is a user input mistake (a floor needs
    >= 2 renders), not an internal bug. The CLI-seam guard fires BEFORE any render
    so it maps to the USAGE exit code (1) with the ``USAGE_REPEATS_TOO_LOW`` code
    via the fatal envelope — never the generic INTERNAL_ERROR path. No plugin /
    Surge needed because the guard precedes all rendering."""
    code = cli.main(["determinism", "--repeats", "1"])
    assert code == int(ExitCode.USAGE)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "USAGE_REPEATS_TOO_LOW"


def test_analyze_wav_emits_report_array(tmp_path, capsys):
    """`analyze --wav` analyzes a standalone wav and emits the wav-analysis JSON
    array (exit 0) — the deferral guard is gone. A synthetic tmp wav is used, so no
    plugin/model is required (keeps this a non-integration test)."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "tone.wav"
    sf.write(
        str(wav),
        np.linspace(-0.4, 0.4, 96000, dtype=np.float32),
        44100,
        subtype="PCM_16",
    )
    code = cli.main(["analyze", "--wav", str(wav)])
    assert code == int(ExitCode.OK)
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["kind"] == "wav-chunk-analysis"
    assert payload[0]["schema_version"] == "1.4.0"
    assert payload[0]["input_provenance"]["original_sample_rate"] == 44100


def test_corpus_verify_failure_exits_nonzero(monkeypatch, capsys):
    """A corpus-integrity failure maps to a nonzero (INPUT) exit; the failure is
    surfaced in the fatal envelope, never a silent pass."""
    failing = VerifyResult(
        ok=False,
        items=(
            ItemVerification(
                name="tone",
                path="signals/tone_1k_2s.wav",
                expected_sha256="a" * 64,
                actual_sha256="b" * 64,
                ok=False,
                reason="hash-mismatch",
            ),
        ),
    )
    monkeypatch.setattr(cli, "corpus_verify", lambda: failing)
    code = cli.main(["corpus", "verify"])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["component"] == "corpus"


def test_corpus_verify_clean_exits_zero(monkeypatch, capsys):
    """A clean corpus verifies to exit 0 with an ok=True JSON report."""
    clean = VerifyResult(
        ok=True,
        items=(
            ItemVerification(
                name="tone",
                path="signals/tone_1k_2s.wav",
                expected_sha256="a" * 64,
                actual_sha256="a" * 64,
                ok=True,
                reason=None,
            ),
        ),
    )
    monkeypatch.setattr(cli, "corpus_verify", lambda: clean)
    code = cli.main(["corpus", "verify"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_probe_unavailable_reports_graceful(monkeypatch, capsys, tmp_path):
    """When the perception adapter is unavailable, `probe` reports the unavailable
    status and exits 0 (advisory, never fatal)."""
    monkeypatch.setattr(cli, "_probe_adapter", lambda _args: NullAdapter())
    code = cli.main(["probe", "--fixtures", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "unavailable"


# --- FIX 1 (dogfood): iterate --direction stable is reachable at the CLI --------


@pytest.mark.parametrize("direction", get_args(IterateDirection))
def test_iterate_every_direction_choice_dispatches(monkeypatch, direction):
    """FIX 1 (RED-proving): every ``IterateDirection`` literal value — INCLUDING
    ``stable`` — is accepted by the ``iterate`` argparse ``--direction`` choices and
    dispatches to the engine (which is stubbed here, so no Surge/render is needed).

    RED against the pre-fix hardcoded ``choices=("increase","decrease","change")``:
    ``iterate --direction stable`` raised a USAGE argparse error (exit 1) at
    ``parse_args`` and NEVER reached the engine, even though iterate.py fully
    implements the ``stable`` verdict. The three movement directions must still work.
    """
    captured: dict[str, str] = {}

    def _stub_iterate(args):
        captured["direction"] = args.direction
        return int(ExitCode.OK)

    monkeypatch.setitem(cli._HANDLERS, "iterate", _stub_iterate)
    code = cli.main(
        [
            "iterate",
            "--baseline",
            "b.json",
            "--candidate",
            "c.json",
            "--metric",
            "deterministic.summary.spectral_centroid_hz",
            "--direction",
            direction,
            "--plugin",
            "Surge XT.vst3",
        ]
    )
    assert code == int(ExitCode.OK)
    assert captured["direction"] == direction


# --- FIX 2 (dogfood): a nonexistent --plugin path is a typed InputError ----------


def test_analyze_nonexistent_plugin_path_is_input_error(capsys):
    """FIX 2 (RED-proving): ``analyze --plugin <missing> --spec <valid spec>`` maps a
    nonexistent plugin path to a typed INPUT error (exit 2, ``PLUGIN_PATH_NOT_FOUND``)
    at the CLI seam, BEFORE any render.

    RED against the pre-fix unguarded path: a bare ``FileNotFoundError`` escaped to the
    generic handler and was misreported as ``INTERNAL_ERROR`` (exit 1). The spec is a
    real committed valid spec, so the ONLY fault is the plugin path.
    """
    valid_spec = cli._REPO_ROOT / "specs" / "surge_lowpass_open.json"
    code = cli.main(
        [
            "analyze",
            "--plugin",
            "/no/such.vst3",
            "--spec",
            str(valid_spec),
        ]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "fatal-error"
    assert payload["error"]["code"] == "PLUGIN_PATH_NOT_FOUND"
    assert payload["error"]["component"] == "analyze"


# --- FIX 3 (dogfood): absent probe fixtures -> clean typed error, not a crash ----


class _AvailableReadingAdapter:
    """An AVAILABLE perception adapter that READS the wav in ``describe`` (like the
    real QwenLocalAdapter). With the FIX 3 guard removed, a missing fixture would
    raise ``FileNotFoundError`` here (-> INTERNAL_ERROR exit 1); the guard turns that
    into the typed ``PROBE_FIXTURES_NOT_FOUND`` input error (exit 2)."""

    id = "stub"
    grounding = "advisory-freetext"

    def describe(self, wav_path, deterministic=None) -> PerceptionBlock:
        Path(wav_path).read_bytes()  # FileNotFoundError if the fixture is absent
        return PerceptionBlock(
            status="ok",
            grounding="advisory-freetext",
            adapter=AdapterInfo(
                id="stub", model="m", quant="none", runtime="test",
                model_sha256="0" * 64,
            ),
            description="a bright tone",
            grounding_map={},
            disclaimer="Advisory only.",
        )

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            available=True, runtime="test", model_id="m", reason=None
        )


def test_probe_absent_fixtures_is_typed_error(monkeypatch, capsys, tmp_path):
    """FIX 3 (RED-proving the crash->typed-error mapping): with an AVAILABLE model and
    an empty fixtures dir, ``probe`` emits a typed INPUT error (exit 2,
    ``PROBE_FIXTURES_NOT_FOUND``) — never a raw ``FileNotFoundError`` INTERNAL_ERROR."""
    monkeypatch.setattr(
        cli, "_probe_adapter", lambda _args: _AvailableReadingAdapter()
    )
    code = cli.main(["probe", "--fixtures", str(tmp_path)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "fatal-error"
    assert payload["error"]["code"] == "PROBE_FIXTURES_NOT_FOUND"
    assert payload["error"]["component"] == "perception"


@pytest.mark.integration
def test_probe_runs_with_default_committed_fixtures(qwen_model, capsys):
    """FIX 3 (integration, model-gated): with the real Qwen model present, ``probe``
    runs out of the box against the committed default fixture set (no ``--fixtures``
    needed) and exits 0 with an ok status over all 5 A/B pairs. Skips when the model
    weights are absent (``qwen_model`` fixture, explicit reason)."""
    code = cli.main(["probe"])
    assert code == int(ExitCode.OK)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "probe"
    assert payload["status"] == "ok"
    assert payload["m_total"] == 5


# --- Item C (dogfood): iterate --brief reduced output ------------------------


def test_iterate_brief_omits_embedded_reports(monkeypatch, capsys):
    """Item C (RED-proving): ``iterate ... --brief`` emits a reduced object (verdict +
    delta + expectation) and OMITS the two embedded ~12KB baseline/candidate
    AnalysisReports, while the default (no --brief) emits the full IterateDelta that
    DOES embed them.

    A fake engine drives a REAL IterateDelta so no Surge/model is needed: the plugin +
    spec loads are stubbed, ``analyze_plugin_spec`` returns hand-authored reports, and a
    cached floor is returned so the measure path is skipped. The verdict is a genuine
    ``run_iterate`` PASS (1000 -> 200 Hz decrease, floor 15 Hz)."""
    from tests.test_iterate import METRIC, _floors, _report

    reports = iter([_report(1000.0), _report(200.0), _report(1000.0), _report(200.0)])
    monkeypatch.setattr(cli, "_require_plugin", lambda plugin, **k: Path("Surge XT.vst3"))
    monkeypatch.setattr(
        cli, "_load_spec", lambda *a, **k: (SimpleNamespace(patch_class="noisy"), "sha")
    )
    monkeypatch.setattr(cli, "analyze_plugin_spec", lambda *a, **k: next(reports))
    monkeypatch.setattr(cli, "read_floors", lambda *a, **k: _floors(15.0))

    argv = [
        "iterate", "--plugin", "Surge XT.vst3", "--baseline", "b.json",
        "--candidate", "c.json", "--metric", METRIC, "--direction", "decrease",
    ]

    # Default (full) DOES embed the baseline + candidate reports.
    assert cli.main(argv) == int(ExitCode.OK)
    full = json.loads(capsys.readouterr().out)
    assert full["kind"] == "iterate-delta"
    assert "baseline" in full
    assert "candidate" in full
    assert full["verdict"] == "PASS"

    # --brief emits ONLY verdict + delta + expectation, no embedded reports.
    assert cli.main(argv + ["--brief"]) == int(ExitCode.OK)
    brief = json.loads(capsys.readouterr().out)
    assert set(brief.keys()) == {"verdict", "delta", "expectation"}
    assert "baseline" not in brief
    assert "candidate" not in brief
    assert brief["verdict"] == "PASS"
    assert brief["delta"]["matches_expectation"] is True
    assert brief["delta"]["abs_delta"] == -800.0
    assert brief["expectation"]["metric"] == METRIC


# --- Finding 4 (Gemini review, final batch): render_dir cleanup on failure ----


def _fake_render_meta() -> RenderMeta:
    """Minimal backend-owned RenderMeta for the render-outcome fakes below."""
    return RenderMeta(
        sample_rate_hz=48000,
        block_size=512,
        channels=1,
        duration_s=0.1,
        wav_subtype="PCM_F32",
        wav_sha256="0" * 64,
        render_wall_ms=1,
    )


def _patch_render_prereqs(monkeypatch, render_impl):
    """Bypass spec load + backend construction; install ``render_impl`` as the
    orchestrator's ``render`` so ``_run_render`` reaches it without a real plugin,
    spec, or Surge XT."""
    monkeypatch.setattr(cli, "_load_spec", lambda *a, **k: (SimpleNamespace(), "sha"))
    monkeypatch.setattr(cli, "_backend", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli.render_orchestrator, "render", render_impl)


def _record_render_dirs(monkeypatch, tmp_path):
    """Route ``_run_render``'s ``mkdtemp`` under ``tmp_path`` and record every dir
    it creates, so a leak (dir still present after the command) is detectable."""
    created: list[Path] = []
    real_mkdtemp = cli.tempfile.mkdtemp

    def _recording_mkdtemp(*args, **kwargs):
        kwargs.setdefault("dir", str(tmp_path))
        made = real_mkdtemp(*args, **kwargs)
        created.append(Path(made))
        return made

    monkeypatch.setattr(cli.tempfile, "mkdtemp", _recording_mkdtemp)
    return created


def test_render_out_cleans_render_dir_on_failure(monkeypatch, tmp_path):
    """Finding 4 (RED-proving): ``render --out <dir>`` where the orchestrator's
    ``render`` raises must NOT leak the transient render dir on disk, and must
    surface the mapped RENDER exit code.

    RED against the pre-fix no-finally code: the ``shutil.rmtree(render_dir)`` lived
    INSIDE the ``if args.out:`` block AFTER the copy, so a raising render never
    reached it and the mkdtemp'd dir leaked — the ``not d.exists()`` assertion
    would fail."""
    created = _record_render_dirs(monkeypatch, tmp_path)

    def _boom(*_a, **_k):
        raise RenderError(
            "RENDER_SUBPROCESS_CRASH", "render blew up", component="render"
        )

    _patch_render_prereqs(monkeypatch, _boom)

    # FIX 2: ``_require_plugin`` now validates the plugin path exists at the CLI
    # seam, so this test supplies a real (empty) plugin file to reach the render.
    plugin = tmp_path / "plugin.vst3"
    plugin.write_bytes(b"")
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "render",
            "--plugin",
            str(plugin),
            "--spec",
            "/fake/spec.json",
            "--out",
            str(out_dir),
        ]
    )

    # The failure surfaces as the mapped RENDER exit code.
    assert code == int(ExitCode.RENDER)
    # The transient render dir was created and then reclaimed (no on-disk leak).
    assert created
    assert all(not d.exists() for d in created)


def test_render_without_out_keeps_wav_dir(monkeypatch, tmp_path, capsys):
    """Finding 4 (behavior-preservation): with NO ``--out`` the rendered wav IS the
    deliverable left in the render dir for the caller to read; that dir MUST be
    kept (only its path is printed), never cleaned by the new try/finally."""
    created = _record_render_dirs(monkeypatch, tmp_path)

    def _ok(spec, plugin, backend, **_k):
        # Place the wav inside the mkdtemp'd render dir (the last recorded one).
        render_dir = created[-1]
        wav_path = render_dir / "render.wav"
        wav_path.write_bytes(b"RIFF")
        return SimpleNamespace(
            wav_path=wav_path,
            backend="fake",
            backend_version="0.0.0",
            ref_sha256=None,
            seed=None,
            render_meta=_fake_render_meta(),
        )

    _patch_render_prereqs(monkeypatch, _ok)

    # FIX 2: supply a real plugin path so ``_require_plugin``'s existence check passes.
    plugin = tmp_path / "plugin.vst3"
    plugin.write_bytes(b"")
    code = cli.main(
        ["render", "--plugin", str(plugin), "--spec", "/fake/spec.json"]
    )

    assert code == int(ExitCode.OK)
    # The render dir is KEPT (the wav is the deliverable) — no cleanup on this path.
    assert created
    render_dir = created[-1]
    assert render_dir.exists()
    # The printed wav_path is the deliverable inside the kept render dir.
    payload = json.loads(capsys.readouterr().out)
    assert payload["wav_path"] == str(render_dir / "render.wav")
    assert Path(payload["wav_path"]).exists()
