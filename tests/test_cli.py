"""Tests for the CLI skeleton: argument parsing, unknown-command usage error,
and the fatal-envelope -> exit-code path (design sections 3.6/4.5; plan C5)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from sonoscope import cli
from sonoscope.backends.base import PluginInfo, RenderMeta, RenderRequest, RenderResult
from sonoscope.errors import RenderError
from sonoscope.schema import ExitCode


def test_iterate_args_parse_to_namespace():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "iterate",
            "--baseline",
            "base.json",
            "--candidate",
            "cand.json",
            "--metric",
            "deterministic.summary.spectral_centroid_hz",
            "--direction",
            "decrease",
            "--min-effect",
            "50",
            "--plugin",
            "Surge XT.vst3",
        ]
    )
    assert vars(args) == {
        "command": "iterate",
        # global (parent) flag
        "json": False,
        # render-spec flags (item D: now scoped to render/analyze/iterate/determinism)
        "sample_rate": None,
        "block_size": None,
        "channels": None,
        "seed": None,
        # iterate-specific
        "baseline": "base.json",
        "candidate": "cand.json",
        "metric": "deterministic.summary.spectral_centroid_hz",
        "direction": "decrease",
        "min_effect": 50.0,
        "plugin": "Surge XT.vst3",
        # item C: --brief flag (default off)
        "brief": False,
    }


def test_unknown_command_exits_usage(capsys):
    code = cli.main(["frobnicate"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    # `message` is argparse-generated (version-dependent); every other field
    # of the fatal envelope is asserted by exact equality.
    assert payload["kind"] == "fatal-error"
    assert payload["schema_version"] == "1.5.0"  # SCHEMA_VERSION bump (all_channels_silent)
    assert payload["error"]["code"] == "USAGE_INVALID_ARGS"
    assert payload["error"]["severity"] == "fatal"
    assert payload["error"]["component"] == "cli"


def test_fatal_envelope_shape(monkeypatch, capsys):
    def boom(_args):
        raise RenderError(
            "RENDER_SUBPROCESS_CRASH",
            "boom",
            detail={"signal": "SIGSEGV"},
            component="render",
        )

    monkeypatch.setitem(cli._HANDLERS, "render", boom)
    monkeypatch.setattr(cli, "_now_iso", lambda: "2026-07-04T12:00:00Z")

    code = cli.main(["render", "--plugin", "x.vst3", "--spec", "s.json"])
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": "1.5.0",  # SCHEMA_VERSION bump (all_channels_silent)
        "kind": "fatal-error",
        "generated_at": "2026-07-04T12:00:00Z",
        "sonoscope_version": "0.1.0",
        "error": {
            "code": "RENDER_SUBPROCESS_CRASH",
            "message": "boom",
            "detail": {"signal": "SIGSEGV"},
            "severity": "fatal",
            "component": "render",
        },
    }


def test_analyze_requires_one_input_both():
    # --wav and --plugin are mutually exclusive: giving both is a usage error.
    assert cli.main(["analyze", "--wav", "a.wav", "--plugin", "p.vst3"]) == 1


def test_analyze_requires_one_input_neither():
    # neither given: the required mutually-exclusive group errors -> usage.
    assert cli.main(["analyze"]) == 1


# --- Item B (dogfood): top-level --version -----------------------------------


def test_version_flag_prints_version_and_exits_zero(capsys):
    """Item B (RED-proving): ``sonoscope --version`` prints the package version and
    exits 0 (argparse's 'version' action raises SystemExit(0)), instead of the
    pre-change USAGE exit 1 (no such flag / required subcommand missing)."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sonoscope {cli.__version__}"


# --- Item D (dogfood): global flags scoped to their consuming commands --------


def test_doctor_rejects_scoped_perception_flag():
    """Item D (RED-proving): ``--perception`` is scoped to 'analyze', so
    ``doctor --perception`` is now an argparse USAGE error (exit 1) rather than a
    silently-ignored flag."""
    assert cli.main(["doctor", "--perception"]) == 1


def test_corpus_rejects_scoped_adapter_flag():
    """Item D (RED-proving): ``--adapter`` is scoped to analyze/probe, so
    ``corpus verify --adapter null`` is a USAGE error (exit 1)."""
    assert cli.main(["corpus", "verify", "--adapter", "null"]) == 1


def test_schema_rejects_scoped_seed_flag():
    """Item D (RED-proving): ``--seed`` is scoped to the render-family commands, so
    ``schema --seed`` (the exact case cited in the dogfood finding) is a USAGE error
    (exit 1), not a silently-ignored flag."""
    assert cli.main(["schema", "--kind", "analysis", "--seed", "7"]) == 1


def test_render_rejects_scoped_perception_flag():
    """Item D: ``--perception`` does not leak onto 'render' (render has no perception
    pass); passing it is a USAGE error (exit 1)."""
    assert cli.main(["render", "--plugin", "p.vst3", "--spec", "s.json",
                     "--perception"]) == 1


def test_analyze_still_accepts_perception_and_adapter():
    """Item D (GREEN): the consuming command 'analyze' still accepts the flags whose
    handler reads them."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["analyze", "--plugin", "p.vst3", "--spec", "s.json",
         "--perception", "--adapter", "null"]
    )
    assert args.perception is True
    assert args.adapter == "null"


def test_probe_still_accepts_adapter():
    """Item D (GREEN): 'probe' still accepts --adapter (its handler reads it)."""
    parser = cli.build_parser()
    args = parser.parse_args(["probe", "--adapter", "null"])
    assert args.adapter == "null"


def test_render_still_accepts_out_and_render_spec_flags():
    """Item D (GREEN): 'render' still accepts --out (consumed) and the render-spec
    flags now scoped onto it."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["render", "--plugin", "p.vst3", "--spec", "s.json", "--out", "o.wav",
         "--sample-rate", "48000", "--block-size", "512", "--channels", "2",
         "--seed", "7"]
    )
    assert args.out == "o.wav"
    assert args.sample_rate == 48000
    assert args.block_size == 512
    assert args.channels == 2
    assert args.seed == 7


def test_schema_still_accepts_out():
    """Item D (GREEN): 'schema' still accepts --out (its handler writes schema files)."""
    parser = cli.build_parser()
    args = parser.parse_args(["schema", "--kind", "analysis", "--out", "schemas/"])
    assert args.out == "schemas/"


def test_determinism_still_accepts_render_spec_flags():
    """Item D (GREEN): 'determinism' still accepts the render-spec flags scoped onto it."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["determinism", "--plugin", "p.vst3", "--spec", "s.json", "--seed", "3"]
    )
    assert args.seed == 3


# --- Item E (dogfood): the dead --schema-version-check flag is removed --------


def test_schema_version_check_flag_removed():
    """Item E (RED-proving): the dead ``--schema-version-check`` flag (never read in
    ``src/``) is gone — passing it is a USAGE error (exit 1) and it is absent from the
    parsed namespace of a command that used to inherit it."""
    assert cli.main(["analyze", "--plugin", "p.vst3", "--spec", "s.json",
                     "--schema-version-check"]) == 1
    parser = cli.build_parser()
    args = parser.parse_args(["analyze", "--plugin", "p.vst3", "--spec", "s.json"])
    assert not hasattr(args, "schema_version_check")


# --- C2: the analyze-midi subcommand -----------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
#: A clean corpus ``.mid`` (4 note pairs, explicit note_off) for the file source.
CORPUS_MID = _REPO_ROOT / "corpus" / "midi" / "phrase_4note.mid"


class _FakeMidiReport:
    """A minimal stand-in for a MidiAnalysisReport with a controllable verdict.

    The C2 handler only reads ``.midi.verdict`` and calls ``.model_dump_json()``,
    so a fake keeps the RED-path / kwarg-forwarding tests free of a real capture.
    """

    def __init__(self, verdict: str) -> None:
        self.midi = types.SimpleNamespace(verdict=verdict)

    def model_dump_json(self) -> str:
        return json.dumps({"kind": "midi-analysis", "verdict": self.midi.verdict})


def test_analyze_midi_missing_source_is_input_error(capsys):
    """Neither --plugin nor --file -> typed InputError (exit 2), clear message."""
    code = cli.main(["analyze-midi"])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SOURCE_MISSING"
    assert payload["error"]["component"] == "midi"


def test_analyze_midi_both_sources_is_input_error(capsys):
    """Both --plugin and --file -> mutually-exclusive InputError (exit 2)."""
    code = cli.main(
        ["analyze-midi", "--plugin", "p.clap", "--file", str(CORPUS_MID)]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SOURCE_CONFLICT"


def test_analyze_midi_plugin_without_spec_is_input_error(tmp_path, capsys):
    """--plugin (valid path) but no --spec -> InputError (exit 2)."""
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")  # exists, so _require_plugin passes to the spec gate
    code = cli.main(["analyze-midi", "--plugin", str(plugin)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_MISSING"


def test_analyze_midi_bad_plugin_path_is_input_error(capsys):
    """A supplied-but-nonexistent --plugin path -> PLUGIN_PATH_NOT_FOUND (exit 2)."""
    code = cli.main(
        ["analyze-midi", "--plugin", "/no/such/ReferenceSequencer.clap", "--spec", "s.json"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "PLUGIN_PATH_NOT_FOUND"


def test_analyze_midi_file_source_prints_report(capsys):
    """--file (a real corpus .mid) -> a valid midi-analysis report, exit 0."""
    code = cli.main(["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000"])
    assert code == int(ExitCode.OK)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "midi-analysis"
    assert payload["input"]["source"] == "file"
    assert payload["midi"]["verdict"] == "PASS"


def test_analyze_midi_file_source_requires_sample_rate(capsys):
    """--file without --sample-rate -> InputError (exit 2)."""
    code = cli.main(["analyze-midi", "--file", str(CORPUS_MID)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_FILE_SAMPLE_RATE_MISSING"


def test_analyze_midi_fail_on_red_exits_nonzero(monkeypatch, capsys):
    """--fail-on-red maps a RED verdict to the ANALYSIS exit code; the report prints."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("RED"))
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000",
         "--fail-on-red"]
    )
    assert code == int(ExitCode.ANALYSIS)
    assert code != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "midi-analysis"
    assert payload["verdict"] == "RED"


def test_analyze_midi_red_without_flag_exits_zero(monkeypatch, capsys):
    """Without --fail-on-red a RED report still prints and exits 0."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("RED"))
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000"]
    )
    assert code == int(ExitCode.OK)
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "RED"


def test_analyze_midi_slice_and_overrides_forwarded(monkeypatch):
    """--offset/--length/--unit, --expected and --offvel0 reach analyze_midi."""
    captured: dict = {}

    def fake_analyze(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return _FakeMidiReport("PASS")

    monkeypatch.setattr(cli, "analyze_midi", fake_analyze)
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000",
         "--offset", "6000", "--length", "6000", "--unit", "samples",
         "--expected", "golden.json", "--offvel0", "normalize"]
    )
    assert code == int(ExitCode.OK)
    slice_spec = captured["slice_spec"]
    assert slice_spec.offset == 6000
    assert slice_spec.length == 6000
    assert slice_spec.unit == "samples"
    assert slice_spec.rebase is True
    assert captured["expected"] == "golden.json"
    assert captured["offvel0_policy"] == "normalize"


def test_analyze_midi_incoherent_slice_is_input_error(monkeypatch, capsys):
    """--length without --offset is an incoherent slice -> InputError (exit 2)."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000",
         "--length", "6000", "--unit", "samples"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SLICE_INCOHERENT"


def test_analyze_midi_plugin_spec_builds_request(tmp_path, monkeypatch):
    """A --plugin + --spec run builds a MidiCaptureRequest from the spec fields."""
    from sonoscope.backends.midi_capture import MidiCaptureRequest

    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    spec.write_text(
        json.dumps(
            {
                "tempo_bpm": 120.0,
                "start_position_beats": 0.0,
                "duration_beats": 1.0,
                "tsig_num": 4,
                "tsig_den": 4,
                "sample_rate": 48000,
                "block_size": 512,
            }
        )
    )
    captured: dict = {}

    def fake_analyze(source, **kwargs):
        captured["source"] = source
        return _FakeMidiReport("PASS")

    monkeypatch.setattr(cli, "analyze_midi", fake_analyze)
    code = cli.main(
        ["analyze-midi", "--plugin", str(plugin), "--spec", str(spec),
         "--plugin-id", "com.example.reference-sequencer"]
    )
    assert code == int(ExitCode.OK)
    request = captured["source"]
    assert isinstance(request, MidiCaptureRequest)
    assert request.plugin_path == plugin
    assert request.tempo_bpm == 120.0
    assert request.sample_rate == 48000
    assert request.block_size == 512
    assert request.plugin_id == "com.example.reference-sequencer"


# --- C2 hardening: cross-source flag rejection + strict spec typing -----------


def _valid_capture_spec() -> dict:
    """The 7 required transport/render fields of a valid --plugin capture spec."""
    return {
        "tempo_bpm": 120.0,
        "start_position_beats": 0.0,
        "duration_beats": 1.0,
        "tsig_num": 4,
        "tsig_den": 4,
        "sample_rate": 48000,
        "block_size": 512,
    }


# Fix 1 (cross-source flags silently ignored): --spec/--plugin-id with --file, and
# --sample-rate/--tempo with --plugin, used to be silently discarded. They must now
# be a typed InputError (exit 2, INPUT_MIDI_FLAG_CONFLICT) — mirrors the both/neither
# source rigor (conflicting input is a typed error, never silent).


def test_analyze_midi_plugin_with_sample_rate_is_flag_conflict(
    tmp_path, monkeypatch, capsys
):
    """--sample-rate belongs to the --file source; supplied WITH --plugin it must be
    a typed InputError, never silently dropped (the transport comes from --spec)."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    spec.write_text(json.dumps(_valid_capture_spec()))
    code = cli.main(
        ["analyze-midi", "--plugin", str(plugin), "--spec", str(spec),
         "--sample-rate", "44100"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_FLAG_CONFLICT"
    assert payload["error"]["component"] == "midi"


def test_analyze_midi_plugin_with_tempo_is_flag_conflict(
    tmp_path, monkeypatch, capsys
):
    """--tempo belongs to the --file source; supplied WITH --plugin -> InputError."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    spec.write_text(json.dumps(_valid_capture_spec()))
    code = cli.main(
        ["analyze-midi", "--plugin", str(plugin), "--spec", str(spec),
         "--tempo", "140"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_FLAG_CONFLICT"


def test_analyze_midi_file_with_spec_is_flag_conflict(monkeypatch, capsys):
    """--spec belongs to the --plugin source; supplied WITH --file -> InputError."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000",
         "--spec", "capture.json"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_FLAG_CONFLICT"


def test_analyze_midi_file_with_plugin_id_is_flag_conflict(monkeypatch, capsys):
    """--plugin-id belongs to the --plugin source; supplied WITH --file -> InputError."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    code = cli.main(
        ["analyze-midi", "--file", str(CORPUS_MID), "--sample-rate", "48000",
         "--plugin-id", "com.example.reference-sequencer"]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_FLAG_CONFLICT"


# Fix 2 (loose type coercion in _load_midi_capture_request): a mistyped spec value
# must be a typed InputError, not a silent truthy/None coercion.


def test_analyze_midi_spec_playing_string_is_invalid(tmp_path, monkeypatch, capsys):
    """playing:"false" (a JSON string) must be rejected, NOT coerced truthy to True."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    data = _valid_capture_spec()
    data["playing"] = "false"
    spec.write_text(json.dumps(data))
    code = cli.main(["analyze-midi", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_INVALID"


def test_analyze_midi_spec_plugin_id_nonstring_is_invalid(
    tmp_path, monkeypatch, capsys
):
    """A non-string plugin_id (int 123) must be rejected, NOT silently coerced to None."""
    monkeypatch.setattr(cli, "analyze_midi", lambda *a, **k: _FakeMidiReport("PASS"))
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    data = _valid_capture_spec()
    data["plugin_id"] = 123
    spec.write_text(json.dumps(data))
    code = cli.main(["analyze-midi", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_INVALID"


# Fix 3 (spec-loader error-branch coverage): the _load_midi_capture_request failure
# branches had zero CLI coverage. These assert each maps to its exact typed code.


def test_analyze_midi_spec_unreadable_is_input_error(tmp_path, capsys):
    """A missing/unreadable --spec file -> INPUT_MIDI_SPEC_UNREADABLE (exit 2)."""
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    code = cli.main(
        ["analyze-midi", "--plugin", str(plugin), "--spec",
         str(tmp_path / "nope.json")]
    )
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_UNREADABLE"


def test_analyze_midi_spec_not_json_is_input_error(tmp_path, capsys):
    """A non-JSON --spec body -> INPUT_MIDI_SPEC_INVALID (exit 2)."""
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    spec.write_text("this is not json {")
    code = cli.main(["analyze-midi", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_INVALID"


def test_analyze_midi_spec_non_object_is_input_error(tmp_path, capsys):
    """A JSON array (non-object) --spec body -> INPUT_MIDI_SPEC_INVALID (exit 2)."""
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    spec.write_text(json.dumps([1, 2, 3]))
    code = cli.main(["analyze-midi", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_INVALID"


def test_analyze_midi_spec_missing_required_field_is_input_error(tmp_path, capsys):
    """A spec missing a required field (tempo_bpm) -> INPUT_MIDI_SPEC_INVALID (exit 2)."""
    plugin = tmp_path / "ReferenceSequencer.clap"
    plugin.write_bytes(b"fake")
    spec = tmp_path / "capture.json"
    data = _valid_capture_spec()
    del data["tempo_bpm"]
    spec.write_text(json.dumps(data))
    code = cli.main(["analyze-midi", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INPUT_MIDI_SPEC_INVALID"


# --- render-override flags: --sample-rate/--block-size/--channels/--seed ------
# These four render-family flags (added by _add_render_spec_flags) now OVERRIDE
# the spec's render.<field> when constructing the RenderRequest. The RED-proving
# tests below drive the real CLI handler end to end and capture the RenderRequest
# that render_orchestrator builds, so a flag that is merely parsed-but-ignored
# fails the override assertion (the request keeps the spec value).

#: Spec render values the override tests assert AGAINST (each flag must beat one).
_SPEC_SAMPLE_RATE = 48000
_SPEC_BLOCK_SIZE = 512
_SPEC_CHANNELS = 2
_SPEC_SEED = 99


def _instrument_spec_file(tmp_path: Path, name: str = "spec.json") -> Path:
    """Write a valid inline-notes instrument spec with explicit render params.

    Inline notes route to the MIDI stimulus path with no corpus ref, so the
    override tests need no corpus/network — only the render() precedence logic.
    """
    data = {
        "stimulus": {
            "kind": "midi",
            "notes": [{"pitch": 60, "vel": 100, "on": 0.0, "off": 1.0}],
        },
        "render": {
            "sample_rate_hz": _SPEC_SAMPLE_RATE,
            "block_size": _SPEC_BLOCK_SIZE,
            "channels": _SPEC_CHANNELS,
            "seed": _SPEC_SEED,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _fake_instrument_info() -> PluginInfo:
    """An instrument PluginInfo so render() routes to the inline-notes MIDI path."""
    return PluginInfo(
        name="FakeSynth",
        format="vst3",
        params=[],
        input_channels=0,
        output_channels=1,
        is_instrument=True,
        needs_gui=False,
        latency_samples=0,
    )


class _ProbeOnlyBackend:
    """A backend whose probe reports an instrument; render is never called here
    because render_orchestrator.render_in_subprocess is monkeypatched to capture
    the constructed RenderRequest instead of spawning a subprocess."""

    id = "fake-capture"
    version = "0.0.0"

    def probe(self, plugin_path: Path) -> PluginInfo:
        return _fake_instrument_info()

    def render(self, req: RenderRequest) -> RenderResult:  # pragma: no cover
        raise AssertionError("render_in_subprocess is monkeypatched in these tests")


def _capture_render_requests(monkeypatch) -> list[RenderRequest]:
    """Route render() through a fake backend + capture the RenderRequest it builds.

    Replaces the CLI backend factory with a probe-only fake and intercepts
    render_orchestrator.render_in_subprocess so the real precedence logic in
    render() runs (probe -> resolve -> build_stimulus -> RenderRequest) without a
    subprocess/plugin. Returns the list of captured requests (one per render)."""
    reqs: list[RenderRequest] = []

    def fake_subproc(backend, req: RenderRequest) -> RenderResult:
        reqs.append(req)
        meta = RenderMeta(
            sample_rate_hz=req.sample_rate_hz,
            block_size=req.block_size,
            channels=req.channels,
            duration_s=0.1,
            wav_subtype="PCM_F32",
            wav_sha256="0" * 64,
            render_wall_ms=0,
            warnings=[],
        )
        return RenderResult(
            wav_path=Path("/tmp/fake-capture.wav"), render_meta=meta, warnings=[]
        )

    monkeypatch.setattr(cli, "_backend", lambda render_dir=None: _ProbeOnlyBackend())
    monkeypatch.setattr(
        cli.render_orchestrator, "render_in_subprocess", fake_subproc
    )
    return reqs


def test_render_sample_rate_flag_overrides_spec(tmp_path, monkeypatch, capsys):
    """RED->GREEN: --sample-rate 44100 beats the spec's render.sample_rate_hz=48000
    in the RenderRequest render() builds (before wiring the flag was ignored and
    the request kept 48000)."""
    reqs = _capture_render_requests(monkeypatch)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["render", "--plugin", str(plugin), "--spec", str(spec),
         "--sample-rate", "44100"]
    )
    assert code == int(ExitCode.OK)
    assert reqs[0].sample_rate_hz == 44100  # override reached the request
    # untouched fields fall back to the spec values.
    assert reqs[0].block_size == _SPEC_BLOCK_SIZE
    assert reqs[0].channels == _SPEC_CHANNELS
    assert reqs[0].seed == _SPEC_SEED


def test_render_block_size_flag_overrides_spec(tmp_path, monkeypatch, capsys):
    """RED->GREEN: --block-size 1024 beats the spec's render.block_size=512."""
    reqs = _capture_render_requests(monkeypatch)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["render", "--plugin", str(plugin), "--spec", str(spec),
         "--block-size", "1024"]
    )
    assert code == int(ExitCode.OK)
    assert reqs[0].block_size == 1024
    assert reqs[0].sample_rate_hz == _SPEC_SAMPLE_RATE


def test_render_channels_flag_overrides_spec(tmp_path, monkeypatch, capsys):
    """RED->GREEN: --channels 1 beats the spec's render.channels=2."""
    reqs = _capture_render_requests(monkeypatch)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["render", "--plugin", str(plugin), "--spec", str(spec),
         "--channels", "1"]
    )
    assert code == int(ExitCode.OK)
    assert reqs[0].channels == 1
    assert reqs[0].sample_rate_hz == _SPEC_SAMPLE_RATE


def test_render_seed_flag_overrides_spec(tmp_path, monkeypatch, capsys):
    """RED->GREEN: --seed 7 beats the spec's render.seed=99 (and is echoed on the
    RenderOutcome.seed printed by the handler)."""
    reqs = _capture_render_requests(monkeypatch)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["render", "--plugin", str(plugin), "--spec", str(spec), "--seed", "7"]
    )
    assert code == int(ExitCode.OK)
    assert reqs[0].seed == 7
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed"] == 7  # the forwarded seed reflects the override


def test_render_no_flags_uses_spec_values(tmp_path, monkeypatch, capsys):
    """Precedence GREEN: with no override flags, every RenderRequest field falls
    back to the spec's render values (unchanged behavior)."""
    reqs = _capture_render_requests(monkeypatch)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(["render", "--plugin", str(plugin), "--spec", str(spec)])
    assert code == int(ExitCode.OK)
    assert reqs[0].sample_rate_hz == _SPEC_SAMPLE_RATE
    assert reqs[0].block_size == _SPEC_BLOCK_SIZE
    assert reqs[0].channels == _SPEC_CHANNELS
    assert reqs[0].seed == _SPEC_SEED


def test_analyze_forwards_render_overrides(tmp_path, monkeypatch):
    """Forwarding smoke: the analyze handler forwards all four overrides into
    analyze_plugin_spec (which forwards them to render() via **render_kwargs)."""
    captured: dict = {}

    def fake_aps(spec, plugin_path, backend, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model_dump_json=lambda: "{}")

    monkeypatch.setattr(cli, "_backend", lambda render_dir=None: object())
    monkeypatch.setattr(cli, "analyze_plugin_spec", fake_aps)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["analyze", "--plugin", str(plugin), "--spec", str(spec),
         "--sample-rate", "44100", "--block-size", "1024",
         "--channels", "1", "--seed", "7"]
    )
    assert code == int(ExitCode.OK)
    assert captured["sample_rate_hz"] == 44100
    assert captured["block_size"] == 1024
    assert captured["channels"] == 1
    assert captured["seed"] == 7


def test_iterate_forwards_render_overrides(tmp_path, monkeypatch):
    """Forwarding smoke: the iterate handler forwards the overrides into BOTH the
    baseline and candidate analyze_plugin_spec calls."""
    captured: list[dict] = []
    fake_report = types.SimpleNamespace(
        input=types.SimpleNamespace(
            plugin=types.SimpleNamespace(binary_sha256="b" * 64)
        )
    )

    def fake_aps(spec, plugin_path, backend, **kwargs):
        captured.append(dict(kwargs))
        return fake_report

    monkeypatch.setattr(cli, "_backend", lambda render_dir=None: object())
    monkeypatch.setattr(cli, "analyze_plugin_spec", fake_aps)
    # A non-None floors skips the measure_floors branch; run_iterate is faked so
    # the handler completes and prints a delta.
    monkeypatch.setattr(cli, "read_floors", lambda binary, patch_class: object())
    monkeypatch.setattr(
        cli, "run_iterate",
        lambda *a, **k: types.SimpleNamespace(model_dump_json=lambda: "{}"),
    )
    baseline = _instrument_spec_file(tmp_path, name="base.json")
    candidate = _instrument_spec_file(tmp_path, name="cand.json")
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    code = cli.main(
        ["iterate", "--plugin", str(plugin), "--baseline", str(baseline),
         "--candidate", str(candidate),
         "--metric", "deterministic.summary.spectral_centroid_hz",
         "--direction", "decrease", "--channels", "1"]
    )
    assert code == int(ExitCode.OK)
    assert len(captured) == 2  # baseline + candidate
    assert captured[0]["channels"] == 1
    assert captured[1]["channels"] == 1


def test_determinism_forwards_render_overrides(tmp_path, monkeypatch, capsys):
    """Forwarding smoke: the determinism handler forwards the overrides into the
    seed render_orchestrator.render call."""
    captured: list[dict] = []

    def fake_render(spec, plugin, backend, **kwargs):
        captured.append(dict(kwargs))
        return types.SimpleNamespace(
            resolved=types.SimpleNamespace(
                resolved_sha256="r" * 64,
                stimulus=types.SimpleNamespace(ref="corpus/midi/x.mid"),
            )
        )

    monkeypatch.setattr(cli, "_backend", lambda render_dir=None: object())
    monkeypatch.setattr(cli.render_orchestrator, "render", fake_render)
    monkeypatch.setattr(cli, "binary_sha256", lambda plugin: "b" * 64)
    monkeypatch.setattr(
        cli, "measure_floors",
        lambda fn, **kw: types.SimpleNamespace(model_dump_json=lambda: "{}"),
    )
    monkeypatch.setattr(cli, "write_floors", lambda floors: None)
    plugin = tmp_path / "p.vst3"
    plugin.write_bytes(b"fake")
    spec = _instrument_spec_file(tmp_path)
    code = cli.main(
        ["determinism", "--plugin", str(plugin), "--spec", str(spec), "--seed", "7"]
    )
    assert code == int(ExitCode.OK)
    assert captured[0]["seed"] == 7


def test_render_override_flag_help_says_override(capsys):
    """The four flags' help now describes an OVERRIDE, dropping the old
    'reserved; not yet applied' wording."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["render", "--help"])
    help_text = capsys.readouterr().out
    assert "reserved; not yet applied" not in help_text
    assert "override" in help_text.lower()
