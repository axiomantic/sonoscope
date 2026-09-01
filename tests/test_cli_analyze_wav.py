"""CLI-level tests for ``analyze --wav`` (design §4, §11, §12).

Drive :func:`sonoscope.cli.main` with argv and capture stdout / stderr / exit code.
Synthetic wavs are written to ``tmp_path`` (no integration marker; keeps the
default ``-m "not integration"`` gate green).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonoscope import __version__
from sonoscope.cli import main
from sonoscope.schema import ExitCode


def _wav(tmp_path, sr=44100, n=96000, name="a.wav"):
    sf.write(
        str(tmp_path / name),
        np.linspace(-0.4, 0.4, n, dtype=np.float32),
        sr,
        subtype="PCM_16",
    )
    return tmp_path / name


def test_analyze_wav_emits_json_array(tmp_path, capsys):
    rc = main(["analyze", "--wav", str(_wav(tmp_path))])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == int(ExitCode.OK)
    assert isinstance(payload, list)
    assert len(payload) == 1
    chunk = payload[0]
    assert chunk["kind"] == "wav-chunk-analysis"
    assert chunk["schema_version"] == "1.5.0"
    assert chunk["sonoscope_version"] == __version__
    p = chunk["input_provenance"]
    assert p["original_sample_rate"] == 44100
    assert p["n_channels"] == 1
    assert p["source_subtype"] == "PCM_16"
    assert p["resample_res_type"] == "soxr_hq"
    assert p["soxr_version"] == "1.1.0"
    assert p["chunk_index"] == 0
    assert p["n_chunks"] == 1
    assert chunk["descriptors"] is not None
    assert chunk["descriptor_gate"] is None
    # Ungated run emits nothing on stderr (no aggregate line).
    assert captured.err == ""


def test_analyze_wav_unit_beats_is_argparse_usage_error(tmp_path):
    # "beats" is not in the AudioSliceUnit choices (samples/seconds) -> argparse
    # rejection -> USAGE exit (1), never a crash or a fabricated report.
    rc = main(
        ["analyze", "--wav", str(_wav(tmp_path)), "--offset", "0", "--unit", "beats"]
    )
    assert rc == int(ExitCode.USAGE)


def test_analyze_wav_rejects_plugin_flag(tmp_path, capsys):
    rc = main(["analyze", "--wav", str(_wav(tmp_path)), "--sample-rate", "48000"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == int(ExitCode.INPUT)
    assert payload["error"]["code"] == "INPUT_WAV_FLAG_CONFLICT"
    assert payload["error"]["component"] == "analyze"
    assert payload["error"]["detail"]["flags"] == ["--sample-rate"]


def test_analyze_plugin_rejects_wav_flag(capsys):
    # --offset is a wav-only slice flag; supplied with --plugin it is a typed
    # conflict (exit 2), rejected BEFORE any plugin-path resolution.
    rc = main(["analyze", "--plugin", "X.vst3", "--offset", "0"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == int(ExitCode.INPUT)
    assert payload["error"]["code"] == "INPUT_ANALYZE_FLAG_CONFLICT"
    assert payload["error"]["component"] == "analyze"
    assert payload["error"]["detail"]["flags"] == ["--offset"]


def test_analyze_wav_slice_incoherent(tmp_path, capsys):
    # --length without --offset is an incoherent slice (exit 2).
    rc = main(["analyze", "--wav", str(_wav(tmp_path)), "--length", "1"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == int(ExitCode.INPUT)
    assert payload["error"]["code"] == "INPUT_WAV_SLICE_INCOHERENT"
    assert payload["error"]["component"] == "analyze"


def test_analyze_wav_fail_on_red_requires_spec(tmp_path, capsys):
    rc = main(["analyze", "--wav", str(_wav(tmp_path)), "--fail-on-red"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == int(ExitCode.USAGE)
    assert payload["error"]["code"] == "USAGE_FAIL_ON_RED_REQUIRES_SPEC"


def test_analyze_wav_gate_stderr_shape(tmp_path, capsys):
    # An empty expectation spec requires nothing -> per-chunk PASS -> aggregate
    # GREEN with no red chunks and no reasons. The richer aggregate shape
    # (verdict / red_chunks / reasons) is emitted as one compact JSON line on
    # stderr, distinct from the plugin path's {verdict, reasons} shape.
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps({"expect_present": []}))
    rc = main(
        [
            "analyze",
            "--wav",
            str(_wav(tmp_path)),
            "--expect-descriptors",
            str(expect),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == int(ExitCode.OK)
    # The report still prints as the single JSON array on stdout, now with a
    # per-chunk PASS gate persisted.
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["descriptor_gate"]["verdict"] == "PASS"
    assert payload[0]["descriptor_gate"]["reasons"] == []
    # The aggregate line on stderr.
    assert captured.err == (
        '{"verdict":"GREEN","red_chunks":[],"reasons":[]}\n'
    )


def test_analyze_wav_fail_on_red_maps_to_analysis_exit(tmp_path, capsys):
    # The ramp signal never emits "bright" (see
    # test_wav_orchestrator.test_analyze_wav_per_chunk_gate_red_is_really_evaluated),
    # so requiring it forces a genuine per-chunk RED -> aggregate RED. With
    # --fail-on-red this MUST map to ExitCode.ANALYSIS (4), not the OK/USAGE/INPUT
    # codes used elsewhere in this file — if the exit-4 mapping regressed (e.g. the
    # `if args.fail_on_red and verdict == "RED"` check in _run_analyze_wav were
    # dropped or inverted), rc would stay 0 and this assertion would catch it.
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps({"expect_present": ["bright"]}))
    rc = main(
        [
            "analyze",
            "--wav",
            str(_wav(tmp_path)),
            "--expect-descriptors",
            str(expect),
            "--fail-on-red",
        ]
    )
    captured = capsys.readouterr()

    assert rc == int(ExitCode.ANALYSIS)
    # The report still prints in full on stdout (report-then-gate contract) as the
    # single JSON array, never swallowed by the gate failure.
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    # The aggregate verdict/red_chunks/reasons shape on stderr.
    err_payload = json.loads(captured.err)
    assert err_payload == {
        "verdict": "RED",
        "red_chunks": [0],
        "reasons": ["chunk[0] DESC_MISSING: bright"],
    }


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "wav"
    / "sine_44100_pcm16.wav"
)


@pytest.mark.integration
def test_analyze_wav_real_fixture_end_to_end(capsys):
    # Integration (design §15.3, m4): a committed real 44.1k PCM_16 wav, run
    # end-to-end through the CLI (no synthetic tmp_path fixture). Proves the
    # honest provenance stamp against sf.info(fixture) directly, not an
    # assumed value — this is the net-new source_subtype capture check.
    from sonoscope.schema.models import WavAnalysisReport

    rc = main(["analyze", "--wav", str(FIXTURE)])
    captured = capsys.readouterr()

    assert rc == int(ExitCode.OK)
    report = WavAnalysisReport.model_validate_json(captured.out)
    assert len(report.root) == 1
    p = report.root[0].input_provenance

    expected_subtype = sf.info(str(FIXTURE)).subtype  # net-new API verification
    assert expected_subtype == "PCM_16"
    assert p.source_subtype == expected_subtype
    assert p.original_sample_rate == 44100
    assert p.resample_res_type == "soxr_hq"
    assert p.soxr_version == "1.1.0"

    window = report.root[0].input_provenance.analyzed_window
    assert window.native_sample_rate == 44100
    audio, native_sr = sf.read(str(FIXTURE), dtype="float32", always_2d=True)
    n_native = audio.shape[0]
    assert window.native_offset_samples == 0
    assert window.native_length_samples == n_native

    # Realized 48k length: soxr's actual output, not a naive sample-rate-ratio
    # estimate (soxr may differ by ±1). Computed via the SAME resample call
    # wav_orchestrator._resample_chunk makes, so this catches provenance
    # mis-stamping the analyzed_samples_48k field independently of whatever
    # value the orchestrator happens to compute for itself.
    from sonoscope.wav_orchestrator import _resample_chunk

    chunk_native = np.ascontiguousarray(audio.T, dtype=np.float32)  # (ch, n)
    resampled, res_type, soxr_version = _resample_chunk(chunk_native, native_sr)
    assert res_type == "soxr_hq"
    assert soxr_version == "1.1.0"
    assert window.analyzed_samples_48k == resampled.shape[1]


def test_analyze_wav_offset_slice(tmp_path, capsys):
    # Happy-path --offset: proves the _audio_slice -> _resolve_region adapter
    # threads a non-zero offset end-to-end via the CLI (previously --offset was
    # only exercised via error paths above). 96000 native samples leaves 95000
    # samples after offset 1000 -> well above MIN_ANALYZE_SAMPLES_48K.
    rc = main(["analyze", "--wav", str(_wav(tmp_path)), "--offset", "1000"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == int(ExitCode.OK)
    window = payload[0]["input_provenance"]["analyzed_window"]
    assert window["native_offset_samples"] == 1000


def test_analyze_wav_nonexistent_file_is_input_error(capsys):
    """``analyze --wav <missing>`` maps to a typed INPUT error (exit 2,
    ``INPUT_WAV_UNREADABLE``) at the CLI seam, mirroring
    ``test_analyze_nonexistent_plugin_path_is_input_error`` for the plugin source."""
    rc = main(["analyze", "--wav", "/no/such/file.wav"])
    assert rc == int(ExitCode.INPUT)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "fatal-error"
    assert payload["error"]["code"] == "INPUT_WAV_UNREADABLE"
    assert payload["error"]["component"] == "analyze"
