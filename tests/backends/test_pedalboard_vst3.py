"""Tests for the PedalboardVST3Backend (Task E3, design §4.1, §9).

Split (per the E3 spec):

- **Non-integration** (default gate, no Surge XT): the raw_state hash-stamp
  validation and the seed record helpers are pure and testable without a real
  plugin — ``test_stale_raw_state_hard_errors`` and
  ``test_seed_forwarded_and_recorded``.
- **Integration** (``@pytest.mark.integration``, requires Surge XT): probe +
  instrument render + effect render exercise the real pedalboard 0.9.23 ↔ Surge
  XT path confirmed by B1.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonoscope.backends.base import (
    ParamInfo,
    PluginInfo,
    RawState,
    RenderMeta,
    RenderRequest,
)
from sonoscope.backends.pedalboard_vst3 import (
    AudioStimulus,
    MidiStimulus,
    PedalboardVST3Backend,
    _apply_params,
    _reinject_raw_state,
    _render_audio,
    binary_sha256,
    seed_record_warnings,
    validate_raw_state,
)
from sonoscope.errors import RenderError
from sonoscope.schema import ExitCode

# Exact seed-record string (design §8: seed forwarded where honored + recorded
# for reproducibility; the backend exposes no plugin-honored seed path in v1).
_SEED_RECORD = (
    "seed-forwarded: 1234 (backend has no plugin-honored seed path; "
    "recorded for reproducibility, measured floor is source of truth)"
)


def _rms_dbfs(data: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(data.astype(np.float64)))))
    return 20.0 * math.log10(rms) if rms > 0.0 else float("-inf")


# --- Non-integration: raw_state stamp validation (RED fixture, §12.1) --------


def test_stale_raw_state_hard_errors(tmp_path):
    # A raw_state stamped to a DIFFERENT binary_sha256 than the loaded binary is
    # STALE -> hard RenderError, never a silent proceed to default state (§9).
    fake_binary = tmp_path / "Fake.vst3"
    fake_binary.write_bytes(b"not the real plugin bytes")
    actual = binary_sha256(fake_binary)

    stale = RawState(
        captured=True,
        plugin_binary_sha256="d" * 64,
        blob_ref=str(tmp_path / "state.blob"),
    )
    assert stale.plugin_binary_sha256 != actual

    with pytest.raises(RenderError) as excinfo:
        validate_raw_state(stale, actual)
    assert excinfo.value.code == "RAW_STATE_STALE"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER

    # GREEN: a stamp matching the loaded binary validates and requests injection.
    fresh = RawState(
        captured=True,
        plugin_binary_sha256=actual,
        blob_ref=str(tmp_path / "state.blob"),
    )
    assert validate_raw_state(fresh, actual) is True

    # Not captured / absent -> no injection, no raise.
    assert validate_raw_state(RawState(captured=False), actual) is False
    assert validate_raw_state(None, actual) is False


def test_seed_forwarded_and_recorded():
    # The request seed round-trips into the recorded render metadata (§8, M8).
    entries = seed_record_warnings(1234)
    assert entries == [_SEED_RECORD]

    meta = RenderMeta(
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        duration_s=1.0,
        wav_subtype="PCM_F32",
        wav_sha256="0" * 64,
        render_wall_ms=1,
        warnings=entries,
    )
    assert meta.warnings == [_SEED_RECORD]

    # No seed -> nothing recorded.
    assert seed_record_warnings(None) == []


# --- Non-integration: pure hard-error branches (RED-proving, no Surge) -------
# These four branches are the "green-mirage" gaps: pure guards whose only proof
# of behavior is that they HARD-ERROR with an exact code/component/exit_code
# rather than silently proceeding. Each is provable with fakes/minimal objects.


def _minimal_plugin_info(name: str, param_names: list[str]) -> PluginInfo:
    """A minimal PluginInfo with float params (no Surge, no plugin load)."""
    return PluginInfo(
        name=name,
        format="vst3",
        params=[
            ParamInfo(
                name=pname, index=i, kind="float", num_steps=None, default=0.0
            )
            for i, pname in enumerate(param_names)
        ],
        input_channels=2,
        output_channels=2,
        is_instrument=True,
        needs_gui=False,
        latency_samples=0,
    )


class _ResolvedParam:
    """Minimal ResolvedParam-shaped object (name/index/value) for _apply_params."""

    def __init__(self, name, index, value):
        self.name = name
        self.index = index
        self.value = value


class _FakePlugin:
    """Minimal plugin exposing a dict `.parameters` keyed by python_name."""

    def __init__(self, keys):
        self.parameters = {k: object() for k in keys}


def test_unstamped_raw_state_hard_errors():
    # The other half of the §9 guard (mirrors test_stale_raw_state_hard_errors):
    # captured=True but no plugin_binary_sha256 stamp cannot be validated against
    # the loaded binary -> hard RenderError, never a silent proceed to default.
    unstamped = RawState(
        captured=True, plugin_binary_sha256=None, blob_ref="ignored"
    )
    with pytest.raises(RenderError) as excinfo:
        validate_raw_state(unstamped, "a" * 64)
    assert excinfo.value.code == "RAW_STATE_UNSTAMPED"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER


def test_reinject_missing_blob_hard_errors(tmp_path):
    # A validated (captured + matching stamp) state whose blob_ref does not point
    # at an existing file cannot be re-injected -> hard RAW_STATE_BLOB_MISSING.
    # The guard fires before touching the plugin, so a bare object() suffices.
    missing_blob = tmp_path / "does-not-exist.blob"
    assert missing_blob.exists() is False
    state = RawState(
        captured=True,
        plugin_binary_sha256="a" * 64,
        blob_ref=str(missing_blob),
    )
    with pytest.raises(RenderError) as excinfo:
        _reinject_raw_state(object(), state)
    assert excinfo.value.code == "RAW_STATE_BLOB_MISSING"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER


def test_unknown_stimulus_type_hard_errors():
    # A req.stimulus that is neither MidiStimulus nor AudioStimulus falls through
    # both dispatch overloads -> hard STIMULUS_TYPE_UNKNOWN (never a silent no-op).
    req = RenderRequest(
        plugin_path=Path("/plugins/Fake.vst3"),
        plugin_format="vst3",
        stimulus=object(),  # unknown stimulus type
        param_set=None,
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        raw_state=None,
        seed=None,
    )
    info = _minimal_plugin_info("Fake", ["cutoff"])
    with pytest.raises(RenderError) as excinfo:
        _render_audio(object(), info, req)
    assert excinfo.value.code == "STIMULUS_TYPE_UNKNOWN"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER


def test_apply_param_not_on_plugin_hard_errors():
    # A resolved param whose name is absent AND whose index is out of range for
    # the probed PluginInfo cannot be applied -> hard PARAM_NOT_ON_PLUGIN, never
    # a silent skip (which would drop a requested param setting).
    plugin = _FakePlugin(["cutoff"])
    info = _minimal_plugin_info("Fake", ["cutoff"])
    bogus = _ResolvedParam(name="nonexistent", index=99, value=0.5)
    with pytest.raises(RenderError) as excinfo:
        _apply_params(plugin, info, [bogus])
    assert excinfo.value.code == "PARAM_NOT_ON_PLUGIN"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER


# --- Non-integration: render-dir temp leak (Finding 1, RED-proving) ----------
# The backend hosts pedalboard (a pinned dep) but writes wavs via the pure
# canonical encoder, so ``_write_wav`` is exercisable with plain numpy audio and
# NO Surge XT / plugin load — a default-gate test of the temp-dir lifecycle.


def test_write_wav_reuses_single_temp_dir_and_close_cleans():
    # Finding 1: with NO render_dir the backend must not mkdtemp a fresh dir on
    # every render (an unbounded disk leak). It creates at most ONE instance-owned
    # temp dir, reused across renders, and close() removes it.
    backend = PedalboardVST3Backend()  # no render_dir -> owned-temp-dir path
    audio = np.zeros((2, 64), dtype=np.float32)

    wav1 = backend._write_wav(audio, 48000)
    wav2 = backend._write_wav(audio, 48000)
    wav3 = backend._write_wav(audio, 48000)

    # RED-proving: under the old per-render mkdtemp each wav lived in a DIFFERENT
    # temp dir; the fix reuses ONE dir, so all three share a parent.
    assert wav1.parent == wav2.parent
    assert wav2.parent == wav3.parent
    # Distinct uuid-named wavs, all present in that single owned dir.
    assert len({wav1, wav2, wav3}) == 3
    assert wav1.is_file() is True
    assert wav2.is_file() is True
    assert wav3.is_file() is True

    owned = wav1.parent
    assert owned.is_dir() is True

    # close() removes the instance-owned temp dir -> no leftover leak. Idempotent.
    backend.close()
    assert owned.exists() is False
    backend.close()  # second close is a no-op, never raises


def test_close_does_not_remove_caller_render_dir(tmp_path):
    # A caller-supplied render_dir is owned by the CALLER (the CLI/orchestrator
    # lifecycle, Finding 1) and must NEVER be removed by the backend's close().
    backend = PedalboardVST3Backend(render_dir=tmp_path)
    audio = np.zeros((2, 64), dtype=np.float32)

    wav = backend._write_wav(audio, 48000)
    assert wav.parent == tmp_path

    backend.close()
    assert tmp_path.is_dir() is True
    assert wav.is_file() is True


def test_backend_context_manager_cleans_owned_temp_dir():
    # The context manager cleans the instance-owned temp dir on exit (Finding 1).
    audio = np.zeros((2, 64), dtype=np.float32)
    with PedalboardVST3Backend() as backend:
        wav = backend._write_wav(audio, 48000)
        owned = wav.parent
        assert owned.is_dir() is True
    assert owned.exists() is False


# --- Non-integration: probe() maps raw build faults to RenderError -----------
# Finding 1 (Gemini cycle 4): probe() must not let a RAW exception from
# _build_plugin_info (channel detection / param parsing) escape as an unmapped
# INTERNAL_ERROR (exit 1). The failure is forced through the smallest seams
# (fake _load + stubbed binary_sha256) so NO Surge XT / real plugin is needed.


def test_probe_wraps_raw_build_failure_as_render_error(monkeypatch, tmp_path):
    # A RAW exception (e.g. ValueError) from _build_plugin_info must be mapped to
    # a typed RenderError with code PLUGIN_PROBE_FAILED, component "render", exit
    # RENDER (3) — never a bare ValueError -> unmapped INTERNAL_ERROR (exit 1).
    # RED-proving against the unguarded ``info = _build_plugin_info(plugin)`` call.
    import sonoscope.backends.pedalboard_vst3 as mod

    backend = PedalboardVST3Backend()
    # Seams: a stable cache key (no real bundle hash) + a fake load (no Surge) so
    # the test reaches _build_plugin_info without loading a real plugin.
    monkeypatch.setattr(mod, "binary_sha256", lambda _p: "a" * 64)
    monkeypatch.setattr(backend, "_load", lambda _p: object())

    def _raise_raw(_plugin):
        raise ValueError("channel detection blew up")

    monkeypatch.setattr(mod, "_build_plugin_info", _raise_raw)

    with pytest.raises(RenderError) as excinfo:
        backend.probe(tmp_path / "Fake.vst3")
    assert excinfo.value.code == "PLUGIN_PROBE_FAILED"
    assert excinfo.value.component == "render"
    assert excinfo.value.exit_code == ExitCode.RENDER


def test_probe_passes_through_typed_render_error(monkeypatch, tmp_path):
    # Counterpart guard against DOUBLE-wrapping: if _build_plugin_info already
    # raises a TYPED SonoscopeError (RenderError), probe() must propagate it
    # UNCHANGED (same code + identity) — not re-wrap it as PLUGIN_PROBE_FAILED.
    # Proves the ``except SonoscopeError: raise`` branch fires before the broad
    # ``except Exception`` wrap.
    import sonoscope.backends.pedalboard_vst3 as mod

    backend = PedalboardVST3Backend()
    monkeypatch.setattr(mod, "binary_sha256", lambda _p: "b" * 64)
    monkeypatch.setattr(backend, "_load", lambda _p: object())

    typed = RenderError(
        "PARAM_NOT_ON_PLUGIN", "already a typed render error", component="render"
    )

    def _raise_typed(_plugin):
        raise typed

    monkeypatch.setattr(mod, "_build_plugin_info", _raise_typed)

    with pytest.raises(RenderError) as excinfo:
        backend.probe(tmp_path / "Fake.vst3")
    assert excinfo.value is typed
    assert excinfo.value.code == "PARAM_NOT_ON_PLUGIN"


# --- Integration (Surge XT): probe + render paths (B1-confirmed) -------------


@pytest.mark.integration
def test_probe_enumerates_named_params(surge_vst3_path):
    backend = PedalboardVST3Backend()
    info = backend.probe(surge_vst3_path)

    assert info.name == "Surge XT"
    assert info.is_instrument is True
    # B1: Surge XT 1.3.4 exposes exactly 775 named params via plugin.parameters.
    assert len(info.params) == 775
    # Enumeration order + normalized python_name keys (B1 §2 sample).
    assert info.params[0].name == "m1"
    assert info.params[0].index == 0
    assert info.params[12].name == "global_volume"

    by_name = {p.name: p for p in info.params}
    # Kind classification via type/valid_values/is_boolean/range (B1 C5).
    assert by_name["global_volume"].kind == "float"
    assert by_name["global_volume"].num_steps is None
    assert by_name["active_scene"].kind == "stepped"
    assert by_name["active_scene"].num_steps == 2
    assert by_name["fx_a1_fx_type"].kind == "stepped"
    assert by_name["fx_a1_fx_type"].num_steps == 30


@pytest.mark.integration
def test_instrument_render_nonsilent(surge_vst3_path, tmp_path):
    import mido

    backend = PedalboardVST3Backend(render_dir=tmp_path)
    stim = MidiStimulus(
        messages=[
            mido.Message("note_on", note=60, velocity=100, time=0.0),
            mido.Message("note_off", note=60, velocity=0, time=1.9),
        ],
        duration_s=2.0,
    )
    req = RenderRequest(
        plugin_path=surge_vst3_path,
        plugin_format="vst3",
        stimulus=stim,
        param_set=(),
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        raw_state=None,
        seed=None,
    )
    result = backend.render(req)

    assert result.render_meta.wav_subtype == "PCM_F32"
    assert result.render_meta.sample_rate_hz == 48000
    assert result.render_meta.channels == 2
    assert result.wav_path.is_file() is True
    # wav_sha256 stamps the exact bytes on disk.
    assert (
        result.render_meta.wav_sha256
        == hashlib.sha256(result.wav_path.read_bytes()).hexdigest()
    )

    data, sr = sf.read(str(result.wav_path), dtype="float32")
    assert sr == 48000
    assert _rms_dbfs(data) > -80.0


@pytest.mark.integration
def test_effect_render_shapes(surge_vst3_path, tmp_path):
    backend = PedalboardVST3Backend(render_dir=tmp_path)
    n = 24000
    t = np.arange(n) / 48000.0
    sweep = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    stereo_in = np.stack([sweep, sweep])  # (channels, samples)

    stim = AudioStimulus(audio=stereo_in, sample_rate_hz=48000)
    req = RenderRequest(
        plugin_path=surge_vst3_path,
        plugin_format="vst3",
        stimulus=stim,
        param_set=(),
        sample_rate_hz=48000,
        block_size=512,
        channels=2,
        raw_state=None,
        seed=None,
    )
    result = backend.render(req)

    assert result.render_meta.channels == 2
    assert result.render_meta.sample_rate_hz == 48000
    assert result.render_meta.wav_subtype == "PCM_F32"

    data, sr = sf.read(str(result.wav_path), dtype="float32")
    assert sr == 48000
    assert data.dtype == np.float32
    assert data.shape[1] == 2  # stereo out
