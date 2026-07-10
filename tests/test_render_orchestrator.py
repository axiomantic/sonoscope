"""RenderOrchestrator tests (Task E5, design §5 data-flow steps 1-3).

Green-mirage discipline: the ref-verification path is exercised against the REAL
pinned corpus (and a mutated TEMP copy) so a stub that skips verification cannot
pass; the happy path spawns a real ``multiprocessing`` render child with a FAKE
backend and asserts the recorded ``wav_sha256`` matches the bytes on disk.

The fake backends + render helper are defined at MODULE TOP LEVEL because the
``"spawn"`` start method re-imports this module in the child and unpickles the
fake by qualified name (a closure/local class would not be spawn-picklable) —
mirroring ``tests/backends/test_subprocess_render.py``.

Assertions are exact-equality (Level 4+): exact error ``code`` / ``exit_code`` /
``component``, exact stimulus TYPE + message tuples, exact ``wav_sha256``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonoscope import corpus as corpus_mod
from sonoscope.backends.base import (
    AudioStimulus,
    MidiStimulus,
    PluginInfo,
    RenderMeta,
    RenderRequest,
    RenderResult,
)
from sonoscope.errors import InputError
from sonoscope.render_orchestrator import RenderOutcome, build_stimulus, render
from sonoscope.schema import ExitCode
from sonoscope.spec import Spec

# Deterministic fake-render shape (a fresh spawn interpreter must reproduce the
# wav bytes, hence its sha256, exactly).
_FAKE_SR = 48000
_FAKE_FRAMES = _FAKE_SR // 10  # 0.1 s
_FAKE_CHANNELS = 1
_FAKE_WAV_SUBTYPE = "PCM_F32"
_SOUNDFILE_SUBTYPE = "FLOAT"


# --- spawn-picklable fake backend + helpers (module top-level) ---------------


def _fake_render(wav_dir: str, req: RenderRequest) -> RenderResult:
    """Write a real deterministic wav + return a ``RenderMeta`` matching it."""
    import uuid

    data = (np.arange(_FAKE_FRAMES, dtype=np.float32) % 7) * 0.01
    wav_path = Path(wav_dir) / f"fake-{uuid.uuid4().hex}.wav"
    sf.write(str(wav_path), data, _FAKE_SR, subtype=_SOUNDFILE_SUBTYPE)
    wav_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    meta = RenderMeta(
        sample_rate_hz=req.sample_rate_hz,
        block_size=req.block_size,
        channels=_FAKE_CHANNELS,
        duration_s=_FAKE_FRAMES / _FAKE_SR,
        wav_subtype=_FAKE_WAV_SUBTYPE,
        wav_sha256=wav_sha256,
        render_wall_ms=0,
        warnings=[],
    )
    return RenderResult(wav_path=wav_path, render_meta=meta, warnings=[])


class _FakeInstrumentBackend:
    """A fake instrument backend: probe reports ``is_instrument=True`` with no
    params (so ``resolve`` passes trivially); render writes a deterministic wav."""

    id = "fake-instrument"
    version = "9.9.9"

    def __init__(self, wav_dir: str) -> None:
        self.wav_dir = str(wav_dir)

    def probe(self, plugin_path: Path) -> PluginInfo:
        return _instrument_info()

    def render(self, req: RenderRequest) -> RenderResult:
        return _fake_render(self.wav_dir, req)


def _instrument_info() -> PluginInfo:
    """An instrument PluginInfo (routes to the MIDI stimulus path, B1 C4)."""
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


def _effect_info() -> PluginInfo:
    """An effect PluginInfo (routes to the audio stimulus path, B1 C4)."""
    return PluginInfo(
        name="FakeEffect",
        format="vst3",
        params=[],
        input_channels=2,
        output_channels=2,
        is_instrument=False,
        needs_gui=False,
        latency_samples=0,
    )


def _midi_spec(ref: str = "corpus/midi/c3_sustain_2s.mid", **overrides) -> Spec:
    base = {"stimulus": {"kind": "midi", "ref": ref}}
    base.update(overrides)
    return Spec.model_validate(base)


def _temp_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the real corpus into a temp tree; return (corpus_root, manifest)."""
    dst = tmp_path / "corpus"
    shutil.copytree(corpus_mod.DEFAULT_CORPUS_ROOT, dst)
    return dst, dst / "manifest.toml"


# --- REQUIRED acceptance tests (exact names) ---------------------------------


def test_missing_corpus_ref_is_input_error(tmp_path: Path) -> None:
    """RED-proving: a stimulus ref that is not a pinned corpus item must raise an
    INPUT error (exit 2) and NEVER silently proceed to a render. The fake backend
    probes + the spec resolves cleanly, so the ONLY failure point is the bad ref;
    a stub that skipped ref-verification would spawn a render and fail this."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    spec = _midi_spec(ref="corpus/midi/does_not_exist.mid")

    with pytest.raises(InputError) as exc_info:
        render(
            spec,
            Path("/nonexistent.vst3"),
            _FakeInstrumentBackend(str(wav_dir)),
        )

    err = exc_info.value
    assert err.code == "STIMULUS_REF_UNKNOWN"
    assert err.exit_code == ExitCode.INPUT
    assert int(err.exit_code) == 2
    assert err.component == "corpus"
    # No wav was produced: the orchestrator refused before dispatching a render.
    assert list(wav_dir.iterdir()) == []


def test_happy_path_produces_wav_and_meta(tmp_path: Path) -> None:
    """A fake instrument backend renders (in a spawn child) against the REAL
    pinned corpus MIDI ref; the orchestrator returns a wav that exists on disk
    and a RenderMeta whose ``wav_sha256`` exactly matches the file bytes."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    spec = _midi_spec()  # real corpus c3_sustain ref -> ref-verification runs

    outcome = render(
        spec,
        Path("/nonexistent.vst3"),
        _FakeInstrumentBackend(str(wav_dir)),
    )

    assert isinstance(outcome, RenderOutcome)
    assert outcome.wav_path.exists()
    wav_sha256 = hashlib.sha256(outcome.wav_path.read_bytes()).hexdigest()
    assert outcome.render_meta.wav_sha256 == wav_sha256
    # Orchestrator-owned report.render residuals (C3) + deferred determinism.
    assert outcome.backend == "fake-instrument"
    assert outcome.backend_version == "9.9.9"
    assert outcome.determinism is None
    # ref_sha256 is the verified pin for the real corpus MIDI item.
    c3 = {item.path: item for item in corpus_mod.list_items()}["midi/c3_sustain_2s.mid"]
    assert outcome.ref_sha256 == c3.sha256


# --- ref-verification RED coverage (drift is caught) -------------------------


def test_corpus_ref_hash_mismatch_is_input_error(tmp_path: Path) -> None:
    """RED-proving: a corpus item whose on-disk bytes drift from the pinned
    sha256 is an INPUT error (exit 2) — the pins-are-law tripwire. A stub that
    trusted the manifest without hashing the file would miss this."""
    corpus_root, manifest = _temp_corpus(tmp_path)
    mid = corpus_root / "midi" / "c3_sustain_2s.mid"
    tampered = bytearray(mid.read_bytes())
    tampered[-1] ^= 0x01  # flip one byte -> hash drift
    mid.write_bytes(bytes(tampered))

    with pytest.raises(InputError) as exc_info:
        build_stimulus(
            _midi_spec(),
            _instrument_info(),
            corpus_root=corpus_root,
            manifest_path=manifest,
        )

    err = exc_info.value
    assert err.code == "STIMULUS_REF_HASH_MISMATCH"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "corpus"


def test_corpus_ref_missing_file_is_input_error(tmp_path: Path) -> None:
    """RED-proving: a pinned item whose file is absent on disk is an INPUT error,
    never a silent proceed to a default/empty stimulus."""
    corpus_root, manifest = _temp_corpus(tmp_path)
    (corpus_root / "midi" / "c3_sustain_2s.mid").unlink()

    with pytest.raises(InputError) as exc_info:
        build_stimulus(
            _midi_spec(),
            _instrument_info(),
            corpus_root=corpus_root,
            manifest_path=manifest,
        )

    err = exc_info.value
    assert err.code == "STIMULUS_REF_MISSING"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "corpus"


# --- instrument-vs-effect routing (PluginInfo.is_instrument, B1 C4) ----------


def test_instrument_routing_builds_midi_stimulus() -> None:
    """GREEN: an instrument plugin routes the corpus MIDI ref to a MidiStimulus
    with the exact loaded messages + the verified ref_sha256."""
    stimulus, ref_sha256 = build_stimulus(_midi_spec(), _instrument_info())

    assert type(stimulus) is MidiStimulus
    assert [(m.type, m.note, m.velocity, m.time) for m in stimulus.messages] == [
        ("note_on", 48, 100, 0.0),
        ("note_off", 48, 0, 2.0),
    ]
    c3 = {item.path: item for item in corpus_mod.list_items()}["midi/c3_sustain_2s.mid"]
    assert ref_sha256 == c3.sha256


def test_effect_routing_builds_audio_stimulus() -> None:
    """GREEN: an effect plugin routes a corpus SIGNAL ref to an AudioStimulus with
    channel-major float32 audio at the corpus sample rate + the verified pin."""
    spec = Spec.model_validate(
        {"stimulus": {"kind": "tone", "ref": "corpus/signals/tone_1k_2s.wav"}}
    )
    stimulus, ref_sha256 = build_stimulus(spec, _effect_info())

    assert type(stimulus) is AudioStimulus
    assert stimulus.sample_rate_hz == 48000
    assert stimulus.audio.dtype == np.float32
    assert stimulus.audio.shape == (1, 96000)  # mono tone, channel-major
    tone = {item.path: item for item in corpus_mod.list_items()}[
        "signals/tone_1k_2s.wav"
    ]
    assert ref_sha256 == tone.sha256


def test_inline_notes_build_midi_stimulus() -> None:
    """GREEN: inline notes on an instrument spec build a MidiStimulus directly
    (no corpus ref -> ref_sha256 is None)."""
    spec = Spec.model_validate(
        {
            "stimulus": {
                "kind": "midi",
                "notes": [{"pitch": 60, "vel": 100, "on": 0.0, "off": 1.0}],
            }
        }
    )
    stimulus, ref_sha256 = build_stimulus(spec, _instrument_info())

    assert type(stimulus) is MidiStimulus
    assert ref_sha256 is None
    assert [(m.type, m.note, m.velocity, m.time) for m in stimulus.messages] == [
        ("note_on", 60, 100, 0.0),
        ("note_off", 60, 0, 1.0),
    ]
    assert stimulus.duration_s == 1.0


def test_instrument_with_signal_ref_is_input_error() -> None:
    """RED-proving: an instrument plugin pointed at a SIGNAL corpus item (not a
    MIDI item) is a stimulus/plugin mismatch INPUT error — never a silent
    coercion into an empty MIDI stream."""
    spec = Spec.model_validate(
        {"stimulus": {"kind": "sweep", "ref": "corpus/signals/logsweep_20-20k_2s.wav"}}
    )
    with pytest.raises(InputError) as exc_info:
        build_stimulus(spec, _instrument_info())

    err = exc_info.value
    assert err.code == "STIMULUS_KIND_MISMATCH"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "corpus"
