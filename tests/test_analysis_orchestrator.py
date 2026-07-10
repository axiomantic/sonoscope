"""AnalysisOrchestrator tests (Task F1, design §4.4, §5 step 4, §3.6).

Green-mirage discipline for the analyze engine's load-bearing guarantees:

- **Deterministic-first + perception-never-fatal.** The deterministic and
  tripwire blocks are populated from the ground-truth layer BEFORE perception is
  called; a perception adapter that *raises* (or times out) degrades to
  ``perception.status == "error"`` while the loop continues and the report is
  still assembled — the RED-proving guarantee that perception can never abort the
  analysis (§10.2, §12.4).
- **M7 errors[] aggregation.** A perception degradation surfaces as an exact
  ``errors[]`` entry (severity ``error``, component ``perception``); the internal
  ``seed-forwarded:`` machine record is filtered out of the user-facing
  ``report.render.warnings`` (E3 stopgap) while a real coercion warning survives.
- **InputBlock assembly.** ``plugin.binary_sha256`` is the real content hash of
  the plugin bundle; an inline-notes stimulus (no pinned corpus ref) uses the
  documented ``ref="inline"`` / ``ref_sha256=""`` sentinel without changing C1.

Assertions are exact-equality (Level 4+): exact enum statuses, exact ``ErrorItem``
objects, exact sentinel strings, exact hashes, and a full model round-trip.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonoscope import corpus as corpus_mod
from sonoscope.analysis_orchestrator import (
    INLINE_STIMULUS_REF,
    INLINE_STIMULUS_REF_SHA256,
    PERCEPTION_DEGRADED_CODE,
    PLUGIN_BINARY_UNREADABLE_CODE,
    analyze_plugin_spec,
    analyze_render_outcome,
)
from sonoscope.backends.base import (
    PluginInfo,
    RenderMeta,
    RenderRequest,
    RenderResult,
)
from sonoscope.backends.pedalboard_vst3 import binary_sha256
from sonoscope.errors import AnalysisError, InputError
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.render_orchestrator import RenderOutcome
from sonoscope.resolver import ResolvedSpec
from sonoscope.schema import ExitCode
from sonoscope.schema.models import (
    AnalysisReport,
    DeterministicBlock,
    ErrorItem,
    PerceptionBlock,
)
from sonoscope.spec import Note, RenderSpec, Spec, StimulusSpec

_SR = 48000  # frozen analysis rate (D1 requires the wav to be 48 kHz)
_SEED_WARNING = "seed-forwarded: 42 (recorded for reproducibility)"
_COERCION_WARNING = "block size 512 -> 480 (coerced)"


# --- real-wav + RenderOutcome builders --------------------------------------


def _write_wav(path: Path, *, duration_s: float = 0.5, freq: float = 1000.0) -> None:
    """Write a real 48 kHz stereo float32 tone so D1/D2 run on genuine audio."""
    n = int(_SR * duration_s)
    t = np.arange(n, dtype=np.float64) / _SR
    tone = (0.25 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)  # (frames, channels) for soundfile
    sf.write(str(path), stereo, _SR, subtype="FLOAT")


def _fake_plugin(tmp_path: Path) -> Path:
    """A fake ``.vst3`` bundle dir with one file so binary_sha256 tree-hashes it."""
    bundle = tmp_path / "Fake.vst3"
    bundle.mkdir()
    (bundle / "plugin.bin").write_bytes(b"fake-vst3-content")
    return bundle


def _resolved(
    stimulus: StimulusSpec,
    *,
    expected_audio: bool = True,
    patch_class: str = "noisy",
    resolved_sha256: str = "a" * 64,
) -> ResolvedSpec:
    return ResolvedSpec(
        spec_version="1.0.0",
        stimulus=stimulus,
        patch_class=patch_class,  # type: ignore[arg-type]
        expected_audio=expected_audio,
        params=(),
        render=RenderSpec(),
        resolved_sha256=resolved_sha256,
    )


def _outcome(
    tmp_path: Path,
    *,
    stimulus: StimulusSpec,
    ref_sha256: str | None,
    warnings: list[str] | None = None,
    expected_audio: bool = True,
    duration_s: float = 0.5,
    sr: int = _SR,
) -> RenderOutcome:
    wav = tmp_path / "render.wav"
    _write_wav(wav, duration_s=duration_s) if sr == _SR else _write_wav_at(
        wav, sr, duration_s
    )
    wav_sha256 = hashlib.sha256(wav.read_bytes()).hexdigest()
    meta = RenderMeta(
        sample_rate_hz=sr,
        block_size=512,
        channels=2,
        duration_s=duration_s,
        wav_subtype="PCM_F32",
        wav_sha256=wav_sha256,
        render_wall_ms=7,
        warnings=list(warnings or []),
    )
    return RenderOutcome(
        wav_path=wav,
        render_meta=meta,
        backend="pedalboard-vst3",
        backend_version="0.9.23",
        resolved=_resolved(stimulus, expected_audio=expected_audio),
        ref_sha256=ref_sha256,
        seed=42,
        determinism=None,
    )


def _write_wav_at(path: Path, sr: int, duration_s: float) -> None:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    tone = (0.25 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    sf.write(str(path), np.stack([tone, tone], axis=1), sr, subtype="FLOAT")


def _tone_stimulus() -> StimulusSpec:
    return StimulusSpec(kind="tone", ref="corpus/signals/tone_1k_2s.wav")


def _inline_stimulus() -> StimulusSpec:
    return StimulusSpec(
        kind="midi", notes=[Note(pitch=60, vel=100, on=0.0, off=0.5)]
    )


class _RaisingAdapter:
    """A perception adapter whose describe() ALWAYS raises (crash path)."""

    id = "raising"
    grounding = "advisory-freetext"

    def describe(self, wav_path, deterministic=None):  # noqa: ANN001
        raise RuntimeError("model exploded")

    def health(self):
        raise RuntimeError("unused")


class _SlowAdapter:
    """A perception adapter whose describe() sleeps well past the watchdog
    timeout (exercises the TIMEOUT half of perception-never-fatal, not the
    raise path)."""

    id = "slow"
    grounding = "advisory-freetext"

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s

    def describe(self, wav_path, deterministic=None):  # noqa: ANN001
        time.sleep(self.sleep_s)
        raise AssertionError("watchdog should have abandoned this call")

    def health(self):
        raise RuntimeError("unused")


class _CountingRaisingAdapter:
    """A perception adapter that records whether it was invoked and WOULD raise
    if called — used to prove the perception_enabled=False short-circuit never
    touches the adapter."""

    id = "counting"
    grounding = "advisory-freetext"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, wav_path, deterministic=None):  # noqa: ANN001
        self.calls += 1
        raise RuntimeError("adapter must not be called when disabled")

    def health(self):
        raise RuntimeError("unused")


class _ReturnedStatusAdapter:
    """A perception adapter whose describe() RETURNS a chosen-status block
    WITHOUT raising (e.g. G1's cooperative per-token timeout returns
    status=='error' itself instead of raising). Used to prove the returned-block
    degradation path is logged into errors[] for 'error' but NOT for the
    graceful 'unavailable'/'disabled' statuses."""

    id = "returned"
    grounding = "advisory-freetext"

    def __init__(self, block: PerceptionBlock) -> None:
        self._block = block

    def describe(self, wav_path, deterministic=None):  # noqa: ANN001
        return self._block

    def health(self):
        raise RuntimeError("unused")


# --- REQUIRED acceptance tests (exact names) ---------------------------------


def test_report_validates_against_schema(tmp_path: Path) -> None:
    """The assembled report is a valid AnalysisReport that survives a full
    JSON round-trip, and the deterministic ground-truth block is ALWAYS present."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        spec_ref="specs/tone.json",
        adapter=NullAdapter(),
    )

    assert isinstance(report, AnalysisReport)
    assert isinstance(report.deterministic, DeterministicBlock)
    # Ground truth actually computed (not a stub): summary + integrity populated.
    assert report.deterministic.summary.sample_rate_hz == _SR
    assert report.deterministic.summary.channels == 2
    assert len(report.deterministic.summary.mfcc_mean) == 13
    # Full round-trip through JSON re-validates to an identical model.
    reloaded = AnalysisReport.model_validate_json(report.model_dump_json())
    assert reloaded == report


def test_perception_disabled_still_valid(tmp_path: Path) -> None:
    """NullAdapter -> perception.status == 'disabled' exactly, report still valid
    (graceful-degradation GREEN, §12.4)."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=NullAdapter(),
    )

    assert report.perception == PerceptionBlock(status="disabled", grounding="none")
    # Deterministic ground truth is intact regardless of perception.
    assert report.deterministic.summary.channels == 2
    assert report.errors == []


def test_perception_disabled_flag_skips_adapter(tmp_path: Path) -> None:
    """perception_enabled=False short-circuits BEFORE the adapter: an adapter that
    would raise if called is never invoked, and perception.status == 'disabled'
    (a distinct branch from NullAdapter's own disabled status)."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    adapter = _CountingRaisingAdapter()

    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=adapter,
        perception_enabled=False,
    )

    assert report.perception == PerceptionBlock(status="disabled", grounding="none")
    # The short-circuit never touched the adapter (would have raised otherwise).
    assert adapter.calls == 0
    assert report.errors == []


def test_perception_error_does_not_fail_loop(tmp_path: Path) -> None:
    """RED-proving: a perception adapter that RAISES degrades to
    status == 'error' while the loop CONTINUES — the deterministic block is
    intact and no exception escapes (exit 0)."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_RaisingAdapter(),
    )

    assert report.perception == PerceptionBlock(status="error", grounding="none")
    # Deterministic + tripwires survived the perception crash. The clean
    # 0.25-amp 1 kHz stereo tone fixture is deterministically PASS (binding
    # exact-equality rule; no `in`-membership on the value under test).
    assert report.deterministic.summary.channels == 2
    assert report.tripwires.overall == "PASS"


def test_deterministic_first_ordering(tmp_path: Path) -> None:
    """Deterministic + tripwires are populated even when perception errors —
    proving the deterministic-first ordering guarantee."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_RaisingAdapter(),
    )

    assert report.perception.status == "error"
    # Deterministic block fully populated (ground truth).
    assert report.deterministic.summary.duration_s == pytest.approx(0.5)
    assert report.deterministic.integrity.silence_threshold_dbfs == -80.0
    # Tripwires evaluated over the four §4.4 checks.
    assert [r.id for r in report.tripwires.results] == [
        "silent-output",
        "nan-inf",
        "denormal",
        "clipping",
    ]


# --- M7 errors[] aggregation + seed-prefix filtering -------------------------


def test_perception_error_adds_errors_entry(tmp_path: Path) -> None:
    """M7: perception degradation surfaces as an exact cross-cutting errors[]
    entry (severity 'error', component 'perception')."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_RaisingAdapter(),
    )

    assert report.errors == [
        ErrorItem(
            code=PERCEPTION_DEGRADED_CODE,
            message="perception adapter failed; analysis continued without it",
            detail={"adapter": "raising"},
            severity="error",
            component="perception",
        )
    ]


def test_perception_timeout_degrades_not_fatal(tmp_path: Path) -> None:
    """RED-proving the TIMEOUT half of perception-never-fatal: an adapter whose
    describe() sleeps well past a small ``perception_timeout_s`` override is
    ABANDONED by the watchdog and degrades to status == 'error' + an exact
    errors[] entry, with the deterministic block intact, NO exception escaping,
    and the call returning promptly (the watchdog does not hang for the full
    sleep)."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)

    start = time.perf_counter()
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_SlowAdapter(sleep_s=3.0),
        perception_timeout_s=0.1,
    )
    elapsed = time.perf_counter() - start

    # Timeout degraded, not raised: exact error block + exact errors[] entry.
    assert report.perception == PerceptionBlock(status="error", grounding="none")
    assert report.errors == [
        ErrorItem(
            code=PERCEPTION_DEGRADED_CODE,
            message="perception adapter failed; analysis continued without it",
            detail={"adapter": "slow"},
            severity="error",
            component="perception",
        )
    ]
    # Deterministic ground truth survived the perception timeout.
    assert report.deterministic.summary.channels == 2
    # The watchdog abandoned the slow call: returned well before the 3 s sleep
    # (proves it did not hang waiting for describe() to finish).
    assert elapsed < 2.0


def test_perception_returned_error_adds_errors_entry(tmp_path: Path) -> None:
    """RED-proving (Gemini cycle 3): an adapter that RETURNS a
    ``status=='error'`` block WITHOUT raising (G1's cooperative per-token
    timeout) is the SAME degradation as the raise path, so it MUST also emit the
    exact PERCEPTION_DEGRADED errors[] entry. The returned block is passed
    through intact, the deterministic ground truth is untouched, and no exception
    escapes (exit 0). RED against the pre-fix code, which passed the returned
    block through with an empty errors[]."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    returned = PerceptionBlock(status="error", grounding="none")
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_ReturnedStatusAdapter(returned),
    )

    # The returned block is passed through unchanged AND logged to errors[].
    assert report.perception == returned
    assert report.perception.status == "error"
    assert report.errors == [
        ErrorItem(
            code=PERCEPTION_DEGRADED_CODE,
            message="perception adapter failed; analysis continued without it",
            detail={"adapter": "returned"},
            severity="error",
            component="perception",
        )
    ]
    # Deterministic ground truth survived the returned-error degradation.
    assert report.deterministic.summary.channels == 2
    assert report.tripwires.overall == "PASS"


@pytest.mark.parametrize("status", ["unavailable", "disabled"])
def test_perception_returned_nonerror_adds_no_errors_entry(
    tmp_path: Path, status: str
) -> None:
    """Graceful-degradation contract: a RETURNED ``unavailable``/``disabled``
    block is NOT a degradation and MUST NOT produce an errors[] entry — only a
    ``status=='error'`` block does. Guards against the returned-error fix
    over-reaching to the graceful non-error statuses."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    returned = PerceptionBlock(status=status, grounding="none")
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_ReturnedStatusAdapter(returned),
    )

    assert report.perception == returned
    assert report.errors == []


def test_seed_warning_filtered_from_render_warnings(tmp_path: Path) -> None:
    """M2/M7: the internal 'seed-forwarded:' machine record is filtered out of the
    user-facing report.render.warnings while a real coercion warning survives
    verbatim."""
    outcome = _outcome(
        tmp_path,
        stimulus=_tone_stimulus(),
        ref_sha256="b" * 64,
        warnings=[_SEED_WARNING, _COERCION_WARNING],
    )
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=NullAdapter(),
    )

    assert report.render.warnings == [_COERCION_WARNING]


# --- InputBlock assembly: binary_sha256 + inline-notes sentinel --------------


def test_binary_sha256_in_input_block(tmp_path: Path) -> None:
    """plugin.binary_sha256 is the real content hash of the plugin bundle."""
    plugin = _fake_plugin(tmp_path)
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    report = analyze_render_outcome(
        outcome,
        plugin_path=plugin,
        spec_sha256="c" * 64,
        spec_ref="specs/tone.json",
        adapter=NullAdapter(),
    )

    assert report.input.plugin.binary_sha256 == binary_sha256(plugin)
    assert report.input.plugin.format == "vst3"
    assert report.input.plugin.backend == "pedalboard-vst3"
    assert report.input.param_set.spec_sha256 == "c" * 64
    assert report.input.param_set.ref == "specs/tone.json"
    assert report.input.stimulus.ref == "corpus/signals/tone_1k_2s.wav"
    assert report.input.stimulus.ref_sha256 == "b" * 64


def test_inline_notes_use_sentinel_stimulus_ref(tmp_path: Path) -> None:
    """An inline-notes stimulus (no pinned corpus ref; E5 returns ref_sha256=None)
    uses the documented ref='inline' / ref_sha256='' sentinel to satisfy the
    required-str C1 StimulusRef fields WITHOUT changing the schema."""
    outcome = _outcome(tmp_path, stimulus=_inline_stimulus(), ref_sha256=None)
    report = analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=NullAdapter(),
    )

    assert report.input.stimulus.ref == INLINE_STIMULUS_REF
    assert report.input.stimulus.ref_sha256 == INLINE_STIMULUS_REF_SHA256
    assert INLINE_STIMULUS_REF == "inline"
    assert INLINE_STIMULUS_REF_SHA256 == ""


# --- deterministic-layer failure is a fatal ANALYSIS error (exit 4) ----------


def test_unanalyzable_wav_is_analysis_error(tmp_path: Path) -> None:
    """RED-proving: a wav the deterministic layer cannot process (wrong sample
    rate) is a fatal ANALYSIS error (exit 4), never a silent/partial report."""
    outcome = _outcome(
        tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64, sr=44100
    )
    with pytest.raises(AnalysisError) as exc_info:
        analyze_render_outcome(
            outcome,
            plugin_path=_fake_plugin(tmp_path),
            spec_sha256="c" * 64,
            adapter=NullAdapter(),
        )

    err = exc_info.value
    assert err.exit_code == ExitCode.ANALYSIS
    assert int(err.exit_code) == 4
    assert err.component == "analyze"


def test_missing_plugin_path_is_typed_error(tmp_path: Path) -> None:
    """RED-proving: a missing/unreadable plugin bundle at the top of
    analyze_render_outcome maps to a typed InputError (exit 2), never a raw
    FileNotFoundError/OSError escaping the typed-error contract."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    missing = tmp_path / "does_not_exist.vst3"

    with pytest.raises(InputError) as exc_info:
        analyze_render_outcome(
            outcome,
            plugin_path=missing,
            spec_sha256="c" * 64,
            adapter=NullAdapter(),
        )

    err = exc_info.value
    assert err.code == PLUGIN_BINARY_UNREADABLE_CODE
    assert err.exit_code == ExitCode.INPUT
    assert int(err.exit_code) == 2
    assert err.component == "analyze"


# --- end-to-end plugin/spec engine (render via E5 -> assemble) ---------------
# Module top-level fake backend so the "spawn" render child can unpickle it by
# qualified name (mirrors tests/test_render_orchestrator.py).


def _fake_render(wav_dir: str, req: RenderRequest) -> RenderResult:
    import uuid

    n = int(_SR * 0.5)
    t = np.arange(n, dtype=np.float64) / _SR
    data = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    wav_path = Path(wav_dir) / f"fake-{uuid.uuid4().hex}.wav"
    sf.write(str(wav_path), data, _SR, subtype="FLOAT")
    wav_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    meta = RenderMeta(
        sample_rate_hz=_SR,
        block_size=req.block_size,
        channels=1,
        duration_s=0.5,
        wav_subtype="PCM_F32",
        wav_sha256=wav_sha256,
        render_wall_ms=0,
        warnings=[],
    )
    return RenderResult(wav_path=wav_path, render_meta=meta, warnings=[])


class _FakeInstrumentBackend:
    id = "fake-instrument"
    version = "9.9.9"

    def __init__(self, wav_dir: str) -> None:
        self.wav_dir = str(wav_dir)

    def probe(self, plugin_path: Path) -> PluginInfo:
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

    def render(self, req: RenderRequest) -> RenderResult:
        return _fake_render(self.wav_dir, req)


def test_plugin_spec_end_to_end_assembles_report(tmp_path: Path) -> None:
    """analyze_plugin_spec renders via E5 (spawn child, fake backend, REAL corpus
    MIDI ref) then assembles a valid AnalysisReport whose render.wav_sha256
    matches the produced wav."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    plugin = _fake_plugin(tmp_path)
    spec = Spec.model_validate(
        {"stimulus": {"kind": "midi", "ref": "corpus/midi/c3_sustain_2s.mid"}}
    )

    report = analyze_plugin_spec(
        spec,
        plugin,
        _FakeInstrumentBackend(str(wav_dir)),
        spec_sha256="d" * 64,
        spec_ref="specs/note.json",
        adapter=NullAdapter(),
    )

    assert isinstance(report, AnalysisReport)
    assert report.perception.status == "disabled"
    assert report.input.plugin.binary_sha256 == binary_sha256(plugin)
    assert report.input.stimulus.ref == "corpus/midi/c3_sustain_2s.mid"
    c3 = {item.path: item for item in corpus_mod.list_items()}[
        "midi/c3_sustain_2s.mid"
    ]
    assert report.input.stimulus.ref_sha256 == c3.sha256
    assert report.render.backend == "fake-instrument"
    assert report.render.backend_version == "9.9.9"
