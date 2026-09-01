"""AnalysisOrchestrator — the ``analyze`` command's engine (Task F1, by design).

``wav + render-context -> AnalysisReport``. It runs the deterministic ground-truth
layer FIRST (D1 summary + D2 integrity -> ``deterministic``; D3 -> ``tripwires``),
then — only if perception is enabled — calls the advisory adapter (default the
always-present C4 :class:`NullAdapter`) for the ``perception`` block, and finally
assembles the full versioned :class:`AnalysisReport` (C1) with every hash/stamp.

Design invariants enforced here:

- **Deterministic-first ordering (by design).** ``deterministic`` and
  ``tripwires`` are computed and populated BEFORE the perception adapter is ever
  called, so a useful, gate-able report exists even when perception fails.
- **Perception is never fatal (by design).** A perception adapter crash OR a
  60 s timeout degrades to ``perception.status == "error"`` and the loop
  CONTINUES (exit 0). The advisory layer can never abort the analysis; only the
  ground-truth layer failing is fatal (``AnalysisError``, exit 4).
- **M7 errors[] aggregation authority.** ``report.render.warnings`` is the
  backend's own low-risk coercion log copied verbatim — MINUS the internal
  ``seed-forwarded:`` machine record (E3 stopgap), which is filtered so the
  internal prefix never leaks to users. ``report.errors[]`` is the cross-cutting
  non-fatal issue list that F1 aggregates; in F1's layer it is populated by
  perception degradation (severity ``error``, component ``perception``). Fatal
  conditions never land here — they raise a :class:`SonoscopeError` (C5) and
  produce the fatal envelope.
- **InputBlock assembly (E5 handoff).** ``plugin.binary_sha256`` is computed via
  the backend's :func:`binary_sha256` (E5's :class:`RenderOutcome` does not carry
  it). An inline-notes stimulus has no pinned corpus ref (E5 returns
  ``ref_sha256 is None``); the required-``str`` C1 :class:`StimulusRef` fields are
  satisfied with the documented ``ref="inline"`` / ``ref_sha256=""`` sentinel
  WITHOUT modifying the frozen C1 contract.
- **Determinism sub-block (by design).** E5 leaves ``render.determinism`` deferred to
  the F-layer. F1 does NOT depend on the F2 floors engine, so a single-shot
  ``analyze`` synthesizes a ``noise_floor_measured == False`` placeholder
  (``repeats == 1``, no per-feature floors); a caller that HAS measured floors
  (H1/F3, wiring F2) may inject a fully-populated :class:`RenderDeterminism`.

CLI-wiring boundary (I6): F1 builds only the engine; wiring ``analyze --wav`` /
``analyze --plugin/--spec`` to it is owned by H1.

Note on ``analyze --wav`` (bare-wav mode): a pre-rendered wav with no plugin
cannot populate a truthful ``input.plugin`` — C1 requires ``plugin.format`` to be
``vst3``/``au`` and a full render provenance block. That path is intentionally NOT
implemented here; see the F1 handoff notes. The two engines below both require a
render context (an :class:`RenderOutcome` or an E5 render), which always carries a
real plugin identity.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import librosa
import numpy as np
import soundfile as sf

from sonoscope import __version__
from sonoscope.backends.base import RenderBackend
from sonoscope.backends.pedalboard_vst3 import binary_sha256
from sonoscope.descriptors.advisory import produce_advisory
from sonoscope.descriptors.deriver import derive_descriptors
from sonoscope.descriptors.summary import render_summary
from sonoscope.errors import AnalysisError, InputError
from sonoscope.features.integrity import compute_integrity
from sonoscope.features.librosa_features import compute_summary, params_sha256
from sonoscope.features.tripwires import evaluate_tripwires
from sonoscope.perception.base import PerceptionAdapter
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.render_orchestrator import RenderOutcome, render
from sonoscope.schema.models import (
    AnalysisReport,
    DeterministicBlock,
    DeterminismFloors,
    ErrorItem,
    InputBlock,
    LibraryInfo,
    ParamSetRef,
    PerceptionBlock,
    PluginRef,
    RawStateBlock,
    RenderBlock,
    RenderDeterminism,
    StimulusRef,
)
from sonoscope.spec import Spec

_T = TypeVar("_T")

# --- constants (by design) ---------------------------------------------------

#: Perception hard timeout (by design). On expiry the call is
#: ABANDONED (a native inference cannot be safely killed from Python; the daemon
#: watchdog thread is discarded) and perception degrades to status "error".
PERCEPTION_TIMEOUT_S: float = 60.0

#: Inline-notes stimulus sentinel (E5 handoff): an inline-notes render has no
#: pinned corpus ref, so E5 returns ``ref_sha256 is None``. C1's StimulusRef.ref /
#: ref_sha256 are required ``str``; these documented sentinels satisfy them WITHOUT
#: changing the frozen C1 contract. ``ref="inline"`` marks the source; the empty
#: ``ref_sha256`` marks "no pinned bytes to certify".
INLINE_STIMULUS_REF: str = "inline"
INLINE_STIMULUS_REF_SHA256: str = ""

#: E3 records the forwarded render seed into ``RenderMeta.warnings`` under this
#: prefix. It is a machine record, not a user-facing coercion warning, so F1
#: filters it out of ``report.render.warnings`` (see the render_orchestrator
#: module docstring). Kept in sync with ``pedalboard_vst3.seed_record_warnings``.
SEED_WARNING_PREFIX: str = "seed-forwarded:"

#: errors[] code for a perception degradation (crash/timeout). severity "error",
#: component "perception" (degraded-but-completed analysis).
PERCEPTION_DEGRADED_CODE: str = "PERCEPTION_DEGRADED"

#: Fatal code when the deterministic ground-truth layer cannot process the wav
#: (unreadable wav / numeric fault) -> AnalysisError, exit 4.
DETERMINISTIC_ANALYSIS_FAILED_CODE: str = "DETERMINISTIC_ANALYSIS_FAILED"

#: Fatal code when the plugin bundle cannot be hashed (missing/unreadable path)
#: -> InputError, exit 2. A bad plugin path is an INPUT-contract failure (C5:
#: missing/unreadable input), mapped to a typed error instead of a raw OSError.
PLUGIN_BINARY_UNREADABLE_CODE: str = "PLUGIN_BINARY_UNREADABLE"

_LIBRARY_NAME = "librosa"
_ANALYZE_COMPONENT = "analyze"
_PERCEPTION_COMPONENT = "perception"


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 ``...Z`` stamp (``generated_at``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- deterministic ground-truth layer ----------------------------------------


def _load_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a rendered wav as channel-major float32 ``(channels, frames)`` + sr.

    soundfile reads frame-major ``(frames, channels)``; transpose to the
    channel-major layout D1/D2 expect (mirrors ``render_orchestrator._load_audio``).
    """
    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(data.T, dtype=np.float32)
    return audio, int(sample_rate)


def _deterministic_block(wav_path: Path) -> tuple[DeterministicBlock, object]:
    """Run the D1 summary + D2 integrity layer -> ``deterministic`` block.

    The deterministic layer is GROUND TRUTH: any failure to read/analyze the wav
    is a FATAL :class:`AnalysisError` (exit 4), never a silent/partial
    report. Returns the block plus the raw summary (D3 needs it for tripwires).
    """
    try:
        audio, sample_rate = _load_wav(wav_path)
        summary_result = compute_summary(audio, sample_rate)
        integrity = compute_integrity(audio)
    except AnalysisError:
        raise
    except Exception as exc:  # deliberate ground-truth boundary -> exit 4
        raise AnalysisError(
            DETERMINISTIC_ANALYSIS_FAILED_CODE,
            f"deterministic feature layer failed on {wav_path}: {exc}",
            detail={"wav_path": str(wav_path), "cause": str(exc)},
            component=_ANALYZE_COMPONENT,
        ) from exc

    block = DeterministicBlock(
        library=LibraryInfo(
            name=_LIBRARY_NAME,
            version=librosa.__version__,
            params_sha256=params_sha256(),
        ),
        summary=summary_result.summary,
        integrity=integrity,
        notes=summary_result.notes,
    )
    return block, summary_result.summary


# --- perception layer (advisory, never fatal) --------------------------------


def _call_with_timeout(fn: Callable[[], _T], timeout_s: float) -> _T:
    """Run ``fn`` on a daemon watchdog thread; raise ``TimeoutError`` on expiry.

    On timeout the thread is ABANDONED (a native inference call is not safely
    interruptible from Python; the daemon thread runs to completion in the
    background and its result is discarded) — the design's abandon-and-discard
    watchdog (by design). Exceptions raised by ``fn`` propagate to the caller.
    """
    box: dict[str, _T] = {}
    err: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            err["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"call exceeded {timeout_s}s and was abandoned")
    if "error" in err:
        raise err["error"]
    return box["value"]


def _perception_block(
    adapter: PerceptionAdapter,
    wav_path: Path,
    deterministic: DeterministicBlock,
    timeout_s: float,
) -> tuple[PerceptionBlock, Optional[ErrorItem]]:
    """Call the advisory adapter, degrading a crash/timeout to status "error".

    Deliberate graceful-degradation boundary: ANY adapter failure (an
    exception it raises, or the watchdog timeout) yields a ``status == "error"``
    block AND a cross-cutting ``errors[]`` entry, while the analysis loop
    continues. Perception is advisory and can never abort the run.
    """
    # Capture adapter.id BEFORE the try so a pathological adapter whose ``id``
    # dereference raises cannot escape the never-fatal boundary via the handler
    # (by design).
    adapter_id = adapter.id
    try:
        block = _call_with_timeout(
            lambda: adapter.describe(wav_path, deterministic), timeout_s
        )
    except Exception:  # noqa: BLE001 - perception-never-fatal boundary
        error = ErrorItem(
            code=PERCEPTION_DEGRADED_CODE,
            message="perception adapter failed; analysis continued without it",
            detail={"adapter": adapter_id},
            severity="error",
            component=_PERCEPTION_COMPONENT,
        )
        return PerceptionBlock(status="error", grounding="none"), error
    # An adapter may RETURN a status "error" block WITHOUT raising (e.g. G1's
    # cooperative per-token timeout). That is the same degradation as the raise
    # path, so it MUST also produce the cross-cutting errors[] entry (Gemini
    # review). Only status "error" degrades; a returned
    # "unavailable"/"disabled" block is graceful non-degradation and gets NO
    # errors[] entry. The raise path above already returned early, so this check
    # cannot double-emit.
    if block.status == "error":
        error = ErrorItem(
            code=PERCEPTION_DEGRADED_CODE,
            message="perception adapter failed; analysis continued without it",
            detail={"adapter": adapter_id},
            severity="error",
            component=_PERCEPTION_COMPONENT,
        )
        return block, error
    return block, None


# --- render.determinism synthesis (F1 has no F2 dep) --------------------------


def _not_measured_determinism(
    outcome: RenderOutcome, binary_hash: str, generated_at: str
) -> RenderDeterminism:
    """Synthesize a "no floor measured" determinism sub-block for a single render.

    F1 does not depend on the F2 floors engine, and a single ``analyze`` render
    cannot measure a nondeterminism floor (that needs N renders). This honest
    placeholder declares ``noise_floor_measured == False``, ``repeats == 1``, no
    bit-identical claim, and an empty per-feature floors map. A caller that HAS
    measured floors injects a fully-populated block instead.
    """
    floors = DeterminismFloors(
        generated_at=generated_at,
        binary_sha256=binary_hash,
        patch_class=outcome.resolved.patch_class,
        resolved_sha256=outcome.resolved.resolved_sha256,
        stimulus_ref=outcome.resolved.stimulus.ref or INLINE_STIMULUS_REF,
        repeats=1,
        is_bit_identical=False,
        floors={},
    )
    return RenderDeterminism(
        repeats=1,
        is_bit_identical=False,
        patch_class=outcome.resolved.patch_class,
        noise_floor_measured=False,
        floors_ref="",
        floors=floors,
    )


# --- InputBlock / RenderBlock assembly (E5 handoff) ---------------------------


def _stimulus_ref_fields(outcome: RenderOutcome) -> tuple[str, str]:
    """Resolve the ``StimulusRef`` ref / ref_sha256, using the inline sentinel.

    A corpus-ref stimulus carries the E5-verified pinned ``ref_sha256``. An
    inline-notes stimulus has no pinned ref (E5 returns ``ref_sha256 is None``);
    it uses the documented ``ref="inline"`` / ``ref_sha256=""`` sentinel.
    """
    stim = outcome.resolved.stimulus
    if stim.ref is not None:
        return stim.ref, (outcome.ref_sha256 or INLINE_STIMULUS_REF_SHA256)
    return INLINE_STIMULUS_REF, INLINE_STIMULUS_REF_SHA256


def _input_block(
    outcome: RenderOutcome,
    plugin_path: Path,
    plugin_format: Literal["vst3", "au"],
    binary_hash: str,
    spec_ref: str,
    spec_sha256: str,
) -> InputBlock:
    """Assemble ``report.input``. ``binary_sha256`` is F1-computed (E5's
    RenderOutcome does not carry it); ``raw_state`` is uncaptured in v1 (capture
    tooling is deferred)."""
    meta = outcome.render_meta
    stim_ref, stim_ref_sha256 = _stimulus_ref_fields(outcome)
    return InputBlock(
        plugin=PluginRef(
            path=str(plugin_path),
            format=plugin_format,
            binary_sha256=binary_hash,
            backend=outcome.backend,
        ),
        stimulus=StimulusRef(
            kind=outcome.resolved.stimulus.kind,
            ref=stim_ref,
            ref_sha256=stim_ref_sha256,
            sample_rate_hz=meta.sample_rate_hz,
            duration_s=meta.duration_s,
        ),
        param_set=ParamSetRef(
            ref=spec_ref,
            spec_sha256=spec_sha256,
            resolved_sha256=outcome.resolved.resolved_sha256,
        ),
        raw_state=RawStateBlock(captured=False),
    )


def _render_block(
    outcome: RenderOutcome, determinism: RenderDeterminism
) -> RenderBlock:
    """Assemble ``report.render`` from the E5 RenderMeta + orchestrator
    residuals, filtering the internal ``seed-forwarded:`` record out of the
    user-facing ``warnings`` (M2/M7)."""
    meta = outcome.render_meta
    user_warnings = [
        w for w in meta.warnings if not w.startswith(SEED_WARNING_PREFIX)
    ]
    return RenderBlock(
        sample_rate_hz=meta.sample_rate_hz,
        block_size=meta.block_size,
        channels=meta.channels,
        duration_s=meta.duration_s,
        wav_subtype=meta.wav_subtype,  # type: ignore[arg-type]
        backend=outcome.backend,
        backend_version=outcome.backend_version,
        wav_sha256=meta.wav_sha256,
        render_wall_ms=meta.render_wall_ms,
        determinism=determinism,
        warnings=user_warnings,
    )


# --- public engines ---------------------------------------------------------


def analyze_render_outcome(
    outcome: RenderOutcome,
    *,
    plugin_path: Path,
    spec_sha256: str,
    plugin_format: Literal["vst3", "au"] = "vst3",
    spec_ref: str = "inline",
    adapter: Optional[PerceptionAdapter] = None,
    perception_enabled: bool = True,
    determinism: Optional[RenderDeterminism] = None,
    perception_timeout_s: float = PERCEPTION_TIMEOUT_S,
    generated_at: Optional[str] = None,
    sonoscope_version: str = __version__,
) -> AnalysisReport:
    """Assemble a full :class:`AnalysisReport` from an E5 render context.

    Deterministic-first: the ground-truth ``deterministic`` + ``tripwires`` blocks
    are computed BEFORE perception, so the report is useful even when the advisory
    adapter fails. Perception (default the C4 :class:`NullAdapter`) is called only
    when ``perception_enabled``; a crash or ``perception_timeout_s`` timeout
    degrades to ``status == "error"`` + an ``errors[]`` entry and the loop
    continues (never fatal). ``determinism`` may be injected (measured
    floors from F2/H1); when ``None`` a not-measured single-render placeholder is
    synthesized. A deterministic-layer failure raises
    :class:`AnalysisError` (exit 4).
    """
    generated_at = generated_at or _now_iso()
    # Guard the binary hash so a missing/unreadable plugin bundle maps to the
    # typed-error contract (InputError, exit 2 — an INPUT-contract failure per
    # C5) instead of a raw FileNotFoundError/OSError escaping.
    try:
        binary_hash = binary_sha256(plugin_path)
    except OSError as exc:
        raise InputError(
            PLUGIN_BINARY_UNREADABLE_CODE,
            f"plugin bundle is missing or unreadable: {plugin_path}: {exc}",
            detail={"plugin_path": str(plugin_path), "cause": str(exc)},
            component=_ANALYZE_COMPONENT,
        ) from exc

    # GROUND TRUTH FIRST (deterministic + tripwires), before any
    # perception call, so a useful report survives a perception failure.
    deterministic, summary = _deterministic_block(outcome.wav_path)
    tripwires = evaluate_tripwires(
        summary,
        deterministic.integrity,
        outcome.resolved.stimulus.kind,
        outcome.resolved.expected_audio,
    )

    # Advisory perception (never fatal). Aggregate degradation into
    # the cross-cutting errors[] list (M7).
    errors: list[ErrorItem] = []
    if not perception_enabled:
        perception = PerceptionBlock(status="disabled", grounding="none")
    else:
        perception, perception_error = _perception_block(
            adapter or NullAdapter(),
            outcome.wav_path,
            deterministic,
            perception_timeout_s,
        )
        if perception_error is not None:
            errors.append(perception_error)

    # DESCRIPTORS — measured/hybrid deriver runs ALWAYS and NEVER depends on
    # advisory or perception status (pure, deterministic).
    descriptors = derive_descriptors(
        summary, is_silent=deterministic.integrity.all_channels_silent
    )

    # Advisory is best-effort; failure degrades to measured-only, exit 0.
    # produce_advisory returns (advisory, coverage, dropped, err).
    advisory, adv_cov, adv_dropped, adv_err = produce_advisory(perception)
    if adv_err is not None:
        errors.append(adv_err)

    # Stamp coverage provenance onto the library whenever it was computed
    # (coverage is not None iff there were candidate terms to map, even if zero
    # matched). Nested model_copy(update=) bypasses validation, so only
    # well-typed values are passed.
    if adv_cov is not None:
        descriptors = descriptors.model_copy(
            update={
                "library": descriptors.library.model_copy(
                    update={
                        "advisory_coverage": adv_cov,
                        "advisory_dropped": adv_dropped,
                    }
                ),
            }
        )

    # Attach the mapped advisory terms + rewrite the human summary only when at
    # least one term mapped. When advisory == [] the measured-only summary from
    # derive_descriptors is kept as-is (empty-advisory case).
    if advisory:
        descriptors = descriptors.model_copy(
            update={
                "advisory": advisory,
                "summary": render_summary(
                    descriptors.measured, descriptors.hybrid, advisory
                ),
            }
        )

    # Attach render.determinism (injected measured floors, or the
    # not-measured single-render placeholder since F1 has no F2 dep).
    render_determinism = determinism or _not_measured_determinism(
        outcome, binary_hash, generated_at
    )

    # Assemble the versioned report (all hashes/stamps).
    return AnalysisReport(
        generated_at=generated_at,
        sonoscope_version=sonoscope_version,
        input=_input_block(
            outcome, plugin_path, plugin_format, binary_hash, spec_ref, spec_sha256
        ),
        render=_render_block(outcome, render_determinism),
        deterministic=deterministic,
        tripwires=tripwires,
        perception=perception,
        descriptors=descriptors,
        errors=errors,
    )


def analyze_plugin_spec(
    spec: Spec,
    plugin_path: Path,
    backend: RenderBackend,
    *,
    spec_sha256: str,
    plugin_format: Literal["vst3", "au"] = "vst3",
    spec_ref: str = "inline",
    adapter: Optional[PerceptionAdapter] = None,
    perception_enabled: bool = True,
    determinism: Optional[RenderDeterminism] = None,
    perception_timeout_s: float = PERCEPTION_TIMEOUT_S,
    generated_at: Optional[str] = None,
    sonoscope_version: str = __version__,
    **render_kwargs: object,
) -> AnalysisReport:
    """Render via E5 then assemble the report — the ``analyze --plugin/--spec``
    engine.

    Thin composition of the two already-tested layers: E5
    :func:`sonoscope.render_orchestrator.render` (resolve -> subprocess render ->
    RenderOutcome) and :func:`analyze_render_outcome` (deterministic-first
    assembly). ``**render_kwargs`` forwards E5's optional ``raw_state`` /
    ``corpus_root`` / ``manifest_path`` arguments unchanged.
    """
    outcome = render(
        spec,
        plugin_path,
        backend,
        plugin_format=plugin_format,
        **render_kwargs,  # type: ignore[arg-type]
    )
    return analyze_render_outcome(
        outcome,
        plugin_path=plugin_path,
        spec_sha256=spec_sha256,
        plugin_format=plugin_format,
        spec_ref=spec_ref,
        adapter=adapter,
        perception_enabled=perception_enabled,
        determinism=determinism,
        perception_timeout_s=perception_timeout_s,
        generated_at=generated_at,
        sonoscope_version=sonoscope_version,
    )
