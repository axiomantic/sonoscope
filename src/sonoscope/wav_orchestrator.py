"""analyze --wav engine: native-rate load -> native-unit slice -> per-chunk
resample-to-48k -> deterministic summary + descriptors -> honest provenance ->
ALWAYS-ARRAY report of independently-analyzed chunks (by design).

Mirrors midi_orchestrator.py: this module builds ONLY the engine; cli.py owns
wiring. No existing deterministic/gate code is modified — it is composed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import soxr

from sonoscope import __version__
from sonoscope.descriptors.deriver import derive_descriptors
from sonoscope.errors import InputError
from sonoscope.features.descriptor_gate import (
    ExpectedDescriptors,
    evaluate_descriptors,
)
from sonoscope.features.integrity import compute_integrity
from sonoscope.features.librosa_features import compute_summary, params_sha256
from sonoscope.schema.models import (
    AnalyzedWindow,
    DescriptorGateResult,
    DeterministicBlock,
    InputProvenance,
    LibraryInfo,
    WavAnalysisReport,
    WavChunkAnalysis,
)

# Constants (frozen; determinism = same-env reproducibility, by design).
TARGET_SR: int = 48000
RES_TYPE: str = "soxr_hq"
MAX_CHUNK_SECONDS_DEFAULT: float = 600.0          # 10 min; frozen by design
MIN_ANALYZE_SAMPLES_48K: int = 2048               # == FROZEN_PARAMS["n_fft"] (by design)

AudioSliceUnit = Literal["samples", "seconds"]    # beats/ticks impossible on raw wav

_COMPONENT = "analyze"

# Slice / chunk error taxonomy (by design). The orchestrator raises these typed
# InputError codes; InputError maps to exit 2 via main().
INPUT_WAV_SLICE_INVALID = "INPUT_WAV_SLICE_INVALID"
INPUT_WAV_SLICE_OUT_OF_RANGE = "INPUT_WAV_SLICE_OUT_OF_RANGE"
INPUT_WAV_SLICE_TOO_SHORT = "INPUT_WAV_SLICE_TOO_SHORT"
INPUT_WAV_CHUNK_PARAM_INVALID = "INPUT_WAV_CHUNK_PARAM_INVALID"
INPUT_WAV_CHUNK_TOO_SMALL = "INPUT_WAV_CHUNK_TOO_SMALL"
INPUT_WAV_UNREADABLE = "INPUT_WAV_UNREADABLE"

_LIBRARY_NAME = "librosa"


@dataclass(frozen=True)
class WavFileSource:
    """A standalone wav source for :func:`analyze_wav`."""
    path: Union[str, Path]


@dataclass(frozen=True)
class AudioSlice:
    """Optional native-unit analysis window (half-open [offset, offset+length))."""
    offset: float
    length: Optional[float]
    unit: AudioSliceUnit


def _read_metadata(f: sf.SoundFile) -> tuple[int, int, int, str]:
    """Read wav METADATA from an open handle WITHOUT reading any audio.

    Returns ``(n_native_frames, native_sr, n_channels, subtype)``. This is the
    seek-based loader's cheap header read: the region and chunk tiling are resolved
    from these COUNTS alone, so only the analyzed frames are ever pulled off disk
    (peak memory ~one chunk, not the whole file). ``str(f.subtype)`` is the net-new
    provenance stamp (by design).
    """
    return int(f.frames), int(f.samplerate), int(f.channels), str(f.subtype)


def _read_chunk_native(f: sf.SoundFile, start: int, length: int) -> np.ndarray:
    """Seek to ``start`` and read exactly ``length`` native frames as channel-major
    float32 ``(ch, n)``.

    ``f.seek(start)`` + ``f.read(frames=length, ...)`` reads ONLY this chunk from
    disk — byte-identical to the old whole-file read then ``audio[:, start:start+length]``
    slice, but with peak memory of a single chunk. ``soundfile.read(always_2d=True)``
    returns frame-major ``(n_frames, n_channels)``; compute_summary requires
    channel-major ``(n_channels, n_samples)``. The ``.T`` transpose is LOAD-BEARING
    (C1): without it a stereo file is silently misread. ``ascontiguousarray`` yields a
    fresh C-contiguous buffer the caller owns.
    """
    try:
        f.seek(start)
        data = f.read(frames=length, dtype="float32", always_2d=True)  # (n, ch)
    except Exception as exc:
        raise InputError(
            INPUT_WAV_UNREADABLE,
            f"could not read {length} frames at {start}: {exc}",
            detail={"start": start, "requested": length, "cause": str(exc)},
            component=_COMPONENT,
        ) from exc
    if data.shape[0] != length:
        raise InputError(
            INPUT_WAV_UNREADABLE,
            f"short read: requested {length} frames at {start}, "
            f"got {data.shape[0]}",
            detail={"start": start, "requested": length,
                    "realized": int(data.shape[0])},
            component=_COMPONENT,
        )
    return np.ascontiguousarray(data.T, dtype=np.float32)              # (ch, n)


def _unit_to_native(value: float, unit: AudioSliceUnit, native_sr: int) -> int:
    """Convert a slice bound in ``unit`` to an integer native-sample count.

    Mirrors ``midi_input._unit_to_samples`` (samples/seconds only): ``samples`` is
    ``round(value)``; ``seconds`` is ``round(value * native_sr)``.
    """
    if unit == "samples":
        return round(value)
    return round(value * native_sr)  # seconds


def _resolve_region(
    n_native: int, native_sr: int, slice_spec: Optional[AudioSlice]
) -> tuple[int, int]:
    """Resolve the half-open native region ``[off, off+len)`` clamped to
    ``[0, n_native)``.

    Native-units-FIRST: the region is resolved on the native-sr signal BEFORE any
    resample, so audio that will be discarded is never resampled. Returns
    ``(off_n, len_n)``. Non-finite/negative bounds -> ``INPUT_WAV_SLICE_INVALID``;
    offset at/after EOF or empty-after-clamp -> ``INPUT_WAV_SLICE_OUT_OF_RANGE``.
    """
    if slice_spec is None:
        return 0, n_native

    off = slice_spec.offset
    length = slice_spec.length
    # Non-finite checked BEFORE sign (NaN/inf < 0 are both False; mirror apply_slice).
    if not math.isfinite(off):
        raise InputError(INPUT_WAV_SLICE_INVALID,
                         f"slice offset must be finite, got {off}",
                         detail={"reason": "non_finite_offset", "offset": off},
                         component=_COMPONENT)
    if off < 0:
        raise InputError(INPUT_WAV_SLICE_INVALID,
                         f"slice offset must be >= 0, got {off}",
                         detail={"reason": "negative_offset", "offset": off},
                         component=_COMPONENT)
    if length is not None and not math.isfinite(length):
        raise InputError(INPUT_WAV_SLICE_INVALID,
                         f"slice length must be finite, got {length}",
                         detail={"reason": "non_finite_length", "length": length},
                         component=_COMPONENT)
    if length is not None and length < 0:
        raise InputError(INPUT_WAV_SLICE_INVALID,
                         f"slice length must be >= 0, got {length}",
                         detail={"reason": "negative_length", "length": length},
                         component=_COMPONENT)

    off_n = _unit_to_native(off, slice_spec.unit, native_sr)
    if off_n >= n_native:
        raise InputError(INPUT_WAV_SLICE_OUT_OF_RANGE,
                         f"slice offset {off_n} at/after EOF ({n_native} samples)",
                         detail={"offset_native": off_n, "n_native": n_native},
                         component=_COMPONENT)
    if length is None:
        len_n = n_native - off_n
    else:
        len_n = _unit_to_native(length, slice_spec.unit, native_sr)
    # Clamp to EOF.
    end_n = min(off_n + len_n, n_native)
    len_n = end_n - off_n
    if len_n <= 0:
        raise InputError(INPUT_WAV_SLICE_OUT_OF_RANGE,
                         "empty analysis region after clamp",
                         detail={"offset_native": off_n, "length_native": len_n},
                         component=_COMPONENT)
    return off_n, len_n


def _resample_chunk(
    chunk_native: np.ndarray, native_sr: int
) -> tuple[np.ndarray, Optional[str], Optional[str]]:
    """Resample a channel-major ``(ch, n)`` native chunk to 48k with ``soxr_hq``.

    48k input is a GENUINE no-op (never resample 48k->48k, which would perturb
    samples); provenance ``res_type``/``soxr_version`` are then ``None``. Otherwise
    resample per-channel on the sample axis (``axis=-1``) with EXPLICIT
    ``res_type="soxr_hq"`` (by design) — NO ingest downmix (multichannel reduction
    stays inside compute_summary's I5 policy). The realized 48k length is soxr's
    actual output (may differ from ``round(len*48000/native_sr)`` by +-1) and is
    returned as-is. Returns ``(out48k, res_type, soxr_version)``.
    """
    if native_sr == TARGET_SR:
        # 48k no-op. Copy ONLY when the input could alias a caller-owned buffer
        # or isn't C-contiguous (e.g. a leading-column view like audio[:, cs:ce],
        # whose .base is not None) — that copy still kills the aliasing
        # risk. But _read_chunk_native already yields a FRESH, owned, C-contiguous
        # float32 array per chunk (.base is None), so copying it would be a
        # redundant ~115 MB alloc on a 10-min stereo chunk; pass it through as-is.
        if chunk_native.base is not None or not chunk_native.flags["C_CONTIGUOUS"]:
            return chunk_native.copy(), None, None
        return chunk_native, None, None
    out = librosa.resample(
        chunk_native, orig_sr=native_sr, target_sr=TARGET_SR,
        res_type=RES_TYPE, axis=-1,
    )
    return np.ascontiguousarray(out, dtype=np.float32), RES_TYPE, soxr.__version__


def _min_native(native_sr: int) -> int:
    """Smallest native span that provably resamples to >= n_fft (2048) @48k.

    ``ceil(2048 * native_sr / 48000)`` maps to the ideal 48k length; the ``+2``
    absorbs soxr's +-1 resample-length variance (by design). T6 empirically
    confirmed 44100->48000 of a 44100-sample chunk realizes 48001 samples, so the
    margin is load-bearing, not cosmetic.

    When ``native_sr == TARGET_SR`` the resample is a genuine no-op (by design:
    "48k input is a GENUINE no-op ... never resample 48k->48k"): zero variance,
    so the +-1 margin does not apply and 2048 is the true floor. Applying the
    margin here would wrongly reject an exactly-2048-sample 48k slice.
    """
    if native_sr == TARGET_SR:
        return MIN_ANALYZE_SAMPLES_48K
    return math.ceil(MIN_ANALYZE_SAMPLES_48K * native_sr / TARGET_SR) + 2


def _chunk_bounds(
    start: int, length: int, native_sr: int, max_chunk_seconds: float
) -> list[tuple[int, int]]:
    """Tile the native region ``[start, start + length)`` into contiguous,
    non-overlapping chunks covering it exactly (by design, D2).

    Default is ONE chunk (the whole region); auto-chunking engages only when the
    region exceeds ``max_chunk_seconds`` worth of native samples. Chunk size in
    native units is ``round(max_chunk_seconds * native_sr)``. A trailing remainder
    shorter than ``min_native`` is folded into the preceding chunk (remainder-fold)
    so there is never a silent too-short tail. Threshold validation is fail-loud
    (never clamp). Returns a list of ``(start, end)`` native ranges, length >= 1.
    """
    if not math.isfinite(max_chunk_seconds) or max_chunk_seconds <= 0:
        raise InputError(INPUT_WAV_CHUNK_PARAM_INVALID,
                         f"--max-chunk-seconds must be finite and > 0, got {max_chunk_seconds}",
                         detail={"max_chunk_seconds": max_chunk_seconds},
                         component=_COMPONENT)
    min_native = _min_native(native_sr)
    if length < min_native:
        raise InputError(INPUT_WAV_SLICE_TOO_SHORT,
                         f"analysis region {length} native samples is below the "
                         f"minimum {min_native} (n_fft @48k)",
                         detail={"reason": "region_below_n_fft",
                                 "region_native": length, "min_native": min_native},
                         component=_COMPONENT)
    chunk_native = round(max_chunk_seconds * native_sr)
    if chunk_native < min_native:
        eff_min_seconds = min_native / native_sr
        raise InputError(INPUT_WAV_CHUNK_TOO_SMALL,
                         f"--max-chunk-seconds {max_chunk_seconds} yields a "
                         f"{chunk_native}-sample chunk, below the {min_native}-sample "
                         f"minimum; use at least {eff_min_seconds} seconds",
                         detail={"max_chunk_seconds": max_chunk_seconds,
                                 "chunk_native": chunk_native, "min_native": min_native,
                                 "effective_min_seconds": eff_min_seconds},
                         component=_COMPONENT)

    end = start + length
    bounds: list[tuple[int, int]] = []
    cs = start
    while cs < end:
        ce = min(cs + chunk_native, end)
        bounds.append((cs, ce))
        cs = ce
    # Remainder-fold: only the FINAL chunk can be shorter than chunk_native, and
    # every non-final chunk is exactly chunk_native (>= min_native). If that final
    # chunk is below min_native, merge it into the preceding chunk so no degenerate
    # too-short tail is ever emitted.
    if len(bounds) >= 2 and (bounds[-1][1] - bounds[-1][0]) < min_native:
        last = bounds.pop()
        prev = bounds.pop()
        bounds.append((prev[0], last[1]))
    return bounds


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_provenance(
    *,
    native_sr: int,
    n_channels: int,
    subtype: str,
    res_type: Optional[str],
    soxr_version: Optional[str],
    off_n: int,
    chunk_len_native: int,
    analyzed_48k: int,
    max_chunk_seconds: float,
    chunk_index: int,
    n_chunks: int,
) -> InputProvenance:
    """Stamp the honest per-chunk provenance (by design).

    The native sample rate, subtype, resample res_type/soxr_version, and the
    realized 48k window are all TRUE values — never faked pristine-48k.
    ``analysis_dtype`` and ``channel_reduction`` are frozen literals carried by the
    model default (float32 / mean_spectral_max_peak); they are asserted by the
    schema and pinned by the round-trip tests.
    """
    return InputProvenance(
        original_sample_rate=native_sr,
        n_channels=n_channels,
        source_subtype=subtype,
        resample_res_type=res_type,
        soxr_version=soxr_version,
        analyzed_window=AnalyzedWindow(
            native_offset_samples=off_n,
            native_length_samples=chunk_len_native,
            native_sample_rate=native_sr,
            analyzed_samples_48k=analyzed_48k,
        ),
        max_chunk_seconds=max_chunk_seconds,
        chunk_index=chunk_index,
        n_chunks=n_chunks,
    )


def _analyze_chunk(
    f: sf.SoundFile,
    native_sr: int,
    n_channels: int,
    subtype: str,
    cs: int,
    ce: int,
    max_chunk_seconds: float,
    chunk_index: int,
    n_chunks: int,
    generated_at: str,
    sonoscope_version: str,
    expectation: Optional[ExpectedDescriptors] = None,
    expectation_spec_sha256: Optional[str] = None,
) -> WavChunkAnalysis:
    """Analyze the native chunk ``[cs, ce)`` (seek-read from ``f``) into a
    ``WavChunkAnalysis``.

    seek+read native ``[cs, ce)`` -> resample to 48k -> R1 realized-length guard ->
    compute_summary (@48000) -> compute_integrity -> derive_descriptors -> honest
    provenance -> optional per-chunk descriptor gate (by design). Only this chunk's
    frames are pulled off disk (peak memory ~one chunk). When ``expectation`` is
    supplied, this chunk's ``descriptors`` are compared by ``evaluate_descriptors``
    (``block_kind="audio"``) and the byte-stable per-chunk PASS/RED verdict is
    persisted as its own ``DescriptorGateResult``; the aggregate cross-chunk
    verdict is derived separately by :func:`aggregate_gate`.
    """
    chunk_native = _read_chunk_native(f, cs, ce - cs)
    chunk48k, res_type, soxr_version = _resample_chunk(chunk_native, native_sr)
    analyzed_48k = int(chunk48k.shape[1])
    # R1: last-line honest guard on the REALIZED 48k length. The min_native+2
    # pre-check in _chunk_bounds should normally prevent this, but the resampled
    # length is the ground truth compute_summary's n_fft (2048) depends on.
    if analyzed_48k < MIN_ANALYZE_SAMPLES_48K:
        raise InputError(
            INPUT_WAV_SLICE_TOO_SHORT,
            f"resampled chunk {analyzed_48k} samples < n_fft "
            f"{MIN_ANALYZE_SAMPLES_48K} @48k",
            detail={"reason": "below_n_fft",
                    "analyzed_samples_48k": analyzed_48k,
                    "min": MIN_ANALYZE_SAMPLES_48K},
            component=_COMPONENT,
        )
    summary_result = compute_summary(chunk48k, TARGET_SR)  # 48k guard passes honestly
    integrity = compute_integrity(chunk48k)
    deterministic = DeterministicBlock(
        library=LibraryInfo(
            name=_LIBRARY_NAME,
            version=librosa.__version__,
            params_sha256=params_sha256(),
        ),
        summary=summary_result.summary,
        integrity=integrity,
        notes=summary_result.notes,
    )
    descriptors = derive_descriptors(
        summary_result.summary, is_silent=integrity.all_channels_silent
    )
    descriptor_gate: Optional[DescriptorGateResult] = None
    if expectation is not None:
        evaluation = evaluate_descriptors(descriptors, expectation, block_kind="audio")
        descriptor_gate = DescriptorGateResult(
            verdict=evaluation.verdict,
            reasons=evaluation.reasons,
            spec_sha256=expectation_spec_sha256,
        )
    provenance = _build_provenance(
        native_sr=native_sr, n_channels=n_channels, subtype=subtype,
        res_type=res_type, soxr_version=soxr_version, off_n=cs,
        chunk_len_native=(ce - cs), analyzed_48k=analyzed_48k,
        max_chunk_seconds=max_chunk_seconds, chunk_index=chunk_index,
        n_chunks=n_chunks,
    )
    return WavChunkAnalysis(
        generated_at=generated_at,
        sonoscope_version=sonoscope_version,
        input_provenance=provenance,
        deterministic=deterministic,
        descriptors=descriptors,
        descriptor_gate=descriptor_gate,
    )


def analyze_wav(
    source: WavFileSource,
    *,
    slice_spec: Optional[AudioSlice] = None,
    max_chunk_seconds: Optional[float] = None,
    expectation: Optional[ExpectedDescriptors] = None,
    expectation_spec_sha256: Optional[str] = None,
    generated_at: Optional[str] = None,
    sonoscope_version: str = __version__,
) -> WavAnalysisReport:
    """Compose the full wav pipeline into a ``WavAnalysisReport`` (always len >= 1).

    Native load -> native-unit region resolve -> tile into chunks -> per-chunk
    resample/analyze -> optional per-chunk descriptor gate -> ALWAYS-ARRAY report
    of independently-analyzed chunks (by design). When
    ``expectation`` is supplied (a loader-validated :class:`ExpectedDescriptors`),
    each chunk persists its own PASS/RED ``descriptor_gate`` with
    ``expectation_spec_sha256``; the cross-chunk GREEN/RED aggregate is derived by
    :func:`aggregate_gate` and emitted by the CLI (Task 10).
    """
    path = Path(source.path)
    threshold = (
        MAX_CHUNK_SECONDS_DEFAULT if max_chunk_seconds is None else max_chunk_seconds
    )
    # I/O-domain narrowing (review): ONLY genuine I/O faults map to
    # INPUT_WAV_UNREADABLE. The file OPEN and the header METADATA read are the two
    # load steps that can fail as "unreadable"; each is wrapped narrowly. Everything
    # downstream (region resolve, chunk tiling, per-chunk seek-read, resample,
    # compute_summary/integrity, derive_descriptors, gate eval, model construction)
    # runs OUTSIDE that mapping so an internal defect on a perfectly readable file
    # propagates RAW instead of being misreported as INPUT_WAV_UNREADABLE. Typed
    # slice/chunk InputErrors and the seek/read/short-read guard InputError (raised
    # inside _read_chunk_native) likewise propagate as-is. The handle still closes
    # on any error via the `with` block.
    try:
        handle = sf.SoundFile(str(path))
    except Exception as exc:  # open/malformed-header failure -> exit 2
        raise InputError(
            INPUT_WAV_UNREADABLE,
            f"could not open wav {path}: {exc}",
            detail={"path": str(path), "cause": str(exc)},
            component=_COMPONENT,
        ) from exc
    with handle as f:
        # Open ONCE; read header metadata without pulling audio. Region + chunk
        # tiling are resolved from COUNTS alone, then each chunk is seek-read
        # individually (peak memory ~one chunk, not the whole file). The handle
        # stays open across the sequential per-chunk seeks.
        try:
            n_native, native_sr, n_channels, subtype = _read_metadata(f)
        except Exception as exc:  # header read failure -> exit 2
            raise InputError(
                INPUT_WAV_UNREADABLE,
                f"could not read wav header {path}: {exc}",
                detail={"path": str(path), "cause": str(exc)},
                component=_COMPONENT,
            ) from exc
        off_n, len_n = _resolve_region(n_native, native_sr, slice_spec)
        bounds = _chunk_bounds(off_n, len_n, native_sr, threshold)
        generated_at = generated_at or _now_iso()
        n_chunks = len(bounds)
        chunks = [
            _analyze_chunk(f, native_sr, n_channels, subtype, cs, ce,
                           threshold, i, n_chunks, generated_at, sonoscope_version,
                           expectation=expectation,
                           expectation_spec_sha256=expectation_spec_sha256)
            for i, (cs, ce) in enumerate(bounds)
        ]
    return WavAnalysisReport(chunks)


def aggregate_gate(report: WavAnalysisReport) -> tuple[str, list[int], list[str]]:
    """Cross-chunk descriptor-gate verdict (by design).

    Returns ``(verdict, red_chunks, reasons)`` where ``verdict`` is a NEW
    GREEN/RED token set — ``"RED"`` iff ANY chunk's ``descriptor_gate`` is RED,
    else ``"GREEN"``. This aggregate token is INTENTIONALLY distinct from the
    per-chunk PASS/RED (``DescriptorGateResult.verdict``): the per-chunk gate
    answers "did THIS chunk match?" (PASS/RED); the aggregate answers "did the
    whole file pass the gate?" (GREEN/RED). ``red_chunks`` lists the RED chunk
    indices in order; each entry in ``reasons`` is the chunk's own reason prefixed
    ``"chunk[i] "`` for per-chunk attribution. Chunks with no gate (``None``, i.e.
    no expectation supplied) never contribute RED. The CLI (Task 10) emits this as
    a single stderr line and keys ``--fail-on-red`` off ``verdict == "RED"``.
    """
    red_chunks: list[int] = []
    reasons: list[str] = []
    for i, chunk in enumerate(report.root):
        gate = chunk.descriptor_gate
        if gate is not None and gate.verdict == "RED":
            red_chunks.append(i)
            reasons.extend(f"chunk[{i}] {reason}" for reason in gate.reasons)
    verdict = "RED" if red_chunks else "GREEN"
    return verdict, red_chunks, reasons
