"""Pydantic v2 models — the single source of truth for the sonoscope
analysis JSON contract (design sections 3.2-3.6).

Field names match the design's JSON keys exactly. All models forbid extra
fields so a stray key is a hard ``ValidationError`` rather than silent drift.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

SCHEMA_VERSION = "1.4.0"  # bumped from 1.3.0; additive/minor (wav-analysis kind + input_provenance)

# --- Enums / literal aliases (design sections 3.2-3.6) ----------------------

PatchClass = Literal["noise_free", "noisy"]
PerceptionStatus = Literal["ok", "unavailable", "disabled", "error"]
Grounding = Literal["advisory-freetext", "structured-vocab", "none"]
TripwireVerdict = Literal["PASS", "RED", "ERROR"]
IterateVerdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]
IterateDirection = Literal["increase", "decrease", "change", "stable"]
Component = Literal[
    "render", "analyze", "perception", "resolver", "corpus", "schema", "cli", "midi"
]
Severity = Literal["warning", "error"]
StimulusKind = Literal[
    "midi", "audio", "impulse", "sweep", "pink_noise", "tone", "silence"
]
PluginFormat = Literal["vst3", "au"]
FloorMethod = Literal["variance", "range"]

# midi-analysis report kind (design section 5). The captured event list is
# ground truth; every verdict is a pure exact-enum function of it.
MidiEventType = Literal["note_on", "note_off"]
MidiVerdict = Literal["PASS", "RED"]  # ERROR is reserved to the fatal envelope
MidiField = Literal["channel", "note", "velocity", "type"]
MidiSource = Literal["plugin", "file"]  # plugin-capture vs standalone .mid


class ExitCode(IntEnum):
    """Process exit codes (design section 3.6 exit-code table)."""

    OK = 0
    USAGE = 1
    INPUT = 2
    RENDER = 3
    ANALYSIS = 4
    ENVIRONMENT = 5


class _Strict(BaseModel):
    """Base: forbid unknown fields so the contract cannot silently drift."""

    model_config = ConfigDict(extra="forbid")


# --- Determinism floors (design section 3.5) --------------------------------


class FloorEntry(_Strict):
    floor: float
    unit: str
    method: FloorMethod
    repeats: int
    timestamp: str
    binary_sha256: str
    patch_class: PatchClass


class DeterminismFloors(_Strict):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["determinism-floors"] = "determinism-floors"
    generated_at: str
    binary_sha256: str
    patch_class: PatchClass
    resolved_sha256: str
    stimulus_ref: str
    repeats: int
    is_bit_identical: bool
    floors: dict[str, FloorEntry]


# --- input block (design section 3.2) ---------------------------------------


class PluginRef(_Strict):
    path: str
    format: PluginFormat
    binary_sha256: str
    backend: str


class StimulusRef(_Strict):
    kind: StimulusKind
    ref: str
    ref_sha256: str
    sample_rate_hz: int
    duration_s: float


class ParamSetRef(_Strict):
    ref: str
    spec_sha256: str
    resolved_sha256: str


class RawStateBlock(_Strict):
    captured: bool
    plugin_binary_sha256: Optional[str] = None
    blob_ref: Optional[str] = None


class InputBlock(_Strict):
    plugin: PluginRef
    stimulus: StimulusRef
    param_set: ParamSetRef
    raw_state: RawStateBlock


# --- render block (design section 3.2) --------------------------------------


class RenderDeterminism(_Strict):
    repeats: int
    is_bit_identical: bool
    patch_class: PatchClass
    noise_floor_measured: bool
    floors_ref: str
    floors: DeterminismFloors


class RenderBlock(_Strict):
    sample_rate_hz: int
    block_size: int
    channels: int
    duration_s: float
    wav_subtype: Literal["PCM_F32"] = "PCM_F32"
    backend: str
    backend_version: str
    wav_sha256: str
    render_wall_ms: int
    determinism: RenderDeterminism
    warnings: list[str] = Field(default_factory=list)


# --- deterministic block (design section 3.2) -------------------------------


class LibraryInfo(_Strict):
    name: str
    version: str
    params_sha256: str


class DeterministicSummary(_Strict):
    duration_s: float
    sample_rate_hz: int
    channels: int
    rms_dbfs: float
    peak_dbfs: float
    crest_factor_db: float
    dc_offset: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    zero_crossing_rate: float
    onset_count: int
    onset_rate_hz: float
    tempo_bpm: Optional[float] = None
    tempo_confidence: Optional[float] = None
    mfcc_mean: list[float]
    mfcc_std: list[float]


class IntegrityBlock(_Strict):
    is_silent: bool
    silence_threshold_dbfs: float
    has_nan: bool
    has_inf: bool
    has_denormal: bool
    clip_count: int
    clip_fraction: float
    dc_offset_exceeds: bool
    dc_offset_threshold: float


class DeterministicBlock(_Strict):
    library: LibraryInfo
    summary: DeterministicSummary
    integrity: IntegrityBlock
    notes: list[str] = Field(default_factory=list)


# --- tripwires block (design section 3.2) -----------------------------------


class TripwireResult(_Strict):
    id: str
    verdict: TripwireVerdict
    detail: Optional[str] = None


class TripwiresBlock(_Strict):
    expected_audio: bool
    results: list[TripwireResult]
    overall: TripwireVerdict


# --- perception block (design section 3.2; nullability per I7) ---------------


class AdapterInfo(_Strict):
    id: str
    model: str
    quant: str
    runtime: str
    model_sha256: str


class PerceptionStructured(_Strict):
    brightness: str
    noisiness: str
    dynamics: str


_OK_REQUIRED = ("adapter", "description", "grounding_map", "disclaimer")


class PerceptionBlock(_Strict):
    status: PerceptionStatus
    grounding: Grounding
    adapter: Optional[AdapterInfo] = None
    description: Optional[str] = None
    structured: Optional[PerceptionStructured] = None
    grounding_map: Optional[dict[str, str]] = None
    disclaimer: Optional[str] = None

    @model_validator(mode="after")
    def _require_adapter_output_when_ok(self) -> "PerceptionBlock":
        # I7: a status=="ok" block must carry adapter output. This is the only
        # direction strictly enforced here: the four `_OK_REQUIRED` fields are
        # required only when status=="ok". Non-ok statuses
        # (disabled/unavailable/error) are expected to carry none, but that is a
        # convention this validator does NOT enforce. `structured` stays optional
        # even when ok (present only for grounding=="structured-vocab").
        if self.status == "ok":
            missing = [f for f in _OK_REQUIRED if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    "perception.status=='ok' requires non-null "
                    + ", ".join(missing)
                )
        return self


# --- descriptors block (design section 3; grounding is structural) ----------
# Three grounding-separated arrays: the array a descriptor lives in *is* its
# grounding claim (measured = ground truth, hybrid = metric-anchored opinion,
# advisory = labeled AI opinion). No new Grounding literal (design section 4,
# Option A). Additive on AnalysisReport (wired in C1) and MidiAnalysisReport
# (defined, unwired until C2). SCHEMA_VERSION bumped 1.1.0 -> 1.2.0 for this
# additive descriptors block (MINOR; major stays 1).

Direction = Literal["high", "low", "value"]
# "high"/"low" = gated term fired because a metric crossed a threshold in that
# direction. "value" = readout term (tempo-audio, rhythmic-density): carries the
# raw metric value, no threshold.

AdvisorySource = Literal["lalm-mapped"]
# Cycle 1 has exactly one advisory source (Qwen2-Audio freeform -> curated map).
# C7 extends this literal; ap6 owns the extension at C7 integration.


class MeasuredDescriptor(_Strict):
    """Deterministic ground-truth term. Gate-eligible unless estimated is True."""

    term: str
    value: float
    metric: str
    direction: Direction
    threshold: Optional[float] = None
    estimated: bool = False
    confidence: Optional[float] = None


class HybridDescriptor(_Strict):
    """Metric-anchored opinion. Deterministic firing, but not ground truth."""

    term: str
    anchor_metric: str
    anchor_value: float
    direction: Direction
    confidence: Optional[float] = None


class AdvisoryDescriptor(_Strict):
    """Labeled AI opinion mapped onto the bounded vocabulary. Never ground truth."""

    term: str
    source: AdvisorySource
    confidence: Optional[float] = None


class DescriptorsLibrary(_Strict):
    """Provenance sub-block. Mirrors DeterministicBlock.library / LibraryInfo."""

    thresholds_sha256: str
    deriver_version: str
    advisory_coverage: Optional[float] = None
    advisory_dropped: Optional[int] = None


class DescriptorsBlock(_Strict):
    measured: list[MeasuredDescriptor] = Field(default_factory=list)
    hybrid: list[HybridDescriptor] = Field(default_factory=list)
    advisory: list[AdvisoryDescriptor] = Field(default_factory=list)
    summary: str
    library: DescriptorsLibrary


class DescriptorGateResult(_Strict):
    """Persisted descriptor-gate verdict (the `analyze --expect-descriptors` outcome).

    The comparator's PASS/RED verdict + priority-ordered RED reason ids, plus the
    sha256 of the raw operator-authored expectation-spec bytes for provenance.
    Additive-optional on AnalysisReport (SCHEMA_VERSION bumped 1.2.0 -> 1.3.0);
    it does NOT replace the CLI's single-line stderr verdict signal. `reasons` is
    empty for a PASS; `spec_sha256` is None when no expectation spec was supplied.
    """

    verdict: Literal["PASS", "RED"]
    reasons: list[str] = Field(default_factory=list)
    spec_sha256: Optional[str] = None


# --- errors (design section 3.6) --------------------------------------------


class ErrorItem(_Strict):
    code: str
    message: str
    detail: Optional[dict[str, Any]] = None
    severity: Severity
    component: Component


class FatalErrorDetail(_Strict):
    code: str
    message: str
    detail: Optional[dict[str, Any]] = None
    severity: Literal["fatal"] = "fatal"
    component: Component


class FatalError(_Strict):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["fatal-error"] = "fatal-error"
    generated_at: str
    sonoscope_version: str
    error: FatalErrorDetail


# --- top-level analysis report (design section 3.2) -------------------------


class AnalysisReport(_Strict):
    schema_version: str = SCHEMA_VERSION
    generated_at: str
    sonoscope_version: str
    input: InputBlock
    render: RenderBlock
    deterministic: DeterministicBlock
    tripwires: TripwiresBlock
    perception: PerceptionBlock
    descriptors: Optional[DescriptorsBlock] = None  # NEW (C1) — after perception
    # Additive-optional descriptor-gate verdict (None when un-gated); backward
    # compatible with pre-1.3.0 readers/JSON. Audio-only this cycle.
    descriptor_gate: Optional[DescriptorGateResult] = None
    errors: list[ErrorItem] = Field(default_factory=list)


# --- iterate / delta report (design section 3.4) ----------------------------


class Expectation(_Strict):
    metric: str
    direction: IterateDirection
    min_effect: Optional[float] = None


class Delta(_Strict):
    metric: str
    baseline_value: float
    candidate_value: float
    abs_delta: float
    measured_floor: float
    noise_threshold_multiplier: float = 3.0
    noise_threshold: float
    significant: bool
    matches_expectation: bool


class IterateDelta(_Strict):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["iterate-delta"] = "iterate-delta"
    baseline: AnalysisReport
    candidate: AnalysisReport
    expectation: Expectation
    delta: Delta
    verdict: IterateVerdict


# --- midi-analysis report (design section 5) --------------------------------
# A distinct report kind for MIDI-generator plugins, NOT an overload of the
# audio AnalysisReport. Ranges are validated (channel 0-15, note/velocity
# 0-127) so a malformed event is a hard ValidationError, never silent drift.


class MidiEvent(_Strict):
    t_samples: int = Field(ge=0)  # absolute capture position (design section 3)
    t_ticks: int = Field(ge=0)  # @960 PPQ (design section 4 decode rule)
    type: MidiEventType
    channel: int = Field(ge=0, le=15)
    note: int = Field(ge=0, le=127)
    velocity: int = Field(ge=0, le=127)


class MistimedEvent(_Strict):
    # Timing gate is SAMPLES per the ap8 contract; delta_ticks is carried
    # alongside for human readability (design section 6 timing tripwire).
    expected: MidiEvent
    actual: MidiEvent
    delta_samples: int
    delta_ticks: int


class WrongFieldEvent(_Strict):
    # An otherwise-matched pair differing on exactly one field (design section 6
    # wrong-field tripwire; EXACT, zero tolerance).
    expected: MidiEvent
    actual: MidiEvent
    field: MidiField


class ExpectedVsActual(_Strict):
    # Present only when an expected list was supplied (design section 5/6).
    matched: int
    missing: list[MidiEvent]
    extra: list[MidiEvent]
    mistimed: list[MistimedEvent]
    wrong_field: list[WrongFieldEvent]


class MidiIntegrity(_Strict):
    # note_on/note_off pairing over the window (design section 6). A note_on
    # with no matching off -> stuck_notes (the M5 firewall tripwire).
    every_note_on_has_off: bool
    stuck_notes: list[MidiEvent]
    dangling_offs: list[MidiEvent]


class MidiCaptureMeta(_Strict):
    # Capture provenance (design section 5). Fields the orchestrator needs to
    # reproduce and audit the capture; determinism fields (events_sha256,
    # block_size_invariant) are design section 4/6 capture-integrity inputs and
    # default to None for a standalone-file source that has no capture repeats.
    # Divisor / positive-value constraints so a malformed capture is a hard
    # ValidationError, never a silent divide-by-zero in the tick math. C1: the
    # file-source path sets block_size=0 as a deliberate "not-applicable"
    # sentinel (a static .mid file has no processing block size), so block_size
    # is ge=0 (allow 0) NOT gt=0. duration_samples is ge=0: a zero-length
    # window is representable; only negatives are rejected.
    sample_rate: int = Field(gt=0)  # divisor in tick math
    block_size: int = Field(ge=0)  # 0 = file-source sentinel (C1); reject <0
    duration_samples: int = Field(ge=0)  # zero-length window ok; reject <0
    ppq: int = Field(default=960, gt=0)  # divisor
    tempo_bpm: float = Field(gt=0)  # divisor in downstream beat math
    start_position_beats: float
    duration_beats: float
    tsig_num: int = Field(gt=0)
    tsig_den: int = Field(gt=0)  # divisor
    plugin_id: Optional[str] = None
    plugin_name: Optional[str] = None
    source: MidiSource
    binary_sha256: Optional[str] = None  # plugin binary hash (plugin source)
    events_sha256: Optional[str] = None  # byte-identity determinism hash
    block_size_invariant: Optional[bool] = None
    timing_tolerance_samples: int = 1


class MidiBlock(_Strict):
    # The pure-function output (design section 6): ground-truth events + the
    # verdict and priority-ordered reasons[] (each entry a firing tripwire id).
    capture_meta: MidiCaptureMeta
    events: list[MidiEvent]
    expected_vs_actual: Optional[ExpectedVsActual] = None
    integrity: MidiIntegrity
    verdict: MidiVerdict
    reasons: list[str]


class MidiPluginRef(_Strict):
    # A CLAP note-effect plugin captured via the C host (design section 3).
    path: str
    binary_sha256: Optional[str] = None
    plugin_id: Optional[str] = None
    plugin_name: Optional[str] = None


class MidiFileRef(_Strict):
    # A standalone .mid file (the v2 file-source addition, design section 5).
    path: str
    file_sha256: Optional[str] = None


class MidiTransportRef(_Strict):
    # The transport/render slice the capture was driven over (design section 3).
    sample_rate: int
    block_size: int
    tempo_bpm: float
    start_position_beats: float
    duration_beats: float
    tsig_num: int
    tsig_den: int
    playing: bool = True


class MidiExpectedRef(_Strict):
    # Provenance of the optional --expected list (design section 7).
    ref: str
    spec_sha256: Optional[str] = None
    event_count: int


class MidiInputBlock(_Strict):
    # Provenance: exactly one of plugin/file per `source` (design section 5).
    source: MidiSource
    plugin: Optional[MidiPluginRef] = None
    file: Optional[MidiFileRef] = None
    transport: MidiTransportRef
    expected: Optional[MidiExpectedRef] = None

    @model_validator(mode="after")
    def _require_source_ref(self) -> "MidiInputBlock":
        # A plugin source must carry a plugin ref (and no file ref); a file
        # source the inverse. An inconsistent block is a hard error, never a
        # silently mismatched provenance record (mirrors PerceptionBlock).
        if self.source == "plugin":
            if self.plugin is None or self.file is not None:
                raise ValueError(
                    "input.source=='plugin' requires a plugin ref and no file ref"
                )
        else:  # source == "file"
            if self.file is None or self.plugin is not None:
                raise ValueError(
                    "input.source=='file' requires a file ref and no plugin ref"
                )
        return self


class MidiAnalysisReport(_Strict):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["midi-analysis"] = "midi-analysis"
    generated_at: str
    sonoscope_version: str
    input: MidiInputBlock
    midi: MidiBlock
    descriptors: Optional[DescriptorsBlock] = None  # DEFINED — wired in C2
    errors: list[ErrorItem] = Field(default_factory=list)


# --- wav-analysis report kind (design section 10, D1) -----------------------
# A distinct report kind for `analyze --wav <file>`: an ALWAYS-ARRAY report of
# independently-analyzed chunks (WavAnalysisReport = RootModel[list[...]],
# length >= 1). Additive-only; AnalysisReport / MidiAnalysisReport are UNTOUCHED.
# input_provenance is REQUIRED here (and ONLY here) — params_sha256 hashes only
# the frozen params and cannot distinguish resampled-from-44.1k vs native-48k
# input, so the distinguishing truth is carried in input_provenance instead.


class AnalyzedWindow(_Strict):
    # The native-unit analysis window plus the realized 48k length (design
    # section 10.2). Half-open [native_offset_samples, +native_length_samples).
    native_offset_samples: int
    native_length_samples: int
    native_sample_rate: int
    analyzed_samples_48k: int


class InputProvenance(_Strict):
    # NOTE: params_sha256 CANNOT distinguish resampled-from-44.1k vs native-48k
    # input (it hashes only the 8 frozen params) — a KNOWN LIMITATION (design
    # section 10.4). The distinguishing truth is carried here: original_sample_rate
    # + resample_res_type + soxr_version. params_sha256 is NOT labeled "honest".
    original_sample_rate: int
    n_channels: int
    source_subtype: str
    analysis_dtype: Literal["float32"] = "float32"
    resample_res_type: Optional[str] = None
    soxr_version: Optional[str] = None
    channel_reduction: Literal["mean_spectral_max_peak"] = "mean_spectral_max_peak"
    analyzed_window: AnalyzedWindow
    max_chunk_seconds: float
    chunk_index: int
    n_chunks: int


class WavChunkAnalysis(_Strict):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["wav-chunk-analysis"] = "wav-chunk-analysis"
    generated_at: str
    sonoscope_version: str
    input_provenance: InputProvenance
    deterministic: DeterministicBlock  # reused unchanged
    descriptors: DescriptorsBlock  # required — derive_descriptors always runs
    descriptor_gate: Optional[DescriptorGateResult] = None
    errors: list[ErrorItem] = Field(default_factory=list)


class WavAnalysisReport(RootModel[Annotated[list[WavChunkAnalysis], Field(min_length=1)]]):
    """Top-level JSON ARRAY of independently-analyzed chunks (D2). Length >= 1."""
