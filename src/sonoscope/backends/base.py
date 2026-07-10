"""RenderBackend interface + render data objects (design 4.1).

Format-agnostic render contract so the pedalboard VST3 backend (v1) and a
native CLAP backend (v2) drop in with no schema change. These are pure,
importable definitions with no heavy runtime deps (no pedalboard/librosa),
so they can be built and tested before backend integration.

I1 subset contract: ``RenderMeta`` holds ONLY the backend-owned fields — a
strict subset of ``report.render`` (schema ``RenderBlock``). The F-layer
orchestrator fills the residual ``{backend, backend_version, determinism}``.
``RawState`` is the schema ``RawStateBlock`` (single source of truth), which
maps 1:1 to ``report.input.raw_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..schema.models import RawStateBlock

if TYPE_CHECKING:
    # M3: AudioStimulus.audio is typed np.ndarray, but the contract layer keeps
    # no heavy runtime deps — with `from __future__ import annotations` the
    # annotation is a string, so numpy is only needed for static type checking.
    import numpy as np

# RawState maps 1:1 to report.input.raw_state (design 4.1). Reuse the schema
# model directly rather than duplicating the shape.
RawState = RawStateBlock

ParamKind = Literal["float", "bool", "stepped"]
_VALID_PARAM_KINDS: tuple[str, ...] = ("float", "bool", "stepped")


@dataclass
class ParamInfo:
    """A single introspected plugin parameter (from runtime probe, never
    hardcoded). ``num_steps`` is populated for ``stepped`` params."""

    name: str
    index: int
    kind: ParamKind
    num_steps: Optional[int]
    default: float

    def __post_init__(self) -> None:
        if self.kind not in _VALID_PARAM_KINDS:
            raise ValueError(
                f"ParamInfo.kind must be one of {_VALID_PARAM_KINDS}, "
                f"got {self.kind!r}"
            )


@dataclass
class PluginInfo:
    """Result of ``RenderBackend.probe`` — the plugin's introspected surface."""

    name: str
    format: str
    params: list[ParamInfo]
    input_channels: int
    output_channels: int
    is_instrument: bool
    needs_gui: bool
    latency_samples: int


@dataclass
class RenderMeta:
    """Backend-owned render metadata (I1). Field set is a strict subset of
    schema ``RenderBlock``; the orchestrator adds backend / backend_version /
    determinism."""

    sample_rate_hz: int
    block_size: int
    channels: int
    duration_s: float
    wav_subtype: str
    wav_sha256: str
    render_wall_ms: int
    warnings: list[str] = field(default_factory=list)


# --- runtime stimulus contract (backend-agnostic) ---------------------------
# The stimulus dataclasses live in the contract layer (not a concrete backend)
# so E5/other backends can build/consume them without coupling to the pedalboard
# module. E5 (RenderOrchestrator) selects which stimulus to build from
# PluginInfo.is_instrument (by design / B1 C4); a backend dispatches its render
# overload on the concrete stimulus TYPE it receives.


@dataclass
class MidiStimulus:
    """Instrument-path stimulus: ``mido`` messages + the render duration (B1)."""

    messages: Sequence[Any]
    duration_s: float


@dataclass
class AudioStimulus:
    """Effect-path stimulus: channel-major float32 audio + its sample rate (B1)."""

    audio: np.ndarray
    sample_rate_hz: int


@dataclass
class RenderRequest:
    """Everything a backend needs to render one wav.

    ``stimulus`` and ``param_set`` are the ``Stimulus`` / ``ResolvedParamSet``
    types, built in later tracks (E1/E2); typed ``Any`` here to keep the
    contract layer free of forward dependencies.
    """

    plugin_path: Path
    plugin_format: Literal["vst3", "au"]
    stimulus: Any
    param_set: Any
    sample_rate_hz: int
    block_size: int
    channels: int
    raw_state: Optional[RawState] = None
    seed: Optional[int] = None


@dataclass
class RenderResult:
    """What a backend returns: the wav on disk plus its ``RenderMeta``."""

    wav_path: Path
    render_meta: RenderMeta
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class RenderBackend(Protocol):
    """Format-agnostic render contract (design 4.1)."""

    id: str
    version: str

    def probe(self, plugin_path: Path) -> PluginInfo:
        """Introspect params, I/O, and GUI-need for the plugin at the path."""
        ...

    def render(self, req: RenderRequest) -> RenderResult:
        """Render one wav for the request and return its metadata."""
        ...
