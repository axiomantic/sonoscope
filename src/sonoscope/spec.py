"""Versioned MIDI / param spec model (Task E2, design §6.2).

The CLI consumes a versioned input spec (JSON/TOML) carrying its own
``spec_version`` — distinct from the analysis-report ``schema_version``
(``sonoscope.schema.models.SCHEMA_VERSION``). This module is the Pydantic
model for that input contract; :mod:`sonoscope.resolver` validates it against a
probed :class:`~sonoscope.backends.base.PluginInfo` and produces the resolved
param vector.

Design §6.2 shape (verbatim keys):

- ``spec_version`` — semver; unknown MAJOR is a hard fail at resolve time.
- ``stimulus`` — a corpus ``ref`` OR inline ``notes`` (exactly one), plus ``kind``.
- ``patch_class`` — ``{noise_free|noisy}`` determinism regime; default ``"noisy"``.
- ``expected_audio`` — optional override; ``null`` ⇒ derive from stimulus kind.
- ``params`` — ``by_name`` (preferred) / ``by_index`` (fallback); values are the
  canonical normalized ``[0.0, 1.0]`` float domain (I8). Range/step validation is
  the resolver's job (not a Pydantic constraint) so an out-of-range value is a
  hard INPUT error, never a silent parse-time coercion.
- ``render`` — sample rate / block size / channels / optional seed.

The type aliases ``PatchClass`` and ``StimulusKind`` are imported from the schema
models (single source of truth) rather than redefined.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sonoscope.schema.models import PatchClass, StimulusKind

#: Version of the input-spec contract this build understands (design §6.2). Its
#: MAJOR gates compatibility in the resolver; distinct from the analysis
#: ``SCHEMA_VERSION``.
SPEC_VERSION = "1.0.0"


class _StrictSpec(BaseModel):
    """Base: forbid unknown fields so a stray spec key is a hard error."""

    model_config = ConfigDict(extra="forbid")


class Note(_StrictSpec):
    """One inline MIDI note (design §6.2 inline-notes form)."""

    pitch: int
    vel: int
    on: float
    off: float


class StimulusSpec(_StrictSpec):
    """Stimulus: a corpus ``ref`` OR inline ``notes`` — exactly one (design §6.2).

    ``kind`` drives ``expected_audio`` derivation (§4.4). The resolver does NOT
    verify a corpus ``ref``'s sha256 here — that is the RenderOrchestrator's job
    (E5); this model only carries the reference forward.
    """

    kind: StimulusKind
    ref: Optional[str] = None
    notes: Optional[list[Note]] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "StimulusSpec":
        has_ref = self.ref is not None
        has_notes = self.notes is not None
        # "stimulus ref (corpus) OR inline notes" — exactly one, never both/neither.
        if has_ref == has_notes:
            raise ValueError(
                "stimulus requires exactly one of 'ref' or 'notes' "
                f"(got ref={self.ref!r}, notes={'set' if has_notes else None})"
            )
        # Inline notes are a MIDI construct; a signal stimulus must use a ref.
        if has_notes and self.kind != "midi":
            raise ValueError(
                f"inline 'notes' is only valid for kind='midi', got {self.kind!r}"
            )
        return self


class ParamsSpec(_StrictSpec):
    """Param references addressable ``by_name`` (preferred) or ``by_index`` (§6.2).

    Values are typed ``float`` with NO range constraint here on purpose: the I8
    ``[0.0, 1.0]`` domain (and step validity) is enforced by the resolver so a
    violation is a hard INPUT error, not a silent Pydantic coercion. ``by_index``
    keys are the string form of the integer param index (JSON/TOML key form).
    """

    by_name: dict[str, float] = Field(default_factory=dict)
    by_index: dict[str, float] = Field(default_factory=dict)


class RenderSpec(_StrictSpec):
    """Render parameters (design §6.2). ``seed`` is forwarded where honored (§8)."""

    sample_rate_hz: int = 48000
    block_size: int = 512
    channels: int = 2
    seed: Optional[int] = None


class Spec(_StrictSpec):
    """The versioned MIDI/param spec (design §6.2). ``patch_class`` defaults to
    ``"noisy"`` — the conservative determinism regime (§6.2, C2)."""

    spec_version: str = SPEC_VERSION
    stimulus: StimulusSpec
    patch_class: PatchClass = "noisy"
    expected_audio: Optional[bool] = None
    params: ParamsSpec = Field(default_factory=ParamsSpec)
    render: RenderSpec = Field(default_factory=RenderSpec)
