"""Pure, deterministic one-line ``descriptors`` summary renderer (by design).

The ``measured:`` clause renders, in order, the gated measured terms (bare
name), then the hybrid feel-terms (bare name), then the readout terms rendered
with their value (``tempo-audio`` -> ``"128 BPM"``, integer-rounded;
``rhythmic-density`` -> ``"6.0 onsets/s"``, one decimal). When both measured and
hybrid are empty the clause renders ``"measured: (none)"``. The
``; advisory: ...`` clause is appended only when ``advisory`` is non-empty.

No I/O, clock, or RNG: identical inputs -> byte-identical string.
"""

from __future__ import annotations

from sonoscope.schema.models import (
    AdvisoryDescriptor,
    HybridDescriptor,
    MeasuredDescriptor,
)


def _render_readout(descriptor: MeasuredDescriptor) -> str:
    if descriptor.term == "tempo-audio":
        return f"{round(descriptor.value)} BPM"
    if descriptor.term == "rhythmic-density":
        return f"{descriptor.value:.1f} onsets/s"
    # ── C2 MIDI measured value-readouts (additive; audio terms above untouched) ──
    if descriptor.term == "note-density":
        return f"{descriptor.value:.2f} notes/s"
    if descriptor.term == "register":
        return f"note {descriptor.value:.1f}"
    if descriptor.term == "pitch-range":
        return f"{descriptor.value:.0f} st"
    if descriptor.term == "polyphony":
        return f"{descriptor.value:.0f} voices"
    if descriptor.term == "velocity-dynamics":
        return f"velocity std {descriptor.value:.1f}"
    if descriptor.term == "ioi":
        return f"{descriptor.value:.3f}s IOI"
    # No other readout terms exist; fall back to the bare name.
    return descriptor.term


def render_summary(
    measured: list[MeasuredDescriptor],
    hybrid: list[HybridDescriptor],
    advisory: list[AdvisoryDescriptor],
) -> str:
    """Render the one-line human summary. Pure and deterministic."""
    gated = [m.term for m in measured if m.direction in ("high", "low")]
    hybrid_terms = [h.term for h in hybrid]
    readouts = [_render_readout(m) for m in measured if m.direction == "value"]

    parts = gated + hybrid_terms + readouts
    measured_clause = "measured: " + (", ".join(parts) if parts else "(none)")

    if advisory:
        advisory_clause = ", ".join(a.term for a in advisory)
        return f"{measured_clause}; advisory: {advisory_clause}"
    return measured_clause
