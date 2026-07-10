"""Advisory producer (by design, Task T5) — cycle-1 curated-map form.

Two-stage producer whose deterministic stage 2 (``map_freeform_to_advisory``) is a
PURE function: a fixed description string always yields the same
``(advisory, matched, total)``. Only stage 1 (Qwen generating the freeform
description) is non-deterministic, and it is reused from the perception block already
computed at the orchestrator hook (no second Qwen call).

``produce_advisory`` is the never-fatal boundary (by design): a not-ok perception
or an unexpected mapping error degrades to measured-only (exit 0). A mapping-stage
exception is reported as a dedicated ``ADVISORY_DEGRADED`` *warning* ErrorItem — never
the perception path's ``PERCEPTION_DEGRADED`` — so the two degradation sources are not
conflated.

Candidate-extraction lexicon (``_CANDIDATE_SURFACE_FORMS``) is SCOPE-DEFERRED /
NON-FROZEN for cycle 1 (F6): a working set sufficient for the tests + worked example.
It grows per the living playbook; the ``coverage >= 0.60`` bar is a tunable quality
signal, never a gate (advisory is never fatal).
"""

from __future__ import annotations

from typing import Optional

from sonoscope.descriptors.vocab import (
    ADVISORY_BASE_CONFIDENCE,
    CURATED_SYNONYM_MAP,
)
from sonoscope.schema.models import AdvisoryDescriptor, ErrorItem, PerceptionBlock

# Dedicated, non-fatal advisory degradation code (by design). Deliberately NOT the
# perception path's PERCEPTION_DEGRADED_CODE so an advisory-mapping failure is not
# confused with a perception-adapter failure.
ADVISORY_DEGRADED_CODE: str = "ADVISORY_DEGRADED"


def _normalize(text: str) -> list[str]:
    """Deterministic normalization: lowercase, punctuation -> space, collapse
    whitespace, split into tokens. Pure."""
    lowered = text.lower()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered)
    return cleaned.split()


# Normalized surface form -> canonical bounded term. The curated map's human-readable
# keys ("outer space", "trance-like") are normalized ONCE here so matching happens on
# a consistent token basis. Deterministic (source dict is fixed).
_NORMALIZED_SYNONYM_MAP: dict[tuple[str, ...], str] = {
    tuple(_normalize(surface)): canonical
    for surface, canonical in CURATED_SYNONYM_MAP.items()
}

# Recognized-but-unmapped candidate surface forms (NON-FROZEN, F6): forms the LALM
# uses that count toward `total` (candidates seen) but resolve to no canonical term,
# so they are dropped. Kept small + sufficient for the worked example; grows per the
# living playbook.
_UNMAPPED_SURFACE_FORMS: tuple[str, ...] = (
    "brooding",
    "warbly",
)

# Candidate lexicon scanned in the extraction step: every mapped surface form (so a
# mappable term is always a candidate) plus the recognized-but-unmapped forms.
_CANDIDATE_SURFACE_FORMS: frozenset[tuple[str, ...]] = frozenset(
    _NORMALIZED_SYNONYM_MAP
) | frozenset(tuple(_normalize(f)) for f in _UNMAPPED_SURFACE_FORMS)


def _phrase_position(tokens: list[str], phrase: tuple[str, ...]) -> Optional[int]:
    """Lowest start index where `phrase` appears contiguously in `tokens`, else None."""
    n = len(phrase)
    if n == 0:
        return None
    # Compare list slices against a list built once from `phrase` (behavior-preserving
    # optimization): avoids re-tupling each window. `list == list` has identical
    # element-wise semantics to the prior `tuple(window) == phrase` comparison.
    phrase_list = list(phrase)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_list:
            return i
    return None


def map_freeform_to_advisory(
    description: str,
) -> tuple[list[AdvisoryDescriptor], int, int]:
    """PURE stage-2 map: freeform description -> ``(advisory, matched, total)``.

    Bit-identical for a fixed string. ``total`` = distinct candidate surface forms
    found (against ``_CANDIDATE_SURFACE_FORMS``); ``matched`` = distinct canonical
    bounded terms produced. Emission order is mapping order = first appearance of each
    canonical term in the (normalized) description. Every emitted descriptor carries
    ``source == "lalm-mapped"`` and ``confidence == ADVISORY_BASE_CONFIDENCE``.
    """
    tokens = _normalize(description)

    # Candidates: each distinct known surface form present, ordered by first
    # appearance (deterministic; frozenset iteration order is erased by the sort).
    found: list[tuple[int, tuple[str, ...]]] = []
    for surface in _CANDIDATE_SURFACE_FORMS:
        pos = _phrase_position(tokens, surface)
        if pos is not None:
            found.append((pos, surface))
    found.sort()
    total = len(found)

    advisory: list[AdvisoryDescriptor] = []
    seen_canonical: set[str] = set()
    for _pos, surface in found:
        canonical = _NORMALIZED_SYNONYM_MAP.get(surface)
        if canonical is None or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        advisory.append(
            AdvisoryDescriptor(
                term=canonical,
                source="lalm-mapped",
                confidence=ADVISORY_BASE_CONFIDENCE,
            )
        )
    matched = len(seen_canonical)
    return advisory, matched, total


def produce_advisory(
    perception: PerceptionBlock,
) -> tuple[list[AdvisoryDescriptor], Optional[float], Optional[int], Optional[ErrorItem]]:
    """Never-fatal advisory producer. Returns ``(advisory, coverage, dropped, err)``.

    - Perception not ``status=='ok'`` or no description -> ``([], None, None, None)``
      (measured-only, no coverage to report).
    - Normal run -> ``coverage = matched/total`` (None at ``total == 0``),
      ``dropped = None if total == 0 else total - matched`` (F3: no signal => no
      coverage AND no drop count), ``err = None``.
    - Any unexpected error -> ``([], None, None, ErrorItem(ADVISORY_DEGRADED_CODE,
      severity="warning", component="analyze", ...))``. NEVER raises through the
      boundary; exit 0.
    """
    if perception.status != "ok" or perception.description is None:
        return [], None, None, None

    try:
        advisory, matched, total = map_freeform_to_advisory(perception.description)
    except Exception as exc:  # noqa: BLE001 - advisory-never-fatal boundary
        err = ErrorItem(
            code=ADVISORY_DEGRADED_CODE,
            message="advisory mapping failed; analysis continued without it",
            detail={"error_type": type(exc).__name__},
            severity="warning",
            component="analyze",
        )
        return [], None, None, err

    coverage: Optional[float] = None if total == 0 else matched / total
    dropped: Optional[int] = None if total == 0 else total - matched
    return advisory, coverage, dropped, None
