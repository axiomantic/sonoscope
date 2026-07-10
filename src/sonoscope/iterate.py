"""Iterate engine — the ``iterate`` command's engine (Task F3, by design).

``iterate`` closes the render -> listen -> feedback loop: it asserts that a defined
parameter change produced the expected, *measurable* deterministic feature delta —
one that clears the plugin's own measured nondeterminism floor. :func:`run_iterate`
is the **significance gate**: it consumes an already-analyzed baseline + candidate
:class:`AnalysisReport` (F1 = render + analyze) and the measured
:class:`DeterminismFloors` (F2 = read/measure the floor for
``(binary_sha256, patch_class)``), computes the R2 noise-thresholded delta,
and returns a verdict of ``PASS`` / ``FAIL`` / ``INCONCLUSIVE``.

CLI-wiring boundary (I6): F3 builds only this engine; wiring the ``iterate``
command (rendering + analyzing both specs via F1, reading/measuring the floor via
F2, then calling :func:`run_iterate`) is owned by H1. This module does NOT touch
``cli.py``.

Design invariants enforced here:

- **Significance is floor-relative, never assumed (by design).** ``noise_threshold`` is
  the resolved absolute product ``measured_floor * noise_threshold_multiplier``
  (default 3x); a delta counts as ``significant`` only when its *magnitude* clears
  that threshold. A change buried in the plugin's own noise is reported
  ``INCONCLUSIVE`` — a distinct, honest outcome, NOT a green. This is the
  false-PASS guard the F3 tests prove.
- **Signed delta, magnitude significance (by design, C1).** ``abs_delta`` records the
  SIGNED difference ``candidate_value - baseline_value`` (the C1 reference stores
  ``-850.3`` for a decrease); direction is recoverable from its sign, while
  significance compares its magnitude to the threshold.
- **Presence-instability -> INCONCLUSIVE (F2 handoff).** F2 records a per-feature
  ``repeats`` = how many of the N floor-measuring renders had the feature present.
  When that shrinks below the full N (an unstable present<->absent feature, e.g.
  octave-mitigated tempo), the measured floor is unreliable — its small/zero value
  must NOT be over-trusted into a confident "significant". Such a metric is
  reported ``INCONCLUSIVE`` rather than a false PASS.
- **No aesthetic judgment (by design).** The engine asserts only measurable, thresholded
  deltas and structural health; it never decides whether a patch "sounds good".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from sonoscope.errors import InputError
from sonoscope.schema.models import (
    AnalysisReport,
    Delta,
    DescriptorsBlock,
    DeterminismFloors,
    Expectation,
    IterateDirection,
    IterateDelta,
    IterateVerdict,
    MeasuredDescriptor,
)

# --- constants (by design) ---------------------------------------------------

#: Default noise-threshold multiplier: ``noise_threshold = measured_floor * 3x``
#: (``noise_threshold_multiplier``). Conservative default so a
#: "significant" delta is provably several times the plugin's measured noise.
DEFAULT_NOISE_THRESHOLD_MULTIPLIER: float = 3.0

#: Component stamp for the typed input errors this engine raises. There is no
#: dedicated ``iterate`` component in the C1 enum; iterate is analysis-domain, so
#: its metric/floor contract failures map to the ``analyze`` component (C1).
_ITERATE_COMPONENT = "analyze"

#: errors[]/fatal codes (UPPER_SNAKE). Raised as :class:`InputError`
#: (exit 2) because each is an INPUT-contract failure on the ``--metric`` the
#: caller supplied — never a silent no-op.
_METRIC_NOT_FOUND_CODE = "ITERATE_METRIC_NOT_FOUND"
_METRIC_UNAVAILABLE_CODE = "ITERATE_METRIC_UNAVAILABLE"
_FLOOR_MISSING_CODE = "ITERATE_FLOOR_MISSING"
_MULTIPLIER_INVALID_CODE = "ITERATE_MULTIPLIER_INVALID"


def _extract_metric(report: AnalysisReport, metric: str) -> float:
    """Resolve a dotted ``metric`` path (e.g. ``deterministic.summary.<feat>``) to
    a numeric value on ``report``.

    Walks Pydantic-model attributes, dict keys (so a ``floors.<key>`` map field
    resolves), and numeric list indices (so an MFCC coefficient path like
    ``deterministic.summary.mfcc_mean.3`` resolves). Every failure is a hard
    :class:`InputError` (exit 2), never a silent fallback:

    - a path segment that does not exist / cannot be descended -> metric not found;
    - a resolved ``None`` (a suppressed feature, e.g. octave-mitigated
      ``tempo_bpm``) or a non-numeric leaf -> metric unavailable (no delta can be
      computed).
    """
    current: object = report
    for part in metric.split("."):
        if isinstance(current, BaseModel):
            if part not in type(current).model_fields:
                raise InputError(
                    _METRIC_NOT_FOUND_CODE,
                    f"metric path segment {part!r} is not a field of "
                    f"{type(current).__name__} (metric {metric!r})",
                    detail={"metric": metric, "segment": part},
                    component=_ITERATE_COMPONENT,
                )
            current = getattr(current, part)
        elif isinstance(current, dict):
            # A dict node (e.g. a ``floors.<key>`` map): a present
            # key descends; a missing key is the SAME typed not-found INPUT error
            # (exit 2) as a missing model field, never a silent fallback.
            if part not in current:
                raise InputError(
                    _METRIC_NOT_FOUND_CODE,
                    f"metric path segment {part!r} is not a key of the dict "
                    f"(metric {metric!r})",
                    detail={"metric": metric, "segment": part},
                    component=_ITERATE_COMPONENT,
                )
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError:
                raise InputError(
                    _METRIC_NOT_FOUND_CODE,
                    f"metric path segment {part!r} is not a valid list index "
                    f"(metric {metric!r})",
                    detail={"metric": metric, "segment": part},
                    component=_ITERATE_COMPONENT,
                ) from None
            try:
                current = current[index]
            except IndexError:
                raise InputError(
                    _METRIC_NOT_FOUND_CODE,
                    f"metric index {index} out of range (metric {metric!r})",
                    detail={"metric": metric, "index": index},
                    component=_ITERATE_COMPONENT,
                ) from None
        else:
            raise InputError(
                _METRIC_NOT_FOUND_CODE,
                f"cannot descend into metric segment {part!r} (metric {metric!r})",
                detail={"metric": metric, "segment": part},
                component=_ITERATE_COMPONENT,
            )

    if current is None:
        raise InputError(
            _METRIC_UNAVAILABLE_CODE,
            f"metric {metric!r} is null in the report (feature suppressed); "
            "no delta can be computed",
            detail={"metric": metric},
            component=_ITERATE_COMPONENT,
        )
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise InputError(
            _METRIC_UNAVAILABLE_CODE,
            f"metric {metric!r} resolved to a non-numeric value "
            f"{type(current).__name__}; iterate compares numeric features only",
            detail={"metric": metric, "kind": type(current).__name__},
            component=_ITERATE_COMPONENT,
        )
    return float(current)


def _matches_expectation(
    direction: IterateDirection, signed_delta: float, significant: bool
) -> bool:
    """Whether the observed change matches the asserted hypothesis ``direction``.

    - ``increase`` / ``decrease`` -> the sign of ``signed_delta`` agrees;
    - ``change`` -> any significant change (either direction) satisfies it;
    - ``stable`` -> the metric did NOT significantly move.

    Note ``matches_expectation`` is a directional record; the final verdict
    combines it with significance (a sub-floor change is ``INCONCLUSIVE``
    regardless).
    """
    if direction == "increase":
        return signed_delta > 0
    if direction == "decrease":
        return signed_delta < 0
    if direction == "change":
        return significant
    # "stable"
    return not significant


def _verdict(
    direction: IterateDirection, significant: bool, matches_expectation: bool
) -> IterateVerdict:
    """The direction-aware verdict.

    ``stable`` is an inverted assertion (reading A): it claims the metric did
    NOT meaningfully move, so it ``PASS``es exactly when the delta is sub-floor
    (``significant is False``) and ``FAIL``s when a supra-floor change occurred. It
    never returns ``INCONCLUSIVE`` — a sub-floor change is the *asserted* outcome,
    not an absence of evidence.

    For the movement directions (``increase`` / ``decrease`` / ``change``) a
    sub-floor change (``significant is False``) is ``INCONCLUSIVE`` — the change was
    smaller than the measured nondeterminism floor, so no claim can be made (this is
    NOT a PASS). Above the floor, the direction decides: ``PASS`` when it matches the
    hypothesis, ``FAIL`` when it moved the wrong way.
    """
    if direction == "stable":
        return "PASS" if not significant else "FAIL"
    if not significant:
        return "INCONCLUSIVE"
    if matches_expectation:
        return "PASS"
    return "FAIL"


def run_iterate(
    baseline: AnalysisReport,
    candidate: AnalysisReport,
    floors: DeterminismFloors,
    *,
    metric: str,
    direction: IterateDirection,
    min_effect: Optional[float] = None,
    noise_threshold_multiplier: float = DEFAULT_NOISE_THRESHOLD_MULTIPLIER,
) -> IterateDelta:
    """Compute the R2-thresholded iterate delta report + verdict.

    ``baseline`` / ``candidate`` are F1 :class:`AnalysisReport`\\ s (render +
    analyze) differing by the tested parameter change; ``floors`` is the F2
    :class:`DeterminismFloors` measured for ``(binary_sha256, patch_class)``. The
    ``metric`` dotted path (e.g. ``deterministic.summary.spectral_centroid_hz``)
    selects the feature to compare and keys its per-feature floor.

    Computation (by design):

    - ``abs_delta = candidate_value - baseline_value`` (signed; the C1 reference
      stores ``-850.3`` for a decrease);
    - ``noise_threshold = measured_floor * noise_threshold_multiplier`` (default
      3x; both operands stored so significance is auditable);
    - ``significant`` iff the delta *magnitude* clears the effective threshold AND
      the metric's floor is reliable (present in every floor-measuring render);
    - ``matches_expectation`` per ``direction``;
    - verdict ``INCONCLUSIVE`` when not significant (``|abs_delta|`` at/below the
      threshold — a sub-floor change), else ``PASS`` / ``FAIL`` by direction.

    ``min_effect`` (optional) raises the effective significance threshold to
    ``max(noise_threshold, min_effect)`` — an additional absolute effect floor the
    caller requires — WITHOUT changing the stored ``noise_threshold`` field (which
    always records ``measured_floor * multiplier``). It never lets a sub-floor
    change be called significant.

    An unknown ``metric`` (absent from the report or the floors object) or a
    suppressed/non-numeric metric value is a hard :class:`InputError` (exit 2).
    """
    # Guard the noise-threshold multiplier (mirrors determinism.py's ``repeats < 2``
    # guard): a non-positive multiplier would collapse or invert the significance
    # threshold, silently turning sub-floor noise into a "significant" delta. This
    # is an INPUT-contract failure on a caller-supplied knob (exit 2). A negative
    # ``min_effect`` is harmless (it loses the ``max()`` to ``noise_threshold``), so
    # only the multiplier is guarded here.
    if noise_threshold_multiplier <= 0:
        raise InputError(
            _MULTIPLIER_INVALID_CODE,
            f"noise_threshold_multiplier must be > 0; got "
            f"{noise_threshold_multiplier}",
            detail={"noise_threshold_multiplier": noise_threshold_multiplier},
            component=_ITERATE_COMPONENT,
        )

    baseline_value = _extract_metric(baseline, metric)
    candidate_value = _extract_metric(candidate, metric)

    floor_entry = floors.floors.get(metric)
    if floor_entry is None:
        raise InputError(
            _FLOOR_MISSING_CODE,
            f"no measured floor for metric {metric!r} in the floors object "
            f"(available: {sorted(floors.floors)[:8]}...)",
            detail={"metric": metric},
            component=_ITERATE_COMPONENT,
        )

    measured_floor = floor_entry.floor
    noise_threshold = measured_floor * noise_threshold_multiplier

    signed_delta = candidate_value - baseline_value
    magnitude = abs(signed_delta)

    # Presence-instability guard (F2 handoff): a feature present in fewer than the
    # full N floor-measuring renders has an unreliable floor; do not over-trust its
    # small/zero value into a confident significance claim.
    floor_reliable = floor_entry.repeats >= floors.repeats

    # min_effect is an additional absolute effect floor; never below the
    # measured noise threshold.
    effective_threshold = noise_threshold
    if min_effect is not None:
        effective_threshold = max(noise_threshold, min_effect)

    significant = magnitude > effective_threshold and floor_reliable
    matches = _matches_expectation(direction, signed_delta, significant)
    verdict = _verdict(direction, significant, matches)

    return IterateDelta(
        baseline=baseline,
        candidate=candidate,
        expectation=Expectation(
            metric=metric, direction=direction, min_effect=min_effect
        ),
        delta=Delta(
            metric=metric,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            abs_delta=signed_delta,
            measured_floor=measured_floor,
            noise_threshold_multiplier=noise_threshold_multiplier,
            noise_threshold=noise_threshold,
            significant=significant,
            matches_expectation=matches,
        ),
        verdict=verdict,
    )


# --- descriptor term-diff (A5; iterate-descriptors) -------------------------
#
# The descriptor-term REGRESSION signal is a SET comparison over the non-estimated
# measured terms (estimated rows are excluded — their raw values, e.g. octave-
# mitigated tempo-audio, are expected to churn and are not ground truth), NOT a
# numeric-metric significance gate like ``run_iterate``. A term appearing or
# disappearing, or flipping firing direction, IS the regression. Raw value drift,
# by contrast, is EXPECTED across renders, so it is reported separately in a
# tolerance-banded advisory sub-list rather than as part of the regression set.
#
# Observe-vs-assert (deliberate): this diff applies ONLY the non-estimated filter.
# It intentionally does NOT reapply the descriptor gate's stricter operational
# eligibility (per-block-kind context-set membership + duplicate rejection). A diff
# OBSERVES which terms moved between two already-produced reports; the gate ASSERTS
# and REJECTS a single report against its context. Keeping the diff block-kind-
# agnostic (no descriptor_gate import) is the right call for an observation tool.
#
# These are plain frozen dataclasses, NOT Pydantic report-schema models: the diff
# is a CLI-side comparison artifact that never persists into the versioned report
# JSON (mirrors the descriptor gate's stderr-only verdict). It shares no symbols
# with ``IterateDelta`` / ``run_iterate``.


@dataclass(frozen=True)
class ValueDrift:
    """One both-present term whose measured value moved beyond ``value_tolerance``.

    Advisory only (NOT part of the regression set). ``baseline_value`` /
    ``candidate_value`` are the raw measured values; a non-finite value on either
    side is reported here verbatim and sanitized to JSON ``null`` at the CLI
    serialization boundary.
    """

    term: str
    baseline_value: float
    candidate_value: float


@dataclass(frozen=True)
class DescriptorTermDiff:
    """The descriptor-term diff between a baseline and a candidate report.

    ``added`` / ``removed`` / ``direction_changed`` are the REGRESSION set (sorted
    term lists); ``value_drift`` is the SEPARATE tolerance-banded advisory sub-list.
    """

    added: list[str]
    removed: list[str]
    direction_changed: list[str]
    value_drift: list[ValueDrift]


def _eligible_measured(block: DescriptorsBlock) -> dict[str, MeasuredDescriptor]:
    """The non-estimated measured rows of a block keyed by term (estimated excluded).

    Only the ``estimated`` filter is applied — the diff does NOT reapply the
    descriptor gate's context-set/duplicate constraints (observe-vs-assert, see the
    section comment above). Keying by ``term`` means a corrupt block carrying
    duplicate non-estimated rows for the same term would last-write-win here.
    Produced reports never emit duplicate terms, so this is an accepted
    corrupt-input limitation, taken deliberately to keep the diff pure and simple.
    """
    return {d.term: d for d in block.measured if not d.estimated}


def _value_drifted(
    baseline_value: float, candidate_value: float, tolerance: float
) -> bool:
    """Whether a both-present term's value moved enough to report as drift.

    A non-finite value on either side is ALWAYS reported (the magnitude comparison
    is undefined, and a NaN/Inf descriptor value is itself notable); otherwise the
    absolute delta must strictly exceed ``tolerance``.
    """
    if not (math.isfinite(baseline_value) and math.isfinite(candidate_value)):
        return True
    return abs(candidate_value - baseline_value) > tolerance


def diff_descriptor_terms(
    baseline: DescriptorsBlock,
    candidate: DescriptorsBlock,
    *,
    value_tolerance: float = 0.0,
) -> DescriptorTermDiff:
    """Compute the descriptor-term regression + value-drift diff (A5).

    Both sides reduce to their non-estimated measured terms (``estimated`` rows
    excluded; the gate's context-set/duplicate constraints are intentionally NOT
    reapplied — observe-vs-assert). ``added`` / ``removed`` are the sorted
    set-differences over the term
    keys; ``direction_changed`` is the sorted set of both-present terms whose
    ``MeasuredDescriptor.direction`` differs. ``value_drift`` is the sorted list of
    both-present terms whose measured ``value`` moved by more than ``value_tolerance``
    (a non-finite value on either side is always reported). The function is pure and
    never raises on non-finite descriptor *values*.

    ``value_tolerance`` itself must be a finite, non-negative number: a negative
    tolerance makes ``abs(delta) > tolerance`` always true (false-positive drift) and
    a non-finite tolerance makes it always false (zero drift). A bad ``value_tolerance``
    raises ``ValueError`` (defense-in-depth for direct callers; the CLI guards its own
    input before reaching this).
    """
    if not math.isfinite(value_tolerance) or value_tolerance < 0:
        raise ValueError(
            "value_tolerance must be a finite, non-negative number; "
            f"got {value_tolerance}"
        )

    base = _eligible_measured(baseline)
    cand = _eligible_measured(candidate)
    base_terms = set(base)
    cand_terms = set(cand)

    added = sorted(cand_terms - base_terms)
    removed = sorted(base_terms - cand_terms)
    both = sorted(base_terms & cand_terms)

    direction_changed = [t for t in both if base[t].direction != cand[t].direction]

    value_drift = [
        ValueDrift(
            term=t,
            baseline_value=base[t].value,
            candidate_value=cand[t].value,
        )
        for t in both
        if _value_drifted(base[t].value, cand[t].value, value_tolerance)
    ]

    return DescriptorTermDiff(
        added=added,
        removed=removed,
        direction_changed=direction_changed,
        value_drift=value_drift,
    )
