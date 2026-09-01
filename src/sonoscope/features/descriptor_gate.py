"""Pure descriptor assertion/gate comparator.

The FIRST reader of ``report.descriptors``: ``evaluate_descriptors`` compares an
operator-authored expectation spec against a :class:`DescriptorsBlock` and yields
a byte-stable ``PASS``/``RED`` verdict plus a deterministic ``reasons[]``. Pure
(no I/O, clock, RNG): identical ``(block, spec, block_kind)`` → identical output.

Mirrors ``features/midi_tripwires.py`` (frozen-dataclass result, module-level
reason-id constants, ``verdict = RED if reasons else PASS``). Because every
descriptor reason id is RED-producing (no informational entries), the simple
``verdict = "RED" if reasons else "PASS"`` rule is correct here — a separate
``_has_red`` predicate (needed by ``evaluate_midi`` only because IT carries a
non-RED note) is deliberately NOT copied.

Zero new runtime dependencies: stdlib ``math`` + existing ``pydantic`` + the
frozen report models. The spec-input model (``ExpectedDescriptors``) is defined
here as field-only; its structural/eligibility validators land in the loader
(Task 2). Nothing in this module is added to the frozen ``schema/models.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from sonoscope.descriptors.vocab import BOUNDED_ADVISORY_VOCAB
from sonoscope.errors import InputError
from sonoscope.schema.models import DescriptorsBlock, MeasuredDescriptor

# --- Reason-id constants (SSOT for the builder and the tests) -----------------
DESC_MISSING = "DESC_MISSING"
DESC_UNEXPECTED = "DESC_UNEXPECTED"
DESC_VALUE_ABSENT = "DESC_VALUE_ABSENT"
DESC_VALUE_NONFINITE = "DESC_VALUE_NONFINITE"
DESC_VALUE_OUT_OF_RANGE = "DESC_VALUE_OUT_OF_RANGE"

DESCRIPTORS_BLOCK_MALFORMED = "DESCRIPTORS_BLOCK_MALFORMED"

# Error code for a malformed operator-authored expectation spec (loader stage).
DESCRIPTORS_EXPECTED_SPEC_INVALID = "DESCRIPTORS_EXPECTED_SPEC_INVALID"

# --- Context-scoped eligibility sets ------------------------------------------
# Gate-eligible measured terms per block kind. Estimated measured terms are NOT
# gate-eligible even though they live in ``measured`` (by design,
# grounding taxonomy); the audio set below excludes ``tempo-audio`` for that
# reason.
AUDIO_GATE_ELIGIBLE_TERMS: frozenset[str] = frozenset(
    {
        "bright",
        "dark",
        "loud",
        "quiet",
        "compressed",
        "dynamic",
        "busy",
        "spare",
        "dense",
        "rhythmic-density",
        # Emitted alone when the integrity layer reports the whole file silent.
        # Gate-eligible so an operator can assert either polarity ("this render
        # must be silent", "this render must not be silent").
        "silent",
    }
)  # tempo-audio EXCLUDED (estimated=True)

MIDI_GATE_ELIGIBLE_TERMS: frozenset[str] = frozenset(
    {
        "note-density",
        "register",
        "pitch-range",
        "polyphony",
        "velocity-dynamics",
        "ioi",
    }
)  # 6; frozen ap12 interface, dormant this cycle (no MIDI producer until C2)

# MIDI defensive import — keep the path dormant, never raise at import time.
# ap12 has not landed ``MIDI_MEASURED_TERMS`` yet; the local frozenset above is
# the design-authoritative set this cycle. ``_VOCAB_MIDI`` is future-proofing and
# is not read by any wired path now.
try:  # pragma: no cover - dormant until ap12 lands MIDI_MEASURED_TERMS
    from sonoscope.descriptors.vocab import MIDI_MEASURED_TERMS as _VOCAB_MIDI  # noqa: F401
except ImportError:  # ap12 has not landed MIDI_MEASURED_TERMS yet
    _VOCAB_MIDI = None

# Known descriptor terms that are NEVER gate-eligible under ANY block kind: the
# three composite hybrid terms (``deriver.py``'s ``_hybrid`` emits ``driving``/
# ``punchy``/``warm`` into ``block.hybrid``; ``warm`` is ALSO a bounded advisory
# evocative term, so it lives in BOUNDED_ADVISORY_VOCAB too), the estimated
# measured audio term (``tempo-audio``), and the entire frozen bounded advisory
# vocabulary (``block.advisory`` labels). Used ONLY by the loader to DISAMBIGUATE
# an ineligible spec term: a member here is a real-but-non-gateable descriptor
# (``term_not_gate_eligible``, L6) rather than a typo (``unknown_term``, L4) or an
# other-context measured term (``cross_context_term``, L5). Kept in lock-step with
# ``_hybrid`` by the ``test_hybrid_terms_match_deriver_ast`` drift-guard so
# ``warm``'s L6 classification never silently depends on the advisory overlap.
_HYBRID_TERMS: frozenset[str] = frozenset({"driving", "punchy", "warm"})
_ESTIMATED_MEASURED_TERMS: frozenset[str] = frozenset({"tempo-audio"})
_KNOWN_NON_GATE_ELIGIBLE_TERMS: frozenset[str] = (
    _HYBRID_TERMS | _ESTIMATED_MEASURED_TERMS | BOUNDED_ADVISORY_VOCAB
)


# --- Spec-input model (field-only in Task 1; loader validators in Task 2) -----


def _spec_invalid(reason: str, message: str, **extra: Any) -> InputError:
    """Build the one fail-loud error for every spec-load failure.

    Raised DIRECTLY (not as ``ValueError``) from the ``_ExpectValue`` after-validator
    and from the loader's eligibility pass. Because ``InputError`` is a
    :class:`SonoscopeError`, NOT a ``ValueError``/``AssertionError``, pydantic v2
    does not wrap it into a ``ValidationError``; the exact ``detail`` propagates
    unchanged straight out of ``model_validate``.
    """
    return InputError(
        DESCRIPTORS_EXPECTED_SPEC_INVALID,
        message,
        detail={"reason": reason, **extra},
        component="analyze",
    )


class _ExpectValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str
    min: Optional[float] = None
    max: Optional[float] = None
    equals: Optional[float] = None
    tolerance: Optional[float] = None

    @model_validator(mode="after")
    def _validate_structure(self) -> "_ExpectValue":
        """Fail loud on a structurally-malformed expect_value (L7–L12).

        Exactly one bound form is legal: an ``equals`` point (optionally with a
        finite, non-negative ``tolerance``, default ``0.0``) OR a band
        (``min``/``max``, at least one side, ``min <= max``, all finite). Any
        other shape raises ``DESCRIPTORS_EXPECTED_SPEC_INVALID`` directly.
        """
        has_equals = self.equals is not None
        has_min = self.min is not None
        has_max = self.max is not None

        # L7: no bound at all — an entry that asserts nothing.
        if not (has_equals or has_min or has_max):
            raise _spec_invalid(
                "expect_value_no_bound",
                "expect_value entry has no bound (need equals or min/max)",
            )

        if has_equals:
            # L8: equals is a point form; combining it with a band is ambiguous.
            if has_min or has_max:
                raise _spec_invalid(
                    "expect_value_mixed_bound",
                    "expect_value entry mixes equals with min/max",
                )
            # L12: a non-finite equals target is never a valid point.
            if not math.isfinite(self.equals):
                raise _spec_invalid(
                    "non_finite_bound", "expect_value equals is not finite"
                )
            # L10: tolerance (when supplied) must be finite and non-negative.
            if self.tolerance is not None and (
                not math.isfinite(self.tolerance) or self.tolerance < 0.0
            ):
                raise _spec_invalid(
                    "bad_tolerance",
                    "expect_value tolerance must be finite and non-negative",
                )
        else:
            # Band form. L9: tolerance is only meaningful alongside equals.
            if self.tolerance is not None:
                raise _spec_invalid(
                    "tolerance_without_equals",
                    "expect_value tolerance requires equals",
                )
            # L12: band edges must be finite (unbounded sides omit the key).
            if has_min and not math.isfinite(self.min):
                raise _spec_invalid("non_finite_bound", "expect_value min is not finite")
            if has_max and not math.isfinite(self.max):
                raise _spec_invalid("non_finite_bound", "expect_value max is not finite")
            # L11: an inverted band matches nothing; that is operator error.
            if has_min and has_max and self.min > self.max:
                raise _spec_invalid(
                    "min_gt_max", "expect_value min is greater than max"
                )
        return self


class ExpectedDescriptors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expect_present: list[str] = []
    expect_absent: list[str] = []
    expect_value: list[_ExpectValue] = []


# --- Result type --------------------------------------------------------------


@dataclass(frozen=True)
class DescriptorEvaluation:
    verdict: Literal["PASS", "RED"]
    reasons: list[str]


# --- Eligibility + extraction -------------------------------------------------


def _eligible_set(block_kind: Literal["audio", "midi"]) -> frozenset[str]:
    return AUDIO_GATE_ELIGIBLE_TERMS if block_kind == "audio" else MIDI_GATE_ELIGIBLE_TERMS


def _extract_eligible(
    block: DescriptorsBlock, block_kind: Literal["audio", "midi"]
) -> dict[str, MeasuredDescriptor]:
    """Gate-eligible measured map: ``{term: row}`` for non-estimated rows.

    Estimated measured terms are not gate-eligible (by design,
    grounding taxonomy) and are skipped here.

    ``hybrid``, ``advisory``, and ``estimated is True`` measured rows are invisible.
    A duplicate gate-eligible term or a term outside the context set is a malformed
    block → fail loud (never silent last-wins / silent drop).
    """
    eligible: dict[str, MeasuredDescriptor] = {}
    valid = _eligible_set(block_kind)
    for row in block.measured:
        if row.estimated:
            continue
        if row.term in eligible:
            raise InputError(
                DESCRIPTORS_BLOCK_MALFORMED,
                "descriptors block has a duplicate gate-eligible term",
                detail={"reason": "duplicate_term", "term": row.term},
                component="analyze",
            )
        if row.term not in valid:
            raise InputError(
                DESCRIPTORS_BLOCK_MALFORMED,
                "descriptors block emits a term outside the context eligibility set",
                detail={"reason": "unknown_emitted_term", "term": row.term},
                component="analyze",
            )
        eligible[row.term] = row
    return eligible


# --- Reason builder (fixed section order + per-entry precedence) --------------


def _value_reason(entry: _ExpectValue, eligible: dict[str, MeasuredDescriptor]) -> Optional[str]:
    """At most one reason per ``expect_value`` entry (ABSENT > NONFINITE > RANGE)."""
    if entry.term not in eligible:
        return f"{DESC_VALUE_ABSENT}: {entry.term}"
    value = eligible[entry.term].value
    if math.isnan(value) or math.isinf(value):
        return f"{DESC_VALUE_NONFINITE}: {entry.term} value={value!r}"
    if entry.equals is not None:
        tolerance = entry.tolerance if entry.tolerance is not None else 0.0
        if abs(value - entry.equals) <= tolerance:
            return None
        return (
            f"{DESC_VALUE_OUT_OF_RANGE}: {entry.term} value={value!r} "
            f"not within {tolerance!r} of {entry.equals!r}"
        )
    lo_val = entry.min if entry.min is not None else -math.inf
    hi_val = entry.max if entry.max is not None else math.inf
    if lo_val <= value <= hi_val:
        return None
    lo = repr(entry.min) if entry.min is not None else "-inf"
    hi = repr(entry.max) if entry.max is not None else "inf"
    return f"{DESC_VALUE_OUT_OF_RANGE}: {entry.term} value={value!r} not in [{lo}, {hi}]"


def _build_reasons(
    eligible: dict[str, MeasuredDescriptor], spec: ExpectedDescriptors
) -> list[str]:
    reasons: list[str] = []
    # 1. expect_present misses, in spec-declared order.
    for term in spec.expect_present:
        if term not in eligible:
            reasons.append(f"{DESC_MISSING}: {term}")
    # 2. expect_absent violations, in spec-declared order.
    for term in spec.expect_absent:
        if term in eligible:
            reasons.append(f"{DESC_UNEXPECTED}: {term}")
    # 3. expect_value failures, in spec-declared order (one reason per entry).
    for entry in spec.expect_value:
        reason = _value_reason(entry, eligible)
        if reason is not None:
            reasons.append(reason)
    return reasons


# --- Public comparator --------------------------------------------------------


def evaluate_descriptors(
    block: DescriptorsBlock,
    spec: ExpectedDescriptors,
    *,
    block_kind: Literal["audio", "midi"],
) -> DescriptorEvaluation:
    """Compare a loader-validated ``spec`` against ``block`` → PASS/RED verdict.

    Presumes a loader-validated ``ExpectedDescriptors``: structural validation
    (bounds present, types correct) is the loader's job (Task 2); this
    comparator's contract is defined only over loader-validated specs.
    """
    eligible = _extract_eligible(block, block_kind)
    reasons = _build_reasons(eligible, spec)
    return DescriptorEvaluation(verdict="RED" if reasons else "PASS", reasons=reasons)


# --- Spec loader (fail-loud; path OR in-memory dict) --------------------------


def _load_spec_source(source: Union[str, Path, dict[str, Any]]) -> Any:
    """Return the deserialized spec: read+parse a path, else pass through.

    Models ``midi_input._load_expected_source``: a path is read as UTF-8 then
    ``json.loads``-ed, mapping ``OSError``/``UnicodeDecodeError`` (both →
    ``unreadable``) and ``json.JSONDecodeError`` (→ ``unparseable``) to a typed
    ``InputError`` (never a silent skip); a non-path source passes through for
    the caller's type/shape checks.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _spec_invalid(
                "unreadable", f"descriptor expectation spec not readable at {path}"
            ) from exc
        except UnicodeDecodeError as exc:
            # ``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``, so it
            # is NOT caught above; without this branch a non-UTF-8 spec file would
            # escape as a generic INTERNAL_ERROR instead of the typed input error.
            raise _spec_invalid(
                "unreadable",
                f"descriptor expectation spec at {path} is not valid UTF-8: {exc}",
            ) from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise _spec_invalid(
                "unparseable",
                f"descriptor expectation spec at {path} is not valid JSON: {exc}",
            ) from exc
    return source


def _classify_ineligible_term(
    term: str, block_kind: Literal["audio", "midi"]
) -> str:
    """Disambiguation reason for a spec term outside the block_kind's set.

    ``cross_context_term`` (L5) if it is the OTHER context's measured term;
    ``term_not_gate_eligible`` (L6) if it is a known-but-non-gateable descriptor
    (hybrid / estimated-measured / advisory); ``unknown_term`` (L4) otherwise.
    """
    other = MIDI_GATE_ELIGIBLE_TERMS if block_kind == "audio" else AUDIO_GATE_ELIGIBLE_TERMS
    if term in other:
        return "cross_context_term"
    if term in _KNOWN_NON_GATE_ELIGIBLE_TERMS:
        return "term_not_gate_eligible"
    return "unknown_term"


def _enforce_term_eligibility(
    spec: ExpectedDescriptors, block_kind: Literal["audio", "midi"]
) -> None:
    """A1 load-time gate: every referenced term MUST be gate-eligible.

    Rejecting cross-context / non-gate-eligible / unknown terms at LOAD (not
    silently at compare time) closes the vacuous-``expect_absent`` green mirage:
    an ``expect_absent`` on a term that could never be emitted would otherwise
    always PASS.
    """
    eligible = _eligible_set(block_kind)
    referenced = [
        *spec.expect_present,
        *spec.expect_absent,
        *(entry.term for entry in spec.expect_value),
    ]
    for term in referenced:
        if term not in eligible:
            raise _spec_invalid(
                _classify_ineligible_term(term, block_kind),
                f"expectation spec references ineligible term {term!r} "
                f"for block_kind {block_kind!r}",
                term=term,
                block_kind=block_kind,
            )


def load_expected_descriptors(
    source: Union[str, Path, dict[str, Any]],
    *,
    block_kind: Literal["audio", "midi"],
) -> ExpectedDescriptors:
    """Load + fail-loud-validate an operator expectation spec.

    ``source`` is EITHER a path to a JSON file OR an already-deserialized dict.
    Returns a fully validated :class:`ExpectedDescriptors`; NEVER silent-skips a
    malformed spec. Failures raise ``InputError(DESCRIPTORS_EXPECTED_SPEC_INVALID,
    ..., component="analyze")`` (exit 2) with an exact ``detail["reason"]``:

    - source stage: ``unreadable`` / ``unparseable`` (path) or ``not_an_object``
      (top level is not a JSON object);
    - structural (``_ExpectValue`` validator, raised DIRECTLY): ``expect_value_no_bound``,
      ``expect_value_mixed_bound``, ``tolerance_without_equals``, ``bad_tolerance``,
      ``min_gt_max``, ``non_finite_bound``;
    - pydantic-native (coercion, this function's ``except`` branch):
      ``unknown_top_level_key`` / ``unknown_nested_key`` (``extra="forbid"``, split
      by error loc depth) or the ``invalid_field_type`` fallback (any other
      coercion failure);
    - eligibility (A1, post-construct): ``cross_context_term`` / ``term_not_gate_eligible``
      / ``unknown_term``.
    """
    obj = _load_spec_source(source)
    if not isinstance(obj, dict):
        raise _spec_invalid(
            "not_an_object",
            "descriptor expectation spec must be a JSON object, "
            f"got {type(obj).__name__}",
        )
    try:
        spec = ExpectedDescriptors.model_validate(obj)
    except ValidationError as exc:
        # The structural-rule InputErrors are NOT ValidationErrors, so they never
        # reach here; this branch handles ONLY pydantic-native coercion failures.
        first_error = exc.errors()[0]
        if first_error["type"] == "extra_forbidden":
            # An unknown key. A top-level extra key has loc length 1
            # (e.g. ('bogus',)); one nested in an expect_value entry has a deeper
            # loc (e.g. ('expect_value', 0, 'bogus')). Report which one precisely.
            reason = (
                "unknown_top_level_key"
                if len(first_error["loc"]) == 1
                else "unknown_nested_key"
            )
        else:
            # INTENTIONAL catch-all: any pydantic-native coercion error that is NOT
            # extra_forbidden (wrong field type, bad numeric parse, etc.). The
            # L1-L13 reason list is NOT an exhaustive per-field-type enumeration —
            # every remaining pydantic ValidationError folds into this one reason.
            reason = "invalid_field_type"
        raise _spec_invalid(
            reason, f"descriptor expectation spec is invalid: {exc.error_count()} error(s)"
        ) from exc
    _enforce_term_eligibility(spec, block_kind)
    return spec
