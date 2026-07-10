"""Advisory producer tests (Task T5 / design §8).

Exact-equality only (F7): the pure ``map_freeform_to_advisory`` unit is bit-identical
for a fixed description, so every assertion builds the exact expected value and asserts
``==``. ``produce_advisory`` is the never-fatal boundary: its four paths (not-ok
perception, ``total == 0``, normal, unexpected error) each return the exact 4-tuple the
orchestrator hook consumes.

The vocabulary drift guard (``test_vocab_exact_membership``) hard-fails on any change to
the frozen 67-term ``BOUNDED_ADVISORY_VOCAB``. ``test_qwen_to_map_integration`` is the
real Qwen-freeform->map deliverable, ``@pytest.mark.integration`` (deselected in core CI,
measured-stability not bit-identity).
"""

from __future__ import annotations

import pytest

from sonoscope.descriptors import advisory as advisory_mod
from sonoscope.descriptors.advisory import (
    ADVISORY_DEGRADED_CODE,
    map_freeform_to_advisory,
    produce_advisory,
)
from sonoscope.descriptors.vocab import (
    ADVISORY_BASE_CONFIDENCE,
    BOUNDED_ADVISORY_VOCAB,
    CURATED_SYNONYM_MAP,
)
from sonoscope.schema.models import AdvisoryDescriptor, AdapterInfo, PerceptionBlock


# --- exact frozen vocabulary membership (drift guard, design §8 / F9) --------

_EXPECTED_VOCAB: frozenset[str] = frozenset(
    {
        # Category F — affect / mood (13)
        "tense",
        "calm",
        "dark",
        "uplifting",
        "melancholy",
        "aggressive",
        "playful",
        "ominous",
        "serene",
        "anxious",
        "triumphant",
        "nostalgic",
        "euphoric",
        # Category G — evocative / aesthetic (15)
        "ethereal",
        "cosmic",
        "dreamy",
        "lush",
        "glassy",
        "retro/vintage",
        "futuristic",
        "campy",
        "cinematic",
        "psychedelic",
        "gritty",
        "warm",
        "organic/synthetic",
        "alien",
        "epic",
        # Category H — genre / style (36)
        "ambient",
        "drone",
        "techno",
        "house",
        "trance",
        "minimal",
        "dnb",
        "jungle",
        "breakbeat",
        "garage",
        "hip-hop",
        "trap",
        "lo-fi",
        "dubstep",
        "rock",
        "indie",
        "punk",
        "post-rock",
        "metal",
        "djent",
        "doom",
        "black",
        "jazz",
        "fusion",
        "swing",
        "bebop",
        "classical",
        "orchestral",
        "film-score",
        "folk",
        "country",
        "pop",
        "synth-pop",
        "noise",
        "glitch",
        "idm",
        # advisory-lean hybrids (3)
        "hypnotic",
        "spacious",
        "wall-of-sound",
    }
)


def _ok_perception(description: str) -> PerceptionBlock:
    """A status=='ok' perception block carrying a canned freeform description.

    status=='ok' requires the four adapter-output fields (models.py _OK_REQUIRED);
    they are stubbed so the block validates and the producer reads .description.
    """
    return PerceptionBlock(
        status="ok",
        grounding="advisory-freetext",
        adapter=AdapterInfo(
            id="fake",
            model="fake-model",
            quant="none",
            runtime="none",
            model_sha256="a" * 64,
        ),
        description=description,
        grounding_map={},
        disclaimer="advisory-only; not ground truth",
    )


def test_vocab_exact_membership() -> None:
    """BOUNDED_ADVISORY_VOCAB is exactly the frozen 67 terms — drift hard-fails."""
    assert set(BOUNDED_ADVISORY_VOCAB) == set(_EXPECTED_VOCAB)
    assert len(BOUNDED_ADVISORY_VOCAB) == 67


def test_map_bit_identical() -> None:
    """map_freeform_to_advisory is pure: identical string -> identical result."""
    desc = "a spacey hypnotic brooding warbly pad"
    first = map_freeform_to_advisory(desc)
    second = map_freeform_to_advisory(desc)
    assert first == second


def test_advisory_base_confidence_pinned() -> None:
    """Literal anchor for the advisory base confidence (green-mirage guard, C1.6).

    Every other confidence assertion in this suite builds its expected record with
    ``confidence=ADVISORY_BASE_CONFIDENCE`` — importing the same production constant
    under test — so a wrong base value would ship undetected. This test pins the
    constant to the concrete literal 0.6, and ``test_map_coverage_worked_example``
    pins an *emitted* descriptor's confidence to the same literal.
    """
    assert ADVISORY_BASE_CONFIDENCE == 0.6


def test_map_coverage_worked_example() -> None:
    """Design §8 worked example: candidates {spacey, hypnotic, brooding, warbly}
    -> total==4, matched==2; advisory is exactly [cosmic, hypnotic] in mapping
    (first-appearance) order."""
    desc = "a spacey hypnotic brooding warbly texture"
    advisory, matched, total = map_freeform_to_advisory(desc)

    assert total == 4
    assert matched == 2
    assert advisory == [
        AdvisoryDescriptor(
            term="cosmic", source="lalm-mapped", confidence=ADVISORY_BASE_CONFIDENCE
        ),
        AdvisoryDescriptor(
            term="hypnotic", source="lalm-mapped", confidence=ADVISORY_BASE_CONFIDENCE
        ),
    ]
    # Literal confidence pin (green-mirage guard, C1.6): assert against the concrete
    # literal 0.6, NOT the imported constant, so a wrong base value fails here.
    assert advisory[0].confidence == 0.6
    assert advisory[1].confidence == 0.6


def test_map_all_values_in_vocab() -> None:
    """Every canonical value of the curated map is a bounded-vocab member."""
    assert set(CURATED_SYNONYM_MAP.values()) - set(BOUNDED_ADVISORY_VOCAB) == set()


def test_produce_advisory_not_ok_perception() -> None:
    """A non-ok perception yields the empty 4-tuple (no coverage to report)."""
    perception = PerceptionBlock(status="disabled", grounding="none")
    assert produce_advisory(perception) == ([], None, None, None)


def test_produce_advisory_total_zero() -> None:
    """ok perception whose description maps nothing -> coverage None, advisory [],
    dropped None (F3: no signal => no coverage AND no drop count)."""
    perception = _ok_perception("the quick brown fox jumped")
    advisory, coverage, dropped, err = produce_advisory(perception)

    assert advisory == []
    assert coverage is None
    assert dropped is None
    assert err is None


def test_produce_advisory_normal_coverage() -> None:
    """A mapping-bearing description plumbs exact coverage + dropped."""
    perception = _ok_perception("a spacey hypnotic brooding texture")
    advisory, coverage, dropped, err = produce_advisory(perception)

    assert advisory == [
        AdvisoryDescriptor(
            term="cosmic", source="lalm-mapped", confidence=ADVISORY_BASE_CONFIDENCE
        ),
        AdvisoryDescriptor(
            term="hypnotic", source="lalm-mapped", confidence=ADVISORY_BASE_CONFIDENCE
        ),
    ]
    # candidates {spacey, hypnotic, brooding} -> total 3, matched 2.
    assert coverage == 2 / 3
    assert dropped == 1
    assert err is None


def test_produce_advisory_never_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected mapping error degrades to measured-only + a warning ErrorItem;
    NO exception propagates through the boundary."""

    def _boom(description: str):  # noqa: ANN202 - test double
        raise RuntimeError("mapping exploded")

    monkeypatch.setattr(advisory_mod, "map_freeform_to_advisory", _boom)

    perception = _ok_perception("a spacey hypnotic pad")
    advisory, coverage, dropped, err = produce_advisory(perception)

    assert advisory == []
    assert coverage is None
    assert dropped is None
    assert err is not None
    assert err.code == ADVISORY_DEGRADED_CODE
    assert err.code == "ADVISORY_DEGRADED"
    assert err.severity == "warning"
    assert err.component == "analyze"


def test_cosmic_positive_golden() -> None:
    """A known-cosmic canned description maps to exactly a cosmic advisory record."""
    advisory, matched, total = map_freeform_to_advisory(
        "an outer space pad drifting through the galactic void"
    )
    assert advisory == [
        AdvisoryDescriptor(
            term="cosmic", source="lalm-mapped", confidence=ADVISORY_BASE_CONFIDENCE
        )
    ]
    assert matched == 1


@pytest.mark.integration
def test_qwen_to_map_integration(qwen_model: None) -> None:
    """Full Qwen freeform -> curated map path (F13 — REAL deliverable).

    requires the local Qwen2-Audio stack; exercises the non-deterministic
    freeform->map path — measured-stability check, not bit-identity.

    Deselected in core CI (`-m "not integration"`). Asserts the mapped term is
    PRESENT (measured-stability), never bit-identity of the Qwen description.
    """
    import os
    from pathlib import Path

    from sonoscope.perception.qwen_local import QwenLocalAdapter

    clip_env = os.environ.get("SONOSCOPE_COSMIC_CLIP")
    if clip_env is None or not Path(clip_env).is_file():
        pytest.skip(
            "cosmic advisory clip absent; set $SONOSCOPE_COSMIC_CLIP to a known-"
            "cosmic 48 kHz wav (integration artifact absent, never a silent pass)"
        )

    adapter = QwenLocalAdapter()
    perception = adapter.describe(Path(clip_env))
    advisory, _matched, _total = map_freeform_to_advisory(perception.description or "")

    assert "cosmic" in {d.term for d in advisory}
