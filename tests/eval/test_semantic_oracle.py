"""LLM semantic-eval TEST oracle (Task T6, design §9.2).

This is a SEMANTIC ORACLE, not a deterministic gate: it asks a constrained LLM
whether each produced advisory/hybrid term falls in the right conceptual category
over the frozen vocabulary's category space, and asserts >=95% agreement across a
fixture set (measured-stability, NOT bit-identity). It is ``@pytest.mark.integration``
and DESELECTED in core CI (``uv run pytest -m "not integration"``); it NEVER gates
production and never makes any descriptor fatal.

The ``allowed_labels`` are sourced from ``BOUNDED_ADVISORY_VOCAB`` (the single source
of truth shared with the advisory producer, design §8/§9) so the label space cannot
diverge from the vocabulary the descriptors are drawn from.

The external LLM eval endpoint and its pinned fixture corpus are a later-cycle
deliverable (the production constrained-classify gate is deferred to C7, design §9);
until they land, this oracle skips with the explicit design-mandated reason. The
skip keeps core-CI collection green (no heavy import at module import time) while
preserving the oracle's contract and single-source-of-truth wiring.
"""

from __future__ import annotations

import pytest

from sonoscope.descriptors.vocab import BOUNDED_ADVISORY_VOCAB

#: Exact skip reason mandated by design §9.2 — do NOT reword.
_SKIP_REASON = (
    "requires an LLM eval endpoint; semantic oracle, not a deterministic gate "
    "— advisory quality check only, never a production gate."
)


@pytest.mark.integration
def test_produced_descriptors_fall_in_right_category() -> None:
    """Oracle: do the produced descriptors fall in the right conceptual category?

    Over the frozen category space (``allowed_labels`` sourced from
    ``BOUNDED_ADVISORY_VOCAB``), a constrained LLM is asked whether each produced
    advisory/hybrid term conceptually applies to its clip; the oracle asserts
    >=95% agreement across the fixture set (measured-stability bar, NOT
    bit-identity). It NEVER gates production and never makes a descriptor fatal.

    SKIP REASON (explicit, design §9.2): 'requires an LLM eval endpoint; semantic
    oracle, not a deterministic gate — advisory quality check only, never a
    production gate.'
    """
    # allowed_labels is sourced from the frozen vocabulary (single source of
    # truth) so this oracle's label space cannot drift from the advisory
    # producer's — exercised here even though the LLM agreement measurement is
    # deferred to the integration endpoint + fixture corpus (C7).
    allowed_labels = tuple(sorted(BOUNDED_ADVISORY_VOCAB))
    # Literal anchor (NOT a tautology against the same constant): the frozen vocab
    # size is 67. A drift in count trips here; membership drift is guarded by
    # tests/descriptors/test_advisory.py::test_vocab_exact_membership.
    assert len(allowed_labels) == 67

    pytest.skip(_SKIP_REASON)
