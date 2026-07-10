"""Shared constrained-classifier primitive (by design, Task T6).

``constrained_classify`` calls an injectable LLM backend constrained to a fixed
label space, validates the response is EXACTLY one of ``allowed_labels``, and on
violation re-prompts (including the violation) up to ``max_retries`` — retry to
convergence. On non-convergence it raises :class:`ConstrainedClassifyError`.

This is the reusable primitive only. The PRODUCTION gate that reuses it (freeform
description -> constrained classification, replacing cycle-1's deterministic
curated advisory map) is DEFERRED to C7 (by design). In cycle 1 the sole
caller is the ``@pytest.mark.integration`` semantic-eval oracle
(``tests/eval/test_semantic_oracle.py``), so the default backend is exercised only
under integration; deterministic unit tests inject a fake via ``model=`` and no
real model loads in core CI.

The backend is an injectable parameter (by design, F1): ``ClassifierBackend`` is
``Callable[[str, float], str]`` — ``(prompt, temperature) -> raw label text``. The
param name ``model`` and the ``ClassifierBackend`` type are part of the FROZEN
signature so unit tests and any C7 reuse cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

#: Injectable classifier backend: ``(prompt, temperature) -> raw label text``.
#: Frozen part of the ``constrained_classify`` signature (by design, F1).
ClassifierBackend = Callable[[str, float], str]


class ConstrainedClassifyError(Exception):
    """Raised when the backend fails to converge on an allowed label.

    Signals that after ``max_retries`` attempts no response was EXACTLY one of
    ``allowed_labels``. In cycle 1 this is not wired into any production path;
    the integration oracle treats it as a test failure/skip signal (by design).
    """


def _default_qwen_backend(prompt: str, temperature: float) -> str:
    """Default backend: the local Qwen2-Audio stack (by design).

    The cycle-1 intent is to reuse the local Qwen2-Audio stack already in the
    perception path (``perception/qwen_local.py``) for text-constrained
    classification, with NO new model dependency. That stack currently exposes
    only an audio ``describe`` entry point; a text-only constrained-generation
    entry point is part of the C7 production gate and is not built in cycle 1
    (by design). Because the only cycle-1 caller is the integration-marked
    semantic-eval oracle — which requires an external LLM eval endpoint and is
    skipped in core CI — this default is never invoked in core CI. Callers that
    need a working backend today MUST inject one via ``model=`` (as the unit
    tests do). Invoking the default before the C7 text endpoint exists raises a
    clear, actionable error rather than silently degrading.
    """
    raise NotImplementedError(
        "the default Qwen text-classification backend is deferred to C7; "
        "inject a backend via constrained_classify(..., model=<ClassifierBackend>) "
        "or run under an integration LLM eval endpoint"
    )


def constrained_classify(
    prompt: str,
    allowed_labels: Sequence[str],
    *,
    model: ClassifierBackend = _default_qwen_backend,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Classify ``prompt`` into EXACTLY one of ``allowed_labels`` (by design).

    Calls ``model(prompt, temperature)`` with ``temperature`` defaulting to 0.0.
    If the response is exactly a member of ``allowed_labels`` it is returned. On
    a violation the prompt is re-issued with the violation appended (re-prompting
    with the offending label) and the backend is queried again, up to
    ``max_retries`` total attempts (retry-to-convergence). If no attempt yields an
    allowed label, :class:`ConstrainedClassifyError` is raised.

    ``model`` is injectable so the deterministic label-validation/retry logic is
    unit-tested with a fake backend (``model=fake``); the real Qwen backend is
    exercised only under integration.
    """
    allowed = tuple(allowed_labels)
    if not allowed:
        raise ValueError("allowed_labels must be non-empty")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    allowed_set = set(allowed)
    current_prompt = prompt
    last_response: str | None = None

    for _ in range(max_retries):
        response = model(current_prompt, temperature).strip()
        if response in allowed_set:
            return response
        last_response = response
        # Re-prompt with the violation so the backend can converge (by design).
        # ACCUMULATE the rejection history: append the newest invalid response to
        # the growing prompt (rather than rebuilding from the original `prompt` +
        # only the latest violation) so the backend sees EVERY previously rejected
        # label and cannot re-propose one it already tried.
        current_prompt = (
            f"{current_prompt}\n\n"
            f"Your previous answer {response!r} was not allowed. "
            f"Respond with EXACTLY one of: {', '.join(allowed)}."
        )

    raise ConstrainedClassifyError(
        f"backend did not converge on an allowed label after {max_retries} "
        f"attempts; last response was {last_response!r}, "
        f"allowed labels are {list(allowed)}"
    )
