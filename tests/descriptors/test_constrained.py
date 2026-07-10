"""Deterministic unit tests for the constrained-classifier primitive (Task T6).

These exercise ONLY the non-LLM parts of ``constrained_classify`` — label
validation, retry-to-convergence, non-convergence failure, and temperature
pass-through — by injecting a deterministic fake backend (``model=fake``). No
real model loads in core CI (design §9.1). Assertions are exact-equality.
"""

from __future__ import annotations

import pytest

from sonoscope.descriptors.constrained import (
    ConstrainedClassifyError,
    constrained_classify,
)


class _FakeBackend:
    """Deterministic ``ClassifierBackend`` stand-in.

    Returns the queued responses in order; once a single response remains it is
    returned on every subsequent call (models the "always invalid" case). Records
    every ``(prompt, temperature)`` it is called with so tests can assert the
    temperature that was passed through.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, float]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(self, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


def test_accepts_exact_label() -> None:
    fake = _FakeBackend(["techno"])
    result = constrained_classify("prompt", ("techno", "house"), model=fake)
    assert result == "techno"
    assert fake.call_count == 1


def test_retries_then_converges() -> None:
    fake = _FakeBackend(["technoish", "techno"])
    result = constrained_classify("prompt", ("techno", "house"), model=fake)
    assert result == "techno"
    # One invalid attempt, then one valid attempt == exactly one retry.
    assert fake.call_count == 2


def test_raises_on_non_convergence() -> None:
    fake = _FakeBackend(["technoish"])
    with pytest.raises(ConstrainedClassifyError):
        constrained_classify("prompt", ("techno", "house"), model=fake)
    # Default max_retries == 3 => exactly three attempts before giving up.
    assert fake.call_count == 3


def test_strips_whitespace_from_response() -> None:
    # A real LLM often returns trailing/leading whitespace or a newline. The
    # response must be stripped before validation so "techno\n" is accepted on the
    # FIRST attempt rather than treated as a violation and retried.
    fake = _FakeBackend(["techno\n"])
    result = constrained_classify("prompt", ("techno", "house"), model=fake)
    assert result == "techno"
    assert fake.call_count == 1


def test_retry_prompt_accumulates_violation_history() -> None:
    # Across retries the re-prompt must ACCUMULATE every prior rejected label, not
    # reset to the original prompt + only the latest violation. With two distinct
    # invalid labels before a valid one, the THIRD prompt the backend sees must
    # mention BOTH earlier rejected labels so the model has the full rejection
    # history and cannot re-propose an already-rejected label.
    fake = _FakeBackend(["technoish", "housey", "techno"])
    result = constrained_classify("prompt", ("techno", "house"), model=fake)
    assert result == "techno"
    assert fake.call_count == 3
    third_prompt = fake.calls[2][0]
    assert "technoish" in third_prompt
    assert "housey" in third_prompt


def test_temperature_zero_default() -> None:
    fake = _FakeBackend(["techno"])
    constrained_classify("prompt", ("techno", "house"), model=fake)
    assert fake.calls[0][1] == 0.0


def test_raises_on_empty_allowed_labels() -> None:
    # An empty label space can NEVER match, so it is silent misuse: without an
    # up-front guard the retry loop would waste every attempt and then raise
    # ConstrainedClassifyError. Surface it as a clear ValueError instead. The
    # fake is a valid ClassifierBackend (never invoked when the guard fires).
    fake = _FakeBackend(["techno"])
    with pytest.raises(ValueError):
        constrained_classify("prompt", (), model=fake)


def test_raises_on_nonpositive_max_retries() -> None:
    # max_retries < 1 skips the retry loop entirely (the backend is never
    # queried) and then raises ConstrainedClassifyError — silent misuse. Surface
    # it as a clear ValueError. The fake is a valid ClassifierBackend (never
    # invoked when the guard fires).
    fake = _FakeBackend(["techno"])
    with pytest.raises(ValueError):
        constrained_classify("prompt", ("techno", "house"), model=fake, max_retries=0)
