"""JSON Schema generation from the Pydantic models (design section 3.7).

The Pydantic models in :mod:`sonoscope.schema.models` are the single source of
truth. This module emits a **draft 2020-12** JSON Schema *generated from* those
models (never hand-maintained) so the published contract cannot drift from the
code. It backs the ``schema`` command and the ``check_schema_version`` guard.
"""

from __future__ import annotations

from typing import Any

from sonoscope.errors import UsageError
from sonoscope.schema.models import (
    SCHEMA_VERSION,
    AnalysisReport,
    DeterminismFloors,
    FatalError,
    IterateDelta,
    MidiAnalysisReport,
    WavAnalysisReport,
)

#: The JSON Schema draft 2020-12 dialect URI. Pydantic v2 generates 2020-12
#: shapes but does not stamp ``$schema``; we set it explicitly (design 3.7).
DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"

_MODEL_BY_KIND: dict[str, type[Any]] = {
    "analysis": AnalysisReport,
    "iterate-delta": IterateDelta,
    "determinism-floors": DeterminismFloors,
    "fatal-error": FatalError,
    "midi-analysis": MidiAnalysisReport,
    "wav-analysis": WavAnalysisReport,
}

#: The kinds the ``schema`` command understands (design 4.5 / 3.7).
SCHEMA_KINDS = tuple(_MODEL_BY_KIND)


def json_schema_for(kind: str) -> dict[str, Any]:
    """Return the draft-2020-12 JSON Schema for one report ``kind``.

    ``kind`` is one of :data:`SCHEMA_KINDS`. An unknown kind is a usage error
    (mapped to exit code 1), never a silent fallback.
    """
    try:
        model = _MODEL_BY_KIND[kind]
    except KeyError:
        raise UsageError(
            "USAGE_UNKNOWN_SCHEMA_KIND",
            f"unknown schema kind {kind!r}; expected one of "
            + ", ".join(SCHEMA_KINDS),
            detail={"kind": kind, "expected": list(SCHEMA_KINDS)},
            component="schema",
        ) from None
    schema = model.model_json_schema()
    # Stamp the dialect last so our explicit 2020-12 stamp wins even if pydantic
    # later emits its own ``$schema`` key.
    return {**schema, "$schema": DRAFT_2020_12_URI}


def check_schema_version(major: int) -> None:
    """Validate a report ``schema_version`` *major* against this build.

    Per design 3.1 a consumer must reject a major it does not understand. A
    mismatch raises :class:`~sonoscope.errors.UsageError` (exit 1); a matching
    major returns ``None``.
    """
    current_major = int(SCHEMA_VERSION.split(".")[0])
    if major != current_major:
        raise UsageError(
            "USAGE_SCHEMA_VERSION_MISMATCH",
            f"unsupported schema_version major {major}; this build understands "
            f"major {current_major} (schema_version {SCHEMA_VERSION})",
            detail={"requested_major": major, "supported_major": current_major},
            component="schema",
        )
    return None
