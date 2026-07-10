"""Tests for the SonoscopeError hierarchy, class->exit-code map, and the
fatal-error envelope rendering (design section 3.6; plan Task C5)."""

from __future__ import annotations

import pytest

from sonoscope.errors import (
    EXIT_CODE_BY_CLASS,
    AnalysisError,
    InputError,
    MidiCaptureError,
    RenderError,
    SonoscopeEnvironmentError,
    UsageError,
)
from sonoscope.schema import ExitCode, FatalError


def test_error_class_exit_code_map():
    # The single authoritative class->exit-code map (plan C5 / design 3.6).
    assert EXIT_CODE_BY_CLASS == {
        UsageError: 1,
        InputError: 2,
        RenderError: 3,
        # B1: a MIDI-capture/host failure is the RENDER-family analogue (exit 3).
        MidiCaptureError: 3,
        AnalysisError: 4,
        SonoscopeEnvironmentError: 5,
    }
    # ...and it is exactly the schema ExitCode single source of truth.
    assert EXIT_CODE_BY_CLASS == {
        UsageError: ExitCode.USAGE,
        InputError: ExitCode.INPUT,
        RenderError: ExitCode.RENDER,
        MidiCaptureError: ExitCode.RENDER,
        AnalysisError: ExitCode.ANALYSIS,
        SonoscopeEnvironmentError: ExitCode.ENVIRONMENT,
    }


@pytest.mark.parametrize(
    "cls,expected",
    [
        (UsageError, ExitCode.USAGE),
        (InputError, ExitCode.INPUT),
        (RenderError, ExitCode.RENDER),
        (AnalysisError, ExitCode.ANALYSIS),
        (SonoscopeEnvironmentError, ExitCode.ENVIRONMENT),
    ],
)
def test_subclass_maps_to_exact_exit_code(cls, expected):
    err = cls("SOME_CODE", "a message", component="cli")
    assert err.exit_code == expected
    assert err.exit_code == int(expected)


def test_error_renders_fatal_envelope():
    err = RenderError(
        "RENDER_SUBPROCESS_CRASH",
        "Render subprocess exited with signal SIGSEGV after 1 retry.",
        detail={"signal": "SIGSEGV", "retries": 1},
        component="render",
    )
    fatal = err.to_fatal_error(
        sonoscope_version="0.1.0",
        generated_at="2026-07-04T12:00:00Z",
    )
    assert isinstance(fatal, FatalError)
    assert fatal.model_dump() == {
        "schema_version": "1.4.0",  # SCHEMA_VERSION bump (wav-analysis + input_provenance)
        "kind": "fatal-error",
        "generated_at": "2026-07-04T12:00:00Z",
        "sonoscope_version": "0.1.0",
        "error": {
            "code": "RENDER_SUBPROCESS_CRASH",
            "message": "Render subprocess exited with signal SIGSEGV after 1 retry.",
            "detail": {"signal": "SIGSEGV", "retries": 1},
            "severity": "fatal",
            "component": "render",
        },
    }
