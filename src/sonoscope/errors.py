"""The ``SonoscopeError`` hierarchy and fatal-error envelope rendering.

Design section 3.6 defines the exit-code contract and the fatal envelope. The
``ExitCode`` enum and the ``FatalError``/``FatalErrorDetail`` models are the
single source of truth in ``sonoscope.schema`` (plan C1); this module imports
them and never redefines them.

Each exception subclass maps to exactly one exit code via the authoritative
``EXIT_CODE_BY_CLASS`` map, whose values are the ``ExitCode`` enum members
(so they equal the design's exit-code table 1..5).
"""

from __future__ import annotations

from typing import Optional

from sonoscope.schema import Component, ExitCode, FatalError, FatalErrorDetail


class SonoscopeError(Exception):
    """Base for all fatal, exit-code-mapped sonoscope errors.

    Carries the structured fields of a design 3.6 error object. Non-fatal
    issues do NOT use this class; they are collected in ``report.errors[]``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        detail: Optional[dict] = None,
        component: Component = "cli",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.component: Component = component

    @property
    def exit_code(self) -> ExitCode:
        """The mapped process exit code for this concrete error class."""
        return EXIT_CODE_BY_CLASS[type(self)]

    def to_fatal_error(
        self, *, sonoscope_version: str, generated_at: str
    ) -> FatalError:
        """Render this error to the design 3.6 fatal-error envelope model."""
        return FatalError(
            generated_at=generated_at,
            sonoscope_version=sonoscope_version,
            error=FatalErrorDetail(
                code=self.code,
                message=self.message,
                detail=self.detail,
                component=self.component,
            ),
        )


class UsageError(SonoscopeError):
    """Bad flags, unknown command, schema-version major mismatch (exit 1)."""


class InputError(SonoscopeError):
    """Input contract failure: missing/hash-mismatched item, bad param (exit 2)."""


class RenderError(SonoscopeError):
    """Render failed: backend load/crash, GUI-init required w/o raw_state (exit 3)."""


class MidiCaptureError(SonoscopeError):
    """MIDI capture failed: C host crash/error, malformed handoff (by design).

    Component ``"midi"``. A capture/host failure is the MIDI-path analogue of a
    :class:`RenderError`, so it maps to the same ``RENDER`` exit code (3): the C
    CLAP MIDI host is a spawned foreign binary whose crash/error must surface as a
    structured fatal error, never a silent empty capture.
    """


class AnalysisError(SonoscopeError):
    """Deterministic feature layer failed: unreadable wav, numeric fault (exit 4)."""


class SonoscopeEnvironmentError(SonoscopeError):
    """Pin/lockfile drift or required-runtime hard failure (exit 5)."""


# The single authoritative class -> exit-code map (design 3.6). Values are the
# schema ExitCode members, so this map both defines and matches the contract.
EXIT_CODE_BY_CLASS: dict[type[SonoscopeError], ExitCode] = {
    UsageError: ExitCode.USAGE,
    InputError: ExitCode.INPUT,
    RenderError: ExitCode.RENDER,
    # A MIDI-capture/host failure is the RENDER-family analogue (by design).
    MidiCaptureError: ExitCode.RENDER,
    AnalysisError: ExitCode.ANALYSIS,
    SonoscopeEnvironmentError: ExitCode.ENVIRONMENT,
}
