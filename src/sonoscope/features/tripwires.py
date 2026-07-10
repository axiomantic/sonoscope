"""Tripwire evaluator + expected_audio derivation (Task D3, by design).

Pure function ``evaluate_tripwires(summary, integrity, stimulus_kind,
spec_expected_audio) -> TripwiresBlock``. It maps the D1 deterministic summary
and D2 integrity flags — the ground-truth layer — plus the stimulus/spec
expectation context onto the ``tripwires`` block of the analysis contract
(schema ``TripwiresBlock``): the four first-class tripwires (``silent-output``,
``nan-inf``, ``denormal``, ``clipping``), each with an exact
``TripwireVerdict``, plus the ``overall`` roll-up.

Design invariants honored here:

- **Deterministic = ground truth ("Architecture invariants").** Every
  verdict is a pure function of the already-computed summary/integrity values;
  no re-analysis, no randomness, exact enum equality.
- **expected_audio derivation (by design).** Derived from the stimulus kind
  (``silence`` → ``False``; any sounding stimulus → ``True``); an explicit spec
  override WINS over the derived default. An all-muting param set is the one
  ``False`` case that is *not* auto-detected and must be declared via the spec
  override (by design, table row 4).
- **Silent-output keys on the summary's mean-channel ``rms_dbfs`` (R3;
  D2-review design input).** The silent-output tripwire compares the summary's
  ``rms_dbfs`` against the frozen ``silence_threshold_dbfs`` — NOT
  ``integrity.is_silent``. ``integrity.is_silent`` OR-combines across channels
  ("at least one channel silent"), which is a *dead-channel* signal, not a
  whole-render silence signal; keying the R3 tripwire off it would false-RED an
  audible stereo render with one dead channel. The single source of truth for
  the -80 dBFS cutoff is ``integrity.silence_threshold_dbfs`` (D1's hashed
  ``FROZEN_PARAMS``), read off the integrity block passed in.
- **overall roll-up (schema line 225).** ``overall = RED`` if ANY tripwire
  is ``RED``, else ``PASS``. ``ERROR`` is reserved for a distinct upstream
  error condition (a failed/absent analysis surfaced by the orchestrator) and is
  intentionally NOT produced here — this pure evaluator, given a valid
  summary+integrity, only ever emits ``PASS``/``RED`` so ``ERROR`` is never
  conflated with a healthy ``RED`` finding.
- **No dc-offset tripwire (exactly four by design; D2-review NIT).** The four
  documented tripwires do not include a dc-offset verdict, so the D2 side effect where
  an all-Inf/NaN channel also sets ``integrity.dc_offset_exceeds`` cannot be
  double-counted: the ``nan-inf`` tripwire owns has_nan/has_inf and nothing here
  keys on ``dc_offset_exceeds``.

D3 imports D1/D2 by module path and does NOT edit ``features/__init__`` (D1 owns
that surface), ``librosa_features.py``, or ``integrity.py``.
"""

from __future__ import annotations

from typing import Optional

from sonoscope.schema.models import (
    DeterministicSummary,
    IntegrityBlock,
    StimulusKind,
    TripwireResult,
    TripwiresBlock,
    TripwireVerdict,
)

# --- Verdict constants (schema TripwireVerdict values; SSOT for comparisons) -
PASS: TripwireVerdict = "PASS"
RED: TripwireVerdict = "RED"
# ERROR is reserved for a distinct upstream error condition (see module docstring
# "overall roll-up"); this evaluator never emits it, but the alias documents the
# third enum member and keeps the reference next to PASS/RED.
ERROR: TripwireVerdict = "ERROR"

# --- Tripwire ids (JSON example, lines 220-223; stable order) ----------------
SILENT_OUTPUT_ID = "silent-output"
NAN_INF_ID = "nan-inf"
DENORMAL_ID = "denormal"
CLIPPING_ID = "clipping"

# --- expected_audio derivation table (lines 512-517) --------------------------
# ``silence`` expects silence; every other stimulus kind is a sounding stimulus
# (note-on/instrument MIDI, or an effect fed audio-in: impulse/sweep/tone/
# pink_noise/audio-stem) and expects non-silence. The all-muting param-set
# ``False`` case (row 4) is intentionally absent: it is NOT auto-detected and is
# expressed only via an explicit spec override. Every ``StimulusKind`` member is
# listed so an added kind fails loudly (KeyError) rather than defaulting silently.
_DERIVED_EXPECTED_AUDIO: dict[StimulusKind, bool] = {
    "silence": False,
    "midi": True,
    "audio": True,
    "impulse": True,
    "sweep": True,
    "pink_noise": True,
    "tone": True,
}


def derive_expected_audio(
    stimulus_kind: StimulusKind,
    spec_expected_audio: Optional[bool] = None,
) -> bool:
    """Derive ``expected_audio`` from the stimulus, spec override winning.

    ``spec_expected_audio`` is the optional ``expected_audio`` field from the
    resolved spec: ``None`` means "derive from the stimulus kind";
    ``True``/``False`` overrides the derived default unconditionally. An unmapped
    ``stimulus_kind`` raises ``ValueError`` — deterministic ground truth admits no
    silent default for an unknown kind.
    """
    # Spec override wins over the derived default (line 519).
    if spec_expected_audio is not None:
        return spec_expected_audio
    try:
        return _DERIVED_EXPECTED_AUDIO[stimulus_kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown stimulus_kind {stimulus_kind!r}; cannot derive "
            f"expected_audio. Known kinds: "
            f"{sorted(_DERIVED_EXPECTED_AUDIO)}"
        ) from exc


def _silent_output_result(
    summary: DeterministicSummary,
    integrity: IntegrityBlock,
    expected_audio: bool,
) -> TripwireResult:
    """R3 silent-output tripwire — keys on ``summary.rms_dbfs`` (not is_silent).

    RED iff audio was expected AND the mean-channel RMS is at/below the frozen
    ``silence_threshold_dbfs`` (the single source of truth read from the
    integrity block). This is the primary guard against the "headless render
    silently produced default/empty state" failure.
    """
    threshold = integrity.silence_threshold_dbfs
    rms = summary.rms_dbfs
    output_is_silent = rms <= threshold
    if expected_audio and output_is_silent:
        return TripwireResult(
            id=SILENT_OUTPUT_ID,
            verdict=RED,
            detail=(
                f"rms_dbfs {rms:.1f} <= {threshold:.1f} dBFS "
                "(audio expected, output silent)"
            ),
        )
    if not expected_audio:
        return TripwireResult(
            id=SILENT_OUTPUT_ID,
            verdict=PASS,
            detail=f"audio not expected (rms_dbfs {rms:.1f})",
        )
    return TripwireResult(
        id=SILENT_OUTPUT_ID,
        verdict=PASS,
        detail=f"rms_dbfs {rms:.1f} > {threshold:.1f} dBFS",
    )


def _nan_inf_result(integrity: IntegrityBlock) -> TripwireResult:
    """NaN/Inf tripwire — RED iff any non-finite sample was detected (M1)."""
    if integrity.has_nan or integrity.has_inf:
        return TripwireResult(
            id=NAN_INF_ID,
            verdict=RED,
            detail=(
                f"has_nan={integrity.has_nan} has_inf={integrity.has_inf}"
            ),
        )
    return TripwireResult(id=NAN_INF_ID, verdict=PASS, detail=None)


def _denormal_result(integrity: IntegrityBlock) -> TripwireResult:
    """Denormal tripwire — RED iff subnormal float32 samples were detected."""
    if integrity.has_denormal:
        return TripwireResult(
            id=DENORMAL_ID,
            verdict=RED,
            detail="has_denormal=True (subnormal float32 samples present)",
        )
    return TripwireResult(id=DENORMAL_ID, verdict=PASS, detail=None)


def _clipping_result(integrity: IntegrityBlock) -> TripwireResult:
    """Clipping tripwire — RED iff any sample reached full scale (|x| >= 1.0)."""
    if integrity.clip_count > 0:
        return TripwireResult(
            id=CLIPPING_ID,
            verdict=RED,
            detail=(
                f"clip_count {integrity.clip_count} "
                f"(clip_fraction {integrity.clip_fraction})"
            ),
        )
    return TripwireResult(
        id=CLIPPING_ID,
        verdict=PASS,
        detail=f"clip_fraction {integrity.clip_fraction}",
    )


def evaluate_tripwires(
    summary: DeterministicSummary,
    integrity: IntegrityBlock,
    stimulus_kind: StimulusKind,
    spec_expected_audio: Optional[bool] = None,
) -> TripwiresBlock:
    """Evaluate the ``tripwires`` block from the deterministic ground truth.

    Pure function of the D1 ``summary``, the D2 ``integrity`` flags, and the
    expectation context (``stimulus_kind`` + optional spec ``expected_audio``
    override). Produces the four documented tripwires in documented order and the
    ``overall`` roll-up (RED if any tripwire RED, else PASS). ``ERROR`` is
    reserved for upstream failures and is never produced here.
    """
    expected_audio = derive_expected_audio(stimulus_kind, spec_expected_audio)

    # Fixed order matches the JSON example (lines 220-223).
    results = [
        _silent_output_result(summary, integrity, expected_audio),
        _nan_inf_result(integrity),
        _denormal_result(integrity),
        _clipping_result(integrity),
    ]

    # overall = RED if ANY tripwire is RED, else PASS (ERROR reserved).
    overall: TripwireVerdict = (
        RED if any(r.verdict == RED for r in results) else PASS
    )

    return TripwiresBlock(
        expected_audio=expected_audio,
        results=results,
        overall=overall,
    )
