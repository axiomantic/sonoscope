"""Tripwire-evaluator tests (Task D3, design §4.4 + §9).

Green-mirage discipline: every tripwire that can fire ``RED`` ships a RED
fixture built from a *real* fault (silent output when audio is expected, an
injected NaN/Inf, a subnormal flag, a non-zero clip count) paired with a GREEN
clean case, asserted with **exact enum equality** against the schema's
``TripwireVerdict`` values so a constant ``"PASS"`` stub cannot pass.

Coverage map (D3 acceptance, impl plan lines 292-295):

- ``test_silent_when_audio_expected_is_RED`` — silent block (rms below -80) +
  ``expected_audio=True`` → ``silent-output`` verdict EXACTLY ``"RED"`` and
  ``overall == "RED"``.
- ``test_silence_stimulus_silent_is_PASS`` — ``silence`` stimulus derives
  ``expected_audio=False``; silent output is PASS (no false RED).
- ``test_expected_audio_override`` — spec override wins over the derived value
  in both directions (False beats derived True; True beats derived False).
- ``test_clipping_red`` / ``test_nan_red`` — RED verdicts flow to ``overall``.

Additional green-mirage / design-input coverage:

- ``test_inf_red`` + clean pair for the ``nan-inf`` tripwire's Inf leg.
- ``test_denormal_red`` / ``test_denormal_clean_pass`` — the ``denormal``
  tripwire (design §4.4 lists it) gets its RED+GREEN pair.
- ``test_silent_output_keys_on_summary_rms_not_integrity_is_silent`` — proves
  the D2-review design input: the silent-output verdict keys on the summary's
  mean-channel ``rms_dbfs``, NOT ``integrity.is_silent`` (which OR-combines a
  dead single channel).
- ``test_expected_audio_derivation_table`` — the full §4.4 derivation table.
- ``test_healthy_render_all_pass`` — a clean render yields all-PASS + PASS
  overall (GREEN baseline).

Inputs are real ``DeterministicSummary`` / ``IntegrityBlock`` instances so a
stub cannot pass on a truthiness shortcut.
"""

from __future__ import annotations

from typing import get_args

import pytest

from sonoscope.features.tripwires import (
    CLIPPING_ID,
    DENORMAL_ID,
    NAN_INF_ID,
    SILENT_OUTPUT_ID,
    _DERIVED_EXPECTED_AUDIO,
    derive_expected_audio,
    evaluate_tripwires,
)
from sonoscope.schema.models import (
    DeterministicSummary,
    IntegrityBlock,
    StimulusKind,
    TripwireResult,
    TripwiresBlock,
)

# Frozen -80 dBFS silence cutoff (design §4.2); the summary's mean-channel
# rms_dbfs is compared against this by the silent-output tripwire.
_SILENCE_THRESHOLD_DBFS = -80.0
_DC_OFFSET_THRESHOLD = 0.001


def _summary(rms_dbfs: float = -18.0) -> DeterministicSummary:
    """A structurally healthy summary; ``rms_dbfs`` is the knob under test."""
    return DeterministicSummary(
        duration_s=1.0,
        sample_rate_hz=48000,
        channels=2,
        rms_dbfs=rms_dbfs,
        peak_dbfs=-6.0,
        crest_factor_db=12.0,
        dc_offset=0.0,
        spectral_centroid_hz=1000.0,
        spectral_bandwidth_hz=800.0,
        spectral_rolloff_hz=4000.0,
        spectral_flatness=0.1,
        zero_crossing_rate=0.05,
        onset_count=0,
        onset_rate_hz=0.0,
        tempo_bpm=None,
        tempo_confidence=None,
        mfcc_mean=[0.0] * 13,
        mfcc_std=[0.0] * 13,
    )


def _integrity(
    *,
    is_silent: bool = False,
    has_nan: bool = False,
    has_inf: bool = False,
    has_denormal: bool = False,
    clip_count: int = 0,
    clip_fraction: float = 0.0,
    dc_offset_exceeds: bool = False,
) -> IntegrityBlock:
    """A clean integrity block; each fault flag is an explicit override."""
    return IntegrityBlock(
        is_silent=is_silent,
        silence_threshold_dbfs=_SILENCE_THRESHOLD_DBFS,
        has_nan=has_nan,
        has_inf=has_inf,
        has_denormal=has_denormal,
        clip_count=clip_count,
        clip_fraction=clip_fraction,
        dc_offset_exceeds=dc_offset_exceeds,
        dc_offset_threshold=_DC_OFFSET_THRESHOLD,
    )


def _result(block: TripwiresBlock, tripwire_id: str) -> TripwireResult:
    """Fetch the single result with ``tripwire_id`` (exactly one must exist)."""
    matches = [r for r in block.results if r.id == tripwire_id]
    assert len(matches) == 1, f"expected exactly one '{tripwire_id}', got {matches}"
    return matches[0]


# --- silent-output tripwire (R3, the primary guard) -------------------------


def test_silent_when_audio_expected_is_RED() -> None:
    # Real fault: audio expected (note-on/instrument MIDI) but the mean-channel
    # RMS is at/below -80 dBFS → the classic silent headless render.
    block = evaluate_tripwires(
        summary=_summary(rms_dbfs=-95.0),
        integrity=_integrity(is_silent=True),
        stimulus_kind="midi",
    )
    assert block.expected_audio is True
    assert _result(block, SILENT_OUTPUT_ID).verdict == "RED"
    assert block.overall == "RED"


def test_silence_stimulus_silent_is_PASS() -> None:
    # A silence stimulus derives expected_audio=False, so a silent output is
    # correct — no false RED.
    block = evaluate_tripwires(
        summary=_summary(rms_dbfs=-95.0),
        integrity=_integrity(is_silent=True),
        stimulus_kind="silence",
    )
    assert block.expected_audio is False
    assert _result(block, SILENT_OUTPUT_ID).verdict == "PASS"
    assert block.overall == "PASS"


def test_silent_output_keys_on_summary_rms_not_integrity_is_silent() -> None:
    # D2-review design input: the silent-output verdict keys on the summary's
    # MEAN-channel rms_dbfs, NOT integrity.is_silent (which OR-combines and would
    # flag a single dead channel of an otherwise-loud stereo render).
    # is_silent=True (dead channel) but mean rms is loud → must NOT be RED.
    loud_but_one_dead = evaluate_tripwires(
        summary=_summary(rms_dbfs=-12.0),
        integrity=_integrity(is_silent=True),
        stimulus_kind="tone",
    )
    assert _result(loud_but_one_dead, SILENT_OUTPUT_ID).verdict == "PASS"
    assert loud_but_one_dead.overall == "PASS"

    # Reciprocal: mean rms below threshold fires RED even if integrity.is_silent
    # happens to be False (proving the key is rms_dbfs, not the flag).
    quiet_flag_off = evaluate_tripwires(
        summary=_summary(rms_dbfs=-95.0),
        integrity=_integrity(is_silent=False),
        stimulus_kind="tone",
    )
    assert _result(quiet_flag_off, SILENT_OUTPUT_ID).verdict == "RED"
    assert quiet_flag_off.overall == "RED"


def test_silent_output_boundary_is_inclusive() -> None:
    # The silent-output predicate is inclusive: rms_dbfs <= silence_threshold
    # (design §9 "at/below"). Sample the boundary EXACTLY at the -80 dBFS cutoff
    # with audio expected (tone) and a clean integrity block. RED is required, so
    # a <=→< off-by-one mutation (which makes -80.0 < -80.0 == False → PASS) is
    # killed rather than surviving unsampled.
    block = evaluate_tripwires(
        summary=_summary(rms_dbfs=_SILENCE_THRESHOLD_DBFS),
        integrity=_integrity(),
        stimulus_kind="tone",
    )
    assert block.expected_audio is True
    assert _result(block, SILENT_OUTPUT_ID).verdict == "RED"
    assert block.overall == "RED"


# --- expected_audio derivation (§4.4 S2) ------------------------------------


def test_expected_audio_override() -> None:
    # Spec override False beats a derived True (note-on/instrument MIDI): a
    # deliberately-muted patch marked expected_audio=False must not RED on
    # silence.
    overridden_false = evaluate_tripwires(
        summary=_summary(rms_dbfs=-95.0),
        integrity=_integrity(is_silent=True),
        stimulus_kind="midi",
        spec_expected_audio=False,
    )
    assert overridden_false.expected_audio is False
    assert _result(overridden_false, SILENT_OUTPUT_ID).verdict == "PASS"
    assert overridden_false.overall == "PASS"

    # And the reverse: override True beats a derived False (silence stimulus),
    # so a silent output DOES fire RED.
    overridden_true = evaluate_tripwires(
        summary=_summary(rms_dbfs=-95.0),
        integrity=_integrity(is_silent=True),
        stimulus_kind="silence",
        spec_expected_audio=True,
    )
    assert overridden_true.expected_audio is True
    assert _result(overridden_true, SILENT_OUTPUT_ID).verdict == "RED"
    assert overridden_true.overall == "RED"


def test_expected_audio_derivation_table() -> None:
    # Exact §4.4 table: silence → False; every other stimulus kind → True.
    assert derive_expected_audio("silence") is False
    for kind in ("midi", "audio", "impulse", "sweep", "pink_noise", "tone"):
        assert derive_expected_audio(kind) is True, kind
    # Override wins in both directions regardless of the derived default.
    assert derive_expected_audio("silence", spec_expected_audio=True) is True
    assert derive_expected_audio("midi", spec_expected_audio=False) is False


def test_unknown_stimulus_kind_hard_errors() -> None:
    # Deterministic ground truth: no silent default for an unmapped kind.
    with pytest.raises(ValueError):
        derive_expected_audio("banjo")  # type: ignore[arg-type]


def test_derivation_table_covers_every_stimulus_kind() -> None:
    # Exhaustiveness lock: the derivation table must map EXACTLY the set of
    # StimulusKind schema members. If a new kind is added to the Literal but not
    # to _DERIVED_EXPECTED_AUDIO, the prod path only fails at runtime (ValueError)
    # with no test flagging it — this turns that silent gap into a CI failure.
    schema_kinds = set(get_args(StimulusKind))

    # The private table maps exactly the schema's kinds — no missing, no extra.
    assert set(_DERIVED_EXPECTED_AUDIO) == schema_kinds

    # And every kind derives a bool through the public function without raising.
    for kind in schema_kinds:
        assert isinstance(derive_expected_audio(kind), bool), kind

    # An obviously-invalid kind still hard-errors (guards against a bare-default).
    with pytest.raises(ValueError):
        derive_expected_audio("banjo")  # type: ignore[arg-type]


# --- nan-inf tripwire -------------------------------------------------------


def test_nan_red() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(has_nan=True),
        stimulus_kind="tone",
    )
    assert _result(block, NAN_INF_ID).verdict == "RED"
    assert block.overall == "RED"


def test_inf_red() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(has_inf=True),
        stimulus_kind="tone",
    )
    assert _result(block, NAN_INF_ID).verdict == "RED"
    assert block.overall == "RED"


def test_nan_inf_clean_pass() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(),
        stimulus_kind="tone",
    )
    assert _result(block, NAN_INF_ID).verdict == "PASS"


# --- denormal tripwire ------------------------------------------------------


def test_denormal_red() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(has_denormal=True),
        stimulus_kind="tone",
    )
    assert _result(block, DENORMAL_ID).verdict == "RED"
    assert block.overall == "RED"


def test_denormal_clean_pass() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(),
        stimulus_kind="tone",
    )
    assert _result(block, DENORMAL_ID).verdict == "PASS"


# --- clipping tripwire ------------------------------------------------------


def test_clipping_red() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(clip_count=42, clip_fraction=0.001),
        stimulus_kind="tone",
    )
    assert _result(block, CLIPPING_ID).verdict == "RED"
    assert block.overall == "RED"


def test_clipping_clean_pass() -> None:
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(clip_count=0, clip_fraction=0.0),
        stimulus_kind="tone",
    )
    assert _result(block, CLIPPING_ID).verdict == "PASS"


# --- block shape + healthy baseline -----------------------------------------


def test_healthy_render_all_pass() -> None:
    # GREEN baseline: a clean, audible render → every tripwire PASS, overall
    # PASS, and exactly the four §4.4 tripwires in the documented order.
    block = evaluate_tripwires(
        summary=_summary(rms_dbfs=-18.0),
        integrity=_integrity(),
        stimulus_kind="tone",
    )
    assert [r.id for r in block.results] == [
        SILENT_OUTPUT_ID,
        NAN_INF_ID,
        DENORMAL_ID,
        CLIPPING_ID,
    ]
    assert all(r.verdict == "PASS" for r in block.results)
    assert block.overall == "PASS"
    assert block.expected_audio is True


def test_multiple_faults_overall_red() -> None:
    # Any RED → overall RED; here clipping and nan both fire.
    block = evaluate_tripwires(
        summary=_summary(),
        integrity=_integrity(has_nan=True, clip_count=3, clip_fraction=0.0001),
        stimulus_kind="tone",
    )
    assert _result(block, NAN_INF_ID).verdict == "RED"
    assert _result(block, CLIPPING_ID).verdict == "RED"
    assert block.overall == "RED"
