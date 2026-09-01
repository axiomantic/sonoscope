"""Task 1 — pure ``evaluate_descriptors`` comparator.

RED+GREEN per fireable check: every reason id, PASS, and each malformed-block
``InputError`` path has an exact-equality assertion so both a constant-``PASS``
stub and a constant-``RED`` stub fail. Plus the drift-guards (AST source-scan +
corpus emittability) and the MIDI contract/skip.
"""

from __future__ import annotations

import ast
import json
import pathlib
from typing import Any

import pytest

from sonoscope.descriptors.deriver import derive_descriptors
from sonoscope.errors import InputError
from sonoscope.features.descriptor_gate import (
    AUDIO_GATE_ELIGIBLE_TERMS,
    DESCRIPTORS_BLOCK_MALFORMED,
    DESCRIPTORS_EXPECTED_SPEC_INVALID,
    MIDI_GATE_ELIGIBLE_TERMS,
    DescriptorEvaluation,
    ExpectedDescriptors,
    _ExpectValue,
    _HYBRID_TERMS,
    evaluate_descriptors,
    load_expected_descriptors,
)
from sonoscope.schema.models import (
    AdvisoryDescriptor,
    DescriptorsBlock,
    DescriptorsLibrary,
    HybridDescriptor,
    MeasuredDescriptor,
)


def _block(measured, hybrid=None, advisory=None) -> DescriptorsBlock:
    """F7 helper: supply the required ``summary`` + ``library`` on every block."""
    return DescriptorsBlock(
        measured=list(measured),
        hybrid=list(hybrid or []),
        advisory=list(advisory or []),
        summary="",
        library=DescriptorsLibrary(thresholds_sha256="x", deriver_version="test"),
    )


def _example_block() -> DescriptorsBlock:
    """Example audio block (gate-eligible: bright/loud/busy/rhythmic-density)."""
    return _block(
        [
            MeasuredDescriptor(
                term="bright", value=3200.0, metric="spectral_centroid_hz", direction="high"
            ),
            MeasuredDescriptor(
                term="loud", value=-12.0, metric="rms_dbfs", direction="high"
            ),
            MeasuredDescriptor(
                term="busy", value=9.0, metric="onset_rate_hz", direction="high"
            ),
            MeasuredDescriptor(
                term="rhythmic-density", value=9.0, metric="onset_rate_hz", direction="value"
            ),
        ]
    )


# --- Step 1.1 — GREEN baseline ------------------------------------------------


def test_evaluate_pass_full_spec() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(
        expect_present=["bright", "loud"],
        expect_absent=["dark", "quiet"],
        expect_value=[
            _ExpectValue(term="bright", min=2500.0),
            _ExpectValue(term="rhythmic-density", min=8.0, max=12.0),
        ],
    )
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


# --- Step 1.3 — one test per reason id (exact single-element reasons) ----------


def test_desc_missing() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_present=["dense"])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_MISSING: dense"]
    )


def test_desc_unexpected() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_absent=["bright"])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_UNEXPECTED: bright"]
    )


def test_desc_value_absent() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="quiet", min=-30.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_VALUE_ABSENT: quiet"]
    )


def test_desc_value_nonfinite_nan() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=float("nan"), metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", min=-6.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_VALUE_NONFINITE: loud value=nan"]
    )


def test_desc_value_nonfinite_inf() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=float("inf"), metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", min=-6.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_VALUE_NONFINITE: loud value=inf"]
    )


def test_desc_value_out_of_range_min_only() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", min=-6.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=["DESC_VALUE_OUT_OF_RANGE: loud value=-12.0 not in [-6.0, inf]"],
    )


def test_desc_value_out_of_range_max_only() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="bright", max=2500.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=["DESC_VALUE_OUT_OF_RANGE: bright value=3200.0 not in [-inf, 2500.0]"],
    )


def test_desc_value_out_of_range_min_max() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="busy", min=10.0, max=20.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=["DESC_VALUE_OUT_OF_RANGE: busy value=9.0 not in [10.0, 20.0]"],
    )


def test_desc_value_out_of_range_equals() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=98.0, metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(
        expect_value=[_ExpectValue(term="loud", equals=100.0, tolerance=0.5)]
    )
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=["DESC_VALUE_OUT_OF_RANGE: loud value=98.0 not within 0.5 of 100.0"],
    )


# --- Step 1.4 — boundary GREENs (inclusive [min,max]; |v-equals|<=tol) ---------


def test_boundary_value_equals_min_passes() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=-6.0, metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", min=-6.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


def test_boundary_value_equals_max_passes() -> None:
    block = _block(
        [
            MeasuredDescriptor(
                term="bright", value=2500.0, metric="spectral_centroid_hz", direction="high"
            )
        ]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="bright", max=2500.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


def test_boundary_equals_at_tolerance_passes() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=100.5, metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(
        expect_value=[_ExpectValue(term="loud", equals=100.0, tolerance=0.5)]
    )
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


def test_boundary_equals_default_tolerance_zero_exact_passes() -> None:
    block = _block(
        [MeasuredDescriptor(term="loud", value=50.0, metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", equals=50.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


def test_equals_default_tolerance_zero_mismatch_out_of_range_red() -> None:
    # S2: no tolerance supplied → defaults to 0.0; any deviation is OUT_OF_RANGE.
    # Pairs the exact-match PASS above: proves the 0.0 default is a real bound,
    # not an "always accept" fallthrough.
    block = _block(
        [MeasuredDescriptor(term="loud", value=50.0, metric="rms_dbfs", direction="high")]
    )
    spec = ExpectedDescriptors(expect_value=[_ExpectValue(term="loud", equals=49.0)])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=["DESC_VALUE_OUT_OF_RANGE: loud value=50.0 not within 0.0 of 49.0"],
    )


# --- Step 1.5 — ordering + per-entry precedence -------------------------------


def test_multi_section_red_fixed_order() -> None:
    block = _example_block()
    spec = ExpectedDescriptors(
        expect_present=["dense"],
        expect_absent=["bright"],
        expect_value=[
            _ExpectValue(term="quiet", min=-30.0),
            _ExpectValue(term="loud", min=-6.0),
        ],
    )
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=[
            "DESC_MISSING: dense",
            "DESC_UNEXPECTED: bright",
            "DESC_VALUE_ABSENT: quiet",
            "DESC_VALUE_OUT_OF_RANGE: loud value=-12.0 not in [-6.0, inf]",
        ],
    )


def test_per_entry_precedence_absent_over_out_of_range() -> None:
    block = _example_block()
    # 'quiet' is absent AND carries a band; precedence: ABSENT beats OUT_OF_RANGE.
    spec = ExpectedDescriptors(
        expect_value=[_ExpectValue(term="quiet", min=-30.0, max=-10.0)]
    )
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_VALUE_ABSENT: quiet"]
    )


# --- Step 1.6 — malformed-block InputErrors (exact .code + .detail) -----------


def test_duplicate_gate_eligible_term_raises() -> None:
    block = _block(
        [
            MeasuredDescriptor(
                term="bright", value=3200.0, metric="spectral_centroid_hz", direction="high"
            ),
            MeasuredDescriptor(
                term="bright", value=4000.0, metric="spectral_centroid_hz", direction="high"
            ),
        ]
    )
    spec = ExpectedDescriptors()
    with pytest.raises(InputError) as exc:
        evaluate_descriptors(block, spec, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_BLOCK_MALFORMED
    assert exc.value.detail == {"reason": "duplicate_term", "term": "bright"}


def test_unknown_emitted_term_raises() -> None:
    block = _block(
        [MeasuredDescriptor(term="wobble", value=1.0, metric="m", direction="value")]
    )
    spec = ExpectedDescriptors()
    with pytest.raises(InputError) as exc:
        evaluate_descriptors(block, spec, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_BLOCK_MALFORMED
    assert exc.value.detail == {"reason": "unknown_emitted_term", "term": "wobble"}


# --- Step 1.7 — invisibility of estimated / hybrid / advisory content ----------


def test_estimated_hybrid_advisory_invisible_to_gate() -> None:
    block = _block(
        measured=[
            MeasuredDescriptor(
                term="bright", value=3200.0, metric="spectral_centroid_hz", direction="high"
            ),
            MeasuredDescriptor(
                term="tempo-audio",
                value=120.0,
                metric="tempo_bpm",
                direction="value",
                estimated=True,
            ),
        ],
        hybrid=[
            HybridDescriptor(
                term="driving",
                anchor_metric="driving_composite",
                anchor_value=0.7,
                direction="high",
            )
        ],
        advisory=[AdvisoryDescriptor(term="cosmic", source="lalm-mapped")],
    )
    spec = ExpectedDescriptors(expect_present=["tempo-audio", "driving", "cosmic"])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED",
        reasons=[
            "DESC_MISSING: tempo-audio",
            "DESC_MISSING: driving",
            "DESC_MISSING: cosmic",
        ],
    )


# --- Step 1.8 — MIDI-context dispatch + cross-context isolation ---------------
# Closes the ``block_kind="midi"`` branch of the eligibility dispatch without the
# C2 producer: a hand-built block carrying a MIDI-vocabulary measured row is only
# accepted under ``block_kind="midi"`` (the MIDI eligibility set), and is rejected
# as an out-of-context emitted term under ``block_kind="audio"`` — and vice versa.


def test_midi_dispatch_gate_eligible_and_value_pass() -> None:
    # A MIDI-vocabulary measured row is emittable + gate-eligible ONLY under the
    # MIDI set. If dispatch always returned the AUDIO set, "note-density" would be
    # an out-of-context emitted term → InputError instead of this PASS.
    block = _block(
        [
            MeasuredDescriptor(
                term="note-density", value=8.0, metric="notes_per_second", direction="value"
            )
        ]
    )
    spec = ExpectedDescriptors(
        expect_present=["note-density"],
        expect_absent=["polyphony"],
        expect_value=[_ExpectValue(term="note-density", min=4.0, max=12.0)],
    )
    assert evaluate_descriptors(block, spec, block_kind="midi") == DescriptorEvaluation(
        verdict="PASS", reasons=[]
    )


def test_midi_context_audio_term_in_spec_is_missing() -> None:
    # Cross-context (a): an AUDIO term asserted via expect_present against a MIDI
    # block is simply not in the eligible map → DESC_MISSING (not treated present).
    block = _block(
        [
            MeasuredDescriptor(
                term="note-density", value=8.0, metric="notes_per_second", direction="value"
            )
        ]
    )
    spec = ExpectedDescriptors(expect_present=["bright"])
    assert evaluate_descriptors(block, spec, block_kind="midi") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_MISSING: bright"]
    )


def test_audio_context_midi_term_in_spec_is_missing() -> None:
    # Cross-context (b): a MIDI term asserted against an audio block is absent from
    # the audio eligible map → DESC_MISSING.
    block = _example_block()
    spec = ExpectedDescriptors(expect_present=["note-density"])
    assert evaluate_descriptors(block, spec, block_kind="audio") == DescriptorEvaluation(
        verdict="RED", reasons=["DESC_MISSING: note-density"]
    )


def test_midi_term_emitted_in_audio_block_raises() -> None:
    # Cross-context isolation (strong form): a MIDI-vocabulary emitted row is out
    # of context under block_kind="audio" → malformed block. This pins the set to
    # the block_kind argument; the mirror PASS above pins the "midi" branch.
    block = _block(
        [
            MeasuredDescriptor(
                term="note-density", value=8.0, metric="notes_per_second", direction="value"
            )
        ]
    )
    spec = ExpectedDescriptors()
    with pytest.raises(InputError) as exc:
        evaluate_descriptors(block, spec, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_BLOCK_MALFORMED
    assert exc.value.detail == {"reason": "unknown_emitted_term", "term": "note-density"}


# --- Step 1.11 — drift-guard: AST source-scan of deriver.py -------------------


def _deriver_non_estimated_terms() -> set[str]:
    src = pathlib.Path("src/sonoscope/descriptors/deriver.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    terms: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MeasuredDescriptor":
            kw = {k.arg: k.value for k in node.keywords}
            est = kw.get("estimated")
            is_estimated = isinstance(est, ast.Constant) and est.value is True
            term_node = kw.get("term")
            if (
                not is_estimated
                and isinstance(term_node, ast.Constant)
                and isinstance(term_node.value, str)
            ):
                terms.add(term_node.value)
    return terms


def test_audio_eligible_terms_match_deriver_ast() -> None:
    assert _deriver_non_estimated_terms() == set(AUDIO_GATE_ELIGIBLE_TERMS)


def _deriver_hybrid_terms() -> set[str]:
    # deriver._hybrid builds its rows by looping over a literal tuple of
    # (term, anchor_metric, score) tuples; scan that loop's iterable and collect
    # the first (``term``) element of every inner tuple. Robust to threshold /
    # score-expr changes because it keys off the loop structure, not the scores.
    src = pathlib.Path("src/sonoscope/descriptors/deriver.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    terms: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_hybrid":
            for sub in ast.walk(node):
                if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Tuple):
                    for elt in sub.iter.elts:
                        if isinstance(elt, ast.Tuple) and elt.elts:
                            first = elt.elts[0]
                            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                                terms.add(first.value)
    return terms


def test_hybrid_terms_match_deriver_ast() -> None:
    # Drift-guard: _HYBRID_TERMS must mirror the exact set of hybrid terms the
    # deriver emits (driving/punchy/warm). If deriver adds/removes a hybrid and
    # _HYBRID_TERMS is not updated, this fails.
    assert _deriver_hybrid_terms() == set(_HYBRID_TERMS)


# --- Step 1.12 — drift-guard: corpus emittability -----------------------------


def _summary(**overrides: Any):
    from sonoscope.schema.models import DeterministicSummary

    base: dict[str, Any] = {
        "duration_s": 8.0,
        "sample_rate_hz": 48000,
        "channels": 2,
        "rms_dbfs": -25.0,
        "peak_dbfs": -10.0,
        "crest_factor_db": 10.0,
        "dc_offset": 0.0,
        "spectral_centroid_hz": 2000.0,
        "spectral_bandwidth_hz": 1000.0,
        "spectral_rolloff_hz": 4000.0,
        "spectral_flatness": 0.1,
        "zero_crossing_rate": 0.1,
        "onset_count": 0,
        "onset_rate_hz": 4.0,
        "tempo_bpm": None,
        "tempo_confidence": None,
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [0.0] * 13,
    }
    base.update(overrides)
    return DeterministicSummary(**base)


def test_every_audio_eligible_term_is_emittable() -> None:
    # High summary → bright, loud, dynamic, busy, dense, rhythmic-density.
    s_high = _summary(
        spectral_centroid_hz=4000.0,
        rms_dbfs=-10.0,
        crest_factor_db=20.0,
        onset_rate_hz=10.0,
        spectral_bandwidth_hz=3000.0,
    )
    # Low summary → dark, quiet, compressed, spare, rhythmic-density.
    s_low = _summary(
        spectral_centroid_hz=500.0,
        rms_dbfs=-40.0,
        crest_factor_db=3.0,
        onset_rate_hz=1.0,
        spectral_bandwidth_hz=1000.0,
    )
    emitted: set[str] = set()
    for s in (s_high, s_low):
        emitted |= {d.term for d in derive_descriptors(s).measured if not d.estimated}
    # ``silent`` is reachable only through the whole-file-silence gate, which
    # suppresses every other term, so it needs its own pass.
    emitted |= {
        d.term
        for d in derive_descriptors(s_low, is_silent=True).measured
        if not d.estimated
    }
    assert emitted == set(AUDIO_GATE_ELIGIBLE_TERMS)


# --- Step 1.14 — MIDI contract (frozen interface) + skipped integration -------


def test_midi_gate_eligible_terms_frozen() -> None:
    assert MIDI_GATE_ELIGIBLE_TERMS == frozenset(
        {"note-density", "register", "pitch-range", "polyphony", "velocity-dynamics", "ioi"}
    )


@pytest.mark.integration
def test_midi_gate_eligible_terms_pending_c2() -> None:
    pytest.skip("C2 producer not landed")


# =============================================================================
# Task 2 — load_expected_descriptors: fail-loud spec loader + validators.
#
# RED+GREEN per L-row. Every fail-loud path asserts the EXACT ``.code`` +
# ``.detail`` (full-dict equality), so a stub loader that never validates
# (returns a bare/empty ``ExpectedDescriptors`` without raising) fails every
# L-row test. Two GREEN round-trips (path + dict) prove valid specs load.
# =============================================================================


# --- Step 2.1 — GREEN: loader round-trip + path/dict parity -------------------


def test_load_valid_dict_and_path_parity(tmp_path) -> None:
    payload = {
        "expect_present": ["bright"],
        "expect_absent": ["dark"],
        "expect_value": [{"term": "loud", "min": -6.0}],
    }
    from_dict = load_expected_descriptors(payload, block_kind="audio")
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    from_path = load_expected_descriptors(str(p), block_kind="audio")
    expected = ExpectedDescriptors(
        expect_present=["bright"],
        expect_absent=["dark"],
        expect_value=[_ExpectValue(term="loud", min=-6.0)],
    )
    assert from_dict == expected
    assert from_path == expected


def test_load_valid_equals_form_dict() -> None:
    # A valid ``equals`` spec (tolerance omitted) round-trips; proves the
    # equals/tolerance branch of the structural validator accepts a well-formed
    # entry rather than rejecting everything.
    payload = {"expect_value": [{"term": "loud", "equals": -6.0}]}
    assert load_expected_descriptors(payload, block_kind="audio") == ExpectedDescriptors(
        expect_value=[_ExpectValue(term="loud", equals=-6.0)]
    )


# --- Step 2.2 — L1–L13: one exact-.code + exact-.detail RED test per row -------


def test_load_l1_not_an_object() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors([1, 2], block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "not_an_object"}
    assert exc.value.component == "analyze"


def test_load_l2_unreadable(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(str(missing), block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "unreadable"}
    assert exc.value.component == "analyze"


def test_load_l2_non_utf8(tmp_path) -> None:
    p = tmp_path / "non-utf8.json"
    p.write_bytes(b"\xff\xfe not valid utf-8 bytes")
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(str(p), block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "unreadable"}
    assert exc.value.component == "analyze"
    assert exc.value.exit_code == 2


def test_load_l2_unparseable(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(str(p), block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "unparseable"}
    assert exc.value.component == "analyze"


def test_load_l3_unknown_top_level_key() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors({"bogus": 1}, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "unknown_top_level_key"}
    assert exc.value.component == "analyze"


def test_load_l4_unknown_term() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors({"expect_present": ["wobble"]}, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {
        "reason": "unknown_term",
        "term": "wobble",
        "block_kind": "audio",
    }
    assert exc.value.component == "analyze"


def test_load_l5_cross_context_term() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors({"expect_present": ["note-density"]}, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {
        "reason": "cross_context_term",
        "term": "note-density",
        "block_kind": "audio",
    }
    assert exc.value.component == "analyze"


@pytest.mark.parametrize("term", ["driving", "warm", "cosmic", "tempo-audio"])
def test_load_l6_term_not_gate_eligible(term: str) -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors({"expect_present": [term]}, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {
        "reason": "term_not_gate_eligible",
        "term": term,
        "block_kind": "audio",
    }
    assert exc.value.component == "analyze"


def test_load_l7_expect_value_no_bound() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright"}]}, block_kind="audio"
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "expect_value_no_bound"}
    assert exc.value.component == "analyze"


def test_load_l8_expect_value_mixed_bound() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright", "equals": 1.0, "min": 0.0}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "expect_value_mixed_bound"}
    assert exc.value.component == "analyze"


def test_load_l9_tolerance_without_equals() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright", "min": 0.0, "tolerance": 0.5}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "tolerance_without_equals"}
    assert exc.value.component == "analyze"


@pytest.mark.parametrize("tolerance", [-0.5, float("inf")])
def test_load_l10_bad_tolerance(tolerance: float) -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright", "equals": 1.0, "tolerance": tolerance}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "bad_tolerance"}
    assert exc.value.component == "analyze"


def test_load_l11_min_gt_max() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright", "min": 10.0, "max": 5.0}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "min_gt_max"}
    assert exc.value.component == "analyze"


def test_load_l12_non_finite_bound() -> None:
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "bright", "min": float("inf")}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "non_finite_bound"}
    assert exc.value.component == "analyze"


def test_load_l13_invalid_field_type() -> None:
    # A wrong-typed top-level field (str where list[str] is required) raises a
    # pydantic-native ValidationError whose errors()[0]["type"] is NOT
    # "extra_forbidden"; the loader's committed fallback maps it to
    # "invalid_field_type" (a first-class, TESTED L-row, not a silent catch-all).
    with pytest.raises(InputError) as exc:
        load_expected_descriptors({"expect_present": "not-a-list"}, block_kind="audio")
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "invalid_field_type"}
    assert exc.value.component == "analyze"


def test_load_nested_unknown_key() -> None:
    # A bogus key NESTED inside an expect_value entry also trips pydantic's
    # extra="forbid", but its error loc is deep (('expect_value', 0, 'bogus')),
    # NOT top-level. The loader distinguishes it as ``unknown_nested_key`` rather
    # than the imprecise top-level ``unknown_top_level_key`` (L3).
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {"expect_value": [{"term": "loud", "equals": 1.0, "bogus": 2}]},
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {"reason": "unknown_nested_key"}
    assert exc.value.component == "analyze"


def test_load_eligibility_precedence_first_ineligible_reported() -> None:
    # Ineligible terms appear in expect_present AND expect_absent AND expect_value.
    # _enforce_term_eligibility concatenates present + absent + value(entry.term)
    # and raises on the FIRST ineligible term, so expect_present's 'wobble'
    # (unknown_term) is reported — not expect_absent's 'cosmic' or expect_value's
    # 'note-density'. Pins the documented precedence order.
    with pytest.raises(InputError) as exc:
        load_expected_descriptors(
            {
                "expect_present": ["wobble"],
                "expect_absent": ["cosmic"],
                "expect_value": [{"term": "note-density", "min": 0.0}],
            },
            block_kind="audio",
        )
    assert exc.value.code == DESCRIPTORS_EXPECTED_SPEC_INVALID
    assert exc.value.detail == {
        "reason": "unknown_term",
        "term": "wobble",
        "block_kind": "audio",
    }
    assert exc.value.component == "analyze"
