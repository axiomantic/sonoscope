"""Descriptors orchestrator-hook tests (Task T4 / design §7).

The hook wires the pure ``derive_descriptors`` into ``analyze_render_outcome``
after perception and before report assembly. With the T4 advisory stub
(``produce_advisory`` returning ``([], None, None, None)``) the hook is a pure
pass-through of the deriver: every ``AnalysisReport`` carries a populated
``descriptors`` block whose ``measured``/``hybrid``/``library`` are exactly the
deriver's output for that render's ``DeterministicSummary``, ``advisory == []``,
and the measured-only ``summary`` string.

Assertions are exact-equality only (F7): whole-block ``==`` against an
independently re-derived expected value, exact ``[]`` for advisory, and an exact
``DescriptorsLibrary`` for the provenance sub-block. No ``is not None``,
``isinstance``, or truthiness.
"""

from __future__ import annotations

from pathlib import Path

from sonoscope.analysis_orchestrator import analyze_render_outcome
from sonoscope.descriptors.advisory import map_freeform_to_advisory
from sonoscope.descriptors.deriver import derive_descriptors
from sonoscope.descriptors.summary import render_summary
from sonoscope.descriptors.thresholds import DERIVER_VERSION, thresholds_sha256
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.schema.models import AdapterInfo, DescriptorsLibrary, PerceptionBlock

from tests.test_analysis_orchestrator import (
    _ReturnedStatusAdapter,
    _fake_plugin,
    _outcome,
    _tone_stimulus,
)


def _report(tmp_path: Path, *, perception_enabled: bool = True):
    """Drive analyze_render_outcome on a real (non-silent) 1 kHz tone render."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    return analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=NullAdapter(),
        perception_enabled=perception_enabled,
    )


def _ok_perception(description: str) -> PerceptionBlock:
    """A status=='ok' perception block carrying a canned freeform description
    (status=='ok' requires the four adapter-output fields, models.py _OK_REQUIRED)."""
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


def _report_with_perception(tmp_path: Path, perception: PerceptionBlock):
    """Drive analyze_render_outcome with an adapter that RETURNS `perception`."""
    outcome = _outcome(tmp_path, stimulus=_tone_stimulus(), ref_sha256="b" * 64)
    return analyze_render_outcome(
        outcome,
        plugin_path=_fake_plugin(tmp_path),
        spec_sha256="c" * 64,
        adapter=_ReturnedStatusAdapter(perception),
    )


def test_report_carries_descriptors(tmp_path: Path) -> None:
    """The report's descriptors block is exactly the deriver's output for that
    render's summary (whole-block exact-equality: measured, hybrid, advisory==[],
    measured-only summary, and library)."""
    report = _report(tmp_path)
    expected = derive_descriptors(report.deterministic.summary)

    assert report.descriptors == expected


def test_measured_present_when_perception_disabled(tmp_path: Path) -> None:
    """Measured is independent of perception: with perception disabled the report
    still carries the exact deriver measured list (full-list exact-equality)."""
    report = _report(tmp_path, perception_enabled=False)
    expected = derive_descriptors(report.deterministic.summary)

    assert report.perception.status == "disabled"
    assert report.descriptors.measured == expected.measured


def test_measured_present_when_advisory_absent(tmp_path: Path) -> None:
    """With the advisory stub returning [], advisory is exactly [] and the summary
    is the exact measured-only render (exact string equality)."""
    report = _report(tmp_path)

    assert report.descriptors.advisory == []
    assert report.descriptors.summary == render_summary(
        report.descriptors.measured, report.descriptors.hybrid, []
    )


def test_library_stamped_in_report(tmp_path: Path) -> None:
    """The library provenance sub-block is stamped exactly: pinned digest, deriver
    version, and un-plumbed advisory coverage/dropped (None at this stage)."""
    report = _report(tmp_path)
    expected_library = DescriptorsLibrary(
        thresholds_sha256=thresholds_sha256(),
        deriver_version=DERIVER_VERSION,
        advisory_coverage=None,
        advisory_dropped=None,
    )

    assert report.descriptors.library == expected_library


# --- T5 coverage plumbing (design §11.7) — locks the T4 model_copy branches --
# that were dead under the advisory stub (exact-equality only, F7).


def test_advisory_coverage_plumbed_when_total_positive(tmp_path: Path) -> None:
    """A status=='ok' perception whose description maps >=1 candidate plumbs exact
    advisory_coverage (== matched/total), advisory_dropped (== total-matched), a
    non-empty advisory list, and a summary carrying the advisory clause."""
    description = "a spacey hypnotic brooding texture"
    report = _report_with_perception(tmp_path, _ok_perception(description))
    expected_advisory, matched, total = map_freeform_to_advisory(description)

    assert total == 3
    assert matched == 2
    assert report.descriptors.library.advisory_coverage == matched / total
    assert report.descriptors.library.advisory_dropped == total - matched
    assert report.descriptors.advisory == expected_advisory
    assert report.descriptors.summary == render_summary(
        report.descriptors.measured, report.descriptors.hybrid, expected_advisory
    )


def test_advisory_coverage_none_when_total_zero(tmp_path: Path) -> None:
    """A status=='ok' perception whose description maps nothing keeps coverage None
    and advisory [] (the total==0 guarded path — advisory_dropped stays None)."""
    report = _report_with_perception(
        tmp_path, _ok_perception("the quick brown fox jumped")
    )

    assert report.descriptors.library.advisory_coverage is None
    assert report.descriptors.library.advisory_dropped is None
    assert report.descriptors.advisory == []


def test_advisory_coverage_zero_when_matched_zero(tmp_path: Path) -> None:
    """A status=='ok' perception whose description yields candidates (total>0) that
    map to NO canonical term (matched==0) stamps coverage 0.0 (total>0, so not the
    None-guarded path) and advisory_dropped == total, while advisory stays [] and the
    summary is measured-only (no advisory clause). Distinct from total==0 (coverage
    None) — this is the total>0/matched==0 advisory path."""
    description = "brooding warbly"
    expected_advisory, matched, total = map_freeform_to_advisory(description)

    # Candidates found but none mappable: both surface forms are in
    # advisory._UNMAPPED_SURFACE_FORMS, so they count toward total yet drop.
    assert total == 2
    assert matched == 0
    assert expected_advisory == []

    report = _report_with_perception(tmp_path, _ok_perception(description))

    assert report.descriptors.library.advisory_coverage == 0.0
    assert report.descriptors.library.advisory_dropped == 2
    assert report.descriptors.advisory == []
    assert report.descriptors.summary == render_summary(
        report.descriptors.measured, report.descriptors.hybrid, []
    )
