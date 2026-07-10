"""Tests for the productized probe engine (Task G1 / M9, src/sonoscope/probe.py).

The engine runs the A/B fixture judgment through a PerceptionAdapter and emits a
probe verdict (PASS / WEAK / FAIL) per the design §10.3 threshold:
PASS = ratio >= 0.80, WEAK = 0.50 <= ratio < 0.80,
FAIL = ratio < 0.50. All tests here are NON-integration: a stub adapter supplies
canned descriptions so the judge + verdict logic runs without the ~16 GB model.

Green-mirage discipline (AGENTS.md): each check ships a RED case (wrong-direction
descriptions, a broken verdict boundary) alongside the GREEN case, so a judge or
threshold that ignored direction/ratio would fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sonoscope.probe import (
    FixturePair,
    ProbeFixturesMissing,
    ProbeReport,
    ProbeUnavailable,
    classify_verdict,
    judge_pair,
    run_probe,
)
from sonoscope.schema.models import AdapterInfo, PerceptionBlock


class _StubAdapter:
    """PerceptionAdapter stub: returns a canned description keyed by wav filename."""

    id = "stub"
    grounding = "advisory-freetext"

    def __init__(self, descs: dict[str, str], available: bool = True) -> None:
        self._descs = descs
        self._available = available

    def describe(self, wav_path, deterministic=None) -> PerceptionBlock:
        return PerceptionBlock(
            status="ok",
            grounding="advisory-freetext",
            adapter=AdapterInfo(
                id="stub", model="m", quant="none", runtime="test",
                model_sha256="0" * 64,
            ),
            description=self._descs[Path(wav_path).name],
            grounding_map={},
            disclaimer="Advisory only.",
        )

    def health(self):
        from sonoscope.perception.base import AdapterHealth

        return AdapterHealth(
            available=self._available,
            runtime="test",
            model_id="m",
            reason=None if self._available else "adapter down",
        )


def test_classify_verdict_thresholds():
    # Exact PASS/WEAK/FAIL boundaries (design §10.3).
    assert classify_verdict(5, 5) == "PASS"
    assert classify_verdict(4, 5) == "PASS"   # 0.80 -> PASS (the B3 threshold)
    assert classify_verdict(3, 5) == "WEAK"   # 0.60 -> WEAK
    assert classify_verdict(3, 6) == "WEAK"   # 0.50 -> WEAK (lower boundary)
    assert classify_verdict(2, 5) == "FAIL"   # 0.40 -> FAIL
    assert classify_verdict(0, 5) == "FAIL"


def test_judge_pair_correct_and_incorrect():
    pair = FixturePair(
        key="cutoff",
        desc="brightness",
        metric="centroid_hz",
        higher_side="a",
        wav_a=Path("a.wav"),
        wav_b=Path("b.wav"),
    )
    # GREEN: higher-centroid side (a) called bright, lower side (b) called dark.
    assert judge_pair(pair, "a bright tone", "a dark muffled sound") is True
    # RED: descriptions swapped -> the direction does not track -> incorrect.
    assert judge_pair(pair, "a dark muffled sound", "a bright tone") is False


def _touch_pairs(tmp_path, names: tuple[str, ...]) -> None:
    """Create empty wav files so ``run_probe``'s up-front fixture-existence guard
    passes; the ``_StubAdapter`` keys canned descriptions by filename and never reads
    the bytes, so empty files suffice for the judge/aggregate logic under test."""
    for name in names:
        (tmp_path / name).write_bytes(b"")


def test_run_probe_aggregates_and_verdict(tmp_path):
    _touch_pairs(tmp_path, ("hi.wav", "lo.wav", "phi.wav", "plo.wav"))
    pairs = [
        FixturePair("cutoff", "brightness", "centroid_hz", "a",
                    tmp_path / "hi.wav", tmp_path / "lo.wav"),
        FixturePair("pitch", "pitch", "centroid_hz", "a",
                    tmp_path / "phi.wav", tmp_path / "plo.wav"),
    ]
    descs = {
        "hi.wav": "a bright tone",
        "lo.wav": "a dark muffled sound",
        "phi.wav": "bright and high-pitched",
        "plo.wav": "dark and low-pitched",
    }
    report = run_probe(_StubAdapter(descs), pairs)

    assert isinstance(report, ProbeReport)
    assert report.n_correct == 2
    assert report.m_total == 2
    assert report.ratio == 1.0
    assert report.verdict == "PASS"
    assert [j.correct for j in report.pairs] == [True, True]


def test_run_probe_weak_when_half_track(tmp_path):
    _touch_pairs(tmp_path, ("hi.wav", "lo.wav", "phi.wav", "plo.wav"))
    pairs = [
        FixturePair("cutoff", "brightness", "centroid_hz", "a",
                    tmp_path / "hi.wav", tmp_path / "lo.wav"),
        FixturePair("pitch", "pitch", "centroid_hz", "a",
                    tmp_path / "phi.wav", tmp_path / "plo.wav"),
    ]
    descs = {
        "hi.wav": "a bright tone",
        "lo.wav": "a dark muffled sound",       # cutoff: correct
        "phi.wav": "a dark muffled sound",
        "plo.wav": "a bright tone",              # pitch: swapped -> incorrect
    }
    report = run_probe(_StubAdapter(descs), pairs)
    assert report.n_correct == 1
    assert report.ratio == 0.5
    assert report.verdict == "WEAK"


def test_run_probe_raises_when_adapter_unavailable():
    pairs = [
        FixturePair("cutoff", "brightness", "centroid_hz", "a",
                    Path("hi.wav"), Path("lo.wav")),
    ]
    with pytest.raises(ProbeUnavailable):
        run_probe(_StubAdapter({}, available=False), pairs)


def test_run_probe_raises_when_fixtures_absent(tmp_path):
    """FIX 3 (RED-proving): an AVAILABLE adapter with fixture wavs that do not exist
    on disk raises ``ProbeFixturesMissing`` (validated up front, before any
    ``describe``) rather than letting a raw ``FileNotFoundError`` escape from the
    adapter. The missing paths are surfaced on the exception."""
    wav_a = tmp_path / "cutoff__cutoff_high.wav"
    wav_b = tmp_path / "cutoff__cutoff_low.wav"  # neither written -> both absent
    pairs = [
        FixturePair("cutoff", "brightness", "centroid_hz", "a", wav_a, wav_b),
    ]
    with pytest.raises(ProbeFixturesMissing) as excinfo:
        run_probe(_StubAdapter({}, available=True), pairs)
    assert excinfo.value.missing == [wav_a, wav_b]


def test_run_probe_health_checked_before_fixtures(tmp_path):
    """Ordering guard: a no-model run still degrades to ``ProbeUnavailable`` even when
    fixtures are absent — the health check precedes fixture validation, so a
    missing-model machine never sees ``ProbeFixturesMissing``."""
    pairs = [
        FixturePair(
            "cutoff", "brightness", "centroid_hz", "a",
            tmp_path / "absent_a.wav", tmp_path / "absent_b.wav",
        ),
    ]
    with pytest.raises(ProbeUnavailable):
        run_probe(_StubAdapter({}, available=False), pairs)
