"""Productized probe engine (design §10.3).

The perception feasibility gate, productized: this engine consumes an
already-rendered A/B fixture set, runs each side through a
:class:`PerceptionAdapter`, applies a deterministic directional judge, and emits
a probe verdict.

Judgment (honest, automated first-pass). Each A/B pair has a KNOWN deterministic
direction (fixed by librosa before the model is asked — the same ground truth the
fixtures encode). A pair is CORRECT iff the description of the metric-higher
side carries a "higher" directional term AND the lower side carries a "lower"
term, drawn from the term sets found reliable in earlier feasibility testing
(brightness/pitch via spectral centroid; noisiness via flatness; loudness via
RMS — kept as a documented axis even though loudness was the model's weak axis,
since the deterministic core owns level). This is a keyword directional check,
not a semantic parser.

Verdict thresholds (design §10.3):
PASS = ratio >= 0.80, WEAK = 0.50 <= ratio < 0.80, FAIL = ratio < 0.50.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .perception.base import PerceptionAdapter

__all__ = [
    "FixturePair",
    "PairJudgment",
    "ProbeFixturesMissing",
    "ProbeReport",
    "ProbeUnavailable",
    "ProbeVerdict",
    "classify_verdict",
    "default_fixture_pairs",
    "judge_pair",
    "run_probe",
]

ProbeVerdict = Literal["PASS", "WEAK", "FAIL"]
Metric = Literal["centroid_hz", "flatness", "rms_dbfs"]
Side = Literal["a", "b"]

_PASS_RATIO = 0.80
_WEAK_RATIO = 0.50

# Directional term sets per metric (derived from the observed reliable axes in
# earlier feasibility testing). Lowercase substring matches against the model's
# free-text description; substrings ("nois") catch inflections (noisy/noise).
_DIRECTION_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "centroid_hz": {
        "high": ("bright", "high", "sharp"),
        "low": ("dark", "muffled", "dull", "low"),
    },
    "flatness": {
        "high": ("nois",),
        "low": ("clear", "tonal", "tone", "pure"),
    },
    "rms_dbfs": {
        "high": ("loud", "strong"),
        "low": ("quiet", "faint", "soft"),
    },
}


class ProbeUnavailable(Exception):
    """Raised when the perception adapter cannot run the probe (health() reports
    unavailable), so a caller never mistakes an all-incorrect no-model run for a
    genuine FAIL verdict."""


class ProbeFixturesMissing(Exception):
    """Raised when the adapter IS available but one or more fixture wavs referenced
    by the pairs do not exist on disk. Validated up front (before any ``describe``)
    so an available-model probe run fails with this clear typed error instead of a
    raw ``FileNotFoundError`` from deep inside the adapter. ``missing`` lists the
    absent wav paths for the caller to surface."""

    def __init__(self, missing: list[Path]) -> None:
        self.missing = missing
        super().__init__(
            f"{len(missing)} probe fixture wav(s) missing: "
            + ", ".join(str(p) for p in missing)
        )


@dataclass(frozen=True)
class FixturePair:
    """One A/B fixture pair with a KNOWN deterministic direction.

    ``metric`` names the librosa metric that fixes correctness; ``higher_side``
    is the side ('a' or 'b') that metric is higher on. ``wav_a`` / ``wav_b`` are
    the already-rendered fixture wavs fed to the adapter.
    """

    key: str
    desc: str
    metric: Metric
    higher_side: Side
    wav_a: Path
    wav_b: Path


@dataclass(frozen=True)
class PairJudgment:
    """The adapter's descriptions for one pair plus the directional correctness."""

    key: str
    a_description: str
    b_description: str
    correct: bool


@dataclass(frozen=True)
class ProbeReport:
    """Aggregate probe result: per-pair judgments, the N-of-M score, and verdict."""

    pairs: list[PairJudgment]
    n_correct: int
    m_total: int
    ratio: float
    verdict: ProbeVerdict


def classify_verdict(n_correct: int, m_total: int) -> ProbeVerdict:
    """Map an N-of-M score to the PASS/WEAK/FAIL verdict (design §10.3).

    PASS = ratio >= 0.80 (for M=5 this is the B3 ">= 4 of 5" threshold),
    WEAK = 0.50 <= ratio < 0.80, FAIL = ratio < 0.50.
    """
    if m_total <= 0:
        raise ValueError("m_total must be positive")
    ratio = n_correct / m_total
    if ratio >= _PASS_RATIO:
        return "PASS"
    if ratio >= _WEAK_RATIO:
        return "WEAK"
    return "FAIL"


def _has_term(description: str, terms: tuple[str, ...]) -> bool:
    lowered = description.lower()
    return any(term in lowered for term in terms)


def judge_pair(pair: FixturePair, a_description: str, b_description: str) -> bool:
    """Directional judge: does the pair's descriptions track the KNOWN metric
    direction? Correct iff the metric-higher side carries a "higher" term and the
    lower side carries a "lower" term (design §10.3, B3 axes)."""
    terms = _DIRECTION_TERMS[pair.metric]
    if pair.higher_side == "a":
        higher_desc, lower_desc = a_description, b_description
    else:
        higher_desc, lower_desc = b_description, a_description
    return _has_term(higher_desc, terms["high"]) and _has_term(lower_desc, terms["low"])


def run_probe(adapter: PerceptionAdapter, pairs: list[FixturePair]) -> ProbeReport:
    """Run the A/B fixture judgment through ``adapter`` and emit a verdict.

    Raises :class:`ProbeUnavailable` when the adapter reports it cannot run, so a
    missing model surfaces as an explicit unavailability rather than a fake FAIL.
    Raises :class:`ProbeFixturesMissing` when the adapter is available but a fixture
    wav is absent, so a mispointed/missing fixture dir is a clean typed error rather
    than a raw ``FileNotFoundError`` from inside ``describe``.
    """
    health = adapter.health()
    if not health.available:
        raise ProbeUnavailable(health.reason or "perception adapter unavailable")

    # Validate all fixture wavs exist BEFORE describing any (health checked first so
    # a no-model run still degrades gracefully to ProbeUnavailable, never this error).
    missing = [
        wav
        for pair in pairs
        for wav in (pair.wav_a, pair.wav_b)
        if not wav.exists()
    ]
    if missing:
        raise ProbeFixturesMissing(missing)

    judgments: list[PairJudgment] = []
    for pair in pairs:
        a_block = adapter.describe(pair.wav_a)
        b_block = adapter.describe(pair.wav_b)
        a_desc = a_block.description or ""
        b_desc = b_block.description or ""
        judgments.append(
            PairJudgment(
                key=pair.key,
                a_description=a_desc,
                b_description=b_desc,
                correct=judge_pair(pair, a_desc, b_desc),
            )
        )

    n_correct = sum(1 for j in judgments if j.correct)
    m_total = len(judgments)
    ratio = n_correct / m_total if m_total else 0.0
    return ProbeReport(
        pairs=judgments,
        n_correct=n_correct,
        m_total=m_total,
        ratio=ratio,
        verdict=classify_verdict(n_correct, m_total),
    )


# Canonical A/B fixture metadata: key, human desc, metric, metric-higher side,
# and the (a_label, b_label) that form the committed fixture filenames
# "{key}__{label}.wav".
_CANONICAL_PAIRS: tuple[tuple[str, str, Metric, Side, str, str], ...] = (
    ("cutoff", "lowpass cutoff HIGH vs LOW (brightness)", "centroid_hz", "a",
     "cutoff_high", "cutoff_low"),
    ("noisiness", "tonal osc vs noise osc", "flatness", "b", "tonal", "noisy"),
    ("pitch", "high note vs low note", "centroid_hz", "a", "pitch_high", "pitch_low"),
    ("timbre", "rich saw vs pure sine", "centroid_hz", "a", "saw", "sine"),
    ("loudness", "loud vs quiet", "rms_dbfs", "a", "loud", "quiet"),
)


def default_fixture_pairs(fixtures_dir: Path) -> list[FixturePair]:
    """Build the canonical 5-pair A/B set against ``fixtures_dir``.

    Filenames follow the committed convention ``{key}__{label}.wav``. The caller
    supplies the directory (the engine does not couple to a fixed location), so
    the CLI can point this at wherever the probe fixtures are provisioned.
    """
    pairs: list[FixturePair] = []
    for key, desc, metric, higher_side, a_label, b_label in _CANONICAL_PAIRS:
        pairs.append(
            FixturePair(
                key=key,
                desc=desc,
                metric=metric,
                higher_side=higher_side,
                wav_a=fixtures_dir / f"{key}__{a_label}.wav",
                wav_b=fixtures_dir / f"{key}__{b_label}.wav",
            )
        )
    return pairs
