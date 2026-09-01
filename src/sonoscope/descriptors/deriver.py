"""Pure audio measured/hybrid deriver (by design).

``derive_descriptors(summary)`` maps a :class:`DeterministicSummary` onto a
:class:`DescriptorsBlock`: up to 11 ``measured`` rows (eight single-metric gated
terms, one multi-metric gated ``dense``, two conditional readouts) and three
``hybrid`` composites, at frozen absolute thresholds (``DERIVER_THRESHOLDS``).

Purity contract: no I/O, clock, RNG, or perception dependency. Identical
``summary`` -> byte-identical ``measured`` + ``hybrid`` + ``library``. The
``advisory`` list is left empty and ``library.advisory_coverage`` /
``advisory_dropped`` stay ``None`` (the orchestrator merges advisory later,
by design). ``library`` is stamped with ``thresholds_sha256()`` and
``DERIVER_VERSION``; ``summary`` is the measured-only render.

Boundary-operator asymmetry (by design, intentional — do NOT harmonize):
measured gated terms fire with strict ``>``/``<`` (at-threshold = no fire);
hybrid feel-terms fire with inclusive ``>=`` (at-threshold = fire).

Digital silence short-circuits the whole deriver: when the caller passes
``is_silent=True`` (the integrity layer's whole-file verdict) the output is the
single measured term ``silent`` and nothing else. Without that gate the timbral
terms fire on the absence of signal — an all-zeros buffer reads as "dark, quiet,
compressed, warm", the last of those a hybrid composed from the MFCCs of nothing.

``spare`` requires positive onset activity (``0 < onset_rate_hz < max``): a
fully silent clip (``onset_rate_hz == 0.0``) is not "sparse" but empty, mirroring
the same-rationale suppression of the ``rhythmic-density`` readout at zero. This
is what makes ``measured == []`` reachable for a degenerate summary (F2).
"""

from __future__ import annotations

from sonoscope.descriptors.summary import render_summary
from sonoscope.descriptors.thresholds import (
    DERIVER_THRESHOLDS,
    DERIVER_VERSION,
    thresholds_sha256,
)
from sonoscope.schema.models import (
    DescriptorsBlock,
    DescriptorsLibrary,
    DeterministicSummary,
    HybridDescriptor,
    MeasuredDescriptor,
)

T = DERIVER_THRESHOLDS


def norm(x: float, lo: float, hi: float) -> float:
    """``clamp((x - lo) / (hi - lo), 0.0, 1.0)`` (by design)."""
    if hi == lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _measured(summary: DeterministicSummary) -> list[MeasuredDescriptor]:
    rows: list[MeasuredDescriptor] = []

    # Gated single-metric terms, in table order (by design).
    if summary.spectral_centroid_hz > T["bright.centroid_hz_min"]:
        rows.append(
            MeasuredDescriptor(
                term="bright",
                value=summary.spectral_centroid_hz,
                metric="spectral_centroid_hz",
                direction="high",
                threshold=T["bright.centroid_hz_min"],
            )
        )
    if summary.spectral_centroid_hz < T["dark.centroid_hz_max"]:
        rows.append(
            MeasuredDescriptor(
                term="dark",
                value=summary.spectral_centroid_hz,
                metric="spectral_centroid_hz",
                direction="low",
                threshold=T["dark.centroid_hz_max"],
            )
        )
    if summary.rms_dbfs > T["loud.rms_dbfs_min"]:
        rows.append(
            MeasuredDescriptor(
                term="loud",
                value=summary.rms_dbfs,
                metric="rms_dbfs",
                direction="high",
                threshold=T["loud.rms_dbfs_min"],
            )
        )
    if summary.rms_dbfs < T["quiet.rms_dbfs_max"]:
        rows.append(
            MeasuredDescriptor(
                term="quiet",
                value=summary.rms_dbfs,
                metric="rms_dbfs",
                direction="low",
                threshold=T["quiet.rms_dbfs_max"],
            )
        )
    if summary.crest_factor_db < T["compressed.crest_db_max"]:
        rows.append(
            MeasuredDescriptor(
                term="compressed",
                value=summary.crest_factor_db,
                metric="crest_factor_db",
                direction="low",
                threshold=T["compressed.crest_db_max"],
            )
        )
    if summary.crest_factor_db > T["dynamic.crest_db_min"]:
        rows.append(
            MeasuredDescriptor(
                term="dynamic",
                value=summary.crest_factor_db,
                metric="crest_factor_db",
                direction="high",
                threshold=T["dynamic.crest_db_min"],
            )
        )
    if summary.onset_rate_hz > T["busy.onset_hz_min"]:
        rows.append(
            MeasuredDescriptor(
                term="busy",
                value=summary.onset_rate_hz,
                metric="onset_rate_hz",
                direction="high",
                threshold=T["busy.onset_hz_min"],
            )
        )
    if 0.0 < summary.onset_rate_hz < T["spare.onset_hz_max"]:
        rows.append(
            MeasuredDescriptor(
                term="spare",
                value=summary.onset_rate_hz,
                metric="onset_rate_hz",
                direction="low",
                threshold=T["spare.onset_hz_max"],
            )
        )

    # Gated multi-metric term: dense = AND of onset activity and spectral
    # fullness. The emitted record carries the onset anchor (value + threshold);
    # bandwidth is a co-gate, not the emitted threshold (by design).
    if (
        summary.onset_rate_hz > T["dense.onset_hz_min"]
        and summary.spectral_bandwidth_hz > T["dense.bandwidth_hz_min"]
    ):
        rows.append(
            MeasuredDescriptor(
                term="dense",
                value=summary.onset_rate_hz,
                metric="onset_rate_hz+spectral_bandwidth_hz",
                direction="high",
                threshold=T["dense.onset_hz_min"],
            )
        )

    # Readouts (conditional, direction="value"). rhythmic-density is suppressed
    # at onset_rate_hz == 0.0; tempo-audio is emitted only when its gate passes.
    if summary.onset_rate_hz > 0.0:
        rows.append(
            MeasuredDescriptor(
                term="rhythmic-density",
                value=summary.onset_rate_hz,
                metric="onset_rate_hz",
                direction="value",
                threshold=None,
            )
        )
    if (
        summary.tempo_bpm is not None
        and summary.onset_count >= T["tempo.min_onsets"]
        and T["tempo.bpm_min"] <= summary.tempo_bpm <= T["tempo.bpm_max"]
    ):
        rows.append(
            MeasuredDescriptor(
                term="tempo-audio",
                value=summary.tempo_bpm,
                metric="tempo_bpm",
                direction="value",
                threshold=None,
                estimated=True,
                confidence=summary.tempo_confidence,
            )
        )

    return rows


def _driving_score(summary: DeterministicSummary) -> float:
    onset = norm(summary.onset_rate_hz, T["driving.onset_lo"], T["driving.onset_hi"])
    tempo = (
        norm(summary.tempo_bpm, T["driving.tempo_lo"], T["driving.tempo_hi"])
        if summary.tempo_bpm is not None
        else 0.0
    )
    rms = norm(summary.rms_dbfs, T["driving.rms_lo"], T["driving.rms_hi"])
    return (
        T["driving.w_onset"] * onset
        + T["driving.w_tempo"] * tempo
        + T["driving.w_rms"] * rms
    )


def _punchy_score(summary: DeterministicSummary) -> float:
    crest = norm(summary.crest_factor_db, T["punchy.crest_lo"], T["punchy.crest_hi"])
    onset = norm(summary.onset_rate_hz, T["punchy.onset_lo"], T["punchy.onset_hi"])
    return T["punchy.w_crest"] * crest + T["punchy.w_onset"] * onset


def _warm_score(summary: DeterministicSummary) -> float:
    segment = summary.mfcc_mean[
        int(T["warm.mfcc_lo_idx"]) : int(T["warm.mfcc_hi_idx"]) + 1
    ]
    # An empty/short mfcc_mean yields an empty slice; the low-mid term then
    # contributes a NEUTRAL 0.0 rather than raising ZeroDivisionError. The centroid
    # term still contributes, so the warm score is only partially neutralized.
    mfcc_low_mid = sum(segment) / len(segment) if segment else 0.0
    centroid = norm(
        summary.spectral_centroid_hz, T["warm.centroid_lo"], T["warm.centroid_hi"]
    )
    mfcc = norm(mfcc_low_mid, T["warm.mfcc_lo"], T["warm.mfcc_hi"])
    return T["warm.w_centroid"] * (1 - centroid) + T["warm.w_mfcc"] * mfcc


def _hybrid(summary: DeterministicSummary) -> list[HybridDescriptor]:
    rows: list[HybridDescriptor] = []
    for term, anchor_metric, score in (
        ("driving", "driving_composite", _driving_score(summary)),
        ("punchy", "punchy_composite", _punchy_score(summary)),
        ("warm", "warm_composite", _warm_score(summary)),
    ):
        if score >= T[f"{term}.fire"]:
            rows.append(
                HybridDescriptor(
                    term=term,
                    anchor_metric=anchor_metric,
                    anchor_value=score,
                    direction="high",
                    confidence=score,
                )
            )
    return rows


def _silent_block(summary: DeterministicSummary) -> DescriptorsBlock:
    """The whole-file-silent output: exactly one measured term, no hybrids.

    ``threshold`` is None because the gate is NOT a deriver threshold — it is the
    integrity layer's frozen silence cutoff, and copying that value here would be
    the second source of truth this design exists to avoid. ``value`` reports the
    observed level for the reader; nothing in this function compares it.
    """
    measured = [
        MeasuredDescriptor(
            term="silent",
            value=summary.rms_dbfs,
            metric="rms_dbfs",
            direction="low",
            threshold=None,
        )
    ]
    return DescriptorsBlock(
        measured=measured,
        hybrid=[],
        advisory=[],
        summary=render_summary(measured, [], []),
        library=DescriptorsLibrary(
            thresholds_sha256=thresholds_sha256(),
            deriver_version=DERIVER_VERSION,
        ),
    )


def derive_descriptors(
    summary: DeterministicSummary, *, is_silent: bool = False
) -> DescriptorsBlock:
    """Pure deriver: DeterministicSummary -> DescriptorsBlock (by design).

    ``is_silent`` is the integrity layer's WHOLE-FILE silence verdict
    (``IntegrityBlock.all_channels_silent``), passed in rather than recomputed:
    ``summary.rms_dbfs`` is the mean across channels, so re-deriving silence here
    would create a second, subtly different definition that could drift from the
    integrity layer's. The caller owns the predicate; this function stays pure.

    When it is set, the file has no signal to interpret, so every measured and
    hybrid term is suppressed in favour of the single ``silent`` term. The
    suppressed terms are not merely absent — an empty list would be
    indistinguishable from a deriver that failed to run.
    """
    if is_silent:
        return _silent_block(summary)
    measured = _measured(summary)
    hybrid = _hybrid(summary)
    library = DescriptorsLibrary(
        thresholds_sha256=thresholds_sha256(),
        deriver_version=DERIVER_VERSION,
    )
    return DescriptorsBlock(
        measured=measured,
        hybrid=hybrid,
        advisory=[],
        summary=render_summary(measured, hybrid, []),
        library=library,
    )
