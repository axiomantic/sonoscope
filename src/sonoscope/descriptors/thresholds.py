"""Frozen deriver thresholds + their digest (by design).

Mirrors ``features/librosa_features.py``'s ``FROZEN_PARAMS``/``params_sha256``
**exactly** — a flat ``dict[str, int | float | str | bool]`` with dotted,
namespaced keys (JSON-stable, change-detecting) and a module-level free function
using the identical canonical serialization recipe.

Separation from ``params_sha256`` (design Q7): this digest is a deliberate
**sibling**, not folded into ``params_sha256``. ``params_sha256`` versions the
*feature-extraction* params; ``thresholds_sha256`` versions the *interpretation*
thresholds. They break independently — a threshold recalibration must not
masquerade as a feature-extraction change, and vice versa.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

DERIVER_VERSION = "1.0.0"

# Flat, dotted, JSON-serialisable leaves only (int|float|str|bool) so the
# digest is stable and changes iff a value changes. CALIBRATION-PENDING (C1.5).
DERIVER_THRESHOLDS: dict[str, int | float | str | bool] = {
    # gated single-metric
    "bright.centroid_hz_min": 2500.0,
    "dark.centroid_hz_max": 800.0,
    "loud.rms_dbfs_min": -18.0,
    "quiet.rms_dbfs_max": -35.0,
    "compressed.crest_db_max": 6.0,
    "dynamic.crest_db_min": 15.0,
    "busy.onset_hz_min": 8.0,
    "spare.onset_hz_max": 2.0,
    # gated multi-metric (AND)
    "dense.onset_hz_min": 8.0,
    "dense.bandwidth_hz_min": 2000.0,
    # readout gate params
    "tempo.min_onsets": 4,
    "tempo.bpm_min": 40.0,
    "tempo.bpm_max": 300.0,
    # hybrid composites — norms + weights + fire thresholds
    "driving.fire": 0.6,
    "driving.w_onset": 0.5, "driving.w_tempo": 0.3, "driving.w_rms": 0.2,
    "driving.onset_lo": 0.0, "driving.onset_hi": 12.0,
    "driving.tempo_lo": 60.0, "driving.tempo_hi": 180.0,
    "driving.rms_lo": -40.0, "driving.rms_hi": -6.0,
    "punchy.fire": 0.6,
    "punchy.w_crest": 0.6, "punchy.w_onset": 0.4,
    "punchy.crest_lo": 4.0, "punchy.crest_hi": 18.0,
    "punchy.onset_lo": 0.0, "punchy.onset_hi": 12.0,
    "warm.fire": 0.6,
    "warm.w_centroid": 0.6, "warm.w_mfcc": 0.4,
    "warm.centroid_lo": 500.0, "warm.centroid_hi": 4000.0,
    "warm.mfcc_lo_idx": 1, "warm.mfcc_hi_idx": 5,
    "warm.mfcc_lo": -200.0, "warm.mfcc_hi": 200.0,
}


def thresholds_sha256(params: Mapping[str, object] = DERIVER_THRESHOLDS) -> str:
    """SHA-256 of the frozen threshold set.

    Serialised canonically (sorted keys, compact separators) so the digest is
    stable across runs and changes iff a threshold value changes. Mirrors
    ``features.librosa_features.params_sha256``.
    """
    payload = json.dumps(
        dict(params), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
