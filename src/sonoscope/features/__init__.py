"""Deterministic ground-truth feature layer (Librosa).

D1 owns this package's public export surface (M1). D2 (integrity) and D3
(tripwires) import by module path and must NOT edit this file; the names they
and downstream reproducibility checks (F2) need are re-exported here:

- ``compute_summary`` / ``SummaryResult`` / ``params_sha256`` / ``FROZEN_PARAMS``
  — the frozen-param summary computation (by design).
- ``close`` / ``REL_TOL`` / ``ABS_TOL`` — the shared reproducibility comparator
  (I3), re-exported from :mod:`sonoscope.features.tolerance`.
"""

from sonoscope.features.librosa_features import (
    FROZEN_PARAMS,
    NOTE_TEMPO_IMPLAUSIBLE,
    NOTE_TEMPO_LOW_ONSETS,
    SummaryResult,
    compute_summary,
    params_sha256,
)
from sonoscope.features.tolerance import ABS_TOL, REL_TOL, close

__all__ = [
    "compute_summary",
    "SummaryResult",
    "params_sha256",
    "FROZEN_PARAMS",
    "NOTE_TEMPO_LOW_ONSETS",
    "NOTE_TEMPO_IMPLAUSIBLE",
    "close",
    "REL_TOL",
    "ABS_TOL",
]
