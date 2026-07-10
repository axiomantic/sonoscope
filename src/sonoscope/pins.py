"""Pinned dependency versions: single source of truth for ``doctor`` / CI.

Pins are law (plan Section 1.1): every deterministic-core dependency is pinned to
an EXACT version in ``pyproject.toml`` and captured in ``uv.lock`` with hashes.
These constants mirror those ``==`` pins so ``doctor`` can hard-fail on drift.

The audio-QA pipeline's ground-truth determinism depends on these exact
versions; changing one requires re-measuring determinism floors.

``pydantic`` is intentionally excluded from the exact-match set: it is pinned as
a compatible range (``>=2.7,<3``) in ``pyproject.toml`` because only its v2 API
surface is load-bearing, not a specific patch release.

Digest re-exports: both feature-extraction and interpretation-layer digests are
re-exported here so drift is checkable through the single pins module, consistent
with AGENTS.md "pins are law". ``params_sha256``
(``features.librosa_features.params_sha256``) versions the feature-extraction
params; ``thresholds_sha256`` (``descriptors.thresholds.thresholds_sha256``,
by design) versions the interpretation layer. They are siblings: extraction and
interpretation each carry an independent digest.
"""

from sonoscope.descriptors.thresholds import thresholds_sha256 as thresholds_sha256
from sonoscope.features.librosa_features import params_sha256 as params_sha256

PEDALBOARD_VERSION = "0.9.23"
LIBROSA_VERSION = "0.10.2"
MIDO_VERSION = "1.3.2"
NUMPY_VERSION = "1.26.4"
SOUNDFILE_VERSION = "0.12.1"
SOXR_VERSION = "1.1.0"

# Distribution name -> pinned version. Keys are PyPI distribution names as used
# by importlib.metadata.version(); doctor iterates this map to detect drift.
PINNED_VERSIONS = {
    "pedalboard": PEDALBOARD_VERSION,
    "librosa": LIBROSA_VERSION,
    "mido": MIDO_VERSION,
    "numpy": NUMPY_VERSION,
    "soundfile": SOUNDFILE_VERSION,
    "soxr": SOXR_VERSION,
}
