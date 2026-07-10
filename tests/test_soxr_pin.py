"""Pin-guard for soxr (D3): the resample stage makes soxr a DIRECT, load-bearing
dependency, so its version is pinned and asserted here. This test also verifies the
``soxr.__version__`` provenance API (design §14.1) that feeds
``input_provenance.soxr_version`` — the stamp is proven, not assumed.
"""

import importlib.metadata as importlib_metadata

import soxr

from sonoscope.pins import PINNED_VERSIONS, SOXR_VERSION


def test_soxr_pinned_to_1_1_0():
    assert SOXR_VERSION == "1.1.0"
    assert PINNED_VERSIONS["soxr"] == "1.1.0"


def test_installed_soxr_matches_pin():
    assert importlib_metadata.version("soxr") == SOXR_VERSION


def test_soxr_dunder_version_provenance_api():
    # The exact API read that feeds input_provenance.soxr_version (§9/§10.2).
    assert soxr.__version__ == "1.1.0"
