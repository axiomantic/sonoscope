"""Pin the frozen deriver thresholds + their digest.

``thresholds_sha256`` mirrors ``features.librosa_features.params_sha256`` exactly
(canonical JSON, sorted keys, compact separators). The digest is a hard pin:
any change to a threshold value drifts the hash and fails ``test_digest_drifts``.
The pinned ``EXPECTED_THRESHOLDS_SHA256`` anchors the CALIBRATION-PENDING
placeholder set — recalibration will intentionally re-pin it.
"""

from sonoscope.descriptors.thresholds import (
    DERIVER_THRESHOLDS,
    DERIVER_VERSION,
    thresholds_sha256,
)

# Real digest of the placeholder set (computed from first RED run).
EXPECTED_THRESHOLDS_SHA256 = (
    "8a30a4cb477803982949d7cb9f4f22a6c5980241c626e6a9d2e2e39325bcd3d3"
)


def test_digest_is_stable_and_exact():
    digest = thresholds_sha256()
    assert len(digest) == 64
    assert digest == EXPECTED_THRESHOLDS_SHA256


def test_digest_drifts_on_change():
    perturbed = dict(DERIVER_THRESHOLDS)
    perturbed["bright.centroid_hz_min"] = (
        perturbed["bright.centroid_hz_min"] + 1.0
    )
    assert thresholds_sha256(perturbed) != thresholds_sha256()


def test_digest_reexported_via_pins():
    from sonoscope.descriptors.thresholds import thresholds_sha256 as d
    from sonoscope.pins import thresholds_sha256 as p

    assert p() == d()


def test_deriver_version_pinned():
    assert DERIVER_VERSION == "1.1.0"
