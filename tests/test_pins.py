"""A2: prove the runtime dependency closure is pinned and honored.

Pins are law (plan Section 1.1): the installed version of every deterministic-core
dependency must EXACTLY equal the constant recorded in ``sonoscope.pins`` and the
uv lockfile must be in sync. Drift is a hard fail.
"""

import importlib.metadata as importlib_metadata
import shutil
import subprocess
from pathlib import Path

import pytest

from sonoscope.pins import PINNED_VERSIONS

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_versions_match_pins():
    installed = {dist: importlib_metadata.version(dist) for dist in PINNED_VERSIONS}
    assert installed == PINNED_VERSIONS


def test_lockfile_not_drifted():
    if shutil.which("uv") is None:
        pytest.skip("uv binary not on PATH; cannot verify lockfile drift")
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_params_sha256_reexported_via_pins():
    from sonoscope.pins import params_sha256 as p
    from sonoscope.features.librosa_features import params_sha256 as d
    assert p() == d()
