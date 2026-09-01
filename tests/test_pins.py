"""A2: prove the runtime dependency closure is pinned and honored.

Pins are law (plan Section 1.1): the installed version of every deterministic-core
dependency must EXACTLY equal the constant recorded in ``sonoscope.pins`` and the
uv lockfile must be in sync. Drift is a hard fail.
"""

import importlib.metadata as importlib_metadata
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sonoscope.pins import PINNED_VERSIONS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pinned_index_env() -> dict[str, str]:
    """Env that makes ``uv`` resolve against PyPI regardless of local uv config.

    The committed lockfile records ``https://pypi.org/simple`` URLs. A developer
    machine configured against a mirror (``~/.config/uv/uv.toml``) makes
    ``uv lock --check`` report drift for the mirror's URLs alone, so without
    this the test asserts the local uv config rather than the lockfile.
    """
    return {
        **os.environ,
        "UV_NO_CONFIG": "1",
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
    }


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
        env=_pinned_index_env(),
    )
    assert result.returncode == 0, result.stderr


def test_params_sha256_reexported_via_pins():
    from sonoscope.pins import params_sha256 as p
    from sonoscope.features.librosa_features import params_sha256 as d
    assert p() == d()
