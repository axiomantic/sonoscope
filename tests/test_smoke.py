"""Scaffold smoke test: the package imports and exposes its version."""

import sonoscope


def test_version() -> None:
    assert sonoscope.__version__ == "0.1.0"
