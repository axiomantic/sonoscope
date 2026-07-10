"""Float-tolerance comparator tests (Task D1, I3).

RED + GREEN for the shared reproducibility comparator. The abs+rel rule
``|a - b| <= ABS_TOL + REL_TOL * |b|`` governs bit-repro / reproducibility
checks only (never ``iterate`` significance). ``test_just_outside_tolerance_fails``
is the RED-proving case: it demonstrates the comparator actually *rejects* a
pair placed just past the bound (a comparator that always returned True would
pass every other test but fail this one).
"""

from sonoscope.features import ABS_TOL as PKG_ABS_TOL
from sonoscope.features import REL_TOL as PKG_REL_TOL
from sonoscope.features import close as pkg_close
from sonoscope.features.tolerance import ABS_TOL, REL_TOL, close


def test_constants_exact():
    # Exact-equality on the frozen constants (I3): 1e-4 / 1e-9.
    assert REL_TOL == 1e-4
    assert ABS_TOL == 1e-9


def test_within_tolerance_passes():
    # A pair strictly inside the bound must compare close (GREEN).
    b = 1000.0
    bound = ABS_TOL + REL_TOL * abs(b)
    a = b + bound * 0.5
    assert close(a, b) is True


def test_just_outside_tolerance_fails():
    # A pair placed just past the bound must be rejected (RED-proving:
    # this is the only test that fails if `close` always returns True).
    b = 1000.0
    bound = ABS_TOL + REL_TOL * abs(b)
    a = b + bound * (1.0 + 1e-6)
    assert close(a, b) is False


def test_close_on_equal_values():
    # Identical values are trivially close in both orders.
    assert close(3.14, 3.14) is True
    assert close(0.0, 0.0) is True


def test_package_reexports_tolerance():
    # D1 owns __init__ exports: close / REL_TOL / ABS_TOL must be re-exported
    # (D2/D3 and F2 import these from the package surface).
    assert PKG_REL_TOL == 1e-4
    assert PKG_ABS_TOL == 1e-9
    assert pkg_close(1.0, 1.0) is True
