"""Float-comparison tolerance for reproducibility checks (by design, I3).

The shared abs+rel comparator used by cross-run / cross-feature reproducibility
comparisons (bit-repro regime). It is deliberately narrow in
scope: it governs *reproducibility* only. The *significance* of an ``iterate``
delta is judged separately by the R2 noise floor and MUST NOT use
these constants.
"""

from __future__ import annotations

# Reproducibility tolerance constants (I3). Frozen and applied uniformly to all
# scalar features. Exact values are contract-relevant (tested for equality).
REL_TOL: float = 1e-4
ABS_TOL: float = 1e-9


def close(a: float, b: float) -> bool:
    """Return True iff ``a`` and ``b`` match under the abs+rel rule (I3).

    Implements ``|a - b| <= ABS_TOL + REL_TOL * |b|``. ``b`` is the reference
    operand for the relative term (asymmetric by construction, matching the
    documented rule).
    """
    return abs(a - b) <= ABS_TOL + REL_TOL * abs(b)
