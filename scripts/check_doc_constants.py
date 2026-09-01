"""Lint the code-derived constants embedded in README.md / docs/using-sonoscope.md.

Not a pytest: documentation consistency is a lint concern. This checks only values
the code OWNS (schema/deriver versions, the pin count, threshold hashes) — never
prose and never plugin-render floats, which are not reproducible run-to-run
(a Surge XT render moved `spectral_centroid_hz` by ~1.7 Hz across six renders).

Exit 0 = every occurrence matches the running code. Exit 1 = drift, listed by
file:line with expected-vs-found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from sonoscope.descriptors.midi_deriver import (
    MIDI_DERIVER_THRESHOLDS,
    MIDI_DERIVER_VERSION,
)
from sonoscope.descriptors.thresholds import (
    DERIVER_THRESHOLDS,
    DERIVER_VERSION,
    thresholds_sha256,
)
from sonoscope.pins import PINNED_VERSIONS
from sonoscope.schema.models import SCHEMA_VERSION

DOCS = ("README.md", "docs/using-sonoscope.md")


def _checks() -> list[tuple[str, re.Pattern[str], set[str]]]:
    """(label, pattern capturing the value, set of values the code permits)."""
    return [
        (
            "schema_version",
            re.compile(r'"schema_version":\s*"([^"]+)"'),
            {SCHEMA_VERSION},
        ),
        (
            "pin count",
            re.compile(r"(\d+) pinned dependencies match"),
            {str(len(PINNED_VERSIONS))},
        ),
        (
            "deriver_version",
            re.compile(r'"deriver_version":\s*"([^"]+)"'),
            {DERIVER_VERSION, MIDI_DERIVER_VERSION},
        ),
        (
            "thresholds_sha256",
            re.compile(r'"thresholds_sha256":\s*"([0-9a-f]+)"'),
            {
                thresholds_sha256(DERIVER_THRESHOLDS),
                thresholds_sha256(MIDI_DERIVER_THRESHOLDS),
            },
        ),
    ]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    checks = _checks()
    drift: list[str] = []
    checked = 0

    for rel in DOCS:
        path = root / rel
        if not path.is_file():
            drift.append(f"{rel}: missing")
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for label, pattern, allowed in checks:
                for found in pattern.findall(line):
                    checked += 1
                    if found not in allowed:
                        drift.append(
                            f"{rel}:{lineno}: {label} is {found!r}, "
                            f"code says {sorted(allowed)!r}"
                        )

    for line in drift:
        print(f"DRIFT {line}")
    print(f"checked {checked} constant occurrences across {len(DOCS)} docs")
    if drift:
        print(f"FAIL: {len(drift)} drifted")
        return 1
    print("OK: no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
