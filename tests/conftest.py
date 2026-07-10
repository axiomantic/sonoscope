"""Shared pytest fixtures.

Integration-marked tests (Surge XT, spike VST3, Qwen model) install their own
skip guards as those external artifacts are introduced in later tasks (A4, B2,
A5). Task E3 introduces the first Surge-XT-requiring integration tests, so the
``surge_vst3_path`` skip guard lives here (per AGENTS.md: an integration test
skips with an explicit reason string when its external artifact is absent, never
a silent pass).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# A4-confirmed system install location for the pinned Surge XT VST3 bundle.
SURGE_XT_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

# --- MIDI-capture integration skip guards ------------------------------------
# The live MIDI-capture golden/determinism test needs BOTH the
# built C CLAP host (build/clap_midi_host) AND the real ReferenceSequencer.clap. Each guard
# skips with an EXPLICIT reason naming what is missing (AGENTS.md testing
# discipline: an absent integration artifact is a loud skip, never a silent pass).

#: Env override for a non-default ReferenceSequencer.clap location (checked before the
#: spike-documented CLAP install path below).
REFSEQ_CLAP_ENV = "SONOSCOPE_REFSEQ_CLAP"
#: Default install location of the reference sequencer's .clap
#: (bundle id ``com.example.reference-sequencer``).
REFSEQ_CLAP_DEFAULT = (
    Path.home() / "Library" / "Audio" / "Plug-Ins" / "CLAP" / "ReferenceSequencer.clap"
)


def _resolve_clap_binary(bundle: Path) -> Path | None:
    """Resolve a ``.clap`` reference to the dlopen-able Mach-O the C host loads.

    The C host (``tools/clap_midi_host/clap_midi_host.c``) ``dlopen``s the path it
    is given verbatim; a macOS ``.clap`` bundle is a DIRECTORY, so the inner
    ``Contents/MacOS/<stem>`` binary must be resolved by the caller (by design,
    the caller passes the inner binary). A bare-binary ``.clap`` is used
    as-is. Returns ``None`` when no loadable binary is present.
    """
    if bundle.is_dir():
        inner = bundle / "Contents" / "MacOS" / bundle.stem
        return inner if inner.exists() else None
    return bundle if bundle.exists() else None


@pytest.fixture(scope="module")
def clap_midi_host_path() -> Path:
    """Path to the built C CLAP MIDI host, or skip with an explicit reason.

    The host binary is a gitignored build artifact
    (``scripts/build_clap_midi_host.sh``); its absence is a loud skip.
    """
    from sonoscope.backends.midi_capture import DEFAULT_HOST_PATH

    if not DEFAULT_HOST_PATH.exists():
        pytest.skip(
            f"clap_midi_host not built at {DEFAULT_HOST_PATH} "
            "(integration artifact absent; run scripts/build_clap_midi_host.sh)"
        )
    return DEFAULT_HOST_PATH


@pytest.fixture(scope="module")
def refseq_clap_path() -> Path:
    """Path to the real ReferenceSequencer.clap's dlopen-able binary, or skip with a reason.

    Resolution order: ``$SONOSCOPE_REFSEQ_CLAP`` (a ``.clap`` bundle or binary),
    else the spike-documented ``~/Library/Audio/Plug-Ins/CLAP/ReferenceSequencer.clap``. A
    bundle is resolved to its inner ``Contents/MacOS`` binary (the C host dlopens
    the path verbatim). Skips (explicit reason) when the plugin is absent or has
    no loadable binary.
    """
    raw = os.environ.get(REFSEQ_CLAP_ENV)
    bundle = Path(raw) if raw else REFSEQ_CLAP_DEFAULT
    if not bundle.exists():
        pytest.skip(
            f"ReferenceSequencer.clap not found at {bundle} "
            f"(integration artifact absent; set ${REFSEQ_CLAP_ENV} or install to "
            f"{REFSEQ_CLAP_DEFAULT})"
        )
    binary = _resolve_clap_binary(bundle)
    if binary is None:
        pytest.skip(
            f"ReferenceSequencer.clap at {bundle} has no dlopen-able binary "
            f"(expected {bundle}/Contents/MacOS/{bundle.stem}; artifact malformed)"
        )
    return binary


@pytest.fixture
def surge_vst3_path() -> Path:
    """Path to the installed Surge XT VST3, or skip with an explicit reason.

    Integration tests depending on this fixture are deselected on a machine
    without Surge XT (design §12.3, AGENTS.md testing discipline).
    """
    if not SURGE_XT_VST3.exists():
        pytest.skip(
            f"Surge XT not installed at {SURGE_XT_VST3} "
            "(integration artifact absent; run scripts/install_surge_xt.sh)"
        )
    return SURGE_XT_VST3


@pytest.fixture
def qwen_model() -> None:
    """Skip (explicit reason) unless the Qwen2-Audio perception artifact is present.

    G1 integration tests need BOTH the ``sonoscope[perception]`` extra
    (transformers/torch) AND the pinned model weights in the local HF cache. The
    processor is small and loads WITHOUT pulling the ~16 GB weight shards, so it
    is a cheap presence probe: if it resolves at the pinned revision, the full
    snapshot (fetched by ``scripts/fetch_qwen_model.sh``) is present. Skips with
    an explicit reason when either is absent (AGENTS.md: never a silent pass).
    """
    from sonoscope.perception.qwen_local import HF_REPO, MODEL_REVISION

    try:
        from transformers import AutoProcessor
    except Exception as exc:  # noqa: BLE001 - explicit skip reason on absence
        pytest.skip(
            f"perception extra not installed ({exc!r}); "
            "run: uv sync --extra perception"
        )

    try:
        AutoProcessor.from_pretrained(
            HF_REPO, revision=MODEL_REVISION, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - weights absent -> explicit skip
        pytest.skip(
            f"Qwen2-Audio weights not in local cache ({type(exc).__name__}); "
            "run: scripts/fetch_qwen_model.sh"
        )
