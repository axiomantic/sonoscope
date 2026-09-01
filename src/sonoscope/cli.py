"""sonoscope command-line entry point.

C5 skeleton: argument parsing + dispatch + the design 3.6 exit-code / fatal
envelope contract. Command engines are NotImplementedError stubs here; the
real handlers are wired in H1 (per the plan's CLI-wiring boundary, I6).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional, get_args

from pydantic import ValidationError

from sonoscope import __version__, doctor, render_orchestrator
from sonoscope.analysis_orchestrator import analyze_plugin_spec
from sonoscope.backends.midi_capture import MidiCaptureRequest
from sonoscope.backends.pedalboard_vst3 import PedalboardVST3Backend, binary_sha256
from sonoscope.corpus import list_items
from sonoscope.corpus import verify as corpus_verify
from sonoscope.determinism import (
    DEFAULT_REPEATS,
    measure_floors,
    read_floors,
    write_floors,
)
from sonoscope.errors import (
    InputError,
    SonoscopeEnvironmentError,
    SonoscopeError,
    UsageError,
)
from sonoscope.features.descriptor_gate import (
    ExpectedDescriptors,
    evaluate_descriptors,
    load_expected_descriptors,
)
from sonoscope.features.midi_tripwires import Offvel0Policy
from sonoscope.iterate import (
    DescriptorTermDiff,
    diff_descriptor_terms,
    run_iterate,
)
from sonoscope.midi_input import SliceUnit
from sonoscope.midi_orchestrator import MidiFileSource, MidiSlice, analyze_midi
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.perception.qwen_local import QwenLocalAdapter
from sonoscope.probe import (
    ProbeFixturesMissing,
    ProbeUnavailable,
    default_fixture_pairs,
    run_probe,
)
from sonoscope.schema import (
    Component,
    ExitCode,
    FatalError,
    FatalErrorDetail,
    IterateDirection,
)
from sonoscope.schema.generate import SCHEMA_KINDS, json_schema_for
from sonoscope.schema.models import (
    AnalysisReport,
    DescriptorGateResult,
    WavAnalysisReport,
)
from sonoscope.spec import Spec
from sonoscope.wav_orchestrator import (
    AudioSlice,
    AudioSliceUnit,
    WavFileSource,
    aggregate_gate,
    analyze_wav,
)

# Repo root (src/sonoscope/cli.py -> parents[2]) for repo-relative defaults.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default location of the probe A/B fixture set (by design). Points at the
# committed fixture set (``corpus/qwen_probe/``, deterministic Surge renders) so
# ``probe`` works out of the box. Overridable via
# ``--fixtures``; the probe checks adapter health BEFORE reading fixtures, so a
# missing model degrades gracefully even when this directory is absent, and an
# available model with absent fixtures raises the typed PROBE_FIXTURES_NOT_FOUND
# input error (never a raw FileNotFoundError crash).
DEFAULT_PROBE_FIXTURES = _REPO_ROOT / "corpus" / "qwen_probe"

# analyze --wav (standalone wav analysis) cross-flag typed input errors (exit 2).
# The wav-path slice/chunk flags are valid ONLY with --wav; the plugin-render
# flags are valid ONLY with --plugin. A flag supplied for the WRONG source is a
# typed InputError (mirrors the analyze-midi cross-source rigor) — never a
# silently discarded value. An incoherent slice (--length/--unit without
# --offset) is likewise a typed InputError.
INPUT_WAV_FLAG_CONFLICT = "INPUT_WAV_FLAG_CONFLICT"
INPUT_ANALYZE_FLAG_CONFLICT = "INPUT_ANALYZE_FLAG_CONFLICT"
INPUT_WAV_SLICE_INCOHERENT = "INPUT_WAV_SLICE_INCOHERENT"

# Exit code for internal/unexpected failures not covered by the design 3.6
# table (which defines only 0..5). Uses the conventional "general error" code.
_GENERIC_EXIT = 1

# analyze --expect-descriptors gates report.descriptors. A spec supplied against a
# report that carries NO descriptors block is a hard INPUT error (exit 2): the
# operator asked to gate a block that does not exist, so we fail loud rather than
# silently PASS. Descriptors are an analysis-domain artifact -> component "analyze".
DESCRIPTORS_NO_BLOCK = "DESCRIPTORS_NO_BLOCK"

# iterate-descriptors loads two AnalysisReport JSONs from disk. A path that is
# unreadable, non-JSON, or valid-JSON-but-not-an-AnalysisReport is a hard INPUT
# error (exit 2): the operator pointed the term diff at a file it cannot use, so
# we fail loud (never a fabricated/empty diff). The ``side`` key names which of the
# two report inputs failed. Descriptors are an analysis-domain artifact -> "analyze".
DESCRIPTORS_REPORT_INVALID = "DESCRIPTORS_REPORT_INVALID"

# iterate-descriptors accepts BOTH report shapes: the plugin-path single
# ``AnalysisReport`` (a JSON object) and the wav-path ``WavAnalysisReport`` (a JSON
# ARRAY of per-chunk analyses). Two arrays are diffed CHUNK-WISE (chunk i vs chunk i),
# which needs equal chunk counts; unequal counts would force a silent truncation to
# the shorter list, so they are a hard INPUT error (exit 2) naming BOTH counts. A
# MIXED pair (one array, one object) is not comparable at all — a single report has no
# chunk axis to align against — so it is its own distinct INPUT error rather than a
# guess. Both are analysis-domain -> component "analyze".
DESCRIPTORS_CHUNK_COUNT_MISMATCH = "DESCRIPTORS_CHUNK_COUNT_MISMATCH"
DESCRIPTORS_REPORT_SHAPE_MISMATCH = "DESCRIPTORS_REPORT_SHAPE_MISMATCH"

# --- analyze-midi (C2) typed input errors ------------------------------------
# All are INPUT-contract violations (component "midi") -> InputError, exit 2.
# Per eek's C2 spec the source (mutual-exclusion) and file/spec-completeness
# checks are typed InputErrors (exit 2), NOT argparse USAGE errors (exit 1): a
# missing/conflicting/incomplete SOURCE is user INPUT, mapped at the CLI seam
# before any capture — never a bare argparse exit-1 or an escaping crash.
MIDI_SOURCE_MISSING_CODE = "INPUT_MIDI_SOURCE_MISSING"
MIDI_SOURCE_CONFLICT_CODE = "INPUT_MIDI_SOURCE_CONFLICT"
# C2 hardening (fix 1): a flag that belongs to the OTHER source (e.g. --sample-rate
# with --plugin, or --spec with --file) is a source-mismatched flag. It used to be
# silently discarded; it is now a typed InputError, mirroring the both/neither
# source rigor (conflicting input is a typed error, never silent).
MIDI_FLAG_CONFLICT_CODE = "INPUT_MIDI_FLAG_CONFLICT"
MIDI_SPEC_MISSING_CODE = "INPUT_MIDI_SPEC_MISSING"
MIDI_SPEC_UNREADABLE_CODE = "INPUT_MIDI_SPEC_UNREADABLE"
MIDI_SPEC_INVALID_CODE = "INPUT_MIDI_SPEC_INVALID"
MIDI_FILE_SAMPLE_RATE_MISSING_CODE = "INPUT_MIDI_FILE_SAMPLE_RATE_MISSING"
MIDI_SLICE_INCOHERENT_CODE = "INPUT_MIDI_SLICE_INCOHERENT"

# The transport/render fields a --plugin capture spec MUST carry (they map 1:1 to
# the non-default MidiCaptureRequest fields; plugin_path comes from --plugin and
# plugin_id/playing/state_b64 are optional). A missing field is a hard InputError
# (exit 2), never a silently-defaulted transport.
_MIDI_SPEC_REQUIRED_FIELDS: tuple[str, ...] = (
    "tempo_bpm",
    "start_position_beats",
    "duration_beats",
    "tsig_num",
    "tsig_den",
    "sample_rate",
    "block_size",
)


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises ``UsageError`` instead of ``SystemExit`` so
    bad flags / unknown commands map to the design 3.6 USAGE exit code (1)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError("USAGE_INVALID_ARGS", message, component="cli")


def _now_iso() -> str:
    """UTC timestamp in the design's ``...Z`` form (monkeypatched in tests)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _global_parser() -> argparse.ArgumentParser:
    """Parent parser of the flag that applies to every command.

    Item D (dogfood LLM-ergonomics): every OTHER flag used to live here and so
    appeared on ALL subcommands via ``parents=[g]`` — including ones whose handler
    never read it (``doctor --perception``, ``corpus --adapter``, ``schema --seed``
    parsed but did nothing, misleading a tool-discovering LLM). Each such flag now
    lives ONLY on the subparser(s) whose handler actually consumes it. ``--json``
    stays global (the structured-output convention is command-agnostic; ``doctor``
    is the current consumer).
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the structured JSON report to stdout (consumed by 'doctor').",
    )
    return parent


def _add_render_spec_flags(p: argparse.ArgumentParser) -> None:
    """Add the render-spec override flags (item D) to a render-family subparser.

    Scoped to the commands that build a render spec/backend — render / analyze /
    iterate / determinism — per item D's mapping.

    Each flag, when supplied, OVERRIDES the corresponding ``spec.render.<field>``
    when the render request is built (the precedence is applied in a single place,
    :func:`sonoscope.render_orchestrator.render`); omitting a flag falls back to the
    spec value (unchanged behavior). The handlers forward ``args.sample_rate`` /
    ``args.block_size`` / ``args.channels`` / ``args.seed`` into that render seam.
    """
    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override the render sample rate in Hz (default: the spec's render.sample_rate_hz).",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the render block size in frames (default: the spec's render.block_size).",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Override the render channel count (default: the spec's render.channels).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the deterministic render seed (default: the spec's render.seed).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full command surface (design 4.5).

    Item A (dogfood): every subcommand + flag carries a ``help=``/``description=`` and
    a usage ``epilog`` example so an LLM discovering the tool via ``--help`` learns what
    each command does and how to invoke it. Item B: a top-level ``--version``. Item D:
    flags are scoped to the subcommands whose handler consumes them (see
    ``_global_parser`` / ``_add_render_spec_flags``).
    """
    g = _global_parser()
    parser = _ArgumentParser(
        prog="sonoscope",
        description=(
            "Deterministic audio-QA harness: render a built plugin to a wav and run "
            "versioned machine-listening analysis (tripwires, feature deltas, "
            "determinism floors) for regression/CI QA."
        ),
    )
    # Item B (dogfood): a top-level --version so an LLM can discover the installed
    # version without an integration run. argparse's 'version' action prints to stdout
    # and raises SystemExit(0), short-circuiting the required-subcommand check.
    parser.add_argument(
        "--version",
        action="version",
        version=f"sonoscope {__version__}",
        help="Print the sonoscope version and exit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser(
        "render",
        parents=[g],
        help="Render a plugin+spec to a wav and print the render metadata.",
        description=(
            "Resolve the spec against the plugin, render it in a crash-isolated "
            "subprocess, and print the render metadata JSON. With --out the wav is "
            "written to that path."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope render --plugin 'Surge XT.vst3' --spec spec.json --out out.wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_render.add_argument(
        "--plugin", default=None, help="Path to the plugin bundle (VST3) to render."
    )
    p_render.add_argument(
        "--spec",
        default=None,
        help="Path to the JSON render spec (param-set + stimulus).",
    )
    p_render.add_argument(
        "--out",
        default=None,
        help=(
            "Write the rendered wav here (a directory or trailing-'/' path receives "
            "'<name>.wav')."
        ),
    )
    _add_render_spec_flags(p_render)

    p_analyze = sub.add_parser(
        "analyze",
        parents=[g],
        help="Render a plugin+spec and emit the deterministic analysis report.",
        description=(
            "Render the plugin+spec, run the deterministic librosa ground-truth "
            "analysis, and print the versioned analysis report JSON. Perception is "
            "opt-in via --perception."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope analyze --plugin 'Surge XT.vst3' --spec spec.json --perception"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p_analyze.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--wav",
        default=None,
        help=(
            "Analyze a standalone wav at its native rate: slice in native units, "
            "resample the analyzed slice to 48 kHz (soxr_hq), and emit an array of "
            "independently-analyzed chunks (use --offset/--length/--unit/"
            "--max-chunk-seconds to window/chunk)."
        ),
    )
    src.add_argument(
        "--plugin",
        default=None,
        help="Path to the plugin bundle (VST3) to render and analyze.",
    )
    p_analyze.add_argument(
        "--spec",
        default=None,
        help="Path to the JSON render spec (param-set + stimulus).",
    )
    p_analyze.add_argument(
        "--perception",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the advisory perception (LALM) pass (default: off; "
            "--no-perception forces off)."
        ),
    )
    p_analyze.add_argument(
        "--adapter",
        default=None,
        help=(
            "Perception adapter selector ('null' forces the no-op adapter; "
            "default: local Qwen)."
        ),
    )
    p_analyze.add_argument(
        "--expect-descriptors",
        default=None,
        help="Path to a descriptor expectation spec JSON; gates report.descriptors.",
    )
    p_analyze.add_argument(
        "--fail-on-red",
        action="store_true",
        default=False,
        help="Exit 4 (ANALYSIS) if the descriptor verdict is RED (requires --expect-descriptors).",
    )
    _add_render_spec_flags(p_analyze)
    # --wav-only native slice/chunk flags (by design). Rejected with --plugin as a
    # typed INPUT_ANALYZE_FLAG_CONFLICT; the render-spec flags above are rejected
    # with --wav as INPUT_WAV_FLAG_CONFLICT.
    p_analyze.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Wav analysis-window slice start in --unit (enables the slice; --wav only).",
    )
    p_analyze.add_argument(
        "--length",
        type=float,
        default=None,
        help="Wav slice length in --unit (omit -> to end of file; --wav only).",
    )
    # DRY: source the unit choices from the AudioSliceUnit literal so the CLI can
    # never drift from _resolve_region's accepted units (mirrors analyze-midi --unit).
    p_analyze.add_argument(
        "--unit",
        choices=get_args(AudioSliceUnit),
        default=None,
        help="Wav slice unit: %(choices)s (default: samples; --wav only).",
    )
    p_analyze.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help="Override the frozen auto-chunk threshold in native seconds (--wav only).",
    )

    # C2 (by design): analyze-midi accepts EITHER a --plugin CLAP
    # capture (transport/render from --spec) OR a standalone --file .mid. The two
    # sources are NOT an argparse mutually-exclusive group: the input-source spec
    # maps both/neither to a typed InputError (exit 2), so both flags are plain
    # optionals validated in the handler (an argparse mutex group would emit exit 1).
    p_analyze_midi = sub.add_parser(
        "analyze-midi",
        parents=[g],
        help="Capture/load a MIDI stream and emit the deterministic MIDI-analysis report.",
        description=(
            "Analyze a MIDI event stream from EITHER a CLAP note-effect plugin capture "
            "(--plugin + --spec, via the C host) OR a standalone .mid file (--file). "
            "Runs the deterministic MIDI tripwires (the stuck-note firewall, plus the "
            "expected-vs-actual diff when --expected is given) and prints the versioned "
            "midi-analysis report JSON. --fail-on-red gates CI on a RED verdict."
        ),
        epilog=(
            "Examples:\n"
            "  # Capture a CLAP note-effect plugin against a golden and gate CI on RED:\n"
            "  sonoscope analyze-midi --plugin 'ReferenceSequencer.clap' --spec capture.json \\\n"
            "    --expected golden.json --fail-on-red\n"
            "  # Analyze a standalone .mid, windowing beats [1, 3):\n"
            "  sonoscope analyze-midi --file phrase.mid --sample-rate 48000 \\\n"
            "    --offset 1 --length 2 --unit beats"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_analyze_midi.add_argument(
        "--plugin",
        default=None,
        help="Path to the CLAP note-effect plugin (.clap) to capture (with --spec).",
    )
    p_analyze_midi.add_argument(
        "--file",
        default=None,
        help="Path to a standalone .mid file to analyze (with --sample-rate).",
    )
    p_analyze_midi.add_argument(
        "--spec",
        default=None,
        help=(
            "Path to the JSON capture spec (tempo_bpm, start_position_beats, "
            "duration_beats, tsig_num, tsig_den, sample_rate, block_size) — required "
            "with --plugin."
        ),
    )
    p_analyze_midi.add_argument(
        "--plugin-id",
        default=None,
        help="CLAP plugin id to select within a multi-plugin .clap bundle (--plugin only).",
    )
    p_analyze_midi.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate in Hz for the --file source (required with --file).",
    )
    p_analyze_midi.add_argument(
        "--tempo",
        type=float,
        default=None,
        help=(
            "Tempo (BPM) override for the --file source (optional; default: the file's "
            "own tempo, or 120)."
        ),
    )
    p_analyze_midi.add_argument(
        "--expected",
        default=None,
        help="Path to an expected-event golden (JSON list) to diff against (optional).",
    )
    p_analyze_midi.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Analysis-window slice start, expressed in --unit; enables the slice.",
    )
    p_analyze_midi.add_argument(
        "--length",
        type=float,
        default=None,
        help="Analysis-window slice length in --unit (omit to slice to the end of the stream).",
    )
    # DRY: source the slice-unit choices from the E2 SliceUnit literal so the CLI can
    # never drift from apply_slice's accepted units (mirrors iterate --direction).
    p_analyze_midi.add_argument(
        "--unit",
        choices=get_args(SliceUnit),
        default=None,
        help="Slice unit for --offset/--length: %(choices)s (default: samples).",
    )
    # DRY: source the offvel0 choices from the E1 Offvel0Policy literal.
    p_analyze_midi.add_argument(
        "--offvel0",
        choices=get_args(Offvel0Policy),
        default=None,
        help=(
            "Override the note_on-velocity-0 policy (default: 'red' for --plugin, "
            "'normalize' for --file)."
        ),
    )
    p_analyze_midi.add_argument(
        "--fail-on-red",
        action="store_true",
        default=False,
        help=(
            "Exit with the ANALYSIS code (4) when the report verdict is RED (the report "
            "still prints to stdout; without this flag RED prints and exits 0)."
        ),
    )

    p_iterate = sub.add_parser(
        "iterate",
        parents=[g],
        help="Compare a baseline vs candidate spec on one metric (thresholded delta).",
        description=(
            "Analyze a baseline and a candidate spec (deterministic-only), then emit "
            "the significance-gated metric delta + verdict against the measured "
            "determinism floor."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope iterate --plugin 'Surge XT.vst3' --baseline base.json \\\n"
            "    --candidate cand.json \\\n"
            "    --metric deterministic.summary.spectral_centroid_hz "
            "--direction decrease --brief"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_iterate.add_argument(
        "--baseline", default=None, help="Path to the baseline JSON spec."
    )
    p_iterate.add_argument(
        "--candidate", default=None, help="Path to the candidate JSON spec."
    )
    p_iterate.add_argument(
        "--metric",
        default=None,
        help=(
            "Dotted metric path to compare "
            "(e.g. deterministic.summary.spectral_centroid_hz)."
        ),
    )
    # FIX 1 (dogfood): source the ``--direction`` choices from the
    # ``IterateDirection`` schema literal (single source of truth) so the CLI can
    # never drift from the engine's verdict logic again. The literal includes
    # ``stable`` (an inverted "did NOT move" assertion, iterate.py); hardcoding a
    # subset here silently made ``--direction stable`` an argparse USAGE error even
    # though the engine fully supports it.
    p_iterate.add_argument(
        "--direction",
        choices=get_args(IterateDirection),
        default=None,
        help="Expected metric movement: %(choices)s.",
    )
    p_iterate.add_argument(
        "--min-effect",
        type=float,
        default=None,
        help=(
            "Minimum effect size (metric units) the change must clear on top of the "
            "noise floor."
        ),
    )
    p_iterate.add_argument(
        "--plugin",
        default=None,
        help="Path to the plugin bundle (VST3) to render both specs with.",
    )
    # Item C (dogfood): --brief emits a display-only reduced object (verdict + delta +
    # expectation) instead of the full IterateDelta that embeds both ~12KB reports.
    p_iterate.add_argument(
        "--brief",
        action="store_true",
        default=False,
        help=(
            "Emit only verdict+delta+expectation (display-only reduced form; omits the "
            "embedded baseline/candidate reports — NOT the canonical iterate-delta "
            "schema)."
        ),
    )
    _add_render_spec_flags(p_iterate)

    # iterate-descriptors (A5): a SEPARATE subcommand from the numeric ``iterate``.
    # It consumes two already-produced AnalysisReport JSONs (run ``analyze`` twice ->
    # two report files) and emits the descriptor-term regression diff, so it needs no
    # plugin/render seam of its own.
    p_iter_desc = sub.add_parser(
        "iterate-descriptors",
        parents=[g],
        help="Diff the descriptor terms of two analysis-report JSONs (term regression).",
        description=(
            "Compare the gate-eligible measured descriptor terms of a baseline and a "
            "candidate analysis report (each produced by a separate 'analyze' run and "
            "saved to a JSON file). Emits a single-line JSON object: the regression set "
            "(added/removed/direction_changed terms) plus a tolerance-banded value-drift "
            "advisory sub-list. Added/removed/direction changes are the regression signal; "
            "raw value drift is expected across renders and is banded by --value-tolerance. "
            "Wav-path reports ('analyze --wav' emits a JSON array of per-chunk analyses) "
            "are compared CHUNK-WISE: chunk i vs chunk i, emitting "
            "{\"chunk_count\":N,\"chunks\":[...]}. The chunk counts must match, and both "
            "reports must come from the same analyze path; either mismatch is a hard "
            "input error (exit 2), never a partial comparison."
        ),
        epilog=(
            "Example (two-invocation workflow):\n"
            "  sonoscope analyze --plugin 'Surge XT.vst3' --spec base.json > baseline.json\n"
            "  sonoscope analyze --plugin 'Surge XT.vst3' --spec cand.json > candidate.json\n"
            "  sonoscope iterate-descriptors --baseline baseline.json \\\n"
            "    --candidate candidate.json --value-tolerance 0.5\n"
            "\n"
            "Example (wav path, compared chunk-wise):\n"
            "  sonoscope analyze --wav before.wav > baseline.json\n"
            "  sonoscope analyze --wav after.wav > candidate.json\n"
            "  sonoscope iterate-descriptors --baseline baseline.json \\\n"
            "    --candidate candidate.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_iter_desc.add_argument(
        "--baseline", default=None, help="Path to the baseline analysis-report JSON."
    )
    p_iter_desc.add_argument(
        "--candidate", default=None, help="Path to the candidate analysis-report JSON."
    )
    p_iter_desc.add_argument(
        "--value-tolerance",
        type=float,
        default=0.0,
        help=(
            "Absolute value delta a both-present term must exceed to be reported in the "
            "value_drift advisory list (default 0.0; does not affect the regression set)."
        ),
    )

    p_det = sub.add_parser(
        "determinism",
        parents=[g],
        help="Render N times and measure the per-feature nondeterminism floors.",
        description=(
            "Render the spec --repeats times and derive the per-feature determinism "
            "floors (the noise thresholds), persist them to the floor cache, and print "
            "them."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope determinism --plugin 'Surge XT.vst3' --spec spec.json --repeats 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_det.add_argument(
        "--plugin", default=None, help="Path to the plugin bundle (VST3) to render."
    )
    p_det.add_argument(
        "--spec",
        default=None,
        help="Path to the JSON render spec (param-set + stimulus).",
    )
    p_det.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of renders to measure the floor over (must be >= 2; default: 5).",
    )
    p_det.add_argument(
        "--method",
        choices=("range", "variance"),
        default="range",
        help="Floor statistic: 'range' (max-min) or 'variance'. Default: range.",
    )
    _add_render_spec_flags(p_det)

    p_probe = sub.add_parser(
        "probe",
        parents=[g],
        help="Run the R6 Qwen A/B perception feasibility gate.",
        description=(
            "Run the perception adapter over the committed A/B fixture set and print "
            "the discrimination verdict. A KNOWN-absent model degrades to "
            "status:'unavailable' (exit 0), never a fake FAIL."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope probe --fixtures corpus/qwen_probe"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_probe.add_argument(
        "--fixtures",
        default=None,
        help="Directory of A/B probe fixture wavs (default: the committed B3 set).",
    )
    p_probe.add_argument(
        "--adapter",
        default=None,
        help=(
            "Perception adapter selector ('null' forces the no-op adapter; "
            "default: local Qwen)."
        ),
    )

    p_schema = sub.add_parser(
        "schema",
        parents=[g],
        help="Emit the draft-2020-12 JSON Schema for a report kind.",
        description=(
            "Print (or write with --out) the draft-2020-12 JSON Schema generated from "
            "the Pydantic models for the requested report kind."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope schema --kind analysis --out schemas/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_schema.add_argument(
        "--kind",
        choices=SCHEMA_KINDS,
        default="analysis",
        help="Report kind to emit the schema for (default: analysis).",
    )
    p_schema.add_argument(
        "--out",
        default=None,
        help=(
            "Write the schema here (a directory or trailing-'/' path receives "
            "'<kind>.schema.json')."
        ),
    )

    p_corpus = sub.add_parser(
        "corpus",
        parents=[g],
        help="List corpus items or verify their pinned checksums.",
        description=(
            "List the pinned corpus items, or verify each against its pinned sha256 "
            "(a missing/hash-drifted item is a hard error — pins are law)."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope corpus verify"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_corpus.add_argument(
        "action",
        choices=("list", "verify"),
        help="'list' the corpus items or 'verify' their pinned checksums.",
    )

    sub.add_parser(
        "doctor",
        parents=[g],
        help="Run environment checks + the latency benchmark.",
        description=(
            "Run the environment/dependency checks and latency benchmark. Prints a "
            "human report to stderr; with --json also prints the structured report to "
            "stdout. Any error-severity check exits 5."
        ),
        epilog=(
            "Example:\n"
            "  sonoscope doctor --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser


# --- command engines (H1: real handlers wired to their engine modules) ------


def _backend(render_dir: Optional[Path] = None) -> PedalboardVST3Backend:
    """Construct the v1 render backend (E3). Its ``__init__`` imports pedalboard;
    a load failure surfaces as the mapped fatal error via the CLI boundary.

    ``render_dir`` (Finding 1): the CLI owns a per-command temp render dir and
    passes it here so the backend writes every render into that caller-owned dir
    instead of mkdtemp-ing (and leaking) a fresh dir per render. The dir is
    cleaned by the handler once the report/result is serialized."""
    return PedalboardVST3Backend(render_dir=render_dir)


def _load_spec(
    spec_path: Optional[str], *, component: Component
) -> tuple[Spec, str]:
    """Load + validate a versioned input spec (E2) and hash its bytes.

    Returns ``(Spec, spec_sha256)`` where ``spec_sha256`` is the sha256 of the
    on-disk spec bytes (the C1 ``param_set.spec_sha256`` field). A missing
    ``--spec``, an unreadable file, or an invalid spec is a hard error (never a
    silent default), mapped to the design 3.6 exit codes.
    """
    if not spec_path:
        raise UsageError(
            "USAGE_MISSING_SPEC",
            "--spec is required for this command",
            component=component,
        )
    path = Path(spec_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputError(
            "INPUT_SPEC_UNREADABLE",
            f"spec file is missing or unreadable: {path}: {exc}",
            detail={"spec": str(path)},
            component=component,
        ) from exc
    try:
        spec = Spec.model_validate_json(raw)
    except ValidationError as exc:
        raise InputError(
            "INPUT_SPEC_INVALID",
            f"invalid spec {path}: {exc}",
            detail={"spec": str(path)},
            component=component,
        ) from exc
    return spec, hashlib.sha256(raw).hexdigest()


def _load_report(
    path: Optional[str], *, side: Literal["baseline", "candidate"]
) -> AnalysisReport | WavAnalysisReport:
    """Load + validate an on-disk analysis report JSON (A5, iterate-descriptors).

    Accepts BOTH report shapes the ``analyze`` command emits:

    - a JSON **object** -> the plugin-path :class:`AnalysisReport`;
    - a JSON **array** -> the wav-path :class:`WavAnalysisReport` (one entry per
      chunk). Dispatch is on the parsed JSON's own top-level type, so the shape is
      never guessed and a malformed array still fails loud.

    Models :func:`_load_spec`: read bytes, parse, then validate. Every failure is a
    hard :class:`InputError` (exit 2), never a silent skip:

    - an unreadable/missing path -> ``detail["reason"] == "unreadable"``;
    - non-JSON bytes, or valid JSON that is neither a valid ``AnalysisReport`` nor a
      valid ``WavAnalysisReport`` -> ``detail["reason"] == "invalid_report"``.

    ``side`` (``"baseline"``/``"candidate"``) is echoed into ``detail`` so a caller can
    tell which of the two report inputs failed. Both models are imported read-only
    from the frozen schema.

    A missing flag is a USAGE error, not an INPUT error: ``--baseline`` /
    ``--candidate`` are argparse ``default=None`` (not ``required``), so an omitted
    flag arrives here as ``path=None``. Guard it FIRST (mirrors ``_load_spec``'s
    ``USAGE_MISSING_SPEC`` guard) so it becomes a typed ``USAGE_MISSING_REPORT``
    (exit 1, ``cli`` component) instead of letting ``Path(None)`` raise a bare
    ``TypeError`` that escapes to the generic handler and is misreported as an
    ``INTERNAL_ERROR`` (also exit 1) — a sonoscope-bug shape for a user mistake.
    """
    if not path:
        raise UsageError(
            "USAGE_MISSING_REPORT",
            f"--{side} is required for iterate-descriptors",
            component="cli",
        )
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise InputError(
            DESCRIPTORS_REPORT_INVALID,
            f"{side} report file is missing or unreadable: {p}: {exc}",
            detail={"reason": "unreadable", "side": side, "path": str(p)},
            component="analyze",
        ) from exc
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise InputError(
            DESCRIPTORS_REPORT_INVALID,
            f"{side} report is not valid JSON: {p}: {exc}",
            detail={"reason": "invalid_report", "side": side, "path": str(p)},
            component="analyze",
        ) from exc
    model: type[AnalysisReport] | type[WavAnalysisReport] = (
        WavAnalysisReport if isinstance(parsed, list) else AnalysisReport
    )
    try:
        return model.model_validate(parsed)
    except ValidationError as exc:
        raise InputError(
            DESCRIPTORS_REPORT_INVALID,
            f"{side} report is not a valid analysis report: {p}: {exc}",
            detail={"reason": "invalid_report", "side": side, "path": str(p)},
            component="analyze",
        ) from exc


def _require_plugin(plugin: Optional[str], *, component: Component) -> Path:
    """Return the validated plugin path (shared seam for every ``--plugin`` command).

    A missing ``--plugin`` flag is a USAGE error (exit 1). A supplied-but-nonexistent
    plugin path is USER INPUT, so it is a typed :class:`InputError` (exit 2,
    ``PLUGIN_PATH_NOT_FOUND``) raised HERE at the CLI seam, BEFORE any render/probe —
    FIX 2 (dogfood): without this guard a bad ``--plugin`` path let a bare
    ``FileNotFoundError`` escape to the generic handler and be misreported as an
    ``INTERNAL_ERROR`` (exit 1). Applies to analyze/render/iterate/determinism, which
    all resolve their plugin through this one helper.
    """
    if not plugin:
        raise UsageError(
            "USAGE_MISSING_PLUGIN",
            "--plugin is required for this command",
            component=component,
        )
    path = Path(plugin)
    if not path.exists():
        raise InputError(
            "PLUGIN_PATH_NOT_FOUND",
            f"plugin path does not exist: {path}",
            detail={"plugin": str(path)},
            component=component,
        )
    return path


def _analyze_adapter(args: argparse.Namespace):
    """Select the perception adapter for ``analyze`` (default: G1 QwenLocalAdapter,
    graceful when the model is absent). ``--adapter null`` forces the NullAdapter."""
    if args.adapter == "null":
        return NullAdapter()
    return QwenLocalAdapter()


def _run_render(args: argparse.Namespace) -> int:
    """`render` engine (E5): resolve -> subprocess-render -> wav + render meta.

    Writes the wav to ``--out`` when given and prints a render summary (wav path +
    backend + RenderMeta) as JSON to stdout (debug/inspection; no C1 report kind).
    """
    plugin = _require_plugin(args.plugin, component="render")
    spec, _spec_sha256 = _load_spec(args.spec, component="render")
    # Finding 1: own the render dir here so the backend never mkdtemps-per-render.
    # A single ``render`` produces exactly one wav; that wav IS the deliverable, so
    # (unlike analyze/iterate/determinism) the dir is NOT unconditionally cleaned:
    # with --out the wav is copied out and the transient dir is removed; without
    # --out the wav is left in place and its path is printed for the caller.
    render_dir = Path(tempfile.mkdtemp(prefix="sonoscope-render-"))
    # Finding 4 (Gemini review, final batch): with --out the wav is copied out, so
    # the transient render dir is the wav's throwaway staging area and MUST be
    # reclaimed even when render/copy raises (try/finally) — otherwise a failing
    # render leaks the dir on disk. Without --out the wav IS the deliverable left in
    # render_dir for the caller to read, so that dir is intentionally kept (only its
    # path is printed); cleanup is scoped to the --out case only.
    cleanup_render_dir = bool(args.out)
    try:
        # Forward the render-override flags (None -> spec value; see
        # _add_render_spec_flags / render_orchestrator.render precedence).
        outcome = render_orchestrator.render(
            spec,
            plugin,
            _backend(render_dir),
            sample_rate_hz=args.sample_rate,
            block_size=args.block_size,
            channels=args.channels,
            seed=args.seed,
        )

        dest_wav = outcome.wav_path
        if args.out:
            out_path = Path(args.out)
            if out_path.is_dir() or args.out.endswith(os.sep):
                out_path = out_path / outcome.wav_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(outcome.wav_path, out_path)
            dest_wav = out_path

        print(
            json.dumps(
                {
                    "kind": "render",
                    "wav_path": str(dest_wav),
                    "backend": outcome.backend,
                    "backend_version": outcome.backend_version,
                    "ref_sha256": outcome.ref_sha256,
                    "seed": outcome.seed,
                    "render_meta": dataclasses.asdict(outcome.render_meta),
                }
            )
        )
        return int(ExitCode.OK)
    finally:
        if cleanup_render_dir:
            shutil.rmtree(render_dir, ignore_errors=True)


def _run_analyze(args: argparse.Namespace) -> int:
    """`analyze` engine (F1): render -> deterministic ground truth -> report.

    ``--wav`` (standalone wav analysis) loads the wav at native rate, resamples the
    analyzed slice to 48 kHz, and emits an ARRAY of independently-analyzed chunks
    (dispatched to :func:`_run_analyze_wav`). ``--plugin/--spec`` runs the F1
    ``analyze_plugin_spec`` path. Perception is opt-in (``--perception``; off the
    critical path by design); when disabled the report carries the ``disabled``
    perception block.

    Descriptor gating is opt-in via ``--expect-descriptors SPEC`` (optionally with
    ``--fail-on-red``) and is SHARED across both source paths (loaded once, above the
    branch). Output contract: the report JSON stays on STDOUT as the SINGLE
    unchanged document; when gating is active the verdict+reasons are emitted as one
    compact single-line JSON object on STDERR. With ``--fail-on-red``, a RED verdict
    returns exit 4 (ANALYSIS); otherwise a RED verdict still prints but exits 0.
    ``--fail-on-red`` without ``--expect-descriptors`` is a USAGE error (exit 1).

    Cross-flag rigor: the wav-only slice/chunk flags are rejected with ``--plugin``
    and the plugin-render flags are rejected with ``--wav``, each a typed InputError
    (exit 2) — never a silently discarded value.
    """
    # Cross-flag precondition (C4): --fail-on-red gates the DESCRIPTOR verdict, so it
    # is meaningless without a spec to gate against. Enforced at the TOP — before any
    # spec load or render — so a misuse never wastes a render. It is a USAGE error
    # (exit 1), NOT argparse's own exit code, so the message names the exact misuse.
    if args.fail_on_red and not args.expect_descriptors:
        raise UsageError(
            "USAGE_FAIL_ON_RED_REQUIRES_SPEC",
            "analyze --fail-on-red requires --expect-descriptors",
            component="cli",
        )
    # Fail-fast: load + validate the descriptor expectation spec BEFORE the render, so a
    # malformed spec exits 2 without ever wasting a render or emitting a report.
    expectation = None
    expectation_spec_sha256: Optional[str] = None
    if args.expect_descriptors:
        expectation = load_expected_descriptors(
            args.expect_descriptors, block_kind="audio"
        )
        # Provenance for the persisted descriptor_gate: sha256 of the raw spec bytes.
        # load_expected_descriptors has already proven the path readable, so this
        # read cannot surface a failure the loader did not already fail loudly on.
        expectation_spec_sha256 = hashlib.sha256(
            Path(args.expect_descriptors).read_bytes()
        ).hexdigest()

    # Source dispatch (by design). The --wav and --plugin flags are an argparse
    # mutually-exclusive required group, so exactly one is set. The gate expectation
    # is loaded above (shared), then threaded into whichever path runs.
    if args.wav is not None:
        _reject_wav_cross_flags(args)
        return _run_analyze_wav(args, expectation, expectation_spec_sha256)
    _reject_plugin_cross_flags(args)

    plugin = _require_plugin(args.plugin, component="analyze")
    spec, spec_sha256 = _load_spec(args.spec, component="analyze")

    perception_enabled = args.perception is True
    adapter = _analyze_adapter(args) if perception_enabled else None
    # Finding 1: own the transient render dir; the wav is consumed to build the
    # report, so the dir is cleaned once the report is serialized to stdout.
    with tempfile.TemporaryDirectory(prefix="sonoscope-render-") as render_dir:
        report = analyze_plugin_spec(
            spec,
            plugin,
            _backend(Path(render_dir)),
            spec_sha256=spec_sha256,
            spec_ref=args.spec or "inline",
            adapter=adapter,
            perception_enabled=perception_enabled,
            # Render-override flags -> render() via analyze_plugin_spec's
            # **render_kwargs (None -> spec value; single precedence point).
            sample_rate_hz=args.sample_rate,
            block_size=args.block_size,
            channels=args.channels,
            seed=args.seed,
        )
        # Legacy path (no gate): byte-identical single-document stdout, exit OK.
        if expectation is None:
            print(report.model_dump_json())
            return int(ExitCode.OK)
        # Gated path. Every fatal check (missing block) MUST precede the report print:
        # _emit_fatal writes the fatal envelope to stdout, so validating first keeps
        # stdout to EXACTLY ONE JSON document on every path.
        if report.descriptors is None:
            raise InputError(
                DESCRIPTORS_NO_BLOCK,
                "analyze --expect-descriptors requires a descriptors block in the report",
                detail={"reason": "no_descriptors_block"},
                component="analyze",
            )
        evaluation = evaluate_descriptors(
            report.descriptors, expectation, block_kind="audio"
        )
        # Persist the verdict into the versioned report (additive descriptor_gate
        # field; the model is mutable, so set the attribute in place) BEFORE the
        # report print, so it appears in the stdout report JSON.
        report.descriptor_gate = DescriptorGateResult(
            verdict=evaluation.verdict,
            reasons=evaluation.reasons,
            spec_sha256=expectation_spec_sha256,
        )
        print(report.model_dump_json())
        # The stderr verdict signal is ADDITIVE-complementary to the persisted field
        # (not a replacement): one compact JSON object on stderr, byte-stable for CI
        # matching.
        print(
            json.dumps(
                {"verdict": evaluation.verdict, "reasons": evaluation.reasons},
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        if args.fail_on_red and evaluation.verdict == "RED":
            return int(ExitCode.ANALYSIS)
        return int(ExitCode.OK)


def _reject_wav_cross_flags(args: argparse.Namespace) -> None:
    """Reject plugin-render flags supplied with ``--wav`` (typed InputError, exit 2).

    ``--wav`` analyzes an existing wav, so the render/perception flags (which drive a
    plugin render) are meaningless. Supplied WITH ``--wav`` they are the WRONG-source
    flags, rejected as a typed InputError rather than silently discarded (mirrors the
    analyze-midi cross-source rigor). ``--perception`` is ``BooleanOptionalAction``
    default ``None`` so a naked ``--wav`` run does not false-positive.
    """
    offenders = [
        name
        for name, value in (
            ("--spec", args.spec),
            ("--perception", args.perception),
            ("--adapter", args.adapter),
            ("--sample-rate", args.sample_rate),
            ("--block-size", args.block_size),
            ("--channels", args.channels),
            ("--seed", args.seed),
        )
        if value is not None
    ]
    if offenders:
        raise InputError(
            INPUT_WAV_FLAG_CONFLICT,
            f"--wav does not accept plugin-path flags: {', '.join(offenders)}",
            detail={"flags": offenders},
            component="analyze",
        )


def _reject_plugin_cross_flags(args: argparse.Namespace) -> None:
    """Reject wav-only slice/chunk flags supplied with ``--plugin`` (exit 2).

    The native slice/chunk flags only make sense against a standalone wav; supplied
    WITH ``--plugin`` they are the WRONG-source flags, a typed InputError rather than
    a silently discarded value.
    """
    offenders = [
        name
        for name, value in (
            ("--offset", args.offset),
            ("--length", args.length),
            ("--unit", args.unit),
            ("--max-chunk-seconds", args.max_chunk_seconds),
        )
        if value is not None
    ]
    if offenders:
        raise InputError(
            INPUT_ANALYZE_FLAG_CONFLICT,
            f"--plugin does not accept wav-path flags: {', '.join(offenders)}",
            detail={"flags": offenders},
            component="analyze",
        )


def _audio_slice(args: argparse.Namespace) -> Optional[AudioSlice]:
    """Assemble the optional native-unit wav slice (by design), or ``None``.

    A slice is requested when any of ``--offset/--length/--unit`` is given. It MUST
    carry an ``--offset`` (the window start); ``--length`` is optional (omit -> to the
    end of the file) and ``--unit`` defaults to ``samples``. A ``--length``/``--unit``
    without an ``--offset`` is an incoherent slice -> typed :class:`InputError`
    (exit 2). Mirrors ``_midi_slice``.
    """
    if args.offset is None and args.length is None and args.unit is None:
        return None
    if args.offset is None:
        raise InputError(
            INPUT_WAV_SLICE_INCOHERENT,
            "a wav slice requires --offset (with optional --length and --unit); "
            "got --length/--unit without --offset",
            detail={
                "offset": args.offset,
                "length": args.length,
                "unit": args.unit,
            },
            component="analyze",
        )
    return AudioSlice(
        offset=args.offset, length=args.length, unit=args.unit or "samples"
    )


def _run_analyze_wav(
    args: argparse.Namespace,
    expectation: Optional[ExpectedDescriptors],
    expectation_spec_sha256: Optional[str],
) -> int:
    """`analyze --wav` path: native-rate wav -> per-chunk 48k analysis -> array report.

    Prints the :class:`WavAnalysisReport` as a single JSON ARRAY on stdout. When a
    descriptor expectation is supplied (shared ``--expect-descriptors``), each chunk
    carries its own gate and the cross-chunk aggregate ``{verdict, red_chunks,
    reasons}`` (richer than the plugin path's ``{verdict, reasons}``) is emitted as
    one compact JSON line on stderr; ``--fail-on-red`` maps an aggregate RED verdict
    to the ANALYSIS exit code (4) AFTER printing, so a CI gate keeps the report.
    """
    slice_spec = _audio_slice(args)
    report = analyze_wav(
        WavFileSource(args.wav),
        slice_spec=slice_spec,
        max_chunk_seconds=args.max_chunk_seconds,
        expectation=expectation,
        expectation_spec_sha256=expectation_spec_sha256,
    )
    # The report is always the single JSON array on stdout.
    print(report.model_dump_json())
    if expectation is None:
        return int(ExitCode.OK)
    verdict, red_chunks, reasons = aggregate_gate(report)
    print(
        json.dumps(
            {"verdict": verdict, "red_chunks": red_chunks, "reasons": reasons},
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    if args.fail_on_red and verdict == "RED":
        return int(ExitCode.ANALYSIS)
    return int(ExitCode.OK)


def _require_opt_str(
    data: dict, key: str, spec_path: Path
) -> Optional[str]:
    """Strictly resolve an optional spec string field (C2 hardening, fix 2).

    Absent, JSON ``null``, or an empty string -> ``None`` (the optional-field
    default). A present NON-string value (e.g. an int) is a mistyped spec field and
    a typed :class:`InputError` (exit 2) — NEVER the silent ``None`` coercion the old
    ``_opt_str`` produced, which masked a typo (``plugin_id: 123``) as "field absent".
    """
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError(
            MIDI_SPEC_INVALID_CODE,
            f"midi capture spec {spec_path} field '{key}' must be a string, got "
            f"{type(value).__name__}",
            detail={"spec": str(spec_path), "field": key},
            component="midi",
        )
    return value or None


def _load_midi_capture_request(
    spec_path: str, plugin_path: Path, plugin_id: Optional[str]
) -> MidiCaptureRequest:
    """Build a :class:`MidiCaptureRequest` from the C2 ``--spec`` JSON (by design).

    The spec carries the transport/render fields (:data:`_MIDI_SPEC_REQUIRED_FIELDS`,
    matching the ``MidiCaptureRequest`` field names); ``plugin_path`` comes from the
    validated ``--plugin`` and ``plugin_id`` from ``--plugin-id`` (overriding a spec
    value). An unreadable file, non-object/invalid JSON, a missing required field, or
    a mistyped value is a typed :class:`InputError` (exit 2) — never a silently
    defaulted transport.
    """
    path = Path(spec_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputError(
            MIDI_SPEC_UNREADABLE_CODE,
            f"midi capture spec is missing or unreadable: {path}: {exc}",
            detail={"spec": str(path)},
            component="midi",
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(
            MIDI_SPEC_INVALID_CODE,
            f"midi capture spec {path} is not valid JSON: {exc}",
            detail={"spec": str(path)},
            component="midi",
        ) from exc
    if not isinstance(data, dict):
        raise InputError(
            MIDI_SPEC_INVALID_CODE,
            f"midi capture spec {path} must be a JSON object of transport/render fields",
            detail={"spec": str(path)},
            component="midi",
        )
    missing = [k for k in _MIDI_SPEC_REQUIRED_FIELDS if k not in data]
    if missing:
        raise InputError(
            MIDI_SPEC_INVALID_CODE,
            f"midi capture spec {path} missing required field(s): {', '.join(missing)}",
            detail={"spec": str(path), "missing": missing},
            component="midi",
        )
    # C2 hardening (fix 2): ``playing`` must be a genuine JSON boolean. The old
    # ``bool(data["playing"])`` accepted any truthy value, so ``"false"`` (a JSON
    # string) coerced to Python ``True`` — a silent transport flip. Reject a non-bool
    # as a typed InputError instead.
    playing = True
    if "playing" in data:
        raw_playing = data["playing"]
        if not isinstance(raw_playing, bool):
            raise InputError(
                MIDI_SPEC_INVALID_CODE,
                f"midi capture spec {path} field 'playing' must be a JSON boolean, "
                f"got {type(raw_playing).__name__}",
                detail={"spec": str(path), "field": "playing"},
                component="midi",
            )
        playing = raw_playing
    # C2 hardening (fix 2): a mistyped optional string (plugin_id/state_b64) is a
    # typed InputError, not the old silent None coercion (validated even when the
    # --plugin-id flag overrides the spec value, since a bad spec field is bad input).
    spec_plugin_id = _require_opt_str(data, "plugin_id", path)
    state_b64 = _require_opt_str(data, "state_b64", path)
    try:
        return MidiCaptureRequest(
            plugin_path=plugin_path,
            tempo_bpm=float(data["tempo_bpm"]),
            start_position_beats=float(data["start_position_beats"]),
            duration_beats=float(data["duration_beats"]),
            tsig_num=int(data["tsig_num"]),
            tsig_den=int(data["tsig_den"]),
            sample_rate=int(data["sample_rate"]),
            block_size=int(data["block_size"]),
            # --plugin-id (flag) overrides the spec's plugin_id when given.
            plugin_id=plugin_id if plugin_id is not None else spec_plugin_id,
            playing=playing,
            state_b64=state_b64,
        )
    except (TypeError, ValueError) as exc:
        raise InputError(
            MIDI_SPEC_INVALID_CODE,
            f"midi capture spec {path} has an invalid field value: {exc}",
            detail={"spec": str(path)},
            component="midi",
        ) from exc


def _reject_cross_source_flags(
    candidates: tuple[tuple[str, object], ...], *, source: str
) -> None:
    """Raise a typed InputError if any source-mismatched flag was supplied (fix 1).

    ``candidates`` is a tuple of ``(flag_name, value)`` pairs that are illegal with
    ``source``; a non-``None`` value is a flag the user set for the WRONG source.
    Mirrors the both/neither source rigor: a conflicting flag is a typed InputError
    (exit 2), never a silently discarded value.
    """
    offenders = [name for name, value in candidates if value is not None]
    if offenders:
        raise InputError(
            MIDI_FLAG_CONFLICT_CODE,
            f"{', '.join(offenders)} "
            f"{'is' if len(offenders) == 1 else 'are'} not valid with {source}; "
            f"remove {'it' if len(offenders) == 1 else 'them'} "
            f"(they belong to the other MIDI source)",
            detail={"source": source, "conflicting": offenders},
            component="midi",
        )


def _midi_source(args: argparse.Namespace):
    """Resolve the exactly-one MIDI source (by design) or raise a typed InputError.

    ``--plugin`` XOR ``--file``: both or neither is a typed :class:`InputError`
    (exit 2). A ``--plugin`` path is validated via the shared ``_require_plugin``
    seam (a bad path -> PLUGIN_PATH_NOT_FOUND, exit 2) and needs ``--spec``; a
    ``--file`` source needs ``--sample-rate`` (the sample timing axis).
    """
    if args.plugin and args.file:
        raise InputError(
            MIDI_SOURCE_CONFLICT_CODE,
            "--plugin and --file are mutually exclusive; give exactly one MIDI source",
            detail={"plugin": args.plugin, "file": args.file},
            component="midi",
        )
    if not args.plugin and not args.file:
        raise InputError(
            MIDI_SOURCE_MISSING_CODE,
            "analyze-midi requires exactly one MIDI source: --plugin <.clap> (with "
            "--spec) or --file <.mid> (with --sample-rate)",
            component="midi",
        )
    if args.plugin:
        # Fix 1: --sample-rate/--tempo belong to the --file source (its sample
        # timing axis / tempo override). Supplied WITH --plugin they would be
        # silently dropped (the transport comes from --spec), so reject them as a
        # typed InputError rather than discard user input.
        _reject_cross_source_flags(
            (("--sample-rate", args.sample_rate), ("--tempo", args.tempo)),
            source="--plugin",
        )
        plugin_path = _require_plugin(args.plugin, component="midi")
        if not args.spec:
            raise InputError(
                MIDI_SPEC_MISSING_CODE,
                "--spec is required with --plugin (the capture transport/render fields)",
                component="midi",
            )
        return _load_midi_capture_request(args.spec, plugin_path, args.plugin_id)

    # file source
    # Fix 1: --spec/--plugin-id belong to the --plugin source (the capture spec /
    # CLAP plugin id). Supplied WITH --file they would be silently dropped, so
    # reject them as a typed InputError rather than discard user input.
    _reject_cross_source_flags(
        (("--spec", args.spec), ("--plugin-id", args.plugin_id)),
        source="--file",
    )
    if args.sample_rate is None:
        raise InputError(
            MIDI_FILE_SAMPLE_RATE_MISSING_CODE,
            "--sample-rate is required with --file (it derives the sample timing axis)",
            component="midi",
        )
    return MidiFileSource(
        path=args.file, sample_rate=args.sample_rate, tempo_bpm=args.tempo
    )


def _midi_slice(args: argparse.Namespace) -> Optional[MidiSlice]:
    """Assemble the optional analysis-window slice (by design), or ``None``.

    A slice is requested when any of ``--offset/--length/--unit`` is given. It MUST
    carry an ``--offset`` (the window start); ``--length`` is optional (omit -> to the
    end) and ``--unit`` defaults to ``samples``. A ``--length``/``--unit`` without an
    ``--offset`` is an incoherent slice -> typed :class:`InputError` (exit 2). The
    engine slices with ``rebase=True`` (MidiSlice default), the operator's "analyze the
    sub-window as if it were the whole capture" intent.
    """
    if args.offset is None and args.length is None and args.unit is None:
        return None
    if args.offset is None:
        raise InputError(
            MIDI_SLICE_INCOHERENT_CODE,
            "a MIDI slice requires --offset (with optional --length and --unit); got "
            "--length/--unit without --offset",
            detail={
                "offset": args.offset,
                "length": args.length,
                "unit": args.unit,
            },
            component="midi",
        )
    return MidiSlice(
        offset=args.offset, length=args.length, unit=args.unit or "samples"
    )


def _run_analyze_midi(args: argparse.Namespace) -> int:
    """`analyze-midi` engine (C2): capture/load a MIDI stream -> deterministic report.

    Source is EITHER a ``--plugin`` capture (a spec-built ``MidiCaptureRequest``, C
    host) OR a ``--file`` .mid (``MidiFileSource``); the two are mutually exclusive and
    exactly one is required (both/neither are typed InputErrors, exit 2). The optional
    ``--expected`` golden, ``--offset/--length/--unit`` slice, and ``--offvel0``
    override are forwarded to the C1 engine (``--offvel0`` None -> the engine's
    per-source default). The versioned report is printed as JSON (mirrors the audio
    ``analyze``'s ``report.model_dump_json()``). ``--fail-on-red`` maps a RED verdict
    to the ANALYSIS exit code (4) AFTER printing the report, so a CI gate keeps the
    gate-able report on stdout; a capture/load/spec failure surfaces as its already-
    typed fatal error, never a fabricated report.
    """
    source = _midi_source(args)
    slice_spec = _midi_slice(args)
    report = analyze_midi(
        source,
        expected=args.expected,
        slice_spec=slice_spec,
        offvel0_policy=args.offvel0,
    )
    print(report.model_dump_json())
    # --fail-on-red: a RED verdict is a deterministic-analysis negative outcome, so it
    # maps to the ANALYSIS exit code (design 3.6, exit 4). The report is already on
    # stdout, so the CI gate both fails AND keeps the RED report for inspection.
    if args.fail_on_red and report.midi.verdict == "RED":
        return int(ExitCode.ANALYSIS)
    return int(ExitCode.OK)


def _run_iterate(args: argparse.Namespace) -> int:
    """`iterate` engine (F3): baseline vs candidate -> R2-thresholded delta.

    Analyzes both specs (deterministic-only), loads the cached ``(binary, patch_class)``
    determinism floors (measuring + caching them when absent), then computes the
    delta + verdict.
    """
    plugin = _require_plugin(args.plugin, component="cli")
    if not args.metric or not args.direction:
        raise UsageError(
            "USAGE_MISSING_EXPECTATION",
            "iterate requires --metric and --direction",
            component="cli",
        )
    baseline_spec, baseline_sha = _load_spec(args.baseline, component="cli")
    candidate_spec, candidate_sha = _load_spec(args.candidate, component="cli")
    # Finding 1: own the transient render dir; every render (both analyses plus the
    # floor-measuring renders) writes into it, and it is cleaned once the delta is
    # serialized to stdout — so the backend never leaks a per-render temp dir.
    with tempfile.TemporaryDirectory(prefix="sonoscope-render-") as render_dir:
        backend = _backend(Path(render_dir))

        # Render-override flags: forwarded to every render on this path (both
        # analyses AND the floor-measuring renders) so the whole compare runs at
        # the overridden render params (None -> spec value; single precedence
        # point in render_orchestrator.render).
        render_overrides = dict(
            sample_rate_hz=args.sample_rate,
            block_size=args.block_size,
            channels=args.channels,
            seed=args.seed,
        )
        baseline_report = analyze_plugin_spec(
            baseline_spec,
            plugin,
            backend,
            spec_sha256=baseline_sha,
            spec_ref=args.baseline,
            perception_enabled=False,
            **render_overrides,
        )
        candidate_report = analyze_plugin_spec(
            candidate_spec,
            plugin,
            backend,
            spec_sha256=candidate_sha,
            spec_ref=args.candidate,
            perception_enabled=False,
            **render_overrides,
        )

        binary = baseline_report.input.plugin.binary_sha256
        patch_class = baseline_spec.patch_class
        floors = read_floors(binary, patch_class)
        if floors is None:
            floors = measure_floors(
                lambda: render_orchestrator.render(
                    baseline_spec, plugin, backend, **render_overrides
                ).wav_path,
                binary_sha256=binary,
                patch_class=patch_class,
                resolved_sha256=baseline_report.input.param_set.resolved_sha256,
                stimulus_ref=baseline_report.input.stimulus.ref,
                repeats=DEFAULT_REPEATS,
            )
            write_floors(floors)

        delta = run_iterate(
            baseline_report,
            candidate_report,
            floors,
            metric=args.metric,
            direction=args.direction,
            min_effect=args.min_effect,
        )
        # Item C (dogfood): ``--brief`` emits a DISPLAY-ONLY reduced object — just the
        # verdict + delta + expectation an LLM needs — selected from the IterateDelta
        # model_dump, OMITTING the two embedded ~12KB baseline/candidate AnalysisReports
        # that dominate the default payload's token cost. This reduced form is NOT the
        # canonical iterate-delta schema (it drops required fields on purpose); the
        # default (no ``--brief``) still prints the full schema-valid IterateDelta.
        if args.brief:
            dumped = delta.model_dump()
            brief = {key: dumped[key] for key in ("verdict", "delta", "expectation")}
            print(json.dumps(brief))
        else:
            print(delta.model_dump_json())
    return int(ExitCode.OK)


def _sanitize_finite(value: float) -> Optional[float]:
    """Map a non-finite float (NaN/Inf) to ``None`` (JSON ``null``); finite passes through.

    The strict-JSON invariant (``json.dumps(..., allow_nan=False)``) rejects the
    ``NaN``/``Infinity`` tokens; sanitizing to ``None`` FIRST keeps the emitted diff a
    valid JSON document that any strict parser round-trips.
    """
    return value if math.isfinite(value) else None


def _descriptor_diff_payload(diff: DescriptorTermDiff) -> dict:
    """The JSON-ready mapping for one :class:`DescriptorTermDiff` (one report pair,
    or one chunk pair in the wav-path chunk-wise form).

    Non-finite drift values are sanitized to ``None`` HERE so every emitter below can
    dump with ``allow_nan=False`` and still produce a valid JSON document.
    """
    return {
        "added": diff.added,
        "removed": diff.removed,
        "direction_changed": diff.direction_changed,
        "value_drift": [
            {
                "term": d.term,
                "baseline_value": _sanitize_finite(d.baseline_value),
                "candidate_value": _sanitize_finite(d.candidate_value),
            }
            for d in diff.value_drift
        ],
    }


def _descriptor_diff_json_line(diff: DescriptorTermDiff) -> str:
    """Serialize a single-report :class:`DescriptorTermDiff` to the exact single-line
    strict-JSON form. Compact separators produce the byte-exact contract the tests pin.
    """
    return json.dumps(
        _descriptor_diff_payload(diff), separators=(",", ":"), allow_nan=False
    )


def _descriptor_chunkwise_diff_json_line(diffs: list[DescriptorTermDiff]) -> str:
    """Serialize the wav-path CHUNK-WISE diff to its single-line strict-JSON form.

    Shape (deliberately distinct from the single-report line so a consumer can tell
    the two apart STRUCTURALLY, not by guessing)::

        {"chunk_count":N,"chunks":[{"chunk_index":0,"added":[],...},...]}

    ``chunks[i]`` carries the same four diff keys as the single-report object plus its
    own ``chunk_index``, mirroring how :func:`aggregate_gate` attributes cross-chunk
    findings per chunk index. ``chunk_count`` equals ``len(chunks)`` and equals the
    (necessarily equal) chunk count of both input reports.
    """
    payload = {
        "chunk_count": len(diffs),
        "chunks": [
            {"chunk_index": i, **_descriptor_diff_payload(diff)}
            for i, diff in enumerate(diffs)
        ],
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _run_iterate_descriptors(args: argparse.Namespace) -> int:
    """`iterate-descriptors` engine (A5): baseline vs candidate report -> term diff.

    Loads two on-disk reports (baseline first, then candidate) via
    :func:`_load_report`; a missing/unparseable/wrong-shape report is a typed
    :class:`InputError` (exit 2) naming the failing ``side``. Both report shapes the
    ``analyze`` command emits are accepted, and the two must AGREE on shape:

    - two single :class:`AnalysisReport` objects (plugin path) -> the one-object diff
      line. A report that parses but carries NO descriptors block is a distinct
      ``DESCRIPTORS_NO_BLOCK`` INPUT error (also naming the side); this guard is
      reachable only here, because ``AnalysisReport.descriptors`` is Optional while
      ``WavChunkAnalysis.descriptors`` is REQUIRED by the schema.
    - two :class:`WavAnalysisReport` arrays (wav path) -> chunk i is diffed against
      chunk i for EVERY i and the chunk-wise object is emitted. Unequal chunk counts
      are ``DESCRIPTORS_CHUNK_COUNT_MISMATCH`` rather than a truncated comparison.
    - a MIXED pair -> ``DESCRIPTORS_REPORT_SHAPE_MISMATCH``; the shapes have no common
      alignment, so guessing one would be wrong.

    The pure :func:`diff_descriptor_terms` does the work in every case, and the result
    is printed as one strict-JSON line on stdout (exit 0).
    """
    # PR review: ``--value-tolerance`` bands value-drift as ``abs(delta) > tol`` in
    # ``diff_descriptor_terms``. A non-finite (nan/inf) or negative tol silently
    # produces WRONG drift output (negative -> everything flagged; nan/inf ->
    # nothing flagged), so reject it loud at the CLI seam BEFORE any report load.
    if not math.isfinite(args.value_tolerance) or args.value_tolerance < 0:
        raise UsageError(
            "USAGE_INVALID_VALUE_TOLERANCE",
            "iterate-descriptors --value-tolerance must be a finite, "
            f"non-negative number; got {args.value_tolerance}",
            component="cli",
        )
    baseline = _load_report(args.baseline, side="baseline")
    candidate = _load_report(args.candidate, side="candidate")

    base_is_wav = isinstance(baseline, WavAnalysisReport)
    cand_is_wav = isinstance(candidate, WavAnalysisReport)
    if base_is_wav != cand_is_wav:
        base_shape = "wav-chunk-array" if base_is_wav else "analysis-report"
        cand_shape = "wav-chunk-array" if cand_is_wav else "analysis-report"
        raise InputError(
            DESCRIPTORS_REPORT_SHAPE_MISMATCH,
            "iterate-descriptors cannot compare a wav-path chunk array against a "
            f"single analysis report; baseline is {base_shape}, candidate is "
            f"{cand_shape}. Produce both reports from the same analyze path.",
            detail={
                "reason": "shape_mismatch",
                "baseline_shape": base_shape,
                "candidate_shape": cand_shape,
            },
            component="analyze",
        )

    if isinstance(baseline, WavAnalysisReport) and isinstance(
        candidate, WavAnalysisReport
    ):
        base_chunks = baseline.root
        cand_chunks = candidate.root
        if len(base_chunks) != len(cand_chunks):
            raise InputError(
                DESCRIPTORS_CHUNK_COUNT_MISMATCH,
                "iterate-descriptors requires equal chunk counts to diff wav-path "
                f"reports chunk-wise; baseline has {len(base_chunks)} chunks, "
                f"candidate has {len(cand_chunks)}",
                detail={
                    "reason": "chunk_count_mismatch",
                    "baseline_chunks": len(base_chunks),
                    "candidate_chunks": len(cand_chunks),
                },
                component="analyze",
            )
        diffs = [
            diff_descriptor_terms(
                b.descriptors, c.descriptors, value_tolerance=args.value_tolerance
            )
            for b, c in zip(base_chunks, cand_chunks)
        ]
        print(_descriptor_chunkwise_diff_json_line(diffs))
        return int(ExitCode.OK)

    if baseline.descriptors is None:
        raise InputError(
            DESCRIPTORS_NO_BLOCK,
            "iterate-descriptors baseline report has no descriptors block",
            detail={"reason": "no_descriptors_block", "side": "baseline"},
            component="analyze",
        )
    if candidate.descriptors is None:
        raise InputError(
            DESCRIPTORS_NO_BLOCK,
            "iterate-descriptors candidate report has no descriptors block",
            detail={"reason": "no_descriptors_block", "side": "candidate"},
            component="analyze",
        )
    diff = diff_descriptor_terms(
        baseline.descriptors,
        candidate.descriptors,
        value_tolerance=args.value_tolerance,
    )
    print(_descriptor_diff_json_line(diff))
    return int(ExitCode.OK)


def _run_determinism(args: argparse.Namespace) -> int:
    """`determinism` engine (F2): render N times -> per-feature floors -> cache.

    An initial render resolves the ``resolved_sha256`` / stimulus ref (the cache
    key context); ``measure_floors`` then renders ``--repeats`` times and derives
    the floors, persisted to the cache and printed to stdout.
    """
    # MINOR-1: measuring a floor is meaningless below two renders. Guard at the
    # CLI seam BEFORE any render so a too-low ``--repeats`` is classified as the
    # user input mistake it is — a USAGE error (exit 1) — rather than falling to
    # ``measure_floors``' raw ``ValueError`` and being misreported as an
    # INTERNAL_ERROR by the generic handler. (``component`` uses this handler's
    # existing "render" value; the Component literal has no "determinism" member.)
    if args.repeats < 2:
        raise UsageError(
            "USAGE_REPEATS_TOO_LOW",
            f"--repeats must be >= 2 to measure a floor; got {args.repeats}",
            component="render",
        )
    plugin = _require_plugin(args.plugin, component="render")
    spec, _spec_sha256 = _load_spec(args.spec, component="render")
    # Finding 1: own the transient render dir; the seed render plus all N repeat
    # renders write into it and each wav is read immediately, so the dir is cleaned
    # once the floors object is serialized — the backend never leaks per render.
    with tempfile.TemporaryDirectory(prefix="sonoscope-render-") as render_dir:
        backend = _backend(Path(render_dir))

        # Render-override flags: forwarded to the seed render AND every repeat
        # render so the measured floor reflects the overridden render params
        # (None -> spec value; single precedence point in render_orchestrator).
        render_overrides = dict(
            sample_rate_hz=args.sample_rate,
            block_size=args.block_size,
            channels=args.channels,
            seed=args.seed,
        )
        seed_outcome = render_orchestrator.render(
            spec, plugin, backend, **render_overrides
        )
        floors = measure_floors(
            lambda: render_orchestrator.render(
                spec, plugin, backend, **render_overrides
            ).wav_path,
            binary_sha256=binary_sha256(plugin),
            patch_class=spec.patch_class,
            resolved_sha256=seed_outcome.resolved.resolved_sha256,
            stimulus_ref=seed_outcome.resolved.stimulus.ref or "inline",
            repeats=args.repeats,
            method=args.method,
        )
        write_floors(floors)
        print(floors.model_dump_json())
    return int(ExitCode.OK)


def _probe_adapter(args: argparse.Namespace):
    """Select the perception adapter for ``probe`` (the R6 Qwen feasibility gate).

    Defaults to the G1 QwenLocalAdapter; ``--adapter null`` forces the NullAdapter
    (which reports unavailable, exercising the graceful path)."""
    if args.adapter == "null":
        return NullAdapter()
    return QwenLocalAdapter()


def _run_probe(args: argparse.Namespace) -> int:
    """`probe` engine (G1): run the R6 A/B fixture judgment -> probe verdict.

    Perception unavailability is NOT an error (advisory, by design): a KNOWN-absent
    model surfaces as ``status:"unavailable"`` with exit 0, never a fake FAIL.
    """
    fixtures_dir = Path(args.fixtures) if args.fixtures else DEFAULT_PROBE_FIXTURES
    adapter = _probe_adapter(args)
    pairs = default_fixture_pairs(fixtures_dir)
    try:
        report = run_probe(adapter, pairs)
    except ProbeUnavailable as exc:
        # Graceful: perception disabled/unavailable is not a failure (exit 0).
        print(
            json.dumps(
                {"kind": "probe", "status": "unavailable", "reason": str(exc)}
            )
        )
        return int(ExitCode.OK)
    except ProbeFixturesMissing as exc:
        # FIX 3 (dogfood): an available model with absent fixture wavs is USER INPUT
        # (a missing/mispointed fixture dir), so it maps to a typed InputError
        # (exit 2, PLUGIN-path-style) with a clear how-to message rather than the
        # raw FileNotFoundError that previously escaped to INTERNAL_ERROR (exit 1).
        raise InputError(
            "PROBE_FIXTURES_NOT_FOUND",
            f"probe fixtures not found under {fixtures_dir}: "
            f"{len(exc.missing)} fixture wav(s) missing. Restore the committed A/B "
            "fixture set, or pass --fixtures <dir> pointing at an "
            "existing fixture set.",
            detail={
                "fixtures_dir": str(fixtures_dir),
                "missing": [str(p) for p in exc.missing],
            },
            component="perception",
        ) from exc

    print(
        json.dumps(
            {
                "kind": "probe",
                "status": "ok",
                "verdict": report.verdict,
                "n_correct": report.n_correct,
                "m_total": report.m_total,
                "ratio": report.ratio,
                "pairs": [
                    {"key": pair.key, "correct": pair.correct}
                    for pair in report.pairs
                ],
            }
        )
    )
    return int(ExitCode.OK)


def _run_corpus(args: argparse.Namespace) -> int:
    """`corpus` engine (E1): list items, or verify pinned checksums (R4).

    ``verify`` maps a corpus-integrity FAILURE (missing / hash-drifted item) to a
    hard INPUT error (exit 2) — pins are law, never a silent pass.
    """
    if args.action == "list":
        items = list_items()
        print(
            json.dumps(
                {
                    "kind": "corpus",
                    "action": "list",
                    "items": [
                        {
                            "name": item.name,
                            "path": item.path,
                            "kind": item.kind,
                            "sha256": item.sha256,
                        }
                        for item in items
                    ],
                }
            )
        )
        return int(ExitCode.OK)

    result = corpus_verify()
    if not result.ok:
        raise InputError(
            "INPUT_CORPUS_DRIFT",
            f"corpus verify failed: {len(result.failures)} item(s) missing or "
            "hash-drifted",
            detail={
                "failures": [
                    {"name": f.name, "path": f.path, "reason": f.reason}
                    for f in result.failures
                ]
            },
            component="corpus",
        )
    print(
        json.dumps(
            {
                "kind": "corpus",
                "action": "verify",
                "ok": True,
                "items": [
                    {"name": item.name, "ok": item.ok} for item in result.items
                ],
            }
        )
    )
    return int(ExitCode.OK)


def _run_doctor(args: argparse.Namespace) -> int:
    """`doctor` engine (H1): environment checks + latency benchmark.

    Prints the human-readable report to stderr (by design: human logs to
    stderr) and, with ``--json``, the structured report to stdout. Any
    error-severity check raises the ENVIRONMENT fatal error (exit 5); latency
    over-target is a non-fatal warning (soft criterion, exit stays 0).
    """
    report = doctor.run_doctor()
    print(doctor.format_report(report), file=sys.stderr)
    if args.json:
        print(
            json.dumps(
                {
                    "kind": "doctor",
                    "ok": report.ok,
                    "checks": [
                        {
                            "name": c.name,
                            "severity": c.severity,
                            "detail": c.detail,
                        }
                        for c in report.checks
                    ],
                    "latencies": [
                        {
                            "metric": latency.metric,
                            "measured_s": latency.measured_s,
                            "target_s": latency.target_s,
                            "over_target": latency.over_target,
                        }
                        for latency in report.latencies
                    ],
                }
            )
        )
    if not report.ok:
        failed = "; ".join(
            f"{c.name}: {c.detail}" for c in report.failed_checks
        )
        raise SonoscopeEnvironmentError(
            "ENVIRONMENT_DOCTOR_FAILED",
            f"doctor found {len(report.failed_checks)} environment "
            f"failure(s): {failed}",
            detail={"failed_checks": [c.name for c in report.failed_checks]},
            component="cli",
        )
    return int(ExitCode.OK)


def _run_schema(args: argparse.Namespace) -> int:
    """Emit the draft-2020-12 JSON Schema for the requested kind (C2).

    Writes to ``--out`` when given (a directory receives ``<kind>.schema.json``;
    any other path is treated as the target file), otherwise to stdout.
    """
    schema = json_schema_for(args.kind)
    text = json.dumps(schema, indent=2)
    if args.out:
        dest = Path(args.out)
        # A trailing separator marks a directory even when it does not exist yet
        # (matches _run_render); without this a non-existent ``schemas/`` would
        # be written as a FILE named ``schemas`` (Gemini review cycle 3).
        if dest.is_dir() or args.out.endswith(os.sep):
            dest = dest / f"{args.kind}.schema.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n")
    else:
        print(text)
    return int(ExitCode.OK)


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "render": _run_render,
    "analyze": _run_analyze,
    "analyze-midi": _run_analyze_midi,
    "iterate": _run_iterate,
    "iterate-descriptors": _run_iterate_descriptors,
    "determinism": _run_determinism,
    "probe": _run_probe,
    "schema": _run_schema,
    "corpus": _run_corpus,
    "doctor": _run_doctor,
}


def _dispatch(args: argparse.Namespace) -> int:
    return _HANDLERS[args.command](args)


def _emit_fatal(err: SonoscopeError) -> None:
    fatal = err.to_fatal_error(
        sonoscope_version=__version__, generated_at=_now_iso()
    )
    print(fatal.model_dump_json(), file=sys.stdout)


def _emit_generic(exc: Exception) -> None:
    fatal = FatalError(
        generated_at=_now_iso(),
        sonoscope_version=__version__,
        error=FatalErrorDetail(
            code="INTERNAL_ERROR", message=str(exc), component="cli"
        ),
    )
    print(fatal.model_dump_json(), file=sys.stdout)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse, dispatch, and translate failures to the design 3.6 exit codes."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return _dispatch(args)
    except SonoscopeError as exc:
        _emit_fatal(exc)
        return int(exc.exit_code)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        _emit_generic(exc)
        return _GENERIC_EXIT


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
