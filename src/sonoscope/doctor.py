"""`doctor` environment check + latency benchmark (Task H1, by design).

`doctor` answers "is this machine correctly provisioned to produce trustworthy,
reproducible audio-QA results?" It runs a set of environment checks plus a
deterministic-only latency micro-benchmark, and reports a structured
:class:`DoctorReport` the CLI (H1) maps to an exit code.

Checks (by design / plan H1 coverage):

- **pins (A2):** every deterministic-core dependency's installed version must
  EXACTLY equal its :mod:`sonoscope.pins` constant. Drift is a hard ENVIRONMENT
  fault (pins are law, AGENTS.md).
- **lockfile drift (A2):** ``uv lock --check`` must succeed; a drifted/tampered
  ``uv.lock`` is an ENVIRONMENT fault.
- **Surge XT install + factory content (A4):** ``verify_surge_xt.sh`` recomputes
  the pinned VST3/CLAP/factory-content hashes; a nonzero exit (missing install or
  drift) is an ENVIRONMENT fault.
- **backend load (E3):** the pedalboard VST3 backend must import + construct.
- **perception health (G1):** the adapter's ``health()`` is reported, but an
  unavailable model is NOT a failure — perception is advisory and degrades
  gracefully. It is reported as an "ok" check.

Latency benchmark (I2, satisfies DoD): a deterministic-only
micro-benchmark measures the target wall-time paths and compares each to its
target. Over-target is a **non-fatal warning** (soft criterion) — slow hardware
must never hard-fail ``doctor``. Targets live in a single constants location
(:data:`LATENCY_TARGETS_S`).

Exit-code mapping (the CLI, H1): ANY error-severity check -> ENVIRONMENT (exit
5); otherwise exit 0. Latency warnings never change the exit code.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from sonoscope import corpus
from sonoscope.features.librosa_features import compute_summary
from sonoscope.perception.base import AdapterHealth, PerceptionAdapter
from sonoscope.pins import PINNED_VERSIONS
from sonoscope.schema import ExitCode

# Repo root: src/sonoscope/doctor.py -> parents[2] == repo root (mirrors corpus.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SURGE_SCRIPT = REPO_ROOT / "scripts" / "verify_surge_xt.sh"
SURGE_MANIFEST = REPO_ROOT / "pins" / "surge_xt.manifest.toml"

# --- latency budget (SINGLE source of truth for the targets) -----------------
# Targets in wall-clock SECONDS on the target Apple Silicon dev machine (by
# design). Over-target is a soft, non-fatal warning (I2), never an ENVIRONMENT
# failure. Keys are the metric names surfaced in the latency warnings.
LATENCY_TARGETS_S: dict[str, float] = {
    "render_2s": 1.0,
    "deterministic_feature_extraction": 0.5,
    "deterministic_analyze": 2.0,
}

CheckSeverity = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class CheckResult:
    """One environment check outcome. ``severity == "error"`` is the only value
    that flips ``doctor`` to a nonzero (ENVIRONMENT) exit."""

    name: str
    severity: CheckSeverity
    detail: str


@dataclass(frozen=True)
class LatencyResult:
    """One latency path's measured wall time vs its target. ``over_target``
    marks a non-fatal warning (soft criterion, exit stays 0)."""

    metric: str
    measured_s: float
    target_s: float
    over_target: bool


@dataclass(frozen=True)
class DoctorReport:
    """Structured `doctor` result. The CLI (H1) maps :attr:`exit_code`."""

    checks: tuple[CheckResult, ...]
    latencies: tuple[LatencyResult, ...]

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        """Error-severity checks (the ENVIRONMENT faults)."""
        return tuple(c for c in self.checks if c.severity == "error")

    @property
    def latency_warnings(self) -> tuple[LatencyResult, ...]:
        """Over-target latency paths (non-fatal soft-criterion warnings)."""
        return tuple(latency for latency in self.latencies if latency.over_target)

    @property
    def ok(self) -> bool:
        """True iff no error-severity check fired (latency warnings do not count)."""
        return len(self.failed_checks) == 0

    @property
    def exit_code(self) -> int:
        """OK (0) when clean, else ENVIRONMENT (5)."""
        return int(ExitCode.OK if self.ok else ExitCode.ENVIRONMENT)


# A subprocess runner (``subprocess.run``-shaped) so the environment probes are
# injectable for hermetic, non-integration testing (H1 fakes uv / Surge verify).
Runner = Callable[..., "subprocess.CompletedProcess"]

# Module-level defaults so ``run_doctor()`` (called by the CLI with no args) can
# be driven by monkeypatch in tests without touching the CLI call site.
_RUNNER: Runner = subprocess.run


def _default_adapter() -> Optional[PerceptionAdapter]:
    """The default perception adapter probed by ``doctor`` (G1 QwenLocalAdapter).

    Constructing it is cheap, and ``health()`` is a LIGHTWEIGHT probe (NIT-1): it
    reports availability via a runtime-import + HF-cache presence check WITHOUT
    loading the ~16 GB model, so ``doctor`` never pays the multi-GB load cost. A
    KNOWN-absent runtime/model degrades to ``available=False`` gracefully.
    Imported lazily so the deterministic core imports ``doctor`` without the
    perception extra.
    """
    from sonoscope.perception.qwen_local import QwenLocalAdapter

    return QwenLocalAdapter()


# --- individual checks ------------------------------------------------------


def _check_pins() -> CheckResult:
    """Every pinned dependency's installed version must EXACTLY match (A2).

    A pinned dependency that is not installed at all raises
    :class:`importlib.metadata.PackageNotFoundError`. That is recorded as the
    version string ``"not installed"`` so it surfaces as ordinary drift (an
    error-severity check, exit 5) instead of crashing ``doctor`` with an uncaught
    exception — a missing pinned dep is a hard ENVIRONMENT fault (pins are law),
    never a fatal traceback.
    """
    installed: dict[str, str] = {}
    for dist in PINNED_VERSIONS:
        try:
            installed[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            installed[dist] = "not installed"
    drifted = {
        dist: (installed[dist], PINNED_VERSIONS[dist])
        for dist in PINNED_VERSIONS
        if installed[dist] != PINNED_VERSIONS[dist]
    }
    if drifted:
        detail = "; ".join(
            f"{dist}: installed {got} != pinned {want}"
            for dist, (got, want) in sorted(drifted.items())
        )
        return CheckResult("pins", "error", f"dependency pin drift: {detail}")
    return CheckResult(
        "pins", "ok", f"{len(PINNED_VERSIONS)} pinned dependencies match"
    )


def _check_lockfile(repo_root: Path, runner: Runner) -> CheckResult:
    """``uv lock --check`` must succeed; a drifted/tampered lock is ENVIRONMENT."""
    try:
        result = runner(
            ["uv", "lock", "--check"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        # uv absent: cannot verify — report as an error (the pinned toolchain is
        # part of the required environment). Never a silent pass.
        return CheckResult(
            "lockfile", "error", f"uv not available to verify lockfile: {exc}"
        )
    if result.returncode != 0:
        return CheckResult(
            "lockfile",
            "error",
            f"uv lock --check reported drift (exit {result.returncode}): "
            f"{(result.stderr or '').strip()}",
        )
    return CheckResult("lockfile", "ok", "uv.lock in sync")


def _check_surge_xt(
    verify_script: Path, manifest: Path, runner: Runner
) -> CheckResult:
    """Surge XT install + factory content verified via A4's script; nonzero exit
    (missing install or drift) is an ENVIRONMENT fault."""
    if not verify_script.is_file():
        return CheckResult(
            "surge_xt", "error", f"verify script missing: {verify_script}"
        )
    try:
        result = runner(
            ["bash", str(verify_script), str(manifest)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return CheckResult(
            "surge_xt", "error", f"cannot run verify_surge_xt.sh: {exc}"
        )
    if result.returncode != 0:
        return CheckResult(
            "surge_xt",
            "error",
            f"Surge XT missing or drifted (verify exit {result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()}",
        )
    return CheckResult("surge_xt", "ok", "Surge XT install + factory content verified")


def _check_backend_load() -> CheckResult:
    """The v1 render backend (pedalboard VST3, E3) must import + construct."""
    try:
        from sonoscope.backends.pedalboard_vst3 import PedalboardVST3Backend

        backend = PedalboardVST3Backend()
    except Exception as exc:  # noqa: BLE001 - any load fault is an env error
        return CheckResult(
            "backend", "error", f"render backend failed to load: {exc!r}"
        )
    return CheckResult(
        "backend", "ok", f"backend {backend.id} v{backend.version} loaded"
    )


def _check_perception(adapter: Optional[PerceptionAdapter]) -> CheckResult:
    """Report perception adapter health. An unavailable model is NOT a failure —
    perception is advisory and degrades gracefully, so this is always an
    "ok" check; ``detail`` records whether the adapter is available."""
    if adapter is None:
        return CheckResult("perception", "ok", "no perception adapter configured")
    health: AdapterHealth = adapter.health()
    if health.available:
        return CheckResult(
            "perception",
            "ok",
            f"perception available: {health.model_id} ({health.runtime})",
        )
    return CheckResult(
        "perception",
        "ok",
        f"perception unavailable (advisory, non-fatal): "
        f"{health.reason or 'no model'}",
    )


# --- latency benchmark ------------------------------------------------------


def _measure_latencies(repo_root: Path) -> dict[str, float]:
    """Deterministic-only latency micro-benchmark (I2).

    Measures the plugin-free deterministic feature-extraction path on a pinned
    corpus signal stimulus (ground-truth librosa layer). The render / deterministic
    -analyze paths require a plugin (Surge XT) and are therefore measured by the
    integration dogfood (H2), not this always-available default; the evaluation
    (:func:`_evaluate_latencies`) compares whatever metrics are measured against
    their targets, so this default reports the one path that needs no plugin.
    """
    try:
        # The ENTIRE corpus setup+load is guarded (Gemini review cycle 4): not
        # only _load_wav, but also list_items() (which reads the corpus
        # manifest.toml) and the wav_path construction. A missing/unreadable
        # manifest raises FileNotFoundError/OSError from list_items() OUTSIDE a
        # narrower guard; wrapping the whole block degrades gracefully (skip the
        # latency bench) rather than crashing the whole doctor command.
        items = corpus.list_items()
        signal = next((item for item in items if item.kind == "signal"), None)
        if signal is None:
            return {}
        wav_path = corpus.DEFAULT_CORPUS_ROOT / signal.path
        audio, sample_rate = _load_wav(wav_path)
    except (OSError, ValueError):
        # Finding 3 (Gemini review, final batch): a malformed/corrupt
        # ``manifest.toml`` makes ``list_items()`` raise ``tomllib.TOMLDecodeError``,
        # which inherits from ``ValueError`` (NOT ``OSError``) — the OSError-only
        # guard let it escape and crash doctor. Broadening to ``ValueError`` degrades
        # gracefully (skip the latency bench) on a malformed manifest too.
        return {}
    start = time.perf_counter()
    compute_summary(audio, sample_rate)
    elapsed = time.perf_counter() - start
    return {"deterministic_feature_extraction": elapsed}


def _load_wav(wav_path: Path):
    """Read a wav to (audio, sample_rate) via soundfile (mirrors the F-layer)."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def _evaluate_latencies(measured: dict[str, float]) -> tuple[LatencyResult, ...]:
    """Compare each measured latency path to its target; over-target -> warning.

    Only metrics that have a target in :data:`LATENCY_TARGETS_S` are evaluated
    (an unknown key is ignored rather than silently treated as a pass/fail).
    """
    results: list[LatencyResult] = []
    for metric, measured_s in measured.items():
        target_s = LATENCY_TARGETS_S.get(metric)
        if target_s is None:
            continue
        results.append(
            LatencyResult(
                metric=metric,
                measured_s=measured_s,
                target_s=target_s,
                over_target=measured_s > target_s,
            )
        )
    return tuple(results)


# --- orchestration ----------------------------------------------------------

# Module-level default measurer (monkeypatchable for hermetic latency tests).
_MEASURE_LATENCIES: Callable[[Path], dict[str, float]] = _measure_latencies


def run_doctor(
    *,
    repo_root: Optional[Path] = None,
    runner: Optional[Runner] = None,
    adapter: Optional[PerceptionAdapter] = None,
    measure_latencies: Optional[Callable[[Path], dict[str, float]]] = None,
    verify_script: Path = VERIFY_SURGE_SCRIPT,
    surge_manifest: Path = SURGE_MANIFEST,
) -> DoctorReport:
    """Run all environment checks + the latency benchmark -> :class:`DoctorReport`.

    Every dependency is injectable (defaulting to the module-level real
    implementations) so the CLI calls ``run_doctor()`` with no arguments while
    tests drive hermetic fakes. ``adapter`` defaults to the G1 QwenLocalAdapter
    (graceful when the model is absent).
    """
    root = repo_root or REPO_ROOT
    active_runner = runner or _RUNNER
    active_measure = measure_latencies or _MEASURE_LATENCIES
    active_adapter = adapter if adapter is not None else _default_adapter()

    checks: tuple[CheckResult, ...] = (
        _check_pins(),
        _check_lockfile(root, active_runner),
        _check_surge_xt(verify_script, surge_manifest, active_runner),
        _check_backend_load(),
        _check_perception(active_adapter),
    )
    latencies = _evaluate_latencies(active_measure(root))
    return DoctorReport(checks=checks, latencies=latencies)


def format_report(report: DoctorReport) -> str:
    """Human-readable `doctor` report for stderr (by design: human logs to
    stderr). One line per check + per measured latency path."""
    lines: list[str] = ["sonoscope doctor:"]
    symbol = {"ok": "OK  ", "warning": "WARN", "error": "FAIL"}
    for check in report.checks:
        lines.append(f"  [{symbol[check.severity]}] {check.name}: {check.detail}")
    for latency in report.latencies:
        flag = "WARN" if latency.over_target else "OK  "
        lines.append(
            f"  [{flag}] latency:{latency.metric}: "
            f"{latency.measured_s:.3f}s (target {latency.target_s:.3f}s)"
        )
    lines.append("  => " + ("OK" if report.ok else "ENVIRONMENT FAILURE"))
    return "\n".join(lines)
