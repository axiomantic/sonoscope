"""H1: `doctor` environment check + latency benchmark (design §4.5, §7.2, §11).

`doctor` verifies the environment (dependency pins, `uv.lock` drift, Surge XT
install + factory content, model-runtime health, backend load) and runs a
deterministic-only latency micro-benchmark against the §7.2 targets.

Contract under test (H1 spec):

- A hard environment fault (pin drift, tampered lockfile, missing Surge) is an
  `ENVIRONMENT` failure -> exit **5** (design §3.6). Clean env -> exit **0**.
- Perception being unavailable is NOT a failure (graceful degradation, §10.2):
  the perception check reports "ok" and does not flip the exit code.
- A latency path OVER its §7.2 target is a NON-FATAL warning (soft criterion):
  it is surfaced with the exact metric name but doctor STAYS exit 0. Slow
  hardware must never hard-fail `doctor`.

Green-mirage discipline (AGENTS.md): the lockfile-drift / missing-surge tests
are paired with the clean-pass test so a check that only ever passes cannot slip
through. The environment probes (uv, Surge verify) are driven through an injected
command runner (a fake) and a Null perception adapter so the tests are
deterministic, fast, and non-integration — they exercise the REAL doctor mapping
logic (subprocess nonzero -> error check -> exit 5) without the external
artifacts.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from sonoscope import cli, doctor
from sonoscope.perception.null_adapter import NullAdapter
from sonoscope.schema import ExitCode


# --- fakes ------------------------------------------------------------------


def _make_runner(uv_rc: int = 0, surge_rc: int = 0):
    """A fake subprocess runner keyed on the command doctor issues.

    ``uv lock --check`` returns ``uv_rc`` (nonzero == tampered/drifted lock);
    the Surge verify script (`bash verify_surge_xt.sh ...`) returns ``surge_rc``
    (nonzero == missing install / factory-content drift).
    """

    def runner(cmd, **_kwargs) -> CompletedProcess:
        if list(cmd[:3]) == ["uv", "lock", "--check"]:
            return CompletedProcess(
                cmd, uv_rc, "", "" if uv_rc == 0 else "lockfile drift"
            )
        if cmd and cmd[0] == "bash":
            return CompletedProcess(
                cmd, surge_rc, "", "" if surge_rc == 0 else "surge missing"
            )
        return CompletedProcess(cmd, 0, "", "")

    return runner


def _install(monkeypatch, *, uv_rc=0, surge_rc=0, latencies=None):
    """Point doctor's module-level defaults at hermetic fakes so a plain
    ``cli.main(["doctor"])`` exercises the real CLI -> exit-code mapping."""
    monkeypatch.setattr(doctor, "_RUNNER", _make_runner(uv_rc, surge_rc))
    monkeypatch.setattr(doctor, "_default_adapter", lambda: NullAdapter())
    monkeypatch.setattr(
        doctor, "_MEASURE_LATENCIES", lambda _repo_root: dict(latencies or {})
    )


# --- environment checks (RED->GREEN) ----------------------------------------


def test_doctor_clean_passes(monkeypatch):
    """GREEN: pins match, lockfile clean, Surge present, backend loads -> exit 0."""
    _install(monkeypatch, uv_rc=0, surge_rc=0)
    assert cli.main(["doctor"]) == 0


def test_doctor_detects_lockfile_drift(monkeypatch):
    """RED-proving: a tampered/drifted lockfile (`uv lock --check` nonzero) is an
    ENVIRONMENT failure -> exit 5. Paired with test_doctor_clean_passes (uv_rc=0
    -> exit 0) so the check provably differentiates drift from clean."""
    _install(monkeypatch, uv_rc=1, surge_rc=0)
    assert cli.main(["doctor"]) == int(ExitCode.ENVIRONMENT)


def test_doctor_detects_missing_surge(monkeypatch):
    """An uninstalled / drifted Surge XT (verify script nonzero) -> ENVIRONMENT."""
    _install(monkeypatch, uv_rc=0, surge_rc=1)
    assert cli.main(["doctor"]) == int(ExitCode.ENVIRONMENT)


def test_lockfile_check_maps_nonzero_to_error(monkeypatch):
    """Engine-level exact-equality proof: the lockfile check flips to severity
    'error' on drift and the report exit code is ENVIRONMENT."""
    report = doctor.run_doctor(
        runner=_make_runner(uv_rc=1),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: {},
    )
    lockfile = [c for c in report.checks if c.name == "lockfile"]
    assert [c.severity for c in lockfile] == ["error"]
    assert report.exit_code == int(ExitCode.ENVIRONMENT)


def test_surge_check_maps_nonzero_to_error(monkeypatch):
    """MINOR-3 engine-isolation proof (green-mirage guard): the surge_xt check
    flips to severity 'error' when the Surge verify script exits nonzero, and the
    report exit code is ENVIRONMENT. Asserted on the specific ``surge_xt`` check —
    independent of the CLI aggregate exit — so ambient pin/backend drift cannot
    make it pass for the wrong reason. The verify-script path is injected (an
    existing file) so the runner-nonzero MAPPING is exercised, not the
    missing-script branch."""
    report = doctor.run_doctor(
        runner=_make_runner(surge_rc=1),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: {},
        verify_script=Path(__file__),
    )
    surge = [c for c in report.checks if c.name == "surge_xt"]
    assert [c.severity for c in surge] == ["error"]
    assert report.exit_code == int(ExitCode.ENVIRONMENT)


def test_doctor_reports_not_installed_dep(monkeypatch):
    """RED-proving: a pinned dependency that is NOT installed raises
    ``PackageNotFoundError`` from ``importlib.metadata.version``; doctor must
    record it as drift (error severity, exit 5) and NOT crash.

    Monkeypatching ``PINNED_VERSIONS`` to a dist that is genuinely absent makes
    the real ``importlib_metadata.version`` lookup raise. Under the unguarded
    lookup this exception escaped ``_check_pins`` (and ``run_doctor``); the guard
    turns it into ordinary drift. The check detail is asserted exactly.
    """
    monkeypatch.setattr(
        doctor, "PINNED_VERSIONS", {"sonoscope-nonexistent-pkg": "1.0.0"}
    )

    pins = doctor._check_pins()
    assert pins.name == "pins"
    assert pins.severity == "error"
    assert pins.detail == (
        "dependency pin drift: sonoscope-nonexistent-pkg: installed not "
        "installed != pinned 1.0.0"
    )

    # And the full CLI-mapped report is ENVIRONMENT (exit 5) without raising.
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: {},
    )
    assert report.exit_code == int(ExitCode.ENVIRONMENT)


def test_perception_unavailable_is_not_error():
    """Graceful degradation (§10.2): a model-absent adapter is reported 'ok'
    (not an error) and does not flip the exit code."""
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: {},
    )
    perception = [c for c in report.checks if c.name == "perception"]
    assert [c.severity for c in perception] == ["ok"]


# --- latency benchmark (I2, RED->GREEN, soft criterion) ---------------------


def test_latency_flagged_when_over_target():
    """RED-proving (I2): a deterministic-analyze time ABOVE its §7.2 target emits
    a latency warning carrying the exact metric name, yet doctor STAYS exit 0
    (soft criterion, not an ENVIRONMENT failure)."""
    over = {
        "deterministic_analyze": doctor.LATENCY_TARGETS_S["deterministic_analyze"]
        + 1.0
    }
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: over,
    )
    assert [w.metric for w in report.latency_warnings] == ["deterministic_analyze"]
    assert report.exit_code == int(ExitCode.OK)


def test_latency_ok_within_target():
    """GREEN: a deterministic-analyze time UNDER its target emits no latency
    warning (and exit stays 0)."""
    under = {
        "deterministic_analyze": doctor.LATENCY_TARGETS_S["deterministic_analyze"]
        - 0.5
    }
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=lambda _r: under,
    )
    assert report.latency_warnings == ()
    assert report.exit_code == int(ExitCode.OK)


def test_doctor_latency_skipped_when_corpus_missing(monkeypatch):
    """RED-proving (Gemini cycle 3): when the corpus stimulus is not generated,
    ``_load_wav`` raises ``FileNotFoundError`` (an OSError). The latency bench
    must degrade gracefully (skip) rather than crash the whole doctor command.
    Monkeypatch ``_load_wav`` to raise OSError so the real ``_measure_latencies``
    hits the missing-corpus path; assert it returns no metrics AND the full
    doctor still runs to its normal exit code for the other checks. RED against
    the unguarded load, where the OSError escapes and crashes run_doctor."""

    def _raise_missing(_wav_path):
        raise FileNotFoundError("corpus stimulus not generated")

    monkeypatch.setattr(doctor, "_load_wav", _raise_missing)

    # The measurer itself degrades to no metrics instead of raising.
    assert doctor._measure_latencies(Path("/nonexistent")) == {}

    # And the full doctor still runs (no raise) with its normal exit code for the
    # remaining checks, simply omitting the latency metrics.
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=doctor._measure_latencies,
    )
    assert report.latencies == ()
    assert report.exit_code == int(ExitCode.OK)


def test_doctor_latency_skipped_when_corpus_manifest_missing(monkeypatch):
    """RED-proving (Gemini cycle 4): a missing/unreadable corpus ``manifest.toml``
    makes ``corpus.list_items()`` raise ``FileNotFoundError`` (an OSError). That
    call (plus the signal lookup + ``wav_path`` construction) ran OUTSIDE the
    cycle-3 guard, which wrapped only ``_load_wav`` — so the raw OSError escaped
    and crashed the whole doctor command. Broadening the guard to wrap the entire
    setup+load degrades gracefully (skip the latency bench). RED against the code
    where ``list_items()`` is outside the try.

    Paired with ``test_doctor_latency_skipped_when_corpus_missing`` (the _load_wav
    branch) so BOTH corpus-absent paths are proven, not just one."""

    def _raise_missing():
        raise FileNotFoundError("corpus manifest.toml not found")

    monkeypatch.setattr(doctor.corpus, "list_items", _raise_missing)

    # The measurer itself degrades to no metrics instead of raising.
    assert doctor._measure_latencies(Path("/nonexistent")) == {}

    # And the full doctor still runs (no raise) with its normal exit code for the
    # remaining checks, simply omitting the latency metrics.
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=doctor._measure_latencies,
    )
    assert report.latencies == ()
    assert report.exit_code == int(ExitCode.OK)


def test_doctor_latency_skipped_when_corpus_manifest_malformed(monkeypatch):
    """RED-proving (Gemini review, final batch — Finding 3): a MALFORMED/corrupt
    corpus ``manifest.toml`` makes ``corpus.list_items()`` raise
    ``tomllib.TOMLDecodeError``, which inherits from ``ValueError`` (NOT
    ``OSError``). The prior ``except OSError`` guard let it escape and crash the
    whole doctor command. Broadening to ``except (OSError, ValueError)`` degrades
    gracefully (skip the latency bench). RED against the ``except OSError`` alone,
    where the ValueError escapes ``_measure_latencies``.

    A plain ``ValueError`` simulates ``TOMLDecodeError`` (its base class) so the
    test needs no on-disk corrupt manifest."""

    def _raise_malformed():
        raise ValueError("Invalid value (at line 1, column 1)")

    monkeypatch.setattr(doctor.corpus, "list_items", _raise_malformed)

    # The measurer itself degrades to no metrics instead of raising.
    assert doctor._measure_latencies(Path("/nonexistent")) == {}

    # And the full doctor still runs (no raise) with its normal exit code for the
    # remaining checks, simply omitting the latency metrics.
    report = doctor.run_doctor(
        runner=_make_runner(0, 0),
        adapter=NullAdapter(),
        measure_latencies=doctor._measure_latencies,
    )
    assert report.latencies == ()
    assert report.exit_code == int(ExitCode.OK)


def test_latency_bench_warms_up_before_timing(monkeypatch):
    """RED-proving: the reported latency must be the WARM steady state, not a cold
    first call. Measured here: the first ``compute_summary`` in a process costs
    ~2.2 s (librosa/scipy/sklearn import + numba JIT) against a 0.500 s target,
    while every later call costs ~0.03 s — so an unwarmed bench warns on every
    clean machine.

    Asserting on wall-clock seconds would be flaky and would not pin the
    behaviour, so the warm-up is made observable instead: a fake
    ``compute_summary`` counts its invocations. Exactly two calls (one warm-up +
    one timed) is the fix; RED against the unwarmed code, which calls it once."""
    calls: list[tuple[int, int]] = []

    def _fake_compute_summary(audio, sample_rate):
        calls.append((len(audio), sample_rate))
        return None

    monkeypatch.setattr(doctor, "compute_summary", _fake_compute_summary)
    monkeypatch.setattr(
        doctor.corpus,
        "list_items",
        lambda: (SimpleNamespace(kind="signal", path="signals/impulse_2s.wav"),),
    )
    monkeypatch.setattr(doctor, "_load_wav", lambda _wav_path: ([0.0, 0.0], 48000))

    measured = doctor._measure_latencies(Path("/nonexistent"))

    assert len(calls) == 2
    assert calls == [(2, 48000), (2, 48000)]
    assert list(measured) == ["deterministic_feature_extraction"]
