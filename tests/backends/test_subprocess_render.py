"""Tests for the E4 subprocess render wrapper (design §7.1 M6/S1).

NON-integration: these exercise the real ``multiprocessing`` spawn machinery but
use **FAKE** backends only — no Surge XT, no pedalboard. The fake backends and
their helpers are defined at MODULE TOP LEVEL because the ``"spawn"`` start
method re-imports this module in the child and unpickles the fake by qualified
name; a closure or a locally-defined class would not be spawn-picklable.

Coverage (E4 spec):

- ``test_render_via_subprocess_returns_meta`` — a fake renders in the child; the
  parent gets a ``RenderMeta`` matching the wav written to disk.
- ``test_crash_becomes_render_error`` — an in-worker raise surfaces as a mapped
  ``RENDER`` error (exact code/component), **never** a silent success.
- ``test_transient_init_crash_retried_once`` — a fail-once-then-succeed fake
  succeeds on the single retry; a fail-twice fake hard-errors; the retry happens
  exactly once in both cases.
- ``test_post_init_crash_not_retried`` — a crash AFTER init (sentinel present) is
  a hard ``RENDER`` error with NO retry, proving the init/post-init distinction.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import soundfile as sf

from sonoscope.backends import subprocess_render
from sonoscope.backends.base import RenderMeta, RenderRequest, RenderResult
from sonoscope.backends.subprocess_render import render_in_subprocess
from sonoscope.errors import RenderError
from sonoscope.schema import ExitCode

# Deterministic fake-render shape (chosen so the wav content — hence its sha256 —
# is reproducible across the fresh spawn interpreter).
_FAKE_SR = 48000
_FAKE_BLOCK = 512
_FAKE_FRAMES = _FAKE_SR // 10  # 0.1 s
_FAKE_CHANNELS = 1
_FAKE_WAV_SUBTYPE = "PCM_F32"
_SOUNDFILE_SUBTYPE = "FLOAT"

#: Distinct temp-dir prefix for the Finding-1 owned-dir fake, so the leak probe
#: only ever counts dirs THIS test created (never a real backend's).
_OWNED_PREFIX = "sonoscope-test-owneddir-"


# --- spawn-picklable fake backends + helpers (module top-level) --------------


def _bump_counter(counter_path: str) -> int:
    """Increment a file-backed attempt counter and return the new value.

    File-backed because the ``"spawn"`` child is a fresh interpreter — in-memory
    state does not survive. Attempts are sequential (the parent joins each child
    before the next), so this read-modify-write is race-free.
    """
    path = Path(counter_path)
    current = int(path.read_text()) if path.exists() else 0
    current += 1
    path.write_text(str(current))
    return current


def _fake_wav_bytes() -> bytes:
    """Deterministic non-silent mono float32 samples (bytes, for reference)."""
    import numpy as np

    return ((np.arange(_FAKE_FRAMES, dtype=np.float32) % 7) * 0.01).tobytes()


def _fake_render(wav_dir: str, req: RenderRequest) -> RenderResult:
    """Write a real deterministic wav + return a ``RenderMeta`` matching it."""
    import uuid

    import numpy as np

    data = (np.arange(_FAKE_FRAMES, dtype=np.float32) % 7) * 0.01
    wav_path = Path(wav_dir) / f"fake-{uuid.uuid4().hex}.wav"
    sf.write(str(wav_path), data, _FAKE_SR, subtype=_SOUNDFILE_SUBTYPE)
    wav_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    meta = RenderMeta(
        sample_rate_hz=req.sample_rate_hz,
        block_size=req.block_size,
        channels=_FAKE_CHANNELS,
        duration_s=_FAKE_FRAMES / _FAKE_SR,
        wav_subtype=_FAKE_WAV_SUBTYPE,
        wav_sha256=wav_sha256,
        render_wall_ms=0,
        warnings=[],
    )
    return RenderResult(wav_path=wav_path, render_meta=meta, warnings=[])


class _EchoBackend:
    """Renders a deterministic wav in the child; the happy path."""

    id = "fake-echo"
    version = "1"

    def __init__(self, wav_dir: str) -> None:
        self.wav_dir = str(wav_dir)

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def render(self, req: RenderRequest) -> RenderResult:
        return _fake_render(self.wav_dir, req)


class _RaiseBackend:
    """Raises a mapped ``RenderError`` inside the worker (caught → error meta)."""

    id = "fake-raise"
    version = "1"
    CODE = "FAKE_RENDER_BOOM"

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def render(self, req: RenderRequest) -> RenderResult:
        raise RenderError(
            self.CODE,
            "fake worker exploded",
            detail={"why": "test"},
            component="render",
        )


class _CrashOnInitBackend:
    """Hard-crashes the child DURING spawn-unpickle (subprocess INIT) for the
    first ``fail_times`` attempts, then renders successfully.

    ``__setstate__`` runs while the spawn child unpickles the worker args —
    strictly before the worker body writes its init sentinel — so a crash here
    is a *transient subprocess-init* crash (the loader-race analogue, S1).
    """

    id = "fake-crash-init"
    version = "1"

    def __init__(self, counter_path: str, fail_times: int, wav_dir: str) -> None:
        self.counter_path = str(counter_path)
        self.fail_times = int(fail_times)
        self.wav_dir = str(wav_dir)

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        attempt = _bump_counter(self.counter_path)
        if attempt <= self.fail_times:
            os._exit(70)  # die during INIT, before the worker writes its sentinel

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def render(self, req: RenderRequest) -> RenderResult:
        return _fake_render(self.wav_dir, req)


class _CrashInRenderBackend:
    """Hard-crashes INSIDE ``render`` — i.e. AFTER the worker wrote its init
    sentinel — modelling a post-init crash (hard error, no retry, S1)."""

    id = "fake-crash-render"
    version = "1"

    def __init__(self, counter_path: str) -> None:
        self.counter_path = str(counter_path)

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        _bump_counter(self.counter_path)  # count spawns to prove NO retry

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def render(self, req: RenderRequest) -> RenderResult:
        os._exit(70)  # post-init crash: sentinel already written by the worker


class _TrivialBackend:
    """Spawns, lets the worker write its init sentinel, returns a minimal
    ``RenderResult``, and exits 0.

    Used by the MINOR-3 defensive-branch tests, which monkeypatch the PARENT's
    meta reader to inject the payload under test — so the child's own (valid)
    meta is irrelevant. The child must merely spawn cleanly and exit 0 so the
    parent reaches its ``exitcode == 0`` discrimination branch with the init
    sentinel present (proving these are hard errors, not retried init crashes).
    """

    id = "fake-trivial"
    version = "1"

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def render(self, req: RenderRequest) -> RenderResult:
        return RenderResult(
            wav_path=Path("/nonexistent-trivial.wav"),
            render_meta=RenderMeta(
                sample_rate_hz=req.sample_rate_hz,
                block_size=req.block_size,
                channels=req.channels,
                duration_s=0.0,
                wav_subtype=_FAKE_WAV_SUBTYPE,
                wav_sha256="0" * 64,
                render_wall_ms=0,
                warnings=[],
            ),
            warnings=[],
        )


class _OwnedTempDirBackend:
    """Faithful spawn-picklable model of ``PedalboardVST3Backend``'s no-caller-
    ``render_dir`` path (Finding 1, cycle 2).

    Mirrors the real backend's instance-owned temp-dir mechanism exactly: with
    ``_render_dir`` unset it LAZILY ``mkdtemp``s ONE reused ``_owned_temp_dir``
    (via :meth:`_ensure_owned_temp_dir`) and writes every render there;
    :meth:`close` reclaims it. Under spawn isolation ``render`` runs in the child,
    so the pre-fix code creates that dir CHILD-side and orphans it on every
    render (the parent's pickled copy keeps ``_owned_temp_dir is None``, so no
    ``close()`` can reclaim it). The Finding-1 fix hoists creation to the parent.
    """

    id = "fake-owned-tempdir"
    version = "1"

    def __init__(self) -> None:
        self._render_dir = None
        self._owned_temp_dir = None

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def probe(self, plugin_path: Path):  # pragma: no cover - unused by E4
        raise NotImplementedError

    def _ensure_owned_temp_dir(self) -> Path:
        if self._owned_temp_dir is None:
            self._owned_temp_dir = tempfile.mkdtemp(prefix=_OWNED_PREFIX)
        return Path(self._owned_temp_dir)

    def close(self) -> None:
        if self._owned_temp_dir is not None:
            shutil.rmtree(self._owned_temp_dir, ignore_errors=True)
            self._owned_temp_dir = None

    def render(self, req: RenderRequest) -> RenderResult:
        out_dir = (
            Path(self._render_dir)
            if self._render_dir is not None
            else self._ensure_owned_temp_dir()
        )
        return _fake_render(str(out_dir), req)


def _req(sample_rate_hz: int = _FAKE_SR) -> RenderRequest:
    """A minimal spawn-picklable request; the fakes synthesize their own wav."""
    return RenderRequest(
        plugin_path=Path("/nonexistent.vst3"),
        plugin_format="vst3",
        stimulus=None,
        param_set=None,
        sample_rate_hz=sample_rate_hz,
        block_size=_FAKE_BLOCK,
        channels=_FAKE_CHANNELS,
    )


# --- acceptance tests --------------------------------------------------------


def test_render_via_subprocess_returns_meta(tmp_path: Path) -> None:
    """A fake backend renders in a spawn child; the parent gets a RenderMeta
    that exactly matches the wav written to disk."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()

    result = render_in_subprocess(_EchoBackend(str(wav_dir)), _req())

    assert isinstance(result, RenderResult)
    assert result.wav_path.exists()
    wav_sha256 = hashlib.sha256(result.wav_path.read_bytes()).hexdigest()
    expected_meta = RenderMeta(
        sample_rate_hz=_FAKE_SR,
        block_size=_FAKE_BLOCK,
        channels=_FAKE_CHANNELS,
        duration_s=_FAKE_FRAMES / _FAKE_SR,
        wav_subtype=_FAKE_WAV_SUBTYPE,
        wav_sha256=wav_sha256,
        render_wall_ms=0,
        warnings=[],
    )
    assert result.render_meta == expected_meta


def test_crash_becomes_render_error(tmp_path: Path) -> None:
    """An in-worker raise surfaces as a mapped RENDER error — NEVER a silent
    success (absence of outcome:"success" is always an error)."""
    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_RaiseBackend(), _req())

    err = excinfo.value
    assert err.code == "FAKE_RENDER_BOOM"
    assert err.component == "render"
    assert err.detail == {"why": "test"}
    assert err.exit_code == ExitCode.RENDER


def test_transient_init_crash_retried_once(tmp_path: Path) -> None:
    """One bounded retry on a transient subprocess-init crash: fail-once-then-
    succeed succeeds on the retry; fail-twice hard-errors. The retry happens
    exactly once (two child spawns) in both cases."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()

    # Fail once, then succeed on the single retry.
    counter_ok = tmp_path / "counter_ok.txt"
    result = render_in_subprocess(
        _CrashOnInitBackend(str(counter_ok), 1, str(wav_dir)), _req()
    )
    assert isinstance(result, RenderResult)
    assert result.wav_path.exists()
    assert int(counter_ok.read_text()) == 2  # 1 crash + 1 successful retry

    # Fail twice -> hard RENDER error; the retry still happens exactly once.
    counter_fail = tmp_path / "counter_fail.txt"
    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(
            _CrashOnInitBackend(str(counter_fail), 2, str(wav_dir)), _req()
        )
    err = excinfo.value
    assert err.code == "RENDER_SUBPROCESS_CRASH"
    assert err.component == "render"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail["retries"] == 1
    assert int(counter_fail.read_text()) == 2  # exactly one retry, not two


def test_post_init_crash_not_retried(tmp_path: Path) -> None:
    """A crash AFTER init (the worker wrote its sentinel) is a hard RENDER error
    with NO retry — the init/post-init discriminator at work (S1)."""
    counter = tmp_path / "counter_post.txt"
    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_CrashInRenderBackend(str(counter)), _req())

    err = excinfo.value
    assert err.code == "RENDER_SUBPROCESS_CRASH"
    assert err.component == "render"
    assert err.detail["retries"] == 0
    assert err.detail["init_ok"] is True
    assert int(counter.read_text()) == 1  # a single spawn: NO retry after init


# --- MINOR-3: defensive-branch coverage --------------------------------------
# These exercise the parent's discrimination branches by monkeypatching the
# PARENT-side meta reader (``_read_meta``) to inject the payload under test.
# Spawn-safe: the child is a fresh interpreter that re-imports this module
# unpatched and never calls ``_read_meta`` — only the parent does. The child
# (``_TrivialBackend``) still spawns and exits 0 with its init sentinel present,
# so every case below is judged a HARD error (no init-crash retry).


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param({"outcome": "weird"}, id="unknown-outcome"),
        pytest.param({}, id="absent-outcome"),
        pytest.param(None, id="unparseable-or-missing-meta"),
    ],
)
def test_bad_meta_becomes_crash(tmp_path: Path, monkeypatch, meta) -> None:
    """An unknown/absent ``outcome`` or an unparseable/non-dict meta (``None``
    from ``_read_meta``) is never a silent success — it maps to a hard
    RENDER_SUBPROCESS_CRASH (design §7.1: the ONLY return path is
    outcome=="success")."""
    monkeypatch.setattr(subprocess_render, "_read_meta", lambda _p: meta)

    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_TrivialBackend(), _req())

    err = excinfo.value
    assert err.code == "RENDER_SUBPROCESS_CRASH"
    assert err.component == "render"
    assert err.exit_code == ExitCode.RENDER


@pytest.mark.parametrize(
    ("content", "id_"),
    [
        ("this is not json {{{", "unparseable-bytes"),
        ("[1, 2, 3]", "non-dict-list"),
        ("42", "non-dict-scalar"),
    ],
)
def test_read_meta_rejects_unparseable_or_non_dict(
    tmp_path: Path, content: str, id_: str
) -> None:
    """``_read_meta`` maps unparseable JSON, a non-dict JSON value, and a missing
    file all to ``None`` (which the parent then treats as a crash) — never a
    partially-usable dict."""
    meta_path = tmp_path / f"meta-{id_}.json"
    meta_path.write_text(content)
    assert subprocess_render._read_meta(meta_path) is None

    missing = tmp_path / "absent-meta.json"
    assert subprocess_render._read_meta(missing) is None


def test_malformed_error_payload_becomes_render_error(
    tmp_path: Path, monkeypatch
) -> None:
    """An ``outcome:"error"`` payload missing its ``code`` still maps to a
    RenderError (RENDER_WORKER_EXCEPTION) — never a silent success."""
    monkeypatch.setattr(
        subprocess_render,
        "_read_meta",
        lambda _p: {"outcome": "error", "error": {"message": "no code here"}},
    )

    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_TrivialBackend(), _req())

    err = excinfo.value
    assert err.code == "RENDER_WORKER_EXCEPTION"
    assert err.component == "render"
    assert err.exit_code == ExitCode.RENDER


def test_out_of_literal_component_falls_back_to_render(monkeypatch) -> None:
    """Finding 1 (Gemini review, final batch): a worker ``outcome:"error"`` payload
    whose ``component`` is NOT one of the strict schema ``Component`` literals must
    still map to a RenderError whose ``to_fatal_error()`` SUCCEEDS — the parent
    coerces the bogus component to the ``render`` fallback.

    RED against the unvalidated code: the bogus component reached
    ``FatalErrorDetail.component`` (a strict Literal), so ``to_fatal_error()`` raised
    a Pydantic ``ValidationError`` and crashed fatal-error reporting."""
    monkeypatch.setattr(
        subprocess_render,
        "_read_meta",
        lambda _p: {
            "outcome": "error",
            "error": {
                "code": "RENDER_WORKER_EXCEPTION",
                "message": "worker failed with a bogus component",
                "component": "bogus",
            },
        },
    )

    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_TrivialBackend(), _req())

    err = excinfo.value
    # The out-of-literal component is coerced to the valid ``render`` fallback.
    assert err.component == "render"
    assert err.exit_code == ExitCode.RENDER
    # And the fatal-error envelope construction SUCCEEDS (no Pydantic
    # ValidationError) with the fallback component.
    fatal = err.to_fatal_error(
        sonoscope_version="0.0.0-test", generated_at="2026-07-06T00:00:00Z"
    )
    assert fatal.error.component == "render"


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param({"outcome": "success"}, id="missing-both"),
        pytest.param(
            {"outcome": "success", "wav_path": "/some/out.wav"},
            id="missing-render-meta",
        ),
        pytest.param(
            {"outcome": "success", "wav_path": "/some/out.wav", "render_meta": {}},
            id="empty-render-meta-fields",
        ),
        pytest.param(
            {
                "outcome": "success",
                "wav_path": "/some/out.wav",
                "render_meta": {"bogus": 1},
            },
            id="wrong-render-meta-fields",
        ),
    ],
)
def test_malformed_success_payload_becomes_render_error(
    tmp_path: Path, monkeypatch, meta
) -> None:
    """MINOR-1 (RED against pre-fix): a malformed ``outcome:"success"`` payload
    (missing ``wav_path``/``render_meta`` or a ``render_meta`` that does not fit
    RenderMeta) must map to a RenderError with ``reason:"malformed_success"`` —
    NOT a raw KeyError/TypeError out of ``render_in_subprocess`` and NOT a silent
    success. Pre-fix, ``_build_success_result`` raised a bare KeyError/TypeError,
    so ``pytest.raises(RenderError)`` would not match -> RED."""
    monkeypatch.setattr(subprocess_render, "_read_meta", lambda _p: meta)

    with pytest.raises(RenderError) as excinfo:
        render_in_subprocess(_TrivialBackend(), _req())

    err = excinfo.value
    assert err.code == "RENDER_SUBPROCESS_CRASH"
    assert err.component == "render"
    assert err.exit_code == ExitCode.RENDER
    assert err.detail["reason"] == "malformed_success"


# --- Finding 1 (cycle 2): no-render_dir backend must not orphan child temp dirs -


def test_no_render_dir_backend_owns_one_reusable_dir_and_close_reclaims() -> None:
    """A backend with NO caller ``render_dir`` renders across the spawn boundary
    without orphaning a per-render temp dir (Finding 1, cycle 2).

    Proves BOTH:

    (a) the parent gets valid ``RenderResult``s whose wavs SURVIVE for the
        consumer to read after the child exits (the wav lives in an owned dir the
        child never deletes — the parent→child wav contract is intact);
    (b) every render REUSES the SAME single instance-owned dir (no per-render
        leak) and ``backend.close()`` reclaims it — leaving nothing on disk.

    RED against the pre-fix ``render_in_subprocess``: each spawn child lazily
    ``mkdtemp``s its OWN owned dir, so the two renders land in DIFFERENT dirs, two
    dirs leak, and ``close()`` on the parent (whose ``_owned_temp_dir`` stayed
    ``None``) reclaims neither."""
    tmp_root = Path(tempfile.gettempdir())

    def owned_dirs() -> set[Path]:
        return set(tmp_root.glob(_OWNED_PREFIX + "*"))

    before = owned_dirs()
    backend = _OwnedTempDirBackend()
    try:
        first = render_in_subprocess(backend, _req())
        second = render_in_subprocess(backend, _req())

        # (a) both renders succeeded and their wavs survive for the consumer.
        assert isinstance(first, RenderResult)
        assert isinstance(second, RenderResult)
        assert first.wav_path.exists()
        assert second.wav_path.exists()

        # (b) both wavs land in the SAME single instance-owned dir (reused, not a
        # fresh orphan per render): exactly ONE new owned dir for the backend.
        assert first.wav_path.parent == second.wav_path.parent
        new_dirs = owned_dirs() - before
        assert new_dirs == {first.wav_path.parent}
    finally:
        backend.close()

    # (b) close() reclaims the single owned dir: no leak once the caller closes.
    assert owned_dirs() - before == set()
