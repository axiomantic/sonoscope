"""E4 subprocess render wrapper — spawn-isolated render + one-retry crash policy.

By design (M6/S1): every ``backend.render`` runs in a fresh child spawned via
``multiprocessing`` with the ``"spawn"`` start method (never ``fork`` — no
inherited JUCE/GUI thread state). Plugins can crash the interpreter or leak JUCE
state; a fresh process per render gives crash isolation and a clean plugin
lifecycle, and the child's exit frees the plugin/JUCE. A crash surfaces as a
structured ``RENDER`` error — **never** a silent green.

Parent <-> child handoff (M6): the child writes the wav to disk (already done by
the backend) plus a small ``meta.json`` whose top-level ``outcome`` discriminator
removes all ambiguity between "the child reported a failure" and "the child
died":

- ``{"outcome": "success", "wav_path": str, "render_meta": {...RenderMeta...}}``
- ``{"outcome": "error", "error": {code, message, detail, severity, component}}``

The parent DISCRIMINATES on ``outcome``: ``success`` -> build ``RenderResult``;
``error`` -> raise the mapped ``RenderError`` (C5). A **missing/unparseable
meta.json, an unknown/absent outcome, OR a nonzero/signalled child exit** is
treated as a crash. A silent success is therefore impossible: the ONLY path that
returns a ``RenderResult`` is an ``outcome == "success"`` payload from a
clean-exiting child.

Retry policy (S1): exactly ONE bounded retry on a *transient subprocess-init*
crash. The worker writes an **init sentinel** the instant it begins executing
(spawn + unpickle succeeded). A crash with NO sentinel means the child died
during init/bringup (the loader-race analogue) -> transient -> retry once. A
crash WITH the sentinel present means the child died during the render itself ->
post-init -> hard ``RENDER`` error, no retry.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Optional, get_args

from sonoscope.backends.base import RenderBackend, RenderMeta, RenderRequest, RenderResult
from sonoscope.errors import RenderError, SonoscopeError
from sonoscope.schema import Component

_COMPONENT = "render"

#: The canonical set of valid ``Component`` literals, derived from the schema's
#: single source of truth (``get_args(Component)``) rather than hardcoded, so a
#: component read off an untrusted worker payload can be validated before it
#: reaches the strict ``FatalErrorDetail.component`` literal (Gemini review,
#: final batch, Finding 1).
_VALID_COMPONENTS: frozenset[str] = frozenset(get_args(Component))

#: ``meta.json`` outcome discriminator values (M6 handoff schema).
OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"

# Error codes (`error.code`; component == "render").
#: Hard crash: missing/unparseable meta, unknown outcome, or nonzero/signalled exit.
RENDER_SUBPROCESS_CRASH = "RENDER_SUBPROCESS_CRASH"
#: A non-``SonoscopeError`` raised inside the worker, translated at the boundary.
RENDER_WORKER_EXCEPTION = "RENDER_WORKER_EXCEPTION"

#: One initial attempt + one bounded retry (S1: exactly one retry on init crash).
MAX_RENDER_ATTEMPTS = 2

_META_FILENAME = "meta.json"
_SENTINEL_FILENAME = "init-ok"

#: Sentinel for "the backend has no ``_render_dir`` attribute at all" (e.g. a
#: v2 backend that manages its own output), distinct from ``_render_dir is None``
#: ("this backend owns a lazily-created temp dir"). Used by
#: :func:`_ensure_parent_owned_render_dir` (Finding 1, cycle 2).
_RENDER_DIR_UNSET = object()


# --- child worker ------------------------------------------------------------


def _worker(
    backend: RenderBackend,
    req: RenderRequest,
    meta_path: str,
    sentinel_path: str,
) -> None:
    """Child entry point (spawn target): run ``backend.render`` and hand off.

    Reaching this function means spawn + unpickle (subprocess INIT) succeeded, so
    the FIRST action is to write the init sentinel — this is what lets the parent
    distinguish an init crash (no sentinel) from a post-init crash (sentinel
    present). Any caught failure is serialized to an ``outcome:"error"`` payload;
    a hard crash writes nothing and the parent detects it via the missing meta /
    nonzero exit.
    """
    # Init sentinel: the child successfully initialized and is about to render.
    try:
        Path(sentinel_path).write_text("ok")
    except OSError:
        # A sentinel-write failure is harmless: the worst case is the parent
        # treats a later crash as an init crash and retries once. Never fatal.
        pass

    try:
        result = backend.render(req)
    except SonoscopeError as exc:
        # Caught, already-mapped in-worker failure: preserve its exact contract.
        _write_meta(
            meta_path,
            {
                "outcome": OUTCOME_ERROR,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail,
                    "severity": "error",
                    "component": exc.component,
                },
            },
        )
        return
    except Exception as exc:  # noqa: BLE001 - boundary: translate any in-worker fault
        # Any other in-worker fault is translated to a structured render error at
        # the process boundary (mirrors PedalboardVST3Backend._load) so the parent
        # re-raises it rather than seeing an opaque crash.
        _write_meta(
            meta_path,
            {
                "outcome": OUTCOME_ERROR,
                "error": {
                    "code": RENDER_WORKER_EXCEPTION,
                    "message": f"{type(exc).__name__}: {exc}",
                    "detail": {"exception_type": type(exc).__name__},
                    "severity": "error",
                    "component": _COMPONENT,
                },
            },
        )
        return

    _write_meta(
        meta_path,
        {
            "outcome": OUTCOME_SUCCESS,
            "wav_path": str(result.wav_path),
            "render_meta": dataclasses.asdict(result.render_meta),
        },
    )


def _write_meta(meta_path: str, payload: dict) -> None:
    """Atomically write the handoff ``meta.json`` (temp + rename).

    The rename is atomic within the directory, so the parent — which reads only
    after the child has exited — never observes a half-written payload.

    MINOR-2: serialization must never crash the handoff. ``exc.detail`` is a
    typed ``Optional[dict]`` that may hold non-JSON-serializable values; a raw
    ``TypeError`` from :func:`json.dumps` inside the worker's ``except`` handler
    would kill the child with the sentinel present, so the parent would report a
    generic crash and the structured ``code`` would be LOST. ``default=str``
    coerces any non-serializable value to its ``str`` so the structured payload
    (and its ``code``) always round-trips.
    """
    path = Path(meta_path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, path)


# --- parent side -------------------------------------------------------------


def render_in_subprocess(
    backend: RenderBackend, req: RenderRequest
) -> RenderResult:
    """Run ``backend.render(req)`` in a fresh spawn child; return its result.

    Discriminates the M6 handoff on ``outcome`` and applies the S1 one-retry
    policy. Raises a mapped :class:`RenderError` on a caught in-worker failure or
    a hard crash; never returns without an explicit ``outcome:"success"``.
    """
    ctx = multiprocessing.get_context("spawn")

    # Finding 1 (cycle 2): hoist the backend's instance-owned render-dir creation
    # to the PARENT before spawn. A no-caller-``render_dir`` backend otherwise
    # creates that dir LAZILY inside ``backend.render`` — which runs in the spawn
    # CHILD, so the dir is created child-side and the parent's (pickled) backend
    # copy never learns its path. Every render then orphans a fresh temp dir that
    # no ``backend.close()`` can reclaim (an unbounded disk leak across iterative
    # QA). Creating it here, on the parent's backend, means the ONE instance-owned
    # dir is inherited (and reused) by every render child, survives for the
    # parent/consumer to read the wav, and is reclaimable by ``backend.close()``.
    # ``_worker`` therefore MUST NOT ``close()`` the backend: the wav lives inside
    # that owned dir and the consumer reads it after the child exits.
    _ensure_parent_owned_render_dir(backend)

    with tempfile.TemporaryDirectory(prefix="sonoscope-subrender-") as handoff:
        meta_path = Path(handoff) / _META_FILENAME
        sentinel_path = Path(handoff) / _SENTINEL_FILENAME

        attempts = 0
        while True:
            attempts += 1
            # Clear any stale handoff artifacts from a prior attempt so this
            # attempt's crash/success is judged on its own outputs.
            _unlink_quiet(meta_path)
            _unlink_quiet(sentinel_path)

            proc = ctx.Process(
                target=_worker,
                args=(backend, req, str(meta_path), str(sentinel_path)),
            )
            proc.start()
            proc.join()
            exitcode = proc.exitcode

            meta = _read_meta(meta_path)

            # A clean-exiting child with a valid handoff: discriminate on outcome.
            if exitcode == 0 and meta is not None:
                outcome = meta.get("outcome")
                if outcome == OUTCOME_SUCCESS:
                    return _build_success_result(meta)
                if outcome == OUTCOME_ERROR:
                    raise _error_from_meta(meta.get("error"))
                # Unknown/absent outcome -> not a success -> treat as a crash.

            # Crash path: missing/unparseable meta, unknown outcome, or a
            # nonzero/signalled exit. The init sentinel distinguishes a transient
            # init crash (retryable once) from a post-init crash (hard).
            init_ok = sentinel_path.exists()
            if not init_ok and attempts < MAX_RENDER_ATTEMPTS:
                continue  # transient subprocess-init crash: one bounded retry (S1)

            raise _crash_error(
                exitcode=exitcode,
                retries=attempts - 1,
                init_ok=init_ok,
                meta_present=meta is not None,
            )


def _ensure_parent_owned_render_dir(backend: RenderBackend) -> None:
    """Pre-create the backend's instance-owned render dir on the PARENT (Finding 1).

    Only acts when the backend exposes a ``_render_dir`` attribute that is
    ``None`` (the "no caller render_dir; I own a lazily-created temp dir" state)
    AND a callable ``_ensure_owned_temp_dir``. Calling it here binds the ONE
    owned dir to the parent's backend object so it is:

    - inherited by every spawn child via pickle (each child REUSES it instead of
      mkdtemp-ing — and orphaning — a fresh dir per render);
    - alive for the parent/consumer to read the wav after the child exits;
    - reclaimable by ``backend.close()`` (its path lives on the parent object).

    A caller-supplied ``render_dir`` (``_render_dir is not None``) owns the wav
    lifecycle already, and a backend without the attribute (a v2 backend that
    manages its own output) is left untouched — both are no-ops here.
    """
    if getattr(backend, "_render_dir", _RENDER_DIR_UNSET) is not None:
        return  # caller-supplied render_dir already owns the wav lifecycle
    ensure = getattr(backend, "_ensure_owned_temp_dir", None)
    if not callable(ensure):
        return  # backend manages its own output; nothing to hoist
    try:
        ensure()
    except OSError:
        # Parent-side pre-creation is a leak-avoidance hoist, not a new failure
        # surface: on a transient FS error fall back to the child's lazy creation
        # (which maps its own render errors), never changing the
        # RenderResult/RenderError contract for this edge.
        pass


def _read_meta(meta_path: Path) -> Optional[dict]:
    """Read + parse the handoff ``meta.json``; ``None`` if missing/unparseable."""
    if not meta_path.exists():
        return None
    try:
        parsed = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_success_result(meta: dict) -> RenderResult:
    """Rebuild the ``RenderResult`` from an ``outcome:"success"`` payload.

    MINOR-1: defensive (mirrors :func:`_error_from_meta`). A success payload
    missing ``wav_path``/``render_meta``, or carrying a ``render_meta`` that does
    not fit :class:`RenderMeta`, must still surface as a mapped
    :class:`RenderError` — never a raw ``KeyError``/``TypeError`` escaping
    :func:`render_in_subprocess`. The "every failure is a mapped RenderError"
    contract holds even for a garbled success payload.
    """
    wav_path = meta.get("wav_path")
    render_meta_fields = meta.get("render_meta")
    if not isinstance(wav_path, str) or not isinstance(render_meta_fields, dict):
        raise _malformed_success_error(meta)
    try:
        render_meta = RenderMeta(**render_meta_fields)
        warnings = list(render_meta.warnings)
    except (TypeError, ValueError) as exc:
        raise _malformed_success_error(meta, cause=exc) from exc
    return RenderResult(
        wav_path=Path(wav_path),
        render_meta=render_meta,
        warnings=warnings,
    )


def _malformed_success_error(
    meta: dict, *, cause: Optional[Exception] = None
) -> RenderError:
    """Map a garbled ``outcome:"success"`` payload to a crash ``RenderError``.

    MINOR-1: a success payload the parent cannot rebuild into a
    ``RenderResult`` is a corrupted handoff, reported with
    ``reason:"malformed_success"`` (mirrors the :func:`_crash_error` family's
    ``reason`` vocabulary) rather than a raw exception or a silent green.
    """
    detail: dict[str, Any] = {"reason": "malformed_success"}
    if cause is not None:
        detail["cause"] = f"{type(cause).__name__}: {cause}"
    return RenderError(
        RENDER_SUBPROCESS_CRASH,
        "render subprocess reported outcome=success with a malformed payload",
        detail=detail,
        component=_COMPONENT,
    )


def _error_from_meta(error: Any) -> RenderError:
    """Map an ``outcome:"error"`` payload back to a :class:`RenderError` (C5)."""
    if not isinstance(error, dict) or not error.get("code") or not error.get("message"):
        # A malformed error payload must still never be a silent success.
        return RenderError(
            RENDER_WORKER_EXCEPTION,
            "render subprocess reported outcome=error with a malformed payload",
            detail={"raw": error},
            component=_COMPONENT,
        )
    # Finding 1 (final batch): the worker's ``component`` is untrusted. A value
    # outside the strict schema ``Component`` literals constructs ``RenderError``
    # fine but would later raise a Pydantic ``ValidationError`` in
    # ``to_fatal_error()`` (``FatalErrorDetail.component`` is a strict Literal),
    # crashing fatal-error reporting. Coerce any out-of-literal component back to
    # the module's ``render`` component so the fatal envelope always renders.
    component = error.get("component")
    if component not in _VALID_COMPONENTS:
        component = _COMPONENT
    return RenderError(
        str(error["code"]),
        str(error["message"]),
        detail=error.get("detail"),
        component=component,
    )


def _crash_error(
    *, exitcode: Optional[int], retries: int, init_ok: bool, meta_present: bool
) -> RenderError:
    """Build the structured hard-crash ``RENDER`` error (by design)."""
    signalled = exitcode is not None and exitcode < 0
    signal_name: Optional[str] = None
    if signalled:
        assert exitcode is not None
        try:
            signal_name = signal.Signals(-exitcode).name
        except ValueError:
            signal_name = f"SIG{-exitcode}"

    if signalled:
        desc = f"exited with signal {signal_name}"
        reason = "signalled"
    elif exitcode:
        desc = f"exited with code {exitcode}"
        reason = "nonzero_exit"
    elif not meta_present:
        desc = "exited 0 without writing a handoff meta"
        reason = "missing_meta"
    else:
        desc = "exited 0 without a success/error outcome"
        reason = "unknown_outcome"

    retry_word = "retry" if retries == 1 else "retries"
    message = (
        f"Render subprocess {desc} after {retries} {retry_word} "
        f"(post_init={init_ok})."
    )
    return RenderError(
        RENDER_SUBPROCESS_CRASH,
        message,
        detail={
            "exitcode": exitcode,
            "signal": signal_name,
            "retries": retries,
            "init_ok": init_ok,
            "reason": reason,
        },
        component=_COMPONENT,
    )


def _unlink_quiet(path: Path) -> None:
    """Remove a handoff artifact if present; tolerate its absence."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
