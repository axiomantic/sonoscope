"""QwenLocalAdapter — advisory local Qwen2-Audio perception (Task G1, by design).

Wraps Qwen2-Audio-7B-Instruct as an advisory :class:`PerceptionAdapter` (C4). The
deterministic layer is ground truth; this adapter only produces a labeled,
free-text advisory description that is NEVER fatal (by design).

Runtime: the model runs on the **transformers reference runtime** —
``Qwen2AudioForConditionalGeneration`` + ``AutoProcessor``, torch on **MPS
(Metal) fp16 with a CPU fp32 fallback**. (``nexa-gguf`` is a GGUF *file-format*
library, not an inference engine, and has no installable macOS/arm64 wheel.) The
load + describe() invocation below was validated end-to-end in earlier
feasibility testing.

Design guarantees implemented here:

- **In-process singleton (R7).** The ~16 GB model loads exactly once per
  process (module-level ``_STATE``), never per describe() call.
- **60 s hard timeout (M2) via a per-token deadline StoppingCriteria.** B3
  (I5) empirically confirmed a real ``generate()`` is cooperatively interruptible
  at token granularity — a deadline ``StoppingCriteria`` fired and cleanly stopped
  the decode loop. This IMPROVES on the stale plan's watchdog abandon-and-discard
  (which existed only for the un-interruptible native Nexa path that no longer
  exists): the deadline cleanly stops the compute in-process, so on expiry
  describe() returns ``status='error'`` with no abandoned background thread.
- **Graceful ``status='unavailable'`` (by design).** When the runtime (transformers/
  torch) or the weights are KNOWN-absent, describe() returns an unavailable block
  with an explanatory message and never raises into the caller. (A mid-inference
  failure is distinct: it propagates so the F1 boundary records it as an error.)
- **``grounding='advisory-freetext'``** with the exact advisory disclaimer (by design).

Pins are law (AGENTS.md): the model identity constants below are re-exported from
``pins/qwen_model.manifest.toml`` (the B3-reconciled authority) so the adapter
needs no packaged manifest file; ``test_pins_match_manifest`` hard-fails on drift.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..schema.models import AdapterInfo, DeterministicBlock, Grounding, PerceptionBlock
from .base import AdapterHealth

__all__ = ["QwenLocalAdapter", "DISCLAIMER"]

# --- pinned model identity (mirrors pins/qwen_model.manifest.toml; drift-tested) ---
HF_REPO = "Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"
MODEL_ID = "Qwen2-Audio-7B-Instruct"
MODEL_RUNTIME = "transformers"
MODEL_PRECISION = "fp16"
# Weight-integrity anchor: sha256 of model.safetensors.index.json, which
# enumerates + integrity-maps every weight shard (the manifest's weights_index).
MODEL_SHA256 = "b6cc05302d1bd25fbab6915e3a033603c524416b984f661d213f9a1f8e3b3895"
# The manifest weights_index filename (the anchor whose sha256 is MODEL_SHA256).
# The lightweight health() probe (NIT-1) checks this file's presence in the local
# HF cache to confirm the pinned weights are downloaded WITHOUT loading them.
WEIGHTS_INDEX_FILENAME = "model.safetensors.index.json"

# Exact advisory disclaimer (by design): machine-legible epistemic status.
DISCLAIMER = "Advisory only. Not ground truth. May be inaccurate or hallucinated."

# describe() prompt — validated in earlier feasibility testing.
PROMPT = (
    "Listen to this synthesizer sound and describe it. Comment on its "
    "brightness (bright vs dark/muffled), pitch (high vs low), whether it "
    "is a clear tone or noisy, and how loud it is."
)

_DEFAULT_TIMEOUT_S = 60.0        # M2 hard timeout
_DEFAULT_MAX_NEW_TOKENS = 96     # bounded advisory description length (B3)

# --- in-process singleton state (R7) ---------------------------------------------
_STATE: Optional["_LoadedModel"] = None
# Serializes the cold load so a second concurrent caller blocks on the in-flight
# load instead of starting its own ~16 GB load (double-checked locking).
_LOAD_LOCK = threading.Lock()


class _ModelUnavailable(Exception):
    """Raised internally when the runtime/weights are KNOWN-absent. describe()
    converts this into a graceful ``status='unavailable'`` block — it never
    escapes to the caller."""


class _Deadline:
    """Per-token wall-clock deadline used as a ``StoppingCriteria`` (M2 / I5).

    Callable with the transformers stopping-criteria signature; returns True once
    the deadline passes so ``generate()`` stops cleanly at the next decode step.
    Kept transformers-free so the non-integration tests import without the extra.
    """

    def __init__(self, timeout_s: float) -> None:
        self._expiry = time.monotonic() + timeout_s
        self.fired = False

    def __call__(self, input_ids: Any = None, scores: Any = None, **kwargs: Any) -> bool:
        if time.monotonic() >= self._expiry:
            self.fired = True
            return True
        return False


@dataclass
class _LoadedModel:
    """A loaded transformers model + processor bound to a device. Owns the real
    audio-in describe() invocation."""

    model: Any
    processor: Any
    device: str
    sampling_rate: int

    def run(self, wav_path: Path, deadline: _Deadline, max_new_tokens: int) -> str:
        """Run one real audio-in generation, stopped by ``deadline`` (M2)."""
        import librosa
        from transformers import StoppingCriteriaList

        audio, _ = librosa.load(str(wav_path), sr=self.sampling_rate)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(wav_path)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=self.sampling_rate,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic advisory output (B3)
            stopping_criteria=StoppingCriteriaList([deadline]),
        )
        generated = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _device_plan(torch: Any, device_pref: Optional[str]) -> list[tuple[str, Any]]:
    """MPS(Metal) fp16 default with a CPU fp32 fallback (B3).

    ``device_pref`` may pin ``'mps'`` or ``'cpu'``; the default (None) prefers MPS
    when available and always keeps CPU as the fallback.
    """
    plan: list[tuple[str, Any]] = []
    if device_pref in (None, "mps") and torch.backends.mps.is_available():
        plan.append(("mps", torch.float16))
    if device_pref != "mps":
        plan.append(("cpu", torch.float32))
    return plan


def _load_model(
    model_ref: str, revision: Optional[str], device_pref: Optional[str]
) -> _LoadedModel:
    """Load the pinned Qwen2-Audio model once (transformers + torch, MPS->CPU).

    Raises :class:`_ModelUnavailable` ONLY when the runtime or the pinned weights
    are KNOWN-absent — the runtime import fails (``ImportError``), or the weights
    are missing from the local cache under ``local_files_only`` (``FileNotFoundError``
    / ``OSError``) — so the caller can degrade gracefully. Any OTHER load
    failure (MPS OOM, corrupted shard, torch runtime fault) is a genuine
    environment fault and is deliberately allowed to PROPAGATE so the F1 boundary
    records it as an error rather than masking it as "not installed".
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except ImportError as exc:  # runtime-absence detection
        raise _ModelUnavailable(
            f"perception runtime (transformers/torch) not installed: {exc!r}"
        ) from exc

    load_errors: list[str] = []
    for device, dtype in _device_plan(torch, device_pref):
        try:
            processor = AutoProcessor.from_pretrained(
                model_ref, revision=revision, local_files_only=True
            )
            model = (
                Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_ref,
                    revision=revision,
                    local_files_only=True,
                    torch_dtype=dtype,
                )
                .to(device)
                .eval()
            )
        except (FileNotFoundError, OSError) as exc:
            # KNOWN-absent weights under local_files_only: record + try next device.
            # Only weight-absence is swallowed here; other faults propagate.
            load_errors.append(f"{device}: {type(exc).__name__}: {exc}")
            continue
        sampling_rate = processor.feature_extractor.sampling_rate
        return _LoadedModel(
            model=model,
            processor=processor,
            device=device,
            sampling_rate=sampling_rate,
        )

    raise _ModelUnavailable(
        "Qwen2-Audio weights/runtime unavailable: " + "; ".join(load_errors)
    )


def _get_model(
    model_ref: str, revision: Optional[str], device_pref: Optional[str]
) -> _LoadedModel:
    """Return the process-wide singleton, loading it exactly once (R7).

    Thread-safe via double-checked locking under ``_LOAD_LOCK``: the fast path
    checks ``_STATE`` outside the lock, and the slow path re-checks inside the
    lock before loading. This prevents a latent double ~16 GB load when a second
    caller (e.g. a fresh F1 watchdog thread after a prior thread was abandoned on
    timeout) races in while a cold load is in flight — it blocks on the in-flight
    load instead of starting its own.

    A load failure is NOT cached (``_STATE`` stays None on any raise from
    ``_load_model``), so an unavailable/faulting adapter re-attempts on each call
    rather than poisoning the singleton.
    """
    global _STATE
    # Fast path: already loaded — no lock contention on the hot path.
    if _STATE is not None:
        return _STATE
    with _LOAD_LOCK:
        # Slow path: re-check under the lock so a caller that blocked while
        # another thread loaded returns the freshly-loaded singleton instead of
        # starting a second load.
        if _STATE is None:
            _STATE = _load_model(model_ref, revision, device_pref)
        return _STATE


# --- lightweight availability probe (NIT-1) --------------------------------------
# health() reports readiness WITHOUT loading the ~16 GB model or warming the
# in-process singleton: a runtime-presence check (transformers + torch importable)
# plus an HF-cache presence check for the pinned repo+revision weights. This keeps
# ``doctor`` cheap — it previously triggered a full multi-GB load just to report
# availability.

#: Pinned-runtime modules health() probes for importability (never loads them).
_RUNTIME_MODULES: tuple[str, ...] = ("transformers", "torch")


def _missing_runtime_modules() -> list[str]:
    """Pinned-runtime modules NOT importable, checked WITHOUT importing them.

    Uses ``importlib.util.find_spec`` so the heavy model class is never loaded; a
    ``find_spec`` that itself raises (a broken/partial install) counts the module
    as missing rather than propagating.
    """
    import importlib.util

    missing: list[str] = []
    for module in _RUNTIME_MODULES:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)
    return missing


def _weights_cached(model_ref: str, revision: Optional[str]) -> bool:
    """True iff the pinned weights index for ``(model_ref, revision)`` is present
    in the local HF cache, checked WITHOUT loading any weights.

    Uses ``huggingface_hub.try_to_load_from_cache``, which returns a real
    filesystem path (str) only when the snapshot file exists; a not-cached miss
    (``None``) or a known-nonexistent sentinel is treated as absent. If
    ``huggingface_hub`` itself is unimportable the weights cannot be present, so
    this returns False.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    resolved = try_to_load_from_cache(
        model_ref, WEIGHTS_INDEX_FILENAME, revision=revision
    )
    return isinstance(resolved, str)


class QwenLocalAdapter:
    """Advisory local Qwen2-Audio perception adapter (by design, C4)."""

    id = "qwen-local"
    grounding: Grounding = "advisory-freetext"

    def __init__(
        self,
        model_ref: str = HF_REPO,
        revision: Optional[str] = MODEL_REVISION,
        device_pref: Optional[str] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model_ref = model_ref
        self._revision = revision
        self._device_pref = device_pref
        self._timeout_s = timeout_s
        self._max_new_tokens = max_new_tokens

    def describe(
        self,
        wav_path: Path,
        deterministic: Optional[DeterministicBlock] = None,
    ) -> PerceptionBlock:
        """Advisory free-text description of the rendered wav (by design).

        - KNOWN-absent runtime/weights -> ``status='unavailable'`` (never raises).
        - describe() exceeding the hard timeout -> ``status='error'`` (loop-safe),
          via the per-token deadline StoppingCriteria (M2).
        - Otherwise -> ``status='ok'`` free-text block carrying the disclaimer.
        A mid-inference failure is intentionally NOT caught here: it propagates so
        the F1 perception-never-fatal boundary records it as an error.
        """
        try:
            loaded = _get_model(self._model_ref, self._revision, self._device_pref)
        except _ModelUnavailable as exc:
            # Graceful degradation — advisory absence never fails the loop.
            return PerceptionBlock(
                status="unavailable",
                grounding="none",
                description=str(exc),
            )

        # M2: bound the decode loop by the hard timeout via a deadline that
        # generate() checks each token; on expiry it stops cleanly and we report
        # error (improves on the stale watchdog abandon-and-discard, per B3).
        deadline = _Deadline(self._timeout_s)
        text = loaded.run(wav_path, deadline=deadline, max_new_tokens=self._max_new_tokens)
        if deadline.fired:
            return PerceptionBlock(status="error", grounding="none")

        # By design: advisory-freetext delivers no structured term->metric grounding,
        # so grounding_map is the honest empty map (present as the schema requires
        # it non-null when ok, but claiming no cross-checkable term).
        return PerceptionBlock(
            status="ok",
            grounding=self.grounding,
            adapter=AdapterInfo(
                id=self.id,
                model=MODEL_ID,
                quant="none",
                runtime=MODEL_RUNTIME,
                model_sha256=MODEL_SHA256,
            ),
            description=text,
            grounding_map={},
            disclaimer=DISCLAIMER,
        )

    def health(self) -> AdapterHealth:
        """Report readiness with a LIGHTWEIGHT probe (NIT-1, by design).

        Does NOT load the ~16 GB model and does NOT warm the in-process singleton
        (so ``doctor`` stays cheap — a prior version triggered a full multi-GB
        load just to report availability). Availability requires BOTH the pinned
        runtime (transformers + torch importable) AND the pinned weights present
        in the local HF cache for the manifest repo+revision. Either absent ->
        ``available=False`` with an explanatory ``reason`` (graceful
        degradation). ``describe()`` keeps the full-load singleton path unchanged.
        """
        missing = _missing_runtime_modules()
        if missing:
            return AdapterHealth(
                available=False,
                runtime=MODEL_RUNTIME,
                model_id=MODEL_ID,
                reason=(
                    "perception runtime not installed: "
                    + ", ".join(missing)
                    + " missing"
                ),
            )
        if not _weights_cached(self._model_ref, self._revision):
            return AdapterHealth(
                available=False,
                runtime=MODEL_RUNTIME,
                model_id=MODEL_ID,
                reason=(
                    f"pinned weights not cached: {self._model_ref}@{self._revision} "
                    f"({WEIGHTS_INDEX_FILENAME} absent from the local HF cache)"
                ),
            )
        return AdapterHealth(
            available=True,
            runtime=MODEL_RUNTIME,
            model_id=MODEL_ID,
            reason=None,
        )
