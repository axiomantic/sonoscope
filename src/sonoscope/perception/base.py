"""PerceptionAdapter interface + AdapterHealth (design 4.3).

Perception is ADVISORY, never ground truth. The deterministic layer is the
source of truth; adapters only produce a labeled semantic description. An
adapter is always present (NullAdapter) so the loop emits valid JSON with no
model installed. ``PerceptionBlock`` is the schema model (single source of
truth) mapping to ``report.perception``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..schema.models import DeterministicBlock, Grounding, PerceptionBlock

# Re-export so consumers import the perception block from the perception
# package while it remains the single schema definition.
__all__ = ["AdapterHealth", "PerceptionAdapter", "PerceptionBlock"]


@dataclass
class AdapterHealth:
    """Adapter readiness (design 4.3, M3). ``reason`` explains unavailability
    when ``available`` is False. ``runtime`` is e.g. ``"nexa-gguf"`` /
    ``"mlx"`` / ``"none"``."""

    available: bool
    runtime: str
    model_id: str
    reason: Optional[str] = None


@runtime_checkable
class PerceptionAdapter(Protocol):
    """Advisory semantic-description contract (design 4.3)."""

    id: str
    grounding: Grounding

    def describe(
        self,
        wav_path: Path,
        deterministic: Optional[DeterministicBlock] = None,
    ) -> PerceptionBlock:
        """Produce an advisory perception block for the rendered wav."""
        ...

    def health(self) -> AdapterHealth:
        """Report whether the adapter can run (model loaded, runtime ok)."""
        ...
