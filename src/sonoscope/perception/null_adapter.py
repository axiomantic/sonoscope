"""NullAdapter — the always-present perception adapter (design 4.3, 10.2).

Guarantees the analysis loop always emits a valid perception block with no
model installed: describe() returns status=="disabled" (no adapter output),
health() reports available=False. Perception degrades to this gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema.models import DeterministicBlock, PerceptionBlock
from .base import AdapterHealth


class NullAdapter:
    """Perception adapter that does nothing but keep the contract valid."""

    id = "null"
    grounding = "none"

    def describe(
        self,
        wav_path: Path,
        deterministic: Optional[DeterministicBlock] = None,
    ) -> PerceptionBlock:
        return PerceptionBlock(status="disabled", grounding="none")

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            available=False,
            runtime="none",
            model_id="none",
            reason="No perception model configured (NullAdapter).",
        )
