"""NumericalConfig — the explicit reproducibility knobs the engine reads.

Spine invariant 5: config is explicit, never a hidden default. Slice 0 needs
one knob — the finite-difference bump for DV01. MC/PDE/tree knobs migrate with
the slices that first consume them (CLAUDE.md 6b — no speculative fields).

Provenance:
  quarry: python/pricebook/core/numerical_config.py
  source: redesign/02_spine.md (stateless-engine contract, invariant 5)
  oracle: N/A (config value type; fd_bump exercised by the DV01 oracle)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalConfig:
    """Frozen numerical hyperparameters for a pricing run."""

    fd_bump: float = 1e-6  # central-difference step for finite-difference greeks

    def __post_init__(self) -> None:
        if self.fd_bump <= 0:
            raise ValueError(f"fd_bump must be > 0, got {self.fd_bump!r}")
