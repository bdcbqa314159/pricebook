"""NumericalConfig — the explicit reproducibility knobs the engine reads.

Spine invariant 5: config is explicit, never a hidden default. Slice 0 needs
one knob — the finite-difference bump for DV01. MC/PDE/tree knobs migrate with
the slices that first consume them (CLAUDE.md 6b — no speculative fields).

Provenance:
  quarry: python/pricebook/core/numerical_config.py
  source: redesign/02_spine.md (stateless-engine contract, invariant 5)
  oracle: to_dict/from_dict round-trip + schema versioning; fd_bump via the DV01 oracle
  slice:  S00; serialisation-numerical-config (CP-3 #1 — RETIRES core/numerical_config)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NumericalConfig:
    """Frozen numerical hyperparameters for a pricing run."""

    fd_bump: float = 1e-6  # central-difference step for finite-difference greeks
    mc_paths: int = 50_000  # Monte-Carlo sample count
    mc_seed: int = 12345    # RNG seed — fixed so MC is reproducible (referential transparency)

    def __post_init__(self) -> None:
        if self.fd_bump <= 0:
            raise ValueError(f"fd_bump must be > 0, got {self.fd_bump!r}")
        if self.mc_paths <= 0:
            raise ValueError(f"mc_paths must be > 0, got {self.mc_paths!r}")

    def replace(self, **changes: Any) -> "NumericalConfig":
        """A new config with `changes` applied (the frozen dataclass is never mutated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable plain-dict wire form. Carries `schema_version` so a future
        breaking change is a loud reject on read, not a silent misread."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "fd_bump": self.fd_bump,
            "mc_paths": self.mc_paths,
            "mc_seed": self.mc_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumericalConfig":
        """Reconstruct from `to_dict`. A payload with no version is legacy v1; a version
        newer than this reader is refused rather than misread."""
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"NumericalConfig schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(fd_bump=data["fd_bump"], mc_paths=data["mc_paths"], mc_seed=data["mc_seed"])
