"""NumericalConfig — the complete, explicit reproducibility knob set (L0).

Spine invariant 5: config is explicit, never a hidden default. The set is designed
**complete up front** — MC, PDE, tree, quadrature, COS and root-finder knobs are all
here even though their engines migrate later — because retrofitting a foundational
value type is the expensive change (the 12 knobs deferred at CP-3 retire #1 are exactly
this pre-emption). Carries the serialisation pattern: per-class `to_dict`/`from_dict` +
`schema_version`, no framework.

Provenance:
  quarry: python/pricebook/core/numerical_config.py
  source: redesign/16 §2.5 (finance-free numerics — interfaces complete, algorithms deferred)
  oracle: round-trip + schema versioning; positive-knob validation
  slice:  numerics-config (Topic 0 S6)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NumericalConfig:
    """Frozen numerical hyperparameters for a pricing run — the full knob set."""
    # fields-exempt: config aggregate — reproducibility knobs across all numerical methods

    # finite-difference greeks
    fd_bump: float = 1e-6
    # Monte-Carlo
    mc_paths: int = 50_000
    mc_seed: int = 12345
    mc_antithetic: bool = True
    mc_sobol: bool = False
    mc_brownian_bridge: bool = False
    # PDE
    pde_time_steps: int = 100
    pde_space_steps: int = 200
    pde_n_std_devs: float = 5.0
    # lattice / tree
    tree_steps: int = 500
    # quadrature
    quadrature_tol: float = 1e-8
    quadrature_max_iter: int = 200
    # Fourier COS
    cos_n: int = 128
    cos_l: float = 10.0
    # root-finder
    rootfinder_tol: float = 1e-12
    rootfinder_max_iter: int = 200

    def __post_init__(self) -> None:
        for name in ("fd_bump", "mc_paths", "pde_time_steps", "pde_space_steps",
                     "tree_steps", "cos_n"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")

    def replace(self, **changes: Any) -> NumericalConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema_version": _SCHEMA_VERSION}
        d.update(self.__dict__)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumericalConfig:
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(f"NumericalConfig schema v{version} newer than reader v{_SCHEMA_VERSION}")
        return cls(**{k: v for k, v in data.items() if k != "schema_version"})
