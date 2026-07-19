"""NumericalConfig — reproducibility knobs, decomposed by method family (L0).

Spine invariant 5: config is explicit. **Decomposed** into one sub-config per numerical
method (§3b — a config never earns `fields-exempt`; it groups naturally by family and
must split). Each sub-config is ≤5 fields, so `verify.py fields` passes on merit, and
`numerics.monte_carlo.paths` states what it is. Designed complete up front (the engines
migrate later). RNG family is a **pinned invariant** (`RNG_FAMILY`) — not a field (that
would push MonteCarloConfig to 6), but pinned so a switch is never silent (S15).

Provenance:
  quarry: python/pricebook/core/numerical_config.py
  source: redesign/16 §2.5; gate audit F1 (decompose) / S15 (pin RNG)
  oracle: round-trip + schema versioning; positive-knob validation; RNG pinned
  slice:  config-result-shapes (Topic 0 gate rework, F1/S15)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

_SCHEMA_VERSION = 1

# S15 — the pinned RNG family. MC oracles are reproducible only against this generator;
# changing it shifts every MC result, so it is an explicit invariant, never a silent default.
RNG_FAMILY = "pcg64"


@dataclass(frozen=True)
class MonteCarloConfig:
    paths: int = 50_000
    seed: int = 12345
    antithetic: bool = True
    sobol: bool = False
    brownian_bridge: bool = False

    def __post_init__(self) -> None:
        if self.paths <= 0:
            raise ValueError(f"paths must be > 0, got {self.paths!r}")


@dataclass(frozen=True)
class LatticeConfig:
    """PDE grid + tree steps (both discretisation grids — tree folded in per the ruling)."""

    time_steps: int = 100
    space_steps: int = 200
    n_std_devs: float = 5.0
    tree_steps: int = 500

    def __post_init__(self) -> None:
        for name in ("time_steps", "space_steps", "tree_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class IntegrationConfig:
    quadrature_tol: float = 1e-8
    quadrature_max_iter: int = 200
    cos_n: int = 128
    cos_l: float = 10.0

    def __post_init__(self) -> None:
        if self.cos_n <= 0:
            raise ValueError(f"cos_n must be > 0, got {self.cos_n!r}")


@dataclass(frozen=True)
class SolverConfig:
    rootfinder_tol: float = 1e-12
    rootfinder_max_iter: int = 200
    fd_bump: float = 1e-6

    def __post_init__(self) -> None:
        if self.fd_bump <= 0:
            raise ValueError(f"fd_bump must be > 0, got {self.fd_bump!r}")


@dataclass(frozen=True)
class NumericalConfig:
    """The full reproducibility config — one sub-config per method family (4 fields, no
    exemption)."""

    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    lattice: LatticeConfig = LatticeConfig()
    integration: IntegrationConfig = IntegrationConfig()
    solver: SolverConfig = SolverConfig()

    def replace(self, **changes: Any) -> NumericalConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "monte_carlo": asdict(self.monte_carlo),
            "lattice": asdict(self.lattice),
            "integration": asdict(self.integration),
            "solver": asdict(self.solver),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumericalConfig:
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"NumericalConfig schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(
            monte_carlo=MonteCarloConfig(**data["monte_carlo"]),
            lattice=LatticeConfig(**data["lattice"]),
            integration=IntegrationConfig(**data["integration"]),
            solver=SolverConfig(**data["solver"]),
        )
