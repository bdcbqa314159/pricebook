"""NumericalConfig — frozen bundle of numerical hyperparameters.

A `PricingContext` carries an optional `NumericalConfig`. Pricers that
need a numerical choice (number of MC paths, PDE grid size, integration
tolerance, COS method points, etc.) read it from the context — not from
hand-coded defaults buried in their entry-point signatures.

Why this matters
----------------

Numerical choices ARE valuation inputs. A book repriced with `mc_paths=10_000`
and another with `mc_paths=1_000_000` are different valuations. Until the
numerical config is part of the context, those two prices are
indistinguishable from a noisy market move — and impossible to audit.

After this slice the choice is explicit, recorded in the context, and can
be serialised alongside the curves and quotes.

Usage
-----

    from pricebook.core.numerical_config import NumericalConfig
    from pricebook.core.pricing_context import PricingContext

    cfg = NumericalConfig(mc_paths=200_000, mc_seed=42, mc_antithetic=True)
    ctx = PricingContext(valuation_date=date(2026, 6, 11), ...,
                         numerical_config=cfg)
    pricer.price(ctx)              # pricer.price() reads ctx.get_numerical_config()

If `numerical_config` is `None` on the context, `get_numerical_config()`
returns `DEFAULT_NUMERICAL_CONFIG` — a sensible default that matches the
existing library hard-coded numbers (so existing call-sites keep their
behaviour unchanged).

Pricer-side adoption is incremental — a pricer that doesn't yet read the
context's config simply uses its own defaults, and behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class NumericalConfig:
    """Frozen numerical hyperparameters for a pricing run.

    Field defaults match the historical library defaults (where they exist),
    so attaching a default-constructed `NumericalConfig` to an existing
    context is a no-op behaviour-wise.

    Subgroups are documented but not nested as separate dataclasses —
    flatter access for pricers, easier diffing in audit logs.
    """

    # ---- Monte Carlo ----
    mc_paths: int = 50_000
    mc_seed: int | None = None
    mc_antithetic: bool = False
    mc_use_sobol: bool = False
    mc_brownian_bridge: bool = False

    # ---- PDE / Finite-difference ----
    pde_time_steps: int = 100
    pde_space_steps: int = 200
    pde_n_std_devs: float = 4.0          # truncation width in vol units

    # ---- Tree / lattice ----
    tree_steps: int = 100

    # ---- Numerical integration / quadrature ----
    integration_tol: float = 1.0e-9
    integration_max_iter: int = 200

    # ---- COS method (Lévy / jump pricing) ----
    cos_n: int = 1024                    # number of COS terms
    cos_L: float = 10.0                  # truncation interval [-L,L] in stdev units

    # ---- Root-finding (implied vol, par-rate solving, ...) ----
    rootfinder_tol: float = 1.0e-10
    rootfinder_max_iter: int = 100

    # ---- Escape hatch for project-specific knobs ----
    extra: Mapping[str, Any] = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> "NumericalConfig":
        """Return a new `NumericalConfig` with the given fields replaced."""
        return replace(self, **kwargs)


DEFAULT_NUMERICAL_CONFIG = NumericalConfig()
"""Library-wide default. Returned by `PricingContext.get_numerical_config()`
when no config is attached to the context. Frozen — safe to share."""
