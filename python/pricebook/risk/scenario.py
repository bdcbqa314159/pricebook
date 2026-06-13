"""
Scenario risk engine.

A Scenario is a named set of market data perturbations applied to a
PricingContext. The engine computes base PV vs scenario PV for a portfolio.

    scenario = parallel_shift(0.0001)  # +1bp
    result = run_scenarios(portfolio, base_ctx, [scenario])

Standard scenarios: parallel shift, point bump (DV01 ladder), vol bump.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.core.trade import Portfolio


@dataclass
class ScenarioResult:
    """Result of a single scenario."""

    name: str
    base_pv: float
    scenario_pv: float

    @property
    def pnl(self) -> float:
        return self.scenario_pv - self.base_pv



    def to_dict(self) -> dict:
        return vars(self)
# ---------------------------------------------------------------------------
# Scenario constructors
# ---------------------------------------------------------------------------

# Fix T4-RISK33: every scenario constructor in this module used
# ``PricingContext(field=value, ...)`` with only 6 named fields,
# silently dropping ``discount_curves`` (plural), ``inflation_curves``,
# ``repo_curves``, ``reporting_currency``, ``stochastic_credit_models``,
# ``credit_vol_surfaces``, ``credit_correlations``, and
# ``numerical_config``.  Same shape as the v0.993 var.stress_test bug.
# Now all five constructors use ``dataclasses.replace`` to preserve
# every untouched field.  Also: ``parallel_shift`` now bumps the
# plural ``discount_curves`` dict alongside the singular.


def parallel_shift(shift: float, name: str | None = None):
    """Create a parallel rate shift scenario.

    Args:
        shift: shift in rate terms (e.g. 0.0001 = +1bp).
    """
    def apply(ctx: PricingContext) -> PricingContext:
        updates: dict = {}
        if ctx.discount_curve is not None:
            updates["discount_curve"] = ctx.discount_curve.bumped(shift)
        if ctx.discount_curves:
            updates["discount_curves"] = {
                k: v.bumped(shift) for k, v in ctx.discount_curves.items()
            }
        if ctx.projection_curves:
            updates["projection_curves"] = {
                k: v.bumped(shift) for k, v in ctx.projection_curves.items()
            }
        return replace(ctx, **updates) if updates else ctx

    return _Scenario(name or f"parallel_{shift*10000:.0f}bp", apply)


def pillar_bump(pillar_idx: int, shift: float = 0.0001, name: str | None = None):
    """Bump a single curve pillar (for DV01 ladder)."""
    def apply(ctx: PricingContext) -> PricingContext:
        if ctx.discount_curve is None:
            return ctx
        return replace(ctx, discount_curve=ctx.discount_curve.bumped_at(pillar_idx, shift))

    return _Scenario(name or f"pillar_{pillar_idx}_{shift*10000:.0f}bp", apply)


def vol_bump(shift: float, surface_name: str = "ir", name: str | None = None):
    """Bump all vol surfaces by a flat amount."""
    from pricebook.options.vol_surface import FlatVol

    def apply(ctx: PricingContext) -> PricingContext:
        new_vols = dict(ctx.vol_surfaces)
        if surface_name in new_vols:
            old = new_vols[surface_name]
            if hasattr(old, '_vol'):
                new_vols[surface_name] = FlatVol(old._vol + shift)
        return replace(ctx, vol_surfaces=new_vols)

    return _Scenario(name or f"vol_{shift*100:.0f}pct", apply)


def fx_spot_shock(base: str, quote: str, shift_pct: float, name: str | None = None):
    """Shock an FX spot rate by a percentage."""
    def apply(ctx: PricingContext) -> PricingContext:
        new_spots = dict(ctx.fx_spots)
        key = (base, quote)
        if key in new_spots:
            new_spots[key] = new_spots[key] * (1 + shift_pct)
        return replace(ctx, fx_spots=new_spots)

    return _Scenario(name or f"fx_{base}{quote}_{shift_pct*100:.0f}pct", apply)


def credit_spread_shift(
    shift: float,
    names: list[str] | None = None,
    name: str | None = None,
):
    """Parallel credit spread shift: bump all (or named) survival curves.

    Args:
        shift: hazard rate shift (e.g. 0.01 = +100bp).
        names: if provided, only bump these credit curves.
    """
    def apply(ctx: PricingContext) -> PricingContext:
        new_credit = {}
        for k, v in ctx.credit_curves.items():
            if names is None or k in names:
                new_credit[k] = v.bumped(shift) if v else v
            else:
                new_credit[k] = v
        return replace(ctx, credit_curves=new_credit)

    return _Scenario(name or f"credit_{shift*10000:.0f}bp", apply)


class _Scenario:
    """Internal scenario wrapper."""

    def __init__(self, name: str, apply_fn):
        self.name = name
        self._apply = apply_fn

    def apply(self, ctx: PricingContext) -> PricingContext:
        return self._apply(ctx)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_scenarios(
    portfolio: Portfolio,
    base_ctx: PricingContext,
    scenarios: list[_Scenario],
) -> list[ScenarioResult]:
    """Run a list of scenarios against a portfolio.

    Returns:
        List of ScenarioResult, one per scenario.
    """
    base_pv = portfolio.pv(base_ctx)
    results = []
    for s in scenarios:
        bumped_ctx = s.apply(base_ctx)
        scenario_pv = portfolio.pv(bumped_ctx)
        results.append(ScenarioResult(
            name=s.name,
            base_pv=base_pv,
            scenario_pv=scenario_pv,
        ))
    return results


def dv01_ladder(
    portfolio: Portfolio,
    base_ctx: PricingContext,
    shift: float = 0.0001,
) -> list[ScenarioResult]:
    """DV01 ladder: bump each discount curve pillar by shift."""
    if base_ctx.discount_curve is None:
        return []
    n_pillars = len(base_ctx.discount_curve.pillar_dates)
    scenarios = [
        pillar_bump(i, shift, name=f"pillar_{i}")
        for i in range(n_pillars)
    ]
    return run_scenarios(portfolio, base_ctx, scenarios)
