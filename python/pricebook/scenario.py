"""
Scenario risk engine.

A Scenario is a named set of market data perturbations applied to a
PricingContext. The engine computes base PV vs scenario PV for a portfolio.

    scenario = parallel_shift(0.0001)  # +1bp
    result = run_scenarios(portfolio, base_ctx, [scenario])

Standard scenarios: parallel shift, point bump (DV01 ladder), vol bump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
from pricebook.trade import Portfolio


@dataclass
class ScenarioResult:
    """Result of a single scenario."""

    name: str
    base_pv: float
    scenario_pv: float

    @property
    def pnl(self) -> float:
        return self.scenario_pv - self.base_pv


def _times_to_dates(reference_date: date, times):
    """Convert year fractions back to dates (approximate)."""
    return [date.fromordinal(reference_date.toordinal() + int(t * 365))
            for t in times if t > 0]


def _bump_curve(curve: DiscountCurve, shift: float) -> DiscountCurve:
    """Shift a discount curve by a parallel zero-rate bump.

    Each discount factor is adjusted: df_new = df_old * exp(-shift * t).
    """
    times = curve._times
    dfs = curve._dfs

    # Skip t=0 (df=1) when reconstructing
    new_dfs = [float(df * math.exp(-shift * t)) for t, df in zip(times, dfs) if t > 0]
    pillar_dates = _times_to_dates(curve.reference_date, times)

    return DiscountCurve(curve.reference_date, pillar_dates, new_dfs)


def _bump_curve_at_pillar(curve: DiscountCurve, pillar_idx: int, shift: float) -> DiscountCurve:
    """Bump a single pillar of a discount curve."""
    times = curve._times
    dfs = curve._dfs

    # Filter out t=0
    pillar_times = [t for t in times if t > 0]
    pillar_dfs = [float(df) for t, df in zip(times, dfs) if t > 0]
    pillar_dates = _times_to_dates(curve.reference_date, times)

    pillar_dfs[pillar_idx] = pillar_dfs[pillar_idx] * math.exp(-shift * pillar_times[pillar_idx])

    return DiscountCurve(curve.reference_date, pillar_dates, pillar_dfs)


# ---------------------------------------------------------------------------
# Scenario constructors
# ---------------------------------------------------------------------------

def parallel_shift(shift: float, name: str | None = None):
    """Create a parallel rate shift scenario.

    Args:
        shift: shift in rate terms (e.g. 0.0001 = +1bp).
    """
    def apply(ctx: PricingContext) -> PricingContext:
        new_disc = _bump_curve(ctx.discount_curve, shift) if ctx.discount_curve else None
        new_proj = {
            k: _bump_curve(v, shift)
            for k, v in ctx.projection_curves.items()
        }
        return PricingContext(
            valuation_date=ctx.valuation_date,
            discount_curve=new_disc,
            projection_curves=new_proj,
            vol_surfaces=ctx.vol_surfaces,
            credit_curves=ctx.credit_curves,
            fx_spots=ctx.fx_spots,
        )

    return _Scenario(name or f"parallel_{shift*10000:.0f}bp", apply)


def pillar_bump(pillar_idx: int, shift: float = 0.0001, name: str | None = None):
    """Bump a single curve pillar (for DV01 ladder)."""
    def apply(ctx: PricingContext) -> PricingContext:
        new_disc = _bump_curve_at_pillar(ctx.discount_curve, pillar_idx, shift) \
            if ctx.discount_curve else None
        return PricingContext(
            valuation_date=ctx.valuation_date,
            discount_curve=new_disc,
            projection_curves=ctx.projection_curves,
            vol_surfaces=ctx.vol_surfaces,
            credit_curves=ctx.credit_curves,
            fx_spots=ctx.fx_spots,
        )

    return _Scenario(name or f"pillar_{pillar_idx}_{shift*10000:.0f}bp", apply)


def vol_bump(shift: float, surface_name: str = "ir", name: str | None = None):
    """Bump all vol surfaces by a flat amount."""
    from pricebook.vol_surface import FlatVol

    def apply(ctx: PricingContext) -> PricingContext:
        new_vols = dict(ctx.vol_surfaces)
        if surface_name in new_vols:
            old = new_vols[surface_name]
            if hasattr(old, '_vol'):
                new_vols[surface_name] = FlatVol(old._vol + shift)
        return PricingContext(
            valuation_date=ctx.valuation_date,
            discount_curve=ctx.discount_curve,
            projection_curves=ctx.projection_curves,
            vol_surfaces=new_vols,
            credit_curves=ctx.credit_curves,
            fx_spots=ctx.fx_spots,
        )

    return _Scenario(name or f"vol_{shift*100:.0f}pct", apply)


def fx_spot_shock(base: str, quote: str, shift_pct: float, name: str | None = None):
    """Shock an FX spot rate by a percentage."""
    def apply(ctx: PricingContext) -> PricingContext:
        new_spots = dict(ctx.fx_spots)
        key = (base, quote)
        if key in new_spots:
            new_spots[key] = new_spots[key] * (1 + shift_pct)
        return PricingContext(
            valuation_date=ctx.valuation_date,
            discount_curve=ctx.discount_curve,
            projection_curves=ctx.projection_curves,
            vol_surfaces=ctx.vol_surfaces,
            credit_curves=ctx.credit_curves,
            fx_spots=new_spots,
        )

    return _Scenario(name or f"fx_{base}{quote}_{shift_pct*100:.0f}pct", apply)


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
    n_pillars = len([t for t in base_ctx.discount_curve._times if t > 0])
    scenarios = [
        pillar_bump(i, shift, name=f"pillar_{i}")
        for i in range(n_pillars)
    ]
    return run_scenarios(portfolio, base_ctx, scenarios)
