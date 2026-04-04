"""Dashboard data layer: structured reports for any frontend.

All output is plain dicts/lists — serializable to JSON for Plotly,
web APIs, or any other consumer.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade, Portfolio


# ---------------------------------------------------------------------------
# Portfolio risk report
# ---------------------------------------------------------------------------


def portfolio_risk_report(
    portfolio: Portfolio,
    ctx: PricingContext,
    shift: float = 0.0001,
) -> dict[str, Any]:
    """Compute a risk report for a portfolio.

    Returns a dict with:
        - name: portfolio name
        - valuation_date: as-of date
        - total_pv: aggregate PV
        - trades: list of per-trade dicts (id, type, pv)
        - dv01: parallel DV01
    """
    total_pv = 0.0
    trades_out = []

    for i, t in enumerate(portfolio.trades):
        try:
            pv = t.pv(ctx)
        except Exception:
            pv = float("nan")
        total_pv += 0.0 if math.isnan(pv) else pv
        trades_out.append({
            "trade_id": t.trade_id or f"trade_{i}",
            "instrument_type": type(t.instrument).__name__,
            "direction": t.direction,
            "notional_scale": t.notional_scale,
            "pv": pv,
        })

    # Parallel DV01
    dv01 = 0.0
    if ctx.discount_curve is not None:
        bumped_ctx = ctx.replace(discount_curve=ctx.discount_curve.bumped(shift))
        try:
            bumped_pv = portfolio.pv(bumped_ctx)
            dv01 = bumped_pv - total_pv
        except Exception:
            pass

    return {
        "name": portfolio.name,
        "valuation_date": ctx.valuation_date.isoformat(),
        "total_pv": total_pv,
        "dv01": dv01,
        "n_trades": len(portfolio.trades),
        "trades": trades_out,
    }


# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------


def scenario_grid(
    portfolio: Portfolio,
    ctx: PricingContext,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute portfolio PV under multiple scenarios.

    Each scenario is a dict with optional keys:
        - rate_shift: parallel bump to discount curve
        - vol_shift: additive bump to FlatVol surfaces
        - name: scenario label

    Returns:
        {"base_pv": float, "scenarios": [{"name", "pv", "pnl"}, ...]}
    """
    from pricebook.vol_surface import FlatVol

    base_pv = portfolio.pv(ctx)
    results = []

    for s in scenarios:
        name = s.get("name", f"scenario_{len(results)}")
        bumped = ctx

        if "rate_shift" in s and ctx.discount_curve is not None:
            bumped = bumped.replace(
                discount_curve=bumped.discount_curve.bumped(s["rate_shift"])
            )

        if "vol_shift" in s:
            new_vols = {}
            for vn, vs in bumped.vol_surfaces.items():
                if isinstance(vs, FlatVol):
                    new_vols[vn] = FlatVol(vs._vol + s["vol_shift"])
                else:
                    new_vols[vn] = vs
            bumped = bumped.replace(vol_surfaces=new_vols)

        try:
            pv = portfolio.pv(bumped)
        except Exception:
            pv = float("nan")

        results.append({
            "name": name,
            "pv": pv,
            "pnl": pv - base_pv,
        })

    return {
        "base_pv": base_pv,
        "scenarios": results,
    }


# ---------------------------------------------------------------------------
# Trade blotter
# ---------------------------------------------------------------------------


def trade_blotter(
    portfolio: Portfolio,
    ctx: PricingContext,
) -> list[dict[str, Any]]:
    """Trade blotter: list of trades with key attributes.

    Returns a list of dicts, one per trade, suitable for a table view.
    """
    rows = []
    for i, t in enumerate(portfolio.trades):
        inst = t.instrument
        try:
            pv = t.pv(ctx)
        except Exception:
            pv = float("nan")

        row = {
            "trade_id": t.trade_id or f"trade_{i}",
            "instrument_type": type(inst).__name__,
            "direction": "long" if t.direction >= 0 else "short",
            "notional_scale": t.notional_scale,
            "counterparty": t.counterparty,
            "pv": pv,
        }

        # Extract common dates if available
        if hasattr(inst, "start"):
            row["start"] = inst.start.isoformat()
        if hasattr(inst, "end"):
            row["end"] = inst.end.isoformat()
        if hasattr(inst, "maturity"):
            row["maturity"] = inst.maturity.isoformat()

        rows.append(row)

    return rows
