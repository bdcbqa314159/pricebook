"""Pricing engine: JSON in → price/risk/report out.

Single entry point for pricing any pricebook instrument from a JSON
(or dict) specification. Builds curves, constructs instruments,
computes PV and Greeks, returns structured results.

    from pricebook.pricing_engine import price_from_json, price_from_dict

    result_json = price_from_json('''
    {
      "valuation_date": "2026-04-28",
      "market_data": {"discount_curve": {"type": "DiscountCurve", "params": {...}}},
      "trades": [{"type": "irs", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}}]
    }
    ''')

    # Or programmatically:
    result = price_from_dict(request_dict)
"""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

from pricebook.serialization import (
    instrument_from_dict,
    discount_curve_from_dict,
    survival_curve_from_dict,
    spread_curve_from_dict,
    ibor_curve_from_dict,
    funding_curve_from_dict,
    csa_from_dict,
    _str_to_date,
)


def price_from_json(json_str: str) -> str:
    """Price from a JSON string. Returns a JSON string with results.

    This is the main entry point for external callers.
    """
    request = json.loads(json_str)
    result = price_from_dict(request)
    return json.dumps(result, indent=2)


def price_from_dict(request: dict[str, Any]) -> dict[str, Any]:
    """Price from a dict. Returns a dict with results.

    Input schema:
        valuation_date: str (ISO)
        market_data: {discount_curve, projection_curves, survival_curves, vol_surfaces, fx_spots}
        trades: [{type, params}]
        config: {compute_greeks, measures}

    Output schema:
        status: "ok" | "error" | "partial"
        results: [{trade_index, instrument_type, pv, greeks, risk}]
        compute_time_ms: float
    """
    t0 = time.monotonic()
    results = []
    errors = []

    try:
        val_date = _str_to_date(request["valuation_date"])
        config = request.get("config", {})
        compute_greeks = config.get("compute_greeks", False)
        measures = set(config.get("measures", ["pv"]))

        # Build pricing context from market_data
        ctx = _build_context(val_date, request.get("market_data", {}))

        # Price each trade
        for i, trade_d in enumerate(request.get("trades", [])):
            try:
                result = _price_one(i, trade_d, ctx, val_date, compute_greeks, measures)
                results.append(result)
            except Exception as e:
                errors.append({"trade_index": i, "error": str(e), "type": type(e).__name__})
                results.append({
                    "trade_index": i,
                    "instrument_type": trade_d.get("type", "unknown"),
                    "status": "error",
                    "error": str(e),
                })

    except Exception as e:
        errors.append({"trade_index": -1, "error": str(e), "type": type(e).__name__})

    elapsed = (time.monotonic() - t0) * 1000
    status = "ok" if not errors else ("partial" if any(r.get("status") != "error" for r in results) else "error")

    return {
        "status": status,
        "results": results,
        "compute_time_ms": round(elapsed, 3),
        "n_trades": len(request.get("trades", [])),
    }


def _build_context(val_date: date, market_data: dict) -> Any:
    """Build a PricingContext from the market_data section."""
    from pricebook.pricing_context import PricingContext
    from pricebook.discount_curve import DiscountCurve
    from pricebook.vol_surface import FlatVol

    discount = None
    if "discount_curve" in market_data:
        discount = discount_curve_from_dict(market_data["discount_curve"])

    # Flat rate shortcut
    if discount is None and "flat_rate" in market_data:
        discount = DiscountCurve.flat(val_date, market_data["flat_rate"])

    if discount is None:
        discount = DiscountCurve.flat(val_date, 0.03)

    # Projection curves
    projection_curves = {}
    for name, cd in market_data.get("projection_curves", {}).items():
        if cd.get("type") == "IBORCurve":
            from pricebook.ibor_curve import IBORCurve
            ibor = ibor_curve_from_dict(cd)
            projection_curves[name] = ibor.projection_curve
        else:
            projection_curves[name] = discount_curve_from_dict(cd)

    # Vol surfaces
    vol_surfaces: dict[str, Any] = {}
    for name, vd in market_data.get("vol_surfaces", {}).items():
        if isinstance(vd, dict) and vd.get("type") == "FlatVol":
            vol_surfaces[name] = FlatVol(vd["vol"])
        elif isinstance(vd, (int, float)):
            vol_surfaces[name] = FlatVol(float(vd))

    # Credit curves
    credit_curves = {}
    for name, cd in market_data.get("survival_curves", {}).items():
        credit_curves[name] = survival_curve_from_dict(cd)

    # FX spots
    fx_spots: dict[tuple[str, str], float] = {}
    for pair_str, rate in market_data.get("fx_spots", {}).items():
        if "/" in pair_str:
            base, quote = pair_str.split("/")
            fx_spots[(base, quote)] = rate

    return PricingContext(
        valuation_date=val_date,
        discount_curve=discount,
        projection_curves=projection_curves if projection_curves else None,
        vol_surfaces=vol_surfaces if vol_surfaces else None,
        credit_curves=credit_curves if credit_curves else None,
        fx_spots=fx_spots if fx_spots else None,
    )


def _price_one(
    index: int,
    trade_d: dict,
    ctx: Any,
    val_date: date,
    compute_greeks: bool,
    measures: set[str],
) -> dict[str, Any]:
    """Price a single trade from its dict specification."""
    params = dict(trade_d.get("params", {}))
    inst_type = trade_d["type"]

    # Types that use 'maturity' as-is (not converted to 'end')
    _MATURITY_TYPES = {"bond", "fx_forward"}

    # Resolve date fields: tenor → date, add start if missing
    if "maturity" in params and inst_type not in _MATURITY_TYPES:
        mat = params.pop("maturity")
        if isinstance(mat, str):
            if len(mat) <= 4:
                params["end"] = _tenor_to_date(val_date, mat)
            else:
                params["end"] = date.fromisoformat(mat)
        else:
            params["end"] = mat
    elif "maturity" in params and inst_type in _MATURITY_TYPES:
        # Convert tenor to date but keep as 'maturity'
        mat = params["maturity"]
        if isinstance(mat, str) and len(mat) <= 4:
            params["maturity"] = _tenor_to_date(val_date, mat)

    if "start" not in params and "end" in params and inst_type not in _MATURITY_TYPES:
        params["start"] = val_date

    # Remove non-constructor convenience fields
    params.pop("currency", None)

    # Construct instrument
    instrument = instrument_from_dict({"type": inst_type, "params": params})

    # Price — try pv_ctx first, fall back to pv with curve(s)
    if hasattr(instrument, "pv_ctx"):
        try:
            pv = instrument.pv_ctx(ctx)
        except (TypeError, AttributeError, KeyError):
            # pv_ctx may fail if context is missing required data (e.g. survival curve for CDS)
            pv = _pv_fallback(instrument, ctx, val_date)
    elif hasattr(instrument, "pv"):
        try:
            pv = instrument.pv(ctx.discount_curve)
        except TypeError:
            # Some instruments need extra curves (e.g. CDS needs survival_curve)
            from pricebook.survival_curve import SurvivalCurve
            sc = None
            if ctx.credit_curves:
                sc = next(iter(ctx.credit_curves.values()))
            if sc is None:
                sc = SurvivalCurve.flat(val_date, 0.02)
            try:
                pv = instrument.pv(ctx.discount_curve, sc)
            except TypeError:
                pv = instrument.pv(ctx.discount_curve)
    else:
        raise ValueError(f"Instrument {inst_type} has no pricing method")

    result: dict[str, Any] = {
        "trade_index": index,
        "instrument_type": inst_type,
        "status": "ok",
        "pv": pv,
    }

    # Greeks
    if compute_greeks:
        greeks = _compute_greeks(instrument, ctx, measures)
        if greeks:
            result["greeks"] = greeks

    return result


def _pv_fallback(instrument, ctx, val_date):
    """Try pv() with various argument combinations."""
    from pricebook.survival_curve import SurvivalCurve
    sc = None
    if ctx.credit_curves:
        sc = next(iter(ctx.credit_curves.values()))
    if sc is None:
        sc = SurvivalCurve.flat(val_date, 0.02)
    try:
        return instrument.pv(ctx.discount_curve, sc)
    except TypeError:
        return instrument.pv(ctx.discount_curve)


def _compute_greeks(instrument, ctx, measures: set[str]) -> dict[str, float]:
    """Compute requested Greeks via bump-and-reprice."""
    greeks: dict[str, float] = {}
    curve = ctx.discount_curve

    if "dv01" in measures and hasattr(instrument, "dv01"):
        try:
            greeks["dv01"] = instrument.dv01(curve)
        except Exception:
            pass
    elif "dv01" in measures:
        # Generic bump-and-reprice DV01
        try:
            bumped = curve.bumped(0.0001)
            if hasattr(instrument, "pv_ctx"):
                base = instrument.pv_ctx(ctx)
                bumped_ctx = ctx.replace(discount_curve=bumped)
                up = instrument.pv_ctx(bumped_ctx)
            else:
                base = instrument.pv(curve)
                up = instrument.pv(bumped)
            greeks["dv01"] = up - base
        except Exception:
            pass

    if "delta" in measures and hasattr(instrument, "greeks"):
        try:
            g = instrument.greeks(curve)
            if "delta" in g:
                greeks["delta"] = g["delta"]
        except Exception:
            pass

    return greeks


def _tenor_to_date(ref: date, tenor: str) -> date:
    """Convert tenor string to date: '3M' → +3 months, '5Y' → +5 years."""
    from dateutil.relativedelta import relativedelta
    from datetime import timedelta

    tenor = tenor.strip().upper()
    if tenor.endswith("D"):
        return ref + timedelta(days=int(tenor[:-1]))
    elif tenor.endswith("W"):
        return ref + timedelta(weeks=int(tenor[:-1]))
    elif tenor.endswith("M"):
        return ref + relativedelta(months=int(tenor[:-1]))
    elif tenor.endswith("Y"):
        return ref + relativedelta(years=int(tenor[:-1]))
    raise ValueError(f"Unknown tenor format: {tenor}")
