"""Serialization framework: to_dict / from_dict for curves, instruments, trades.

Every serializable object produces a dict with:
    {"type": "class_name", "params": {...}}

The instrument registry maps type strings to classes for deserialization.
"""

from __future__ import annotations

import json
from datetime import date
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_to_str(d: date) -> str:
    return d.isoformat()


def _str_to_date(s: str) -> date:
    return date.fromisoformat(s)


def _enum_to_str(e) -> str:
    return e.value


# ---------------------------------------------------------------------------
# Curve serialization
# ---------------------------------------------------------------------------


def discount_curve_to_dict(curve) -> dict[str, Any]:
    """Serialize a DiscountCurve to a dict."""
    from pricebook.discount_curve import DiscountCurve
    # Extract pillar data (excluding t=0 prepended point)
    pillar_times = [float(t) for t in curve._times if t > 0]
    pillar_dfs = [float(df) for t, df in zip(curve._times, curve._dfs) if t > 0]
    pillar_dates = curve.pillar_dates

    return {
        "type": "DiscountCurve",
        "params": {
            "reference_date": _date_to_str(curve.reference_date),
            "dates": [_date_to_str(d) for d in pillar_dates],
            "dfs": pillar_dfs,
            "day_count": _enum_to_str(curve.day_count),
        },
    }


def discount_curve_from_dict(d: dict[str, Any]):
    """Deserialize a DiscountCurve from a dict."""
    from pricebook.discount_curve import DiscountCurve
    from pricebook.day_count import DayCountConvention
    p = d["params"]
    return DiscountCurve(
        reference_date=_str_to_date(p["reference_date"]),
        dates=[_str_to_date(s) for s in p["dates"]],
        dfs=p["dfs"],
        day_count=DayCountConvention(p["day_count"]),
    )


def survival_curve_to_dict(curve) -> dict[str, Any]:
    """Serialize a SurvivalCurve to a dict."""
    pillar_times = [float(t) for t in curve._times if t > 0]
    pillar_survs = [float(s) for t, s in zip(curve._times, curve._survs) if t > 0]
    ref = curve.reference_date
    pillar_dates = list(curve._pillar_dates)
    return {
        "type": "SurvivalCurve",
        "params": {
            "reference_date": _date_to_str(ref),
            "dates": [_date_to_str(d) for d in pillar_dates],
            "survival_probs": pillar_survs,
            "day_count": _enum_to_str(curve.day_count),
        },
    }


def survival_curve_from_dict(d: dict[str, Any]):
    """Deserialize a SurvivalCurve from a dict."""
    from pricebook.survival_curve import SurvivalCurve
    from pricebook.day_count import DayCountConvention
    p = d["params"]
    return SurvivalCurve(
        reference_date=_str_to_date(p["reference_date"]),
        dates=[_str_to_date(s) for s in p["dates"]],
        survival_probs=p["survival_probs"],
        day_count=DayCountConvention(p["day_count"]),
    )


# ---------------------------------------------------------------------------
# PricingContext serialization
# ---------------------------------------------------------------------------


def pricing_context_to_dict(ctx) -> dict[str, Any]:
    """Serialize a PricingContext to a dict."""
    from pricebook.vol_surface import FlatVol

    d: dict[str, Any] = {
        "valuation_date": _date_to_str(ctx.valuation_date),
    }

    if ctx.discount_curve is not None:
        d["discount_curve"] = discount_curve_to_dict(ctx.discount_curve)

    if ctx.projection_curves:
        d["projection_curves"] = {
            name: discount_curve_to_dict(c)
            for name, c in ctx.projection_curves.items()
        }

    if ctx.vol_surfaces:
        vols = {}
        for name, vs in ctx.vol_surfaces.items():
            if isinstance(vs, FlatVol):
                vols[name] = {"type": "FlatVol", "vol": vs._vol}
        d["vol_surfaces"] = vols

    if ctx.credit_curves:
        d["credit_curves"] = {
            name: survival_curve_to_dict(c)
            for name, c in ctx.credit_curves.items()
        }

    if ctx.fx_spots:
        d["fx_spots"] = {
            f"{base}/{quote}": rate
            for (base, quote), rate in ctx.fx_spots.items()
        }

    return d


def pricing_context_from_dict(d: dict[str, Any]):
    """Deserialize a PricingContext from a dict."""
    from pricebook.pricing_context import PricingContext
    from pricebook.vol_surface import FlatVol

    discount = None
    if "discount_curve" in d:
        discount = discount_curve_from_dict(d["discount_curve"])

    projection = {}
    for name, cd in d.get("projection_curves", {}).items():
        projection[name] = discount_curve_from_dict(cd)

    vol_surfaces: dict[str, object] = {}
    for name, vd in d.get("vol_surfaces", {}).items():
        if vd["type"] == "FlatVol":
            vol_surfaces[name] = FlatVol(vd["vol"])

    credit = {}
    for name, cd in d.get("credit_curves", {}).items():
        credit[name] = survival_curve_from_dict(cd)

    fx_spots: dict[tuple[str, str], float] = {}
    for pair_str, rate in d.get("fx_spots", {}).items():
        base, quote = pair_str.split("/")
        fx_spots[(base, quote)] = rate

    return PricingContext(
        valuation_date=_str_to_date(d["valuation_date"]),
        discount_curve=discount,
        projection_curves=projection,
        vol_surfaces=vol_surfaces,
        credit_curves=credit,
        fx_spots=fx_spots,
    )


# ---------------------------------------------------------------------------
# Instrument serialization
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    """Serialize a single value for JSON compatibility."""
    if isinstance(v, date):
        return _date_to_str(v)
    if isinstance(v, Enum):
        return v.value
    if v is None:
        return None
    # Handle CurrencyPair (has base/quote Currency enums)
    if hasattr(v, "base") and hasattr(v, "quote") and hasattr(v.base, "value"):
        return f"{v.base.value}/{v.quote.value}"
    return v


def _instrument_to_dict(inst, type_name: str, fields: list[str]) -> dict[str, Any]:
    """Generic instrument serialization given field names."""
    params = {}
    for f in fields:
        v = getattr(inst, f)
        params[f] = _serialize_value(v)
    return {"type": type_name, "params": params}




# --- Per-instrument serialization ---

_INSTRUMENT_FIELDS: dict[str, tuple[str, list[str]]] = {
    # Core rates
    "irs": ("pricebook.swap.InterestRateSwap", [
        "start", "end", "fixed_rate", "direction", "notional",
        "fixed_frequency", "float_frequency", "fixed_day_count", "float_day_count",
        "spread",
    ]),
    # ois_swap and basis_swap handled via custom serialisers below
    # (they don't store all constructor args as direct attributes)
    "deposit": ("pricebook.deposit.Deposit", [
        "start", "end", "rate", "notional", "day_count",
    ]),
    "fra": ("pricebook.fra.FRA", [
        "start", "end", "strike", "notional",
    ]),
    # Bonds
    "bond": ("pricebook.bond.FixedRateBond", [
        "issue_date", "maturity", "coupon_rate", "frequency", "face_value", "day_count",
    ]),
    # Credit
    "cds": ("pricebook.cds.CDS", [
        "start", "end", "spread", "notional", "recovery", "frequency", "day_count",
    ]),
    "cln": ("pricebook.cln.CreditLinkedNote", [
        "start", "end", "coupon_rate", "notional", "recovery", "leverage",
        "floating", "frequency", "day_count",
    ]),
    # Options
    "swaption": ("pricebook.swaption.Swaption", [
        "expiry", "swap_end", "strike", "swaption_type", "notional",
        "fixed_frequency", "float_frequency", "fixed_day_count", "float_day_count",
    ]),
    "capfloor": ("pricebook.capfloor.CapFloor", [
        "start", "end", "strike", "option_type", "notional", "frequency", "day_count",
    ]),
    # Loans
    "term_loan": ("pricebook.loan.TermLoan", [
        "start", "end", "spread", "notional", "amort_rate", "frequency", "day_count",
    ]),
    "revolver": ("pricebook.loan.RevolvingFacility", [
        "start", "end", "max_commitment", "drawn_amount", "drawn_spread",
        "undrawn_fee", "frequency", "day_count",
    ]),
    # FX
    "fx_forward": ("pricebook.fx_forward.FXForward", [
        "pair", "maturity", "strike", "notional",
    ]),
}

# TRS handled separately due to polymorphic underlying
_TRS_MODULE = "pricebook.trs.TotalReturnSwap"
_TRS_FIELDS = [
    "notional", "start", "end", "repo_spread", "haircut", "sigma",
]


_CLASS_TO_TYPE = {
    "InterestRateSwap": "irs",
    "OISSwap": "ois_swap",
    "BasisSwap": "basis_swap",
    "Deposit": "deposit",
    "FixedRateBond": "bond",
    "FRA": "fra",
    "CDS": "cds",
    "CreditLinkedNote": "cln",
    "Swaption": "swaption",
    "CapFloor": "capfloor",
    "TermLoan": "term_loan",
    "RevolvingFacility": "revolver",
    "FXForward": "fx_forward",
    "TotalReturnSwap": "trs",
}


def instrument_to_dict(inst) -> dict[str, Any]:
    """Serialize any registered instrument to a dict."""
    type_key = _CLASS_TO_TYPE.get(type(inst).__name__)
    if type_key is None:
        raise ValueError(f"No serialization registered for {type(inst).__name__}")

    # Custom handlers for instruments that don't store all constructor args
    if type_key == "trs":
        return _trs_to_dict(inst)
    if type_key == "ois_swap":
        return _ois_swap_to_dict(inst)
    if type_key == "basis_swap":
        return _basis_swap_to_dict(inst)

    _, fields = _INSTRUMENT_FIELDS[type_key]

    params = {}
    for f in fields:
        v = getattr(inst, f)
        params[f] = _serialize_value(v)

    return {"type": type_key, "params": params}


def _ois_swap_to_dict(ois) -> dict[str, Any]:
    """Custom OISSwap serialisation: extract fixed_rate from fixed leg."""
    return {"type": "ois_swap", "params": {
        "start": _date_to_str(ois.start),
        "end": _date_to_str(ois.end),
        "fixed_rate": ois.fixed_leg.rate,
        "notional": ois.notional,
        "day_count": _enum_to_str(ois.day_count),
    }}


def _basis_swap_to_dict(bs) -> dict[str, Any]:
    """Custom BasisSwap serialisation: extract start/end from legs."""
    start = bs.leg1.cashflows[0].accrual_start
    end = bs.leg1.cashflows[-1].accrual_end
    return {"type": "basis_swap", "params": {
        "start": _date_to_str(start),
        "end": _date_to_str(end),
        "spread": bs.spread,
        "notional": bs.notional,
    }}


def _trs_to_dict(trs) -> dict[str, Any]:
    """Custom TRS serialisation: handles polymorphic underlying."""
    params: dict[str, Any] = {}
    for f in _TRS_FIELDS:
        params[f] = _serialize_value(getattr(trs, f))

    # Underlying: float (equity spot) or instrument
    underlying = trs.underlying
    if isinstance(underlying, (int, float)):
        params["underlying"] = {"type": "equity_spot", "value": float(underlying)}
    else:
        params["underlying"] = instrument_to_dict(underlying)

    return {"type": "trs", "params": params}


def _trs_from_dict(params: dict) -> Any:
    """Custom TRS deserialisation: resolves polymorphic underlying."""
    cls = _import_class(_TRS_MODULE)

    # Resolve underlying
    underlying_d = params.pop("underlying")
    if isinstance(underlying_d, (int, float)):
        underlying = float(underlying_d)
    elif isinstance(underlying_d, dict):
        if underlying_d.get("type") == "equity_spot":
            underlying = float(underlying_d["value"])
        else:
            underlying = instrument_from_dict(underlying_d)
    else:
        underlying = float(underlying_d)

    # Resolve dates
    for key in ("start", "end"):
        if key in params and isinstance(params[key], str):
            params[key] = _str_to_date(params[key])

    return cls(underlying=underlying, **params)


def instrument_from_dict(d: dict[str, Any]):
    """Deserialize an instrument from a dict."""
    type_key = d["type"]
    params = dict(d["params"])

    # Custom handlers
    if type_key == "trs":
        return _trs_from_dict(params)
    if type_key == "ois_swap":
        _resolve_params(type_key, params)
        cls = _import_class("pricebook.ois.OISSwap")
        return cls(**params)
    if type_key == "basis_swap":
        _resolve_params(type_key, params)
        cls = _import_class("pricebook.basis_swap.BasisSwap")
        return cls(**params)

    if type_key not in _INSTRUMENT_FIELDS:
        raise ValueError(f"Unknown instrument type: {type_key}")

    module_class, fields = _INSTRUMENT_FIELDS[type_key]

    # Resolve enums and dates in params
    _resolve_params(type_key, params)

    # Import and construct
    cls = _import_class(module_class)
    return cls(**params)


_DATE_FIELDS = frozenset(("start", "end", "expiry", "swap_end", "issue_date", "maturity"))
_FREQ_FIELDS = frozenset(("frequency", "fixed_frequency", "float_frequency",
                           "leg1_frequency", "leg2_frequency"))
_DC_FIELDS = frozenset(("day_count", "fixed_day_count", "float_day_count"))


def _resolve_params(type_key: str, params: dict) -> None:
    """Convert string values back to dates and enums."""
    from pricebook.day_count import DayCountConvention
    from pricebook.schedule import Frequency

    for key, val in list(params.items()):
        if val is None:
            continue

        if isinstance(val, str) and key in _DATE_FIELDS:
            params[key] = _str_to_date(val)
        elif key in _FREQ_FIELDS and isinstance(val, (str, int)):
            params[key] = Frequency(int(val))
        elif key in _DC_FIELDS and isinstance(val, str):
            params[key] = DayCountConvention(val)
        elif key == "direction" and isinstance(val, str):
            from pricebook.swap import SwapDirection
            params[key] = SwapDirection(val)
        elif key == "swaption_type" and isinstance(val, str):
            from pricebook.swaption import SwaptionType
            params[key] = SwaptionType(val)
        elif key == "option_type" and isinstance(val, str):
            from pricebook.black76 import OptionType
            params[key] = OptionType(val)
        elif key == "pair" and isinstance(val, str):
            from pricebook.currency import CurrencyPair, Currency
            base_s, quote_s = val.split("/")
            params[key] = CurrencyPair(Currency(base_s), Currency(quote_s))


def _import_class(dotted_path: str):
    """Import a class from a dotted module path like 'pricebook.swap.InterestRateSwap'."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 2:
        module_path, class_name = parts
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise ValueError(f"Cannot parse class path: {dotted_path}")


# ---------------------------------------------------------------------------
# Trade serialization
# ---------------------------------------------------------------------------


def trade_to_dict(trade) -> dict[str, Any]:
    """Serialize a Trade to a dict."""
    d: dict[str, Any] = {
        "direction": trade.direction,
        "notional_scale": trade.notional_scale,
        "trade_id": trade.trade_id,
        "counterparty": trade.counterparty,
    }
    if trade.trade_date:
        d["trade_date"] = _date_to_str(trade.trade_date)
    d["instrument"] = instrument_to_dict(trade.instrument)
    return d


def trade_from_dict(d: dict[str, Any]):
    """Deserialize a Trade from a dict."""
    from pricebook.trade import Trade
    inst = instrument_from_dict(d["instrument"])
    return Trade(
        instrument=inst,
        direction=d.get("direction", 1),
        notional_scale=d.get("notional_scale", 1.0),
        trade_date=_str_to_date(d["trade_date"]) if d.get("trade_date") else None,
        counterparty=d.get("counterparty", ""),
        trade_id=d.get("trade_id", ""),
    )


def portfolio_to_dict(portfolio) -> dict[str, Any]:
    """Serialize a Portfolio to a dict."""
    return {
        "name": portfolio.name,
        "trades": [trade_to_dict(t) for t in portfolio.trades],
    }


def portfolio_from_dict(d: dict[str, Any]):
    """Deserialize a Portfolio from a dict."""
    from pricebook.trade import Portfolio
    trades = [trade_from_dict(td) for td in d["trades"]]
    return Portfolio(trades=trades, name=d.get("name", ""))


# ---------------------------------------------------------------------------
# JSON convenience
# ---------------------------------------------------------------------------


def to_json(obj, **kwargs) -> str:
    """Serialize any supported object to JSON string."""
    from pricebook.pricing_context import PricingContext
    from pricebook.trade import Trade, Portfolio
    from pricebook.discount_curve import DiscountCurve
    from pricebook.survival_curve import SurvivalCurve

    if isinstance(obj, PricingContext):
        d = pricing_context_to_dict(obj)
    elif isinstance(obj, DiscountCurve):
        d = discount_curve_to_dict(obj)
    elif isinstance(obj, SurvivalCurve):
        d = survival_curve_to_dict(obj)
    elif isinstance(obj, Portfolio):
        d = portfolio_to_dict(obj)
    elif isinstance(obj, Trade):
        d = trade_to_dict(obj)
    else:
        # Try instrument registry (covers all registered types)
        d = instrument_to_dict(obj)

    return json.dumps(d, indent=2, **kwargs)


def from_json(s: str):
    """Deserialize any supported object from a JSON string."""
    d = json.loads(s)
    if "trades" in d:
        return portfolio_from_dict(d)
    if "instrument" in d:
        return trade_from_dict(d)
    if "valuation_date" in d and "type" not in d:
        return pricing_context_from_dict(d)
    if "type" in d:
        t = d["type"]
        if t == "DiscountCurve":
            return discount_curve_from_dict(d)
        if t == "SurvivalCurve":
            return survival_curve_from_dict(d)
        return instrument_from_dict(d)
    raise ValueError("Cannot determine object type from JSON")


# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------


def get_instrument_class(type_key: str):
    """Get an instrument class by type key (e.g. 'irs', 'cds')."""
    if type_key not in _INSTRUMENT_FIELDS:
        raise KeyError(
            f"Unknown instrument type '{type_key}'. "
            f"Available: {list(_INSTRUMENT_FIELDS.keys())}"
        )
    module_class, _ = _INSTRUMENT_FIELDS[type_key]
    return _import_class(module_class)


def list_instruments() -> list[str]:
    """List all registered instrument type keys."""
    return list(_INSTRUMENT_FIELDS.keys())


def load_trade(data: dict[str, Any]) -> "Trade":
    """Load a single trade from a dict. Convenience wrapper around trade_from_dict."""
    return trade_from_dict(data)


def load_portfolio(data: list[dict] | dict) -> "Portfolio":
    """Load a portfolio from a list of trade dicts or a portfolio dict.

    Accepts either:
        - A portfolio dict: {"name": "...", "trades": [...]}
        - A plain list of trade dicts: [{...}, {...}]
    """
    from pricebook.trade import Portfolio
    if isinstance(data, list):
        trades = [trade_from_dict(td) for td in data]
        return Portfolio(trades=trades)
    return portfolio_from_dict(data)
