"""Serialization — canonical entry point.

All serialisation goes through pricebook.serialisable. This module
re-exports the public API and provides backward-compat aliases.

    from pricebook.core.serialization import to_dict, from_dict, to_json, from_json

    d = to_dict(my_irs)       # calls my_irs.to_dict()
    irs = from_dict(d)        # dispatches via registry
    s = to_json(my_irs)       # to_dict + json.dumps
    obj = from_json(s)        # json.loads + from_dict
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pricebook.core.serialisable import (
    _REGISTRY,
    _register,
    _serialise_atom as serialise_value,
    from_dict as _new_from_dict,
)


# ---------------------------------------------------------------------------
# Lazy registration — auto-discover every @serialisable* module under pricebook
# ---------------------------------------------------------------------------
#
# Why auto-discovery (and not a curated import list)?
# A curated whitelist is a silent-broken footgun: every new module with
# `@serialisable` / `@serialisable_convention` would have to be remembered
# and added here. Audit A.12 B1 found ~29 modules with @serialisable types
# already missing from the curated list. `from_dict` would raise "Unknown
# type" only when that *specific* type happened to be deserialised — by
# which time the failure mode is mysterious.
#
# pkgutil.walk_packages + importlib.import_module walks the entire pricebook
# tree once on first use, populates the registry transparently, and skips
# any module that fails to import (which would be a separate bug —
# import-time errors are not silenced here, they're just not allowed to
# block deserialisation of other types).

_loaded = False
_failed_imports: list[tuple[str, str]] = []


def _ensure_loaded() -> None:
    """Auto-discover and import every submodule of `pricebook` so that all
    classes decorated with `@serialisable` / `@serialisable_convention` (and
    direct calls to `_register`) populate the global registry.

    Idempotent — guarded by a module-level flag. First call walks the tree;
    subsequent calls are O(1).

    Failures: any module that fails to import is recorded in
    `_failed_imports` (name, exception-class-name). The audit test
    `test_serialization_autodiscovery` asserts this list is empty so we
    catch regressions at CI time rather than at deserialise time.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    import importlib
    import pkgutil
    import pricebook

    for module_info in pkgutil.walk_packages(
        pricebook.__path__, prefix="pricebook."
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 — any import failure is recorded
            _failed_imports.append((module_info.name, type(exc).__name__))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_dict(obj) -> dict[str, Any]:
    """Serialize any object that has to_dict()."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise ValueError(f"{type(obj).__name__} has no to_dict() method")


def from_dict(d: dict[str, Any]):
    """Deserialize from {"type": ..., "params": {...}}."""
    _ensure_loaded()
    return _new_from_dict(d)


def to_json(obj, **kwargs) -> str:
    """Serialize to JSON string."""
    return json.dumps(to_dict(obj), indent=2, **kwargs)


def from_json(s: str):
    """Deserialize from JSON string."""
    return from_dict(json.loads(s))


def registered_types() -> list[str]:
    """List all registered type keys."""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Backward-compat aliases (used by pricing_engine.py, pricing_server.py, etc.)
# ---------------------------------------------------------------------------

_str_to_date = date.fromisoformat

def deserialise_date(s: str) -> date:
    return date.fromisoformat(s)

def deserialise_enum(enum_cls, v):
    from enum import Enum
    if isinstance(v, enum_cls):
        return v
    return enum_cls(v)

def deserialise_currency_pair(s: str):
    from pricebook.core.currency import CurrencyPair, Currency
    base_s, quote_s = s.split("/")
    return CurrencyPair(Currency(base_s), Currency(quote_s))

# Old names → new dispatch
instrument_to_dict = to_dict
instrument_from_dict = from_dict
list_instruments = registered_types
trade_to_dict = to_dict
trade_from_dict = from_dict
portfolio_to_dict = to_dict
portfolio_from_dict = from_dict
discount_curve_to_dict = to_dict
discount_curve_from_dict = from_dict
survival_curve_to_dict = to_dict
survival_curve_from_dict = from_dict
spread_curve_to_dict = to_dict
spread_curve_from_dict = from_dict
ibor_curve_to_dict = to_dict
ibor_curve_from_dict = from_dict
funding_curve_to_dict = to_dict
funding_curve_from_dict = from_dict
csa_to_dict = to_dict
csa_from_dict = from_dict
multi_currency_curves_to_dict = to_dict
multi_currency_curves_from_dict = from_dict
pricing_context_to_dict = to_dict
pricing_context_from_dict = from_dict

def get_instrument_class(type_key: str):
    _ensure_loaded()
    if type_key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown type '{type_key}'. Available: {available}")
    return _REGISTRY[type_key]

def load_trade(data: dict[str, Any]):
    """Load a Trade from a dict. Supports both new and legacy formats."""
    if "type" in data and data["type"] == "trade":
        return from_dict(data)
    # Legacy format: {"instrument": {...}, "trade_id": ..., ...}
    if "instrument" in data:
        from pricebook.core.trade import Trade
        inst = from_dict(data["instrument"])
        return Trade(
            instrument=inst,
            direction=data.get("direction", 1),
            notional_scale=data.get("notional_scale", 1.0),
            trade_date=date.fromisoformat(data["trade_date"]) if data.get("trade_date") else None,
            counterparty=data.get("counterparty", ""),
            trade_id=data.get("trade_id", ""),
        )
    return from_dict(data)

def load_portfolio(data):
    """Load a Portfolio. Accepts list of trade dicts or portfolio dict."""
    from pricebook.core.trade import Portfolio
    if isinstance(data, list):
        trades = [load_trade(td) for td in data]
        return Portfolio(trades=trades)
    if "type" in data and data["type"] == "portfolio":
        return from_dict(data)
    # Legacy: {"name": ..., "trades": [...]}
    if "trades" in data:
        trades = [load_trade(td) for td in data["trades"]]
        return Portfolio(trades=trades, name=data.get("name", ""))
    return from_dict(data)
