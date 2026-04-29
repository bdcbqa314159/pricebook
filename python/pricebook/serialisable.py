"""Serialisable mixin: auto to_dict/from_dict from constructor annotations.

The atom of the serialisation system. Every class that crosses the wire
inherits from Serialisable or uses the @serialisable decorator.

Usage — two things to declare:
    class MyInstrument(Serialisable):
        _SERIAL_TYPE = "my_instrument"
        _SERIAL_FIELDS = ["start", "end", "rate", "notional", "day_count"]

        def __init__(self, start: date, end: date, rate: float, ...):
            self.start = start
            ...

That's it. to_dict() and from_dict() are auto-generated.

How it works:
    to_dict():  reads self.X for each field in _SERIAL_FIELDS
                auto-converts: date→isoformat, Enum→.value, Serialisable→.to_dict()

    from_dict(): reads __init__ type hints to know how to deserialise each field
                 auto-converts: str→date, str→Enum, dict→from_dict()

For classes with polymorphic fields (e.g. TRS.underlying), override from_dict.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, get_type_hints, Union, get_origin, get_args

# ---------------------------------------------------------------------------
# Registry: type_key → class
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def _register(cls: type) -> None:
    key = getattr(cls, "_SERIAL_TYPE", None)
    if key and key not in _REGISTRY:
        _REGISTRY[key] = cls


def from_dict(d: dict[str, Any]) -> Any:
    """Deserialise from {"type": ..., "params": {...}}. Registry dispatch."""
    t = d.get("type")
    if t is None:
        raise ValueError("Dict has no 'type' key")
    if t not in _REGISTRY:
        raise ValueError(f"Unknown type '{t}'. Registered: {', '.join(sorted(_REGISTRY.keys()))}")
    return _REGISTRY[t].from_dict(d)


# ---------------------------------------------------------------------------
# Atom serialisers: value → JSON-native
# ---------------------------------------------------------------------------

def _serialise_atom(v: Any) -> Any:
    """Convert a single value to a JSON-native type."""
    if v is None:
        return None
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    # Nested serialisable
    if hasattr(v, "to_dict"):
        return v.to_dict()
    # List of dates or serialisables
    if isinstance(v, list):
        return [_serialise_atom(x) for x in v]
    # CurrencyPair (has .base.value, .quote.value)
    if hasattr(v, "base") and hasattr(v, "quote") and hasattr(v.base, "value"):
        return f"{v.base.value}/{v.quote.value}"
    return v


def _deserialise_atom(v: Any, hint: type) -> Any:
    """Convert a JSON value back to the target type using the type hint."""
    if v is None:
        return None

    # Unwrap Optional (X | None)
    origin = get_origin(hint)
    if origin is Union:
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            hint = args[0]
        else:
            # Union of multiple types — can't auto-resolve, return as-is
            return v

    # date
    if hint is date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(v)

    # Enum subclass
    if isinstance(hint, type) and issubclass(hint, Enum):
        if isinstance(v, hint):
            return v
        return hint(v)

    # Nested serialisable (has _SERIAL_TYPE)
    if isinstance(hint, type) and hasattr(hint, "_SERIAL_TYPE"):
        if isinstance(v, dict):
            return from_dict(v)
        return v

    # list[date] — check if hint is list[X]
    if get_origin(hint) is list:
        args = get_args(hint)
        if args and args[0] is date:
            return [date.fromisoformat(x) if isinstance(x, str) else x for x in v]

    # Primitives
    return v


# ---------------------------------------------------------------------------
# The Mixin
# ---------------------------------------------------------------------------

class Serialisable:
    """Mixin for auto-serialisation.

    Subclass declares:
        _SERIAL_TYPE: str          — wire type key
        _SERIAL_FIELDS: list[str]  — constructor param names to serialise

    Gets for free:
        to_dict() → {"type": _SERIAL_TYPE, "params": {...}}
        from_dict(d) → cls(**resolved_params)

    Override from_dict() for special cases (polymorphic fields).
    """

    _SERIAL_TYPE: str = ""
    _SERIAL_FIELDS: list[str] = []

    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses that have _SERIAL_TYPE."""
        super().__init_subclass__(**kwargs)
        if getattr(cls, "_SERIAL_TYPE", ""):
            _register(cls)

    def to_dict(self) -> dict[str, Any]:
        params = {}
        for field in self._SERIAL_FIELDS:
            v = getattr(self, field)
            params[field] = _serialise_atom(v)
        return {"type": self._SERIAL_TYPE, "params": params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        p = d["params"]
        hints = _get_init_hints(cls)
        kwargs = {}
        for field in cls._SERIAL_FIELDS:
            if field not in p:
                continue  # use constructor default
            v = p[field]
            if field in hints:
                kwargs[field] = _deserialise_atom(v, hints[field])
            else:
                kwargs[field] = v
        return cls(**kwargs)


def _get_init_hints(cls: type) -> dict[str, type]:
    """Get type hints from __init__, cached."""
    try:
        return get_type_hints(cls.__init__)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Decorator alternative (for classes that can't inherit)
# ---------------------------------------------------------------------------

def serialisable(serial_type: str, fields: list[str]):
    """Decorator: add Serialisable behaviour without inheritance.

    @serialisable("irs", ["start", "end", "fixed_rate", ...])
    class InterestRateSwap:
        ...
    """
    def decorator(cls):
        cls._SERIAL_TYPE = serial_type
        cls._SERIAL_FIELDS = fields

        def to_dict(self) -> dict[str, Any]:
            params = {}
            for field in self._SERIAL_FIELDS:
                v = getattr(self, field)
                params[field] = _serialise_atom(v)
            return {"type": self._SERIAL_TYPE, "params": params}

        @classmethod
        def cls_from_dict(klass, d: dict[str, Any]) -> Any:
            p = d["params"]
            hints = _get_init_hints(klass)
            kwargs = {}
            for field in klass._SERIAL_FIELDS:
                if field not in p:
                    continue
                v = p[field]
                if field in hints:
                    kwargs[field] = _deserialise_atom(v, hints[field])
                else:
                    kwargs[field] = v
            return klass(**kwargs)

        cls.to_dict = to_dict
        cls.from_dict = cls_from_dict
        _register(cls)
        return cls

    return decorator
