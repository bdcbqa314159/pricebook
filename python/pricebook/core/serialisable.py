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

For frozen dataclasses (convention objects), use @serialisable_convention:
    @serialisable_convention("sovereign_conventions")
    @dataclass(frozen=True)
    class SovereignConventions:
        market_code: str
        country: str
        ...

    # Fields auto-derived from dataclasses.fields().
    # to_dict() returns flat dict (no "type"/"params" nesting — pure data).
    # from_dict() reconstructs with enum/date resolution.

How it works:
    to_dict():  reads self.X for each field in _SERIAL_FIELDS
                auto-converts: date→isoformat, Enum→.value, Serialisable→.to_dict()
                emits `schema_version` (top-level) or `_schema_version`
                (in flat convention dicts) carrying `_SERIAL_SCHEMA_VERSION`.

    from_dict(): reads __init__ type hints to know how to deserialise each field
                 auto-converts: str→date, str→Enum, dict→from_dict()
                 validates schema_version — absent → v1; future version → raise.

For classes with polymorphic fields (e.g. TRS.underlying), override from_dict.

Schema versioning
-----------------

Every serialised dict carries the writer's `_SERIAL_SCHEMA_VERSION`. The
reader checks the version is recognised; existing data without the field
is treated as v1 silently. Classes bump `_SERIAL_SCHEMA_VERSION` when
they make a *breaking* wire-format change (renamed field, changed unit,
restructured payload) and either implement migration from older versions
or refuse to deserialise them with a clear error.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, get_type_hints, Union, get_origin, get_args

# ---------------------------------------------------------------------------
# Registry: type_key → class
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Schema-version helpers
# ---------------------------------------------------------------------------

# Top-level key in non-convention payloads ("type"/"params" envelope).
_SCHEMA_VERSION_KEY = "schema_version"

# Key used in flat convention payloads. Underscore prefix avoids any clash
# with real convention fields (which are conventional Python identifiers).
_SCHEMA_VERSION_KEY_FLAT = "_schema_version"


def _check_schema_version(cls: type, found: int | None) -> None:
    """Raise if `found` exceeds the class's declared `_SERIAL_SCHEMA_VERSION`.

    Absent (`None`) is treated as v1 — that's what existing on-disk and
    in-test payloads serialised before this slice carry, and we never want
    to break them.

    Classes that introduce a *breaking* wire change bump
    `_SERIAL_SCHEMA_VERSION` and are responsible for handling older versions
    (migrate or raise with their own message).
    """
    expected = getattr(cls, "_SERIAL_SCHEMA_VERSION", 1)
    v = 1 if found is None else int(found)
    if v > expected:
        raise ValueError(
            f"Cannot deserialise {cls.__name__}: payload says "
            f"{_SCHEMA_VERSION_KEY}={v}, but this build only supports up to "
            f"{expected}. The payload was written by a newer pricebook — "
            f"upgrade this environment."
        )


def make_payload(instance: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Build the standard `{type, params, schema_version}` envelope for a
    `Serialisable` instance whose `to_dict` is hand-written.

    Classes that override `to_dict` (rather than inheriting the auto-generated
    one) MUST go through this helper instead of hand-rolling the dict —
    otherwise `schema_version` is forgotten (audit B.1 B2). Wider context: G1
    P3 Slice 2 added the schema-version slot but only to `Serialisable.to_dict`
    and the two decorators; custom overrides were quietly outside that
    coverage. `make_payload` closes the gap.

        def _my_to_dict(self):
            return make_payload(self, {
                "foo": self.foo,
                "bar": _serialise_atom(self.bar),
            })
    """
    return {
        "type": instance._SERIAL_TYPE,
        "params": params,
        _SCHEMA_VERSION_KEY: getattr(instance, "_SERIAL_SCHEMA_VERSION", 1),
    }


def read_payload(d: dict[str, Any], cls: type) -> dict[str, Any]:
    """Inverse of `make_payload`: validate `schema_version` against `cls`'s
    declared `_SERIAL_SCHEMA_VERSION` and return the `params` dict.

    Replaces hand-rolled `p = d["params"]` in custom `from_dict` overrides.

        @classmethod
        def _my_from_dict(cls, d):
            p = read_payload(d, cls)
            return cls(foo=p["foo"], ...)
    """
    _check_schema_version(cls, d.get(_SCHEMA_VERSION_KEY))
    return d["params"]


def _register(cls: type) -> None:
    """Register a class. Validates _SERIAL_FIELDS exist on the class or its __init__."""
    key = getattr(cls, "_SERIAL_TYPE", None)
    if key and key not in _REGISTRY:
        # Validate fields at registration time (catches typos early)
        fields = getattr(cls, "_SERIAL_FIELDS", [])
        if fields:
            import inspect
            init_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
            for f in fields:
                if f not in init_params:
                    import warnings
                    warnings.warn(
                        f"Serialisation: {cls.__name__}._SERIAL_FIELDS contains '{f}' "
                        f"which is not in __init__ parameters: {sorted(init_params)}",
                        stacklevel=3,
                    )
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
    """Convert a single value to a JSON-native type.

    Handles: None, date, Enum, bool, int, float, str, numpy scalars,
    nested serialisable objects, lists, CurrencyPair.
    """
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
    # numpy scalars → native Python (np.float64 → float, np.int64 → int)
    if hasattr(v, "item") and callable(v.item):
        return v.item()
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

    # Unwrap Optional (X | None) — handles both typing.Union and types.UnionType (3.10+)
    import types as _types
    origin = get_origin(hint)
    if origin is Union or isinstance(hint, _types.UnionType):
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
            # Convention objects use flat dicts (no "type"/"params" nesting).
            # If the dict has no "type" key, call hint.from_dict directly.
            if "type" in v:
                return from_dict(v)
            return hint.from_dict(v)
        return v

    # CurrencyPair — deserialise from "EUR/USD" string
    if isinstance(hint, type) and hint.__name__ == "CurrencyPair":
        if isinstance(v, str) and "/" in v:
            from pricebook.core.currency import CurrencyPair, Currency
            base_str, quote_str = v.split("/")
            return CurrencyPair(Currency(base_str), Currency(quote_str))
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
        _SERIAL_SCHEMA_VERSION: int  — bump on breaking wire-format change

    Gets for free:
        to_dict() → {"type": _SERIAL_TYPE, "params": {...},
                     "schema_version": _SERIAL_SCHEMA_VERSION}
        from_dict(d) → cls(**resolved_params)  (validates schema_version)

    Override from_dict() for special cases (polymorphic fields).
    """

    _SERIAL_TYPE: str = ""
    _SERIAL_FIELDS: list[str] = []
    _SERIAL_SCHEMA_VERSION: int = 1

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
        return {
            "type": self._SERIAL_TYPE,
            "params": params,
            _SCHEMA_VERSION_KEY: self._SERIAL_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        _check_schema_version(cls, d.get(_SCHEMA_VERSION_KEY))
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
    """Get type hints from __init__. Returns empty dict if hints can't be resolved."""
    try:
        return get_type_hints(cls.__init__)
    except Exception as e:
        import warnings
        warnings.warn(
            f"Could not resolve type hints for {cls.__name__}.__init__: {e}. "
            f"from_dict() will not auto-resolve dates/enums for this class.",
            stacklevel=3,
        )
        return {}


# ---------------------------------------------------------------------------
# Decorator alternative (for classes that can't inherit)
# ---------------------------------------------------------------------------

def serialisable(serial_type: str, fields: list[str], schema_version: int = 1):
    """Decorator: add Serialisable behaviour without inheritance.

    @serialisable("irs", ["start", "end", "fixed_rate", ...])
    class InterestRateSwap:
        ...

    Pass `schema_version=N` (N >= 2) when a class introduces a *breaking*
    wire-format change. Older payloads (absent version, or version < N) are
    handled by the class's own from_dict — bump the version AND implement
    the migration logic, never one without the other.
    """
    def decorator(cls):
        cls._SERIAL_TYPE = serial_type
        cls._SERIAL_FIELDS = fields
        cls._SERIAL_SCHEMA_VERSION = schema_version

        def to_dict(self) -> dict[str, Any]:
            params = {}
            for field in self._SERIAL_FIELDS:
                v = getattr(self, field)
                params[field] = _serialise_atom(v)
            return {
                "type": self._SERIAL_TYPE,
                "params": params,
                _SCHEMA_VERSION_KEY: self._SERIAL_SCHEMA_VERSION,
            }

        @classmethod
        def cls_from_dict(klass, d: dict[str, Any]) -> Any:
            _check_schema_version(klass, d.get(_SCHEMA_VERSION_KEY))
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


# ---------------------------------------------------------------------------
# Convention decorator (for frozen dataclasses — pure data)
# ---------------------------------------------------------------------------

def serialisable_convention(serial_type: str, schema_version: int = 1):
    """Decorator: add serialisation to frozen dataclasses (convention objects).

    Unlike @serialisable, this auto-derives _SERIAL_FIELDS from
    dataclasses.fields() and produces flat dicts (no "type"/"params"
    nesting) — because conventions are pure data, not polymorphic
    instruments.

    @serialisable_convention("sovereign_conventions")
    @dataclass(frozen=True)
    class SovereignConventions:
        market_code: str
        country: str
        currency: str
        frequency: Frequency
        ...

    conv.to_dict()
    # → {"market_code": "UST", "country": "US", "currency": "USD",
    #    "frequency": 6, ..., "_schema_version": 1}

    SovereignConventions.from_dict(d)
    # → SovereignConventions(market_code="UST", ...)

    Bump `schema_version` on a breaking wire-format change. The version
    travels in the flat dict under the reserved key `_schema_version` —
    underscore-prefixed to avoid colliding with real convention fields.
    """
    import dataclasses

    def decorator(cls):
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"@serialisable_convention requires a dataclass, got {cls}")

        field_names = [f.name for f in dataclasses.fields(cls)]
        cls._SERIAL_TYPE = serial_type
        cls._SERIAL_FIELDS = field_names
        cls._SERIAL_SCHEMA_VERSION = schema_version

        # Flat to_dict (no type/params nesting — pure data)
        def to_dict(self) -> dict[str, Any]:
            d = {}
            for field in field_names:
                v = getattr(self, field)
                d[field] = _serialise_atom(v)
            d[_SCHEMA_VERSION_KEY_FLAT] = self._SERIAL_SCHEMA_VERSION
            return d

        @classmethod
        def cls_from_dict(klass, d: dict[str, Any]) -> Any:
            # Accept both flat dict and {"type": ..., "params": {...}} format
            if "params" in d and "type" in d:
                p = d["params"]
                version = d.get(_SCHEMA_VERSION_KEY)
            else:
                p = d
                version = d.get(_SCHEMA_VERSION_KEY_FLAT)
            _check_schema_version(klass, version)
            hints = _get_init_hints(klass)
            kwargs = {}
            for field in field_names:
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
        # Register so from_dict dispatch works for nested conventions
        _register(cls)
        return cls

    return decorator
