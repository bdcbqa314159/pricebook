"""Regression tests for the A.11 B4-B7 serialisable LOW fixes landed in T-LOW-CLEANUP.

* B4 — numpy `.item()` duck-test is narrowed to `isinstance(v, np.generic)`;
  objects with a callable `.item` that aren't numpy scalars must not be
  silently flattened.
* B5 — `CurrencyPair` deserialisation rejects malformed payloads
  ("EUR" or "EUR/USD/extra") with a clear ValueError instead of crashing
  on unpack or silently dropping the third token.
* B6 — Missing "params" key in a `Serialisable.from_dict` payload raises a
  structured `ValueError` naming the class instead of bare `KeyError`.
* B7 — `IntEnum` field that comes back as a JSON string (numeric) round-trips
  cleanly via int-coercion.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pytest

from pricebook.core.serialisable import (
    Serialisable,
    _deserialise_atom,
    _serialise_atom,
)


# ── B4 — numpy duck-test narrow ──────────────────────────────────────


class _ItemMethodNotNumpy:
    """Has a callable .item attribute but isn't numpy. Pre-fix would have
    been flattened by the broad duck-test."""

    def item(self):
        return "i-am-not-a-numpy-scalar"


def test_b4_numpy_item_narrowed_to_np_generic():
    np_scalar = np.float64(1.5)
    assert _serialise_atom(np_scalar) == 1.5

    not_numpy = _ItemMethodNotNumpy()
    # Should NOT call .item() on this — it falls through to the final
    # "return v" branch unchanged.
    result = _serialise_atom(not_numpy)
    assert result is not_numpy


# ── B5 — graceful CurrencyPair parse ─────────────────────────────────


def test_b5_currency_pair_too_few_parts_raises():
    from pricebook.core.currency import CurrencyPair
    with pytest.raises(ValueError, match="CurrencyPair payload must be"):
        _deserialise_atom("EUR", CurrencyPair)


def test_b5_currency_pair_too_many_parts_raises():
    from pricebook.core.currency import CurrencyPair
    with pytest.raises(ValueError, match="CurrencyPair payload must be"):
        _deserialise_atom("EUR/USD/extra", CurrencyPair)


def test_b5_currency_pair_round_trip_unchanged():
    from pricebook.core.currency import CurrencyPair, Currency
    pair = _deserialise_atom("EUR/USD", CurrencyPair)
    assert pair == CurrencyPair(Currency.EUR, Currency.USD)


# ── B6 — structured ValueError on missing "params" ──────────────────


class _DemoSerialisable(Serialisable):
    _SERIAL_TYPE = "demo_low_fix"
    _SERIAL_FIELDS = ["x"]

    def __init__(self, x: int = 0):
        self.x = x


def test_b6_missing_params_key_raises_structured():
    with pytest.raises(ValueError, match="missing required 'params' key"):
        _DemoSerialisable.from_dict({"type": "demo_low_fix"})


def test_b6_missing_params_message_names_the_class():
    with pytest.raises(ValueError, match="_DemoSerialisable"):
        _DemoSerialisable.from_dict({"type": "demo_low_fix"})


# ── B7 — IntEnum from string ─────────────────────────────────────────


class _DemoIntEnum(IntEnum):
    FOO = 1
    BAR = 3


def test_b7_int_enum_from_string():
    # JSON typically round-trips small ints as strings; the deserialiser
    # must coerce before the Enum lookup.
    assert _deserialise_atom("3", _DemoIntEnum) is _DemoIntEnum.BAR


def test_b7_int_enum_from_int_still_works():
    assert _deserialise_atom(1, _DemoIntEnum) is _DemoIntEnum.FOO


def test_b7_int_enum_already_member_passthrough():
    assert _deserialise_atom(_DemoIntEnum.FOO, _DemoIntEnum) is _DemoIntEnum.FOO


class TestRegistryCollision:
    """v1.236: a duplicate _SERIAL_TYPE must fail loud, not silently drop a class."""

    def test_distinct_class_same_serial_type_raises(self):
        from pricebook.core import serialisable as S

        class _CollideA(S.Serialisable):
            _SERIAL_TYPE = "collide_test_key"
            _SERIAL_FIELDS = ["x"]

            def __init__(self, x=1):
                self.x = x

        with pytest.raises(ValueError, match="collision"):
            class _CollideB(S.Serialisable):
                _SERIAL_TYPE = "collide_test_key"
                _SERIAL_FIELDS = ["y"]

                def __init__(self, y=2):
                    self.y = y

    def test_same_class_reregister_is_idempotent(self):
        from pricebook.core import serialisable as S

        class _IdemProbe(S.Serialisable):
            _SERIAL_TYPE = "idem_test_key"
            _SERIAL_FIELDS = ["z"]

            def __init__(self, z=3):
                self.z = z

        S._register(_IdemProbe)  # again — must not raise
        S._register(_IdemProbe)
