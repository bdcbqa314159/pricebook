"""Tests for `_deserialise_atom` polymorphic dispatch (fixes A.11 B1 + B2).

Pre-fix limitations (documented in `AUDIT_L0_CORE.md` A.11):

- B1: `_deserialise_atom` only reconstructed `list[date]`. Any other
  parameterised list (`list[SomeSerialisable]`, `list[Enum]`) silently
  returned the raw list of dicts/values.
- B2: `Union[A, B, None]` with 3+ args returned the raw value, bypassing
  registry-dispatched reconstruction even when the value was a tagged dict
  with a `"type"` key.

This slice adds proper dispatch for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pytest

from pricebook.core.serialisable import (
    Serialisable,
    _deserialise_atom,
    from_dict,
    make_payload,
    read_payload,
    serialisable,
    serialisable_convention,
)


# ============================================================
# Fixtures — small Serialisable types only used in these tests
# ============================================================

class _Leaf(Serialisable):
    _SERIAL_TYPE = "_polymorphic_test_leaf"
    _SERIAL_FIELDS = ["x"]

    def __init__(self, x: int):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, _Leaf) and self.x == other.x


class _Branch(Serialisable):
    _SERIAL_TYPE = "_polymorphic_test_branch"
    _SERIAL_FIELDS = ["label"]

    def __init__(self, label: str):
        self.label = label

    def __eq__(self, other):
        return isinstance(other, _Branch) and self.label == other.label


@serialisable_convention("_polymorphic_test_convention")
@dataclass(frozen=True)
class _Conv:
    name: str
    n: int


class _Colour(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# A container whose __init__ has parameterised-list and polymorphic-Union hints.
class _Container(Serialisable):
    _SERIAL_TYPE = "_polymorphic_test_container"
    _SERIAL_FIELDS = ["leaves", "convs", "colours", "underlying"]

    def __init__(
        self,
        leaves: list[_Leaf],
        convs: list[_Conv],
        colours: list[_Colour],
        underlying: _Leaf | _Branch | None = None,
    ):
        self.leaves = leaves
        self.convs = convs
        self.colours = colours
        self.underlying = underlying


# ============================================================
# A.11 B1 — list[Serialisable] dispatch
# ============================================================

class TestListOfSerialisable:
    def test_list_of_leaves_round_trip(self):
        c = _Container(
            leaves=[_Leaf(1), _Leaf(2), _Leaf(3)],
            convs=[],
            colours=[],
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        assert rebuilt.leaves == [_Leaf(1), _Leaf(2), _Leaf(3)]

    def test_list_of_convention_round_trip(self):
        c = _Container(
            leaves=[],
            convs=[_Conv(name="a", n=1), _Conv(name="b", n=2)],
            colours=[],
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        # Convention dicts are flat (no envelope) — dispatch path takes the
        # else branch and calls `inner.from_dict` directly.
        assert rebuilt.convs == [_Conv(name="a", n=1), _Conv(name="b", n=2)]

    def test_list_of_enum_round_trip(self):
        c = _Container(
            leaves=[],
            convs=[],
            colours=[_Colour.RED, _Colour.BLUE],
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        assert rebuilt.colours == [_Colour.RED, _Colour.BLUE]
        assert all(isinstance(c, _Colour) for c in rebuilt.colours)

    def test_empty_list(self):
        c = _Container(leaves=[], convs=[], colours=[])
        rebuilt = _Container.from_dict(c.to_dict())
        assert rebuilt.leaves == []
        assert rebuilt.convs == []
        assert rebuilt.colours == []


# ============================================================
# A.11 B2 — Union[A, B, None] dispatch
# ============================================================

class TestPolymorphicUnion:
    def test_union_resolves_via_type_tag(self):
        """When the value is a dict with a 'type' key, registry-dispatch
        the underlying type even though the hint has 3+ Union args."""
        c = _Container(
            leaves=[], convs=[], colours=[],
            underlying=_Leaf(42),
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        assert isinstance(rebuilt.underlying, _Leaf)
        assert rebuilt.underlying == _Leaf(42)

    def test_union_other_branch(self):
        """Same hint, different concrete type — the type tag picks the right one."""
        c = _Container(
            leaves=[], convs=[], colours=[],
            underlying=_Branch(label="north"),
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        assert isinstance(rebuilt.underlying, _Branch)
        assert rebuilt.underlying == _Branch(label="north")

    def test_union_with_none(self):
        c = _Container(
            leaves=[], convs=[], colours=[],
            underlying=None,
        )
        d = c.to_dict()
        rebuilt = _Container.from_dict(d)
        assert rebuilt.underlying is None


# ============================================================
# Direct atom-dispatch tests (cover the path without a full to_dict round trip)
# ============================================================

class TestDeserialiseAtomDirect:
    def test_list_of_serialisable_atom(self):
        v = [_Leaf(1).to_dict(), _Leaf(2).to_dict()]
        out = _deserialise_atom(v, list[_Leaf])
        assert out == [_Leaf(1), _Leaf(2)]

    def test_polymorphic_union_atom(self):
        v = _Leaf(7).to_dict()
        # The Union[_Leaf, _Branch, None] argument list has 2 non-None args,
        # so the previous behaviour returned v raw. New behaviour: dispatch.
        from typing import Union
        out = _deserialise_atom(v, Union[_Leaf, _Branch, None])
        assert isinstance(out, _Leaf)
        assert out == _Leaf(7)

    def test_union_without_type_tag_stays_raw(self):
        """If the value isn't a dict with a type tag, we still can't auto-resolve."""
        from typing import Union
        out = _deserialise_atom(42, Union[int, str, None])
        assert out == 42
