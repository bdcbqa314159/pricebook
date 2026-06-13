"""Regression for L2 Wave-2 audit — `Dual` defined `__eq__` without
`__hash__`, breaking hashability and violating the Python hash/eq
contract.

Pre-fix: defining ``__eq__`` automatically set ``__hash__ = None``,
making ``Dual`` unhashable.  Any attempt to put a Dual in a dict or set
raised ``TypeError: unhashable type: 'Dual'`` — surprising the user since
Dual is conceptually a "number-like" object.

Additionally, even if hashability had been left intact via inheritance,
the standard Python hash/eq contract requires ``a == b ⇒ hash(a) ==
hash(b)``.  Dual's ``__eq__`` compares ONLY ``val`` (so that
``Dual(1, 2) == 1.0`` is True — a deliberate design choice for float
compatibility), so the matching ``__hash__`` must also depend only on
``val``.

Post-fix: ``__hash__`` is defined to return ``hash(self.val)``, making
Dual hashable AND consistent with the existing ``__eq__`` semantics.
"""

from __future__ import annotations

import pytest

from pricebook.numerical.auto_diff import Dual


class TestDualIsHashable:
    def test_hash_returns_int(self):
        h = hash(Dual(1.0, 2.0))
        assert isinstance(h, int)

    def test_can_be_dict_key(self):
        d = {Dual(1.0, 2.0): "x", Dual(3.0, 4.0): "y"}
        assert d[Dual(1.0, 2.0)] == "x"
        assert d[Dual(3.0, 4.0)] == "y"

    def test_can_be_in_set(self):
        s = {Dual(1.0, 0.0), Dual(2.0, 0.0)}
        assert Dual(1.0, 0.0) in s
        assert Dual(3.0, 0.0) not in s


class TestHashEqContract:
    """a == b ⇒ hash(a) == hash(b)."""

    def test_equal_duals_with_different_der_have_same_hash(self):
        """Dual.__eq__ ignores der; __hash__ must too."""
        a = Dual(1.0, 2.0)
        b = Dual(1.0, 99.0)
        assert a == b
        assert hash(a) == hash(b)

    def test_dual_and_equal_float_have_same_hash(self):
        """Dual(1.0, x) == 1.0 — must hash to the same as 1.0."""
        d = Dual(1.0, 2.0)
        assert d == 1.0
        assert hash(d) == hash(1.0)


class TestHashableButNotCollapsed:
    """Distinct vals → distinct hashes (with vanishingly small collision
    probability in any realistic case)."""

    def test_distinct_vals_distinct_hashes(self):
        h1 = hash(Dual(1.0, 0.0))
        h2 = hash(Dual(2.0, 0.0))
        # Could collide by coincidence but extremely unlikely.
        assert h1 != h2
