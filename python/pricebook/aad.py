"""
Adjoint Algorithmic Differentiation (AAD).

Tape-based reverse-mode AD: records a computation graph during the
forward pass, then traverses it backward to compute all derivatives
in a single pass.

Translated from the CompFinance C++ engine (Savine, "Modern Computational
Finance: AAD and Parallel Simulations").

    from pricebook.aad import Number, Tape

    tape = Tape()
    Number.tape = tape

    x = Number(2.0)
    y = Number(3.0)
    z = x * y + x.exp()

    z.propagate_to_start()
    print(x.adjoint)  # dz/dx = y + exp(x) = 3 + e^2 ≈ 10.389
    print(y.adjoint)  # dz/dy = x = 2.0

Usage with pricing:
    tape = Tape()
    Number.tape = tape

    spot = Number(100.0)
    vol = Number(0.20)
    price = black_scholes(spot, vol, ...)

    price.propagate_to_start()
    delta = spot.adjoint  # dPrice/dSpot
    vega = vol.adjoint    # dPrice/dVol
"""

from __future__ import annotations

import math as _math


class Node:
    """A single operation recorded on the tape.

    Stores partial derivatives to each argument and pointers to
    argument nodes for adjoint propagation.
    """

    __slots__ = ("adjoint", "_derivatives", "_children")

    def __init__(self, derivatives: list[float], children: list[Node]):
        self.adjoint: float = 0.0
        self._derivatives = derivatives
        self._children = children

    def propagate(self) -> None:
        """Propagate this node's adjoint to its children."""
        if not self.adjoint:
            return
        for der, child in zip(self._derivatives, self._children):
            child.adjoint += der * self.adjoint


class Tape:
    """Records the computation graph as a list of nodes.

    Supports mark/rewind for efficient MC Greeks: mark after setup,
    rewind per path, propagate per path, then final propagation
    through setup operations.
    """

    def __init__(self):
        self._nodes: list[Node] = []
        self._mark: int = 0

    def record(self, derivatives: list[float], children: list[Node]) -> Node:
        """Record an operation and return the new node."""
        node = Node(derivatives, children)
        self._nodes.append(node)
        return node

    def record_leaf(self) -> Node:
        """Record a leaf (input variable) with no children."""
        node = Node([], [])
        self._nodes.append(node)
        return node

    def mark(self) -> None:
        """Mark the current tape position."""
        self._mark = len(self._nodes)

    def rewind_to_mark(self) -> None:
        """Remove all nodes after the mark, resetting their adjoints."""
        for node in self._nodes[self._mark:]:
            node.adjoint = 0.0
        del self._nodes[self._mark:]

    def clear(self) -> None:
        """Clear the entire tape."""
        self._nodes.clear()
        self._mark = 0

    def reset_adjoints(self) -> None:
        """Zero all adjoints without clearing the tape."""
        for node in self._nodes:
            node.adjoint = 0.0

    def propagate(self, from_idx: int, to_idx: int) -> None:
        """Propagate adjoints backward from from_idx to to_idx (inclusive)."""
        for i in range(from_idx, to_idx - 1, -1):
            self._nodes[i].propagate()

    def propagate_mark_to_start(self) -> None:
        """Propagate from mark position backward to start."""
        if self._mark > 0:
            self.propagate(self._mark - 1, 0)

    @property
    def size(self) -> int:
        return len(self._nodes)


# Global tape
_global_tape = Tape()


class Number:
    """Active floating-point number with automatic differentiation.

    All arithmetic is recorded on the tape. After computing a result,
    call result.propagate_to_start() to compute all gradients.
    """

    tape: Tape = _global_tape

    __slots__ = ("_value", "_node")

    def __init__(self, value: float):
        self._value = float(value)
        self._node = Number.tape.record_leaf()

    @classmethod
    def _from_op(cls, value: float, derivatives: list[float], children: list[Node]) -> Number:
        """Create a Number from an operation (internal)."""
        obj = object.__new__(cls)
        obj._value = value
        obj._node = Number.tape.record(derivatives, children)
        return obj

    @property
    def value(self) -> float:
        return self._value

    @property
    def adjoint(self) -> float:
        return self._node.adjoint

    @adjoint.setter
    def adjoint(self, val: float):
        self._node.adjoint = val

    @property
    def node(self) -> Node:
        return self._node

    def propagate_to_start(self) -> None:
        """Set this node's adjoint to 1 and propagate all the way back."""
        self._node.adjoint = 1.0
        idx = Number.tape._nodes.index(self._node)
        Number.tape.propagate(idx, 0)

    def propagate_to_mark(self) -> None:
        """Set adjoint to 1 and propagate back to the mark position."""
        self._node.adjoint = 1.0
        idx = Number.tape._nodes.index(self._node)
        Number.tape.propagate(idx, Number.tape._mark)

    # --- Arithmetic operators ---

    def __add__(self, other):
        if isinstance(other, Number):
            return Number._from_op(
                self._value + other._value,
                [1.0, 1.0],
                [self._node, other._node],
            )
        other = float(other)
        return Number._from_op(self._value + other, [1.0], [self._node])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            return Number._from_op(
                self._value - other._value,
                [1.0, -1.0],
                [self._node, other._node],
            )
        other = float(other)
        return Number._from_op(self._value - other, [1.0], [self._node])

    def __rsub__(self, other):
        other = float(other)
        return Number._from_op(other - self._value, [-1.0], [self._node])

    def __mul__(self, other):
        if isinstance(other, Number):
            return Number._from_op(
                self._value * other._value,
                [other._value, self._value],
                [self._node, other._node],
            )
        other = float(other)
        return Number._from_op(self._value * other, [other], [self._node])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            val = self._value / other._value
            return Number._from_op(
                val,
                [1.0 / other._value, -self._value / (other._value ** 2)],
                [self._node, other._node],
            )
        other = float(other)
        return Number._from_op(self._value / other, [1.0 / other], [self._node])

    def __rtruediv__(self, other):
        other = float(other)
        val = other / self._value
        return Number._from_op(val, [-other / (self._value ** 2)], [self._node])

    def __neg__(self):
        return Number._from_op(-self._value, [-1.0], [self._node])

    def __pow__(self, other):
        if isinstance(other, Number):
            val = self._value ** other._value
            return Number._from_op(
                val,
                [other._value * self._value ** (other._value - 1),
                 val * _math.log(max(self._value, 1e-300))],
                [self._node, other._node],
            )
        other = float(other)
        val = self._value ** other
        return Number._from_op(
            val, [other * self._value ** (other - 1)], [self._node],
        )

    # --- Comparison (value only, no tape) ---

    def __lt__(self, other):
        return self._value < (other._value if isinstance(other, Number) else other)

    def __le__(self, other):
        return self._value <= (other._value if isinstance(other, Number) else other)

    def __gt__(self, other):
        return self._value > (other._value if isinstance(other, Number) else other)

    def __ge__(self, other):
        return self._value >= (other._value if isinstance(other, Number) else other)

    def __eq__(self, other):
        return self._value == (other._value if isinstance(other, Number) else other)

    def __float__(self):
        return self._value

    def __repr__(self):
        return f"Number({self._value})"

    # --- Math functions ---

    def exp(self) -> Number:
        val = _math.exp(self._value)
        return Number._from_op(val, [val], [self._node])

    def log(self) -> Number:
        val = _math.log(self._value)
        return Number._from_op(val, [1.0 / self._value], [self._node])

    def sqrt(self) -> Number:
        val = _math.sqrt(self._value)
        return Number._from_op(val, [0.5 / val], [self._node])

    def abs(self) -> Number:
        val = _math.fabs(self._value)
        der = 1.0 if self._value >= 0 else -1.0
        return Number._from_op(val, [der], [self._node])


# --- Module-level math functions ---

def exp(x: Number) -> Number:
    return x.exp()

def log(x: Number) -> Number:
    return x.log()

def sqrt(x: Number) -> Number:
    return x.sqrt()

def maximum(x: Number, y) -> Number:
    """Differentiable max (piecewise linear)."""
    if isinstance(y, Number):
        if x._value >= y._value:
            return Number._from_op(x._value, [1.0, 0.0], [x._node, y._node])
        return Number._from_op(y._value, [0.0, 1.0], [x._node, y._node])
    y = float(y)
    if x._value >= y:
        return Number._from_op(x._value, [1.0], [x._node])
    return Number._from_op(y, [0.0], [x._node])

def norm_cdf(x: Number) -> Number:
    """Standard normal CDF (differentiable)."""
    val = 0.5 * (1.0 + _math.erf(x._value / _math.sqrt(2.0)))
    pdf = _math.exp(-0.5 * x._value**2) / _math.sqrt(2.0 * _math.pi)
    return Number._from_op(val, [pdf], [x._node])
