"""
Adjoint Algorithmic Differentiation (AAD).

Tape-based reverse-mode AD: records a computation graph during the
forward pass, then traverses it backward to compute all derivatives
in a single pass.

Translated from the CompFinance C++ engine. See REFERENCES.md (Savine).

    from pricebook.aad import Number, Tape

    tape = Tape()
    Number.tape = tape

    x = Number(2.0)
    y = Number(3.0)
    z = x * y + x.exp()

    z.propagate_to_start()
    print(x.adjoint)  # dz/dx = y + exp(x) = 3 + e^2 ≈ 10.389
    print(y.adjoint)  # dz/dy = x = 2.0
"""

from __future__ import annotations

import math as _math
import threading


class Node:
    """A single operation recorded on the tape."""

    __slots__ = ("adjoint", "_derivatives", "_children", "_idx")

    def __init__(self, derivatives: list[float], children: list[Node], idx: int):
        self.adjoint: float = 0.0
        self._derivatives = derivatives
        self._children = children
        self._idx = idx

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

    Use as context manager for clean lifecycle:
        with Tape() as tape:
            x = Number(2.0)
            ...
    """

    def __init__(self):
        self._nodes: list[Node] = []
        self._mark: int = 0

    def __enter__(self):
        Number.tape = self
        return self

    def __exit__(self, *args):
        self.clear()

    def record(self, derivatives: list[float], children: list[Node]) -> Node:
        """Record an operation and return the new node."""
        idx = len(self._nodes)
        node = Node(derivatives, children, idx)
        self._nodes.append(node)
        return node

    def record_leaf(self) -> Node:
        """Record a leaf (input variable) with no children."""
        idx = len(self._nodes)
        node = Node([], [], idx)
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


# Thread-local tape storage
_tape_local = threading.local()


def _get_default_tape() -> Tape:
    if not hasattr(_tape_local, "tape"):
        _tape_local.tape = Tape()
    return _tape_local.tape


class Number:
    """Active floating-point number with automatic differentiation.

    All arithmetic is recorded on the tape. After computing a result,
    call result.propagate_to_start() to compute all gradients.
    """

    tape: Tape = _get_default_tape()

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

    def put_on_tape(self) -> None:
        """Re-register this Number on the tape (new leaf node)."""
        self._node = Number.tape.record_leaf()

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
        Number.tape.propagate(self._node._idx, 0)

    def propagate_to_mark(self) -> None:
        """Set adjoint to 1 and propagate back to the mark position."""
        self._node.adjoint = 1.0
        Number.tape.propagate(self._node._idx, Number.tape._mark)

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

    def __iadd__(self, other):
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

    def __isub__(self, other):
        return self.__sub__(other)

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

    def __imul__(self, other):
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

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __neg__(self):
        return Number._from_op(-self._value, [-1.0], [self._node])

    def __pos__(self):
        return self

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

    def __hash__(self):
        return id(self)

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
        der = 1.0 if self._value > 0 else (-1.0 if self._value < 0 else 0.0)
        return Number._from_op(val, [der], [self._node])


# --- Module-level math functions ---

def exp(x: Number) -> Number:
    return x.exp()

def log(x: Number) -> Number:
    return x.log()

def sqrt(x: Number) -> Number:
    return x.sqrt()

def maximum(x, y) -> Number:
    """Differentiable max (piecewise linear)."""
    if not isinstance(x, Number):
        x_val, x_node = float(x), None
    else:
        x_val, x_node = x._value, x._node

    if not isinstance(y, Number):
        y_val, y_node = float(y), None
    else:
        y_val, y_node = y._value, y._node

    if x_val >= y_val:
        val = x_val
        ders = []
        children = []
        if x_node is not None:
            ders.append(1.0)
            children.append(x_node)
        if y_node is not None:
            ders.append(0.0)
            children.append(y_node)
    else:
        val = y_val
        ders = []
        children = []
        if x_node is not None:
            ders.append(0.0)
            children.append(x_node)
        if y_node is not None:
            ders.append(1.0)
            children.append(y_node)

    return Number._from_op(val, ders, children)


def minimum(x, y) -> Number:
    """Differentiable min (piecewise linear)."""
    return -(maximum(-x if isinstance(x, Number) else Number(-float(x)),
                     -y if isinstance(y, Number) else Number(-float(y))))


def norm_cdf(x: Number) -> Number:
    """Standard normal CDF (differentiable)."""
    val = 0.5 * (1.0 + _math.erf(x._value / _math.sqrt(2.0)))
    pdf = _math.exp(-0.5 * x._value**2) / _math.sqrt(2.0 * _math.pi)
    return Number._from_op(val, [pdf], [x._node])


def norm_pdf(x: Number) -> Number:
    """Standard normal PDF (differentiable). d/dx N'(x) = -x * N'(x)."""
    val = _math.exp(-0.5 * x._value**2) / _math.sqrt(2.0 * _math.pi)
    der = -x._value * val
    return Number._from_op(val, [der], [x._node])
