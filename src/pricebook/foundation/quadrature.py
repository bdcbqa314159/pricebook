"""Expectation under a standard normal, by Gauss-Hermite quadrature (L0).

`standard_normal_expectation(f)` returns `E[f(Z)]` for `Z ~ N(0,1)`. Market Model I's year-on-year
value (BERHAN eq 3.1.11) is exactly such an expectation and has no closed form.

**A fixed Gauss-Hermite rule rather than an adaptive integrator, deliberately.**
`scipy.integrate.quad` is adaptive: it chooses its own node set per integrand, carries an internal
error estimate, and can shift between scipy versions — none of which belongs under a pinned
reproducibility contract, and the last of those would make a result depend on a dependency's minor
release. A fixed rule is deterministic, identical on both CI operating systems, and has **no
tolerance knob to turn until a test passes**. Its accuracy is instead a property the oracles
measure, against moments known in closed form.

**The transform, written out because it is where this goes wrong.** `numpy.polynomial.hermgauss`
returns nodes and weights for `∫ e^{−x²} g(x) dx`, not for a standard normal. Substituting
`x = z/√2` gives

    E[f(Z)] = (1/√π) · Σ wᵢ · f(xᵢ·√2)

so the `√2` scaling and the `1/√π` normalisation are both required. Those two constants and the
weights are exactly what a Gauss-Hermite transcription gets wrong, which is why the oracles check
`E[1] = 1` (the normalisation), the even moments `1, 3, 15` (the scaling — a wrong `√2` rescales
every moment by a power of two while leaving `E[1]` untouched), and `E[e^Z] = e^{1/2}` (a
transcendental target the rule does not integrate exactly).

Provenance:
  stock:  — the quarry integrated adaptively and per-site; built fresh as one adapter
  source: Gauss-Hermite quadrature, standard; `numpy.polynomial.legendre`-family API via
          `numpy.polynomial.hermite.hermgauss` (numpy pinned in pyproject, as the rule's nodes are
          part of the reproducibility contract)
  oracle: tests/L0/test_quadrature.py — `E[1]=1`, the odd moments vanish, the even moments are the
          double factorials `1, 3, 15`, `E[e^{tZ}] = e^{t²/2}` at four arguments; convergence in the
          node count, with a companion proving the count is not inert
  crosscheck: none — a numerical quadrature adapter, no finance convention
  slice:  tranche 12(b) — Market Model I (its first consumer)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np

#: Enough for a smooth integrand against a Gaussian weight; the oracles show `E[e^Z]` converged to
#: 1e-13 by 24 nodes, so 64 is comfortable margin rather than a tuned value.
_DEFAULT_NODES = 64


#: The rule's own normalisation, `Σwᵢ = E[1] = 1`, verified at construction. Not decoration: above
#: ~370 nodes `numpy.polynomial.hermite.hermgauss` loses the weights to underflow and returns a
#: rule that **silently integrates to the wrong answer** rather than failing. Measured on numpy
#: 2.4.3: correct to 370; at **371 every weight is 0.0, so `E[1]` is `0.0`** — a finite,
#: plausible-looking, 100%-wrong number, which is the worst of the three outcomes; from ~380 it is
#: `NaN`. A caller asking for a finer rule than the library can build must be refused, never handed
#: a zero (charter §3.4 — failure is a value, never a silent NaN or a quiet zero).
#:
#: The check is **derived, not a hard-coded 370**: it asserts the invariant this module's docstring
#: already names as its oracle, so a future numpy change moves the usable range without moving the
#: guard. Found while building tranche 41, whose market-factor integral wanted 256 nodes at high
#: factor loadings — within range, but one refinement away from the cliff.
_NORMALISATION_TOLERANCE = 1e-12


@lru_cache(maxsize=16)
def _rule(nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The scaled nodes and normalised weights, cached — they depend only on the count."""
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(nodes)
    scaled = tuple(float(x) * math.sqrt(2.0) for x in raw_nodes)
    normalised = tuple(float(w) / math.sqrt(math.pi) for w in raw_weights)
    total = math.fsum(normalised)
    if not abs(total - 1.0) < _NORMALISATION_TOLERANCE:
        raise ValueError(
            f"a {nodes}-point Gauss-Hermite rule does not normalise: its weights sum to {total}, "
            f"not 1. The underlying `hermgauss` loses the weights to underflow above roughly 370 "
            f"nodes, so it cannot build a rule this fine; use fewer nodes. Returning the rule "
            f"anyway would integrate everything to zero or NaN without saying so.")
    return scaled, normalised


def standard_normal_expectation(
    f: Callable[[float], float], nodes: int = _DEFAULT_NODES
) -> float:
    """`E[f(Z)]` for `Z ~ N(0,1)`, by an `nodes`-point Gauss-Hermite rule."""
    if nodes < 1:
        raise ValueError(f"a quadrature rule needs at least one node, got {nodes}")
    points, weights = _rule(nodes)
    return math.fsum(w * f(z) for z, w in zip(points, weights))
