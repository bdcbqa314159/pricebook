"""Hagan–West monotone-convex reconstruction from interval averages (L0, finance-free).

Given knots ``x₀ < x₁ < … < xₙ`` and the average ``aᵢ`` of an unknown function over each
interval ``[xᵢ₋₁, xᵢ]``, reconstruct a continuous ``h`` that (1) reproduces every interval
average exactly — ``(1/Δᵢ)∫h = aᵢ`` — and (2) is locally monotone / convex where the averages
are. Pure interval-average interpolation: NO forwards, discount factors, rates or curves appear
here (that wiring is L1). Hand-rolled — absent from scipy (§7bb). It does NOT fit the point-based
`interpolate(xs, ys, x)` API (that takes point *values*; this takes interval *integrals*), so it
is a distinct primitive, not a mode of `interpolate`.

Scope: the four-region monotone-convex construction + optional positivity. The separate l=0.20
smoothness amelioration (AMF eq 63–80, the less-local variant) is deferred to its first
smoothness-preference consumer.

Provenance:
  quarry: none — new work, absent from both trees (redesign/18 §3)
  source: Hagan & West, "Interpolation Methods for Curve Construction" (AMF 2006): knot
          estimates eq 30–32, interval-average identity eq 33, basic quadratic eq 47,
          four-region construction eq 49–56, positivity eq 58–62.
  oracle: eq-33 interval-average reproduction to 1e-14; per-region pointwise value vs eq 47/49–56
  slice:  hagan-west (T1 slice 5)
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def _knot_estimates(
    knots: tuple[float, ...], averages: tuple[float, ...], positive: bool
) -> tuple[float, ...]:
    """The value at each knot, estimated from the adjacent interval averages (AMF eq 30–32);
    optionally clamped for positivity (eq 60–62)."""
    n = len(averages)
    f = [0.0] * (n + 1)
    for i in range(1, n):  # eq 30 — linear interpolation of the two adjacent averages
        span = knots[i + 1] - knots[i - 1]
        f[i] = (knots[i] - knots[i - 1]) / span * averages[i] + (
            knots[i + 1] - knots[i]
        ) / span * averages[i - 1]
    f[0] = averages[0] - 0.5 * (f[1] - averages[0])  # eq 31 (f'(x₀)=0 collar)
    f[n] = averages[n - 1] - 0.5 * (f[n - 1] - averages[n - 1])  # eq 32
    if positive:  # eq 60–62 — clamp knot values into [0, 2·min(adjacent averages)]
        f[0] = _clamp(f[0], 0.0, 2.0 * averages[0])
        for i in range(1, n):
            f[i] = _clamp(f[i], 0.0, 2.0 * min(averages[i - 1], averages[i]))
        f[n] = _clamp(f[n], 0.0, 2.0 * averages[n - 1])
    return tuple(f)


def _region_g(x: float, g0: float, g1: float) -> float:
    """The monotone-convex deviation ``g(x) = h − aᵢ`` on the unit interval, four regions
    (AMF eq 47, 49–56). ``g(0)=g0``, ``g(1)=g1``, ``∫₀¹g = 0``."""
    if g0 == 0.0 and g1 == 0.0:
        return 0.0
    if x <= 0.0:
        return g0
    if x >= 1.0:
        return g1
    # region (i): the basic quadratic is already monotone (eq 47)
    if (g0 > 0 and -0.5 * g0 >= g1 >= -2 * g0) or (g0 < 0 and -0.5 * g0 <= g1 <= -2 * g0):
        return g0 * (1 - 4 * x + 3 * x * x) + g1 * (-2 * x + 3 * x * x)
    # region (ii): flat then quadratic (eq 49–50)
    if (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
        eta = (g1 + 2 * g0) / (g1 - g0)
        return g0 if x <= eta else g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2
    # region (iii): quadratic then flat (eq 51–52)
    if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
        eta = 3 * g1 / (g1 - g0)
        return g1 + (g0 - g1) * ((eta - x) / eta) ** 2 if x < eta else g1
    # region (iv): same sign — two quadratics through a middle extremum A (eq 53–56)
    eta = g1 / (g1 + g0)
    a = -g0 * g1 / (g0 + g1)
    if x < eta:
        return a + (g0 - a) * ((eta - x) / eta) ** 2
    return a + (g1 - a) * ((x - eta) / (1 - eta)) ** 2


def _region_integral(x: float, g0: float, g1: float) -> float:
    """``∫₀ˣ g`` for the four-region ``g`` — the antiderivative used to integrate ``h``.
    ``_region_integral(1, ·) == 0`` for every region (the eq-33 constraint)."""
    if g0 == 0.0 and g1 == 0.0:
        return 0.0
    x = _clamp(x, 0.0, 1.0)
    if (g0 > 0 and -0.5 * g0 >= g1 >= -2 * g0) or (g0 < 0 and -0.5 * g0 <= g1 <= -2 * g0):
        return g0 * (x - 2 * x * x + x**3) + g1 * (-x * x + x**3)
    if (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
        eta = (g1 + 2 * g0) / (g1 - g0)
        if x <= eta:
            return g0 * x
        return g0 * x + (g1 - g0) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)
    if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
        eta = 3 * g1 / (g1 - g0)
        if x < eta:
            return g1 * x + (g0 - g1) * (eta**3 - (eta - x) ** 3) / (3 * eta * eta)
        at_eta = g1 * eta + (g0 - g1) * eta / 3
        return at_eta + g1 * (x - eta)
    eta = g1 / (g1 + g0)
    a = -g0 * g1 / (g0 + g1)
    if x < eta:
        return a * x + (g0 - a) * (eta**3 - (eta - x) ** 3) / (3 * eta * eta)
    at_eta = a * eta + (g0 - a) * eta / 3
    return at_eta + a * (x - eta) + (g1 - a) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)


@dataclass(frozen=True)
class MonotoneConvex:
    """A reconstructed function from interval averages. `knot_values` are the AMF eq 30–32
    estimates at each knot; `value(x)` evaluates the four-region interpolant; `integral(x)`
    is ``∫_{knots[0]}^x value``. Reproduces every interval average exactly (AMF eq 33)."""

    knots: tuple[float, ...]
    averages: tuple[float, ...]
    knot_values: tuple[float, ...]
    positive: bool

    def _interval(self, x: float) -> int:
        n = len(self.averages)
        for i in range(1, n + 1):
            if x <= self.knots[i] or i == n:
                return i
        return n

    def _check_domain(self, x: float) -> None:
        # the reconstruction is defined only on [knots[0], knots[-1]] — no silent extrapolation
        # (mirrors the interpolation RAISE policy; consistent with the log-linear curve branch)
        if x < self.knots[0] or x > self.knots[-1]:
            raise ValueError(f"{x} outside [{self.knots[0]}, {self.knots[-1]}] — no extrapolation")

    def value(self, x: float) -> float:
        self._check_domain(x)
        i = self._interval(x)
        span = self.knots[i] - self.knots[i - 1]
        xloc = (x - self.knots[i - 1]) / span
        a = self.averages[i - 1]
        return a + _region_g(xloc, self.knot_values[i - 1] - a, self.knot_values[i] - a)

    def integral(self, x: float) -> float:
        self._check_domain(x)
        total = 0.0
        n = len(self.averages)
        for i in range(1, n + 1):
            x_lo, x_hi = self.knots[i - 1], self.knots[i]
            a = self.averages[i - 1]
            if x >= x_hi:
                total += a * (x_hi - x_lo)
            elif x > x_lo:
                span = x_hi - x_lo
                xloc = (x - x_lo) / span
                g0, g1 = self.knot_values[i - 1] - a, self.knot_values[i] - a
                total += a * (x - x_lo) + span * _region_integral(xloc, g0, g1)
                break
            else:
                break
        return total


def monotone_convex(
    knots: tuple[float, ...], averages: tuple[float, ...], positive: bool = False
) -> MonotoneConvex:
    """Build the monotone-convex reconstruction: `knots` (ascending, length n+1), `averages`
    (length n, the average over each interval), `positive` clamps knot estimates ≥ 0 (eq
    60–62). The primitive is finance-free — `knots`/`averages` are abstract."""
    if len(knots) != len(averages) + 1 or len(averages) < 1:
        raise ValueError("need len(knots) == len(averages) + 1, and at least one interval")
    return MonotoneConvex(knots, averages, _knot_estimates(knots, averages, positive), positive)
