"""Root finding (L0 numerical toolkit — migrated on demand).

`bisect_root` only so far: bracketed bisection, which is bulletproof for a
continuous function with a sign change over [lo, hi] — the Jamshidian swaption's
coupon-bond equation is monotonic, so this is all it needs. Newton/Brent variants
migrate from the quarry when a consumer needs faster convergence (CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/core/solvers.py
  source: bisection method (standard)
  oracle: exact roots of known functions (S08 test); drives the swaption solve
  slice:  S08
"""

from __future__ import annotations

from collections.abc import Callable


def bisect_root(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    """Return x in [lo, hi] with f(x) ~ 0. Requires f(lo) and f(hi) to differ in
    sign. Converges when the bracket is narrower than `tol`."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo > 0.0) == (fhi > 0.0):
        raise ValueError(f"f does not change sign on [{lo}, {hi}]: f(lo)={flo}, f(hi)={fhi}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0 or (hi - lo) < tol:
            return mid
        if (fmid > 0.0) == (flo > 0.0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)
