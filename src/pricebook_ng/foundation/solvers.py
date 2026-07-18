"""Root-finding and derivative-free optimisation — finance-free numerics (L0).

Provenance:
  quarry: python/pricebook/core/ (solvers)
  source: bisection; Nelder-Mead downhill simplex (Nelder & Mead 1965)
  oracle: bisect finds a bracketed root; Nelder-Mead minimises a quadratic
  slice:  numerics-config (Topic 0 S6)
"""

from __future__ import annotations

from collections.abc import Callable


def bisect_root(f: Callable[[float], float], lo: float, hi: float,
                tol: float = 1e-14, max_iter: int = 200) -> float:
    """`x` in ``[lo, hi]`` with ``f(x) ≈ 0``; requires `f(lo)`, `f(hi)` to differ in sign."""
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


def nelder_mead(f: Callable[[list[float]], float], x0: list[float],
                step: float = 0.05, tol: float = 1e-12, max_iter: int = 2000) -> list[float]:
    """Minimise `f` over R^n from `x0` by the Nelder-Mead simplex — derivative-free, which
    suits calibration objectives. Converges on both a flat objective and a small simplex."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step * abs(pt[i]) if pt[i] != 0.0 else step
        simplex.append(pt)
    fvals = [f(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda k: fvals[k])
        simplex = [simplex[k] for k in order]
        fvals = [fvals[k] for k in order]
        size = max((abs(simplex[k][j] - simplex[0][j]) for k in range(1, n + 1) for j in range(n)),
                   default=0.0)
        if abs(fvals[-1] - fvals[0]) <= tol and size <= tol:
            break

        centroid = [sum(simplex[k][j] for k in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        reflected = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        f_ref = f(reflected)

        if fvals[0] <= f_ref < fvals[-2]:
            simplex[-1], fvals[-1] = reflected, f_ref
        elif f_ref < fvals[0]:                                    # expand
            expanded = [centroid[j] + 2.0 * (reflected[j] - centroid[j]) for j in range(n)]
            f_exp = f(expanded)
            if f_exp < f_ref:
                simplex[-1], fvals[-1] = expanded, f_exp
            else:
                simplex[-1], fvals[-1] = reflected, f_ref
        else:                                                     # contract
            contracted = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            f_con = f(contracted)
            if f_con < fvals[-1]:
                simplex[-1], fvals[-1] = contracted, f_con
            else:                                                 # shrink toward best
                best = simplex[0]
                for k in range(1, n + 1):
                    simplex[k] = [best[j] + 0.5 * (simplex[k][j] - best[j]) for j in range(n)]
                    fvals[k] = f(simplex[k])
    return simplex[fvals.index(min(fvals))]
