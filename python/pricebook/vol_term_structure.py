"""Vol term structure: forward vol dynamics, calendar spreads, Bergomi 2-factor.

* :func:`forward_vol_from_term` — extract forward vol from ATM term structure.
* :func:`calendar_spread_strategy` — long one tenor, short another.
* :func:`vol_curve_shape` — classify contango/flat/backwardation.
* :class:`Bergomi2Factor` — Bergomi two-factor vol model.

References:
    Bergomi, *Stochastic Volatility Modeling*, CRC, 2016.
    Gatheral, *The Volatility Surface*, Wiley, 2006.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class ForwardVolResult:
    tenor1: float
    tenor2: float
    spot_vol1: float
    spot_vol2: float
    forward_vol: float

def forward_vol_from_term(
    tenors: list[float], atm_vols: list[float],
    T1: float, T2: float,
) -> ForwardVolResult:
    v1 = float(np.interp(T1, tenors, atm_vols))
    v2 = float(np.interp(T2, tenors, atm_vols))
    fwd_var = (v2**2 * T2 - v1**2 * T1) / max(T2 - T1, 1e-10)
    fwd_vol = math.sqrt(max(fwd_var, 0))
    return ForwardVolResult(T1, T2, v1, v2, float(fwd_vol))


@dataclass
class CalendarSpreadResult:
    long_tenor: float
    short_tenor: float
    long_vol: float
    short_vol: float
    spread: float               # long − short vol
    carry_per_day: float

def calendar_spread_strategy(
    tenors: list[float], vols: list[float],
    long_tenor: float, short_tenor: float,
    vega_per_vol_point: float = 1.0,
) -> CalendarSpreadResult:
    v_long = float(np.interp(long_tenor, tenors, vols))
    v_short = float(np.interp(short_tenor, tenors, vols))
    spread = v_long - v_short
    # Calendar carry: long vega decays slower than short
    carry = -spread * vega_per_vol_point / 252
    return CalendarSpreadResult(long_tenor, short_tenor, v_long, v_short,
                                  float(spread), float(carry))


@dataclass
class VolCurveShapeResult:
    shape: str                  # "contango", "flat", "backwardation"
    slope: float
    curvature: float

def vol_curve_shape(tenors: list[float], vols: list[float]) -> VolCurveShapeResult:
    if len(tenors) < 2:
        return VolCurveShapeResult("flat", 0.0, 0.0)
    slope = (vols[-1] - vols[0]) / max(tenors[-1] - tenors[0], 1e-10)
    if len(tenors) >= 3:
        mid = len(tenors) // 2
        curv = vols[mid] - 0.5 * (vols[0] + vols[-1])
    else:
        curv = 0.0
    if slope > 0.005: shape = "contango"
    elif slope < -0.005: shape = "backwardation"
    else: shape = "flat"
    return VolCurveShapeResult(shape, float(slope), float(curv))


@dataclass
class Bergomi2FactorResult:
    vol_paths: np.ndarray       # (n_paths, n_steps+1)
    mean_terminal_vol: float
    vol_of_vol: float

class Bergomi2Factor:
    """Bergomi two-factor forward variance model.
    ξ(t, T) = ξ₀(T) × exp(η₁ W₁(t) + η₂ W₂(t) − ½(η₁²+η₂²)t)
    Two factors capture level and slope of forward variance.
    """
    def __init__(self, xi0: float, eta1: float, eta2: float, rho12: float = 0.0):
        self.xi0 = xi0; self.eta1 = eta1; self.eta2 = eta2; self.rho12 = rho12

    def simulate(self, T: float, n_paths: int = 2000, n_steps: int = 50,
                  seed: int | None = 42) -> Bergomi2FactorResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps; sqrt_dt = math.sqrt(dt)
        z1 = rng.standard_normal((n_paths, n_steps))
        z2 = self.rho12 * z1 + math.sqrt(1-self.rho12**2) * rng.standard_normal((n_paths, n_steps))

        W1 = np.zeros((n_paths, n_steps+1)); W2 = np.zeros((n_paths, n_steps+1))
        for s in range(n_steps):
            W1[:, s+1] = W1[:, s] + z1[:, s] * sqrt_dt
            W2[:, s+1] = W2[:, s] + z2[:, s] * sqrt_dt

        times = np.linspace(0, T, n_steps+1)
        log_xi = (self.eta1 * W1 + self.eta2 * W2
                   - 0.5 * (self.eta1**2 + self.eta2**2) * times)
        vol_paths = math.sqrt(self.xi0) * np.exp(0.5 * log_xi)

        return Bergomi2FactorResult(vol_paths, float(vol_paths[:,-1].mean()),
                                      float(vol_paths[:,-1].std()))
