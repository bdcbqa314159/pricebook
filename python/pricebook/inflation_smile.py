"""Inflation smile: SABR on YoY caplets, ZC cap smile, vol cube.

* :func:`calibrate_inflation_sabr` — SABR per YoY tenor.
* :class:`InflationVolCube` — inflation vol cube.
* :func:`zc_inflation_cap_smile` — ZC cap at multiple strikes.

References:
    Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.
    Kenyon, *Inflation Is Normal*, Risk, 2008.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from pricebook.sabr import sabr_implied_vol
from pricebook.black76 import black76_price, OptionType


@dataclass
class InflationSmileNode:
    expiry: float
    forward_yoy: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    residual: float

def calibrate_inflation_sabr(
    forward_yoy: float, expiry: float,
    strikes: list[float], market_vols: list[float],
    beta: float = 0.5,
) -> InflationSmileNode:
    """Calibrate SABR to inflation caplet smile at one tenor."""
    K = np.array(strikes); V = np.array(market_vols)
    def obj(params):
        a, rho, nu = params
        if a <= 0 or nu <= 0 or abs(rho) >= 0.999: return 1e10
        return sum((sabr_implied_vol(forward_yoy, k, expiry, a, beta, rho, nu) - v)**2
                    for k, v in zip(K, V))
    atm = float(np.interp(forward_yoy, K, V)) if len(K) > 1 else V[0]
    a0 = atm * forward_yoy**(1-beta) if forward_yoy > 0 else 0.01
    res = minimize(obj, [a0, -0.2, 0.3], method='Nelder-Mead', options={'maxiter': 3000})
    a, rho, nu = res.x
    a = max(a, 1e-6); rho = max(-0.999, min(0.999, rho)); nu = max(nu, 1e-6)
    atm_fit = sabr_implied_vol(forward_yoy, forward_yoy, expiry, a, beta, rho, nu)
    return InflationSmileNode(expiry, forward_yoy, a, beta, rho, nu, atm_fit,
                                math.sqrt(res.fun / len(K)))


@dataclass
class InflationVolCube:
    nodes: list[InflationSmileNode] = field(default_factory=list)

    def vol(self, T: float, strike: float) -> float:
        if not self.nodes: raise ValueError("Empty cube")
        exps = sorted(n.expiry for n in self.nodes)
        by_t = {n.expiry: n for n in self.nodes}
        if T <= exps[0]: n = by_t[exps[0]]
        elif T >= exps[-1]: n = by_t[exps[-1]]
        else:
            idx = next(i for i, e in enumerate(exps) if e >= T)
            t0, t1 = exps[idx-1], exps[idx]
            f = (T-t0)/(t1-t0)
            n0, n1 = by_t[t0], by_t[t1]
            n = InflationSmileNode(T, n0.forward_yoy+f*(n1.forward_yoy-n0.forward_yoy),
                n0.alpha+f*(n1.alpha-n0.alpha), n0.beta, n0.rho+f*(n1.rho-n0.rho),
                n0.nu+f*(n1.nu-n0.nu), n0.atm_vol+f*(n1.atm_vol-n0.atm_vol), 0)
        return sabr_implied_vol(n.forward_yoy, strike, T, n.alpha, n.beta, n.rho, n.nu)


@dataclass
class ZCCapSmileResult:
    strikes: np.ndarray
    prices: np.ndarray
    implied_vols: np.ndarray
    forward_zc_rate: float

def zc_inflation_cap_smile(
    forward_zc_rate: float, rate: float, T: float, vol: float,
    strike_offsets: list[float] | None = None,
) -> ZCCapSmileResult:
    """ZC inflation cap prices at multiple strikes."""
    if strike_offsets is None:
        strike_offsets = [-0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02]
    strikes = np.array([forward_zc_rate + off for off in strike_offsets])
    strikes = np.maximum(strikes, 1e-6)
    df = math.exp(-rate * T)
    prices = np.array([black76_price(forward_zc_rate, k, vol, T, df, OptionType.CALL)
                        for k in strikes])
    return ZCCapSmileResult(strikes, prices, np.full_like(strikes, vol), forward_zc_rate)
