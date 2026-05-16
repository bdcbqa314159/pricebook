"""Commodity vol surface: SABR per tenor, vol cube, smile-consistent Kirk.

* :func:`calibrate_commodity_sabr` — SABR per commodity tenor.
* :class:`CommodityVolCube` — full vol cube for a commodity.
* :func:`kirk_spread_smile` — Kirk spread option with smile.

References:
    Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000.
    Kirk, *Correlation in the Energy Markets*, Risk Magazine, 1995.
    Hagan et al., *Managing Smile Risk*, Wilmott, 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from pricebook.sabr import sabr_implied_vol


# ---- SABR per tenor ----

@dataclass
class CommoditySmileNode:
    """SABR calibration at one commodity tenor."""
    expiry: float
    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    residual: float


def calibrate_commodity_sabr(
    forward: float,
    expiry: float,
    strikes: list[float],
    market_vols: list[float],
    beta: float = 1.0,
) -> CommoditySmileNode:
    """Calibrate SABR (α, ρ, ν) to market smile at one tenor.

    For commodities, β choice is asset-specific:
    - β = 1 for crude oil (lognormal)
    - β = 0.5 for metals (CIR-like)
    - β = 0 for power (Bachelier / can be negative prices)

    Args:
        forward: forward commodity price.
        expiry: time to expiry.
        strikes: market strikes.
        market_vols: corresponding market implied vols.
        beta: fixed β.
    """
    K = np.array(strikes)
    V = np.array(market_vols)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 0.999:
            return 1e10
        err = 0.0
        for k, v in zip(K, V):
            try:
                mdl = sabr_implied_vol(forward, k, expiry, alpha, beta, rho, nu)
                err += (mdl - v) ** 2
            except (ValueError, ZeroDivisionError):
                err += 1.0
        return err

    atm_vol = float(np.interp(forward, K, V)) if len(K) > 1 else V[0]
    alpha0 = atm_vol * forward ** (1 - beta) if forward > 0 else 0.3

    x0 = [alpha0, -0.2, 0.4]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8})

    alpha, rho, nu = result.x
    alpha = max(alpha, 1e-6)
    rho = max(-0.9999, min(0.9999, rho))
    nu = max(nu, 1e-6)

    atm = sabr_implied_vol(forward, forward, expiry, alpha, beta, rho, nu)
    residual = math.sqrt(result.fun / len(strikes))

    return CommoditySmileNode(
        expiry=expiry, forward=forward,
        alpha=alpha, beta=beta, rho=rho, nu=nu,
        atm_vol=atm, residual=residual,
    )


# ---- Commodity vol cube ----

@dataclass
class CommodityVolCube:
    """Commodity vol cube: SABR smile per tenor."""
    commodity: str
    nodes: list[CommoditySmileNode] = field(default_factory=list)

    def vol(self, T: float, strike: float) -> float:
        """Implied vol at (T, K) via parameter interpolation."""
        node = self._interp_node(T)
        return sabr_implied_vol(node.forward, strike, T,
                                 node.alpha, node.beta, node.rho, node.nu)

    def atm_vol(self, T: float) -> float:
        node = self._interp_node(T)
        return sabr_implied_vol(node.forward, node.forward, T,
                                 node.alpha, node.beta, node.rho, node.nu)

    def forward(self, T: float) -> float:
        """Forward price at tenor T (interpolated)."""
        return self._interp_node(T).forward

    def _interp_node(self, T: float) -> CommoditySmileNode:
        if not self.nodes:
            raise ValueError("Empty cube")
        exps = sorted(n.expiry for n in self.nodes)
        by_t = {n.expiry: n for n in self.nodes}

        if T <= exps[0]:
            return by_t[exps[0]]
        if T >= exps[-1]:
            return by_t[exps[-1]]

        idx = next(i for i, e in enumerate(exps) if e >= T)
        T0, T1 = exps[idx - 1], exps[idx]
        n0, n1 = by_t[T0], by_t[T1]
        f = (T - T0) / (T1 - T0)

        return CommoditySmileNode(
            expiry=T,
            forward=n0.forward + f * (n1.forward - n0.forward),
            alpha=n0.alpha + f * (n1.alpha - n0.alpha),
            beta=n0.beta,
            rho=n0.rho + f * (n1.rho - n0.rho),
            nu=n0.nu + f * (n1.nu - n0.nu),
            atm_vol=n0.atm_vol + f * (n1.atm_vol - n0.atm_vol),
            residual=0.0,
        )


def build_commodity_cube(
    commodity: str,
    tenors: list[float],
    forwards: dict[float, float],
    market_smiles: dict[float, tuple[list[float], list[float]]],
    beta: float = 1.0,
) -> CommodityVolCube:
    """Build cube from per-tenor smiles.

    market_smiles[T] = (strikes, vols).
    """
    nodes = []
    for T in tenors:
        if T not in market_smiles:
            continue
        F = forwards[T]
        strikes, vols = market_smiles[T]
        node = calibrate_commodity_sabr(F, T, strikes, vols, beta)
        nodes.append(node)

    return CommodityVolCube(commodity, nodes)


# ---- Kirk spread with smile ----

@dataclass
class KirkResult:
    """Kirk spread option result."""
    price: float
    implied_spread_vol: float
    flat_price: float
    smile_price: float
    smile_adjustment: float


def kirk_spread_smile(
    forward1: float,
    forward2: float,
    strike: float,
    vol1: float,
    vol2: float,
    correlation: float,
    rate: float,
    T: float,
    is_call: bool = True,
    smile_adjustment_factor: float = 0.0,
) -> KirkResult:
    """Kirk (1995) spread option with optional smile adjustment.

    Kirk approximation for option on (F₁ − F₂):
        σ² ≈ σ₁² + σ₂² × (F₂/(F₁−K))² − 2 ρ σ₁ σ₂ × (F₂/(F₁−K))

    Then Black-76 on F₁ − F₂ with effective σ.

    Smile adjustment: `smile_adjustment_factor` applied additively to σ
    (approximate — real smile-Kirk requires full vol surface knowledge).

    Args:
        forward1, forward2: forwards of the two commodities.
        strike: spread strike (F₁ − F₂ − K).
        vol1, vol2: ATM vols.
        correlation: between the two commodities.
        smile_adjustment_factor: additive adjustment to σ_spread.
    """
    df = math.exp(-rate * T)

    # Kirk (1995) effective vol: ratio = F2 / (F2 + K)
    denom = forward2 + strike
    if abs(denom) < 1e-6:
        denom = 1e-6
    ratio = forward2 / denom
    sigma_sq = vol1**2 + (vol2 * ratio)**2 - 2 * correlation * vol1 * vol2 * ratio
    sigma = math.sqrt(max(sigma_sq, 1e-10))
    sigma_smile = sigma + smile_adjustment_factor

    # Spread = F1 - F2 - K, priced as ATM option on forward1 − forward2 with K effective
    F_spread = forward1 - forward2
    K_eff = strike

    from pricebook.black76 import black76_price, OptionType
    opt = OptionType.CALL if is_call else OptionType.PUT
    flat_price = black76_price(F_spread, K_eff, sigma, T, df, opt)
    smile_price = black76_price(F_spread, K_eff, sigma_smile, T, df, opt)

    return KirkResult(
        price=float(max(smile_price, 0.0)),
        implied_spread_vol=float(sigma),
        flat_price=float(flat_price),
        smile_price=float(smile_price),
        smile_adjustment=float(smile_price - flat_price),
    )
