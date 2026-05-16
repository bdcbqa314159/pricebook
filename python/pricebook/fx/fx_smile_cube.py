"""FX smile calibration and arbitrage-free vol cube.

Extends :mod:`pricebook.fx_vol_surface` with:

* :func:`calibrate_sabr_fx_tenor` — SABR per expiry tenor with fixed β.
* :class:`FXVolCube` — 3D vol(T, δ, K) with SABR smile at each tenor.
* :func:`svi_fit` — Stochastic Vol Inspired parametrisation.
* :func:`check_butterfly_arbitrage` — second-derivative positivity test.
* :func:`check_calendar_arbitrage` — total variance monotonicity.

References:
    Clark, *FX Option Pricing*, Wiley, 2011, Ch. 3-4.
    Gatheral, *The Volatility Surface*, Wiley, 2006.
    Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.
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
class FXSmileNode:
    """Calibrated SABR smile at one FX tenor."""
    expiry: float
    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    rr_25d: float          # 25D risk reversal (call - put)
    bf_25d: float          # 25D butterfly
    residual: float


def _strike_from_delta_fx(spot, rd, rf, vol, T, delta, is_call):
    """Spot delta to strike conversion."""
    sign = 1 if is_call else -1
    q = rf
    a = abs(delta) * math.exp(q * T)
    a = min(a, 0.9999)
    d1_target = sign * norm.ppf(a)
    return spot * math.exp(-d1_target * vol * math.sqrt(T)
                           + (rd - rf + 0.5 * vol**2) * T)


def calibrate_sabr_fx_tenor(
    spot: float,
    rate_dom: float,
    rate_for: float,
    T: float,
    vol_atm: float,
    vol_25d_call: float,
    vol_25d_put: float,
    vol_10d_call: float | None = None,
    vol_10d_put: float | None = None,
    beta: float = 0.5,
) -> FXSmileNode:
    """Calibrate SABR (α, ρ, ν) at one FX tenor to market quotes.

    Uses ATM + 25D call/put (+ optional 10D) as calibration strikes.
    β is fixed (0.5 common for FX).

    Args:
        spot: FX spot.
        rate_dom, rate_for: rates.
        T: tenor.
        vol_atm: ATM vol.
        vol_25d_call, vol_25d_put: 25D market vols.
        vol_10d_call, vol_10d_put: optional wings.
        beta: fixed CEV exponent.
    """
    F = spot * math.exp((rate_dom - rate_for) * T)

    # Convert deltas to strikes using the quote vol
    K_atm = F * math.exp(0.5 * vol_atm**2 * T)
    K_25c = _strike_from_delta_fx(spot, rate_dom, rate_for, vol_25d_call, T, 0.25, True)
    K_25p = _strike_from_delta_fx(spot, rate_dom, rate_for, vol_25d_put, T, 0.25, False)

    strikes = [K_25p, K_atm, K_25c]
    vols = [vol_25d_put, vol_atm, vol_25d_call]

    if vol_10d_call is not None and vol_10d_put is not None:
        K_10c = _strike_from_delta_fx(spot, rate_dom, rate_for, vol_10d_call, T, 0.10, True)
        K_10p = _strike_from_delta_fx(spot, rate_dom, rate_for, vol_10d_put, T, 0.10, False)
        strikes = [K_10p, K_25p, K_atm, K_25c, K_10c]
        vols = [vol_10d_put, vol_25d_put, vol_atm, vol_25d_call, vol_10d_call]

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 0.999:
            return 1e10
        total = 0.0
        for k, v_mkt in zip(strikes, vols):
            try:
                v_mdl = sabr_implied_vol(F, k, T, alpha, beta, rho, nu)
                total += (v_mdl - v_mkt) ** 2
            except (ValueError, ZeroDivisionError):
                total += 1.0
        return total

    alpha0 = vol_atm * F ** (1 - beta)
    x0 = [alpha0, -0.2, 0.3]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8})

    alpha, rho, nu = result.x
    alpha = max(alpha, 1e-6)
    rho = max(-0.9999, min(0.9999, rho))
    nu = max(nu, 1e-6)

    atm = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
    residual = math.sqrt(result.fun / len(strikes))

    return FXSmileNode(
        expiry=T, forward=F,
        alpha=alpha, beta=beta, rho=rho, nu=nu,
        atm_vol=atm,
        rr_25d=vol_25d_call - vol_25d_put,
        bf_25d=0.5 * (vol_25d_call + vol_25d_put) - vol_atm,
        residual=residual,
    )


# ---- FX Vol Cube ----

@dataclass
class FXVolCube:
    """FX vol cube with SABR smile at each tenor."""
    spot: float
    rate_dom: float
    rate_for: float
    nodes: list[FXSmileNode] = field(default_factory=list)

    def vol(self, T: float, strike: float) -> float:
        """Implied vol at (T, K) via time-interpolated SABR parameters."""
        node = self._interpolate_node(T)
        F = self.spot * math.exp((self.rate_dom - self.rate_for) * T)
        return sabr_implied_vol(F, strike, T, node.alpha, node.beta, node.rho, node.nu)

    def vol_at_delta(self, T: float, delta: float, is_call: bool = True) -> float:
        """Implied vol at (T, δ)."""
        # Iterative: use ATM vol first, compute strike, refine
        node = self._interpolate_node(T)
        K = _strike_from_delta_fx(self.spot, self.rate_dom, self.rate_for,
                                   node.atm_vol, T, delta, is_call)
        # Refine once
        vol = self.vol(T, K)
        K = _strike_from_delta_fx(self.spot, self.rate_dom, self.rate_for,
                                   vol, T, delta, is_call)
        return self.vol(T, K)

    def _interpolate_node(self, T: float) -> FXSmileNode:
        """Linearly interpolate SABR parameters across tenors."""
        if not self.nodes:
            raise ValueError("Cube has no nodes")

        expiries = sorted(n.expiry for n in self.nodes)
        nodes_by_t = {n.expiry: n for n in self.nodes}

        if T <= expiries[0]:
            return nodes_by_t[expiries[0]]
        if T >= expiries[-1]:
            return nodes_by_t[expiries[-1]]

        # Find bracketing tenors
        idx = next(i for i, e in enumerate(expiries) if e >= T)
        T0, T1 = expiries[idx - 1], expiries[idx]
        n0, n1 = nodes_by_t[T0], nodes_by_t[T1]

        f = (T - T0) / (T1 - T0)

        F = self.spot * math.exp((self.rate_dom - self.rate_for) * T)
        return FXSmileNode(
            expiry=T, forward=F,
            alpha=n0.alpha + f * (n1.alpha - n0.alpha),
            beta=n0.beta,
            rho=n0.rho + f * (n1.rho - n0.rho),
            nu=n0.nu + f * (n1.nu - n0.nu),
            atm_vol=n0.atm_vol + f * (n1.atm_vol - n0.atm_vol),
            rr_25d=n0.rr_25d + f * (n1.rr_25d - n0.rr_25d),
            bf_25d=n0.bf_25d + f * (n1.bf_25d - n0.bf_25d),
            residual=0.0,
        )


def build_fx_vol_cube(
    spot: float,
    rate_dom: float,
    rate_for: float,
    tenors: list[float],
    market_quotes: dict[float, dict[str, float]],
    beta: float = 0.5,
) -> FXVolCube:
    """Build FX vol cube from market quotes.

    Args:
        tenors: list of expiry times (years).
        market_quotes: {T → {"atm": vol, "rr_25d": rr, "bf_25d": bf, ...}}
            or {T → {"atm": vol, "25c": vol, "25p": vol}}.
    """
    nodes = []
    for T in tenors:
        q = market_quotes.get(T)
        if q is None:
            continue

        atm = q["atm"]
        if "25c" in q and "25p" in q:
            c, p = q["25c"], q["25p"]
        else:
            rr = q.get("rr_25d", 0.0)
            bf = q.get("bf_25d", 0.0)
            c = atm + bf + rr / 2
            p = atm + bf - rr / 2

        node = calibrate_sabr_fx_tenor(spot, rate_dom, rate_for, T,
                                        atm, c, p, beta=beta)
        nodes.append(node)

    return FXVolCube(spot, rate_dom, rate_for, nodes)


# ---- SVI ----

@dataclass
class SVIParams:
    """SVI (Stochastic Vol Inspired) raw parameters.

    Total variance w(k) = a + b × {ρ(k − m) + sqrt((k − m)² + σ²)}
    where k = log(K/F) is log-moneyness.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry: float


def _svi_variance(k: float, p: SVIParams) -> float:
    """SVI total variance at log-moneyness k."""
    return p.a + p.b * (p.rho * (k - p.m) + math.sqrt((k - p.m)**2 + p.sigma**2))


def svi_fit(
    log_moneyness: list[float],
    implied_vols: list[float],
    expiry: float,
) -> SVIParams:
    """Fit SVI raw parameters to observed smile.

    Minimises Σ (w_model(k) − T × σ²_mkt)².

    Args:
        log_moneyness: k = log(K/F) for each observed point.
        implied_vols: market implied vols (same length).
        expiry: tenor T.
    """
    k = np.array(log_moneyness)
    w_target = np.array(implied_vols)**2 * expiry

    def objective(params):
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 0.999:
            return 1e10
        w_model = np.array([_svi_variance(ki, SVIParams(a, b, rho, m, sigma, expiry))
                            for ki in k])
        return float(np.sum((w_model - w_target)**2))

    # Initial guess
    atm_var = (np.mean(implied_vols)**2) * expiry
    x0 = [atm_var * 0.5, 0.1, 0.0, 0.0, 0.3]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10})

    a, b, rho, m, sigma = result.x
    return SVIParams(a, max(b, 1e-6), max(-0.999, min(0.999, rho)),
                     m, max(sigma, 1e-6), expiry)


def svi_vol(k: float, params: SVIParams) -> float:
    """Implied vol from SVI at log-moneyness k."""
    w = _svi_variance(k, params)
    return math.sqrt(max(w / params.expiry, 1e-10))


# ---- Arbitrage checks ----

@dataclass
class ArbitrageCheckResult:
    """Arbitrage check result."""
    is_arbitrage_free: bool
    violations: list[str]
    n_strikes_checked: int
    n_tenors_checked: int


def check_butterfly_arbitrage(
    cube: FXVolCube,
    T: float,
    strike_range: tuple[float, float] | None = None,
    n_strikes: int = 30,
) -> ArbitrageCheckResult:
    """Check butterfly (no-arb) condition: ∂²C/∂K² ≥ 0 for all K.

    At a fixed tenor T, the call price must be convex in strike.
    Violations indicate a butterfly arbitrage opportunity.
    """
    from pricebook.black76 import black76_price, OptionType

    F = cube.spot * math.exp((cube.rate_dom - cube.rate_for) * T)
    df = math.exp(-cube.rate_dom * T)

    if strike_range is None:
        strike_range = (F * 0.8, F * 1.2)

    strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)
    prices = np.zeros(n_strikes)

    for i, K in enumerate(strikes):
        vol = cube.vol(T, K)
        prices[i] = black76_price(F, K, vol, T, df, OptionType.CALL)

    # Second difference: C(K+h) - 2C(K) + C(K-h)
    violations = []
    for i in range(1, n_strikes - 1):
        d2 = prices[i + 1] - 2 * prices[i] + prices[i - 1]
        if d2 < -1e-8:
            violations.append(f"K={strikes[i]:.4f}: d²C/dK²={d2:.6f} < 0")

    return ArbitrageCheckResult(
        is_arbitrage_free=(len(violations) == 0),
        violations=violations,
        n_strikes_checked=n_strikes - 2,
        n_tenors_checked=1,
    )


def check_calendar_arbitrage(
    cube: FXVolCube,
    tenors: list[float],
    strike_range: tuple[float, float] | None = None,
    n_strikes: int = 20,
) -> ArbitrageCheckResult:
    """Check calendar arbitrage: total variance T × σ²(T, K) is non-decreasing in T.

    For each strike K, w(T₁, K) ≤ w(T₂, K) for T₁ < T₂.
    """
    sorted_tenors = sorted(tenors)
    F0 = cube.spot

    if strike_range is None:
        strike_range = (F0 * 0.8, F0 * 1.2)

    strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)

    violations = []
    for K in strikes:
        prev_w = -math.inf
        for T in sorted_tenors:
            vol = cube.vol(T, K)
            w = vol**2 * T
            if w < prev_w - 1e-8:
                violations.append(
                    f"K={K:.4f}: w(T={T:.2f})={w:.6f} < w(prev)={prev_w:.6f}"
                )
            prev_w = w

    return ArbitrageCheckResult(
        is_arbitrage_free=(len(violations) == 0),
        violations=violations,
        n_strikes_checked=n_strikes,
        n_tenors_checked=len(sorted_tenors),
    )
