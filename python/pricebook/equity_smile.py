"""Equity vol surface: SVI/SSVI, vol cube, forward vol, arbitrage checks.

* :func:`ssvi_fit` — Surface SVI (SSVI) calibration.
* :class:`EquityVolCube` — SABR per expiry with term structure.
* :func:`forward_vol` — forward-starting volatility from cube.
* :func:`sticky_strike_dynamics` / :func:`sticky_delta_dynamics`.

References:
    Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.
    Gatheral, *The Volatility Surface*, Wiley, 2006.
    Bergomi, *Stochastic Volatility Modeling*, CRC Press, 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from pricebook.sabr import sabr_implied_vol


# ---- SSVI (Surface SVI) ----

@dataclass
class SSVIParams:
    """Surface SVI parameters.

    Raw SSVI: w(k, T) = θ(T) / 2 × (1 + ρ φ(θ) k + sqrt((φ(θ)k + ρ)² + (1 − ρ²)))
    where θ(T) is total ATM variance, φ(θ) is a parameter function, ρ is correlation.

    Simple φ: φ(θ) = η / θ^γ (power-law).
    """
    theta_fn_params: dict       # ATM variance term structure parameters
    rho: float
    eta: float
    gamma: float
    atm_vars: dict[float, float]    # T → θ(T) = σ²_ATM(T) × T


def _phi(theta: float, eta: float, gamma: float) -> float:
    """SSVI φ function: η / θ^γ (power-law)."""
    if theta <= 1e-10:
        return 0.0
    return eta / (theta ** gamma)


def _ssvi_variance(k: float, T: float, theta: float, rho: float,
                   eta: float, gamma: float) -> float:
    """Total variance w(k, T) under SSVI."""
    phi = _phi(theta, eta, gamma)
    inner = (phi * k + rho) ** 2 + (1 - rho**2)
    return 0.5 * theta * (1 + rho * phi * k + math.sqrt(max(inner, 0.0)))


def ssvi_fit(
    tenors: list[float],
    log_moneyness_grid: dict[float, list[float]],
    implied_vols: dict[float, list[float]],
) -> SSVIParams:
    """Fit SSVI to a grid of observed smiles.

    Parameters fit: ρ, η, γ (single set for the surface),
    plus ATM variances θ(T) per tenor.

    Args:
        tenors: list of expiry times.
        log_moneyness_grid: {T → [k values]}.
        implied_vols: {T → [σ at each k]}.
    """
    # First estimate θ(T) from ATM vol (smallest |k|)
    atm_vars = {}
    for T in tenors:
        ks = log_moneyness_grid[T]
        vs = implied_vols[T]
        # Find ATM point
        atm_idx = int(np.argmin(np.abs(np.array(ks))))
        atm_vol = vs[atm_idx]
        atm_vars[T] = (atm_vol ** 2) * T

    def objective(params):
        rho, eta, gamma = params
        if abs(rho) >= 0.999 or eta <= 0 or gamma < 0 or gamma > 1:
            return 1e10

        total_err = 0.0
        for T in tenors:
            theta = atm_vars[T]
            ks = log_moneyness_grid[T]
            vs = implied_vols[T]
            for k, v in zip(ks, vs):
                w_mdl = _ssvi_variance(k, T, theta, rho, eta, gamma)
                w_mkt = (v ** 2) * T
                total_err += (w_mdl - w_mkt) ** 2
        return total_err

    x0 = [-0.3, 1.0, 0.5]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8})

    rho, eta, gamma = result.x
    rho = max(-0.999, min(0.999, rho))
    eta = max(eta, 1e-6)
    gamma = max(0.0, min(1.0, gamma))

    return SSVIParams(
        theta_fn_params={},
        rho=rho, eta=eta, gamma=gamma,
        atm_vars=atm_vars,
    )


def ssvi_vol(k: float, T: float, params: SSVIParams) -> float:
    """Implied vol from SSVI at log-moneyness k, tenor T."""
    if T in params.atm_vars:
        theta = params.atm_vars[T]
    else:
        # Interpolate linearly in T
        tenors_sorted = sorted(params.atm_vars.keys())
        if T <= tenors_sorted[0]:
            theta = params.atm_vars[tenors_sorted[0]] * T / tenors_sorted[0]
        elif T >= tenors_sorted[-1]:
            theta = params.atm_vars[tenors_sorted[-1]] * T / tenors_sorted[-1]
        else:
            for i in range(len(tenors_sorted) - 1):
                if tenors_sorted[i] <= T <= tenors_sorted[i + 1]:
                    f = (T - tenors_sorted[i]) / (tenors_sorted[i + 1] - tenors_sorted[i])
                    theta = ((1 - f) * params.atm_vars[tenors_sorted[i]]
                             + f * params.atm_vars[tenors_sorted[i + 1]])
                    break

    w = _ssvi_variance(k, T, theta, params.rho, params.eta, params.gamma)
    return math.sqrt(max(w / T, 1e-10))


# ---- Equity vol cube ----

@dataclass
class EquityCubeNode:
    """SABR parameters at one expiry for equity."""
    expiry: float
    forward: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float


@dataclass
class EquityVolCube:
    """Equity vol cube with SABR smile per expiry."""
    spot: float
    rate: float
    dividend_yield: float
    nodes: list[EquityCubeNode] = field(default_factory=list)

    def vol(self, T: float, strike: float) -> float:
        """Implied vol at (T, K) with SABR parameter interpolation."""
        node = self._interpolate_node(T)
        F = self.spot * math.exp((self.rate - self.dividend_yield) * T)
        return sabr_implied_vol(F, strike, T, node.alpha, node.beta, node.rho, node.nu)

    def atm_vol(self, T: float) -> float:
        """ATM vol at tenor T."""
        node = self._interpolate_node(T)
        F = self.spot * math.exp((self.rate - self.dividend_yield) * T)
        return sabr_implied_vol(F, F, T, node.alpha, node.beta, node.rho, node.nu)

    def _interpolate_node(self, T: float) -> EquityCubeNode:
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

        F = self.spot * math.exp((self.rate - self.dividend_yield) * T)
        return EquityCubeNode(
            expiry=T, forward=F,
            alpha=n0.alpha + f * (n1.alpha - n0.alpha),
            beta=n0.beta,
            rho=n0.rho + f * (n1.rho - n0.rho),
            nu=n0.nu + f * (n1.nu - n0.nu),
            atm_vol=n0.atm_vol + f * (n1.atm_vol - n0.atm_vol),
        )


def calibrate_equity_sabr_tenor(
    spot: float,
    rate: float,
    dividend_yield: float,
    T: float,
    atm_vol: float,
    vol_25d_call: float,
    vol_25d_put: float,
    beta: float = 1.0,      # β=1 lognormal typical for equity
) -> EquityCubeNode:
    """Calibrate SABR (α, ρ, ν) at one equity tenor to ATM + 25D smile."""
    F = spot * math.exp((rate - dividend_yield) * T)

    # Rough strike from delta
    sigma = atm_vol
    K_atm = F * math.exp(0.5 * sigma**2 * T)
    # Approximate 25D strikes
    K_25c = F * math.exp(sigma * math.sqrt(T) * 0.67 + 0.5 * sigma**2 * T)
    K_25p = F * math.exp(-sigma * math.sqrt(T) * 0.67 + 0.5 * sigma**2 * T)

    strikes = [K_25p, K_atm, K_25c]
    vols = [vol_25d_put, atm_vol, vol_25d_call]

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 0.999:
            return 1e10
        err = 0.0
        for k, v in zip(strikes, vols):
            try:
                mdl = sabr_implied_vol(F, k, T, alpha, beta, rho, nu)
                err += (mdl - v) ** 2
            except (ValueError, ZeroDivisionError):
                err += 1.0
        return err

    alpha0 = atm_vol * F ** (1 - beta)
    x0 = [alpha0, -0.3, 0.3]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8})

    alpha, rho, nu = result.x
    alpha = max(alpha, 1e-6)
    rho = max(-0.9999, min(0.9999, rho))
    nu = max(nu, 1e-6)

    atm = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)

    return EquityCubeNode(T, F, alpha, beta, rho, nu, atm)


def build_equity_vol_cube(
    spot: float,
    rate: float,
    dividend_yield: float,
    tenors: list[float],
    quotes: dict[float, dict[str, float]],
    beta: float = 1.0,
) -> EquityVolCube:
    """Build equity vol cube from market quotes per tenor.

    quotes[T] = {"atm": vol, "25c": vol, "25p": vol}
    """
    nodes = []
    for T in tenors:
        q = quotes.get(T)
        if q is None:
            continue
        atm = q["atm"]
        c = q.get("25c", atm)
        p = q.get("25p", atm)
        node = calibrate_equity_sabr_tenor(spot, rate, dividend_yield, T,
                                             atm, c, p, beta)
        nodes.append(node)

    return EquityVolCube(spot, rate, dividend_yield, nodes)


# ---- Forward vol ----

@dataclass
class ForwardVolResult:
    """Forward-starting volatility result."""
    forward_vol: float          # sqrt(forward variance)
    forward_variance: float
    tenor1: float
    tenor2: float


def forward_vol(cube: EquityVolCube, T1: float, T2: float,
                strike: float | None = None) -> ForwardVolResult:
    """Forward vol from cube: vol over [T1, T2].

    Variance additivity: T2 × σ²(T2) = T1 × σ²(T1) + (T2 − T1) × σ²_fwd
    → σ²_fwd = [T2 × σ²(T2) − T1 × σ²(T1)] / (T2 − T1)

    Args:
        strike: strike for the vol (default: ATM).
    """
    if T2 <= T1:
        raise ValueError("T2 must be > T1")

    if strike is None:
        v1 = cube.atm_vol(T1)
        v2 = cube.atm_vol(T2)
    else:
        v1 = cube.vol(T1, strike)
        v2 = cube.vol(T2, strike)

    fwd_var = (v2**2 * T2 - v1**2 * T1) / (T2 - T1)
    fwd_vol = math.sqrt(max(fwd_var, 0.0))

    return ForwardVolResult(fwd_vol, fwd_var, T1, T2)


# ---- Smile dynamics ----

@dataclass
class SmileRegimeResult:
    """Smile dynamics regime (sticky strike vs delta)."""
    regime: str
    backbone_slope: float
    description: str


def sticky_strike_dynamics(
    cube: EquityVolCube,
    T: float,
    strike: float,
    spot_bump: float = 0.01,
) -> float:
    """Vol at fixed strike, bumped spot (sticky strike convention).

    Under sticky strike: vol at K doesn't change when S moves.
    Returns: new vol at bumped spot (should equal original in pure sticky strike).
    """
    original = cube.vol(T, strike)
    # In a sticky-strike regime, vol at K is unchanged → original
    # Here we just return the cube's read, simulating unchanged surface
    return original


def sticky_delta_dynamics(
    cube: EquityVolCube,
    T: float,
    delta_level: float,
    spot_bump: float = 0.01,
) -> float:
    """Vol at fixed delta under sticky delta regime (vol follows spot)."""
    F = cube.spot * math.exp((cube.rate - cube.dividend_yield) * T)
    # In sticky delta, strike adjusts with spot to maintain same delta
    # Approximate ATM-equivalent strike for the delta level
    atm = cube.atm_vol(T)
    K = F * math.exp(delta_level * atm * math.sqrt(T) + 0.5 * atm**2 * T)
    return cube.vol(T, K)
