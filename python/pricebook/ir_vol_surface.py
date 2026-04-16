"""IR volatility surface deepening: SABR vol cube, smile dynamics.

Extends :mod:`pricebook.swaption_vol` with strike-dependent surfaces:

* :class:`SABRSmileNode` — SABR parameters at one (expiry, tenor) point.
* :class:`SwaptionVolCube` — full vol cube with SABR smiles.
* :func:`calibrate_sabr_smile` — fit SABR to market smile at one node.
* :func:`build_vol_cube` — calibrate full cube from market data.
* :func:`smile_dynamics` — backbone analysis (sticky strike vs sticky delta).

References:
    Hagan et al., *Managing Smile Risk*, Wilmott, 2002.
    Rebonato, *Volatility and Correlation*, Wiley, Ch. 16-17.
    Andersen & Piterbarg, *Interest Rate Modeling*, Vol. 2, Ch. 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pricebook.sabr import sabr_implied_vol


# ---- SABR smile at one node ----

@dataclass
class SABRSmileNode:
    """SABR parameters at one (expiry, tenor) point."""
    expiry: float          # option expiry in years
    tenor: float           # swap tenor in years
    forward: float         # forward swap rate
    alpha: float           # initial vol
    beta: float            # CEV exponent
    rho: float             # correlation
    nu: float              # vol of vol
    atm_vol: float         # ATM implied vol from SABR
    residual: float        # calibration residual


@dataclass
class SmileDynamicsResult:
    """Backbone / smile dynamics analysis."""
    regime: str            # "sticky_strike", "sticky_delta", or "mixed"
    backbone_slope: float  # d(ATM_vol)/d(forward) × forward/vol
    description: str


def calibrate_sabr_smile(
    forward: float,
    expiry: float,
    strikes: list[float],
    market_vols: list[float],
    beta: float = 0.5,
) -> SABRSmileNode:
    """Calibrate SABR (α, ρ, ν) to market smile at one node.

    β is typically fixed (0 = normal, 0.5 = CIR, 1 = lognormal).
    The remaining parameters are fitted to minimise Σ(model − market)².

    Args:
        forward: forward swap rate.
        expiry: time to option expiry.
        strikes: market strikes.
        market_vols: corresponding market implied vols (lognormal).
        beta: fixed CEV exponent.
    """
    K = np.array(strikes)
    V = np.array(market_vols)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        total = 0.0
        for k, v_mkt in zip(K, V):
            try:
                v_mdl = sabr_implied_vol(forward, k, expiry, alpha, beta, rho, nu)
                total += (v_mdl - v_mkt) ** 2
            except (ValueError, ZeroDivisionError):
                total += 1.0
        return total

    # Initial guess: alpha from ATM vol
    atm_vol = float(np.interp(forward, K, V)) if len(K) > 1 else V[0]
    alpha0 = atm_vol * forward ** (1 - beta)

    x0 = [alpha0, -0.2, 0.3]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8})

    alpha, rho, nu = result.x
    alpha = max(alpha, 1e-6)
    rho = max(-0.9999, min(0.9999, rho))
    nu = max(nu, 1e-6)

    # ATM vol from calibrated SABR
    atm = sabr_implied_vol(forward, forward, expiry, alpha, beta, rho, nu)
    residual = math.sqrt(result.fun / len(K))

    return SABRSmileNode(expiry, 0.0, forward, alpha, beta, rho, nu, atm, residual)


# ---- Vol cube ----

@dataclass
class VolCubeNode:
    """Internal node in the vol cube."""
    expiry: float
    tenor: float
    sabr: SABRSmileNode


class SwaptionVolCube:
    """Swaption vol cube: vol(expiry, tenor, strike) via SABR smiles.

    At each (expiry, tenor) grid point, a full SABR smile is calibrated.
    Between grid points, SABR parameters are interpolated bilinearly.

    Args:
        nodes: list of calibrated SABR nodes.
        expiries: sorted unique expiry times.
        tenors: sorted unique swap tenors.
    """

    def __init__(self, nodes: list[VolCubeNode],
                 expiries: list[float], tenors: list[float]):
        self.expiries = np.array(sorted(set(expiries)))
        self.tenors = np.array(sorted(set(tenors)))
        self._nodes: dict[tuple[float, float], SABRSmileNode] = {}
        for n in nodes:
            self._nodes[(n.expiry, n.tenor)] = n.sabr

    def vol(self, expiry: float, tenor: float, strike: float,
            forward: float | None = None) -> float:
        """Implied vol at (expiry, tenor, strike).

        Interpolates SABR parameters bilinearly, then evaluates smile.
        """
        sabr = self._interpolate_sabr(expiry, tenor)
        fwd = forward if forward is not None else sabr.forward
        return sabr_implied_vol(fwd, strike, expiry,
                                sabr.alpha, sabr.beta, sabr.rho, sabr.nu)

    def atm_vol(self, expiry: float, tenor: float) -> float:
        """ATM implied vol at (expiry, tenor)."""
        sabr = self._interpolate_sabr(expiry, tenor)
        return sabr_implied_vol(sabr.forward, sabr.forward, expiry,
                                sabr.alpha, sabr.beta, sabr.rho, sabr.nu)

    def smile(self, expiry: float, tenor: float,
              strike_offsets: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (strikes, vols) for the smile at (expiry, tenor).

        Args:
            strike_offsets: offsets from ATM in absolute terms.
                Default: [-200bp, -100bp, ..., +200bp].
        """
        sabr = self._interpolate_sabr(expiry, tenor)
        if strike_offsets is None:
            strike_offsets = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]

        strikes = np.array([sabr.forward + off for off in strike_offsets])
        strikes = np.maximum(strikes, 1e-6)
        vols = np.array([
            sabr_implied_vol(sabr.forward, k, expiry,
                             sabr.alpha, sabr.beta, sabr.rho, sabr.nu)
            for k in strikes
        ])
        return strikes, vols

    def _interpolate_sabr(self, expiry: float, tenor: float) -> SABRSmileNode:
        """Bilinearly interpolate SABR parameters."""
        exp = np.clip(expiry, self.expiries[0], self.expiries[-1])
        ten = np.clip(tenor, self.tenors[0], self.tenors[-1])

        # Find bounding indices
        if len(self.expiries) < 2 or len(self.tenors) < 2:
            # Not enough nodes for bilinear — return nearest
            best_key = min(self._nodes.keys(),
                           key=lambda k: (k[0] - exp)**2 + (k[1] - ten)**2)
            return self._nodes[best_key]

        i = max(0, min(int(np.searchsorted(self.expiries, exp)) - 1, len(self.expiries) - 2))
        j = max(0, min(int(np.searchsorted(self.tenors, ten)) - 1, len(self.tenors) - 2))

        e0, e1 = self.expiries[i], self.expiries[i + 1]
        t0, t1 = self.tenors[j], self.tenors[j + 1]

        fx = (exp - e0) / (e1 - e0) if e1 > e0 else 0.0
        fy = (ten - t0) / (t1 - t0) if t1 > t0 else 0.0

        def get_node(ei, tj):
            key = (ei, tj)
            if key in self._nodes:
                return self._nodes[key]
            # Fallback: nearest
            best_key = min(self._nodes.keys(),
                           key=lambda k: (k[0] - ei)**2 + (k[1] - tj)**2)
            return self._nodes[best_key]

        n00 = get_node(e0, t0)
        n01 = get_node(e0, t1)
        n10 = get_node(e1, t0)
        n11 = get_node(e1, t1)

        def interp(attr):
            v00 = getattr(n00, attr)
            v01 = getattr(n01, attr)
            v10 = getattr(n10, attr)
            v11 = getattr(n11, attr)
            return v00 * (1-fx)*(1-fy) + v01 * (1-fx)*fy + v10 * fx*(1-fy) + v11 * fx*fy

        return SABRSmileNode(
            expiry=expiry, tenor=tenor,
            forward=interp('forward'),
            alpha=interp('alpha'),
            beta=interp('beta'),
            rho=interp('rho'),
            nu=interp('nu'),
            atm_vol=interp('atm_vol'),
            residual=0.0,
        )


def build_vol_cube(
    expiries: list[float],
    tenors: list[float],
    forwards: dict[tuple[float, float], float],
    market_smiles: dict[tuple[float, float], tuple[list[float], list[float]]],
    beta: float = 0.5,
) -> SwaptionVolCube:
    """Build full vol cube by calibrating SABR at each node.

    Args:
        expiries: expiry times (years).
        tenors: swap tenors (years).
        forwards: {(expiry, tenor) → forward_swap_rate}.
        market_smiles: {(expiry, tenor) → (strikes, vols)}.
        beta: fixed SABR β.
    """
    nodes = []
    for exp in expiries:
        for ten in tenors:
            key = (exp, ten)
            if key not in market_smiles:
                continue
            fwd = forwards.get(key, 0.05)
            strikes, vols = market_smiles[key]
            sabr = calibrate_sabr_smile(fwd, exp, strikes, vols, beta)
            sabr_with_tenor = SABRSmileNode(
                exp, ten, sabr.forward, sabr.alpha, sabr.beta,
                sabr.rho, sabr.nu, sabr.atm_vol, sabr.residual,
            )
            nodes.append(VolCubeNode(exp, ten, sabr_with_tenor))

    return SwaptionVolCube(nodes, expiries, tenors)


def smile_dynamics(
    forward: float,
    expiry: float,
    sabr_node: SABRSmileNode,
    bump: float = 0.001,
) -> SmileDynamicsResult:
    """Analyse smile dynamics: sticky strike vs sticky delta.

    Computes the backbone slope: how ATM vol changes with forward rate.

    - Backbone slope ≈ 0 → sticky strike (vol fixed at given K).
    - Backbone slope ≈ −1 → sticky delta (vol moves 1:1 with forward).
    - β ≈ 0 → normal backbone (sticky strike in normal vol).
    - β ≈ 1 → lognormal backbone (sticky strike in lognormal vol).

    Args:
        forward: current forward rate.
        expiry: time to expiry.
        sabr_node: calibrated SABR parameters.
        bump: forward rate bump size.
    """
    f_up = forward + bump
    f_dn = forward - bump

    vol_base = sabr_implied_vol(forward, forward, expiry,
                                sabr_node.alpha, sabr_node.beta,
                                sabr_node.rho, sabr_node.nu)
    vol_up = sabr_implied_vol(f_up, f_up, expiry,
                              sabr_node.alpha, sabr_node.beta,
                              sabr_node.rho, sabr_node.nu)
    vol_dn = sabr_implied_vol(f_dn, f_dn, expiry,
                              sabr_node.alpha, sabr_node.beta,
                              sabr_node.rho, sabr_node.nu)

    # Backbone slope: d(ln σ)/d(ln F)
    if vol_base > 0 and forward > 0:
        slope = (math.log(vol_up) - math.log(vol_dn)) / (math.log(f_up) - math.log(f_dn))
    else:
        slope = 0.0

    if abs(slope) < 0.15:
        regime = "sticky_strike"
        desc = f"Near sticky strike (slope={slope:.3f}): vol anchored at fixed strikes"
    elif slope < -0.7:
        regime = "sticky_delta"
        desc = f"Near sticky delta (slope={slope:.3f}): vol moves with forward"
    else:
        regime = "mixed"
        desc = f"Mixed regime (slope={slope:.3f}): between sticky strike and delta"

    return SmileDynamicsResult(regime, slope, desc)
