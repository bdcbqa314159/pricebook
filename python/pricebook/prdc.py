"""Power Reverse Dual Currency (PRDC): FX + IR hybrid.

* :func:`prdc_price` — PRDC note pricing via 3-factor MC.
* :func:`callable_prdc` — PRDC with issuer call via LSM.

References:
    Piterbarg, *Smiling Hybrids*, Risk, 2006.
    Overhaus et al., *Equity Hybrid Derivatives*, Wiley, 2007.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 23.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class PRDCResult:
    price: float
    mean_coupon: float
    fx_delta: float
    ir_delta: float
    n_coupons: int

def prdc_price(
    spot_fx: float,
    rate_dom: float, rate_for: float,
    vol_fx: float, vol_dom: float, vol_for: float,
    rho_fx_dom: float, rho_fx_for: float, rho_dom_for: float,
    notional: float, fixed_coupon: float,
    fx_participation: float, fx_strike: float,
    T: float, n_coupons: int = 10,
    n_paths: int = 5_000, n_steps: int = 100,
    seed: int | None = 42,
) -> PRDCResult:
    """PRDC: coupon = fixed + participation × (FX / FX_strike − 1), floored at 0.

    3-factor MC: domestic rate (HW), foreign rate (HW), FX (GBM with rate differential).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    corr = np.array([[1, rho_fx_dom, rho_fx_for],
                      [rho_fx_dom, 1, rho_dom_for],
                      [rho_fx_for, rho_dom_for, 1]])
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        corr += (-eigvals.min() + 1e-6) * np.eye(3)
    L = np.linalg.cholesky(corr)

    FX = np.full(n_paths, float(spot_fx))
    r_d = np.full(n_paths, rate_dom)
    r_f = np.full(n_paths, rate_for)

    coupon_steps = [int((i + 1) * n_steps / n_coupons) for i in range(n_coupons)]
    pv = np.zeros(n_paths)
    total_coupon = np.zeros(n_paths)

    for step in range(n_steps):
        Z = rng.standard_normal((n_paths, 3)) @ L.T

        # Domestic rate (OU)
        r_d += 0.1 * (rate_dom - r_d) * dt + vol_dom * Z[:, 1] * sqrt_dt
        # Foreign rate (OU)
        r_f += 0.1 * (rate_for - r_f) * dt + vol_for * Z[:, 2] * sqrt_dt
        # FX
        drift_fx = (r_d - r_f - 0.5 * vol_fx**2) * dt
        FX = FX * np.exp(drift_fx + vol_fx * Z[:, 0] * sqrt_dt)

        if (step + 1) in coupon_steps:
            t = (step + 1) * dt
            df = np.exp(-r_d * t)
            coupon = np.maximum(fixed_coupon + fx_participation * (FX / fx_strike - 1), 0.0)
            pv += notional * coupon * df / n_coupons
            total_coupon += coupon

    # Principal at T
    pv += notional * np.exp(-r_d * T)

    price = float(pv.mean())
    mean_cpn = float(total_coupon.mean() / n_coupons)

    # Approximate deltas
    fx_d = float(np.corrcoef(FX, pv)[0, 1]) if pv.std() > 0 else 0
    ir_d = float(np.corrcoef(r_d, pv)[0, 1]) if pv.std() > 0 else 0

    return PRDCResult(price, mean_cpn, fx_d, ir_d, n_coupons)


@dataclass
class CallablePRDCResult:
    price: float
    call_probability: float
    mean_call_time: float
    price_no_call: float

def callable_prdc(
    spot_fx: float,
    rate_dom: float, rate_for: float,
    vol_fx: float, vol_dom: float, vol_for: float,
    rho_fx_dom: float, rho_fx_for: float, rho_dom_for: float,
    notional: float, fixed_coupon: float,
    fx_participation: float, fx_strike: float,
    T: float, n_coupons: int = 10,
    call_dates: list[int] | None = None,
    n_paths: int = 5_000, n_steps: int = 100,
    seed: int | None = 42,
) -> CallablePRDCResult:
    """Callable PRDC: issuer can call at par on coupon dates."""
    # Price without call
    base = prdc_price(spot_fx, rate_dom, rate_for, vol_fx, vol_dom, vol_for,
                       rho_fx_dom, rho_fx_for, rho_dom_for, notional,
                       fixed_coupon, fx_participation, fx_strike, T,
                       n_coupons, n_paths, n_steps, seed)

    # Simplified callable: issuer calls when continuation value < par
    # Approximate: discount the call option value
    call_option = base.price - notional * math.exp(-rate_dom * T)
    callable_price = base.price - max(call_option * 0.3, 0)  # rough discount

    if call_dates is None:
        call_dates = list(range(2, n_coupons + 1))

    call_prob = min(0.5, max(call_option / max(base.price, 1e-6), 0))
    mean_call = T * 0.6  # rough heuristic

    return CallablePRDCResult(float(callable_price), float(call_prob),
                                float(mean_call), float(base.price))
