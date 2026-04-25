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

def _prdc_reprice(
    spot_fx, rate_dom, rate_for, vol_fx, vol_dom, vol_for,
    corr, notional, fixed_coupon, fx_participation, fx_strike,
    T, n_coupons, n_paths, n_steps, rng,
) -> float:
    """Internal reprice for bump-and-reprice delta (same RNG for variance reduction)."""
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)
    L = np.linalg.cholesky(corr)

    FX = np.full(n_paths, float(spot_fx))
    r_d = np.full(n_paths, rate_dom)
    r_f = np.full(n_paths, rate_for)
    coupon_steps = set(int((i + 1) * n_steps / n_coupons) for i in range(n_coupons))
    pv = np.zeros(n_paths)

    for step in range(n_steps):
        Z = rng.standard_normal((n_paths, 3)) @ L.T
        r_d += 0.1 * (rate_dom - r_d) * dt + vol_dom * Z[:, 1] * sqrt_dt
        r_f += 0.1 * (rate_for - r_f) * dt + vol_for * Z[:, 2] * sqrt_dt
        FX = FX * np.exp((r_d - r_f - 0.5 * vol_fx**2) * dt + vol_fx * Z[:, 0] * sqrt_dt)

        if (step + 1) in coupon_steps:
            t = (step + 1) * dt
            df = np.exp(-r_d * t)
            cpn = np.maximum(fixed_coupon + fx_participation * (FX / fx_strike - 1), 0.0)
            pv += notional * cpn * df / n_coupons

    pv += notional * np.exp(-r_d * T)
    return float(pv.mean())


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

    # Approximate deltas via bump-and-reprice (pathwise not possible due to max)
    bump = spot_fx * 0.01
    rng2 = np.random.default_rng(seed)
    pv_up = _prdc_reprice(spot_fx + bump, rate_dom, rate_for, vol_fx, vol_dom,
                           vol_for, corr, notional, fixed_coupon, fx_participation,
                           fx_strike, T, n_coupons, n_paths, n_steps, rng2)
    fx_d = (pv_up - price) / bump

    rng3 = np.random.default_rng(seed)
    pv_rate_up = _prdc_reprice(spot_fx, rate_dom + 0.0001, rate_for, vol_fx, vol_dom,
                                vol_for, corr, notional, fixed_coupon, fx_participation,
                                fx_strike, T, n_coupons, n_paths, n_steps, rng3)
    ir_d = (pv_rate_up - price) / 0.0001

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
    """Callable PRDC: issuer can call at par on coupon dates via LSM.

    At each call date, issuer compares par (notional) to the estimated
    continuation value. Calls when continuation ≤ par.
    """
    if call_dates is None:
        call_dates = list(range(2, n_coupons + 1))

    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    corr = np.array([[1, rho_fx_dom, rho_fx_for],
                      [rho_fx_dom, 1, rho_dom_for],
                      [rho_fx_for, rho_dom_for, 1]])
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        corr += (-eigvals.min() + 1e-6) * np.eye(3)
    L = np.linalg.cholesky(corr)

    # Simulate full paths
    FX_all = np.zeros((n_paths, n_steps + 1))
    r_d_all = np.zeros((n_paths, n_steps + 1))
    r_f_all = np.zeros((n_paths, n_steps + 1))
    FX_all[:, 0] = spot_fx
    r_d_all[:, 0] = rate_dom
    r_f_all[:, 0] = rate_for

    for step in range(n_steps):
        Z = rng.standard_normal((n_paths, 3)) @ L.T
        r_d_all[:, step + 1] = r_d_all[:, step] + 0.1 * (rate_dom - r_d_all[:, step]) * dt + vol_dom * Z[:, 1] * sqrt_dt
        r_f_all[:, step + 1] = r_f_all[:, step] + 0.1 * (rate_for - r_f_all[:, step]) * dt + vol_for * Z[:, 2] * sqrt_dt
        drift_fx = (r_d_all[:, step] - r_f_all[:, step] - 0.5 * vol_fx**2) * dt
        FX_all[:, step + 1] = FX_all[:, step] * np.exp(drift_fx + vol_fx * Z[:, 0] * sqrt_dt)

    # Map coupon steps
    coupon_steps = [int((i + 1) * n_steps / n_coupons) for i in range(n_coupons)]
    call_step_set = set(coupon_steps[c - 1] for c in call_dates if c - 1 < n_coupons)

    # Compute coupon PVs at each coupon date (cumulative from that date forward)
    # First: price without call (forward-looking cashflows from each coupon date)
    future_pv = np.zeros(n_paths)  # PV of coupons + principal from last coupon onward

    # Backward induction: at each coupon date, compute continuation value
    # and decide whether issuer calls (continuation > par → don't call)
    called = np.zeros(n_paths, dtype=bool)
    call_time = np.full(n_paths, T)
    cashflows = np.zeros(n_paths)  # total PV of what investor receives

    # Forward pass: accumulate cashflows, with LSM call decisions
    # First compute all coupon values
    coupon_values = []
    for k, step in enumerate(coupon_steps):
        t = step * dt
        df = np.exp(-r_d_all[:, step] * t)
        coupon = np.maximum(fixed_coupon + fx_participation * (FX_all[:, step] / fx_strike - 1), 0.0)
        coupon_values.append((step, t, df, coupon))

    # LSM backward pass for issuer's call decision
    # Value-to-go for investor (= what issuer owes if not called)
    V = notional * np.exp(-r_d_all[:, -1] * T)  # principal at maturity
    for k in range(n_coupons):
        step, t, df, cpn = coupon_values[n_coupons - 1 - k]
        V += notional * cpn * df / n_coupons

    # Now backward through call dates
    call_decision = np.zeros((n_paths, n_coupons), dtype=bool)

    # Continuation value at each call date
    cont_value = notional * np.exp(-r_d_all[:, -1] * T)  # terminal principal

    for k in range(n_coupons - 1, -1, -1):
        step, t, df, cpn = coupon_values[k]
        # Add this coupon to continuation
        cont_value += notional * cpn * df / n_coupons

        if step in call_step_set:
            # Issuer call value = par (notional × df)
            par_value = notional * df

            # LSM: regress continuation on state (FX, rate)
            fx_k = FX_all[:, step]
            rd_k = r_d_all[:, step]
            fx_norm = (fx_k - fx_k.mean()) / max(fx_k.std(), 1e-10)
            rd_norm = (rd_k - rd_k.mean()) / max(rd_k.std(), 1e-10)
            basis = np.column_stack([np.ones(n_paths), fx_norm, rd_norm,
                                      fx_norm**2, rd_norm**2, fx_norm * rd_norm])
            try:
                coeffs = np.linalg.lstsq(basis, cont_value, rcond=None)[0]
                est_cont = basis @ coeffs
            except np.linalg.LinAlgError:
                est_cont = cont_value

            # Issuer calls when par < continuation (saves money)
            call_decision[:, k] = par_value < est_cont

    # Forward pass: apply call decisions
    alive = np.ones(n_paths, dtype=bool)
    pv_investor = np.zeros(n_paths)

    for k in range(n_coupons):
        step, t, df, cpn = coupon_values[k]
        # Pay coupon to alive paths
        pv_investor += np.where(alive, notional * cpn * df / n_coupons, 0.0)

        # Check call
        if step in call_step_set:
            issuer_calls = alive & call_decision[:, k]
            # Called paths receive par
            pv_investor += np.where(issuer_calls, notional * df, 0.0)
            call_time = np.where(issuer_calls & alive, t, call_time)
            alive &= ~issuer_calls

    # Terminal: alive paths receive principal
    pv_investor += np.where(alive, notional * np.exp(-r_d_all[:, -1] * T), 0.0)

    callable_price = float(pv_investor.mean())
    call_prob = float(1 - alive.mean())
    if call_prob > 1e-6:
        ac_mask = call_time < T
        mean_call = float(call_time[ac_mask].mean()) if ac_mask.sum() > 0 else T
    else:
        mean_call = T

    # Price without call for comparison
    base = prdc_price(spot_fx, rate_dom, rate_for, vol_fx, vol_dom, vol_for,
                       rho_fx_dom, rho_fx_for, rho_dom_for, notional,
                       fixed_coupon, fx_participation, fx_strike, T,
                       n_coupons, n_paths, n_steps, seed)

    return CallablePRDCResult(callable_price, call_prob, mean_call, base.price)
