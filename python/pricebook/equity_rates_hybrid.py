"""Equity-rates hybrids: callable equity notes, joint simulation.

* :func:`callable_equity_note` — equity payoff with rate call provision.
* :func:`equity_ir_joint_simulate` — joint equity + rates MC.
* :func:`hybrid_autocallable` — autocall with IR floor/cap.

References:
    Overhaus et al., *Equity Hybrid Derivatives*, Wiley, 2007.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 23.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class CallableEquityNoteResult:
    price: float
    equity_delta: float
    rate_delta: float
    call_probability: float

def callable_equity_note(
    spot: float, rate: float, equity_vol: float, rate_vol: float,
    rho: float, notional: float, participation: float,
    strike_pct: float, T: float, call_dates: list[float],
    n_paths: int = 5_000, n_steps: int = 100, seed: int | None = 42,
) -> CallableEquityNoteResult:
    """Equity-linked note with issuer call right."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    S = np.full(n_paths, float(spot))
    r = np.full(n_paths, rate)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)

    call_steps = set(int(t * n_steps / T) for t in call_dates)

    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.standard_normal(n_paths)

        S = S * np.exp((r - 0.5 * equity_vol**2) * dt + equity_vol * z1 * sqrt_dt)
        r += 0.1 * (rate - r) * dt + rate_vol * z2 * sqrt_dt

        if (step + 1) in call_steps:
            t = (step + 1) * dt
            eq_return = S / spot - 1
            cont_value = notional * (1 + participation * np.maximum(eq_return - strike_pct, 0))
            called = alive & (cont_value > notional * 1.05)
            pv += np.where(called, notional * np.exp(-rate * t), 0)
            alive &= ~called

    # Terminal
    eq_return = S / spot - 1
    terminal = notional * (1 + participation * np.maximum(eq_return - strike_pct, 0))
    pv += np.where(alive, terminal * np.exp(-r * T), 0)

    price = float(pv.mean())
    call_prob = float(1 - alive.mean())
    eq_d = float(np.corrcoef(S, pv)[0, 1]) if pv.std() > 0 else 0
    ir_d = float(np.corrcoef(r, pv)[0, 1]) if pv.std() > 0 else 0

    return CallableEquityNoteResult(price, eq_d, ir_d, call_prob)


@dataclass
class JointSimResult:
    equity_paths: np.ndarray
    rate_paths: np.ndarray
    correlation_realised: float

def equity_ir_joint_simulate(
    spot: float, rate: float, equity_vol: float, rate_vol: float,
    rho: float, T: float, n_paths: int = 2000, n_steps: int = 50,
    seed: int | None = 42,
) -> JointSimResult:
    """Joint equity + Hull-White rate simulation."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    S = np.full((n_paths, n_steps + 1), float(spot))
    r = np.full((n_paths, n_steps + 1), rate)

    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        S[:, step+1] = S[:, step] * np.exp((r[:, step] - 0.5*equity_vol**2)*dt + equity_vol*z1*sqrt_dt)
        r[:, step+1] = r[:, step] + 0.1*(rate - r[:, step])*dt + rate_vol*z2*sqrt_dt

    log_r_eq = np.diff(np.log(S), axis=1)
    log_r_ir = np.diff(r, axis=1)
    corr = float(np.mean([np.corrcoef(log_r_eq[p], log_r_ir[p])[0,1]
                            for p in range(min(n_paths, 100))
                            if log_r_eq[p].std() > 1e-10 and log_r_ir[p].std() > 1e-10]))

    return JointSimResult(S, r, corr)


@dataclass
class HybridAutocallResult:
    price: float
    autocall_probability: float
    ir_floor_triggered: float

def hybrid_autocallable(
    spot: float, rate: float, equity_vol: float, rate_vol: float,
    rho: float, notional: float, coupon: float,
    autocall_barrier_pct: float, ir_floor: float,
    T: float, observation_steps: list[int],
    n_paths: int = 5_000, n_steps: int = 100, seed: int | None = 42,
) -> HybridAutocallResult:
    """Autocall with IR floor: only autocalls if equity above barrier AND rate above floor."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    S = np.full(n_paths, float(spot))
    r = np.full(n_paths, rate)
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    ir_floor_count = 0

    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5*equity_vol**2)*dt + equity_vol*z1*sqrt_dt)
        r += 0.1*(rate - r)*dt + rate_vol*z2*sqrt_dt

        if (step + 1) in observation_steps:
            t = (step + 1) * dt
            eq_trigger = S / spot >= autocall_barrier_pct
            ir_ok = r >= ir_floor
            triggered = alive & eq_trigger & ir_ok
            ir_blocked = alive & eq_trigger & ~ir_ok
            ir_floor_count += int(ir_blocked.sum())

            pv += np.where(triggered, (notional + coupon) * math.exp(-rate * t), 0)
            alive &= ~triggered

    pv += np.where(alive, notional * math.exp(-rate * T), 0)

    return HybridAutocallResult(
        float(pv.mean()), float(1 - alive.mean()),
        float(ir_floor_count / max(n_paths * len(observation_steps), 1)))
