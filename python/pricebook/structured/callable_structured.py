"""
Callable steepener, CMS spread notes and callable inverse floaters.

All instruments are priced via the Longstaff-Schwartz Monte Carlo (LSM)
algorithm. At each Bermudan call date the issuer decides whether to call
the note by comparing the continuation value (cost of keeping the note
alive — future coupon obligations discounted back) with the call price
(typically par). The issuer calls when the note is expensive, i.e. when
the continuation value exceeds par.

LSM regression basis: quadratic polynomial in (long_rate, short_rate) or in
the single driving rate, following Piterbarg's recommended state variables for
CMS spread products.

    from pricebook.structured.callable_structured import callable_steepener

    result = callable_steepener(
        long_rate=0.04, short_rate=0.02,
        fixed_coupon=0.0, leverage=3.0, floor=0.0, cap=0.10,
        call_dates_years=[1.0, 2.0, 3.0, 4.0],
        maturity_years=5.0,
        rate=0.04, vol_long=0.005, vol_short=0.004, rho=-0.3,
    )

References:
    Piterbarg, *Rates Squared*, Risk Magazine, March 2004.
    Andersen & Piterbarg, *Interest Rate Modeling*, Vol. III, Ch. 17, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CallableSteepenerResult:
    """Result from callable steepener / CMS-slope note pricing.

    Attributes:
        price: note price per 100 notional.
        straight_price: price without the call option (straight steepener).
        call_option_value: |straight_price - price|, issuer's call value.
        expected_coupon: annualised expected coupon (per period average).
        call_probability: estimated probability the issuer exercises the call.
    """
    price: float
    straight_price: float
    call_option_value: float
    expected_coupon: float
    call_probability: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CallableCMSSpreadResult:
    """Result from callable CMS spread note pricing."""
    price: float
    straight_price: float
    call_option_value: float
    expected_coupon: float
    call_probability: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CallableInverseFloaterResult:
    """Result from callable inverse floater pricing."""
    price: float
    straight_price: float
    call_option_value: float
    expected_coupon: float
    call_probability: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cholesky2(rho: float) -> np.ndarray:
    """2x2 Cholesky factor for correlation rho."""
    rho = max(-0.9999, min(0.9999, rho))
    return np.array([[1.0, 0.0],
                     [rho, math.sqrt(1.0 - rho * rho)]])


def _simulate_two_rates(
    r_long0: float,
    r_short0: float,
    vol_long: float,
    vol_short: float,
    rho: float,
    rate: float,
    maturity_years: float,
    n_steps_per_year: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate correlated GBM rate paths and cumulative discount factors.

    Both rates follow arithmetic (normal) diffusion (driftless in the
    risk-neutral measure is approximated by drift = rate * dt to keep rates
    positive — a common working approximation for short tenors).

    Returns:
        rates_long: shape (n_paths, n_steps + 1)
        rates_short: shape (n_paths, n_steps + 1)
        cum_disc: shape (n_paths, n_steps + 1), P(0, t_i) per path
    """
    n_steps = int(maturity_years * n_steps_per_year)
    dt = maturity_years / n_steps
    L = _cholesky2(rho)
    rng = np.random.default_rng(seed)

    rates_long = np.zeros((n_paths, n_steps + 1))
    rates_short = np.zeros((n_paths, n_steps + 1))
    cum_disc = np.ones((n_paths, n_steps + 1))

    rates_long[:, 0] = r_long0
    rates_short[:, 0] = r_short0

    # Fix T4-STRUCT: pre-fix the ``rate`` parameter was documented in the
    # comment above as providing drift (``drift = rate * dt to keep rates
    # positive``) but the update step never used it — pure driftless ABM
    # with no upward bias.  Now the documented drift is applied each step
    # so paths stay in a sensible range and respect the stated convention.
    sqdt = math.sqrt(dt)
    drift = rate * dt
    for i in range(n_steps):
        Z = rng.standard_normal((2, n_paths))
        dW = L @ Z  # shape (2, n_paths)
        rates_long[:, i + 1] = rates_long[:, i] + drift + vol_long * sqdt * dW[0]
        rates_short[:, i + 1] = rates_short[:, i] + drift + vol_short * sqdt * dW[1]
        # Discount: use average rate over the step
        r_avg = 0.5 * (rates_long[:, i] + rates_long[:, i + 1])
        r_avg = np.maximum(r_avg, 0.0)
        cum_disc[:, i + 1] = cum_disc[:, i] * np.exp(-r_avg * dt)

    return rates_long, rates_short, cum_disc


def _simulate_one_rate(
    r0: float,
    vol: float,
    rate: float,
    maturity_years: float,
    n_steps_per_year: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate single rate paths under normal diffusion.

    Returns:
        rates: shape (n_paths, n_steps + 1)
        cum_disc: shape (n_paths, n_steps + 1)
    """
    n_steps = int(maturity_years * n_steps_per_year)
    dt = maturity_years / n_steps
    rng = np.random.default_rng(seed)
    sqdt = math.sqrt(dt)

    rates = np.zeros((n_paths, n_steps + 1))
    cum_disc = np.ones((n_paths, n_steps + 1))
    rates[:, 0] = r0
    # Fix T4-STRUCT: ``rate`` was silently unused in the rate update; now
    # applies the documented ``rate * dt`` drift (same fix as in
    # _simulate_two_rates above).
    drift = rate * dt

    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        rates[:, i + 1] = rates[:, i] + drift + vol * sqdt * Z
        r_avg = 0.5 * (rates[:, i] + rates[:, i + 1])
        r_avg = np.maximum(r_avg, 0.0)
        cum_disc[:, i + 1] = cum_disc[:, i] * np.exp(-r_avg * dt)

    return rates, cum_disc


def _lsm_call_decision(
    cashflows: np.ndarray,
    cashflow_times: np.ndarray,
    call_step_indices: list[int],
    cum_disc: np.ndarray,
    regressors: np.ndarray,
    call_price: float,
    n_basis: int = 6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """LSM backward induction for issuer call.

    The issuer calls when the continuation value (cost of keeping the note)
    exceeds the call price. We regress the discounted future cashflows onto
    the state variables at each call date.

    Args:
        cashflows: per-path PV of all coupons + principal (pre-computed
            without early call), shape (n_paths, n_periods).
        cashflow_times: step index at which each cashflow occurs.
        call_step_indices: sorted list of tree step indices for call dates.
        cum_disc: cumulative discount factors, shape (n_paths, n_steps+1).
        regressors: state variables, shape (n_paths, n_steps+1, n_reg).
        call_price: issuer's redemption price (per 100).
        n_basis: number of polynomial basis functions.

    Returns:
        (is_called_path, call_step_per_path, call_probability).
    """
    n_paths = cashflows.shape[0]
    n_periods = cashflows.shape[1]

    # Total undiscounted (from t=0) PV of all future cashflows per path
    # We store the step at which each cashflow lands.
    # For LSM: work backwards from last call date.

    # Remaining cashflow: initially all cashflows survive.
    # cashflow_pv[p] = sum of future cashflows discounted to t=0, path p.
    cashflow_pv = np.zeros(n_paths)
    for k in range(n_periods):
        s = int(cashflow_times[k])
        df = cum_disc[:, s]
        cashflow_pv += cashflows[:, k] * df

    is_called = np.zeros(n_paths, dtype=bool)
    call_step = np.full(n_paths, -1, dtype=int)

    for call_s in sorted(call_step_indices, reverse=True):
        # Paths still alive at this call date
        alive = ~is_called
        if alive.sum() < n_basis + 1:
            continue

        # Continuation value for alive paths = PV of future cashflows
        # (cashflows occurring after call_s only)
        cont_pv = np.zeros(n_paths)
        for k in range(n_periods):
            s = int(cashflow_times[k])
            if s > call_s:
                cont_pv += cashflows[:, k] * cum_disc[:, s]

        # Discount to call_s
        df_call = cum_disc[:, call_s]
        df_call = np.where(df_call > 1e-15, df_call, 1e-15)
        cont_at_call = np.where(alive, cont_pv / df_call, 0.0)

        # Regression: estimate continuation value from state vars
        X = regressors[:, call_s, :]  # (n_paths, n_reg)
        y = cont_at_call[alive]
        X_alive = X[alive]

        # Build quadratic basis
        cols = [np.ones(alive.sum())]
        for f in range(X_alive.shape[1]):
            x_col = X_alive[:, f]
            x_mean = x_col.mean()
            x_std = x_col.std()
            if x_std < 1e-12:
                x_std = 1.0
            x_norm = (x_col - x_mean) / x_std
            cols.append(x_norm)
            cols.append(x_norm ** 2)
        basis = np.column_stack(cols[:n_basis])

        try:
            coeffs = np.linalg.lstsq(basis, y, rcond=None)[0]
            cont_estimate = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Issuer calls if estimated continuation value > call_price
        exercise = alive.copy()
        exercise[alive] = cont_estimate > call_price

        is_called[exercise] = True
        call_step[exercise] = call_s

    call_probability = float(is_called.mean())
    return is_called, call_step, call_probability


# ---------------------------------------------------------------------------
# Public API: callable steepener
# ---------------------------------------------------------------------------


def callable_steepener(
    long_rate: float,
    short_rate: float,
    fixed_coupon: float,
    leverage: float,
    floor: float,
    cap: float,
    call_dates_years: list[float],
    maturity_years: float,
    rate: float,
    vol_long: float,
    vol_short: float,
    rho: float,
    n_paths: int = 50_000,
    n_steps_per_year: int = 12,
    seed: int = 42,
) -> CallableSteepenerResult:
    """Callable CMS steepener note priced via LSM Monte Carlo.

    Coupon per period:
        coupon = max(floor, min(cap, fixed_coupon + leverage * (long_rate - short_rate)))

    The issuer calls when the estimated continuation value (PV of remaining
    obligations) exceeds par. LSM regression uses a quadratic basis in
    (long_rate, short_rate).

    Args:
        long_rate: initial long-end CMS rate (e.g. CMS10).
        short_rate: initial short-end CMS rate (e.g. CMS2).
        fixed_coupon: fixed coupon add-on (e.g. 0.0 for pure slope note).
        leverage: multiplier on the slope (e.g. 3.0).
        floor: minimum coupon per period (e.g. 0.0).
        cap: maximum coupon per period (e.g. 0.10).
        call_dates_years: Bermudan call dates (year fractions).
        maturity_years: note maturity.
        rate: risk-free rate for discounting (approximation; tree uses path rates).
        vol_long: annual vol of the long CMS rate.
        vol_short: annual vol of the short CMS rate.
        rho: correlation between long and short rate innovations.
        n_paths: number of MC paths.
        n_steps_per_year: simulation steps per year.
        seed: RNG seed.

    Returns:
        CallableSteepenerResult.
    """
    n_steps = int(maturity_years * n_steps_per_year)
    dt = maturity_years / n_steps
    coupon_steps = list(range(1, n_steps + 1))  # coupon every step

    rates_long, rates_short, cum_disc = _simulate_two_rates(
        long_rate, short_rate, vol_long, vol_short, rho,
        rate, maturity_years, n_steps_per_year, n_paths, seed,
    )

    # Compute per-period coupons (per 100 notional)
    cashflows = np.zeros((n_paths, n_steps))
    for i, s in enumerate(coupon_steps):
        slope = rates_long[:, s] - rates_short[:, s]
        cpn = np.clip(fixed_coupon + leverage * slope, floor, cap)
        cashflows[:, i] = cpn * dt * 100.0

    # Add principal at maturity
    cashflows[:, -1] += 100.0

    cashflow_times = np.array(coupon_steps, dtype=float)

    # Straight note price (no call)
    straight_pv = np.zeros(n_paths)
    for i, s in enumerate(coupon_steps):
        straight_pv += cashflows[:, i] * cum_disc[:, s]
    straight_price = float(straight_pv.mean())

    # Regressors: (long_rate, short_rate) at each step
    regressors = np.stack([rates_long, rates_short], axis=2)  # (n_paths, n_steps+1, 2)

    call_step_indices = [int(round(t / dt)) for t in call_dates_years
                         if 0 < t <= maturity_years + 1e-10]

    is_called, call_step_arr, call_prob = _lsm_call_decision(
        cashflows, cashflow_times, call_step_indices,
        cum_disc, regressors, call_price=100.0, n_basis=6,
    )

    # Callable note price: for called paths, receive call_price at call date;
    # for uncalled paths, receive full cashflow stream.
    callable_pv = np.zeros(n_paths)
    for p in range(n_paths):
        if is_called[p]:
            s = call_step_arr[p]
            # Coupons up to and including call date + call price
            pv = 0.0
            for i, cs in enumerate(coupon_steps):
                if cs <= s:
                    pv += cashflows[p, i] * cum_disc[p, cs]
            pv += 100.0 * cum_disc[p, s]
            callable_pv[p] = pv
        else:
            for i, cs in enumerate(coupon_steps):
                callable_pv[p] += cashflows[p, i] * cum_disc[p, cs]

    price = float(callable_pv.mean())

    # Expected coupon: average per-period coupon across paths and periods
    expected_cpn = float(np.mean(cashflows[:, :-1] / dt / 100.0))  # annualised, excl. principal

    return CallableSteepenerResult(
        price=price,
        straight_price=straight_price,
        call_option_value=max(0.0, straight_price - price),
        expected_coupon=expected_cpn,
        call_probability=call_prob,
    )


# ---------------------------------------------------------------------------
# Public API: callable CMS spread note
# ---------------------------------------------------------------------------


def callable_cms_spread(
    cms_long_tenor: float,
    cms_short_tenor: float,
    strike: float,
    cap: float,
    maturity_years: float,
    call_dates_years: list[float],
    rate: float,
    vol_long: float,
    vol_short: float,
    rho: float,
    n_paths: int = 50_000,
    seed: int = 42,
) -> CallableCMSSpreadResult:
    """Callable CMS spread note priced via LSM.

    Coupon per period:
        coupon = max(0, min(cap, CMS_long - CMS_short - strike))

    The issuer calls when the PV of remaining obligations (estimated via LSM)
    exceeds par. State variables for regression: (CMS_long, CMS_short).

    Args:
        cms_long_tenor: initial long CMS rate (e.g. CMS10 = 0.04).
        cms_short_tenor: initial short CMS rate (e.g. CMS2 = 0.02).
        strike: spread strike rate (e.g. 0.01 = 100 bp).
        cap: maximum coupon rate per period.
        maturity_years: note tenor in years.
        call_dates_years: Bermudan call date schedule (year fractions).
        rate: flat risk-free rate.
        vol_long: annual vol of the long CMS rate.
        vol_short: annual vol of the short CMS rate.
        rho: CMS long/short correlation.
        n_paths: number of MC paths.
        seed: RNG seed.

    Returns:
        CallableCMSSpreadResult.
    """
    n_steps_per_year = 12
    n_steps = int(maturity_years * n_steps_per_year)
    dt = maturity_years / n_steps
    coupon_steps = list(range(1, n_steps + 1))

    rates_long, rates_short, cum_disc = _simulate_two_rates(
        cms_long_tenor, cms_short_tenor, vol_long, vol_short, rho,
        rate, maturity_years, n_steps_per_year, n_paths, seed,
    )

    cashflows = np.zeros((n_paths, n_steps))
    for i, s in enumerate(coupon_steps):
        spread = rates_long[:, s] - rates_short[:, s] - strike
        cpn = np.clip(spread, 0.0, cap)
        cashflows[:, i] = cpn * dt * 100.0

    cashflows[:, -1] += 100.0

    cashflow_times = np.array(coupon_steps, dtype=float)

    straight_pv = np.zeros(n_paths)
    for i, s in enumerate(coupon_steps):
        straight_pv += cashflows[:, i] * cum_disc[:, s]
    straight_price = float(straight_pv.mean())

    regressors = np.stack([rates_long, rates_short], axis=2)
    call_step_indices = [int(round(t / dt)) for t in call_dates_years
                         if 0 < t <= maturity_years + 1e-10]

    is_called, call_step_arr, call_prob = _lsm_call_decision(
        cashflows, cashflow_times, call_step_indices,
        cum_disc, regressors, call_price=100.0, n_basis=6,
    )

    callable_pv = np.zeros(n_paths)
    for p in range(n_paths):
        if is_called[p]:
            s = call_step_arr[p]
            pv = 0.0
            for i, cs in enumerate(coupon_steps):
                if cs <= s:
                    pv += cashflows[p, i] * cum_disc[p, cs]
            pv += 100.0 * cum_disc[p, s]
            callable_pv[p] = pv
        else:
            for i, cs in enumerate(coupon_steps):
                callable_pv[p] += cashflows[p, i] * cum_disc[p, cs]

    price = float(callable_pv.mean())
    expected_cpn = float(np.mean(cashflows[:, :-1] / dt / 100.0))

    return CallableCMSSpreadResult(
        price=price,
        straight_price=straight_price,
        call_option_value=max(0.0, straight_price - price),
        expected_coupon=expected_cpn,
        call_probability=call_prob,
    )


# ---------------------------------------------------------------------------
# Public API: callable inverse floater
# ---------------------------------------------------------------------------


def callable_inverse_floater(
    fixed_rate: float,
    floating_rate: float,
    leverage: float,
    floor: float,
    maturity_years: float,
    call_dates_years: list[float],
    rate: float,
    vol: float,
    n_paths: int = 50_000,
    seed: int = 42,
) -> CallableInverseFloaterResult:
    """Callable inverse floater priced via LSM.

    Coupon per period:
        coupon = max(floor, fixed_rate - leverage * floating_rate)

    As rates rise the coupon falls, making this note attractive when rates
    are expected to fall. The issuer calls when the note becomes expensive
    (i.e. rates have fallen, coupons are high). LSM regression uses the
    simulated floating rate as the single state variable.

    Args:
        fixed_rate: fixed component of the inverse floater coupon.
        floating_rate: initial floating rate (e.g. SOFR = 0.04).
        leverage: multiplier on the floating rate (e.g. 1.0 or 2.0).
        floor: coupon floor (typically 0.0).
        maturity_years: note tenor in years.
        call_dates_years: Bermudan call dates (year fractions).
        rate: risk-free discounting rate (approximation).
        vol: annual vol of the floating rate.
        n_paths: number of MC paths.
        seed: RNG seed.

    Returns:
        CallableInverseFloaterResult.
    """
    n_steps_per_year = 12
    n_steps = int(maturity_years * n_steps_per_year)
    dt = maturity_years / n_steps
    coupon_steps = list(range(1, n_steps + 1))

    rates, cum_disc = _simulate_one_rate(
        floating_rate, vol, rate, maturity_years,
        n_steps_per_year, n_paths, seed,
    )

    cashflows = np.zeros((n_paths, n_steps))
    for i, s in enumerate(coupon_steps):
        cpn = np.maximum(floor, fixed_rate - leverage * rates[:, s])
        cashflows[:, i] = cpn * dt * 100.0

    cashflows[:, -1] += 100.0

    cashflow_times = np.array(coupon_steps, dtype=float)

    straight_pv = np.zeros(n_paths)
    for i, s in enumerate(coupon_steps):
        straight_pv += cashflows[:, i] * cum_disc[:, s]
    straight_price = float(straight_pv.mean())

    # Regressors: single floating rate as state variable (expanded to 3D)
    regressors = rates[:, :, np.newaxis]  # (n_paths, n_steps+1, 1)

    call_step_indices = [int(round(t / dt)) for t in call_dates_years
                         if 0 < t <= maturity_years + 1e-10]

    is_called, call_step_arr, call_prob = _lsm_call_decision(
        cashflows, cashflow_times, call_step_indices,
        cum_disc, regressors, call_price=100.0, n_basis=5,
    )

    callable_pv = np.zeros(n_paths)
    for p in range(n_paths):
        if is_called[p]:
            s = call_step_arr[p]
            pv = 0.0
            for i, cs in enumerate(coupon_steps):
                if cs <= s:
                    pv += cashflows[p, i] * cum_disc[p, cs]
            pv += 100.0 * cum_disc[p, s]
            callable_pv[p] = pv
        else:
            for i, cs in enumerate(coupon_steps):
                callable_pv[p] += cashflows[p, i] * cum_disc[p, cs]

    price = float(callable_pv.mean())
    expected_cpn = float(np.mean(cashflows[:, :-1] / dt / 100.0))

    return CallableInverseFloaterResult(
        price=price,
        straight_price=straight_price,
        call_option_value=max(0.0, straight_price - price),
        expected_coupon=expected_cpn,
        call_probability=call_prob,
    )
