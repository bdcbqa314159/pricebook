"""Bermudan swaption pricing under the G2++ two-factor model.

Two complementary pricing engines are provided:

* :func:`bermudan_swaption_g2pp_tree` — exact 2D trinomial tree backward
  induction.  At each exercise date the node-level swap PV is computed
  analytically using the G2++ ZCB formula and compared against the
  continuation value; the option holder exercises optimally.

* :func:`bermudan_swaption_g2pp_lsm` — Longstaff-Schwartz (LSM) Monte Carlo
  on exact OU factor paths.  Continuation values are regressed on a
  polynomial basis in (x, y) at each exercise date, and the standard LSM
  backward recursion is applied.

* :func:`g2pp_vs_hw1f_bermudan` — convenience wrapper that prices the same
  Bermudan under both a 1-factor Hull-White model and a 2-factor G2++ model
  and returns the comparison as a dictionary.

References:
    Brigo, D. & Mercurio, F., *Interest Rate Models — Theory and Practice*,
    2nd ed., Springer, 2006, Ch. 4.3 (G2++ Bermudan swaption).
    Andersen, L. & Piterbarg, V., *Interest Rate Modeling*, Vol. II, Atlantic
    Financial Press, 2010 (LSM for Bermudan swaptions).
    Longstaff, F. & Schwartz, E., "Valuing American Options by Simulation",
    *Review of Financial Studies*, 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.vasicek import G2PlusPlus
from pricebook.models.g2pp_tree import G2PPTree, g2pp_european_swaption_tree


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BermudanSwaptionG2PPResult:
    """Result container for Bermudan swaption pricing under G2++."""

    price: float
    european_price: float
    early_exercise_premium: float     # price − european_price
    exercise_probabilities: list[float]   # per exercise date (tree) or empty (LSM)
    n_factors: int                    # always 2 for G2++
    method: str                       # "tree" | "lsm"

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "european_price": self.european_price,
            "early_exercise_premium": self.early_exercise_premium,
            "exercise_probabilities": self.exercise_probabilities,
            "n_factors": self.n_factors,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Helper: swap PV at a tree node
# ---------------------------------------------------------------------------

def _swap_pv_at_node(
    tree: G2PPTree,
    t_idx: int,
    xi: int,
    yi: int,
    pay_times: list[float],
    alpha: float,
    strike: float,
    is_payer: bool,
) -> float:
    """Compute swap PV at a single tree node using analytical ZCB prices."""
    t = tree.times[t_idx]
    annuity = sum(
        alpha * tree.zcb_price(t_idx, xi, yi, tp)
        for tp in pay_times if tp > t
    )
    if annuity < 1e-14:
        return 0.0
    P_end = tree.zcb_price(t_idx, xi, yi, pay_times[-1])
    # P(t,t) = 1 — so the floating leg value is (1 − P(t, T_N)) analytically
    swap_pv = annuity * ((1.0 - P_end) / annuity - strike)
    if not is_payer:
        swap_pv = -swap_pv
    return swap_pv


# ---------------------------------------------------------------------------
# Helper: payment schedule
# ---------------------------------------------------------------------------

def _build_pay_times(
    expiry_years: float,
    swap_end_years: float,
    swap_freq: int,
) -> list[float]:
    coupon_dt = 1.0 / swap_freq
    pay_times: list[float] = []
    t = expiry_years + coupon_dt
    while t <= swap_end_years + 1e-9:
        pay_times.append(round(t, 10))
        t += coupon_dt
    if not pay_times:
        raise ValueError(
            f"No swap payment dates after first exercise ({expiry_years}y); "
            f"check swap_end_years ({swap_end_years}y)."
        )
    return pay_times


# ---------------------------------------------------------------------------
# Tree pricer
# ---------------------------------------------------------------------------

def bermudan_swaption_g2pp_tree(
    g2pp: G2PlusPlus,
    exercise_years: list[float],
    swap_end_years: float,
    strike: float,
    is_payer: bool = True,
    n_steps: int = 50,
    swap_freq: int = 2,
) -> BermudanSwaptionG2PPResult:
    """Bermudan swaption via G2++ 2D trinomial tree.

    The tree is built to the last exercise date.  At each exercise date the
    node-level swap PV is computed analytically (via the G2++ ZCB formula)
    and compared against the discounted continuation value.  The holder
    exercises if and only if exercise_value > continuation.

    Parameters
    ----------
    g2pp:
        Calibrated G2PlusPlus instance.
    exercise_years:
        Sorted list of exercise dates in years (ascending).
    swap_end_years:
        Underlying swap maturity in years.
    strike:
        Fixed rate of the underlying swap.
    is_payer:
        True = payer swaption (holder pays fixed, receives float).
    n_steps:
        Number of time steps in the tree (to the last exercise date).
    swap_freq:
        Swap coupon frequency per year (default 2 = semi-annual).

    Returns
    -------
    BermudanSwaptionG2PPResult
    """
    exercise_years = sorted(exercise_years)
    T_last = exercise_years[-1]
    alpha = 1.0 / swap_freq

    # Payment schedule (full swap, starting from first exercise date)
    pay_times = _build_pay_times(exercise_years[0], swap_end_years, swap_freq)

    tree = G2PPTree(g2pp, T_last, n_steps)
    dt = tree.dt

    # Map exercise dates to tree step indices
    exercise_steps: set[int] = set()
    for ey in exercise_years:
        step = int(round(ey / dt))
        step = max(0, min(n_steps, step))
        exercise_steps.add(step)

    # Track exercise counts for probability estimation
    exercise_count: dict[int, float] = {s: 0.0 for s in exercise_steps}
    total_weight: dict[int, float] = {s: 0.0 for s in exercise_steps}

    # Terminal payoff: at last exercise date, intrinsic value of swaption
    last_step = n_steps
    terminal = np.zeros((tree.n_x, tree.n_y))
    if last_step in exercise_steps:
        for xi in range(tree.n_x):
            for yi in range(tree.n_y):
                sv = _swap_pv_at_node(
                    tree, last_step, xi, yi,
                    pay_times, alpha, strike, is_payer,
                )
                terminal[xi, yi] = max(sv, 0.0)

    # Build option_func that applies Bermudan constraint at exercise steps
    # We capture exercise statistics via a mutable container
    _stats: dict[str, object] = {
        "ex_count": {s: 0.0 for s in exercise_steps},
        "tot_weight": {s: 0.0 for s in exercise_steps},
    }

    def _option_func(t_idx: int, xi: int, yi: int, cont: float) -> float:
        if t_idx not in exercise_steps or t_idx == last_step:
            return cont
        sv = _swap_pv_at_node(
            tree, t_idx, xi, yi,
            pay_times, alpha, strike, is_payer,
        )
        ex_val = max(sv, 0.0)
        if ex_val > cont:
            _stats["ex_count"][t_idx] = _stats["ex_count"].get(t_idx, 0.0) + 1.0
            _stats["tot_weight"][t_idx] = _stats["tot_weight"].get(t_idx, 0.0) + 1.0
            return ex_val
        _stats["tot_weight"][t_idx] = _stats["tot_weight"].get(t_idx, 0.0) + 1.0
        return cont

    bermudan_price = tree.backward_induction(terminal, option_func=_option_func)

    # Exercise probabilities (fraction of nodes that exercise at each date)
    ex_probs = []
    for s in sorted(exercise_steps):
        tw = _stats["tot_weight"].get(s, 0.0)
        ec = _stats["ex_count"].get(s, 0.0)
        ex_probs.append(ec / tw if tw > 0 else 0.0)

    # European price: only exercise at last date
    european_result = g2pp_european_swaption_tree(
        g2pp, T_last, swap_end_years, strike,
        is_payer=is_payer, n_steps=n_steps, swap_freq=swap_freq,
    )
    eur_price = european_result.price

    return BermudanSwaptionG2PPResult(
        price=bermudan_price,
        european_price=eur_price,
        early_exercise_premium=bermudan_price - eur_price,
        exercise_probabilities=ex_probs,
        n_factors=2,
        method="tree",
    )


# ---------------------------------------------------------------------------
# LSM pricer
# ---------------------------------------------------------------------------

def bermudan_swaption_g2pp_lsm(
    g2pp: G2PlusPlus,
    exercise_years: list[float],
    swap_end_years: float,
    strike: float,
    is_payer: bool = True,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int = 42,
    swap_freq: int = 2,
) -> BermudanSwaptionG2PPResult:
    """Bermudan swaption under G2++ via Longstaff-Schwartz Monte Carlo.

    Factor paths are simulated using exact OU transitions (no Euler
    discretisation error).  At each exercise date the swap PV is computed
    analytically from the G2++ ZCB formula for each path.  Continuation
    values are estimated by regressing discounted future cash flows on
    the basis functions {1, x, y, x², y², xy} restricted to in-the-money
    paths.

    Parameters
    ----------
    g2pp:
        Calibrated G2PlusPlus instance.
    exercise_years:
        Sorted list of exercise dates in years (ascending).
    swap_end_years:
        Underlying swap maturity in years.
    strike:
        Fixed rate of the underlying swap.
    is_payer:
        True = payer swaption.
    n_paths:
        Number of Monte Carlo paths (default 50 000).
    n_steps:
        Number of simulation steps between exercise dates (default 100
        steps from t=0 to the last exercise date).
    seed:
        Random seed for reproducibility.
    swap_freq:
        Swap coupon frequency per year.

    Returns
    -------
    BermudanSwaptionG2PPResult
    """
    exercise_years = sorted(exercise_years)
    T_last = exercise_years[-1]
    alpha = 1.0 / swap_freq
    pay_times = _build_pay_times(exercise_years[0], swap_end_years, swap_freq)

    rng = np.random.default_rng(seed)
    a, b_ = g2pp.a, g2pp.b
    s1, s2, rho = g2pp.sigma1, g2pp.sigma2, g2pp.rho

    # Simulate fine-grained paths from 0 to T_last
    dt_sim = T_last / n_steps
    times_sim = np.linspace(0.0, T_last, n_steps + 1)

    # Exact OU step parameters
    e_a = math.exp(-a * dt_sim)
    e_b = math.exp(-b_ * dt_sim)
    var_x = s1**2 / (2.0 * a) * (1.0 - math.exp(-2.0 * a * dt_sim)) if a > 0 \
        else s1**2 * dt_sim
    var_y = s2**2 / (2.0 * b_) * (1.0 - math.exp(-2.0 * b_ * dt_sim)) if b_ > 0 \
        else s2**2 * dt_sim
    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    # Correlated normal increments: shape (n_steps, n_paths, 2)
    Z = rng.standard_normal((n_steps, n_paths, 2))
    Z[:, :, 1] = rho * Z[:, :, 0] + math.sqrt(1.0 - rho**2) * Z[:, :, 1]

    # Simulate paths
    x_paths = np.zeros((n_paths, n_steps + 1))
    y_paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        x_paths[:, i + 1] = x_paths[:, i] * e_a + std_x * Z[i, :, 0]
        y_paths[:, i + 1] = y_paths[:, i] * e_b + std_y * Z[i, :, 1]

    # Map exercise dates to simulation step indices
    ex_step_indices: list[int] = [
        int(round(ey / dt_sim)) for ey in exercise_years
    ]

    # Accumulated discount factors along each path (for discounting back)
    # We discount from each exercise date to t=0 using the path short rates.
    # phi(t) is deterministic and cancels in the swap PV (analytically).

    # Helper: G2++ short rate at a node (x, y, t) = x + y + phi(t).
    # Fix T4-BSWG1: same finite-difference defect as G2PPTree._fwd_rate
    # (T4-G2T1) — eps=1e-5 was destroyed by date_from_year_fraction's day
    # rounding, alternating fwd ≈ 0 / fwd ≈ 137·r across the time grid.
    # Use the curve's day-step instantaneous_forward for stability.
    def _phi(t: float) -> float:
        fwd = g2pp.curve.instantaneous_forward(t)
        ea = (1.0 - math.exp(-a * t)) if a > 1e-12 else t
        eb = (1.0 - math.exp(-b_ * t)) if b_ > 1e-12 else t
        ca = ea**2 / (2.0 * a**2) if a > 1e-12 else t**2 / 2.0
        cb = eb**2 / (2.0 * b_**2) if b_ > 1e-12 else t**2 / 2.0
        cab = ea * eb / (a * b_) if a > 1e-12 and b_ > 1e-12 else t**2
        return fwd + s1**2 * ca + s2**2 * cb + rho * s1 * s2 * cab

    # Compute discount factor from 0 to each exercise step via path integral
    # ∫_0^T r dt ≈ sum_{i} r(t_i) * dt_sim
    log_df_paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        t_i = times_sim[i]
        phi_i = _phi(t_i)
        r_i = x_paths[:, i] + y_paths[:, i] + phi_i
        log_df_paths[:, i + 1] = log_df_paths[:, i] - r_i * dt_sim

    # Analytical ZCB at a path state (x, y) at time t to maturity T_mat
    def _zcb_path(x: np.ndarray, y: np.ndarray, t: float, T_mat: float) -> np.ndarray:
        """Vectorised analytical G2++ ZCB price."""
        tau = T_mat - t
        if tau <= 0.0:
            return np.ones(len(x))
        Bx = (1.0 - math.exp(-a * tau)) / a if a > 0 else tau
        By = (1.0 - math.exp(-b_ * tau)) / b_ if b_ > 0 else tau

        def Bk(k, tt):
            return (1.0 - math.exp(-k * tt)) / k if k > 0 else tt

        def _V(tt):
            ca = (tt - 2.0 * Bk(a, tt) + Bk(2.0 * a, tt)) / a**2 if a > 1e-12 else tt**3 / 3
            cb = (tt - 2.0 * Bk(b_, tt) + Bk(2.0 * b_, tt)) / b_**2 if b_ > 1e-12 else tt**3 / 3
            cab = (tt - Bk(a, tt) - Bk(b_, tt) + Bk(a + b_, tt)) / (a * b_) if a > 1e-12 and b_ > 1e-12 else tt**3 / 3
            return s1**2 * ca + s2**2 * cb + 2.0 * rho * s1 * s2 * cab

        from pricebook.core.day_count import date_from_year_fraction as _dyf
        ref = g2pp.curve.reference_date
        P_T = g2pp.curve.df(_dyf(ref, T_mat))
        P_t = g2pp.curve.df(_dyf(ref, t)) if t > 1e-10 else 1.0
        # Fix T4-G2T2: Brigo-Mercurio eq. 4.10 — exponent is
        # 0.5·[V(t, T) − V(0, T) + V(0, t)], not 0.5·[V(0, T) − V(0, t)].
        half_dV = 0.5 * (_V(tau) - _V(T_mat) + _V(t))
        return (P_T / P_t) * np.exp(-Bx * x - By * y + half_dV)

    def _swap_pv_paths(
        x: np.ndarray, y: np.ndarray, t: float
    ) -> np.ndarray:
        """Swap PV for each path at time t."""
        annuity = sum(
            alpha * _zcb_path(x, y, t, tp)
            for tp in pay_times if tp > t
        )
        if not isinstance(annuity, np.ndarray):
            return np.zeros(len(x))
        P_end = _zcb_path(x, y, t, pay_times[-1])
        swap_pv = annuity * ((1.0 - P_end) / np.where(annuity > 1e-14, annuity, 1.0) - strike)
        if not is_payer:
            swap_pv = -swap_pv
        return swap_pv

    # LSM backward induction
    # cash_flows[path] = PV at t=0 of the optimal exercise payoff
    cash_flows = np.zeros(n_paths)

    # At last exercise date: exercise if ITM
    last_step_idx = ex_step_indices[-1]
    t_last = exercise_years[-1]
    x_last = x_paths[:, last_step_idx]
    y_last = y_paths[:, last_step_idx]
    sv_last = _swap_pv_paths(x_last, y_last, t_last)
    itm_last = sv_last > 0.0
    cash_flows[itm_last] = sv_last[itm_last]

    # Discount exercised cash flows back from last exercise date to t=0
    df_last = np.exp(log_df_paths[:, last_step_idx])

    # Rolling back through earlier exercise dates
    # Hold: discounted cash flow from future exercise
    # At each prior exercise date, regress and compare
    discounted_cf = cash_flows * df_last  # PV at t=0 of last-date exercise

    for k in range(len(exercise_years) - 2, -1, -1):
        ex_step = ex_step_indices[k]
        t_ex = exercise_years[k]
        x_ex = x_paths[:, ex_step]
        y_ex = y_paths[:, ex_step]
        sv_ex = _swap_pv_paths(x_ex, y_ex, t_ex)
        itm_ex = sv_ex > 0.0

        if itm_ex.sum() < 5:
            # Too few ITM paths to regress; skip this date
            continue

        # Continuation value: regress discounted_cf on basis for ITM paths
        x_itm = x_ex[itm_ex]
        y_itm = y_ex[itm_ex]
        cf_itm = discounted_cf[itm_ex]

        # Basis: (1, x, y, x^2, y^2, xy)
        basis = np.column_stack([
            np.ones(x_itm.shape[0]),
            x_itm,
            y_itm,
            x_itm**2,
            y_itm**2,
            x_itm * y_itm,
        ])

        # Discount cf back from t=0 to t_ex for regression
        df_ex = np.exp(log_df_paths[itm_ex, ex_step])
        y_reg = cf_itm / df_ex   # continuation PV at t_ex

        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, y_reg, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Full basis for all ITM paths
        basis_full = np.column_stack([
            np.ones(x_itm.shape[0]),
            x_itm, y_itm,
            x_itm**2, y_itm**2,
            x_itm * y_itm,
        ])
        cont_est = basis_full @ coeffs

        # Exercise where exercise value > estimated continuation
        exercise_now = sv_ex[itm_ex] > cont_est

        # Update cash flows for paths that exercise now
        itm_indices = np.where(itm_ex)[0]
        for idx_local, idx_global in enumerate(itm_indices):
            if exercise_now[idx_local]:
                # Override: exercise at this date
                df_to_now = math.exp(log_df_paths[idx_global, ex_step])
                discounted_cf[idx_global] = sv_ex[idx_global] * df_to_now

    bermudan_price = float(np.mean(discounted_cf))

    # European price at last exercise date for comparison
    european_result = g2pp_european_swaption_tree(
        g2pp, T_last, swap_end_years, strike,
        is_payer=is_payer, n_steps=50, swap_freq=swap_freq,
    )
    eur_price = european_result.price

    return BermudanSwaptionG2PPResult(
        price=bermudan_price,
        european_price=eur_price,
        early_exercise_premium=bermudan_price - eur_price,
        exercise_probabilities=[],   # not tracked in LSM (path-based)
        n_factors=2,
        method="lsm",
    )


# ---------------------------------------------------------------------------
# 1F vs 2F comparison
# ---------------------------------------------------------------------------

def g2pp_vs_hw1f_bermudan(
    curve,
    exercise_years: list[float],
    swap_end_years: float,
    strike: float,
    hw_a: float,
    hw_sigma: float,
    g2pp_params: dict,
    n_steps: int = 50,
) -> dict:
    """Compare 1-factor Hull-White and 2-factor G2++ Bermudan swaption prices.

    The 1F Hull-White model is a special case of G2++ with sigma2 → 0 and
    b equal to some large number.  The difference in prices illustrates the
    2F premium that arises from correlation and two-dimensional smile dynamics.

    Parameters
    ----------
    curve:
        DiscountCurve (shared by both models).
    exercise_years:
        Exercise schedule (list of years, ascending).
    swap_end_years:
        Underlying swap maturity in years.
    strike:
        Swap fixed rate.
    hw_a:
        Hull-White mean-reversion speed for the 1F model.
    hw_sigma:
        Hull-White volatility for the 1F model.
    g2pp_params:
        Dictionary with keys ``a``, ``b``, ``sigma1``, ``sigma2``, ``rho``
        for the G2++ model.
    n_steps:
        Tree time steps used for both models.

    Returns
    -------
    dict with keys:
        hw1f_price, g2pp_price, two_factor_premium,
        g2pp_params, hw_params, exercise_years, swap_end_years, strike.
    """
    from pricebook.models.hull_white import HullWhite
    from pricebook.options.bermudan_swaption import bermudan_swaption_tree as _hw_berm

    # 1F Hull-White Bermudan
    hw = HullWhite(a=hw_a, sigma=hw_sigma, curve=curve)
    hw_price = _hw_berm(
        hw,
        exercise_years=exercise_years,
        swap_end_years=swap_end_years,
        strike=strike,
        n_steps=n_steps,
    )

    # 2F G2++ Bermudan (tree)
    g2pp = G2PlusPlus(
        a=g2pp_params["a"],
        b=g2pp_params["b"],
        sigma1=g2pp_params["sigma1"],
        sigma2=g2pp_params["sigma2"],
        rho=g2pp_params["rho"],
        curve=curve,
    )
    g2pp_result = bermudan_swaption_g2pp_tree(
        g2pp,
        exercise_years=exercise_years,
        swap_end_years=swap_end_years,
        strike=strike,
        n_steps=n_steps,
    )

    return {
        "hw1f_price": hw_price,
        "g2pp_price": g2pp_result.price,
        "two_factor_premium": g2pp_result.price - hw_price,
        "g2pp_european_price": g2pp_result.european_price,
        "g2pp_early_exercise_premium": g2pp_result.early_exercise_premium,
        "g2pp_params": g2pp_params,
        "hw_params": {"a": hw_a, "sigma": hw_sigma},
        "exercise_years": exercise_years,
        "swap_end_years": swap_end_years,
        "strike": strike,
    }
