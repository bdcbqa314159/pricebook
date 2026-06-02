"""Black-Derman-Toy (BDT) log-normal short rate tree.

Log-normal rates for markets where Gaussian HW is inappropriate
(high-rate EM currencies, historical periods with high rates).

* :class:`BDTTree` — calibrated BDT trinomial tree.
* :func:`bdt_callable_bond` — callable bond via BDT.
* :func:`bdt_bermudan_swaption` — Bermudan swaption via BDT.

References:
    Black, Derman & Toy, *A One-Factor Model of Interest Rates and
    Its Application to Treasury Bond Options*, Financial Analysts
    Journal, 1990.
    Tuckman & Serrat, *Fixed Income Securities*, 3rd ed., Ch. 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class BDTResult:
    """BDT tree pricing result."""
    price: float
    n_steps: int
    short_rates: np.ndarray     # calibrated rates at step 0
    vol_term: list[float]       # calibrated vol at each step

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "n_steps": self.n_steps,
            "mean_short_rate": float(np.mean(self.short_rates)),
        }


class BDTTree:
    """Black-Derman-Toy recombining binomial tree.

    d(ln r) = [θ(t) − a(t) ln r] dt + σ(t) dW

    At each step, the tree has i+1 nodes (j = 0..i).
    Rate at node (i, j): r(i, j) = a_i × exp(2 × j × σ_i × √dt).

    Calibration:
    - a_i chosen to match market discount factor at step i+1.
    - σ_i from market vol term structure (or flat).

    Args:
        curve: market discount curve.
        vol_term: volatility at each step (or flat vol).
        n_steps: number of tree steps.
    """

    def __init__(
        self,
        curve: DiscountCurve,
        vol_term: list[float] | float = 0.10,
        n_steps: int = 50,
    ):
        self.curve = curve
        self.n_steps = n_steps
        self.dt = 1.0  # annual steps (simplification; can refine)

        if isinstance(vol_term, (int, float)):
            self.vol_term = [float(vol_term)] * n_steps
        else:
            self.vol_term = list(vol_term)
            while len(self.vol_term) < n_steps:
                self.vol_term.append(self.vol_term[-1])

        # Calibrate
        self._a = np.zeros(n_steps)     # median rate at each step
        self._calibrate()

    def _calibrate(self):
        """Calibrate a_i to match market discount factors."""
        ref = self.curve.reference_date
        from datetime import timedelta

        # State prices Q[j] = Arrow-Debreu price at node j
        Q = np.array([1.0])  # initial

        for i in range(self.n_steps):
            n_nodes = i + 1
            sigma_i = self.vol_term[i]
            sqrt_dt = math.sqrt(self.dt)

            # Target: sum of Q × df should match market DF
            t_next = (i + 1) * self.dt
            target_date = ref + timedelta(days=round(t_next * 365.25))
            try:
                target_df = self.curve.df(target_date)
            except Exception:
                target_df = math.exp(-0.04 * t_next)

            # Bisection for a_i
            def _price_at_a(a_val):
                total = 0.0
                for j in range(n_nodes):
                    r_j = a_val * math.exp(2 * j * sigma_i * sqrt_dt)
                    df_j = 1.0 / (1.0 + r_j * self.dt)
                    total += Q[j] * df_j
                return total

            # Bisection
            lo, hi = 1e-6, 2.0
            for _ in range(100):
                mid = (lo + hi) / 2
                if _price_at_a(mid) > target_df:
                    lo = mid
                else:
                    hi = mid
            self._a[i] = (lo + hi) / 2

            # Evolve state prices
            Q_new = np.zeros(n_nodes + 1)
            a_i = self._a[i]
            for j in range(n_nodes):
                r_j = a_i * math.exp(2 * j * sigma_i * sqrt_dt)
                df_j = 1.0 / (1.0 + r_j * self.dt)
                Q_new[j] += 0.5 * Q[j] * df_j
                Q_new[j + 1] += 0.5 * Q[j] * df_j
            Q = Q_new

    def rate(self, step: int, node: int) -> float:
        """Short rate at tree node (step, node)."""
        sigma = self.vol_term[step]
        sqrt_dt = math.sqrt(self.dt)
        return self._a[step] * math.exp(2 * node * sigma * sqrt_dt)

    def zcb_price(self, maturity_steps: int) -> float:
        """Price a zero-coupon bond via the tree."""
        n = min(maturity_steps, self.n_steps)

        # Terminal values: 1.0 at maturity
        V = np.ones(n + 1)

        for i in range(n - 1, -1, -1):
            n_nodes = i + 1
            V_new = np.zeros(n_nodes)
            for j in range(n_nodes):
                r_j = self.rate(i, j)
                df_j = 1.0 / (1.0 + r_j * self.dt)
                V_new[j] = 0.5 * df_j * (V[j] + V[j + 1])
            V = V_new

        return float(V[0])

    def price_cashflows(
        self,
        cashflows: list[tuple[int, float]],
        option_steps: set[int] | None = None,
        option_func: str = "none",
        option_value: float = 100.0,
    ) -> float:
        """Price cashflows via backward induction.

        Args:
            cashflows: list of (step, amount).
            option_steps: steps where option is exercisable.
            option_func: "call" (min), "put" (max), or "none".
            option_value: strike for call/put.
        """
        cf_map = {}
        for step, amount in cashflows:
            cf_map[step] = cf_map.get(step, 0) + amount

        n = self.n_steps
        V = np.full(n + 1, cf_map.get(n, 0.0))

        for i in range(n - 1, -1, -1):
            n_nodes = i + 1
            V_new = np.zeros(n_nodes)
            cf = cf_map.get(i, 0.0)

            for j in range(n_nodes):
                r_j = self.rate(i, j)
                df_j = 1.0 / (1.0 + r_j * self.dt)
                cont = 0.5 * df_j * (V[j] + V[j + 1]) + cf

                if option_steps and i in option_steps:
                    if option_func == "call":
                        cont = min(cont, option_value)
                    elif option_func == "put":
                        cont = max(cont, option_value)

                V_new[j] = cont
            V = V_new

        return float(V[0])


def bdt_callable_bond(
    curve: DiscountCurve,
    coupon: float,
    maturity_years: int,
    call_dates_years: list[int],
    call_price: float = 100.0,
    face: float = 100.0,
    vol: float = 0.10,
    n_steps: int | None = None,
) -> dict:
    """Price a callable bond via BDT tree.

    Args:
        coupon: annual coupon rate.
        maturity_years: bond maturity in years.
        call_dates_years: years at which issuer can call.
        call_price: call price (typically par).
        vol: short rate vol for BDT.
    """
    steps = n_steps or maturity_years
    tree = BDTTree(curve, vol, steps)

    # Cashflows: coupons + principal
    cashflows = []
    for y in range(1, maturity_years + 1):
        step = min(y, steps)
        cashflows.append((step, face * coupon))
    cashflows.append((min(maturity_years, steps), face))

    # Call option
    call_steps = {min(y, steps) for y in call_dates_years}

    callable_price = tree.price_cashflows(cashflows, call_steps, "call", call_price)

    # Straight bond (no call)
    straight_price = tree.price_cashflows(cashflows)

    return {
        "callable_price": callable_price,
        "straight_price": straight_price,
        "call_option_value": straight_price - callable_price,
        "n_steps": steps,
    }


def bdt_bermudan_swaption(
    curve: DiscountCurve,
    swap_rate: float,
    maturity_years: int,
    exercise_years: list[int],
    notional: float = 100.0,
    vol: float = 0.10,
    is_payer: bool = True,
) -> dict:
    """Price a Bermudan swaption via BDT tree."""
    tree = BDTTree(curve, vol, maturity_years)
    n = maturity_years

    exercise_set = {min(y, n) for y in exercise_years}

    # Backward induction
    V = np.zeros(n + 1)

    for i in range(n - 1, -1, -1):
        n_nodes = i + 1
        V_new = np.zeros(n_nodes)

        for j in range(n_nodes):
            r_j = tree.rate(i, j)
            df_j = 1.0 / (1.0 + r_j * tree.dt)
            cont = 0.5 * df_j * (V[j] + V[j + 1])

            if i in exercise_set:
                # Swap value: PV of (fixed - floating) annuity
                remaining = n - i
                annuity = sum(tree.zcb_price(k) for k in range(1, remaining + 1))
                swap_pv = notional * (swap_rate - r_j) * annuity / remaining
                if not is_payer:
                    swap_pv = -swap_pv
                exercise_val = max(swap_pv, 0)
                V_new[j] = max(cont, exercise_val)
            else:
                V_new[j] = cont

        V = V_new

    return {
        "price": float(V[0]),
        "n_exercise_dates": len(exercise_years),
        "maturity_years": maturity_years,
    }
