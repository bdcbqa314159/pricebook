"""
Hull-White short-rate model and trinomial rate tree.

Model: dr = (theta(t) - a*r) dt + sigma dW

theta(t) is calibrated to match the initial term structure so that
model-implied discount factors equal the market curve.

Analytical zero-coupon bond price under Hull-White:
    P(t, T) = A(t,T) * exp(-B(t,T) * r(t))
    B(t,T) = (1 - exp(-a*(T-t))) / a
    A(t,T) from forward rate matching

Rate tree builds a trinomial tree for r, prices instruments via
backward induction.

    hw = HullWhite(a=0.1, sigma=0.01, curve=discount_curve)
    price = hw.zcb_price(t=0, T=5, r0=0.05)
    tree_price = hw.tree_zcb(T=5, n_steps=100)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


class HullWhite:
    """Hull-White one-factor short-rate model.

    Args:
        a: mean reversion speed.
        sigma: short-rate volatility.
        curve: initial discount curve (for calibrating theta(t)).
    """

    def __init__(self, a: float, sigma: float, curve: DiscountCurve):
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.a = a
        self.sigma = sigma
        self.curve = curve

    def B(self, t: float, T: float) -> float:
        """B(t, T) = (1 - exp(-a*(T-t))) / a."""
        tau = T - t
        if tau <= 0:
            return 0.0
        return (1.0 - math.exp(-self.a * tau)) / self.a

    def _forward_rate(self, t: float) -> float:
        """Instantaneous forward rate from the curve at time t."""
        ref = self.curve.reference_date
        dt_days = 1
        d1 = date.fromordinal(ref.toordinal() + int(t * 365))
        d2 = date.fromordinal(ref.toordinal() + int(t * 365) + dt_days)
        df1 = self.curve.df(d1)
        df2 = self.curve.df(d2)
        dt = dt_days / 365.0
        if df2 <= 0 or df1 <= 0:
            return 0.0
        return -math.log(df2 / df1) / dt

    def _log_A(self, t: float, T: float) -> float:
        """log A(t, T) for the analytical bond price."""
        ref = self.curve.reference_date
        d_t = date.fromordinal(ref.toordinal() + int(t * 365))
        d_T = date.fromordinal(ref.toordinal() + int(T * 365))

        df_t = self.curve.df(d_t)
        df_T = self.curve.df(d_T)

        if df_t <= 0 or df_T <= 0:
            return 0.0

        b = self.B(t, T)
        f_t = self._forward_rate(t)

        log_a = math.log(df_T / df_t) + b * f_t - \
            (self.sigma**2 / (4.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * t)) * b**2

        return log_a

    def zcb_price(self, t: float, T: float, r: float) -> float:
        """Analytical zero-coupon bond price P(t, T) given short rate r at time t.

        P(t, T) = A(t,T) * exp(-B(t,T) * r)
        """
        b = self.B(t, T)
        log_a = self._log_A(t, T)
        return math.exp(log_a - b * r)

    def tree_zcb(self, T: float, n_steps: int = 100) -> float:
        """Price a zero-coupon bond maturing at T using a trinomial rate tree.

        The tree is calibrated to match the initial discount curve.
        """
        dt = T / n_steps
        a, sigma = self.a, self.sigma

        # Tree geometry
        dr = sigma * math.sqrt(3.0 * dt)
        j_max = int(math.ceil(0.1835 / (a * dt)))  # standard branching limit

        # Forward rates for theta calibration
        n_nodes = 2 * j_max + 1

        # Build the tree step by step
        # State prices Q[j] at each time step
        Q = np.zeros(n_nodes)
        mid = j_max
        Q[mid] = 1.0  # initial state price at r0

        # r at each node: r_j = r0 + j * dr, shifted by alpha(t)
        # alpha(t) is calibrated so that sum_j Q[j] * exp(-r_j * dt) = df(t+dt) / df(t)
        ref = self.curve.reference_date

        r0 = self._forward_rate(0.0)

        for step in range(n_steps):
            t_now = step * dt
            t_next = (step + 1) * dt

            d_next = date.fromordinal(ref.toordinal() + int(t_next * 365))
            target_df = self.curve.df(d_next)  # absolute DF to t+dt

            # Determine alpha so that sum(Q_new) = target_df
            # sum(Q_new) = exp(-alpha*dt) * sum_j Q[j]*exp(-j*dr*dt) = target_df
            sum_q = 0.0
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                if Q[idx] > 0:
                    sum_q += Q[idx] * math.exp(-j * dr * dt)

            if sum_q > 0 and target_df > 0:
                alpha = -math.log(target_df / sum_q) / dt
            else:
                alpha = r0

            # Transition probabilities for each node
            Q_new = np.zeros(n_nodes)
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                if Q[idx] <= 1e-20:
                    continue

                r_j = alpha + j * dr
                discount = math.exp(-r_j * dt)

                # Branching: determine if we branch up/mid/down normally
                # or need to trim at boundaries
                eta = a * j * dr * dt / dr  # mean drift in units of dr
                k = j  # central node destination

                if k > j_max - 1:
                    k = j_max - 1
                elif k < -j_max + 1:
                    k = -j_max + 1

                # Probabilities (standard trinomial)
                p_u = 1.0 / 6.0 + (j**2 * a**2 * dt**2 - j * a * dt) / 2.0
                p_m = 2.0 / 3.0 - j**2 * a**2 * dt**2
                p_d = 1.0 / 6.0 + (j**2 * a**2 * dt**2 + j * a * dt) / 2.0

                # Clamp probabilities
                p_u = max(0, min(1, p_u))
                p_d = max(0, min(1, p_d))
                p_m = max(0, 1.0 - p_u - p_d)

                contrib = Q[idx] * discount

                k_u = min(k + 1, j_max)
                k_d = max(k - 1, -j_max)

                Q_new[k_u + mid] += contrib * p_u
                Q_new[k + mid] += contrib * p_m
                Q_new[k_d + mid] += contrib * p_d

            Q = Q_new

        # Bond price = sum of state prices at maturity
        return float(Q.sum())

    def tree_european_swaption(
        self,
        expiry_T: float,
        swap_end_T: float,
        strike: float,
        n_steps: int = 100,
        is_payer: bool = True,
    ) -> float:
        """Price a European swaption on the Hull-White tree.

        Simplified: at expiry, compute swap value using analytical bond prices,
        then take max(swap_value, 0) weighted by state prices.
        """
        dt = expiry_T / n_steps
        a, sigma = self.a, self.sigma
        dr = sigma * math.sqrt(3.0 * dt)
        j_max = int(math.ceil(0.1835 / (a * dt)))
        n_nodes = 2 * j_max + 1
        mid = j_max

        Q = np.zeros(n_nodes)
        Q[mid] = 1.0
        ref = self.curve.reference_date
        r0 = self._forward_rate(0.0)

        # Evolve state prices to expiry
        for step in range(n_steps):
            t_now = step * dt
            d_next = date.fromordinal(ref.toordinal() + int((step + 1) * dt * 365))
            target_df = self.curve.df(d_next)

            sum_q = 0.0
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                if Q[idx] > 0:
                    sum_q += Q[idx] * math.exp(-j * dr * dt)

            alpha = -math.log(target_df / max(sum_q, 1e-20)) / dt if target_df > 0 else r0

            Q_new = np.zeros(n_nodes)
            for j in range(-j_max, j_max + 1):
                idx = j + mid
                if Q[idx] <= 1e-20:
                    continue

                r_j = alpha + j * dr
                discount = math.exp(-r_j * dt)
                k = j

                if k > j_max - 1:
                    k = j_max - 1
                elif k < -j_max + 1:
                    k = -j_max + 1

                p_u = max(0, min(1, 1.0/6 + (j**2*a**2*dt**2 - j*a*dt)/2))
                p_d = max(0, min(1, 1.0/6 + (j**2*a**2*dt**2 + j*a*dt)/2))
                p_m = max(0, 1.0 - p_u - p_d)

                contrib = Q[idx] * discount
                Q_new[min(k+1, j_max) + mid] += contrib * p_u
                Q_new[k + mid] += contrib * p_m
                Q_new[max(k-1, -j_max) + mid] += contrib * p_d

            Q = Q_new

        # At expiry: for each node, compute swap value using analytical bond prices
        # Simplified: swap_pv ≈ 1 - P(T_expiry, T_end) - strike * annuity
        # where annuity = sum of year_frac * P(T_expiry, T_i)
        total = 0.0
        for j in range(-j_max, j_max + 1):
            idx = j + mid
            if Q[idx] <= 1e-20:
                continue

            # Get the last alpha (approximate r at this node)
            # This is simplified — a full implementation tracks alpha per step
            r_j = r0 + j * dr  # approximate

            # Bond prices from this node
            p_end = self.zcb_price(expiry_T, swap_end_T, r_j)
            # Simple annuity: annual payments
            n_payments = max(1, int(swap_end_T - expiry_T))
            annuity = 0.0
            for k in range(1, n_payments + 1):
                t_pay = expiry_T + k
                if t_pay <= swap_end_T:
                    annuity += self.zcb_price(expiry_T, t_pay, r_j)

            swap_pv = (1.0 - p_end) - strike * annuity
            if not is_payer:
                swap_pv = -swap_pv

            payoff = max(swap_pv, 0.0)
            total += Q[idx] * payoff

        return total
