"""
Vasicek and G2++ short-rate models.

Vasicek: dr = a*(b - r)*dt + sigma*dW  (constant parameters)
    Analytical ZCB: P(t,T) = A(t,T)*exp(-B(t,T)*r)

G2++: r(t) = x(t) + y(t) + phi(t)  (two-factor)
    dx = -a*x*dt + sigma1*dW1
    dy = -b*y*dt + sigma2*dW2
    dW1*dW2 = rho*dt
    phi(t) calibrated to the initial term structure

    from pricebook.vasicek import Vasicek, G2PlusPlus

    v = Vasicek(a=0.5, b=0.05, sigma=0.01)
    price = v.zcb_price(r=0.05, T=5.0)

    g2 = G2PlusPlus(a=0.5, b=0.1, sigma1=0.01, sigma2=0.008, rho=-0.5, curve=ois)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.special_process import OUProcess
from pricebook.brownian import CorrelatedBM


class Vasicek:
    """Vasicek short-rate model with constant parameters.

    dr = a*(b - r)*dt + sigma*dW

    Args:
        a: mean reversion speed.
        b: long-term mean rate.
        sigma: rate volatility.
    """

    def __init__(self, a: float, b: float, sigma: float):
        self.a = a
        self.b = b
        self.sigma = sigma

    def B(self, tau: float) -> float:
        """B(tau) = (1 - exp(-a*tau)) / a."""
        if self.a == 0:
            return tau
        return (1 - math.exp(-self.a * tau)) / self.a

    def A(self, tau: float) -> float:
        """A(tau) for the ZCB formula: ln(A) = (B-tau)*(a^2*b - sigma^2/2)/a^2 - sigma^2*B^2/(4a)."""
        bt = self.B(tau)
        a, b, sig = self.a, self.b, self.sigma
        if a == 0:
            return math.exp(-0.5 * sig**2 * tau**3 / 3)
        log_a = (bt - tau) * (a**2 * b - 0.5 * sig**2) / a**2 \
                - sig**2 * bt**2 / (4 * a)
        return math.exp(log_a)

    def zcb_price(self, r: float, tau: float) -> float:
        """Zero-coupon bond price P(r, tau) = A(tau) * exp(-B(tau)*r)."""
        return self.A(tau) * math.exp(-self.B(tau) * r)

    def yield_curve(self, r: float, taus: list[float]) -> list[float]:
        """Zero rates R(tau) = -ln(P)/tau for a list of maturities."""
        return [-math.log(self.zcb_price(r, t)) / t if t > 0 else r for t in taus]

    def mean(self, r0: float, t: float) -> float:
        """E[r(t)] = b + (r0 - b)*exp(-a*t)."""
        return self.b + (r0 - self.b) * math.exp(-self.a * t)

    def variance(self, t: float) -> float:
        """Var[r(t)] = sigma^2/(2a) * (1 - exp(-2at))."""
        if self.a == 0:
            return self.sigma**2 * t
        return self.sigma**2 / (2 * self.a) * (1 - math.exp(-2 * self.a * t))

    def simulate(
        self, r0: float, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate rate paths via exact OU simulation. Shape: (n_paths, n_steps+1)."""
        ou = OUProcess(a=self.a, mu=self.b, sigma=self.sigma, seed=seed)
        return ou.sample(x0=r0, T=T, n_steps=n_steps, n_paths=n_paths)

    def caplet_price(
        self, r: float, strike: float, T_option: float, T_pay: float,
        notional: float = 1.0,
    ) -> float:
        """Analytical Vasicek caplet price (Jamshidian decomposition).

        Caplet on [T_option, T_pay] with strike K.
        """
        tau_bond = T_pay - T_option
        P_T = self.zcb_price(r, T_option)
        P_S = self.zcb_price(r, T_pay)

        sigma_p = self.sigma * self.B(tau_bond) * \
            math.sqrt((1 - math.exp(-2 * self.a * T_option)) / (2 * self.a)) \
            if self.a > 0 else self.sigma * tau_bond * math.sqrt(T_option)

        K_bond = 1.0 / (1.0 + strike * tau_bond)

        from pricebook.black76 import _norm_cdf
        if sigma_p <= 0:
            return max(P_S - K_bond * P_T, 0.0) * notional * (1 + strike * tau_bond)

        d1 = math.log(P_S / (K_bond * P_T)) / sigma_p + 0.5 * sigma_p
        d2 = d1 - sigma_p

        return notional * (1 + strike * tau_bond) * \
            (P_S * _norm_cdf(d1) - K_bond * P_T * _norm_cdf(d2))


class G2PlusPlus:
    """Two-factor Hull-White (G2++) model.

    r(t) = x(t) + y(t) + phi(t)
    dx = -a*x*dt + sigma1*dW1
    dy = -b*y*dt + sigma2*dW2
    dW1*dW2 = rho*dt

    phi(t) calibrated so that model discount factors match the input curve.
    """

    def __init__(
        self,
        a: float,
        b: float,
        sigma1: float,
        sigma2: float,
        rho: float,
        curve: DiscountCurve,
    ):
        self.a = a
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.curve = curve

    def _V(self, T: float) -> float:
        """Variance of the integral of the short rate from 0 to T."""
        a, b_ = self.a, self.b
        s1, s2, rho = self.sigma1, self.sigma2, self.rho

        def B(k, t):
            if k == 0:
                return t
            return (1 - math.exp(-k * t)) / k

        Ba = B(a, T)
        Bb = B(b_, T)

        v = (s1**2 / a**2) * (T - 2 * Ba + B(2 * a, T)) + \
            (s2**2 / b_**2) * (T - 2 * Bb + B(2 * b_, T)) + \
            2 * rho * s1 * s2 / (a * b_) * (T - Ba - Bb + B(a + b_, T))

        return v

    def zcb_price(self, x: float, y: float, T: float) -> float:
        """Analytical ZCB: P(x, y, T) = P_market(T) * exp(-Bx*x - By*y + 0.5*V(T))."""
        ref = self.curve.reference_date
        d_T = date_from_year_fraction(ref, T)
        P_mkt = self.curve.df(d_T)

        Bx = (1 - math.exp(-self.a * T)) / self.a if self.a > 0 else T
        By = (1 - math.exp(-self.b * T)) / self.b if self.b > 0 else T
        V = self._V(T)

        return P_mkt * math.exp(-Bx * x - By * y + 0.5 * V)

    def simulate(
        self, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate (x, y) paths. Returns (x_paths, y_paths), each (n_paths, n_steps+1)."""
        cbm = CorrelatedBM(
            [[1, self.rho], [self.rho, 1]], seed=seed,
        )
        dW = cbm.increments(T, n_steps, n_paths)
        dt = T / n_steps

        x = np.zeros((n_paths, n_steps + 1))
        y = np.zeros((n_paths, n_steps + 1))

        e_a = math.exp(-self.a * dt)
        e_b = math.exp(-self.b * dt)
        std_x = self.sigma1 * math.sqrt((1 - math.exp(-2 * self.a * dt)) / (2 * self.a)) \
            if self.a > 0 else self.sigma1 * math.sqrt(dt)
        std_y = self.sigma2 * math.sqrt((1 - math.exp(-2 * self.b * dt)) / (2 * self.b)) \
            if self.b > 0 else self.sigma2 * math.sqrt(dt)

        sqrt_dt = math.sqrt(dt)
        scale_x = std_x / sqrt_dt
        scale_y = std_y / sqrt_dt

        for i in range(n_steps):
            # Exact OU simulation for each factor
            x[:, i + 1] = x[:, i] * e_a + scale_x * dW[:, i, 0]
            y[:, i + 1] = y[:, i] * e_b + scale_y * dW[:, i, 1]

        return x, y

    def discount_factor_mc(
        self, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """MC discount factor to T: E[exp(-∫r ds)]."""
        x, y = self.simulate(T, n_steps, n_paths, seed)
        dt = T / n_steps

        # phi(t) from curve: f(0,t) + V terms
        # Simplified: ∫r ds ≈ sum((x+y)*dt) + ∫phi ds
        # ∫phi ds is deterministic = -ln(P_mkt(T)) + 0.5*V(T) approximately

        integral_xy = (x[:, :-1] + y[:, :-1]).sum(axis=1) * dt

        ref = self.curve.reference_date
        d_T = date_from_year_fraction(ref, T)
        log_P = math.log(self.curve.df(d_T))
        V = self._V(T)

        return np.exp(log_P + 0.5 * V - integral_xy)
