"""
Heath-Jarrow-Morton framework.

Models the entire instantaneous forward rate curve f(t,T) directly:
    df(t,T) = alpha(t,T)*dt + sigma(t,T)*dW(t)

No-drift condition (risk-neutral): alpha(t,T) = sigma(t,T) * ∫_t^T sigma(t,s) ds

Musiela parameterisation: f(t,x) where x = T-t (time to maturity).

    from pricebook.hjm import HJMModel, hjm_simulate

    hjm = HJMModel(initial_forwards, tenors, vol_func=lambda t, x: 0.01)
    paths = hjm.simulate(T=5.0, n_steps=50, n_paths=10000)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction
from datetime import date


class HJMModel:
    """HJM forward rate model in Musiela parameterisation.

    f(t, x) = forward rate at time t for maturity t+x.

    Args:
        initial_forwards: initial forward curve f(0, x) at each tenor.
        tenors: time-to-maturity grid (x values, in years).
        vol_func: volatility function sigma(t, x) → float.
            Default: constant vol.
    """

    def __init__(
        self,
        initial_forwards: list[float] | np.ndarray,
        tenors: list[float] | np.ndarray,
        vol_func=None,
        constant_vol: float = 0.01,
    ):
        self.f0 = np.asarray(initial_forwards, dtype=float)
        self.tenors = np.asarray(tenors, dtype=float)
        self.n_tenors = len(self.tenors)
        self._constant_vol = constant_vol

        if vol_func is not None:
            self._vol = vol_func
        else:
            self._vol = lambda t, x: constant_vol

    @classmethod
    def from_curve(
        cls,
        curve: DiscountCurve,
        tenors: list[float] | None = None,
        constant_vol: float = 0.01,
    ) -> "HJMModel":
        """Build HJM from a discount curve by extracting forward rates."""
        ref = curve.reference_date
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]

        forwards = []
        for x in tenors:
            d1 = date.fromordinal(ref.toordinal() + int(x * 365))
            d2 = date.fromordinal(ref.toordinal() + int((x + 0.01) * 365))
            # Instantaneous forward ≈ -d/dT ln(P(T))
            df1 = curve.df(d1)
            df2 = curve.df(d2)
            dt = 0.01
            if df2 > 0 and df1 > 0:
                f = -math.log(df2 / df1) / dt
            else:
                f = 0.0
            forwards.append(f)

        return cls(forwards, tenors, constant_vol=constant_vol)

    def _drift(self, t: float, sigma_vals: np.ndarray) -> np.ndarray:
        """HJM no-drift condition: alpha(t,x) = sigma(t,x) * ∫_0^x sigma(t,s) ds.

        Discretised: alpha[j] = sigma[j] * sum(sigma[0:j+1] * dx).
        """
        dx = np.diff(self.tenors, prepend=0.0)
        integral = np.cumsum(sigma_vals * dx)
        return sigma_vals * integral

    def simulate(
        self, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate forward curve paths.

        Returns:
            Array of shape (n_paths, n_steps + 1, n_tenors).
            paths[p, i, j] = f(t_i, x_j) for path p.
        """
        dt = T / n_steps
        rng = np.random.default_rng(seed)
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1, self.n_tenors))
        paths[:, 0, :] = self.f0

        for i in range(n_steps):
            t = i * dt
            dW = sqrt_dt * rng.standard_normal((n_paths, 1))

            # Vol at each tenor
            sigma_vals = np.array([self._vol(t, x) for x in self.tenors])

            # Drift from no-arbitrage condition
            drift = self._drift(t, sigma_vals)

            # Musiela drift: d/dx f contribution (finite difference)
            f_curr = paths[:, i, :]
            dfdx = np.zeros_like(f_curr)
            if self.n_tenors > 1:
                dx = self.tenors[1] - self.tenors[0]
                dfdx[:, :-1] = (f_curr[:, 1:] - f_curr[:, :-1]) / dx
                dfdx[:, -1] = dfdx[:, -2]  # flat extrapolation

            paths[:, i + 1, :] = (
                f_curr + (drift + dfdx) * dt + sigma_vals * dW
            )

        return paths

    def discount_factors(self, paths: np.ndarray, dt: float) -> np.ndarray:
        """Compute discount factors from simulated forward curves.

        P(0, T_i) ≈ exp(-sum(f(t_j, 0) * dt) for j=0..i)

        Returns: shape (n_paths, n_steps + 1). df[p, i] = P_path(0, t_i).
        """
        # Use the short rate: f(t, 0) ≈ paths[:, :, 0] (shortest tenor)
        short_rate = paths[:, :, 0]
        cumulative = np.cumsum(short_rate[:, :-1] * dt, axis=1)
        df = np.ones((paths.shape[0], paths.shape[1]))
        df[:, 1:] = np.exp(-cumulative)
        return df

    def zcb_prices(self, paths: np.ndarray, dt: float) -> np.ndarray:
        """Average ZCB prices across paths (should match initial curve)."""
        df = self.discount_factors(paths, dt)
        return df.mean(axis=0)
