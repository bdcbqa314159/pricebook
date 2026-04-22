"""Multi-factor HJM and LIBOR Market Model (LMM/BGM).

Multi-factor HJM: multiple independent Brownian drivers with separate
volatility functions (level, slope, curvature).

LMM: forward LIBOR rates as state variables, lognormal dynamics,
calibration via Rebonato approximation.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import black76_price, OptionType


# ---------------------------------------------------------------------------
# Multi-factor HJM
# ---------------------------------------------------------------------------


class MultiFactorHJM:
    """Multi-factor HJM in Musiela parameterisation.

    Each factor has its own volatility function sigma_k(t, x).
    The no-drift condition is enforced per factor.

    Args:
        initial_forwards: f(0, x) at each tenor.
        tenors: time-to-maturity grid.
        vol_funcs: list of volatility functions sigma_k(t, x) → float.
    """

    def __init__(
        self,
        initial_forwards: list[float] | np.ndarray,
        tenors: list[float] | np.ndarray,
        vol_funcs: list = None,
    ):
        self.f0 = np.asarray(initial_forwards, dtype=float)
        self.tenors = np.asarray(tenors, dtype=float)
        self.n_tenors = len(self.tenors)

        if vol_funcs is None:
            # Default: 2-factor (level + slope)
            vol_funcs = [
                lambda t, x: 0.01,                  # level
                lambda t, x: 0.005 * math.exp(-x),  # slope (decaying)
            ]
        self.vol_funcs = vol_funcs
        self.n_factors = len(vol_funcs)

    def _drift_factor(self, sigma_vals: np.ndarray) -> np.ndarray:
        """HJM drift for one factor: sigma(x) * integral_0^x sigma(s) ds."""
        dx = np.diff(self.tenors, prepend=0.0)
        integral = np.cumsum(sigma_vals * dx)
        return sigma_vals * integral

    def simulate(
        self, T: float, n_steps: int, n_paths: int, seed: int = 42,
    ) -> np.ndarray:
        """Simulate multi-factor HJM forward curve paths.

        Returns:
            Array of shape (n_paths, n_steps+1, n_tenors).
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        rng = np.random.default_rng(seed)

        paths = np.zeros((n_paths, n_steps + 1, self.n_tenors))
        paths[:, 0, :] = self.f0

        for i in range(n_steps):
            t = i * dt
            f_curr = paths[:, i, :]

            # Musiela drift: df/dx
            dfdx = np.zeros_like(f_curr)
            if self.n_tenors > 1:
                dx = self.tenors[1] - self.tenors[0]
                dfdx[:, :-1] = (f_curr[:, 1:] - f_curr[:, :-1]) / dx
                dfdx[:, -1] = dfdx[:, -2]

            total_drift = dfdx.copy()
            total_diffusion = np.zeros_like(f_curr)

            for k in range(self.n_factors):
                sigma_k = np.array([self.vol_funcs[k](t, x) for x in self.tenors])
                drift_k = self._drift_factor(sigma_k)
                dW_k = sqrt_dt * rng.standard_normal((n_paths, 1))

                total_drift += drift_k
                total_diffusion += sigma_k * dW_k

            paths[:, i + 1, :] = f_curr + total_drift * dt + total_diffusion

        return paths

    def discount_factors(self, paths: np.ndarray, dt: float) -> np.ndarray:
        """Discount factors from short rate f(t, 0)."""
        short_rate = paths[:, :, 0]
        cumulative = np.cumsum(short_rate[:, :-1] * dt, axis=1)
        df = np.ones((paths.shape[0], paths.shape[1]))
        df[:, 1:] = np.exp(-cumulative)
        return df


# ---------------------------------------------------------------------------
# LIBOR Market Model (BGM)
# ---------------------------------------------------------------------------


class LMM:
    """LIBOR Market Model (Brace-Gatarek-Musiela).

    Forward LIBOR rates L_i(t) follow lognormal dynamics:
        dL_i / L_i = mu_i * dt + sigma_i * dW_i

    where mu_i is determined by the drift restriction (risk-neutral measure).

    Args:
        initial_rates: L_i(0) for each tenor.
        tenors: fixing times T_0, T_1, ..., T_n.
        vols: instantaneous vols sigma_i for each forward rate.
        tau: accrual period (default 0.25 for quarterly).
    """

    def __init__(
        self,
        initial_rates: list[float] | np.ndarray,
        tenors: list[float] | np.ndarray,
        vols: list[float] | np.ndarray,
        tau: float = 0.25,
    ):
        self.L0 = np.asarray(initial_rates, dtype=float)
        self.tenors = np.asarray(tenors, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        self.tau = tau
        self.n_rates = len(self.L0)

    def _drift(self, L: np.ndarray, j: int, numeraire_idx: int = 0) -> np.ndarray:
        """Risk-neutral drift for L_j under spot measure (vectorised).

        Under the spot measure, the drift for L_j involves forwards
        from k=numeraire_idx+1 to j (not expired forwards).

        μ_j = Σ_{k=numeraire_idx+1}^{j} (τ × σ_j × σ_k × L_k) / (1 + τ × L_k)
        """
        start = numeraire_idx + 1
        if start > j:
            return np.zeros(L.shape[0])
        L_slice = L[:, start:j + 1]
        vol_slice = self.vols[start:j + 1]
        numerator = self.tau * self.vols[j] * vol_slice * L_slice
        denominator = 1.0 + self.tau * L_slice
        return (numerator / denominator).sum(axis=1)

    def simulate(
        self, n_steps_per_period: int = 10, n_paths: int = 10_000, seed: int = 42,
    ) -> np.ndarray:
        """Simulate LMM paths.

        Returns:
            Array of shape (n_paths, n_rates) with terminal forward rates L_i(T_i).
        """
        rng = np.random.default_rng(seed)

        L = np.tile(self.L0, (n_paths, 1))  # (n_paths, n_rates)

        for period in range(self.n_rates):
            T_start = self.tenors[period] if period > 0 else 0.0
            T_end = self.tenors[period]
            dt = self.tau / n_steps_per_period

            for step in range(n_steps_per_period):
                dW = math.sqrt(dt) * rng.standard_normal((n_paths, self.n_rates))

                for j in range(period, self.n_rates):
                    drift = self._drift(L, j, numeraire_idx=period)
                    L[:, j] = L[:, j] * np.exp(
                        (drift - 0.5 * self.vols[j]**2) * dt
                        + self.vols[j] * dW[:, j]
                    )
                    L[:, j] = np.maximum(L[:, j], 0.0)

        return L

    def caplet_price(
        self,
        fixing_idx: int,
        strike: float,
        df: float,
        n_paths: int = 50_000,
        seed: int = 42,
    ) -> float:
        """Price a caplet on L_{fixing_idx} via MC."""
        L_terminal = self.simulate(n_paths=n_paths, seed=seed)
        payoff = np.maximum(L_terminal[:, fixing_idx] - strike, 0.0) * self.tau
        return float(df * payoff.mean())

    @staticmethod
    def rebonato_swaption_vol(
        vols: np.ndarray,
        L0: np.ndarray,
        tau: float,
        T_expiry: float,
    ) -> float:
        """Rebonato approximation for ATM swaption vol.

        σ_swap² × T ≈ Σ_{i,j} w_i × w_j × σ_i × σ_j × ρ_{ij} × T

        Uses annuity weights w_i = τ × P(0,T_{i+1}) × L_i(0) / (A × S)
        where A = annuity, S = swap rate, P = discount factors.

        With unit correlation (ρ=1): σ² × T = (Σ w_i × σ_i)² × T
        Simplified diagonal (ρ=δ_{ij}): σ² × T = Σ w_i² × σ_i² × T
        """
        D = 1.0 / np.cumprod(1.0 + tau * L0)
        annuity = tau * D.sum()
        swap_rate = (1.0 - D[-1]) / annuity
        if abs(swap_rate) < 1e-15 or abs(annuity) < 1e-15:
            return 0.0
        # Annuity weights (Rebonato 2002, eq 6.23)
        weights = tau * D * L0 / (annuity * swap_rate)
        var = np.sum(weights**2 * vols**2) * T_expiry
        return math.sqrt(var / T_expiry) if T_expiry > 0 else 0.0
