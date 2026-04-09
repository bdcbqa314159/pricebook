"""Local volatility: Dupire equation and local vol MC simulation.

Computes the Dupire local vol surface from market implied vols, then
simulates spot paths under local vol dynamics for exotic pricing.

    from pricebook.local_vol import LocalVolSurface, local_vol_mc

    lv = LocalVolSurface.from_implied_vols(spot, rate, strikes, expiries, vols)
    paths = local_vol_mc(spot, rate, lv, T, n_steps, n_paths)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


# ---- Local vol surface ----

class LocalVolSurface:
    """Dupire local vol surface σ_loc(K, T).

    Stored on a (strike, time) grid with bilinear interpolation.

    Args:
        strikes: sorted strike array.
        times: sorted time-to-expiry array.
        local_vols: 2D array of shape (len(times), len(strikes)).
    """

    def __init__(
        self,
        strikes: np.ndarray,
        times: np.ndarray,
        local_vols: np.ndarray,
    ):
        self.strikes = np.asarray(strikes, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.vols = np.asarray(local_vols, dtype=float)

    def vol(self, strike: float, T: float) -> float:
        """Interpolated local vol at (strike, T)."""
        if T <= 0:
            # At T=0, use first time slice
            return float(self._interp_strike(strike, 0))

        # Clamp to grid boundaries
        T = np.clip(T, self.times[0], self.times[-1])
        strike = np.clip(strike, self.strikes[0], self.strikes[-1])

        if len(self.times) == 1:
            return float(self._interp_strike(strike, 0))

        # Bilinear interpolation
        ti = int(np.searchsorted(self.times, T)) - 1
        ti = max(0, min(ti, len(self.times) - 2))
        ft = (T - self.times[ti]) / (self.times[ti + 1] - self.times[ti])

        v0 = self._interp_strike(strike, ti)
        v1 = self._interp_strike(strike, ti + 1)

        return float(v0 * (1 - ft) + v1 * ft)

    def _interp_strike(self, strike: float, time_idx: int) -> float:
        row = self.vols[time_idx]
        if len(self.strikes) == 1:
            return row[0]
        ki = int(np.searchsorted(self.strikes, strike)) - 1
        ki = max(0, min(ki, len(self.strikes) - 2))
        fk = (strike - self.strikes[ki]) / (self.strikes[ki + 1] - self.strikes[ki])
        return row[ki] * (1 - fk) + row[ki + 1] * fk

    @classmethod
    def from_implied_vols(
        cls,
        spot: float,
        rate: float,
        strikes: list[float],
        times: list[float],
        implied_vols: list[list[float]],
        div_yield: float = 0.0,
    ) -> LocalVolSurface:
        """Build local vol surface from Black-Scholes implied vols via Dupire.

        Dupire equation:
            σ_loc²(K,T) = (∂C/∂T + (r-q)K ∂C/∂K + qC) / (0.5 K² ∂²C/∂K²)

        Uses finite differences on the implied vol surface.

        Args:
            spot: current spot price.
            rate: risk-free rate (continuous).
            strikes: sorted strike levels.
            times: sorted times to expiry.
            implied_vols: 2D list, shape (len(times), len(strikes)).
            div_yield: continuous dividend yield.
        """
        K = np.array(strikes)
        T = np.array(times)
        iv = np.array(implied_vols)  # shape (nT, nK)

        nT, nK = iv.shape
        lv = np.zeros_like(iv)

        for i in range(nT):
            for j in range(nK):
                lv[i, j] = _dupire_local_vol(
                    spot, rate, div_yield, K, T, iv, i, j,
                )

        return cls(K, T, lv)


def _dupire_local_vol(
    spot: float,
    rate: float,
    div_yield: float,
    K: np.ndarray,
    T: np.ndarray,
    iv: np.ndarray,
    ti: int,
    ki: int,
) -> float:
    """Compute Dupire local vol at one grid point via finite differences."""
    nT, nK = iv.shape
    sigma = iv[ti, ki]
    t = T[ti]
    k = K[ki]

    if t <= 0 or sigma <= 0:
        return sigma

    # ∂σ/∂T
    if ti == 0:
        dsdt = (iv[1, ki] - iv[0, ki]) / (T[1] - T[0]) if nT > 1 else 0.0
    elif ti == nT - 1:
        dsdt = (iv[ti, ki] - iv[ti - 1, ki]) / (T[ti] - T[ti - 1])
    else:
        dsdt = (iv[ti + 1, ki] - iv[ti - 1, ki]) / (T[ti + 1] - T[ti - 1])

    # ∂σ/∂K
    if ki == 0:
        dsdk = (iv[ti, 1] - iv[ti, 0]) / (K[1] - K[0]) if nK > 1 else 0.0
    elif ki == nK - 1:
        dsdk = (iv[ti, ki] - iv[ti, ki - 1]) / (K[ki] - K[ki - 1])
    else:
        dsdk = (iv[ti, ki + 1] - iv[ti, ki - 1]) / (K[ki + 1] - K[ki - 1])

    # ∂²σ/∂K²
    if nK < 3 or ki == 0 or ki == nK - 1:
        d2sdk2 = 0.0
    else:
        dk1 = K[ki] - K[ki - 1]
        dk2 = K[ki + 1] - K[ki]
        d2sdk2 = 2 * (iv[ti, ki + 1] * dk1 + iv[ti, ki - 1] * dk2 - iv[ti, ki] * (dk1 + dk2)) / (dk1 * dk2 * (dk1 + dk2))

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / k) + (rate - div_yield + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)

    # Dupire numerator
    numerator = (sigma / (2 * t) + dsdt
                 + (rate - div_yield) * k * dsdk)

    # Dupire denominator
    denom = (1 / (k * sigma * sqrt_t)
             + d1 * dsdk / sigma
             + k * sqrt_t * d2sdk2) ** 2
    denom_full = k ** 2 * (denom if denom > 0 else 1e-10)

    # σ_loc² = 2 × numerator / (K² × denominator_factor)
    # Simplified: use the total variance form
    w = sigma ** 2 * t  # total variance
    dw_dt = 2 * sigma * t * dsdt + sigma ** 2
    dw_dk = 2 * sigma * t * dsdk
    d2w_dk2 = 2 * t * (dsdk ** 2 + sigma * d2sdk2)

    # Gatheral's formula for local vol from total variance:
    # σ_loc² = dw/dT / (1 - y/w × dw/dy + 0.25(-1/4 - 1/w + y²/w²)(dw/dy)² + 0.5 d²w/dy²)
    # Simplified approach: just use numerator/denominator
    local_var = max(dw_dt, 1e-10)

    # Denominator corrections from strike dependence
    y = math.log(k / spot)
    if abs(w) > 1e-15:
        dw_dy = k * dw_dk
        d2w_dy2 = k ** 2 * d2w_dk2 + k * dw_dk
        denom_corr = (1
                       - y / w * dw_dy
                       + 0.25 * (-0.25 - 1 / w + y ** 2 / w ** 2) * dw_dy ** 2
                       + 0.5 * d2w_dy2)
        if denom_corr > 0.01:
            local_var = dw_dt / denom_corr

    return math.sqrt(max(local_var, 1e-10))


# ---- Local vol MC ----

def local_vol_mc(
    spot: float,
    rate: float,
    lv_surface: LocalVolSurface,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    div_yield: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate spot paths under local vol dynamics.

    dS/S = (r - q)dt + σ_loc(S, t) dW

    Returns:
        Array of shape (n_paths,) with terminal spot values.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full(n_paths, spot, dtype=float)

    for step in range(n_steps):
        t = step * dt
        Z = rng.standard_normal(n_paths)
        vols = np.array([lv_surface.vol(s, t) for s in S])
        S = S * np.exp(
            (rate - div_yield - 0.5 * vols ** 2) * dt + vols * sqrt_dt * Z
        )

    return S


def local_vol_mc_european(
    spot: float,
    rate: float,
    lv_surface: LocalVolSurface,
    strike: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 100,
    n_paths: int = 50_000,
    div_yield: float = 0.0,
    seed: int = 42,
) -> float:
    """Price a European option under local vol via MC."""
    S_T = local_vol_mc(spot, rate, lv_surface, T, n_steps, n_paths, div_yield, seed)
    df = math.exp(-rate * T)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - strike, 0)
    else:
        payoffs = np.maximum(strike - S_T, 0)

    return float(df * payoffs.mean())
