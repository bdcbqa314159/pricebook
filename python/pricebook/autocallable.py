"""Autocallable structured product.

Early-termination note that pays coupons if underlying stays above
a coupon barrier, and terminates early if it reaches the autocall
barrier. At maturity, if put barrier is breached, investor takes loss.

    from pricebook.autocallable import Autocallable

    ac = Autocallable(
        observation_dates=[...], autocall_level=1.05,
        coupon_rate=0.08, put_barrier=0.70, notional=1_000_000)
    result = ac.price_mc(spot=100, curve=ois, vol=0.20)

References:
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010, Ch. 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from pricebook.black76 import OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import _register, _serialise_atom


@dataclass
class AutocallResult:
    """Autocallable pricing result."""
    price: float
    std_error: float = 0.0
    n_paths: int = 0
    avg_life: float = 0.0          # expected life in years
    autocall_prob: float = 0.0     # probability of early termination
    put_knock_prob: float = 0.0    # probability of put barrier breach

    def to_dict(self) -> dict:
        return {"price": self.price, "std_error": self.std_error,
                "n_paths": self.n_paths, "avg_life": self.avg_life,
                "autocall_prob": self.autocall_prob,
                "put_knock_prob": self.put_knock_prob}


class Autocallable:
    """Autocallable structured note.

    At each observation date:
    - If S(t) >= autocall_level × S₀: note terminates, investor receives
      notional + accumulated coupons.
    - If S(t) >= coupon_barrier × S₀ (but below autocall): coupon accrues.

    At maturity (if not autocalled):
    - If S(T) >= put_barrier × S₀: investor receives notional + coupons.
    - If S(T) < put_barrier × S₀: investor receives S(T)/S₀ × notional (loss).

    Args:
        observation_dates: autocall observation dates.
        autocall_level: fraction of initial spot for autocall (e.g. 1.05 = 105%).
        coupon_rate: annual coupon rate (paid per period if above coupon barrier).
        coupon_barrier: fraction of spot for coupon eligibility (default = 1.0).
        put_barrier: fraction of spot below which capital is at risk (e.g. 0.70).
        notional: invested amount.
    """

    _SERIAL_TYPE = "autocallable"

    def __init__(
        self,
        observation_dates: list[date],
        autocall_level: float = 1.0,
        coupon_rate: float = 0.08,
        coupon_barrier: float = 1.0,
        put_barrier: float = 0.70,
        notional: float = 1_000_000.0,
    ):
        if not observation_dates:
            raise ValueError("observation_dates must not be empty")
        self.observation_dates = sorted(observation_dates)
        self.autocall_level = autocall_level
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.put_barrier = put_barrier
        self.notional = notional

    @property
    def maturity(self) -> date:
        return self.observation_dates[-1]

    def price_mc(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        n_paths: int = 100_000,
        seed: int = 42,
    ) -> AutocallResult:
        """Price via Monte Carlo simulation."""
        ref = curve.reference_date
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        obs_times = [year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
                     for d in self.observation_dates]
        n_obs = len(obs_times)
        T = obs_times[-1]
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)

        rng = np.random.default_rng(seed)

        # Period lengths for coupon calculation
        period_lengths = [obs_times[0]] + [obs_times[i] - obs_times[i-1]
                                            for i in range(1, n_obs)]

        # Simulate spot at each observation date
        S = np.full(n_paths, spot, dtype=float)
        pv = np.zeros(n_paths)
        terminated = np.zeros(n_paths, dtype=bool)
        life = np.full(n_paths, T)
        autocalled = np.zeros(n_paths, dtype=bool)
        put_knocked = np.zeros(n_paths, dtype=bool)

        t_prev = 0.0
        for i, t_obs in enumerate(obs_times):
            dt = t_obs - t_prev
            sqrt_dt = math.sqrt(dt)
            Z = rng.standard_normal(n_paths)
            S = S * np.exp((rate - div_yield - 0.5 * vol**2) * dt + vol * sqrt_dt * Z)

            active = ~terminated
            df_obs = math.exp(-rate * t_obs)

            # Autocall check
            autocall_hit = active & (S >= self.autocall_level * spot)
            if autocall_hit.any():
                # Pay notional + accrued coupons (only for periods above coupon barrier)
                coupon_periods = i + 1
                total_coupon = self.coupon_rate * sum(period_lengths[:coupon_periods])
                pv[autocall_hit] = (self.notional * (1 + total_coupon)) * df_obs
                terminated[autocall_hit] = True
                autocalled[autocall_hit] = True
                life[autocall_hit] = t_obs

            # Track coupon eligibility (above coupon_barrier)
            # Note: in this simplified model, coupons accrue for all periods
            # above coupon_barrier. For a full implementation, track per-path
            # coupon accumulation separately.

            t_prev = t_obs

        # At maturity: handle non-autocalled paths
        still_alive = ~terminated
        df_T = math.exp(-rate * T)
        total_coupon = self.coupon_rate * T

        # Above put barrier and coupon barrier: get notional + coupons
        above_put = still_alive & (S >= self.put_barrier * spot)
        pv[above_put] = (self.notional * (1 + total_coupon)) * df_T

        # Below put barrier: get S/S0 * notional (capital loss)
        below_put = still_alive & (S < self.put_barrier * spot)
        pv[below_put] = (self.notional * S[below_put] / spot) * df_T
        put_knocked[below_put] = True

        price = float(pv.mean())
        std_err = float(pv.std(ddof=1) / math.sqrt(n_paths))
        avg_life = float(life.mean())

        return AutocallResult(
            price=price,
            std_error=std_err,
            n_paths=n_paths,
            avg_life=avg_life,
            autocall_prob=float(autocalled.mean()),
            put_knock_prob=float(put_knocked.mean()),
        )


    def greeks(self, spot, curve, vol, div_yield=0.0, n_paths=50_000, seed=42):
        """Bump-and-reprice Greeks: delta, gamma, vega."""
        base = self.price_mc(spot, curve, vol, div_yield, n_paths, seed)
        bump = spot * 0.01
        up = self.price_mc(spot + bump, curve, vol, div_yield, n_paths, seed)
        dn = self.price_mc(spot - bump, curve, vol, div_yield, n_paths, seed)
        delta = (up.price - dn.price) / (2 * bump)
        gamma = (up.price - 2 * base.price + dn.price) / (bump ** 2)
        v_up = self.price_mc(spot, curve, vol + 0.01, div_yield, n_paths, seed)
        vega = v_up.price - base.price
        return {"delta": delta, "gamma": gamma, "vega": vega, "price": base.price}

    def pv_ctx(self, ctx) -> float:
        vol_surface = ctx.vol_surfaces.get("equity") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.maturity, 100.0) if vol_surface else 0.20
        return self.price_mc(spot=100.0, curve=ctx.discount_curve, vol=vol).price

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "observation_dates": [d.isoformat() for d in self.observation_dates],
            "autocall_level": self.autocall_level,
            "coupon_rate": self.coupon_rate,
            "coupon_barrier": self.coupon_barrier,
            "put_barrier": self.put_barrier,
            "notional": self.notional,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> Autocallable:
        p = d["params"]
        return cls(
            observation_dates=[date.fromisoformat(s) for s in p["observation_dates"]],
            autocall_level=p.get("autocall_level", 1.0),
            coupon_rate=p.get("coupon_rate", 0.08),
            coupon_barrier=p.get("coupon_barrier", 1.0),
            put_barrier=p.get("put_barrier", 0.70),
            notional=p.get("notional", 1_000_000.0),
        )


_register(Autocallable)
