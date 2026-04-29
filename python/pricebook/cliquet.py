"""Cliquet (ratchet) option: periodic reset with local/global caps and floors.

Each period: local_return = clip(S(end)/S(start) - 1, local_floor, local_cap).
Total payoff = clip(Σ local_returns, global_floor, global_cap) × notional.

    from pricebook.cliquet import Cliquet

    cliq = Cliquet(reset_dates=[...], local_floor=0.0, local_cap=0.05,
                    global_floor=0.0, global_cap=0.30, notional=1_000_000)
    result = cliq.price_mc(spot=100, curve=ois, vol=0.20)

References:
    Wilmott, *Paul Wilmott on Quantitative Finance*, Vol. 2, Ch. 25.
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Ch. 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import _register


@dataclass
class CliquetResult:
    """Cliquet pricing result."""
    price: float
    std_error: float = 0.0
    n_paths: int = 0
    avg_total_return: float = 0.0
    avg_capped_periods: float = 0.0   # avg periods hitting local cap
    avg_floored_periods: float = 0.0  # avg periods hitting local floor

    def to_dict(self) -> dict:
        return {"price": self.price, "std_error": self.std_error,
                "n_paths": self.n_paths, "avg_total_return": self.avg_total_return,
                "avg_capped_periods": self.avg_capped_periods,
                "avg_floored_periods": self.avg_floored_periods}


class Cliquet:
    """Cliquet (ratchet) option with local and global caps/floors.

    At each reset period [t_{i-1}, t_i], the local return is:
        R_i = clip(S(t_i)/S(t_{i-1}) - 1, local_floor, local_cap)

    The total payoff at maturity is:
        payoff = notional × clip(Σ R_i, global_floor, global_cap)

    Args:
        reset_dates: period end dates (first period starts at valuation).
        local_floor: minimum return per period (e.g. 0.0 for no downside).
        local_cap: maximum return per period (e.g. 0.05 for 5% cap).
        global_floor: minimum total return (e.g. 0.0).
        global_cap: maximum total return (e.g. 0.30 for 30% cap).
        notional: position size.
    """

    _SERIAL_TYPE = "cliquet"

    def __init__(
        self,
        reset_dates: list[date],
        local_floor: float = 0.0,
        local_cap: float = 0.10,
        global_floor: float = 0.0,
        global_cap: float = 1.0,
        notional: float = 1_000_000.0,
    ):
        if len(reset_dates) < 1:
            raise ValueError("need at least 1 reset date")
        self.reset_dates = sorted(reset_dates)
        self.local_floor = local_floor
        self.local_cap = local_cap
        self.global_floor = global_floor
        self.global_cap = global_cap
        self.notional = notional

    @property
    def maturity(self) -> date:
        return self.reset_dates[-1]

    @property
    def n_periods(self) -> int:
        return len(self.reset_dates)

    def price_mc(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        n_paths: int = 100_000,
        seed: int = 42,
    ) -> CliquetResult:
        """Price via Monte Carlo."""
        ref = curve.reference_date
        reset_times = [year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
                       for d in self.reset_dates]
        T = reset_times[-1]
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)
        df = math.exp(-rate * T)

        rng = np.random.default_rng(seed)

        S_prev = np.full(n_paths, spot, dtype=float)
        total_return = np.zeros(n_paths)
        n_capped = np.zeros(n_paths)
        n_floored = np.zeros(n_paths)

        t_prev = 0.0
        for i, t in enumerate(reset_times):
            dt = t - t_prev
            sqrt_dt = math.sqrt(max(dt, 1e-10))
            Z = rng.standard_normal(n_paths)
            S = S_prev * np.exp((rate - div_yield - 0.5 * vol**2) * dt + vol * sqrt_dt * Z)

            # Local return
            local_ret = S / S_prev - 1.0
            capped = local_ret > self.local_cap
            floored = local_ret < self.local_floor
            n_capped += capped.astype(float)
            n_floored += floored.astype(float)

            clipped = np.clip(local_ret, self.local_floor, self.local_cap)
            total_return += clipped

            S_prev = S.copy()
            t_prev = t

        # Global clip
        payoff = np.clip(total_return, self.global_floor, self.global_cap)
        discounted = df * payoff * self.notional

        price = float(discounted.mean())
        std_err = float(discounted.std(ddof=1) / math.sqrt(n_paths))

        return CliquetResult(
            price=price, std_error=std_err, n_paths=n_paths,
            avg_total_return=float(total_return.mean()),
            avg_capped_periods=float(n_capped.mean()),
            avg_floored_periods=float(n_floored.mean()),
        )

    def pv_ctx(self, ctx) -> float:
        vol_surface = ctx.vol_surfaces.get("equity") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.maturity, 100.0) if vol_surface else 0.20
        return self.price_mc(spot=100.0, curve=ctx.discount_curve, vol=vol).price

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "reset_dates": [d.isoformat() for d in self.reset_dates],
            "local_floor": self.local_floor, "local_cap": self.local_cap,
            "global_floor": self.global_floor, "global_cap": self.global_cap,
            "notional": self.notional,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> Cliquet:
        p = d["params"]
        return cls(
            reset_dates=[date.fromisoformat(s) for s in p["reset_dates"]],
            local_floor=p.get("local_floor", 0.0),
            local_cap=p.get("local_cap", 0.10),
            global_floor=p.get("global_floor", 0.0),
            global_cap=p.get("global_cap", 1.0),
            notional=p.get("notional", 1_000_000.0),
        )


_register(Cliquet)
