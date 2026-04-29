"""Target Redemption Forward (TARF).

Accumulating forward that terminates when accumulated profit reaches
a target level. Popular in FX structured products.

Each period: if S(t) > strike, buyer profits (S(t) - K) × notional.
Accumulated profit is tracked. Once target is hit, contract terminates.
If S(t) < strike, buyer has a leveraged loss (knock-in leverage).

    from pricebook.tarf import TARF

    tarf = TARF(strike=1.10, target=0.10, leverage=2.0,
                fixing_dates=[...], notional=1_000_000)
    result = tarf.price_mc(spot=1.10, curve=ois, vol=0.08)

References:
    Wystup, U. (2017). FX Options and Structured Products, 2nd ed., Ch. 6.
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Ch. 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.serialisable import _register, _serialise_atom


@dataclass
class TARFResult:
    """TARF pricing result."""
    price: float
    std_error: float = 0.0
    n_paths: int = 0
    avg_life: float = 0.0             # expected life in years
    target_hit_prob: float = 0.0      # probability of early termination
    avg_accumulated: float = 0.0      # average accumulated gain at termination

    def to_dict(self) -> dict:
        return {"price": self.price, "std_error": self.std_error,
                "n_paths": self.n_paths, "avg_life": self.avg_life,
                "target_hit_prob": self.target_hit_prob,
                "avg_accumulated": self.avg_accumulated}


class TARF:
    """Target Redemption Forward.

    At each fixing date:
    - If S(t) >= strike (buyer profits):
        gain = (S(t)/strike - 1) × notional
        accumulated += gain
        If accumulated >= target × notional: contract terminates.
    - If S(t) < strike (buyer loses):
        loss = (1 - S(t)/strike) × leverage × notional
        (No accumulation toward target from losses.)

    PV = discounted sum of all cashflows until termination.

    Args:
        strike: forward strike rate.
        target: target profit as fraction of notional (e.g. 0.10 = 10%).
        leverage: loss multiplier when below strike (e.g. 2.0).
        fixing_dates: observation dates.
        notional: per-period notional.
        pivot: alternative strike for gain/loss split (default = strike).
    """

    _SERIAL_TYPE = "tarf"

    def __init__(
        self,
        strike: float,
        target: float,
        leverage: float = 2.0,
        fixing_dates: list[date] | None = None,
        notional: float = 1_000_000.0,
        pivot: float | None = None,
    ):
        if fixing_dates is None or len(fixing_dates) == 0:
            raise ValueError("fixing_dates must not be empty")
        self.strike = strike
        self.target = target
        self.leverage = leverage
        self.fixing_dates = sorted(fixing_dates)
        self.notional = notional
        self.pivot = pivot if pivot is not None else strike

    @property
    def maturity(self) -> date:
        return self.fixing_dates[-1]

    def price_mc(
        self,
        spot: float,
        curve: DiscountCurve,
        vol: float,
        div_yield: float = 0.0,
        n_paths: int = 100_000,
        seed: int = 42,
    ) -> TARFResult:
        """Price via Monte Carlo."""
        ref = curve.reference_date
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        fix_times = [year_fraction(ref, d, DayCountConvention.ACT_365_FIXED)
                     for d in self.fixing_dates]
        T = fix_times[-1]
        rate = -math.log(curve.df(self.maturity)) / max(T, 1e-10)

        rng = np.random.default_rng(seed)
        target_notional = self.target * self.notional

        S = np.full(n_paths, spot, dtype=float)
        pv = np.zeros(n_paths)
        accumulated = np.zeros(n_paths)
        terminated = np.zeros(n_paths, dtype=bool)
        life = np.full(n_paths, T)
        target_hit = np.zeros(n_paths, dtype=bool)

        t_prev = 0.0
        for i, t_fix in enumerate(fix_times):
            dt = t_fix - t_prev
            sqrt_dt = math.sqrt(max(dt, 1e-10))
            Z = rng.standard_normal(n_paths)
            S = S * np.exp((rate - div_yield - 0.5 * vol**2) * dt + vol * sqrt_dt * Z)

            active = ~terminated
            df_t = math.exp(-rate * t_fix)

            # Above pivot: buyer gains
            above = active & (S >= self.pivot)
            gain = (S / self.strike - 1.0) * self.notional
            gain = np.where(above, gain, 0.0)
            pv += gain * df_t * above

            # Accumulate gains
            accumulated += np.where(above, gain, 0.0)

            # Check target
            hit = active & above & (accumulated >= target_notional)
            if hit.any():
                # Cap the gain at target
                excess = accumulated[hit] - target_notional
                pv[hit] -= excess * df_t  # remove excess
                accumulated[hit] = target_notional
                terminated[hit] = True
                target_hit[hit] = True
                life[hit] = t_fix

            # Below pivot: buyer loses (leveraged)
            below = active & ~above & ~terminated
            loss = (1.0 - S / self.strike) * self.leverage * self.notional
            loss = np.where(below, loss, 0.0)
            pv -= loss * df_t * below

            t_prev = t_fix

        price = float(pv.mean())
        std_err = float(pv.std(ddof=1) / math.sqrt(n_paths))

        return TARFResult(
            price=price, std_error=std_err, n_paths=n_paths,
            avg_life=float(life.mean()),
            target_hit_prob=float(target_hit.mean()),
            avg_accumulated=float(accumulated.mean()),
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
        vol_surface = ctx.vol_surfaces.get("fx") if ctx.vol_surfaces else None
        vol = vol_surface.vol(self.maturity, self.strike) if vol_surface else 0.10
        return self.price_mc(spot=self.strike, curve=ctx.discount_curve, vol=vol).price

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "strike": self.strike, "target": self.target,
            "leverage": self.leverage,
            "fixing_dates": [d.isoformat() for d in self.fixing_dates],
            "notional": self.notional,
            "pivot": self.pivot,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> TARF:
        p = d["params"]
        return cls(
            strike=p["strike"], target=p["target"],
            leverage=p.get("leverage", 2.0),
            fixing_dates=[date.fromisoformat(s) for s in p["fixing_dates"]],
            notional=p.get("notional", 1_000_000.0),
            pivot=p.get("pivot"),
        )


_register(TARF)
