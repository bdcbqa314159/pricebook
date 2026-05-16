"""Equity vol desk: surface management, RV strategies, vega ladder, cross-Greeks.

Mid-office tooling for an equity vol trader.

* ``EquityVolSurface`` — strike × expiry surface with parallel / term /
  skew / curvature bumps.
* ``CalendarSpread``, ``RiskReversal`` — vol RV trades.
* ``VarianceSwap`` — pricing via the static call/put strip replication.
* ``vega_ladder`` — aggregate vega by (expiry, strike) bucket.
* ``volga``, ``vanna`` — second-order vol Greeks via finite differences.

References:
    Demeterfi, Derman, Kamal & Zou, *More Than You Ever Wanted To Know
    About Variance Swaps*, Goldman Sachs Quantitative Strategies, 1999.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.black76 import OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.equity_option import equity_option_price, equity_vega
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike


# ---- Vol surface ----

@dataclass
class VolPillar:
    """One pillar in a strike × expiry vol surface."""
    expiry: date
    strikes: list[float]
    vols: list[float]


class EquityVolSurface:
    """Equity vol surface anchored on an ATM strike.

    Wraps :class:`VolSurfaceStrike` with mutable bump operations
    (parallel/term/skew/curvature) and convenience accessors for the
    ATM term-structure and per-expiry skews.

    Args:
        reference_date: valuation date.
        atm_strike: anchor strike for skew/curvature bumps.
        pillars: list of ``VolPillar`` (one per expiry).
        day_count: day count convention for expiry → year fraction.
    """

    def __init__(
        self,
        reference_date: date,
        atm_strike: float,
        pillars: list[VolPillar],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if not pillars:
            raise ValueError("need at least one pillar")
        self.reference_date = reference_date
        self.atm_strike = atm_strike
        self.day_count = day_count
        self._pillars = sorted(pillars, key=lambda p: p.expiry)
        self._rebuild()

    def _rebuild(self) -> None:
        smiles = [VolSmile(p.strikes, p.vols) for p in self._pillars]
        self._surface = VolSurfaceStrike(
            self.reference_date,
            [p.expiry for p in self._pillars],
            smiles,
            day_count=self.day_count,
        )

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Vol at (expiry, strike). Defaults to ATM strike if omitted."""
        k = strike if strike is not None else self.atm_strike
        return self._surface.vol(expiry, k)

    @property
    def expiries(self) -> list[date]:
        return [p.expiry for p in self._pillars]

    @property
    def n_pillars(self) -> int:
        return len(self._pillars)

    def atm_term(self) -> list[tuple[date, float]]:
        """ATM vol term structure as ``(expiry, atm_vol)`` pairs."""
        return [(p.expiry, self.vol(p.expiry, self.atm_strike)) for p in self._pillars]

    def skew_at_expiry(self, expiry: date) -> list[tuple[float, float]]:
        """Strike-vs-vol skew at the nearest expiry pillar."""
        p = self._pillars[self._nearest_pillar(expiry)]
        return list(zip(p.strikes, p.vols))

    def _nearest_pillar(self, expiry: date) -> int:
        ref_t = year_fraction(self.reference_date, expiry, self.day_count)
        best, best_dist = 0, float("inf")
        for i, p in enumerate(self._pillars):
            t = year_fraction(self.reference_date, p.expiry, self.day_count)
            d = abs(t - ref_t)
            if d < best_dist:
                best, best_dist = i, d
        return best

    # ---- Bumps (return new surface, original unchanged) ----

    def _copy_with(self, mutate) -> "EquityVolSurface":
        new = [
            VolPillar(p.expiry, list(p.strikes), mutate(p))
            for p in self._pillars
        ]
        return EquityVolSurface(
            self.reference_date, self.atm_strike, new, self.day_count,
        )

    def bump_parallel(self, dvol: float) -> "EquityVolSurface":
        """Shift every (strike, expiry) vol by ``dvol``."""
        return self._copy_with(lambda p: [v + dvol for v in p.vols])

    def bump_term(self, expiry: date, dvol: float) -> "EquityVolSurface":
        """Shift the smile of the nearest expiry pillar by ``dvol``."""
        idx = self._nearest_pillar(expiry)
        new = []
        for j, p in enumerate(self._pillars):
            vols = (
                [v + dvol for v in p.vols] if j == idx else list(p.vols)
            )
            new.append(VolPillar(p.expiry, list(p.strikes), vols))
        return EquityVolSurface(
            self.reference_date, self.atm_strike, new, self.day_count,
        )

    def bump_skew(self, dvol: float) -> "EquityVolSurface":
        """Tilt the smile: vol += dvol × (K - K_atm)/K_atm at every pillar."""
        def _tilt(p: VolPillar) -> list[float]:
            return [
                v + dvol * (k - self.atm_strike) / self.atm_strike
                for k, v in zip(p.strikes, p.vols)
            ]
        return self._copy_with(_tilt)

    def bump_curvature(self, dvol: float) -> "EquityVolSurface":
        """Quadratic bump centred at the ATM strike."""
        def _curve(p: VolPillar) -> list[float]:
            return [
                v + dvol * ((k - self.atm_strike) / self.atm_strike) ** 2
                for k, v in zip(p.strikes, p.vols)
            ]
        return self._copy_with(_curve)


# ---- RV strategies ----

@dataclass
class CalendarSpread:
    """Same strike, two expiries.

    ``direction = +1`` is long the short-dated leg and short the long-dated
    leg (long front-month vol); ``-1`` is the reverse.
    """
    strike: float
    short_expiry: date
    long_expiry: date
    direction: int = 1
    quantity: float = 1.0
    option_type: OptionType = OptionType.CALL

    def pv(
        self,
        surface: EquityVolSurface,
        spot: float,
        rate: float,
        div_yield: float = 0.0,
    ) -> float:
        t_s = year_fraction(surface.reference_date, self.short_expiry, surface.day_count)
        t_l = year_fraction(surface.reference_date, self.long_expiry, surface.day_count)
        v_s = surface.vol(self.short_expiry, self.strike)
        v_l = surface.vol(self.long_expiry, self.strike)
        p_s = equity_option_price(spot, self.strike, rate, v_s, t_s, self.option_type, div_yield)
        p_l = equity_option_price(spot, self.strike, rate, v_l, t_l, self.option_type, div_yield)
        return self.direction * self.quantity * (p_s - p_l)


@dataclass
class RiskReversal:
    """Long out-of-the-money call, short out-of-the-money put (or reverse).

    ``direction = +1`` is long the call / short the put (bullish skew).
    """
    expiry: date
    call_strike: float
    put_strike: float
    direction: int = 1
    quantity: float = 1.0

    def pv(
        self,
        surface: EquityVolSurface,
        spot: float,
        rate: float,
        div_yield: float = 0.0,
    ) -> float:
        T = year_fraction(surface.reference_date, self.expiry, surface.day_count)
        v_call = surface.vol(self.expiry, self.call_strike)
        v_put = surface.vol(self.expiry, self.put_strike)
        call = equity_option_price(
            spot, self.call_strike, rate, v_call, T, OptionType.CALL, div_yield,
        )
        put = equity_option_price(
            spot, self.put_strike, rate, v_put, T, OptionType.PUT, div_yield,
        )
        return self.direction * self.quantity * (call - put)


@dataclass
class VarianceSwap:
    """Variance swap priced via static call/put strip replication.

    Fair variance, with forward F = S·exp((r-q)·T):

        K²_var ≈ (2·e^{rT} / T) · [
            Σ_{K_i ≤ F} P(K_i) · ΔK / K_i² + Σ_{K_i > F} C(K_i) · ΔK / K_i²
        ]

    Reference: Demeterfi-Derman-Kamal-Zou (1999).
    """
    expiry: date
    var_strike: float          # variance strike (square of vol strike)
    notional: float = 1.0      # variance notional (PV is per unit variance)
    direction: int = 1

    def fair_variance(
        self,
        surface: EquityVolSurface,
        spot: float,
        rate: float,
        div_yield: float = 0.0,
        n_strikes: int = 101,
        k_min_mult: float = 0.3,
        k_max_mult: float = 3.0,
    ) -> float:
        T = year_fraction(surface.reference_date, self.expiry, surface.day_count)
        if T <= 0.0:
            return 0.0
        forward = spot * math.exp((rate - div_yield) * T)
        df = math.exp(-rate * T)

        k_lo = k_min_mult * forward
        k_hi = k_max_mult * forward
        step = (k_hi - k_lo) / (n_strikes - 1)
        ks = [k_lo + i * step for i in range(n_strikes)]

        total = 0.0
        for i, k in enumerate(ks):
            vol = surface.vol(self.expiry, k)
            opt_type = OptionType.PUT if k <= forward else OptionType.CALL
            price = equity_option_price(spot, k, rate, vol, T, opt_type, div_yield)
            # Trapezoidal weights: half-width on the edges.
            if i == 0 or i == n_strikes - 1:
                width = step * 0.5
            else:
                width = step
            total += price * width / (k * k)

        return (2.0 / T) * total / df

    def pv(
        self,
        surface: EquityVolSurface,
        spot: float,
        rate: float,
        div_yield: float = 0.0,
        n_strikes: int = 101,
    ) -> float:
        T = year_fraction(surface.reference_date, self.expiry, surface.day_count)
        df = math.exp(-rate * T)
        fair = self.fair_variance(surface, spot, rate, div_yield, n_strikes)
        return self.direction * self.notional * (fair - self.var_strike) * df


# ---- Vega ladder + cross-Greeks ----

@dataclass
class VegaBucket:
    """Aggregated vega in a single (expiry, strike) bucket."""
    expiry: date
    strike: float
    vega: float


def vega_ladder(
    positions: list[tuple[date, float, float]],
) -> list[VegaBucket]:
    """Aggregate vega across positions by ``(expiry, strike)`` bucket.

    Args:
        positions: list of ``(expiry, strike, vega)`` tuples — already
            signed and scaled by direction × quantity.

    Returns:
        list of ``VegaBucket`` sorted by (expiry, strike).
    """
    agg: dict[tuple[date, float], float] = {}
    for expiry, strike, v in positions:
        key = (expiry, strike)
        agg[key] = agg.get(key, 0.0) + v

    return [
        VegaBucket(expiry=k[0], strike=k[1], vega=v)
        for k, v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]


def total_vega(buckets: list[VegaBucket]) -> float:
    """Sum of vegas across all buckets."""
    return sum(b.vega for b in buckets)


def volga(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_yield: float = 0.0,
    dv: float = 1e-4,
) -> float:
    """Volga ≡ ∂²Price/∂σ² ≡ ∂Vega/∂σ via central difference."""
    v_up = equity_vega(spot, strike, rate, vol + dv, T, div_yield)
    v_dn = equity_vega(spot, strike, rate, vol - dv, T, div_yield)
    return (v_up - v_dn) / (2.0 * dv)


def vanna(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_yield: float = 0.0,
    ds: float = 1e-2,
) -> float:
    """Vanna ≡ ∂²Price/(∂S ∂σ) ≡ ∂Vega/∂S via central difference on spot."""
    v_up = equity_vega(spot + ds, strike, rate, vol, T, div_yield)
    v_dn = equity_vega(spot - ds, strike, rate, vol, T, div_yield)
    return (v_up - v_dn) / (2.0 * ds)
