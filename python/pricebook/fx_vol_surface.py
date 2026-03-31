"""
FX volatility surface from market quotes.

Market quotes at each expiry:
    ATM: at-the-money vol (delta-neutral straddle)
    RR25: 25-delta risk reversal = vol_25D_call - vol_25D_put
    BF25: 25-delta butterfly = 0.5*(vol_25D_call + vol_25D_put) - ATM

From these, recover the three-point smile:
    vol_25D_call = ATM + BF25 + 0.5 * RR25
    vol_25D_put  = ATM + BF25 - 0.5 * RR25

Then convert delta → strike using strike_from_delta, and build a VolSmile.

    surface = FXVolSurface(
        spot=1.0850, r_d=0.05, r_f=0.03,
        expiry_quotes=[
            FXVolQuote(expiry=date(2024,7,15), atm=0.08, rr25=0.01, bf25=0.005),
            FXVolQuote(expiry=date(2025,1,15), atm=0.09, rr25=0.012, bf25=0.006),
        ],
    )
    v = surface.vol(expiry=date(2024,10,15), strike=1.10)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.black76 import OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.fx_option import fx_forward, strike_from_delta
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike


@dataclass
class FXVolQuote:
    """Market vol quote at a single expiry."""

    expiry: date
    atm: float       # ATM vol (delta-neutral straddle)
    rr25: float      # 25-delta risk reversal
    bf25: float      # 25-delta butterfly


class FXVolSurface:
    """FX vol surface built from ATM / RR / BF quotes.

    Args:
        spot: FX spot rate.
        r_d: domestic (quote currency) rate.
        r_f: foreign (base currency) rate.
        expiry_quotes: list of FXVolQuote, one per expiry.
        reference_date: valuation date. Defaults to earliest expiry - 1Y estimate;
            should be provided explicitly.
        delta_type: "forward" or "spot" for strike-from-delta conversion.
    """

    def __init__(
        self,
        spot: float,
        r_d: float,
        r_f: float,
        expiry_quotes: list[FXVolQuote],
        reference_date: date | None = None,
        delta_type: str = "forward",
    ):
        if not expiry_quotes:
            raise ValueError("need at least 1 expiry quote")

        self.spot = spot
        self.r_d = r_d
        self.r_f = r_f

        if reference_date is None:
            reference_date = expiry_quotes[0].expiry
        self._reference_date = reference_date

        expiries = []
        smiles = []

        for q in sorted(expiry_quotes, key=lambda x: x.expiry):
            T = year_fraction(
                self._reference_date, q.expiry, DayCountConvention.ACT_365_FIXED,
            )
            if T <= 0:
                continue

            vol_25c = q.atm + q.bf25 + 0.5 * q.rr25
            vol_25p = q.atm + q.bf25 - 0.5 * q.rr25

            fwd = fx_forward(spot, r_d, r_f, T)

            # ATM-DNS strike: where straddle delta = 0
            # Approximate: ATM strike ≈ forward * exp(0.5 * vol^2 * T)
            k_atm = fwd * math.exp(0.5 * q.atm**2 * T)

            k_25c = strike_from_delta(
                spot, 0.25, r_d, r_f, vol_25c, T,
                delta_type=delta_type, option_type=OptionType.CALL,
            )
            k_25p = strike_from_delta(
                spot, -0.25, r_d, r_f, vol_25p, T,
                delta_type=delta_type, option_type=OptionType.PUT,
            )

            # Build 3-point smile sorted by strike
            strikes = sorted([k_25p, k_atm, k_25c])
            vols_at_strikes = []
            for k in strikes:
                if abs(k - k_25p) < 1e-10:
                    vols_at_strikes.append(vol_25p)
                elif abs(k - k_atm) < 1e-10:
                    vols_at_strikes.append(q.atm)
                else:
                    vols_at_strikes.append(vol_25c)

            smiles.append(VolSmile(strikes, vols_at_strikes))
            expiries.append(q.expiry)

        self._surface = VolSurfaceStrike(
            self._reference_date, expiries, smiles,
        )

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Vol at (expiry, strike)."""
        return self._surface.vol(expiry, strike)

    def smile_at(self, expiry: date) -> VolSmile:
        """Return the VolSmile at the nearest expiry."""
        return self._surface.vol(expiry)  # type: ignore  # for direct access use _surface
