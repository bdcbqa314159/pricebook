"""Crypto volatility surface and realised vol.

24/7 trading requires different vol calculations than TradFi.
No market close, no overnight gaps, continuous theta decay.

* :func:`realised_vol_24_7` — realised vol for 24/7 markets.
* :func:`parkinson_vol` — Parkinson high-low estimator.
* :func:`yang_zhang_vol` — Yang-Zhang OHLC estimator.
* :func:`crypto_theta` — 24/7 theta (no weekends).
* :class:`CryptoVolSurface` — vol surface for crypto options.

References:
    Parkinson, *The Extreme Value Method for Estimating the Variance of
    the Rate of Return*, JB, 1980.
    Yang & Zhang, *Drift Independent Volatility Estimation*, JB, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RealizedVolResult:
    """Realised volatility result."""
    vol: float                  # annualised vol (decimal)
    vol_pct: float              # as percentage
    method: str
    n_observations: int
    lookback_days: int

    def to_dict(self) -> dict:
        return vars(self)


def realised_vol_24_7(
    prices: list[float],
    interval_hours: float = 1.0,
    annualise: bool = True,
) -> RealizedVolResult:
    """Realised volatility for 24/7 crypto markets.

    Unlike TradFi (252 trading days), crypto trades 365 × 24.
    Annualisation factor: √(365 × 24 / interval_hours).

    Args:
        prices: price series (e.g. hourly closes).
        interval_hours: time between observations.
    """
    if len(prices) < 2:
        return RealizedVolResult(0, 0, "close_to_close", 0, 0)

    arr = np.array(prices)
    log_returns = np.log(arr[1:] / arr[:-1])
    vol = float(np.std(log_returns, ddof=1))

    if annualise:
        periods_per_year = 365 * 24 / interval_hours
        vol *= math.sqrt(periods_per_year)

    lookback = len(prices) * interval_hours / 24

    return RealizedVolResult(
        vol=vol, vol_pct=vol * 100,
        method="close_to_close_24_7",
        n_observations=len(prices) - 1,
        lookback_days=int(lookback),
    )


def parkinson_vol(
    highs: list[float],
    lows: list[float],
    interval_hours: float = 24.0,
) -> RealizedVolResult:
    """Parkinson high-low volatility estimator.

    σ² = (1/4ln2) × (1/N) × Σ ln(H/L)²

    More efficient than close-to-close: uses intra-period information.
    5× more efficient for pure diffusion.

    Args:
        highs: high prices per period.
        lows: low prices per period.
        interval_hours: period length.
    """
    if len(highs) < 1 or len(highs) != len(lows):
        return RealizedVolResult(0, 0, "parkinson", 0, 0)

    h = np.array(highs)
    l = np.array(lows)
    log_hl = np.log(h / l)
    var = float(np.mean(log_hl**2)) / (4 * math.log(2))
    vol = math.sqrt(var)

    periods_per_year = 365 * 24 / interval_hours
    vol_ann = vol * math.sqrt(periods_per_year)

    return RealizedVolResult(
        vol=vol_ann, vol_pct=vol_ann * 100,
        method="parkinson",
        n_observations=len(highs),
        lookback_days=int(len(highs) * interval_hours / 24),
    )


def yang_zhang_vol(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    interval_hours: float = 24.0,
) -> RealizedVolResult:
    """Yang-Zhang OHLC volatility estimator.

    Combines overnight (close-to-open), open-to-close, and
    Parkinson (high-low) components. Drift-independent.

    σ²_YZ = σ²_overnight + k × σ²_close-to-close + (1-k) × σ²_Rogers-Satchell

    For 24/7 crypto: overnight component is negligible (no gaps).

    Args:
        opens, highs, lows, closes: OHLC data per period.
        interval_hours: period length.
    """
    n = len(closes)
    if n < 2:
        return RealizedVolResult(0, 0, "yang_zhang", 0, 0)

    o = np.array(opens)
    h = np.array(highs)
    l = np.array(lows)
    c = np.array(closes)

    # Overnight variance (close-to-open) — minimal for 24/7
    log_co = np.log(o[1:] / c[:-1])
    var_overnight = float(np.var(log_co, ddof=1)) if len(log_co) > 1 else 0

    # Close-to-close variance
    log_cc = np.log(c[1:] / c[:-1])
    var_cc = float(np.var(log_cc, ddof=1)) if len(log_cc) > 1 else 0

    # Rogers-Satchell variance
    log_hc = np.log(h / c)
    log_ho = np.log(h / o)
    log_lc = np.log(l / c)
    log_lo = np.log(l / o)
    var_rs = float(np.mean(log_ho * log_hc + log_lo * log_lc))

    # Yang-Zhang combination
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_yz = var_overnight + k * var_cc + (1 - k) * var_rs

    vol = math.sqrt(max(var_yz, 0))
    periods_per_year = 365 * 24 / interval_hours
    vol_ann = vol * math.sqrt(periods_per_year)

    return RealizedVolResult(
        vol=vol_ann, vol_pct=vol_ann * 100,
        method="yang_zhang",
        n_observations=n,
        lookback_days=int(n * interval_hours / 24),
    )


def crypto_theta(
    option_price: float,
    vol: float,
    spot: float,
    T: float,
) -> float:
    """24/7 theta: time decay per calendar day (not trading day).

    Crypto θ = BS_theta / 365  (not / 252 like equity).
    Crypto options decay on weekends too.

    Args:
        option_price: current option value.
        vol: implied vol.
        spot: underlying price.
        T: time to expiry (years).
    """
    if T <= 0 or vol <= 0:
        return 0.0
    # Approximate: θ ≈ −S × σ × N'(d1) / (2√T) / 365
    from pricebook.models.black76 import _norm_pdf
    d1 = (math.log(1) + 0.5 * vol**2 * T) / (vol * math.sqrt(T))  # ATM approx
    return -spot * vol * _norm_pdf(d1) / (2 * math.sqrt(T)) / 365.0


@dataclass
class CryptoVolSurface:
    """Crypto implied vol surface.

    Indexed by (expiry_days, delta) or (expiry_days, strike).
    """
    expiry_days: list[int]
    strikes: list[list[float]]      # per expiry
    vols: list[list[float]]         # per expiry per strike
    spot: float

    def vol(self, days: int, strike: float) -> float:
        """Interpolate vol at (days, strike)."""
        # Find nearest expiry
        idx = min(range(len(self.expiry_days)),
                  key=lambda i: abs(self.expiry_days[i] - days))
        strikes = self.strikes[idx]
        ivs = self.vols[idx]

        # Linear interpolation in strike
        if strike <= strikes[0]:
            return ivs[0]
        if strike >= strikes[-1]:
            return ivs[-1]
        for j in range(len(strikes) - 1):
            if strikes[j] <= strike <= strikes[j + 1]:
                w = (strike - strikes[j]) / (strikes[j + 1] - strikes[j])
                return ivs[j] + w * (ivs[j + 1] - ivs[j])
        return ivs[-1]

    @property
    def atm_term_structure(self) -> list[tuple[int, float]]:
        """ATM vol term structure: [(days, vol), ...]."""
        result = []
        for i, days in enumerate(self.expiry_days):
            atm_vol = self.vol(days, self.spot)
            result.append((days, atm_vol))
        return result

    def to_dict(self) -> dict:
        return {
            "n_expiries": len(self.expiry_days),
            "spot": self.spot,
            "atm_vols": [v for _, v in self.atm_term_structure],
        }
