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
        return dict(vars(self))


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


# ═══════════════════════════════════════════════════════════════
# CD2: Cross-Asset Correlation, Intraday Patterns, Jump Decomp
# ═══════════════════════════════════════════════════════════════

@dataclass
class CrossAssetCorrelation:
    """Cross-asset correlation analysis for crypto."""
    correlation_matrix: np.ndarray
    assets: list[str]
    rolling_window: int
    avg_correlation: float
    min_pair: tuple[str, str, float]
    max_pair: tuple[str, str, float]

    def to_dict(self) -> dict:
        return {
            "assets": self.assets,
            "avg_correlation": self.avg_correlation,
            "min_pair": self.min_pair,
            "max_pair": self.max_pair,
        }


def cross_asset_correlation(
    returns: dict[str, list[float]],
    window: int | None = None,
) -> CrossAssetCorrelation:
    """Compute cross-asset correlation matrix for crypto assets.

    Crypto correlations are typically high (0.6–0.9 for majors)
    and spike toward 1.0 in crashes.

    Args:
        returns: {asset_name: [returns]} dict.
        window: if given, use rolling window (last N observations).
    """
    assets = list(returns.keys())
    n = len(assets)

    # Align to shortest
    min_len = min(len(v) for v in returns.values())
    if window:
        min_len = min(min_len, window)

    data = np.column_stack([np.array(returns[a][-min_len:]) for a in assets])
    corr = np.corrcoef(data, rowvar=False)

    # Find min and max off-diagonal pairs
    min_corr = (assets[0], assets[1], 1.0)
    max_corr = (assets[0], assets[1], -1.0)
    corr_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr[i, j])
            corr_sum += c
            count += 1
            if c < min_corr[2]:
                min_corr = (assets[i], assets[j], c)
            if c > max_corr[2]:
                max_corr = (assets[i], assets[j], c)

    return CrossAssetCorrelation(
        correlation_matrix=corr,
        assets=assets,
        rolling_window=window or min_len,
        avg_correlation=corr_sum / count if count > 0 else 0,
        min_pair=min_corr,
        max_pair=max_corr,
    )


@dataclass
class IntradayVolPattern:
    """Intraday volatility pattern (hourly)."""
    hourly_vol: list[float]     # vol multiplier per hour (0-23 UTC)
    peak_hour: int
    trough_hour: int
    peak_to_trough_ratio: float

    def to_dict(self) -> dict:
        return {
            "peak_hour_utc": self.peak_hour,
            "trough_hour_utc": self.trough_hour,
            "ratio": self.peak_to_trough_ratio,
        }


def intraday_vol_pattern(
    hourly_returns: list[float],
    hours: list[int],
) -> IntradayVolPattern:
    """Compute intraday vol pattern from hourly returns.

    Crypto shows clear intraday patterns:
    - US hours (14–21 UTC): highest vol
    - Asian morning (0–4 UTC): lower vol
    - Weekends: slightly lower overall

    Args:
        hourly_returns: return per hour.
        hours: hour of day (0–23 UTC) per observation.
    """
    # Group returns by hour
    by_hour: dict[int, list[float]] = {h: [] for h in range(24)}
    for r, h in zip(hourly_returns, hours):
        by_hour[h % 24].append(r)

    hourly_vol = []
    for h in range(24):
        obs = by_hour[h]
        if len(obs) > 1:
            hourly_vol.append(float(np.std(obs)))
        else:
            hourly_vol.append(0.0)

    # Normalise to average = 1
    avg = sum(hourly_vol) / 24 if any(hourly_vol) else 1
    if avg > 0:
        hourly_vol = [v / avg for v in hourly_vol]

    peak = int(np.argmax(hourly_vol))
    trough = int(np.argmin([v if v > 0 else float('inf') for v in hourly_vol]))
    ratio = hourly_vol[peak] / hourly_vol[trough] if hourly_vol[trough] > 0 else 1

    return IntradayVolPattern(hourly_vol, peak, trough, ratio)


@dataclass
class JumpDecomposition:
    """Decomposition of returns into diffusion and jump components."""
    total_var: float
    diffusion_var: float
    jump_var: float
    jump_fraction: float        # fraction of variance from jumps
    n_jumps: int
    jump_threshold: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def jump_decomposition(
    returns: list[float],
    threshold_sigma: float = 3.0,
) -> JumpDecomposition:
    """Decompose crypto returns into diffusion and jump components.

    Jumps: returns exceeding threshold × σ.
    Diffusion: remaining returns.

    Crypto typically has higher jump fraction (~20-30%) than
    equities (~5-10%).

    Args:
        returns: return series.
        threshold_sigma: number of σ to classify as jump.
    """
    arr = np.array(returns)
    sigma = float(np.std(arr))
    threshold = threshold_sigma * sigma

    jump_mask = np.abs(arr) > threshold
    n_jumps = int(np.sum(jump_mask))

    total_var = float(np.var(arr))
    jump_returns = arr[jump_mask]
    diffusion_returns = arr[~jump_mask]

    jump_var = float(np.var(jump_returns) * n_jumps / len(arr)) if n_jumps > 0 else 0
    diffusion_var = float(np.var(diffusion_returns) * (len(arr) - n_jumps) / len(arr)) if len(diffusion_returns) > 0 else 0

    jump_fraction = jump_var / total_var if total_var > 0 else 0

    return JumpDecomposition(
        total_var=total_var,
        diffusion_var=diffusion_var,
        jump_var=jump_var,
        jump_fraction=jump_fraction,
        n_jumps=n_jumps,
        jump_threshold=threshold,
    )
