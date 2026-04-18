"""Cross-asset structured notes: fusion, correlation triggers, hybrids.

* :func:`equity_fx_fusion_note` — equity performance in foreign currency.
* :func:`correlation_trigger_note` — coupon triggered by ρ level.
* :func:`commodity_equity_autocall` — equity autocall with commodity knock-out.
* :func:`dual_asset_range_accrual` — accrues if BOTH assets in range.

References:
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
    De Weert, *Exotic Options Trading*, Wiley, 2008.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class FusionNoteResult:
    price: float
    equity_delta: float
    fx_delta: float
    notional: float

def equity_fx_fusion_note(
    equity_paths: np.ndarray, fx_paths: np.ndarray,
    strike_pct: float, participation: float, notional: float,
    discount_factor: float,
) -> FusionNoteResult:
    """Equity performance paid in foreign currency.
    Payoff = notional × max(participation × (S_T/S_0 × FX_T/FX_0 - 1) + 1, floor).
    """
    eq_return = equity_paths[:, -1] / equity_paths[:, 0] - 1
    fx_return = fx_paths[:, -1] / fx_paths[:, 0]
    combined = participation * eq_return * fx_return
    payoff = notional * np.maximum(1 + combined - strike_pct, 0.0)
    price = float(discount_factor * payoff.mean())
    # Approximate deltas
    eq_d = float(np.corrcoef(equity_paths[:, -1], payoff)[0, 1]) if payoff.std() > 0 else 0.0
    fx_d = float(np.corrcoef(fx_paths[:, -1], payoff)[0, 1]) if payoff.std() > 0 else 0.0
    return FusionNoteResult(price, eq_d, fx_d, notional)


@dataclass
class CorrelationTriggerResult:
    price: float
    coupon_probability: float
    mean_realised_corr: float
    threshold: float

def correlation_trigger_note(
    asset1_paths: np.ndarray, asset2_paths: np.ndarray,
    coupon: float, corr_threshold: float, notional: float,
    discount_factor: float, window: int = 20,
) -> CorrelationTriggerResult:
    """Coupon paid if realised correlation stays below threshold.
    Investor is short correlation: gets paid when ρ is low.
    """
    n_paths, n_steps = asset1_paths.shape
    log_r1 = np.diff(np.log(asset1_paths), axis=1)
    log_r2 = np.diff(np.log(asset2_paths), axis=1)
    # Rolling correlation over last `window` steps
    n = min(window, log_r1.shape[1])
    r1_w = log_r1[:, -n:]
    r2_w = log_r2[:, -n:]
    # Per-path correlation
    corrs = np.array([float(np.corrcoef(r1_w[p], r2_w[p])[0, 1])
                       if r1_w[p].std() > 1e-10 and r2_w[p].std() > 1e-10 else 0.0
                       for p in range(n_paths)])
    triggered = corrs < corr_threshold
    payoff = np.where(triggered, notional * (1 + coupon), notional)
    price = float(discount_factor * payoff.mean())
    return CorrelationTriggerResult(price, float(triggered.mean()),
                                     float(corrs.mean()), corr_threshold)


@dataclass
class CommodityEquityAutocallResult:
    price: float
    autocall_probability: float
    commodity_ko_probability: float

def commodity_equity_autocall(
    equity_paths: np.ndarray, commodity_paths: np.ndarray,
    eq_autocall_barrier: float, commodity_ko_level: float,
    coupon: float, notional: float, observation_steps: list[int],
    discount_factors: np.ndarray,
) -> CommodityEquityAutocallResult:
    """Equity autocall with commodity knock-out.
    Autocall if equity ≥ barrier AND commodity hasn't knocked out.
    Commodity KO: if commodity drops below ko_level at any point.
    """
    n_paths = equity_paths.shape[0]
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    eq0 = equity_paths[:, 0]
    # Track commodity KO
    comm_ko = np.any(commodity_paths <= commodity_ko_level, axis=1)
    for obs in observation_steps:
        eq_ratio = equity_paths[:, obs] / eq0
        triggered = alive & ~comm_ko & (eq_ratio >= eq_autocall_barrier)
        pv += np.where(triggered, (notional + coupon) * discount_factors[obs], 0.0)
        alive &= ~triggered
    # Terminal
    pv += np.where(alive, notional * discount_factors[-1], 0.0)
    return CommodityEquityAutocallResult(
        float(pv.mean()), float(1 - alive.mean()), float(comm_ko.mean()))


@dataclass
class DualRangeAccrualResult:
    price: float
    accrual_rate: float
    n_observations: int

def dual_asset_range_accrual(
    asset1_paths: np.ndarray, asset2_paths: np.ndarray,
    range1: tuple[float, float], range2: tuple[float, float],
    coupon_per_obs: float, discount_factor: float, notional: float = 1.0,
) -> DualRangeAccrualResult:
    """Accrues coupon for each observation where BOTH assets are in range."""
    n_paths, n_obs = asset1_paths.shape
    in1 = (asset1_paths[:, 1:] >= range1[0]) & (asset1_paths[:, 1:] <= range1[1])
    in2 = (asset2_paths[:, 1:] >= range2[0]) & (asset2_paths[:, 1:] <= range2[1])
    both = in1 & in2
    count = both.sum(axis=1)
    payoff = notional * coupon_per_obs * count
    price = float(discount_factor * payoff.mean())
    accrual = float(count.mean() / max(n_obs - 1, 1))
    return DualRangeAccrualResult(price, accrual, n_obs - 1)
