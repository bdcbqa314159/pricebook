"""Synthetic swaption vol data per currency.

Generates realistic ATM vol surfaces and smile data for testing and
calibration without external market data feeds.

    from pricebook.options.synthetic_swaption_data import (
        synthetic_atm_surface, synthetic_smile_data,
    )

    surface = synthetic_atm_surface("USD", ref_date)
"""

from __future__ import annotations

from datetime import date

import numpy as np

from pricebook.options.swaption_vol_cube import SwaptionVolCube, build_swaption_vol_cube


# ═══════════════════════════════════════════════════════════════
# Per-currency ATM vol levels (realistic as of 2024)
# ═══════════════════════════════════════════════════════════════

# Base ATM vol in decimal (e.g. 0.006 = 60bp Black vol)
_BASE_VOLS = {
    "USD": 0.0060,  # ~60bp
    "EUR": 0.0055,  # ~55bp (Bachelier would be ~55bp normal)
    "GBP": 0.0065,  # ~65bp
    "JPY": 0.0025,  # ~25bp (low rate → low Black vol)
    "CHF": 0.0040,  # ~40bp
    "CAD": 0.0055,  # ~55bp
    "AUD": 0.0060,  # ~60bp
    "BRL": 0.0200,  # ~200bp (EM, high rates)
    "MXN": 0.0150,  # ~150bp
    "KRW": 0.0050,  # ~50bp
    "ZAR": 0.0120,  # ~120bp
}

_EXPIRIES = [0.5, 1.0, 2.0, 5.0, 10.0]
_TENORS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


def synthetic_atm_surface(
    currency: str,
    reference_date: date,
) -> SwaptionVolCube:
    """Generate a realistic ATM swaption vol surface for a currency.

    Vol decreases with expiry (mean reversion) and tenor (averaging).
    """
    base = _BASE_VOLS.get(currency.upper(), 0.006)

    atm_vols = []
    for i, exp in enumerate(_EXPIRIES):
        row = []
        for j, ten in enumerate(_TENORS):
            # Vol decreases with expiry (HW mean reversion effect)
            exp_decay = 1.0 / (1 + 0.05 * exp)
            # Vol decreases with tenor (swap rate averaging)
            ten_decay = 1.0 / (1 + 0.02 * ten)
            vol = base * exp_decay * ten_decay
            row.append(vol)
        atm_vols.append(row)

    return build_swaption_vol_cube(reference_date, _EXPIRIES, _TENORS, atm_vols)


def synthetic_smile_data(
    currency: str,
    reference_date: date,
    forward: float = 0.04,
) -> dict:
    """Generate synthetic smile data (RR25, BF25) per (expiry, tenor) node.

    Returns dict suitable for `build_swaption_vol_cube(smile_data=...)`.
    """
    base = _BASE_VOLS.get(currency.upper(), 0.006)

    smile = {}
    for exp in [1.0, 5.0, 10.0]:
        for ten in [5.0, 10.0]:
            atm_vol = base / (1 + 0.05 * exp) / (1 + 0.02 * ten)
            rr25 = -0.15 * atm_vol  # 25D risk reversal (negative = put skew)
            bf25 = 0.05 * atm_vol   # 25D butterfly (smile convexity)

            k_atm = forward
            k_put = forward - 0.01
            k_call = forward + 0.01

            # Approximate smile vols from RR/BF
            vol_put = atm_vol + bf25 - 0.5 * rr25
            vol_call = atm_vol + bf25 + 0.5 * rr25

            smile[(exp, ten)] = {
                "forward": forward,
                "strikes": [k_put, k_atm, k_call],
                "vols": [vol_put, atm_vol, vol_call],
            }

    return smile


def synthetic_hw_targets(
    currency: str,
    reference_date: date,
) -> dict[tuple[float, float], float]:
    """Generate synthetic swaption vol targets for HW calibration.

    Returns dict {(expiry_years, tenor_years): atm_vol}.
    """
    base = _BASE_VOLS.get(currency.upper(), 0.006)

    targets = {}
    for exp in [1.0, 2.0, 5.0, 10.0]:
        for ten in [5.0, 10.0]:
            vol = base / (1 + 0.05 * exp) / (1 + 0.02 * ten)
            targets[(exp, ten)] = vol

    return targets
