"""Hull-White calibration per currency.

Integrates curve building, synthetic vol data, and HW calibration
into a single per-currency entry point.

    from pricebook.models.hw_per_currency import calibrate_hw_for_currency

    hw = calibrate_hw_for_currency("USD", ref_date, curve)

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hull_white import HullWhite
from pricebook.models.hw_calibration import calibrate_hull_white, HWCalibrationResult


# Per-currency default HW parameters (used as initial guess / fallback)
_DEFAULTS = {
    # G10: moderate mean reversion, low vol
    "USD": (0.03, 0.010),
    "EUR": (0.03, 0.008),
    "GBP": (0.03, 0.012),
    "JPY": (0.01, 0.005),   # very low rates → lower vol
    "CHF": (0.02, 0.006),   # negative rate environment
    "CAD": (0.03, 0.010),
    "AUD": (0.03, 0.010),
    "NZD": (0.03, 0.010),
    # Nordics
    "SEK": (0.03, 0.008),
    "NOK": (0.03, 0.009),
    "DKK": (0.03, 0.007),   # pegged to EUR
    # EM: higher mean reversion, higher vol
    "BRL": (0.10, 0.030),
    "MXN": (0.08, 0.025),
    "CLP": (0.06, 0.020),
    "COP": (0.06, 0.020),
    "TRY": (0.15, 0.050),   # extreme rates
    "ARS": (0.20, 0.100),   # extreme rates
    "ZAR": (0.06, 0.020),
    # Asia
    "CNY": (0.03, 0.008),
    "KRW": (0.04, 0.010),
    "INR": (0.05, 0.015),
    "SGD": (0.03, 0.008),
    "HKD": (0.03, 0.010),   # pegged to USD
    "THB": (0.04, 0.010),
    "IDR": (0.05, 0.015),
    "MYR": (0.04, 0.010),
    "PHP": (0.05, 0.015),
    # CEE
    "PLN": (0.05, 0.015),
    "CZK": (0.04, 0.012),
    "HUF": (0.06, 0.018),
    # Other
    "ILS": (0.04, 0.012),
    "PEN": (0.05, 0.015),
}


@dataclass
class HWPerCurrencyResult:
    """Result of per-currency HW calibration."""
    model: HullWhite
    currency: str
    a: float
    sigma: float
    calibration: HWCalibrationResult | None
    source: str                # "calibrated" or "default"

    def to_dict(self) -> dict:
        return {"currency": self.currency, "a": self.a, "sigma": self.sigma,
                "source": self.source}


def calibrate_hw_for_currency(
    currency: str,
    reference_date: date,
    curve: DiscountCurve,
    swaption_vols: dict[tuple[float, float], float] | None = None,
    use_synthetic: bool = True,
    n_steps: int = 50,
) -> HWPerCurrencyResult:
    """Calibrate Hull-White for a specific currency.

    If swaption_vols provided, calibrates to market data.
    If use_synthetic=True and no vols, uses synthetic vol surface.
    Otherwise, uses per-currency default parameters.

    Args:
        currency: ISO currency code.
        reference_date: valuation date.
        curve: discount curve for this currency.
        swaption_vols: optional market swaption vols.
        use_synthetic: if True, use synthetic data when market data unavailable.
        n_steps: tree steps for calibration.

    Returns:
        HWPerCurrencyResult with calibrated or default HW model.
    """
    ccy = currency.upper()

    # Step 1: Get vol targets
    if swaption_vols is not None:
        targets = swaption_vols
        source = "calibrated"
    elif use_synthetic:
        from pricebook.options.synthetic_swaption_data import synthetic_hw_targets
        targets = synthetic_hw_targets(ccy, reference_date)
        source = "calibrated_synthetic"
    else:
        targets = None
        source = "default"

    # Step 2: Calibrate or use defaults
    if targets:
        try:
            result = calibrate_hull_white(curve, targets, n_steps=n_steps)
            return HWPerCurrencyResult(
                result.model, ccy, result.a, result.sigma,
                result, source,
            )
        except Exception:
            pass  # fall through to defaults

    # Step 3: Defaults
    a_def, sig_def = _DEFAULTS.get(ccy, (0.03, 0.01))
    hw = HullWhite(a=a_def, sigma=sig_def, curve=curve)
    return HWPerCurrencyResult(hw, ccy, a_def, sig_def, None, "default")


def list_hw_currencies() -> list[str]:
    """Return currencies with HW default parameters."""
    return sorted(_DEFAULTS.keys())
