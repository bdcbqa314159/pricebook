"""Tranche CDS pricing with base correlation calibration.

Standard tranches on credit indices (iTraxx, CDX). Uses Gaussian copula
for expected tranche loss calculation and base correlation calibration.

    from pricebook.tranche_pricing import TrancheCDS, calibrate_base_correlation

    tranche = TrancheCDS(attachment=0.0, detachment=0.03, maturity=date(2031,3,20))
    result = tranche.price(discount_curve, survival_curves, correlation=0.3)

    base_corr = calibrate_base_correlation(tranche_quotes, survival_curves, disc)

References:
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit
    Derivatives. Wiley, Ch. 10-11 — Tranches and Base Correlation.
    Li, D. (2000). On Default Correlation: A Copula Function Approach.
    McGinty, Beinstein et al. (2004). Credit Correlation: A Guide. JPMorgan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy.stats import norm

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.solvers import brentq
from pricebook.serialisable import _register, _serialise_atom


# Standard index tranches
STANDARD_TRANCHES = {
    "equity": (0.00, 0.03),
    "mezzanine_1": (0.03, 0.06),
    "mezzanine_2": (0.06, 0.09),
    "mezzanine_3": (0.09, 0.12),
    "senior": (0.12, 0.22),
    "super_senior": (0.22, 1.00),
}


@dataclass
class TrancheResult:
    """Tranche pricing result."""
    price: float                  # PV of the tranche
    expected_loss: float          # expected tranche loss at maturity
    protection_pv: float
    premium_pv: float
    par_spread: float             # spread that makes PV = 0
    attachment: float
    detachment: float

    def to_dict(self) -> dict:
        return {
            "price": self.price, "expected_loss": self.expected_loss,
            "protection_pv": self.protection_pv, "premium_pv": self.premium_pv,
            "par_spread": self.par_spread,
            "attachment": self.attachment, "detachment": self.detachment,
        }


def expected_tranche_loss(
    attachment: float,
    detachment: float,
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    rho: float,
    T: float,
    recovery: float = 0.4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Expected tranche loss via Gaussian copula MC.

    L_tranche = clip(L_portfolio - a, 0, d-a) / (d-a)

    where L_portfolio = (n_defaults / N) × (1-R).
    """
    width = detachment - attachment
    if width <= 0:
        return 0.0

    n_names = len(survival_curves)
    ref = discount_curve.reference_date

    rng = np.random.default_rng(seed)
    sqrt_rho = math.sqrt(max(rho, 0.0))
    sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))

    M = rng.standard_normal(n_sims)
    eps = rng.standard_normal((n_sims, n_names))
    Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

    T_date = ref + timedelta(days=int(T * 365))
    thresholds = np.array([
        norm.ppf(max(1.0 - sc.survival(T_date), 1e-15))
        for sc in survival_curves
    ])
    defaults = Z < thresholds[np.newaxis, :]
    n_defaults = defaults.sum(axis=1).astype(float)
    portfolio_loss = n_defaults / n_names * (1.0 - recovery)

    tranche_loss = np.clip(portfolio_loss - attachment, 0.0, width) / width
    return float(tranche_loss.mean())


class TrancheCDS:
    """CDS tranche — protection on a slice of index portfolio loss.

    Buyer pays premium on surviving tranche notional.
    Seller pays losses that hit the [attachment, detachment] band.

    Args:
        attachment: lower tranche bound (0.0 for equity).
        detachment: upper tranche bound (0.03 for equity).
        maturity: tranche maturity date.
        spread: running premium (decimal).
        notional: tranche notional.
        recovery: assumed uniform recovery.
        n_names: number of reference entities.
    """

    _SERIAL_TYPE = "tranche_cds"

    def __init__(
        self,
        attachment: float,
        detachment: float,
        maturity: date,
        spread: float = 0.05,
        notional: float = 10_000_000.0,
        recovery: float = 0.4,
        n_names: int = 125,
    ):
        if detachment <= attachment:
            raise ValueError(f"detachment ({detachment}) must be > attachment ({attachment})")
        self.attachment = attachment
        self.detachment = detachment
        self.maturity = maturity
        self.spread = spread
        self.notional = notional
        self.recovery = recovery
        self.n_names = n_names

    @property
    def width(self) -> float:
        return self.detachment - self.attachment

    @property
    def is_equity(self) -> bool:
        return self.attachment == 0.0

    def price(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
        correlation: float = 0.3,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> TrancheResult:
        """Price the tranche via Gaussian copula MC."""
        ref = discount_curve.reference_date
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        df = discount_curve.df(self.maturity)

        el = expected_tranche_loss(
            self.attachment, self.detachment, survival_curves,
            discount_curve, correlation, T, self.recovery, n_sims, seed,
        )

        # Protection leg: (1 - EL_start) → EL_end transition
        protection_pv = el * self.width * self.notional * df

        # Premium leg: spread × surviving notional × annuity
        # Approximate surviving = 1 - EL
        annuity = T * df  # simplified single-period
        premium_pv = self.spread * (1 - el) * self.notional * annuity

        pv = protection_pv - premium_pv

        # Par spread
        if (1 - el) * annuity > 1e-10:
            par_spread = (el * self.width * df) / ((1 - el) * annuity)
        else:
            par_spread = 0.0

        return TrancheResult(
            price=pv, expected_loss=el,
            protection_pv=protection_pv, premium_pv=premium_pv,
            par_spread=par_spread,
            attachment=self.attachment, detachment=self.detachment,
        )

    def pv_ctx(self, ctx) -> float:
        if ctx.credit_curves:
            scs = list(ctx.credit_curves.values())
            while len(scs) < self.n_names:
                scs.append(scs[-1])
            scs = scs[:self.n_names]
            return self.price(ctx.discount_curve, scs).price
        return 0.0

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "attachment": self.attachment, "detachment": self.detachment,
            "maturity": self.maturity.isoformat(), "spread": self.spread,
            "notional": self.notional, "recovery": self.recovery,
            "n_names": self.n_names,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> TrancheCDS:
        p = d["params"]
        return cls(
            attachment=p["attachment"], detachment=p["detachment"],
            maturity=date.fromisoformat(p["maturity"]),
            spread=p.get("spread", 0.05), notional=p.get("notional", 10_000_000.0),
            recovery=p.get("recovery", 0.4), n_names=p.get("n_names", 125),
        )


_register(TrancheCDS)


# ---- Base correlation calibration ----

def calibrate_base_correlation(
    tranche_quotes: dict[float, float],
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    maturity: date | None = None,
    recovery: float = 0.4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> dict[float, float]:
    """Calibrate base correlation from tranche market quotes.

    For each detachment point d, find the correlation ρ(d) such that
    the [0, d] tranche reprices at the market quote.

    Base correlation is monotonically increasing: equity tranche has
    lower correlation than senior (correlation skew).

    Args:
        tranche_quotes: {detachment_point: market_expected_loss}.
            E.g. {0.03: 0.45, 0.06: 0.12, 0.09: 0.05}
        survival_curves: N constituent survival curves.
        discount_curve: risk-free curve.
        maturity: tranche maturity (default: 5Y from ref).

    Returns:
        {detachment_point: base_correlation} — monotonically increasing.
    """
    ref = discount_curve.reference_date
    mat = maturity or (ref + timedelta(days=1825))
    T = year_fraction(ref, mat, DayCountConvention.ACT_365_FIXED)
    n_names = len(survival_curves)

    base_corr = {}
    for det_point in sorted(tranche_quotes.keys()):
        target_el = tranche_quotes[det_point]

        def objective(rho: float) -> float:
            el = expected_tranche_loss(
                0.0, det_point, survival_curves, discount_curve,
                rho, T, recovery, n_sims, seed,
            )
            return el - target_el

        try:
            rho = brentq(objective, 0.001, 0.999)
        except Exception:
            rho = 0.5  # fallback

        base_corr[det_point] = rho

    return base_corr


# ---------------------------------------------------------------------------
# Multi-period tranche pricing
# ---------------------------------------------------------------------------

def price_tranche_multiperiod(
    attachment: float,
    detachment: float,
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    correlation: float,
    spread: float,
    maturity_years: float = 5.0,
    recovery: float = 0.4,
    frequency: int = 4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> TrancheResult:
    """Multi-period tranche pricing with proper premium and protection legs.

    Premium leg: Σ spread × (1 - EL_i) × yf_i × df_i × notional
    Protection leg: Σ (EL_i - EL_{i-1}) × df_i × width × notional

    Computes expected tranche loss at each payment date.
    """
    width = detachment - attachment
    if width <= 0:
        return TrancheResult(0, 0, 0, 0, 0, attachment, detachment)

    ref = discount_curve.reference_date
    dt = 1.0 / frequency
    n_periods = int(maturity_years * frequency)

    # Compute EL at each payment date
    els = [0.0]  # EL at t=0
    for i in range(1, n_periods + 1):
        t = i * dt
        el = expected_tranche_loss(
            attachment, detachment, survival_curves,
            discount_curve, correlation, t, recovery, n_sims, seed,
        )
        els.append(el)

    def _date_at(t_years: float) -> date:
        return ref + timedelta(days=round(t_years * 365))

    # Premium leg
    premium_pv = 0.0
    for i in range(1, n_periods + 1):
        t = i * dt
        df = discount_curve.df(_date_at(t))
        surviving = 1.0 - els[i]
        premium_pv += spread * surviving * dt * df

    # Protection leg
    protection_pv = 0.0
    for i in range(1, n_periods + 1):
        t = i * dt
        df = discount_curve.df(_date_at(t))
        delta_el = els[i] - els[i - 1]
        protection_pv += delta_el * width * df

    pv = protection_pv - premium_pv

    # Par spread
    annuity = sum(
        (1.0 - els[i]) * dt * discount_curve.df(_date_at(i * dt))
        for i in range(1, n_periods + 1)
    )
    par_spread = protection_pv / annuity if annuity > 1e-10 else 0.0

    return TrancheResult(
        price=pv, expected_loss=els[-1],
        protection_pv=protection_pv, premium_pv=premium_pv,
        par_spread=par_spread,
        attachment=attachment, detachment=detachment,
    )


# ---------------------------------------------------------------------------
# Student-t copula for tranche loss
# ---------------------------------------------------------------------------

def expected_tranche_loss_t(
    attachment: float,
    detachment: float,
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    rho: float,
    T: float,
    nu: float = 5.0,
    recovery: float = 0.4,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Expected tranche loss via Student-t copula MC.

    Same structure as Gaussian copula but with t-distributed factors.
    Captures tail dependence: more clustered defaults under stress.

    As ν → ∞, converges to Gaussian copula.

    Args:
        nu: degrees of freedom (lower = fatter tails).
    """
    from scipy.stats import t as t_dist

    width = detachment - attachment
    if width <= 0:
        return 0.0

    n_names = len(survival_curves)
    ref = discount_curve.reference_date

    rng = np.random.default_rng(seed)
    sqrt_rho = math.sqrt(max(rho, 0.0))
    sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))

    # t-distributed systematic and idiosyncratic factors
    # Chi-squared mixing variable (clamped for stability)
    chi2 = np.maximum(rng.chisquare(nu, n_sims), 0.01)
    W = np.sqrt(nu / chi2)  # mixing variable

    M = rng.standard_normal(n_sims) * W
    eps = rng.standard_normal((n_sims, n_names)) * W[:, np.newaxis]
    Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

    T_date = ref + timedelta(days=int(T * 365))
    # Use t-distribution CDF for thresholds
    thresholds = np.array([
        t_dist.ppf(max(1.0 - sc.survival(T_date), 1e-15), nu)
        for sc in survival_curves
    ])
    defaults = Z < thresholds[np.newaxis, :]
    n_defaults = defaults.sum(axis=1).astype(float)
    portfolio_loss = n_defaults / n_names * (1.0 - recovery)

    tranche_loss = np.clip(portfolio_loss - attachment, 0.0, width) / width
    return float(tranche_loss.mean())


# ---------------------------------------------------------------------------
# Correlation sensitivity (rho01)
# ---------------------------------------------------------------------------

def tranche_rho01(
    attachment: float,
    detachment: float,
    survival_curves: list[SurvivalCurve],
    discount_curve: DiscountCurve,
    correlation: float,
    T: float,
    recovery: float = 0.4,
    shift: float = 0.01,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Tranche sensitivity to correlation: dEL/dρ per 1%.

    Equity: rho01 < 0 (higher ρ reduces equity tranche loss).
    Senior: rho01 > 0 (higher ρ increases senior tranche loss).
    """
    el_base = expected_tranche_loss(
        attachment, detachment, survival_curves,
        discount_curve, correlation, T, recovery, n_sims, seed,
    )
    rho_bumped = min(correlation + shift, 0.999)
    el_bumped = expected_tranche_loss(
        attachment, detachment, survival_curves,
        discount_curve, rho_bumped, T, recovery, n_sims, seed,
    )
    return (el_bumped - el_base) / shift


# ---------------------------------------------------------------------------
# Base correlation interpolation
# ---------------------------------------------------------------------------

def interpolate_base_correlation(
    calibrated: dict[float, float],
    detachment: float,
) -> float:
    """Interpolate base correlation at arbitrary detachment point.

    Linear interpolation between calibrated detachment points.
    Extrapolates flat beyond the calibrated range.

    Args:
        calibrated: {detachment_point: base_correlation}.
        detachment: target detachment point.

    Returns:
        Interpolated base correlation.
    """
    if not calibrated:
        return 0.3  # default

    points = sorted(calibrated.items())
    dets = [p[0] for p in points]
    corrs = [p[1] for p in points]

    if detachment <= dets[0]:
        return corrs[0]
    if detachment >= dets[-1]:
        return corrs[-1]

    # Linear interpolation
    for i in range(len(dets) - 1):
        if dets[i] <= detachment <= dets[i + 1]:
            frac = (detachment - dets[i]) / (dets[i + 1] - dets[i])
            return corrs[i] + frac * (corrs[i + 1] - corrs[i])

    return corrs[-1]
