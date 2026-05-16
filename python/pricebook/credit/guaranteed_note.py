"""Guaranteed notes: third-party credit enhancement via joint default modelling.

A guaranteed note is a bond issued by entity A, with a guarantee from
entity B (parent, bank, monoline, sovereign). The investor only loses
if BOTH A and B default — the guarantee provides credit enhancement.

Pricing uses bivariate Gaussian copula for joint default:
    PD_joint(t) = Φ₂(Φ⁻¹(PD_A(t)), Φ⁻¹(PD_B(t)), ρ)

The note defaults only when both issuer and guarantor default.
The effective survival probability is:
    Q_eff(t) = 1 - PD_A(t) - PD_B(t) + PD_joint(t)

    from pricebook.guaranteed_note import (
        GuaranteedNote, guarantee_value, guarantee_spread,
    )

References:
    Hull & White (2001). Valuing Credit Default Swaps II.
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
    Li (2000). On Default Correlation: A Copula Function Approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm, multivariate_normal

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.schedule import Frequency, generate_schedule


def _bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    """Bivariate normal CDF Φ₂(x, y, ρ) via Drezner-Wesolowsky approximation.

    For the special cases ρ=0, ±1, uses exact formulas.
    General case uses the Drezner (1978) approximation with Gauss-Legendre.
    """
    if abs(rho) < 1e-10:
        return norm.cdf(x) * norm.cdf(y)
    if rho > 1 - 1e-10:
        return norm.cdf(min(x, y))
    if rho < -1 + 1e-10:
        return max(norm.cdf(x) + norm.cdf(y) - 1.0, 0.0)

    return multivariate_normal.cdf([x, y], mean=[0, 0], cov=[[1, rho], [rho, 1]])


@dataclass
class GuaranteedNoteResult:
    """Pricing result for a guaranteed note."""
    pv: float                    # full PV (guaranteed)
    pv_unguaranteed: float       # PV without guarantee (issuer only)
    guarantee_value: float       # PV_guaranteed - PV_unguaranteed
    guarantee_spread_bp: float   # implied spread benefit from guarantee
    effective_spread: float      # yield spread of guaranteed note

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "pv_unguaranteed": self.pv_unguaranteed,
            "guarantee_value": self.guarantee_value,
            "guarantee_spread_bp": self.guarantee_spread_bp,
            "effective_spread": self.effective_spread,
        }


class GuaranteedNote:
    """Bond with third-party credit guarantee.

    The note defaults only when BOTH issuer and guarantor default.
    Uses bivariate Gaussian copula for joint default probability.

    Args:
        start: issue date.
        end: maturity date.
        coupon_rate: annual coupon rate.
        notional: face value.
        recovery_issuer: recovery rate if issuer defaults (guarantor saves).
        recovery_guarantor: recovery rate if guarantor defaults (issuer saves).
        recovery_joint: recovery rate if both default.
        correlation: default correlation ρ between issuer and guarantor.
        frequency: coupon frequency.
        day_count: accrual convention.
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float,
        notional: float = 100.0,
        recovery_issuer: float = 0.40,
        recovery_guarantor: float = 0.40,
        recovery_joint: float = 0.20,
        correlation: float = 0.30,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    ):
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if not -1 <= correlation <= 1:
            raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
        for name, val in [("recovery_issuer", recovery_issuer),
                          ("recovery_guarantor", recovery_guarantor),
                          ("recovery_joint", recovery_joint)]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if recovery_joint > min(recovery_issuer, recovery_guarantor):
            raise ValueError(
                f"recovery_joint ({recovery_joint}) cannot exceed "
                f"min(recovery_issuer, recovery_guarantor) = "
                f"{min(recovery_issuer, recovery_guarantor)}"
            )

        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.recovery_issuer = recovery_issuer
        self.recovery_guarantor = recovery_guarantor
        self.recovery_joint = recovery_joint
        self.correlation = correlation
        self.frequency = frequency
        self.day_count = day_count
        self.schedule = generate_schedule(start, end, frequency)

    def _joint_survival(
        self,
        t: date,
        issuer_surv: SurvivalCurve,
        guarantor_surv: SurvivalCurve,
    ) -> float:
        """Probability that the note has NOT defaulted by time t.

        Note survives if at least one of {issuer, guarantor} survives.
        P(note survives) = 1 - P(both default)
                         = 1 - Φ₂(Φ⁻¹(PD_A), Φ⁻¹(PD_B), ρ)
        """

        pd_a = 1 - issuer_surv.survival(t)
        pd_b = 1 - guarantor_surv.survival(t)

        # Clamp to avoid norm.ppf(0) or norm.ppf(1)
        pd_a = max(min(pd_a, 1 - 1e-10), 1e-10)
        pd_b = max(min(pd_b, 1 - 1e-10), 1e-10)

        pd_joint = _bivariate_normal_cdf(
            norm.ppf(pd_a), norm.ppf(pd_b), self.correlation,
        )

        return 1.0 - pd_joint

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        issuer_surv: SurvivalCurve,
        guarantor_surv: SurvivalCurve,
    ) -> float:
        """Full price of the guaranteed note.

        PV = Σ coupon_i × df_i × Q_eff(t_i)
           + principal × df_T × Q_eff(T)
           + recovery terms for default scenarios
        """

        pv = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)

            q_eff = self._joint_survival(t_end, issuer_surv, guarantor_surv)
            q_eff_prev = self._joint_survival(t_start, issuer_surv, guarantor_surv)
            default_prob = q_eff_prev - q_eff

            # Coupon: paid if note survives
            pv += self.notional * self.coupon_rate * yf * df * q_eff

            # Recovery on default: weighted by joint recovery
            pv += self.recovery_joint * self.notional * default_prob * df

        # Principal at maturity
        q_final = self._joint_survival(self.end, issuer_surv, guarantor_surv)
        pv += self.notional * discount_curve.df(self.end) * q_final

        return pv

    def price(
        self,
        discount_curve: DiscountCurve,
        issuer_surv: SurvivalCurve,
        guarantor_surv: SurvivalCurve,
    ) -> GuaranteedNoteResult:
        """Full pricing with guarantee decomposition."""
        # Guaranteed PV
        pv_guaranteed = self.dirty_price(discount_curve, issuer_surv, guarantor_surv)

        # Unguaranteed PV (issuer-only credit risk)
        pv_unguar = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = issuer_surv.survival(t_end)
            surv_prev = issuer_surv.survival(t_start)
            default_prob = surv_prev - surv

            pv_unguar += self.notional * self.coupon_rate * yf * df * surv
            pv_unguar += self.recovery_issuer * self.notional * default_prob * df

        pv_unguar += self.notional * discount_curve.df(self.end) * issuer_surv.survival(self.end)

        guarantee_val = pv_guaranteed - pv_unguar

        # Risk-free annuity for spread calculation
        annuity = sum(
            year_fraction(self.schedule[i-1], self.schedule[i], self.day_count)
            * discount_curve.df(self.schedule[i])
            for i in range(1, len(self.schedule))
        )
        guarantee_spread_bp = (guarantee_val / (annuity * self.notional) * 10_000
                               if annuity > 1e-15 else 0.0)

        # Effective spread: risk-free PV - guaranteed PV → implied spread
        rf_pv = sum(
            self.notional * self.coupon_rate
            * year_fraction(self.schedule[i-1], self.schedule[i], self.day_count)
            * discount_curve.df(self.schedule[i])
            for i in range(1, len(self.schedule))
        ) + self.notional * discount_curve.df(self.end)

        eff_spread = (rf_pv - pv_guaranteed) / (annuity * self.notional) if annuity > 1e-15 else 0.0

        return GuaranteedNoteResult(
            pv=pv_guaranteed,
            pv_unguaranteed=pv_unguar,
            guarantee_value=guarantee_val,
            guarantee_spread_bp=guarantee_spread_bp,
            effective_spread=eff_spread,
        )

    def cs01_issuer(
        self, discount_curve, issuer_surv, guarantor_surv, shift=0.0001,
    ) -> float:
        """Issuer credit sensitivity: PV change per 1bp issuer hazard shift."""
        pv_up = self.dirty_price(discount_curve, issuer_surv.bumped(shift), guarantor_surv)
        pv_dn = self.dirty_price(discount_curve, issuer_surv.bumped(-shift), guarantor_surv)
        return (pv_up - pv_dn) / 2

    def cs01_guarantor(
        self, discount_curve, issuer_surv, guarantor_surv, shift=0.0001,
    ) -> float:
        """Guarantor credit sensitivity: PV change per 1bp guarantor hazard shift."""
        pv_up = self.dirty_price(discount_curve, issuer_surv, guarantor_surv.bumped(shift))
        pv_dn = self.dirty_price(discount_curve, issuer_surv, guarantor_surv.bumped(-shift))
        return (pv_up - pv_dn) / 2

    def rho01(
        self, discount_curve, issuer_surv, guarantor_surv, shift=0.01,
    ) -> float:
        """Correlation sensitivity: PV change per 1% correlation shift."""
        up = GuaranteedNote(
            self.start, self.end, self.coupon_rate, self.notional,
            self.recovery_issuer, self.recovery_guarantor, self.recovery_joint,
            min(self.correlation + shift, 0.999),
            self.frequency, self.day_count,
        )
        dn = GuaranteedNote(
            self.start, self.end, self.coupon_rate, self.notional,
            self.recovery_issuer, self.recovery_guarantor, self.recovery_joint,
            max(self.correlation - shift, -0.999),
            self.frequency, self.day_count,
        )
        return (up.dirty_price(discount_curve, issuer_surv, guarantor_surv)
                - dn.dirty_price(discount_curve, issuer_surv, guarantor_surv)) / 2


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def guarantee_value(
    coupon_rate: float,
    maturity: date,
    notional: float,
    discount_curve: DiscountCurve,
    issuer_surv: SurvivalCurve,
    guarantor_surv: SurvivalCurve,
    correlation: float = 0.30,
    recovery_joint: float = 0.20,
) -> float:
    """Value of a guarantee: PV_guaranteed - PV_unguaranteed."""
    gn = GuaranteedNote(
        discount_curve.reference_date, maturity, coupon_rate, notional,
        correlation=correlation, recovery_joint=recovery_joint,
    )
    result = gn.price(discount_curve, issuer_surv, guarantor_surv)
    return result.guarantee_value


def guarantee_spread(
    coupon_rate: float,
    maturity: date,
    notional: float,
    discount_curve: DiscountCurve,
    issuer_surv: SurvivalCurve,
    guarantor_surv: SurvivalCurve,
    correlation: float = 0.30,
) -> float:
    """Implied spread benefit from a guarantee (in bp)."""
    gn = GuaranteedNote(
        discount_curve.reference_date, maturity, coupon_rate, notional,
        correlation=correlation,
    )
    result = gn.price(discount_curve, issuer_surv, guarantor_surv)
    return result.guarantee_spread_bp
