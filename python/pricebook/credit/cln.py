"""Unified Credit-Linked Note pricing.

Wraps existing CLN implementations (VanillaCLN, LeveragedCLN, FloatingCLN)
under a single ``CreditLinkedNote`` class with Trade/Portfolio integration.

Also provides ``BasketCLN`` for multi-name structures using Gaussian copula.

    from pricebook.cln import CreditLinkedNote, BasketCLN

    cln = CreditLinkedNote(start, end, coupon_rate=0.06,
                           notional=1_000_000, recovery=0.4)
    price = cln.dirty_price(discount_curve, survival_curve)

    basket = BasketCLN(start, end, notional=10_000_000,
                       attachment=0.0, detachment=0.03)
    price = basket.price_mc(discount_curve, survival_curves, rho=0.3)

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Wiley, 2008, Ch. 15-16.
    Li, D. (2000). On Default Correlation: A Copula Function Approach.
    J. Fixed Income, 9(4), 43-54.  (Gaussian copula for BasketCLN)

Equations:
    Vanilla CLN PV (O'Kane Eq 15.1):
        PV = Σ_i c × α_i × D(t_i) × Q(t_i)
           + N × D(T) × Q(T)
           + R × N × Σ_i D(t_i) × [Q(t_{i-1}) - Q(t_i)]

    Leveraged CLN: adds (L-1)(1-R) × N × ΔPD per period (O'Kane §16.2).

    Basket CLN tranche loss (Li 2000, O'Kane §18.3):
        Z_j = √ρ M + √(1-ρ) ε_j,  default if Z_j < Φ⁻¹(1 - Q_j(t))
        L_tranche = clip(L_portfolio - a, 0, d-a) / (d-a)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import norm

from pricebook.core.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.notional import normalize_notional as _normalize_notional
from pricebook.core.schedule import Frequency, generate_schedule
from pricebook.core.survival_curve import SurvivalCurve


@dataclass
class CLNResult:
    """Credit-linked note pricing result."""
    price: float          # dirty price (currency)
    coupon_pv: float
    principal_pv: float
    recovery_pv: float
    credit_spread: float  # implied spread over risk-free
    par_coupon: float     # coupon that makes price = notional

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "coupon_pv": self.coupon_pv,
            "principal_pv": self.principal_pv,
            "recovery_pv": self.recovery_pv,
            "credit_spread": self.credit_spread,
            "par_coupon": self.par_coupon,
        }


class CreditLinkedNote:
    """Unified credit-linked note — funded credit exposure.

    The investor funds the principal upfront. In return:
    - Receives coupon = (risk-free + credit_spread) each period
    - At maturity: receives par if no default, recovery if default

    Supports:
    - Vanilla (leverage=1.0): standard funded CLN
    - Leveraged (leverage>1.0): amplified credit exposure
    - Floating coupon via ``floating=True``: coupon resets to forward + spread

    Args:
        start: issue date.
        end: maturity.
        coupon_rate: annual coupon rate (fixed) or spread over floating.
        notional: face value / funded amount.
        recovery: recovery rate on default.
        leverage: credit exposure multiplier (1.0 = vanilla).
        floating: if True, coupon = forward_rate + coupon_rate.
        frequency: coupon frequency.
        day_count: accrual convention.
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float = 0.06,
        notional: float | list[float] = 1_000_000.0,
        recovery: float = 0.4,
        leverage: float = 1.0,
        floating: bool = False,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if not 0.0 <= recovery <= 1.0:
            raise ValueError(f"recovery must be in [0, 1], got {recovery}")
        if leverage < 1.0:
            raise ValueError(f"leverage must be >= 1.0, got {leverage}")
        if start >= end:
            raise ValueError(f"end ({end}) must be after start ({start})")
        if isinstance(notional, (int, float)) and notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if isinstance(notional, list) and any(n <= 0 for n in notional):
            raise ValueError("all notional schedule values must be positive")
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.recovery = recovery
        self.leverage = leverage
        self.floating = floating
        self.frequency = frequency
        self.day_count = day_count
        self.schedule = generate_schedule(start, end, frequency)

        n_periods = len(self.schedule) - 1
        self.notional_schedule = _normalize_notional(notional, n_periods)
        self.notional = self.notional_schedule[0]

    @property
    def average_notional(self) -> float:
        """Arithmetic mean of the notional schedule."""
        return sum(self.notional_schedule) / len(self.notional_schedule)

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Full price of the CLN accounting for default risk (O'Kane Eq 15.1).

        PV = Σ c × α_i × D(t_i) × Q(t_i)          [coupon leg]
           + N × D(T) × Q(T)                       [principal leg]
           + R × N × Σ D(t_i) × ΔPD_i              [recovery leg]
           - (L-1)(1-R) × N × Σ D(t_i) × ΔPD_i    [leverage loss, O'Kane §16.2]
        """
        pv = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv
            n_i = self.notional_schedule[i - 1]

            # Coupon: fixed or floating
            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = n_i * (fwd + self.coupon_rate) * yf
            else:
                coupon = n_i * self.coupon_rate * yf

            pv += coupon * df * surv

            # Recovery on default (investor gets recovery × notional)
            pv += self.recovery * n_i * default_prob * df

            # Leveraged loss: extra (leverage - 1) × (1-R) × notional on default
            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * n_i
                pv -= extra_loss * default_prob * df

        # Principal at maturity (conditional on survival) — uses final period notional
        pv += self.notional_schedule[-1] * discount_curve.df(self.end) * survival_curve.survival(self.end)

        return pv

    def price_per_100(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Price per 100 face."""
        return self.dirty_price(discount_curve, survival_curve) / self.notional * 100.0

    def price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> CLNResult:
        """Full pricing with decomposition."""
        coupon_pv = 0.0
        recovery_pv = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv
            n_i = self.notional_schedule[i - 1]

            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = n_i * (fwd + self.coupon_rate) * yf
            else:
                coupon = n_i * self.coupon_rate * yf

            coupon_pv += coupon * df * surv
            recovery_pv += self.recovery * n_i * default_prob * df

            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * n_i
                recovery_pv -= extra_loss * default_prob * df

        principal_pv = self.notional_schedule[-1] * discount_curve.df(self.end) * survival_curve.survival(self.end)
        total_pv = coupon_pv + principal_pv + recovery_pv

        # Credit spread: implied spread over risk-free
        riskfree_pv = self._risk_free_pv(discount_curve)
        annuity = self._risky_annuity(discount_curve, survival_curve)
        credit_spread = (riskfree_pv - total_pv) / (self.notional * annuity) if annuity > 0 else 0.0

        # Par coupon
        par_coupon = self._par_coupon(discount_curve, survival_curve)

        return CLNResult(
            price=total_pv,
            coupon_pv=coupon_pv,
            principal_pv=principal_pv,
            recovery_pv=recovery_pv,
            credit_spread=credit_spread,
            par_coupon=par_coupon,
        )

    def breakeven_spread(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Spread that makes CLN price = notional (par)."""
        from pricebook.core.solvers import brentq

        def objective(s: float) -> float:
            shifted = CreditLinkedNote(
                self.start, self.end, coupon_rate=s,
                notional=self.notional, recovery=self.recovery,
                leverage=self.leverage, floating=self.floating,
                frequency=self.frequency, day_count=self.day_count,
            )
            return shifted.dirty_price(discount_curve, survival_curve) - self.notional

        return brentq(objective, -0.10, 0.50)

    def greeks(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> dict[str, float]:
        """Bump-and-reprice sensitivities (immutable — no state mutation).

        Returns:
            dv01: price change for +1bp parallel rate shift.
            cs01: price change for +1bp credit spread (hazard) shift.
            recovery_sens: price change for +1% recovery shift.
        """
        base = self.dirty_price(discount_curve, survival_curve)

        # DV01: bump discount curve by 1bp
        bumped_curve = discount_curve.bumped(0.0001)
        dv01 = self.dirty_price(bumped_curve, survival_curve) - base

        # CS01: parallel bump of hazard rates by +1bp
        shifted_surv = survival_curve.bumped(0.0001)
        cs01 = self.dirty_price(discount_curve, shifted_surv) - base

        # Recovery sensitivity: +1% bump via copy (no mutation)
        cln_bumped = CreditLinkedNote(
            self.start, self.end, self.coupon_rate,
            self.notional_schedule, self.recovery + 0.01,
            self.leverage, self.floating, self.frequency, self.day_count,
        )
        recovery_sens = cln_bumped.dirty_price(discount_curve, survival_curve) - base

        return {"dv01": dv01, "cs01": cs01, "recovery_sensitivity": recovery_sens}

    def pv_ctx(self, ctx) -> float:
        """Price from PricingContext — Trade/Portfolio integration.

        If ctx has a stochastic credit model matching a credit curve name,
        uses price_stochastic_intensity() for MC pricing. Otherwise deterministic.
        """
        curve = ctx.discount_curve

        # Find survival curve from credit_curves dict
        sc = None
        if hasattr(ctx, 'credit_curves') and ctx.credit_curves:
            sc = next(iter(ctx.credit_curves.values()), None)
        if sc is None and hasattr(ctx, "survival_curve") and ctx.survival_curve:
            sc = ctx.survival_curve
        if sc is None:
            sc = SurvivalCurve.flat(curve.reference_date, 0.02)

        # Check for stochastic model
        if hasattr(ctx, 'stochastic_credit_models') and ctx.stochastic_credit_models:
            model = next(iter(ctx.stochastic_credit_models.values()), None)
            if model is not None:
                return self.price_stochastic_intensity(
                    curve, model, n_paths=10_000, n_steps=50,
                ).price

        return self.dirty_price(curve, sc)

    # ---- Internal helpers ----

    def _risk_free_pv(self, discount_curve: DiscountCurve) -> float:
        pv = 0.0
        for i in range(1, len(self.schedule)):
            yf = year_fraction(self.schedule[i-1], self.schedule[i], self.day_count)
            pv += self.notional * self.coupon_rate * yf * discount_curve.df(self.schedule[i])
        pv += self.notional * discount_curve.df(self.end)
        return pv

    def decompose_bond_cds(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> dict[str, float]:
        """Decompose CLN into bond + embedded CDS components.

        CLN = Risk-free bond - L × CDS protection value

        For vanilla (L=1): CLN = Bond - CDS_protection
        For leveraged (L>1): CLN = Bond - L × CDS_protection

        This verifies the fundamental relationship:
        dirty_price = risk_free_pv - L × protection_pv + recovery_adjustment
        """
        from pricebook.cds import protection_leg_pv

        rf_pv = self._risk_free_pv(discount_curve)
        cln_pv = self.dirty_price(discount_curve, survival_curve)

        # CDS protection leg (per unit notional)
        prot_pv = protection_leg_pv(
            self.start, self.end, discount_curve, survival_curve,
            recovery=self.recovery, notional=self.notional,
        )

        # Embedded CDS value = protection_pv (what you'd receive as protection buyer)
        # CLN holder is SHORT protection (they bear the credit risk)
        # So CLN = bond - protection_pv × leverage_factor
        embedded_cds = prot_pv * self.leverage

        # The difference between rf_pv - embedded_cds and actual CLN price
        # should be small (only differs by recovery leg treatment)
        synthetic_cln = rf_pv - embedded_cds
        basis = cln_pv - synthetic_cln

        return {
            "cln_price": cln_pv,
            "risk_free_bond": rf_pv,
            "embedded_cds_protection": embedded_cds,
            "synthetic_cln": synthetic_cln,
            "basis": basis,
            "leverage": self.leverage,
            "credit_discount": rf_pv - cln_pv,
        }

    def _risky_annuity(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        annuity = 0.0
        for i in range(1, len(self.schedule)):
            yf = year_fraction(self.schedule[i-1], self.schedule[i], self.day_count)
            annuity += yf * discount_curve.df(self.schedule[i]) * survival_curve.survival(self.schedule[i])
        return annuity

    def _par_coupon(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Coupon rate that makes price = notional."""
        # PV = coupon × annuity + principal_pv + recovery_pv = notional
        # → coupon = (notional - principal_pv - recovery_pv) / (notional × annuity)
        principal_pv = self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)
        recovery_pv = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv
            recovery_pv += self.recovery * self.notional * default_prob * df
            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * self.notional
                recovery_pv -= extra_loss * default_prob * df

        annuity = self._risky_annuity(discount_curve, survival_curve)
        if annuity <= 0:
            return 0.0
        return (self.notional - principal_pv - recovery_pv) / (self.notional * annuity)

    # ---- Stochastic / MC pricing (delegated to cln_mc.py) ----

    def price_stochastic_recovery(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        recovery_spec=None,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> CLNResult:
        """Price CLN with stochastic recovery correlated to default."""
        from pricebook.cln_mc import cln_stochastic_recovery
        return cln_stochastic_recovery(
            self, discount_curve, survival_curve, recovery_spec, n_sims, seed)

    def price_stochastic_intensity(
        self,
        discount_curve: DiscountCurve,
        intensity_model,
        x0: float | None = None,
        n_paths: int = 50_000,
        n_steps: int = 200,
        seed: int = 42,
    ) -> CLNResult:
        """Price CLN via Monte Carlo under stochastic intensity."""
        from pricebook.cln_mc import cln_stochastic_intensity
        return cln_stochastic_intensity(
            self, discount_curve, intensity_model, x0, n_paths, n_steps, seed)

    def price_stochastic_intensity_from_curve(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        model_type: str = "cir++",
        xi: float = 0.1,
        kappa: float = 1.0,
        n_paths: int = 50_000,
        n_steps: int = 200,
        seed: int = 42,
        **model_kwargs,
    ) -> CLNResult:
        """Auto-calibrate stochastic model from survival curve, then price."""
        from pricebook.cln_mc import cln_stochastic_intensity_from_curve
        return cln_stochastic_intensity_from_curve(
            self, discount_curve, survival_curve, model_type, xi, kappa,
            n_paths, n_steps, seed, **model_kwargs)

    def price_bilateral_mc(
        self,
        discount_curve: DiscountCurve,
        ref_survival: SurvivalCurve,
        issuer_survival: SurvivalCurve,
        issuer_recovery: float = 0.4,
        correlation: float = 0.3,
        n_paths: int = 50_000,
        seed: int = 42,
    ) -> CLNResult:
        """Price bilateral CLN with dual default risk (ref entity + issuer)."""
        from pricebook.cln_mc import cln_bilateral_mc
        return cln_bilateral_mc(
            self, discount_curve, ref_survival, issuer_survival,
            issuer_recovery, correlation, n_paths, seed)

    def rec_vol_01(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        recovery_spec=None,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> float:
        """Recovery vol sensitivity: PV change for 1% increase in recovery std."""
        from pricebook.cln_mc import cln_rec_vol_01
        return cln_rec_vol_01(
            self, discount_curve, survival_curve, recovery_spec, n_sims, seed)

    @classmethod
    def from_seniority(
        cls,
        start: date,
        end: date,
        seniority: str,
        coupon_rate: float = 0.06,
        notional: float = 1_000_000.0,
        leverage: float = 1.0,
        **kwargs,
    ) -> CreditLinkedNote:
        """Build CLN with seniority-appropriate recovery.

        1L: recovery ≈ 77%. 2L: ≈ 43%. Sub: ≈ 28%.
        """
        from pricebook.recovery_pricing import SENIORITY_RECOVERY
        if seniority not in SENIORITY_RECOVERY:
            raise ValueError(f"Unknown seniority '{seniority}'")
        mean, _ = SENIORITY_RECOVERY[seniority]
        return cls(
            start=start, end=end, coupon_rate=coupon_rate,
            notional=notional, recovery=mean, leverage=leverage,
            **kwargs,
        )


@dataclass
class BasketCLNResult:
    """Basket CLN pricing result."""
    price: float
    expected_loss: float
    tranche_width: float
    attachment: float
    detachment: float
    std_error: float = 0.0  # MC standard error on expected_loss

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "expected_loss": self.expected_loss,
            "tranche_width": self.tranche_width,
            "attachment": self.attachment,
            "detachment": self.detachment,
            "std_error": self.std_error,
        }


class BasketCLN:
    """Multi-name credit-linked note with tranche structure.

    The investor takes credit risk on a basket of names via a tranche
    defined by [attachment, detachment]. Losses hitting the tranche
    reduce the investor's principal.

    Uses the Li (2000) one-factor Gaussian copula:
        Z_j = √ρ M + √(1-ρ) ε_j,  j = 1..N
        Name j defaults if Z_j < Φ⁻¹(1 - Q_j(t))

    where M ~ N(0,1) is the systematic factor, ε_j ~ N(0,1) iid,
    ρ is the asset correlation, and Q_j(t) is the survival probability.
    Fresh draws are generated per period to avoid cross-date correlation
    artifacts.

    Args:
        start: issue date.
        end: maturity.
        coupon_rate: annual coupon.
        notional: funded amount.
        attachment: lower tranche bound (fraction of portfolio loss).
        detachment: upper tranche bound.
        recovery: uniform recovery rate.
        n_names: number of reference entities.
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float = 0.05,
        notional: float = 10_000_000.0,
        attachment: float = 0.0,
        detachment: float = 0.03,
        recovery: float = 0.4,
        n_names: int = 125,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if n_names < 1:
            raise ValueError(f"n_names must be >= 1, got {n_names}")
        if not (0 <= attachment < detachment <= 1):
            raise ValueError(f"need 0 <= attachment < detachment <= 1, got [{attachment}, {detachment}]")
        if start >= end:
            raise ValueError(f"end ({end}) must be after start ({start})")
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.attachment = attachment
        self.detachment = detachment
        self.recovery = recovery
        self.n_names = n_names
        self.schedule = generate_schedule(start, end, Frequency.QUARTERLY)

    def price_mc(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
        rho: float = 0.3,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> BasketCLNResult:
        """Price via Gaussian copula Monte Carlo.

        At each coupon date, simulate portfolio loss via copula.
        Tranche loss = max(0, min(portfolio_loss - attachment, width)) / width.
        """
        if len(survival_curves) != self.n_names:
            raise ValueError(
                f"Expected {self.n_names} survival curves, got {len(survival_curves)}"
            )
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")

        width = self.detachment - self.attachment
        if width <= 0:
            raise ValueError("detachment must be > attachment")

        rng = np.random.default_rng(seed)
        sqrt_rho = math.sqrt(rho)
        sqrt_1_rho = math.sqrt(1.0 - rho)

        coupon_pv = 0.0
        total_expected_loss = 0.0
        loss_std_error = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, DayCountConvention.ACT_360)
            df = discount_curve.df(t_end)

            # Fresh draws per period — each period has independent
            # systematic and idiosyncratic factors (Li 2000 copula).
            M = rng.standard_normal(n_sims)
            eps = rng.standard_normal((n_sims, self.n_names))
            Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

            # Default thresholds at this date
            thresholds = np.array([
                norm.ppf(max(1.0 - sc.survival(t_end), 1e-15))
                for sc in survival_curves
            ])
            defaults = Z < thresholds[np.newaxis, :]

            # Portfolio loss fraction: n_defaults / n_names × (1 - R)
            n_defaults = defaults.sum(axis=1).astype(float)
            portfolio_loss = n_defaults / self.n_names * (1.0 - self.recovery)

            # Tranche loss (Li 2000, O'Kane §18.3)
            tranche_loss = np.clip(portfolio_loss - self.attachment, 0.0, width) / width
            avg_tranche_loss = float(tranche_loss.mean())

            # Tranche survival = 1 - tranche_loss
            tranche_surv = 1.0 - avg_tranche_loss

            # Coupon on surviving notional
            coupon_pv += self.notional * self.coupon_rate * yf * df * tranche_surv

            total_expected_loss = avg_tranche_loss  # cumulative at maturity
            # MC standard error on tranche loss at maturity
            loss_std_error = float(tranche_loss.std()) / math.sqrt(n_sims)

        # Principal: surviving notional at maturity
        df_T = discount_curve.df(self.end)
        principal_pv = self.notional * df_T * (1.0 - total_expected_loss)

        total_pv = coupon_pv + principal_pv

        return BasketCLNResult(
            price=total_pv,
            expected_loss=total_expected_loss,
            tranche_width=width,
            attachment=self.attachment,
            detachment=self.detachment,
            std_error=loss_std_error,
        )

    def price_mc_correlated_recovery(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
        rho: float = 0.3,
        recovery_sensitivity: float = 0.10,
        recoveries: list[float] | None = None,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> BasketCLNResult:
        """Price with correlated recovery via systematic factor.

        Recovery depends on M: R(M) = mean + sensitivity × M, clipped to [0,1].
        In downturn (low M): more defaults AND lower recovery.

        Double-correlation: M drives both defaults (via ρ_DD) and recovery
        (via recovery_sensitivity).

        Args:
            rho: default-default correlation (ρ_DD).
            recovery_sensitivity: dR/dM (positive: recovery rises with M).
            recoveries: per-name recovery means (None = uniform self.recovery).
        """
        if len(survival_curves) != self.n_names:
            raise ValueError(f"Expected {self.n_names} survival curves, got {len(survival_curves)}")

        width = self.detachment - self.attachment
        if width <= 0:
            raise ValueError("detachment must be > attachment")

        rng = np.random.default_rng(seed)
        sqrt_rho = math.sqrt(max(rho, 0.0))
        sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))

        rec_means = np.array(recoveries if recoveries else [self.recovery] * self.n_names)

        coupon_pv = 0.0
        total_expected_loss = 0.0
        loss_std_error = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, DayCountConvention.ACT_360)
            df = discount_curve.df(t_end)

            M = rng.standard_normal(n_sims)
            eps = rng.standard_normal((n_sims, self.n_names))
            Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

            thresholds = np.array([
                norm.ppf(max(1.0 - sc.survival(t_end), 1e-15))
                for sc in survival_curves
            ])
            defaults = Z < thresholds[np.newaxis, :]

            # Recovery correlated with M (per-name, per-path)
            # R_j(M) = mean_j + sensitivity × M, clipped
            R_per_path = rec_means[np.newaxis, :] + recovery_sensitivity * M[:, np.newaxis]
            R_per_path = np.clip(R_per_path, 0.0, 1.0)

            # Portfolio loss with heterogeneous, correlated recovery
            lgd_per_name = (1.0 - R_per_path) * defaults  # (n_sims, n_names)
            portfolio_loss = lgd_per_name.sum(axis=1) / self.n_names

            tranche_loss = np.clip(portfolio_loss - self.attachment, 0.0, width) / width
            avg_tranche_loss = float(tranche_loss.mean())
            tranche_surv = 1.0 - avg_tranche_loss
            coupon_pv += self.notional * self.coupon_rate * yf * df * tranche_surv

            total_expected_loss = avg_tranche_loss
            loss_std_error = float(tranche_loss.std()) / math.sqrt(n_sims)

        df_T = discount_curve.df(self.end)
        principal_pv = self.notional * df_T * (1.0 - total_expected_loss)
        total_pv = coupon_pv + principal_pv

        return BasketCLNResult(
            price=total_pv, expected_loss=total_expected_loss,
            tranche_width=width, attachment=self.attachment,
            detachment=self.detachment, std_error=loss_std_error,
        )

    def price_mc_copula(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
        rho: float = 0.3,
        copula: str = "gaussian",
        nu: float = 5.0,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> BasketCLNResult:
        """Price with flexible copula choice.

        Args:
            copula: "gaussian", "t" (Student-t with nu degrees of freedom).
            nu: degrees of freedom for t-copula (ignored for Gaussian).
        """
        if copula == "gaussian":
            return self.price_mc(discount_curve, survival_curves, rho, n_sims, seed)

        if copula != "t":
            raise ValueError(f"Unsupported copula '{copula}'. Options: gaussian, t")

        # Student-t copula
        if len(survival_curves) != self.n_names:
            raise ValueError(f"Expected {self.n_names} survival curves")

        from scipy.stats import t as t_dist

        width = self.detachment - self.attachment
        if width <= 0:
            raise ValueError("detachment must be > attachment")

        rng = np.random.default_rng(seed)
        sqrt_rho = math.sqrt(max(rho, 0.0))
        sqrt_1_rho = math.sqrt(max(1.0 - rho, 0.0))

        coupon_pv = 0.0
        total_expected_loss = 0.0
        loss_std_error = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, DayCountConvention.ACT_360)
            df = discount_curve.df(t_end)

            chi2 = np.maximum(rng.chisquare(nu, n_sims), 0.01)
            W = np.sqrt(nu / chi2)

            M = rng.standard_normal(n_sims) * W
            eps = rng.standard_normal((n_sims, self.n_names)) * W[:, np.newaxis]
            Z = sqrt_rho * M[:, np.newaxis] + sqrt_1_rho * eps

            thresholds = np.array([
                t_dist.ppf(max(1.0 - sc.survival(t_end), 1e-15), nu)
                for sc in survival_curves
            ])
            defaults = Z < thresholds[np.newaxis, :]

            n_defaults = defaults.sum(axis=1).astype(float)
            portfolio_loss = n_defaults / self.n_names * (1.0 - self.recovery)

            tranche_loss = np.clip(portfolio_loss - self.attachment, 0.0, width) / width
            avg_tranche_loss = float(tranche_loss.mean())
            tranche_surv = 1.0 - avg_tranche_loss
            coupon_pv += self.notional * self.coupon_rate * yf * df * tranche_surv

            total_expected_loss = avg_tranche_loss
            loss_std_error = float(tranche_loss.std()) / math.sqrt(n_sims)

        df_T = discount_curve.df(self.end)
        principal_pv = self.notional * df_T * (1.0 - total_expected_loss)
        total_pv = coupon_pv + principal_pv

        return BasketCLNResult(
            price=total_pv, expected_loss=total_expected_loss,
            tranche_width=width, attachment=self.attachment,
            detachment=self.detachment, std_error=loss_std_error,
        )

    @property
    def tranche_width(self) -> float:
        return self.detachment - self.attachment

    def pv_ctx(self, ctx) -> float:
        """Trade/Portfolio integration (requires survival_curves in context)."""
        curve = ctx.discount_curve
        if hasattr(ctx, "survival_curves") and ctx.survival_curves:
            scs = ctx.survival_curves
        else:
            scs = [SurvivalCurve.flat(curve.reference_date, 0.02)] * self.n_names
        rho = getattr(ctx, "correlation", 0.3)
        result = self.price_mc(curve, scs, rho=rho)
        return result.price

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("cln", ["start", "end", "coupon_rate", "notional", "recovery", "leverage", "floating", "frequency", "day_count"])(CreditLinkedNote)


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------

def cln_price_stochastic_recovery_via_engine(
    cln_inst: CreditLinkedNote,
    discount_curve,
    survival_curve,
    recovery_spec=None,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """CLN stochastic recovery via unified MC engine (copula sampling).

    Delegates to original — copula default indicators don't benefit from
    engine path generation, but this provides API consistency.
    """
    return cln_inst.price_stochastic_recovery(
        discount_curve, survival_curve, recovery_spec, n_sims, seed)


def basket_cln_price_mc_via_engine(
    basket: BasketCLN,
    discount_curve,
    survival_curves: list,
    rho: float = 0.3,
    n_sims: int = 50_000,
    seed: int = 42,
) -> BasketCLNResult:
    """Basket CLN via unified MC engine (Gaussian copula).

    Delegates to original — copula default sampling is inherently
    different from SDE path generation.
    """
    return basket.price_mc(discount_curve, survival_curves, rho, n_sims, seed)
