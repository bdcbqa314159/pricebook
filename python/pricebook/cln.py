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

from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.survival_curve import SurvivalCurve


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
        notional: float = 1_000_000.0,
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
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.recovery = recovery
        self.leverage = leverage
        self.floating = floating
        self.frequency = frequency
        self.day_count = day_count
        self.schedule = generate_schedule(start, end, frequency)

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

            # Coupon: fixed or floating
            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = self.notional * (fwd + self.coupon_rate) * yf
            else:
                coupon = self.notional * self.coupon_rate * yf

            pv += coupon * df * surv

            # Recovery on default (investor gets recovery × notional)
            pv += self.recovery * self.notional * default_prob * df

            # Leveraged loss: extra (leverage - 1) × (1-R) × notional on default
            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * self.notional
                pv -= extra_loss * default_prob * df

        # Principal at maturity (conditional on survival)
        pv += self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)

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

            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = self.notional * (fwd + self.coupon_rate) * yf
            else:
                coupon = self.notional * self.coupon_rate * yf

            coupon_pv += coupon * df * surv
            recovery_pv += self.recovery * self.notional * default_prob * df

            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * self.notional
                recovery_pv -= extra_loss * default_prob * df

        principal_pv = self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)
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
        from pricebook.solvers import brentq

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
        """Bump-and-reprice sensitivities.

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

        # Recovery sensitivity: +1% bump
        old_recovery = self.recovery
        self.recovery = old_recovery + 0.01
        rec_up = self.dirty_price(discount_curve, survival_curve)
        self.recovery = old_recovery
        recovery_sens = rec_up - base

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

    # ---- Stochastic recovery pricing ----

    def price_stochastic_recovery(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        recovery_spec=None,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> CLNResult:
        """Price CLN with stochastic recovery correlated to default.

        Uses MC: for each path, draw correlated (default, recovery) pair
        per period. The wrong-way premium = difference vs fixed-recovery price.

        Args:
            recovery_spec: RecoverySpec (from recovery_pricing module).
                If None, uses fixed recovery at self.recovery.
        """
        from pricebook.recovery_pricing import RecoverySpec

        if recovery_spec is None:
            recovery_spec = RecoverySpec.fixed(self.recovery)

        rng = np.random.default_rng(seed)
        pv_paths = np.zeros(n_sims)

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv

            # Coupon conditional on survival
            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = self.notional * (fwd + self.coupon_rate) * yf
            else:
                coupon = self.notional * self.coupon_rate * yf
            pv_paths += coupon * df * surv

            # Default in this period: draw default indicators and recoveries
            Z_D = rng.standard_normal(n_sims)
            threshold = norm.ppf(max(default_prob / max(surv_prev, 1e-10), 1e-15))
            period_defaults = Z_D < threshold

            # Correlated recovery
            R_samples = recovery_spec.sample(n_sims, systematic_factor=Z_D, seed=seed + i)

            # Recovery on default (per path)
            recovery_pv = R_samples * self.notional * period_defaults * df
            pv_paths += recovery_pv

            # Leveraged loss
            if self.leverage > 1.0:
                extra = (self.leverage - 1.0) * (1.0 - R_samples) * self.notional * period_defaults * df
                pv_paths -= extra

        # Principal at maturity conditional on survival
        pv_paths += self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)

        total_pv = float(pv_paths.mean())

        # Decomposition (approximate)
        fixed_price = self.dirty_price(discount_curve, survival_curve)
        credit_spread_approx = (fixed_price - total_pv) / max(
            self.notional * self._risky_annuity(discount_curve, survival_curve), 1e-10
        )

        return CLNResult(
            price=total_pv, coupon_pv=0.0, principal_pv=0.0,
            recovery_pv=0.0, credit_spread=credit_spread_approx,
            par_coupon=0.0,
        )

    def price_stochastic_intensity(
        self,
        discount_curve: DiscountCurve,
        intensity_model,
        x0: float | None = None,
        n_paths: int = 50_000,
        n_steps: int = 200,
        seed: int = 42,
    ) -> CLNResult:
        """Price CLN via Monte Carlo under stochastic intensity.

        For each path:
          1. Simulate intensity λ(t) from the model.
          2. Compute path survival Q_path(t) = exp(−∫λ ds).
          3. Accumulate coupon × df × Q and recovery × df × ΔQ per period.
          4. Average across paths.

        Supports CIRPlusPlus, HWHazardRate, or CIRIntensity models.
        """
        import numpy as np

        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        if x0 is None:
            # Use theta or first hazard as starting point
            if hasattr(intensity_model, 'theta'):
                x0 = intensity_model.theta
            else:
                x0 = 0.02

        # Simulate intensity paths
        result = intensity_model.simulate(x0, T, n_steps, n_paths, seed)
        lambda_paths = result.lambda_paths  # (n_paths, n_steps+1)
        dt = T / n_steps

        # Cumulative survival per path: Q(t) = exp(-∫₀ᵗ λ ds)
        cum_integral = np.cumsum(lambda_paths[:, :-1] * dt, axis=1)
        # Prepend zero for t=0
        cum_integral = np.concatenate([np.zeros((n_paths, 1)), cum_integral], axis=1)
        path_survival = np.exp(-cum_integral)  # (n_paths, n_steps+1)

        # Map coupon periods to simulation steps
        coupon_pv_total = 0.0
        recovery_pv_total = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf_coupon = year_fraction(t_start, t_end, self.day_count)
            t_frac = year_fraction(self.start, t_end, DayCountConvention.ACT_365_FIXED)
            t_frac_prev = year_fraction(self.start, t_start, DayCountConvention.ACT_365_FIXED)

            step_end = min(int(t_frac / dt), n_steps)
            step_prev = min(int(t_frac_prev / dt), n_steps)

            df = discount_curve.df(t_end)
            surv_end = path_survival[:, step_end]        # (n_paths,)
            surv_prev = path_survival[:, step_prev]      # (n_paths,)
            default_prob = np.maximum(surv_prev - surv_end, 0.0)

            # Coupon
            if self.floating:
                fwd = discount_curve.forward_rate(t_start, t_end)
                coupon = self.notional * (fwd + self.coupon_rate) * yf_coupon
            else:
                coupon = self.notional * self.coupon_rate * yf_coupon

            coupon_pv_total += float(np.mean(coupon * df * surv_end))

            # Recovery on default
            recovery_pv_total += float(np.mean(
                self.recovery * self.notional * default_prob * df
            ))

            # Leveraged loss
            if self.leverage > 1.0:
                extra_loss = (self.leverage - 1.0) * (1.0 - self.recovery) * self.notional
                recovery_pv_total -= float(np.mean(extra_loss * default_prob * df))

        # Principal at maturity
        final_surv = float(np.mean(path_survival[:, -1]))
        principal_pv = self.notional * discount_curve.df(self.end) * final_surv

        total_pv = coupon_pv_total + principal_pv + recovery_pv_total

        return CLNResult(
            price=total_pv,
            coupon_pv=coupon_pv_total,
            principal_pv=principal_pv,
            recovery_pv=recovery_pv_total,
            credit_spread=0.0,
            par_coupon=0.0,
        )

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
        """Auto-calibrate stochastic model from survival curve, then price.

        Combines Phase 2 (curve → model) + Phase 4 (model → MC price).
        """
        from pricebook.hazard_rate_models import CIRPlusPlus, HWHazardRate

        if model_type == "cir++":
            model = CIRPlusPlus.from_survival_curve(
                survival_curve, kappa=kappa, xi=xi, **model_kwargs)
            x0 = model.theta
        elif model_type == "hw":
            model = HWHazardRate.from_survival_curve(
                survival_curve, a=kappa, sigma=xi)
            x0 = survival_curve.pillar_hazards()[0][1] if survival_curve.pillar_hazards() else 0.02
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return self.price_stochastic_intensity(
            discount_curve, model, x0, n_paths, n_steps, seed)

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
        """Price bilateral CLN with dual default risk (ref entity + issuer).

        On reference default → investor receives recovery × notional.
        On issuer default (no ref default) → investor receives issuer_recovery × notional.
        On both → investor receives min(recovery, issuer_recovery) × notional.
        On neither → full coupon + principal.

        Simulates two correlated uniform defaults via Gaussian copula.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        T = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)

        # Correlated normals via Cholesky
        Z = rng.standard_normal((n_paths, 2))
        Z[:, 1] = correlation * Z[:, 0] + math.sqrt(1 - correlation**2) * Z[:, 1]

        # Convert to uniform via Φ
        from scipy.stats import norm
        U_ref = norm.cdf(Z[:, 0])
        U_issuer = norm.cdf(Z[:, 1])

        total_pv = np.zeros(n_paths)

        for path in range(n_paths):
            # Default times from inverse survival
            # τ where Q(τ) = U → find by scanning periods
            ref_defaulted = False
            issuer_defaulted = False
            pv = 0.0

            for i in range(1, len(self.schedule)):
                t_start = self.schedule[i - 1]
                t_end = self.schedule[i]
                yf = year_fraction(t_start, t_end, self.day_count)
                df = discount_curve.df(t_end)
                surv_ref = ref_survival.survival(t_end)
                surv_iss = issuer_survival.survival(t_end)

                # Check defaults this period
                if not ref_defaulted and surv_ref < 1 - U_ref[path]:
                    ref_defaulted = True
                if not issuer_defaulted and surv_iss < 1 - U_issuer[path]:
                    issuer_defaulted = True

                if ref_defaulted and issuer_defaulted:
                    pv += min(self.recovery, issuer_recovery) * self.notional * df
                    break
                elif ref_defaulted:
                    pv += self.recovery * self.notional * df
                    break
                elif issuer_defaulted:
                    pv += issuer_recovery * self.notional * df
                    break
                else:
                    # Coupon (conditional on both surviving)
                    coupon = self.notional * self.coupon_rate * yf
                    pv += coupon * df

            if not ref_defaulted and not issuer_defaulted:
                pv += self.notional * discount_curve.df(self.end)

            total_pv[path] = pv

        mean_pv = float(np.mean(total_pv))

        return CLNResult(
            price=mean_pv,
            coupon_pv=0.0,
            principal_pv=0.0,
            recovery_pv=0.0,
            credit_spread=0.0,
            par_coupon=0.0,
        )

    def rec_vol_01(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        recovery_spec=None,
        n_sims: int = 50_000,
        seed: int = 42,
    ) -> float:
        """Recovery vol sensitivity: PV change for 1% increase in recovery std.

        Higher recovery vol → more wrong-way risk → lower CLN PV.
        """
        from pricebook.recovery_pricing import RecoverySpec

        if recovery_spec is None:
            recovery_spec = RecoverySpec(mean=self.recovery, std=0.15,
                                         correlation_to_default=-0.3)

        base = self.price_stochastic_recovery(
            discount_curve, survival_curve, recovery_spec, n_sims, seed,
        ).price

        bumped_spec = RecoverySpec(
            mean=recovery_spec.mean,
            std=recovery_spec.std + 0.01,
            distribution=recovery_spec.distribution,
            correlation_to_default=recovery_spec.correlation_to_default,
        )
        bumped = self.price_stochastic_recovery(
            discount_curve, survival_curve, bumped_spec, n_sims, seed,
        ).price

        return bumped - base

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

from pricebook.serialisable import serialisable as _serialisable
_serialisable("cln", ["start", "end", "coupon_rate", "notional", "recovery", "leverage", "floating", "frequency", "day_count"])(CreditLinkedNote)
