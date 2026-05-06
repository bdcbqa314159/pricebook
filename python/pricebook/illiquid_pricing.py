"""Illiquid bond/loan pricing: matrix pricing, liquidity premium, private placements.

For instruments without observable market prices — private placements,
bilateral loans, unlisted bonds — pricing via comparable spread analysis,
liquidity premium models, and composite spread construction.

    from pricebook.illiquid_pricing import (
        MatrixPricer, LiquidityPremiumModel, PrivatePlacementPricer,
    )

References:
    Amihud (2002). Illiquidity and stock returns. JFM.
    Longstaff (2004). The Flight-to-Liquidity Premium. JFE.
    Dick-Nielsen, Feldhütter, Lando (2012). Corporate bond liquidity. JFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# Matrix Pricing
# ---------------------------------------------------------------------------

@dataclass
class Comparable:
    """A comparable liquid bond for matrix pricing."""
    name: str
    sector: str
    rating: str
    maturity_years: float
    spread_bp: float
    seniority: str = "senior"


@dataclass
class MatrixResult:
    """Matrix pricing result."""
    fair_spread_bp: float       # estimated fair spread (bp)
    confidence_bp: float        # ± confidence interval (bp)
    n_comparables: int          # number of comparables used
    closest_name: str           # most similar comparable
    adjustments: dict           # sector/rating/maturity adjustments

    def to_dict(self) -> dict:
        return {
            "fair_spread_bp": self.fair_spread_bp,
            "confidence_bp": self.confidence_bp,
            "n_comparables": self.n_comparables,
            "closest": self.closest_name,
            "adjustments": self.adjustments,
        }


# Rating order for distance calculation
_RATING_ORDER = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC": 17, "CC": 18, "C": 19, "D": 20,
}


class MatrixPricer:
    """Estimate fair spread from comparable liquid bonds.

    Uses weighted average of comparable spreads, with weights inversely
    proportional to distance in (rating, maturity, sector) space.

    Args:
        comparables: list of comparable liquid bonds.
    """

    def __init__(self, comparables: list[Comparable]):
        if not comparables:
            raise ValueError("comparables list is empty")
        self.comparables = comparables

    def price(
        self,
        target_sector: str,
        target_rating: str,
        target_maturity_years: float,
        target_seniority: str = "senior",
    ) -> MatrixResult:
        """Estimate fair spread for the target bond.

        Distance metric:
            d = w_rating × |rating_diff| + w_maturity × |maturity_diff| + w_sector
        where w_sector = 0 if same sector, 2 otherwise.
        """
        target_rank = _RATING_ORDER.get(target_rating, 10)

        distances = []
        for comp in self.comparables:
            comp_rank = _RATING_ORDER.get(comp.rating, 10)
            d_rating = abs(target_rank - comp_rank) * 10  # 10bp per notch
            d_maturity = abs(target_maturity_years - comp.maturity_years) * 5  # 5bp per year
            d_sector = 0 if comp.sector.lower() == target_sector.lower() else 20  # 20bp cross-sector
            d_seniority = 0 if comp.seniority == target_seniority else 30
            distance = d_rating + d_maturity + d_sector + d_seniority
            distances.append((comp, distance))

        # Weight by inverse distance (add floor to avoid division by zero)
        weights = [(comp, 1.0 / max(d + 1, 1)) for comp, d in distances]
        total_weight = sum(w for _, w in weights)

        if total_weight <= 0:
            return MatrixResult(0, 0, 0, "", {})

        fair_spread = sum(comp.spread_bp * w for comp, w in weights) / total_weight

        # Confidence interval: weighted std deviation
        variance = sum(w * (comp.spread_bp - fair_spread) ** 2 for comp, w in weights) / total_weight
        confidence = math.sqrt(variance)

        # Closest comparable
        closest = min(distances, key=lambda x: x[1])

        # Adjustments breakdown
        adjustments = {
            "rating_adj_bp": (target_rank - _RATING_ORDER.get(closest[0].rating, 10)) * 10,
            "maturity_adj_bp": (target_maturity_years - closest[0].maturity_years) * 5,
            "sector_adj_bp": 0 if closest[0].sector.lower() == target_sector.lower() else 20,
        }

        return MatrixResult(
            fair_spread_bp=fair_spread,
            confidence_bp=confidence,
            n_comparables=len(self.comparables),
            closest_name=closest[0].name,
            adjustments=adjustments,
        )


# ---------------------------------------------------------------------------
# Liquidity Premium Model
# ---------------------------------------------------------------------------

@dataclass
class LiquidityPremiumResult:
    """Liquidity premium estimation result."""
    illiquidity_premium_bp: float    # estimated liquidity premium (bp)
    bid_ask_component_bp: float
    size_component_bp: float
    age_component_bp: float

    def to_dict(self) -> dict:
        return {
            "total_bp": self.illiquidity_premium_bp,
            "bid_ask_bp": self.bid_ask_component_bp,
            "size_bp": self.size_component_bp,
            "age_bp": self.age_component_bp,
        }


class LiquidityPremiumModel:
    """Estimate illiquidity premium using Amihud-style factors.

    Premium = α + β₁ × bid_ask_pct + β₂ × log(issue_size_mm) + β₃ × age_years

    Default coefficients calibrated to USD IG corporate bond data
    (Dick-Nielsen, Feldhütter, Lando 2012).

    Args:
        alpha: constant (bp).
        beta_bid_ask: coefficient on bid-ask spread (bp per 1% bid-ask).
        beta_size: coefficient on log(issue_size in $MM) (bp, negative = bigger → less premium).
        beta_age: coefficient on age in years (bp per year).
    """

    def __init__(
        self,
        alpha: float = 30.0,
        beta_bid_ask: float = 50.0,
        beta_size: float = -8.0,
        beta_age: float = 3.0,
    ):
        self.alpha = alpha
        self.beta_bid_ask = beta_bid_ask
        self.beta_size = beta_size
        self.beta_age = beta_age

    def estimate(
        self,
        bid_ask_pct: float = 0.5,
        issue_size_mm: float = 500.0,
        age_years: float = 2.0,
    ) -> LiquidityPremiumResult:
        """Estimate illiquidity premium in basis points.

        Args:
            bid_ask_pct: bid-ask spread as % of price (e.g. 0.5 = 50 cents on $100).
            issue_size_mm: issue size in millions (larger = more liquid).
            age_years: years since issuance (older = less liquid).
        """
        bid_ask_bp = self.beta_bid_ask * bid_ask_pct
        size_bp = self.beta_size * math.log(max(issue_size_mm, 1))
        age_bp = self.beta_age * age_years

        total = self.alpha + bid_ask_bp + size_bp + age_bp
        total = max(total, 0.0)  # premium cannot be negative

        return LiquidityPremiumResult(
            illiquidity_premium_bp=total,
            bid_ask_component_bp=bid_ask_bp,
            size_component_bp=size_bp,
            age_component_bp=age_bp,
        )


# ---------------------------------------------------------------------------
# Private Placement Pricer
# ---------------------------------------------------------------------------

@dataclass
class PrivatePlacementResult:
    """Private placement pricing result."""
    total_spread_bp: float        # composite spread
    credit_spread_bp: float       # from matrix pricing
    illiquidity_premium_bp: float  # from liquidity model
    complexity_premium_bp: float   # for structural features
    pv: float                     # PV using composite Z-spread
    price_per_100: float          # clean price per 100 face

    def to_dict(self) -> dict:
        return {
            "total_spread_bp": self.total_spread_bp,
            "credit_bp": self.credit_spread_bp,
            "illiquidity_bp": self.illiquidity_premium_bp,
            "complexity_bp": self.complexity_premium_bp,
            "pv": self.pv, "price_100": self.price_per_100,
        }


class PrivatePlacementPricer:
    """Price a private placement using composite spread.

    total_spread = credit_spread + illiquidity_premium + complexity_premium

    PV is computed by discounting at risk-free + total_spread (Z-spread approach).

    Args:
        coupon_rate: annual coupon rate.
        maturity_years: time to maturity.
        notional: face value.
        credit_spread_bp: credit spread from matrix pricing or analyst estimate.
        illiquidity_premium_bp: from LiquidityPremiumModel.
        complexity_premium_bp: additional spread for structural features
            (covenants, embedded options, call protection, etc.).
    """

    def __init__(
        self,
        coupon_rate: float,
        maturity_years: float,
        notional: float = 100.0,
        credit_spread_bp: float = 150.0,
        illiquidity_premium_bp: float = 50.0,
        complexity_premium_bp: float = 25.0,
    ):
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.notional = notional
        self.credit_spread_bp = credit_spread_bp
        self.illiquidity_premium_bp = illiquidity_premium_bp
        self.complexity_premium_bp = complexity_premium_bp

    @property
    def total_spread_bp(self) -> float:
        return self.credit_spread_bp + self.illiquidity_premium_bp + self.complexity_premium_bp

    def price(self, discount_curve: DiscountCurve) -> PrivatePlacementResult:
        """Price the private placement via Z-spread discounting.

        Discount each cashflow at: rf_zero_rate + total_spread.
        """
        from datetime import timedelta

        ref = discount_curve.reference_date
        total_spread = self.total_spread_bp / 10_000
        n_periods = int(self.maturity_years * 2)  # semi-annual
        dt = 0.5

        pv = 0.0
        for i in range(1, n_periods + 1):
            t = i * dt
            cf_date = ref + timedelta(days=int(t * 365))
            # Risk-free DF from curve, then apply Z-spread
            rf_df = discount_curve.df(cf_date)
            spread_df = math.exp(-total_spread * t)
            df = rf_df * spread_df
            coupon = self.notional * self.coupon_rate * dt
            pv += coupon * df

        # Principal at maturity
        t_mat = n_periods * dt
        mat_date = ref + timedelta(days=int(t_mat * 365))
        rf_df_mat = discount_curve.df(mat_date)
        spread_df_mat = math.exp(-total_spread * t_mat)
        pv += self.notional * rf_df_mat * spread_df_mat

        price_100 = pv / self.notional * 100

        return PrivatePlacementResult(
            total_spread_bp=self.total_spread_bp,
            credit_spread_bp=self.credit_spread_bp,
            illiquidity_premium_bp=self.illiquidity_premium_bp,
            complexity_premium_bp=self.complexity_premium_bp,
            pv=pv,
            price_per_100=price_100,
        )
