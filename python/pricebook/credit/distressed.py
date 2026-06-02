"""Distressed debt: DIP financing, fulcrum analysis, exchange offers,
recovery waterfalls, Chapter 11 timeline.

    from pricebook.credit.distressed import (
        DIPLoan, FulcrumAnalysis, RecoveryWaterfall,
        ExchangeOffer, Chapter11Timeline,
    )

References:
    Moyer (2004). Distressed Debt Analysis. J. Ross Publishing.
    Altman & Hotchkiss (2006). Corporate Financial Distress and Bankruptcy.
    Gilson (2010). Creating Value Through Corporate Restructuring, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pricebook.core.serialisable import serialisable as _serialisable


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class CapitalStructureLayer:
    """Single layer in a capital structure."""
    name: str
    notional: float
    seniority: int       # lower = more senior (0 = DIP/super-priority)
    secured: bool = False
    coupon: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class DIPResult:
    """DIP financing analysis result."""
    dip_size: float
    roll_up_amount: float
    carve_out: float
    total_super_priority: float
    expected_recovery_pct: float
    dip_spread: float
    dip_all_in_cost: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class FulcrumResult:
    """Fulcrum security identification."""
    fulcrum_class: str
    fulcrum_recovery_pct: float
    classes_above: list[str]
    classes_below: list[str]
    implied_equity_value: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class RecoveryDistribution:
    """Per-class recovery from absolute priority waterfall."""
    enterprise_value: float
    recoveries: dict[str, float]
    losses: dict[str, float]
    total_claims: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ExchangeResult:
    """Exchange offer / tender analysis."""
    old_value: float
    new_value: float
    consent_fee: float
    exchange_premium: float
    participation_breakeven: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class Chapter11Milestone:
    """Chapter 11 timeline milestone."""
    event: str
    estimated_months: float
    cumulative_months: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class Chapter11Result:
    """Chapter 11 timeline and recovery estimates."""
    milestones: list[Chapter11Milestone]
    estimated_duration_months: float
    recovery_by_class: dict[str, tuple[float, float]]
    administrative_costs_pct: float

    def to_dict(self) -> dict:
        return {
            "milestones": [m.to_dict() for m in self.milestones],
            "estimated_duration_months": self.estimated_duration_months,
            "recovery_by_class": self.recovery_by_class,
            "administrative_costs_pct": self.administrative_costs_pct,
        }


# ═══════════════════════════════════════════════════════════════
# DIP Financing
# ═══════════════════════════════════════════════════════════════

class DIPLoan:
    """Debtor-in-possession financing with super-priority.

    DIP loans sit at the top of the capital structure in bankruptcy.
    Roll-up: converts pre-petition debt into DIP (higher priority).
    Carve-out: professional fees reserved from DIP collateral.

    Args:
        notional: new DIP facility size.
        spread: DIP coupon spread (typically high: 6-12%).
        maturity_months: expected DIP maturity.
        roll_up_amount: pre-petition debt rolled into DIP.
        carve_out: professional fees carve-out.
        upfront_fee: OID / upfront arrangement fee.
    """

    def __init__(
        self,
        notional: float,
        spread: float = 0.08,
        maturity_months: float = 18.0,
        roll_up_amount: float = 0.0,
        carve_out: float = 0.0,
        upfront_fee: float = 0.02,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        self.notional = notional
        self.spread = spread
        self.maturity_months = maturity_months
        self.roll_up_amount = roll_up_amount
        self.carve_out = carve_out
        self.upfront_fee = upfront_fee

    def pv(self, discount_rate: float = 0.05) -> float:
        """PV of DIP assuming ~100% recovery (super-priority).

        Simple PV: notional discounted at risk-free + small spread.
        """
        t = self.maturity_months / 12.0
        total_claim = self.notional + self.roll_up_amount
        coupon_income = total_claim * self.spread * t
        # Near-certain repayment → discount at low rate
        pv = (total_claim + coupon_income) / (1 + discount_rate) ** t
        return pv

    def dip_economics(
        self,
        existing_debt: list[CapitalStructureLayer] | None = None,
    ) -> DIPResult:
        """Analyse DIP economics."""
        total_super = self.notional + self.roll_up_amount
        maturity_years = self.maturity_months / 12.0
        all_in = self.spread + self.upfront_fee / maturity_years if maturity_years > 0 else self.spread

        return DIPResult(
            dip_size=self.notional,
            roll_up_amount=self.roll_up_amount,
            carve_out=self.carve_out,
            total_super_priority=total_super,
            expected_recovery_pct=1.0,  # super-priority
            dip_spread=self.spread,
            dip_all_in_cost=all_in,
        )

    def to_dict(self) -> dict:
        return {
            "notional": self.notional, "spread": self.spread,
            "maturity_months": self.maturity_months,
            "roll_up_amount": self.roll_up_amount,
            "carve_out": self.carve_out, "upfront_fee": self.upfront_fee,
        }


# ═══════════════════════════════════════════════════════════════
# Recovery Waterfall (Absolute Priority)
# ═══════════════════════════════════════════════════════════════

class RecoveryWaterfall:
    """Absolute priority distribution in bankruptcy.

    Each class must be paid in full before any junior class receives
    anything. This is the legal standard (unlike CLO waterfalls which
    are contractual with triggers).
    """

    def __init__(self, capital_structure: list[CapitalStructureLayer]):
        if not capital_structure:
            raise ValueError("capital_structure must not be empty")
        self.capital_structure = sorted(capital_structure, key=lambda l: l.seniority)

    def distribute(self, enterprise_value: float) -> RecoveryDistribution:
        """Distribute enterprise value through capital structure.

        Args:
            enterprise_value: total available value for distribution.
        """
        remaining = max(enterprise_value, 0.0)
        total_claims = sum(l.notional for l in self.capital_structure)
        recoveries: dict[str, float] = {}
        losses: dict[str, float] = {}

        for layer in self.capital_structure:
            recovery = min(remaining, layer.notional)
            recovery_pct = recovery / layer.notional if layer.notional > 0 else 0.0
            recoveries[layer.name] = recovery_pct
            losses[layer.name] = layer.notional - recovery
            remaining -= recovery

        return RecoveryDistribution(
            enterprise_value=enterprise_value,
            recoveries=recoveries,
            losses=losses,
            total_claims=total_claims,
        )


# ═══════════════════════════════════════════════════════════════
# Fulcrum Analysis
# ═══════════════════════════════════════════════════════════════

class FulcrumAnalysis:
    """Identify the fulcrum security in a distressed capital structure.

    The fulcrum is the most senior class that is impaired — it receives
    equity in a reorganisation. Classes above are repaid in full;
    classes below receive nothing.
    """

    def __init__(self, capital_structure: list[CapitalStructureLayer]):
        if not capital_structure:
            raise ValueError("capital_structure must not be empty")
        self.capital_structure = sorted(capital_structure, key=lambda l: l.seniority)

    def identify_fulcrum(self, enterprise_value: float) -> FulcrumResult:
        """Identify fulcrum security for a given enterprise value."""
        remaining = max(enterprise_value, 0.0)
        classes_above = []
        fulcrum_class = ""
        fulcrum_recovery = 0.0

        for layer in self.capital_structure:
            if remaining >= layer.notional:
                classes_above.append(layer.name)
                remaining -= layer.notional
            else:
                fulcrum_class = layer.name
                fulcrum_recovery = remaining / layer.notional if layer.notional > 0 else 0.0
                remaining = 0.0
                break

        # If EV covers everything, last class is "fulcrum" at 100%
        if not fulcrum_class and classes_above:
            fulcrum_class = classes_above.pop()
            fulcrum_recovery = 1.0

        # Classes below fulcrum
        fulcrum_idx = next(
            (i for i, l in enumerate(self.capital_structure) if l.name == fulcrum_class),
            len(self.capital_structure) - 1,
        )
        classes_below = [l.name for l in self.capital_structure[fulcrum_idx + 1:]]

        return FulcrumResult(
            fulcrum_class=fulcrum_class,
            fulcrum_recovery_pct=fulcrum_recovery,
            classes_above=classes_above,
            classes_below=classes_below,
            implied_equity_value=remaining,
        )

    def sensitivity(
        self,
        ev_range: tuple[float, float],
        n_points: int = 20,
    ) -> dict[str, list[tuple[float, float]]]:
        """Recovery % for each class across a range of enterprise values."""
        import numpy as np
        evs = np.linspace(ev_range[0], ev_range[1], n_points)
        wf = RecoveryWaterfall(self.capital_structure)
        result: dict[str, list[tuple[float, float]]] = {
            l.name: [] for l in self.capital_structure
        }
        for ev in evs:
            rd = wf.distribute(float(ev))
            for name, rec in rd.recoveries.items():
                result[name].append((float(ev), rec))
        return result


# ═══════════════════════════════════════════════════════════════
# Exchange Offer
# ═══════════════════════════════════════════════════════════════

class ExchangeOffer:
    """Distressed exchange / tender offer analytics.

    Models the economics for a bondholder deciding whether to tender
    old bonds for new bonds at a discount, typically with a consent fee.
    """

    def __init__(
        self,
        old_notional: float,
        old_price: float,
        new_price: float,
        consent_fee: float = 0.0,
        participation_threshold: float = 0.50,
    ):
        self.old_notional = old_notional
        self.old_price = old_price
        self.new_price = new_price
        self.consent_fee = consent_fee
        self.participation_threshold = participation_threshold

    def exchange_value(self, participation_rate: float = 1.0) -> ExchangeResult:
        """Compute exchange economics.

        Args:
            participation_rate: fraction of holders who tender.
        """
        old_value = self.old_notional * self.old_price / 100
        new_value = self.old_notional * self.new_price / 100
        fee = self.old_notional * self.consent_fee / 100

        premium = new_value + fee - old_value

        # Breakeven: at what participation does exchange > holdout?
        # Simplified: if new_price + consent > old_price, always exchange
        if self.new_price + self.consent_fee > self.old_price:
            breakeven = 0.0  # always better to exchange
        else:
            breakeven = 1.0  # never better (without coercion)

        return ExchangeResult(
            old_value=old_value, new_value=new_value,
            consent_fee=fee, exchange_premium=premium,
            participation_breakeven=breakeven,
        )

    def holdout_value(self, post_exchange_price: float) -> float:
        """Value if you don't tender.

        Post-exchange, holdout bonds may trade differently (covenant changes,
        subordination via exit consents, etc.).
        """
        return self.old_notional * post_exchange_price / 100

    def prisoners_dilemma(self) -> dict[str, float]:
        """Game theory: cooperative vs defect payoffs.

        Cooperate = tender, Defect = holdout.
        """
        exchange = self.exchange_value()
        holdout_same = self.old_notional * self.old_price / 100
        # If exchange succeeds (above threshold): holdouts may face
        # covenant stripping / subordination
        holdout_if_success = self.old_notional * self.old_price * 0.8 / 100  # 20% penalty
        holdout_if_fail = holdout_same  # exchange fails, bonds unchanged

        return {
            "cooperate_payoff": exchange.new_value + exchange.consent_fee,
            "defect_if_exchange_succeeds": holdout_if_success,
            "defect_if_exchange_fails": holdout_if_fail,
            "cooperation_threshold": self.participation_threshold,
        }


# ═══════════════════════════════════════════════════════════════
# Chapter 11 Timeline
# ═══════════════════════════════════════════════════════════════

# Standard milestone durations (months)
_STANDARD_MILESTONES = [
    ("Filing", 0),
    ("DIP Approval", 1),
    ("Claims Bar Date", 4),
    ("Plan Filed", 8),
    ("Disclosure Approved", 10),
    ("Plan Confirmed", 14),
    ("Emergence", 16),
]

_PREPACK_MILESTONES = [
    ("Filing + Pre-pack Plan", 0),
    ("DIP Approval", 0.5),
    ("Plan Confirmed", 2),
    ("Emergence", 3),
]

_COMPLEX_MILESTONES = [
    ("Filing", 0),
    ("DIP Approval", 2),
    ("Claims Bar Date", 6),
    ("Examiner Appointed", 9),
    ("Plan Filed", 18),
    ("Disclosure Approved", 22),
    ("Plan Confirmed", 28),
    ("Emergence", 32),
]


class Chapter11Timeline:
    """Estimated Chapter 11 timeline with recovery ranges.

    Args:
        complexity: "simple" (pre-pack), "standard", or "complex".
    """

    def __init__(self, complexity: str = "standard"):
        if complexity == "simple":
            self._milestones = _PREPACK_MILESTONES
        elif complexity == "complex":
            self._milestones = _COMPLEX_MILESTONES
        else:
            self._milestones = _STANDARD_MILESTONES
        self.complexity = complexity

    def timeline(self) -> list[Chapter11Milestone]:
        """Get milestone timeline."""
        return [
            Chapter11Milestone(event=name, estimated_months=months, cumulative_months=months)
            for name, months in self._milestones
        ]

    def estimate_recovery(
        self,
        ev_range: tuple[float, float],
        capital_structure: list[CapitalStructureLayer],
        administrative_cost_pct: float = 0.05,
    ) -> Chapter11Result:
        """Estimate per-class recovery range accounting for admin costs.

        Args:
            ev_range: (low_ev, high_ev) estimate of reorganisation value.
            capital_structure: capital structure for waterfall.
            administrative_cost_pct: professional fees as % of EV.
        """
        wf = RecoveryWaterfall(capital_structure)

        # Low case: low EV minus admin costs
        ev_low = ev_range[0] * (1 - administrative_cost_pct)
        low_dist = wf.distribute(ev_low)

        # High case: high EV minus admin costs
        ev_high = ev_range[1] * (1 - administrative_cost_pct)
        high_dist = wf.distribute(ev_high)

        recovery_by_class = {}
        for layer in capital_structure:
            low_r = low_dist.recoveries.get(layer.name, 0.0)
            high_r = high_dist.recoveries.get(layer.name, 0.0)
            recovery_by_class[layer.name] = (low_r, high_r)

        milestones = self.timeline()
        duration = milestones[-1].cumulative_months if milestones else 0.0

        return Chapter11Result(
            milestones=milestones,
            estimated_duration_months=duration,
            recovery_by_class=recovery_by_class,
            administrative_costs_pct=administrative_cost_pct,
        )
_serialisable("dip_loan", ["notional", "spread", "maturity_months", "roll_up_amount", "carve_out", "upfront_fee"])(DIPLoan)


# ═══════════════════════════════════════════════════════════════
# Distressed CDS: Upfront Quoting + Implied CPD
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistressedCDSResult:
    """Distressed CDS pricing result (upfront convention)."""
    upfront_pct: float          # upfront payment as % of notional
    running_spread: float       # fixed running coupon (100bp or 500bp)
    implied_cpd: float          # cumulative probability of default
    implied_hazard: float       # flat hazard rate
    recovery: float
    maturity_years: float

    def to_dict(self) -> dict:
        return vars(self)


def distressed_cds_upfront(
    market_spread: float,
    maturity_years: float = 5.0,
    running_coupon: float = 0.05,
    recovery: float = 0.40,
    discount_rate: float = 0.04,
    coupon_frequency: int = 4,
) -> DistressedCDSResult:
    """Convert distressed running spread to upfront payment.

    Distressed CDS trade with an upfront payment plus a fixed running
    coupon (typically 500bp for HY). The upfront is:

        upfront = (market_spread − running_coupon) × RPV01

    where RPV01 is the risky annuity.

    Also computes implied cumulative probability of default (CPD)
    and the implied flat hazard rate.

    Args:
        market_spread: quoted CDS spread (decimal, e.g. 0.05 = 500bp).
        maturity_years: CDS maturity.
        running_coupon: fixed running coupon (decimal).
        recovery: recovery rate.
        discount_rate: risk-free rate for discounting.
        coupon_frequency: payments per year.
    """
    import math

    dt = 1.0 / coupon_frequency
    n = int(maturity_years * coupon_frequency)

    # Implied hazard from market spread
    # spread ≈ hazard × (1 - recovery), so hazard ≈ spread / (1 - recovery)
    if recovery < 1.0:
        implied_hazard = market_spread / (1 - recovery)
    else:
        implied_hazard = 0.0

    # Compute protection leg and RPV01 with implied hazard
    rpv01 = 0.0
    prot_pv = 0.0
    prev_q = 1.0
    for i in range(1, n + 1):
        t = i * dt
        q = math.exp(-implied_hazard * t)
        df = math.exp(-discount_rate * t)
        rpv01 += dt * q * df
        prot_pv += (1 - recovery) * (prev_q - q) * df
        prev_q = q

    # Upfront = protection_leg - running_coupon × RPV01
    upfront = prot_pv - running_coupon * rpv01

    # Implied CPD = 1 - Q(T)
    cpd = 1.0 - math.exp(-implied_hazard * maturity_years)

    return DistressedCDSResult(
        upfront_pct=upfront,
        running_spread=running_coupon,
        implied_cpd=cpd,
        implied_hazard=implied_hazard,
        recovery=recovery,
        maturity_years=maturity_years,
    )


def implied_cpd_from_upfront(
    upfront_pct: float,
    maturity_years: float = 5.0,
    running_coupon: float = 0.05,
    recovery: float = 0.40,
    discount_rate: float = 0.04,
) -> float:
    """Implied cumulative default probability from upfront payment.

    Inverts the upfront formula to get the implied hazard rate,
    then computes CPD = 1 − exp(−λT).

    Uses Newton-Raphson to solve for λ.
    """
    import math

    coupon_frequency = 4
    dt = 1.0 / coupon_frequency
    n = int(maturity_years * coupon_frequency)

    def _upfront_for_hazard(h: float) -> float:
        rpv01 = 0.0
        prot = 0.0
        prev_q = 1.0
        for i in range(1, n + 1):
            t = i * dt
            q = math.exp(-h * t)
            df = math.exp(-discount_rate * t)
            rpv01 += dt * q * df
            prot += (1 - recovery) * (prev_q - q) * df
            prev_q = q
        return prot - running_coupon * rpv01

    # Newton-Raphson
    h = 0.05  # initial guess
    for _ in range(50):
        f = _upfront_for_hazard(h) - upfront_pct
        dh = 0.0001
        fp = (_upfront_for_hazard(h + dh) - _upfront_for_hazard(h - dh)) / (2 * dh)
        if abs(fp) < 1e-15:
            break
        h -= f / fp
        h = max(h, 1e-6)

    return 1.0 - math.exp(-h * maturity_years)


def distressed_basis(
    cds_upfront: float,
    bond_price: float,
    recovery: float = 0.40,
) -> float:
    """Distressed CDS-bond basis.

    basis = implied_bond_price_from_CDS − actual_bond_price

    implied_bond_price = par − cds_upfront × par
    (since upfront ≈ protection value as fraction of notional)

    Positive basis: CDS protection is more expensive than bond discount.
    Negative basis: bond trades wider than CDS (common in distress).

    Args:
        cds_upfront: upfront CDS payment as fraction of notional.
        bond_price: clean bond price (% of par, e.g. 65 for 65 cents).
        recovery: recovery rate.

    Returns:
        Basis in price points (% of par).
    """
    # Bond implied spread from price: spread ≈ (1 - price/100) / duration
    # CDS implied bond price: par - upfront - (1 - recovery) component
    implied_bond = 100 * (1 - cds_upfront)
    return implied_bond - bond_price
