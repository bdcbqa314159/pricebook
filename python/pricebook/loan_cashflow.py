"""Loan cashflow features: SOFR floor, excess cash flow sweep, PIK toggle,
pricing grid, SOFR CSA adjustment.

These are the building blocks that make leveraged loan cashflows realistic.
Each can be composed with TermLoan or used standalone.

    from pricebook.loan_cashflow import (
        FlooredTermLoan, ExcessCashFlowSweep, PIKTermLoan,
        PricingGrid, SOFRCSAAdjustment,
    )

References:
    LSTA (2022). The Handbook of Loan Syndications and Trading.
    S&P LCD. Leveraged Loan Primer, 2023.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.loan import TermLoan
from pricebook.schedule import Frequency, generate_schedule
from pricebook.serialisable import _register, _serialise_atom


# ---------------------------------------------------------------------------
# P1.5: SOFR CSA adjustment
# ---------------------------------------------------------------------------

# ISDA published credit spread adjustments (bps) for LIBOR→SOFR transition
SOFR_CSA_BPS = {
    1: 11.448,    # 1M LIBOR → SOFR + 11.448bp
    3: 26.161,    # 3M LIBOR → SOFR + 26.161bp
    6: 42.826,    # 6M LIBOR → SOFR + 42.826bp
    12: 71.513,   # 12M LIBOR → SOFR + 71.513bp
}


@dataclass
class SOFRCSAAdjustment:
    """SOFR credit spread adjustment for legacy LIBOR loans.

    coupon = SOFR + CSA + spread  (instead of LIBOR + spread)
    """
    tenor_months: int = 3

    @property
    def csa_bps(self) -> float:
        return SOFR_CSA_BPS.get(self.tenor_months, 26.161)

    @property
    def csa_rate(self) -> float:
        return self.csa_bps / 10_000


# ---------------------------------------------------------------------------
# P1.4: Pricing grid
# ---------------------------------------------------------------------------

class PricingGrid:
    """Dynamic spread based on borrower's leverage ratio.

    Spread steps up/down as leverage changes. Lender is short this optionality.

    Args:
        leverage_levels: sorted breakpoints [3.0, 4.0, 5.0].
        spreads: spread at each level [0.020, 0.025, 0.030, 0.035].
            Length = len(leverage_levels) + 1 (includes above highest level).
    """

    def __init__(
        self,
        leverage_levels: list[float],
        spreads: list[float],
    ):
        if len(spreads) != len(leverage_levels) + 1:
            raise ValueError(f"spreads length ({len(spreads)}) must be leverage_levels + 1 ({len(leverage_levels) + 1})")
        self.leverage_levels = sorted(leverage_levels)
        self.spreads = spreads

    def grid_spread(self, leverage: float) -> float:
        """Look up spread for current leverage ratio."""
        for i, level in enumerate(self.leverage_levels):
            if leverage <= level:
                return self.spreads[i]
        return self.spreads[-1]  # above highest level

    def to_dict(self) -> dict:
        return {"leverage_levels": self.leverage_levels, "spreads": self.spreads}

    @classmethod
    def from_dict(cls, d: dict) -> PricingGrid:
        return cls(d["leverage_levels"], d["spreads"])


# ---------------------------------------------------------------------------
# P1.2: Excess cash flow sweep
# ---------------------------------------------------------------------------

@dataclass
class ExcessCashFlowSweep:
    """Mandatory prepayment from excess cash flow.

    sweep_amount = max(EBITDA - capex - debt_service - threshold, 0) × sweep_pct

    Args:
        sweep_pct: fraction of excess cash swept (typically 0.50-0.75).
        capex_budget: expected capex (annual).
        debt_service: annual scheduled principal + interest.
    """
    sweep_pct: float = 0.50
    capex_budget: float = 0.0
    debt_service: float = 0.0

    def mandatory_prepayment(self, ebitda: float) -> float:
        """Compute mandatory prepayment from excess cash flow.

        excess = EBITDA - capex - debt_service
        prepay = max(excess, 0) × sweep_pct
        """
        excess = ebitda - self.capex_budget - self.debt_service
        return max(excess, 0.0) * self.sweep_pct

    def to_dict(self) -> dict:
        return {"sweep_pct": self.sweep_pct, "capex_budget": self.capex_budget,
                "debt_service": self.debt_service}


# ---------------------------------------------------------------------------
# P1.1: Floored term loan
# ---------------------------------------------------------------------------

class FlooredTermLoan(TermLoan):
    """Term loan with embedded SOFR floor.

    coupon = max(forward_rate, floor_rate) + spread

    The floor is an embedded put option on interest rates. When SOFR < floor,
    the lender receives extra carry. This changes the DV01 profile:
    below the floor, the loan behaves like a fixed-rate instrument.

    Args:
        floor_rate: minimum floating rate (e.g. 0.01 = 1% floor).
        csa: optional SOFR CSA adjustment for legacy loans.
        pricing_grid: optional dynamic spread grid.
    """

    _SERIAL_TYPE = "floored_term_loan"

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.03,
        notional: float = 1_000_000.0,
        amort_rate: float = 0.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        floor_rate: float = 0.0,
        csa: SOFRCSAAdjustment | None = None,
        pricing_grid: PricingGrid | None = None,
    ):
        super().__init__(start, end, spread, notional, amort_rate, frequency, day_count)
        self.floor_rate = floor_rate
        self.csa = csa
        self.pricing_grid = pricing_grid

    def cashflows(
        self,
        projection_curve: DiscountCurve,
        leverage_path: list[float] | None = None,
    ) -> list[tuple[date, float, float]]:
        """Generate cashflows with floor and optional grid/CSA.

        Args:
            leverage_path: leverage at each period (for pricing grid).
                If None and grid exists, uses a flat leverage.
        """
        outstanding = self.notional
        amort_amount = self.notional * self.amort_rate
        flows = []
        csa_adj = self.csa.csa_rate if self.csa else 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df1 = projection_curve.df(t_start)
            df2 = projection_curve.df(t_end)
            fwd = (df1 - df2) / (yf * df2)

            # Apply floor
            effective_rate = max(fwd + csa_adj, self.floor_rate)

            # Apply pricing grid if available
            if self.pricing_grid is not None and leverage_path is not None:
                period_idx = min(i - 1, len(leverage_path) - 1)
                current_spread = self.pricing_grid.grid_spread(leverage_path[period_idx])
            else:
                current_spread = self.spread

            interest = outstanding * (effective_rate + current_spread) * yf

            if i == len(self.schedule) - 1:
                principal = outstanding
            else:
                principal = min(amort_amount, outstanding)

            flows.append((t_end, interest, principal))
            outstanding -= principal
            outstanding = max(outstanding, 0.0)

        return flows

    def floor_value(
        self,
        projection_curve: DiscountCurve,
        vol: float = 0.50,
    ) -> float:
        """Value of the embedded floor (sum of floorlets).

        Each period's floorlet: max(floor - forward, 0) × τ × notional × df.
        Priced via Black-76.
        """
        outstanding = self.notional
        total = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)

            df1 = projection_curve.df(t_start)
            df2 = projection_curve.df(t_end)
            fwd = (df1 - df2) / (yf * df2)

            # Time to fixing
            t_fix = year_fraction(projection_curve.reference_date, t_start,
                                   DayCountConvention.ACT_365_FIXED)
            if t_fix <= 0:
                # Already fixed
                floorlet = max(self.floor_rate - fwd, 0.0) * yf * outstanding
            else:
                # Black-76 floorlet (put on rate)
                floorlet_unit = black76_price(fwd, self.floor_rate, vol, t_fix, 1.0,
                                               OptionType.PUT)
                floorlet = floorlet_unit * yf * outstanding

            total += floorlet * projection_curve.df(t_end)
            outstanding -= min(self.notional * self.amort_rate, outstanding)
            outstanding = max(outstanding, 0.0)

        return total

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self._SERIAL_TYPE, "params": {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "notional": self.notional,
            "amort_rate": self.amort_rate,
            "frequency": _serialise_atom(self.frequency),
            "day_count": _serialise_atom(self.day_count),
            "floor_rate": self.floor_rate,
        }}
        if self.csa:
            d["params"]["csa_tenor_months"] = self.csa.tenor_months
        if self.pricing_grid:
            d["params"]["pricing_grid"] = self.pricing_grid.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FlooredTermLoan:
        p = d["params"]
        csa = SOFRCSAAdjustment(p["csa_tenor_months"]) if "csa_tenor_months" in p else None
        grid = PricingGrid.from_dict(p["pricing_grid"]) if "pricing_grid" in p else None
        return cls(
            start=date.fromisoformat(p["start"]), end=date.fromisoformat(p["end"]),
            spread=p["spread"], notional=p.get("notional", 1_000_000.0),
            amort_rate=p.get("amort_rate", 0.0),
            frequency=Frequency(p.get("frequency", 3)),
            day_count=DayCountConvention(p.get("day_count", "ACT/360")),
            floor_rate=p.get("floor_rate", 0.0),
            csa=csa, pricing_grid=grid,
        )


_register(FlooredTermLoan)


# ---------------------------------------------------------------------------
# P1.3: PIK term loan
# ---------------------------------------------------------------------------

class PIKTermLoan(TermLoan):
    """Term loan with PIK (pay-in-kind) toggle.

    During PIK periods, interest capitalises (adds to outstanding) instead
    of being paid in cash. After toggle date, normal cash-pay resumes.

    Args:
        pik_rate: PIK interest rate (may differ from cash-pay spread).
        pik_end: date when PIK period ends and cash-pay resumes.
    """

    _SERIAL_TYPE = "pik_term_loan"

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.03,
        notional: float = 1_000_000.0,
        amort_rate: float = 0.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        pik_rate: float = 0.04,
        pik_end: date | None = None,
    ):
        super().__init__(start, end, spread, notional, amort_rate, frequency, day_count)
        self.pik_rate = pik_rate
        self.pik_end = pik_end or end  # PIK for entire life if not specified

    def cashflows(
        self,
        projection_curve: DiscountCurve,
    ) -> list[tuple[date, float, float]]:
        """Generate cashflows with PIK toggle.

        During PIK: cash_interest = 0, outstanding grows by PIK amount.
        After PIK: normal cash-pay on (now higher) outstanding.
        """
        outstanding = self.notional
        amort_amount = self.notional * self.amort_rate
        flows = []

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df1 = projection_curve.df(t_start)
            df2 = projection_curve.df(t_end)
            fwd = (df1 - df2) / (yf * df2)

            is_pik = t_end <= self.pik_end

            if is_pik:
                # PIK: interest capitalises, no cash coupon
                pik_interest = outstanding * self.pik_rate * yf
                outstanding += pik_interest  # outstanding grows
                cash_interest = 0.0
            else:
                # Cash-pay: normal coupon on (now higher) outstanding
                cash_interest = outstanding * (fwd + self.spread) * yf

            if i == len(self.schedule) - 1:
                principal = outstanding
            else:
                principal = min(amort_amount, outstanding)

            flows.append((t_end, cash_interest, principal))
            if not is_pik:
                outstanding -= principal
                outstanding = max(outstanding, 0.0)

        return flows

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {"type": self._SERIAL_TYPE, "params": {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "spread": self.spread, "notional": self.notional,
            "amort_rate": self.amort_rate,
            "frequency": _serialise_atom(self.frequency),
            "day_count": _serialise_atom(self.day_count),
            "pik_rate": self.pik_rate,
            "pik_end": self.pik_end.isoformat() if self.pik_end else None,
        }}

    @classmethod
    def from_dict(cls, d: dict) -> PIKTermLoan:
        p = d["params"]
        return cls(
            start=date.fromisoformat(p["start"]), end=date.fromisoformat(p["end"]),
            spread=p["spread"], notional=p.get("notional", 1_000_000.0),
            amort_rate=p.get("amort_rate", 0.0),
            frequency=Frequency(p.get("frequency", 3)),
            day_count=DayCountConvention(p.get("day_count", "ACT/360")),
            pik_rate=p.get("pik_rate", 0.04),
            pik_end=date.fromisoformat(p["pik_end"]) if p.get("pik_end") else None,
        )


_register(PIKTermLoan)
