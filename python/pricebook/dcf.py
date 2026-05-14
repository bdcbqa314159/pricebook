"""DCF and enterprise valuation: WACC, terminal value, EV bridge.

    from pricebook.dcf import DCFModel, WACCInputs, compute_wacc

    wacc = WACCInputs(risk_free_rate=0.04, equity_risk_premium=0.05,
                       beta=1.2, cost_of_debt=0.05, tax_rate=0.25,
                       debt_to_total=0.40)
    model = DCFModel(fcfs=[50, 55, 60, 65, 70], wacc_inputs=wacc,
                      terminal_growth=0.02, net_debt=200)
    result = model.value()
    print(result.ev_bridge.equity_value)

References:
    Damodaran (2012). Investment Valuation, 3rd ed.
    Koller, Goedhart & Wessels (2020). Valuation, 7th ed.
    Berk & DeMarzo (2019). Corporate Finance, 5th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class WACCInputs:
    """Weighted average cost of capital inputs.

    WACC = E/(D+E) × Re + D/(D+E) × Rd × (1 - tax)
    Re = Rf + beta × ERP  (CAPM)
    """
    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    tax_rate: float
    debt_to_total: float  # D / (D + E)

    @property
    def cost_of_equity(self) -> float:
        """CAPM cost of equity: Rf + beta × ERP."""
        return self.risk_free_rate + self.beta * self.equity_risk_premium

    @property
    def equity_to_total(self) -> float:
        return 1.0 - self.debt_to_total

    @property
    def wacc(self) -> float:
        re = self.cost_of_equity
        rd_after_tax = self.cost_of_debt * (1 - self.tax_rate)
        return self.equity_to_total * re + self.debt_to_total * rd_after_tax

    def to_dict(self) -> dict:
        return {
            "risk_free_rate": self.risk_free_rate,
            "equity_risk_premium": self.equity_risk_premium,
            "beta": self.beta,
            "cost_of_equity": self.cost_of_equity,
            "cost_of_debt": self.cost_of_debt,
            "tax_rate": self.tax_rate,
            "debt_to_total": self.debt_to_total,
            "wacc": self.wacc,
        }


@dataclass
class TerminalValue:
    """Terminal value result."""
    method: str  # "perpetuity_growth" or "exit_multiple"
    value: float
    terminal_year_fcf: float
    growth_rate: float | None = None
    exit_multiple: float | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method, "value": self.value,
            "terminal_year_fcf": self.terminal_year_fcf,
            "growth_rate": self.growth_rate,
            "exit_multiple": self.exit_multiple,
        }


@dataclass
class EVBridge:
    """Enterprise value to equity value bridge."""
    enterprise_value: float
    net_debt: float
    minority_interest: float = 0.0
    associates: float = 0.0
    equity_value: float = 0.0
    equity_value_per_share: float | None = None

    def to_dict(self) -> dict:
        return {
            "enterprise_value": self.enterprise_value,
            "net_debt": self.net_debt,
            "minority_interest": self.minority_interest,
            "associates": self.associates,
            "equity_value": self.equity_value,
            "equity_value_per_share": self.equity_value_per_share,
        }


@dataclass
class DCFResult:
    """DCF valuation result."""
    pv_fcfs: float
    pv_terminal: float
    enterprise_value: float
    ev_bridge: EVBridge
    wacc: float
    terminal_value: TerminalValue
    scenario: str = "base"

    def to_dict(self) -> dict:
        return {
            "pv_fcfs": self.pv_fcfs,
            "pv_terminal": self.pv_terminal,
            "enterprise_value": self.enterprise_value,
            "ev_bridge": self.ev_bridge.to_dict(),
            "wacc": self.wacc,
            "terminal_value": self.terminal_value.to_dict(),
            "scenario": self.scenario,
        }


@dataclass
class FootballField:
    """Range of valuations from multiple methods."""
    methods: list[str]
    low: list[float]
    mid: list[float]
    high: list[float]

    def to_dict(self) -> dict:
        return {
            "methods": self.methods,
            "low": self.low,
            "mid": self.mid,
            "high": self.high,
        }


# ═══════════════════════════════════════════════════════════════
# Standalone functions
# ═══════════════════════════════════════════════════════════════

def compute_wacc(inputs: WACCInputs) -> float:
    """Compute WACC from inputs."""
    return inputs.wacc


def terminal_value_perpetuity(
    fcf: float,
    wacc: float,
    growth: float,
) -> TerminalValue:
    """Gordon growth model terminal value.

    TV = FCF × (1 + g) / (WACC - g)

    Args:
        fcf: last projected free cash flow.
        wacc: discount rate.
        growth: perpetuity growth rate (must be < wacc).

    Raises:
        ValueError: if growth >= wacc (infinite value).
    """
    if growth >= wacc:
        raise ValueError(
            f"growth ({growth:.2%}) must be < wacc ({wacc:.2%}) for convergence"
        )
    tv = fcf * (1 + growth) / (wacc - growth)
    return TerminalValue(
        method="perpetuity_growth", value=tv,
        terminal_year_fcf=fcf, growth_rate=growth,
    )


def terminal_value_exit_multiple(
    ebitda: float,
    multiple: float,
) -> TerminalValue:
    """Exit multiple terminal value.

    TV = terminal EBITDA × exit multiple

    Args:
        ebitda: terminal year EBITDA.
        multiple: EV/EBITDA exit multiple.
    """
    tv = ebitda * multiple
    return TerminalValue(
        method="exit_multiple", value=tv,
        terminal_year_fcf=ebitda, exit_multiple=multiple,
    )


def ev_to_equity(
    ev: float,
    net_debt: float,
    minority: float = 0.0,
    associates: float = 0.0,
    shares: float | None = None,
) -> EVBridge:
    """Bridge from enterprise value to equity value.

    Equity = EV - net debt - minorities + associates
    """
    equity = ev - net_debt - minority + associates
    per_share = equity / shares if shares and shares > 0 else None

    return EVBridge(
        enterprise_value=ev, net_debt=net_debt,
        minority_interest=minority, associates=associates,
        equity_value=equity, equity_value_per_share=per_share,
    )


# ═══════════════════════════════════════════════════════════════
# DCF Model
# ═══════════════════════════════════════════════════════════════

class DCFModel:
    """Discounted cash flow valuation model.

    Discounts explicit-period FCFs and a terminal value to compute
    enterprise value, then bridges to equity value.

    Args:
        fcfs: projected free cash flows for the explicit forecast period.
        wacc_inputs: WACC calculation inputs.
        terminal_growth: perpetuity growth rate for Gordon model.
        terminal_ebitda: terminal year EBITDA (for exit multiple method).
        terminal_multiple: EV/EBITDA exit multiple (for exit multiple method).
        net_debt: total debt minus cash for EV→equity bridge.
        minority_interest: minority interest for bridge.
        associates: associate interests for bridge.
        shares_outstanding: for per-share equity value.
    """

    def __init__(
        self,
        fcfs: list[float],
        wacc_inputs: WACCInputs,
        terminal_growth: float = 0.02,
        terminal_ebitda: float | None = None,
        terminal_multiple: float | None = None,
        net_debt: float = 0.0,
        minority_interest: float = 0.0,
        associates: float = 0.0,
        shares_outstanding: float | None = None,
    ):
        if not fcfs:
            raise ValueError("fcfs must not be empty")
        self.fcfs = list(fcfs)
        self.wacc_inputs = wacc_inputs
        self.terminal_growth = terminal_growth
        self.terminal_ebitda = terminal_ebitda
        self.terminal_multiple = terminal_multiple
        self.net_debt = net_debt
        self.minority_interest = minority_interest
        self.associates = associates
        self.shares_outstanding = shares_outstanding

    def value(self, method: str = "perpetuity_growth") -> DCFResult:
        """Compute DCF valuation.

        Args:
            method: "perpetuity_growth" (Gordon model) or "exit_multiple".
        """
        wacc = self.wacc_inputs.wacc
        n = len(self.fcfs)

        # PV of explicit period FCFs
        pv_fcfs = sum(
            fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(self.fcfs)
        )

        # Terminal value
        if method == "exit_multiple":
            if self.terminal_ebitda is None or self.terminal_multiple is None:
                raise ValueError(
                    "terminal_ebitda and terminal_multiple required for exit_multiple method"
                )
            tv = terminal_value_exit_multiple(self.terminal_ebitda, self.terminal_multiple)
        else:
            tv = terminal_value_perpetuity(self.fcfs[-1], wacc, self.terminal_growth)

        # PV of terminal value (discounted back to today)
        pv_tv = tv.value / (1 + wacc) ** n

        ev = pv_fcfs + pv_tv
        bridge = ev_to_equity(
            ev, self.net_debt, self.minority_interest,
            self.associates, self.shares_outstanding,
        )

        return DCFResult(
            pv_fcfs=pv_fcfs, pv_terminal=pv_tv,
            enterprise_value=ev, ev_bridge=bridge,
            wacc=wacc, terminal_value=tv,
        )

    def scenario_analysis(
        self,
        scenarios: dict[str, dict],
        weights: dict[str, float] | None = None,
    ) -> list[DCFResult]:
        """Run multiple scenarios with optional probability weights.

        Args:
            scenarios: {name: {field: value}} overrides.
                Supported fields: fcfs, terminal_growth, terminal_ebitda,
                terminal_multiple, net_debt.
            weights: {name: probability} (for reference, not used in calc).

        Returns:
            List of DCFResult, one per scenario.
        """
        results = []
        for name, overrides in scenarios.items():
            fcfs = overrides.get("fcfs", self.fcfs)
            tg = overrides.get("terminal_growth", self.terminal_growth)
            te = overrides.get("terminal_ebitda", self.terminal_ebitda)
            tm = overrides.get("terminal_multiple", self.terminal_multiple)
            nd = overrides.get("net_debt", self.net_debt)
            method = overrides.get("method", "perpetuity_growth")

            model = DCFModel(
                fcfs=fcfs,
                wacc_inputs=self.wacc_inputs,
                terminal_growth=tg,
                terminal_ebitda=te,
                terminal_multiple=tm,
                net_debt=nd,
                minority_interest=self.minority_interest,
                associates=self.associates,
                shares_outstanding=self.shares_outstanding,
            )
            r = model.value(method=method)
            r.scenario = name
            results.append(r)
        return results

    def football_field(self) -> FootballField:
        """Generate valuation range from multiple methods.

        Methods: perpetuity (base ± 1% growth), exit multiple (if configured),
        and WACC sensitivity (±1%).
        """
        wacc = self.wacc_inputs.wacc
        methods = []
        lows = []
        mids = []
        highs = []

        # 1. DCF perpetuity growth
        try:
            mid = self.value("perpetuity_growth")
            low_g = max(self.terminal_growth - 0.01, 0.0)
            high_g = self.terminal_growth + 0.01
            if high_g < wacc:
                m_low = DCFModel(self.fcfs, self.wacc_inputs, low_g,
                                 net_debt=self.net_debt).value()
                m_high = DCFModel(self.fcfs, self.wacc_inputs, high_g,
                                  net_debt=self.net_debt).value()
                methods.append("DCF (perpetuity)")
                lows.append(m_low.ev_bridge.equity_value)
                mids.append(mid.ev_bridge.equity_value)
                highs.append(m_high.ev_bridge.equity_value)
        except ValueError:
            pass

        # 2. DCF exit multiple (if configured)
        if self.terminal_ebitda is not None and self.terminal_multiple is not None:
            mult = self.terminal_multiple
            for low_m, mid_m, high_m in [(mult - 1, mult, mult + 1)]:
                pass
            m_low = DCFModel(self.fcfs, self.wacc_inputs, net_debt=self.net_debt,
                             terminal_ebitda=self.terminal_ebitda,
                             terminal_multiple=mult - 1).value("exit_multiple")
            m_mid = DCFModel(self.fcfs, self.wacc_inputs, net_debt=self.net_debt,
                             terminal_ebitda=self.terminal_ebitda,
                             terminal_multiple=mult).value("exit_multiple")
            m_high = DCFModel(self.fcfs, self.wacc_inputs, net_debt=self.net_debt,
                              terminal_ebitda=self.terminal_ebitda,
                              terminal_multiple=mult + 1).value("exit_multiple")
            methods.append("DCF (exit multiple)")
            lows.append(m_low.ev_bridge.equity_value)
            mids.append(m_mid.ev_bridge.equity_value)
            highs.append(m_high.ev_bridge.equity_value)

        # 3. WACC sensitivity
        try:
            from copy import copy
            wacc_low = WACCInputs(
                self.wacc_inputs.risk_free_rate,
                self.wacc_inputs.equity_risk_premium,
                self.wacc_inputs.beta * 0.9,
                self.wacc_inputs.cost_of_debt,
                self.wacc_inputs.tax_rate,
                self.wacc_inputs.debt_to_total,
            )
            wacc_high = WACCInputs(
                self.wacc_inputs.risk_free_rate,
                self.wacc_inputs.equity_risk_premium,
                self.wacc_inputs.beta * 1.1,
                self.wacc_inputs.cost_of_debt,
                self.wacc_inputs.tax_rate,
                self.wacc_inputs.debt_to_total,
            )
            m_low_w = DCFModel(self.fcfs, wacc_low, self.terminal_growth,
                               net_debt=self.net_debt).value()
            m_mid_w = self.value()
            m_high_w = DCFModel(self.fcfs, wacc_high, self.terminal_growth,
                                net_debt=self.net_debt).value()
            methods.append("WACC sensitivity")
            # Lower beta → lower WACC → higher value
            lows.append(m_high_w.ev_bridge.equity_value)
            mids.append(m_mid_w.ev_bridge.equity_value)
            highs.append(m_low_w.ev_bridge.equity_value)
        except ValueError:
            pass

        return FootballField(methods=methods, low=lows, mid=mids, high=highs)
