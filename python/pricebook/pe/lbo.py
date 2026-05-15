"""LBO deal model: sources & uses, debt schedule, FCF, exit analysis.

Core PE underwriting tool for leveraged buyout deal structuring.

    from pricebook.pe.lbo import LBOModel

    model = LBOModel(
        enterprise_value=500_000_000,
        entry_ebitda=100_000_000,
        equity_pct=0.40,
        senior_debt_turns=4.0,
        mezz_debt_turns=1.0,
        ebitda_growth=0.05,
    )
    result = model.run()
    print(result.exit_analyses[0].equity_irr)

References:
    Rosenbaum & Pearl (2020). Investment Banking, 3rd ed.
    Pignataro (2013). Leveraged Buyouts.
    Kaplan & Strömberg (2009). Leveraged Buyouts and Private Equity.
"""

from __future__ import annotations

from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class SourcesAndUses:
    """LBO sources & uses of funds."""
    # Sources
    equity: float
    senior_debt: float
    mezzanine: float
    rollover_equity: float = 0.0

    # Uses
    enterprise_value: float = 0.0
    transaction_fees: float = 0.0
    financing_fees: float = 0.0
    cash_to_balance_sheet: float = 0.0

    @property
    def total_sources(self) -> float:
        return self.equity + self.senior_debt + self.mezzanine + self.rollover_equity

    @property
    def total_uses(self) -> float:
        return self.enterprise_value + self.transaction_fees + self.financing_fees + self.cash_to_balance_sheet

    def check_balance(self) -> float:
        """Sources minus uses — should be ~0."""
        return self.total_sources - self.total_uses

    def to_dict(self) -> dict:
        return {
            "equity": self.equity, "senior_debt": self.senior_debt,
            "mezzanine": self.mezzanine, "rollover_equity": self.rollover_equity,
            "total_sources": self.total_sources,
            "enterprise_value": self.enterprise_value,
            "transaction_fees": self.transaction_fees,
            "financing_fees": self.financing_fees,
            "cash_to_balance_sheet": self.cash_to_balance_sheet,
            "total_uses": self.total_uses,
            "balance": self.check_balance(),
        }


@dataclass
class FCFProjection:
    """Single year's free cash flow build."""
    year: int
    revenue: float
    ebitda: float
    da: float
    ebit: float
    taxes: float
    capex: float
    nwc_change: float
    fcf: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class DebtYear:
    """Debt schedule for a single year."""
    year: int
    # Senior
    senior_opening: float
    senior_interest: float
    senior_amort: float
    senior_sweep: float
    senior_closing: float
    # Mezzanine
    mezz_opening: float
    mezz_cash_interest: float
    mezz_pik_interest: float
    mezz_closing: float
    # Totals
    total_debt: float
    total_interest: float
    net_leverage: float  # total_debt / ebitda

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ExitAnalysis:
    """Exit scenario at a given year and multiple."""
    exit_year: int
    exit_ebitda: float
    exit_multiple: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    equity_irr: float
    cash_on_cash: float
    moic: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class LBOResult:
    """Complete LBO model output."""
    sources_and_uses: SourcesAndUses
    fcf_projections: list[FCFProjection]
    debt_schedule: list[DebtYear]
    exit_analyses: list[ExitAnalysis]
    entry_multiple: float
    equity_invested: float

    def to_dict(self) -> dict:
        return {
            "sources_and_uses": self.sources_and_uses.to_dict(),
            "fcf_projections": [f.to_dict() for f in self.fcf_projections],
            "debt_schedule": [d.to_dict() for d in self.debt_schedule],
            "exit_analyses": [e.to_dict() for e in self.exit_analyses],
            "entry_multiple": self.entry_multiple,
            "equity_invested": self.equity_invested,
        }


# ═══════════════════════════════════════════════════════════════
# LBO Model
# ═══════════════════════════════════════════════════════════════

class LBOModel:
    """Leveraged buyout deal model.

    Models a PE buyout from entry to exit: sources & uses, EBITDA projection,
    free cash flow build, debt schedule with amortisation and sweeps,
    and exit analysis at various multiples/hold periods.

    Args:
        enterprise_value: target acquisition EV.
        entry_ebitda: LTM EBITDA at acquisition.
        equity_pct: sponsor equity as % of total sources (default 40%).
        senior_debt_turns: senior debt / EBITDA (default 4.0x).
        mezz_debt_turns: mezzanine debt / EBITDA (default 1.0x).
        rollover_equity: management/seller rollover equity.
        transaction_fees_pct: advisory + legal as % of EV.
        financing_fees_pct: debt arrangement fees as % of total debt.
        ebitda_growth: annual EBITDA growth (scalar or per-year list).
        ebitda_margin: EBITDA / revenue.
        tax_rate: corporate tax rate.
        capex_pct_revenue: capex as % of revenue.
        da_pct_revenue: D&A as % of revenue.
        nwc_pct_revenue: NWC as % of revenue (change drives cash).
        senior_rate: senior debt all-in coupon rate.
        senior_amort_pct: annual amortisation as % of initial senior.
        mezz_cash_rate: mezzanine cash coupon rate.
        mezz_pik_rate: mezzanine PIK coupon rate.
        sweep_pct: excess cash flow sweep percentage (0 = no sweep).
        hold_period: maximum holding period for projections.
    """

    def __init__(
        self,
        enterprise_value: float,
        entry_ebitda: float,
        equity_pct: float = 0.40,
        senior_debt_turns: float = 4.0,
        mezz_debt_turns: float = 1.0,
        rollover_equity: float = 0.0,
        transaction_fees_pct: float = 0.02,
        financing_fees_pct: float = 0.02,
        ebitda_growth: float | list[float] = 0.05,
        ebitda_margin: float = 0.20,
        tax_rate: float = 0.25,
        capex_pct_revenue: float = 0.03,
        da_pct_revenue: float = 0.02,
        nwc_pct_revenue: float = 0.10,
        senior_rate: float = 0.06,
        senior_amort_pct: float = 0.01,
        mezz_cash_rate: float = 0.06,
        mezz_pik_rate: float = 0.04,
        sweep_pct: float = 0.50,
        hold_period: int = 5,
    ):
        if enterprise_value <= 0:
            raise ValueError(f"enterprise_value must be positive, got {enterprise_value}")
        if entry_ebitda <= 0:
            raise ValueError(f"entry_ebitda must be positive, got {entry_ebitda}")

        self.enterprise_value = enterprise_value
        self.entry_ebitda = entry_ebitda
        self.entry_multiple = enterprise_value / entry_ebitda
        self.equity_pct = equity_pct
        self.senior_debt_turns = senior_debt_turns
        self.mezz_debt_turns = mezz_debt_turns
        self.rollover_equity = rollover_equity
        self.transaction_fees_pct = transaction_fees_pct
        self.financing_fees_pct = financing_fees_pct
        self.ebitda_margin = ebitda_margin
        self.tax_rate = tax_rate
        self.capex_pct_revenue = capex_pct_revenue
        self.da_pct_revenue = da_pct_revenue
        self.nwc_pct_revenue = nwc_pct_revenue
        self.senior_rate = senior_rate
        self.senior_amort_pct = senior_amort_pct
        self.mezz_cash_rate = mezz_cash_rate
        self.mezz_pik_rate = mezz_pik_rate
        self.sweep_pct = sweep_pct
        self.hold_period = hold_period

        # Normalise growth to per-year list
        if isinstance(ebitda_growth, (int, float)):
            self._growth = [float(ebitda_growth)] * hold_period
        else:
            self._growth = list(ebitda_growth)
            while len(self._growth) < hold_period:
                self._growth.append(self._growth[-1] if self._growth else 0.0)

    # ── Sources & Uses ────────────────────────────────────────

    def sources_and_uses(self) -> SourcesAndUses:
        """Compute deal sources and uses."""
        senior = self.entry_ebitda * self.senior_debt_turns
        mezz = self.entry_ebitda * self.mezz_debt_turns
        total_debt = senior + mezz

        transaction_fees = self.enterprise_value * self.transaction_fees_pct
        financing_fees = total_debt * self.financing_fees_pct

        total_uses = self.enterprise_value + transaction_fees + financing_fees
        equity = total_uses - senior - mezz - self.rollover_equity

        return SourcesAndUses(
            equity=equity,
            senior_debt=senior,
            mezzanine=mezz,
            rollover_equity=self.rollover_equity,
            enterprise_value=self.enterprise_value,
            transaction_fees=transaction_fees,
            financing_fees=financing_fees,
        )

    # ── EBITDA & FCF Projection ───────────────────────────────

    def project_ebitda(self) -> list[float]:
        """Project EBITDA over the hold period."""
        ebitda = [self.entry_ebitda]
        for g in self._growth:
            ebitda.append(ebitda[-1] * (1 + g))
        return ebitda  # length = hold_period + 1 (year 0 = entry)

    def project_fcf(self) -> list[FCFProjection]:
        """Project free cash flows for each year."""
        ebitda_path = self.project_ebitda()
        projections = []
        prev_nwc = ebitda_path[0] / self.ebitda_margin * self.nwc_pct_revenue

        for yr in range(1, self.hold_period + 1):
            ebitda = ebitda_path[yr]
            revenue = ebitda / self.ebitda_margin if self.ebitda_margin > 0 else ebitda

            da = revenue * self.da_pct_revenue
            ebit = ebitda - da
            taxes = max(ebit * self.tax_rate, 0.0)
            capex = revenue * self.capex_pct_revenue

            current_nwc = revenue * self.nwc_pct_revenue
            nwc_change = current_nwc - prev_nwc
            prev_nwc = current_nwc

            fcf = ebitda - taxes - capex - nwc_change

            projections.append(FCFProjection(
                year=yr, revenue=revenue, ebitda=ebitda, da=da,
                ebit=ebit, taxes=taxes, capex=capex,
                nwc_change=nwc_change, fcf=fcf,
            ))
        return projections

    # ── Debt Schedule ─────────────────────────────────────────

    def debt_schedule(self) -> list[DebtYear]:
        """Build annual debt schedule with amortisation, PIK, and sweeps."""
        su = self.sources_and_uses()
        fcfs = self.project_fcf()
        ebitda_path = self.project_ebitda()

        senior = su.senior_debt
        mezz = su.mezzanine
        initial_senior = senior
        schedule = []

        for yr in range(1, self.hold_period + 1):
            fcf_yr = fcfs[yr - 1]

            # Senior interest
            senior_interest = senior * self.senior_rate

            # Senior scheduled amortisation
            senior_amort = min(initial_senior * self.senior_amort_pct, senior)

            # Excess cash flow sweep on senior
            senior_sweep = 0.0
            if self.sweep_pct > 0 and senior > 0:
                debt_service = senior_interest + senior_amort
                excess = fcf_yr.fcf - debt_service - (mezz * self.mezz_cash_rate)
                senior_sweep = min(max(excess * self.sweep_pct, 0.0), senior - senior_amort)

            senior_closing = senior - senior_amort - senior_sweep

            # Mezzanine: cash interest + PIK (PIK capitalises)
            mezz_cash = mezz * self.mezz_cash_rate
            mezz_pik = mezz * self.mezz_pik_rate
            mezz_closing = mezz + mezz_pik  # PIK adds to principal

            total_debt = senior_closing + mezz_closing
            total_interest = senior_interest + mezz_cash  # PIK is non-cash
            net_lev = total_debt / ebitda_path[yr] if ebitda_path[yr] > 0 else float('inf')

            schedule.append(DebtYear(
                year=yr,
                senior_opening=senior, senior_interest=senior_interest,
                senior_amort=senior_amort, senior_sweep=senior_sweep,
                senior_closing=senior_closing,
                mezz_opening=mezz, mezz_cash_interest=mezz_cash,
                mezz_pik_interest=mezz_pik, mezz_closing=mezz_closing,
                total_debt=total_debt, total_interest=total_interest,
                net_leverage=net_lev,
            ))

            senior = senior_closing
            mezz = mezz_closing

        return schedule

    # ── Exit Analysis ─────────────────────────────────────────

    def exit_analysis(self, exit_multiple: float, exit_year: int | None = None) -> ExitAnalysis:
        """Analyse exit at given multiple and year.

        Args:
            exit_multiple: EV / EBITDA at exit.
            exit_year: year of exit (default = hold_period).
        """
        if exit_year is None:
            exit_year = self.hold_period

        if exit_year < 1 or exit_year > self.hold_period:
            raise ValueError(f"exit_year {exit_year} must be in [1, {self.hold_period}]")

        ebitda_path = self.project_ebitda()
        debt_sched = self.debt_schedule()
        su = self.sources_and_uses()

        exit_ebitda = ebitda_path[exit_year]
        ev = exit_ebitda * exit_multiple
        net_debt = debt_sched[exit_year - 1].total_debt
        equity_value = max(ev - net_debt, 0.0)

        equity_invested = su.equity + su.rollover_equity
        moic = equity_value / equity_invested if equity_invested > 0 else 0.0
        cash_on_cash = moic  # simplified (no interim dividends)

        # Equity IRR: invest equity_invested at t=0, receive equity_value at t=exit_year
        equity_irr = _irr_simple(equity_invested, equity_value, exit_year)

        return ExitAnalysis(
            exit_year=exit_year, exit_ebitda=exit_ebitda,
            exit_multiple=exit_multiple, enterprise_value=ev,
            net_debt=net_debt, equity_value=equity_value,
            equity_irr=equity_irr, cash_on_cash=cash_on_cash,
            moic=moic,
        )

    # ── Full Run ──────────────────────────────────────────────

    def run(
        self,
        exit_multiples: list[float] | None = None,
    ) -> LBOResult:
        """Run full LBO model with exit analyses at various multiples.

        Args:
            exit_multiples: list of exit multiples to analyse.
                Defaults to entry multiple ± 2 turns.
        """
        if exit_multiples is None:
            base = self.entry_multiple
            exit_multiples = [base - 2, base - 1, base, base + 1, base + 2]
            exit_multiples = [m for m in exit_multiples if m > 0]

        su = self.sources_and_uses()
        fcfs = self.project_fcf()
        ds = self.debt_schedule()

        exits = []
        for m in exit_multiples:
            exits.append(self.exit_analysis(m))

        return LBOResult(
            sources_and_uses=su,
            fcf_projections=fcfs,
            debt_schedule=ds,
            exit_analyses=exits,
            entry_multiple=self.entry_multiple,
            equity_invested=su.equity + su.rollover_equity,
        )

    # ── Sensitivity Table ─────────────────────────────────────

    def sensitivity_table(
        self,
        row_param: str = "exit_multiple",
        col_param: str = "hold_period",
        row_values: list[float] | None = None,
        col_values: list[float] | None = None,
    ) -> list[list[float]]:
        """Generate IRR sensitivity table across two parameters.

        Args:
            row_param: "exit_multiple" or "ebitda_growth".
            col_param: "hold_period" or "exit_multiple" or "ebitda_growth".
            row_values: values for rows.
            col_values: values for columns.

        Returns:
            2D list of equity IRRs: result[i][j] = IRR(row_values[i], col_values[j]).
        """
        base = self.entry_multiple
        if row_values is None:
            if row_param == "exit_multiple":
                row_values = [base - 2, base - 1, base, base + 1, base + 2]
                row_values = [m for m in row_values if m > 0]
            else:
                row_values = [0.0, 0.03, 0.05, 0.07, 0.10]

        if col_values is None:
            if col_param == "hold_period":
                col_values = [3.0, 4.0, 5.0, 6.0, 7.0]
            elif col_param == "exit_multiple":
                col_values = [base - 2, base - 1, base, base + 1, base + 2]
                col_values = [m for m in col_values if m > 0]
            else:
                col_values = [0.0, 0.03, 0.05, 0.07, 0.10]

        grid = []
        for rv in row_values:
            row = []
            for cv in col_values:
                irr = self._compute_irr_for_params(row_param, rv, col_param, cv)
                row.append(irr)
            grid.append(row)
        return grid

    def _compute_irr_for_params(self, p1: str, v1: float, p2: str, v2: float) -> float:
        """Compute equity IRR for a given pair of parameter overrides."""
        params = {p1: v1, p2: v2}
        exit_mult = params.get("exit_multiple", self.entry_multiple)
        hold = int(params.get("hold_period", self.hold_period))
        growth = params.get("ebitda_growth", self._growth[0] if self._growth else 0.05)

        # Rebuild with overridden growth if needed
        if "ebitda_growth" in params:
            model = LBOModel(
                enterprise_value=self.enterprise_value,
                entry_ebitda=self.entry_ebitda,
                equity_pct=self.equity_pct,
                senior_debt_turns=self.senior_debt_turns,
                mezz_debt_turns=self.mezz_debt_turns,
                rollover_equity=self.rollover_equity,
                transaction_fees_pct=self.transaction_fees_pct,
                financing_fees_pct=self.financing_fees_pct,
                ebitda_growth=growth,
                ebitda_margin=self.ebitda_margin,
                tax_rate=self.tax_rate,
                capex_pct_revenue=self.capex_pct_revenue,
                da_pct_revenue=self.da_pct_revenue,
                nwc_pct_revenue=self.nwc_pct_revenue,
                senior_rate=self.senior_rate,
                senior_amort_pct=self.senior_amort_pct,
                mezz_cash_rate=self.mezz_cash_rate,
                mezz_pik_rate=self.mezz_pik_rate,
                sweep_pct=self.sweep_pct,
                hold_period=max(hold, self.hold_period),
            )
        else:
            model = self

        hold = min(hold, model.hold_period)
        try:
            ea = model.exit_analysis(exit_mult, hold)
            return ea.equity_irr
        except (ValueError, ZeroDivisionError):
            return float('nan')


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _irr_simple(invested: float, terminal: float, years: int) -> float:
    """Simple IRR for single investment → single exit.

    IRR = (terminal / invested)^(1/years) - 1
    """
    if invested <= 0 or terminal <= 0 or years <= 0:
        return 0.0
    return (terminal / invested) ** (1.0 / years) - 1.0
