"""Targeted tests to boost coverage in ts/_stats, ts/_rolling, structured/rates_structured,
structured/structured_notes, and pe/ modules."""

import pytest
import math
import numpy as np
from datetime import date


# ═══════════════════════════════════════════════════════════════
# ts/_stats.py — edge cases and untested functions
# ═══════════════════════════════════════════════════════════════

class TestTSStats:
    def _make_ts(self, values):
        from pricebook.ts._core import TimeSeries
        dates = np.array([date(2024, 1, i + 1) for i in range(len(values))])
        return TimeSeries(dates, np.array(values, dtype=float))

    def test_mean_empty(self):
        from pricebook.ts._stats import mean
        ts = self._make_ts([])
        assert mean(ts) == 0.0

    def test_sharpe_zero_vol(self):
        from pricebook.ts._stats import sharpe
        ts = self._make_ts([0.01] * 10)  # constant returns → zero vol
        assert sharpe(ts) == 0.0

    def test_sortino(self):
        from pricebook.ts._stats import sortino
        ts = self._make_ts([0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015, 0.02, -0.008])
        s = sortino(ts)
        assert s != 0  # should compute a value

    def test_sortino_no_downside(self):
        from pricebook.ts._stats import sortino
        ts = self._make_ts([0.01, 0.02, 0.03, 0.04, 0.05])  # all positive
        s = sortino(ts)
        assert s == 0.0  # no downside → zero sortino

    def test_recovery_time_empty(self):
        from pricebook.ts._stats import recovery_time
        ts = self._make_ts([])
        assert recovery_time(ts) == 0

    def test_information_ratio(self):
        from pricebook.ts._stats import information_ratio
        ts = self._make_ts([0.02, 0.01, 0.03, 0.02, 0.01, 0.03, 0.02, 0.01, 0.03, 0.02])
        bench = self._make_ts([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        ir = information_ratio(ts, bench)
        assert ir > 0

    def test_information_ratio_short(self):
        from pricebook.ts._stats import information_ratio
        ts = self._make_ts([0.01])
        bench = self._make_ts([0.01])
        ir = information_ratio(ts, bench)
        assert ir == 0.0

    def test_tracking_error(self):
        from pricebook.ts._stats import tracking_error
        ts = self._make_ts([0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015, 0.02, -0.008])
        bench = self._make_ts([0.01, 0.005, 0.02, 0.0, 0.01, 0.015, -0.005, 0.01, 0.015, -0.003])
        te = tracking_error(ts, bench)
        assert te > 0

    def test_treynor_ratio(self):
        from pricebook.ts._stats import treynor_ratio
        ts = self._make_ts([0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015, 0.02, -0.008])
        bench = self._make_ts([0.01, 0.005, 0.02, 0.0, 0.01, 0.015, -0.005, 0.01, 0.015, -0.003])
        tr = treynor_ratio(ts, bench)
        assert isinstance(tr, float)

    def test_treynor_short(self):
        from pricebook.ts._stats import treynor_ratio
        ts = self._make_ts([0.01])
        bench = self._make_ts([0.01])
        assert treynor_ratio(ts, bench) == 0.0

    def test_omega_ratio(self):
        from pricebook.ts._stats import omega_ratio
        ts = self._make_ts([0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015, 0.02, -0.008])
        o = omega_ratio(ts)
        assert o > 0

    def test_omega_empty(self):
        from pricebook.ts._stats import omega_ratio
        ts = self._make_ts([])
        assert omega_ratio(ts) == 0.0

    def test_gain_to_pain(self):
        from pricebook.ts._stats import gain_to_pain
        ts = self._make_ts([0.02, -0.01, 0.03, -0.005, 0.01])
        g = gain_to_pain(ts)
        assert g > 0

    def test_gain_to_pain_empty(self):
        from pricebook.ts._stats import gain_to_pain
        ts = self._make_ts([])
        assert gain_to_pain(ts) == 0.0

    def test_kelly_fraction(self):
        from pricebook.ts._stats import kelly_fraction
        k = kelly_fraction(0.6, 2.0)  # 60% win rate, 2:1 payoff
        assert k > 0

    def test_kelly_fraction_zero_ratio(self):
        from pricebook.ts._stats import kelly_fraction
        assert kelly_fraction(0.5, 0.0) == 0.0

    def test_kelly_continuous(self):
        from pricebook.ts._stats import kelly_continuous
        k = kelly_continuous(0.10, 0.04)  # 10% return, 4% variance
        assert k > 0

    def test_kelly_continuous_zero_var(self):
        from pricebook.ts._stats import kelly_continuous
        assert kelly_continuous(0.10, 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════
# ts/_rolling.py — rolling functions
# ═══════════════════════════════════════════════════════════════

class TestTSRolling:
    def _make_ts(self, n=100):
        from pricebook.ts._core import TimeSeries
        rng = np.random.default_rng(42)
        dates = np.array([date.fromordinal(date(2024, 1, 1).toordinal() + i) for i in range(n)])
        return TimeSeries(dates, rng.normal(0.001, 0.01, n))

    def test_rolling_mean(self):
        from pricebook.ts._rolling import rolling_mean
        ts = self._make_ts()
        r = rolling_mean(ts, window=20)
        assert len(r.values) > 0

    def test_rolling_skew(self):
        from pricebook.ts._rolling import rolling_skew
        ts = self._make_ts()
        r = rolling_skew(ts, window=20)
        assert len(r.values) > 0

    def test_rolling_kurtosis(self):
        from pricebook.ts._rolling import rolling_kurtosis
        ts = self._make_ts()
        r = rolling_kurtosis(ts, window=20)
        assert len(r.values) > 0

    def test_rolling_beta(self):
        from pricebook.ts._rolling import rolling_beta
        ts = self._make_ts()
        bench = self._make_ts()
        r = rolling_beta(ts, bench, window=20)
        assert len(r.values) > 0


# ═══════════════════════════════════════════════════════════════
# structured/rates_structured.py — untested products
# ═══════════════════════════════════════════════════════════════

class TestRatesStructured:
    def test_zc_swaption(self):
        from pricebook.structured.rates_structured import zc_swaption
        r = zc_swaption(forward_zc_rate=0.04, strike=0.04, vol=0.20, T_option=1.0,
                         T_swap=5.0, notional=1_000_000, rate=0.04)
        assert r.price > 0
        d = r.to_dict()
        assert "price" in d

    def test_inverse_floater(self):
        from pricebook.structured.rates_structured import inverse_floater
        r = inverse_floater(notional=1_000_000, fixed_rate=0.08, leverage=1.0,
                             rate=0.04, vol=0.01, T=5.0, floor=0.0)
        assert r.price > 0
        d = r.to_dict()
        assert "price" in d

    def test_capped_floater(self):
        from pricebook.structured.rates_structured import capped_floater
        r = capped_floater(notional=1_000_000, spread=0.01, cap_rate=0.06,
                            rate=0.04, vol=0.01, T=5.0)
        assert r.price > 0
        d = r.to_dict()
        assert "price" in d

    def test_cms_spread_range_accrual_to_dict(self):
        from pricebook.structured.rates_structured import cms_spread_range_accrual
        r = cms_spread_range_accrual(
            notional=1_000_000, coupon_rate=0.06,
            range_low=0.0, range_high=0.03,
            cms_long_rate=0.04, cms_short_rate=0.035,
            cms_long_vol=0.008, cms_short_vol=0.006,
            correlation=0.85, rate=0.04, T=3.0, n_paths=5_000,
        )
        d = r.to_dict()
        assert "price" in d

    def test_callable_step_up_to_dict(self):
        from pricebook.structured.rates_structured import callable_step_up_bond
        r = callable_step_up_bond(
            face=1_000_000,
            coupon_schedule=[0.03, 0.035, 0.04, 0.045, 0.05],
            rate=0.04, vol=0.01, T=5.0,
        )
        d = r.to_dict()
        assert "price" in d

    def test_inflation_range_accrual_to_dict(self):
        from pricebook.structured.rates_structured import inflation_range_accrual
        r = inflation_range_accrual(
            notional=1_000_000, coupon_rate=0.05,
            inflation_range_low=0.01, inflation_range_high=0.04,
            initial_inflation=0.025, inflation_vol=0.005,
            rate=0.04, T=3.0, n_paths=5_000,
        )
        d = r.to_dict()
        assert "price" in d


# ═══════════════════════════════════════════════════════════════
# structured/structured_notes.py — to_dict and participation_note
# ═══════════════════════════════════════════════════════════════

class TestStructuredNotes:
    def test_capital_protected_to_dict(self):
        from pricebook.structured.structured_notes import capital_protected_note
        r = capital_protected_note(spot=100, rate=0.04, dividend_yield=0.02, vol=0.20, T=3.0)
        d = r.to_dict()
        assert "price" in d or "value" in d or len(d) > 0

    def test_dual_digital_to_dict(self):
        from pricebook.structured.structured_notes import dual_digital
        r = dual_digital(spot1=100, spot2=50, barrier1=95, barrier2=45,
                          rate=0.04, div1=0.02, div2=0.01, vol1=0.20, vol2=0.25,
                          correlation=0.5, T=1.0, payout=1.0, n_paths=5_000)
        d = r.to_dict()
        assert len(d) > 0

    def test_bonus_certificate_to_dict(self):
        from pricebook.structured.structured_notes import bonus_certificate
        r = bonus_certificate(spot=100, rate=0.04, dividend_yield=0.02, vol=0.20, T=2.0,
                               bonus_level=110, barrier=80, n_paths=5_000)
        d = r.to_dict()
        assert len(d) > 0

    def test_participation_note(self):
        from pricebook.structured.structured_notes import participation_note
        r = participation_note(notional=1000, spot=100, rate=0.04, vol=0.20, T=3.0,
                                protection=0.95, participation=1.5)
        d = r.to_dict()
        assert len(d) > 0

    def test_outperformance_to_dict(self):
        from pricebook.structured.structured_notes import outperformance_certificate
        r = outperformance_certificate(spot=100, rate=0.04, dividend_yield=0.02,
                                        vol=0.20, T=2.0, participation=1.5)
        d = r.to_dict()
        assert len(d) > 0


# ═══════════════════════════════════════════════════════════════
# pe/ — DCF, LBO, performance, desk
# ═══════════════════════════════════════════════════════════════

class TestPE:
    def test_dcf_perpetuity(self):
        from pricebook.pe.dcf import DCFModel, WACCInputs
        wacc = WACCInputs(risk_free_rate=0.03, equity_risk_premium=0.05,
                           beta=1.0, cost_of_debt=0.05, tax_rate=0.25, debt_to_total=0.4)
        model = DCFModel(fcfs=[50, 55, 60, 65, 70], wacc_inputs=wacc, terminal_growth=0.02)
        r = model.value(method="perpetuity_growth")
        assert r.enterprise_value > 0

    def test_dcf_exit_multiple(self):
        from pricebook.pe.dcf import DCFModel, WACCInputs
        wacc = WACCInputs(risk_free_rate=0.03, equity_risk_premium=0.05,
                           beta=1.0, cost_of_debt=0.05, tax_rate=0.25, debt_to_total=0.4)
        model = DCFModel(fcfs=[50, 55, 60, 65, 70], wacc_inputs=wacc,
                          terminal_ebitda=100, terminal_multiple=8.0)
        r = model.value(method="exit_multiple")
        assert r.enterprise_value > 0

    def test_dcf_scenario(self):
        from pricebook.pe.dcf import DCFModel, WACCInputs
        wacc = WACCInputs(risk_free_rate=0.03, equity_risk_premium=0.05,
                           beta=1.0, cost_of_debt=0.05, tax_rate=0.25, debt_to_total=0.4)
        model = DCFModel(fcfs=[50, 55, 60], wacc_inputs=wacc, terminal_growth=0.02)
        scenarios = model.scenario_analysis({
            "base": {}, "bull": {"terminal_growth": 0.03}, "bear": {"terminal_growth": 0.01},
        })
        assert len(scenarios) == 3

    def test_dcf_football_field(self):
        from pricebook.pe.dcf import DCFModel, WACCInputs
        wacc = WACCInputs(risk_free_rate=0.03, equity_risk_premium=0.05,
                           beta=1.0, cost_of_debt=0.05, tax_rate=0.25, debt_to_total=0.4)
        model = DCFModel(fcfs=[50, 55, 60], wacc_inputs=wacc, terminal_growth=0.02)
        ff = model.football_field()
        assert len(ff.methods) > 0

    def test_ev_to_equity(self):
        from pricebook.pe.dcf import ev_to_equity
        bridge = ev_to_equity(1000, 200, shares=100)
        assert bridge.equity_value == 800
        assert bridge.equity_value_per_share == 8.0

    def test_lbo_run(self):
        from pricebook.pe.lbo import LBOModel
        model = LBOModel(enterprise_value=500, entry_ebitda=100)
        r = model.run(exit_multiples=[7.0, 8.0, 9.0])
        assert r.entry_multiple > 0

    def test_lbo_sensitivity(self):
        from pricebook.pe.lbo import LBOModel
        model = LBOModel(enterprise_value=500, entry_ebitda=100)
        table = model.sensitivity_table()
        assert len(table) > 0

    def test_kaplan_schoar_pme(self):
        from pricebook.pe.pe_performance import kaplan_schoar_pme
        r = kaplan_schoar_pme(
            contributions=[100, 50, 30],
            distributions=[0, 20, 80],
            index_returns=[0.10, 0.08, 0.12],
            fund_nav=150,
        )
        assert r.pme_ratio > 0

    def test_commitment_pacing(self):
        from pricebook.pe.pe_performance import commitment_pacing
        r = commitment_pacing(
            target_allocation=0.15, portfolio_value=1_000_000,
            existing_nav=100_000, existing_unfunded=50_000,
            horizon=5,
        )
        assert len(r) > 0

    def test_gp_economics(self):
        from pricebook.pe.pe_performance import gp_economics
        r = gp_economics(fund_size=500_000_000, gross_return=0.15)
        assert r.carry_total > 0 or r.mgmt_fee_total > 0

    def test_pe_desk(self):
        from pricebook.pe.pe_desk import PEBook, PEBookEntry
        entry = PEBookEntry(trade_id="PE001", instrument=None, product_type="fund",
                             fund_manager="Alpha", vintage_year=2020, sector="Tech")
        d = entry.to_dict()
        assert "trade_id" in d

    def test_pe_book(self):
        from pricebook.pe.pe_desk import PEBook, PEBookEntry
        book = PEBook()
        book.add(PEBookEntry("PE001", None, "fund", "Alpha", 2020, "Tech", "US"))
        book.add(PEBookEntry("PE002", None, "fund", "Beta", 2021, "Health", "EU"))
        assert len(book._entries) == 2
