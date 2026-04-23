"""Tests for API Extension (APIX1-APIX12)."""

from datetime import date

import numpy as np
import pytest

import pricebook.api as pb
from pricebook.inflation import CPICurve
from pricebook.repo_term import RepoCurve, RepoRate

REF = date(2026, 4, 21)


# ---- APIX1: Bond Futures ----

class TestAPIX1:
    def test_implied_repo(self):
        assert pb.implied_repo(98, 100, 0.95, 1.5, 2.0, 90) > 0

    def test_hedge_ratio(self):
        assert pb.futures_hedge_ratio(0.08, 0.10, 0.95) == pytest.approx(0.08 / 0.10 * 0.95)

    def test_bond_forward_price(self):
        assert pb.bond_forward_price(101.5, 0.04, 90) > 101.5

    def test_strips(self):
        result = pb.strips("10Y", 0.04, start=REF)
        assert result["n_c_strips"] > 0


# ---- APIX2: Exotic Options ----

class TestAPIX2:
    def test_asian(self):
        assert pb.asian_option(100, 100, 0.20, 1.0) > 0

    def test_digital_call(self):
        assert pb.digital_option(100, 95, 0.20, 1.0, payout=100) > 0

    def test_digital_put(self):
        assert pb.digital_option(100, 105, 0.20, 1.0, payout=100, option_type="put") > 0


# ---- APIX3: Inflation ----

class TestAPIX3:
    def _cpi(self):
        return CPICurve.from_breakevens(REF, 300.0, [date(2031, 4, 21)], [0.025])

    def test_zc_swap(self):
        curve = pb.flat_curve(0.04, REF)
        pv = pb.zc_inflation_swap(0.025, "5Y", curve, self._cpi())
        assert isinstance(pv, float)

    def test_par_rate(self):
        curve = pb.flat_curve(0.04, REF)
        pr = pb.inflation_par_rate("5Y", curve, self._cpi())
        assert 0.01 < pr < 0.05

    def test_linker(self):
        curve = pb.flat_curve(0.04, REF)
        px = pb.inflation_linker("10Y", 0.01, curve, self._cpi(), start=REF)
        assert px > 0


# ---- APIX4: Vol Calibration ----

class TestAPIX4:
    def test_implied_vol(self):
        from pricebook.black76 import black76_price, OptionType
        price = black76_price(100, 100, 0.20, 1.0, 0.96, OptionType.CALL)
        iv = pb.implied_vol(price, 100, 100, 1.0, 0.96)
        assert iv == pytest.approx(0.20, rel=0.01)


# ---- APIX5: Risk ----

class TestAPIX5:
    def test_var(self):
        ret = np.random.default_rng(42).standard_normal(500) * 0.01
        assert pb.var(ret, 0.95) > 0

    def test_stress(self):
        results = pb.stress(1e6, {"rates": -50_000, "equity": 200_000})
        assert len(results) > 0

    def test_drawdown(self):
        dd = pb.drawdown([100, 110, 95, 100])
        assert dd["max"] > 0

    def test_key_rate_dv01(self):
        curve = pb.flat_curve(0.04, REF)
        ladder = pb.key_rate_dv01(lambda c: pb.irs("5Y", 0.04, c, start=REF), curve)
        assert len(ladder) > 0


# ---- APIX6: P&L ----

class TestAPIX6:
    def test_pnl(self):
        result = pb.pnl({"a": 100}, {"a": 90})
        assert result["total"] == pytest.approx(10)

    def test_cashflow_table(self):
        curve = pb.flat_curve(0.04, REF)
        cfs = pb.cashflow_table("5Y", 0.04, curve)
        assert len(cfs) > 0
        assert "leg" in cfs[0]


# ---- APIX7: Stochastic Models ----

class TestAPIX7:
    def test_heston(self):
        assert pb.heston_price(100, 100, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7) > 0

    def test_sabr_vol(self):
        v = pb.sabr_vol(0.04, 0.04, 1.0, 0.03)
        assert v > 0


# ---- APIX8: Repo ----

class TestAPIX8:
    def test_repo_rate(self):
        rc = RepoCurve(REF, [RepoRate(30, 0.04), RepoRate(90, 0.045)])
        assert pb.repo_rate(60, rc) > 0

    def test_forward_repo(self):
        rc = RepoCurve(REF, [RepoRate(30, 0.04), RepoRate(90, 0.045)])
        assert pb.forward_repo(30, 90, rc) > 0


# ---- APIX9: Trading Strategies ----

class TestAPIX9:
    def test_curve_spread(self):
        curve = pb.flat_curve(0.04, REF)
        spread = pb.curve_spread("2Y", "10Y", curve)
        assert isinstance(spread, float)

    def test_butterfly(self):
        curve = pb.flat_curve(0.04, REF)
        fly = pb.butterfly("2Y", "5Y", "10Y", curve)
        assert isinstance(fly, float)

    def test_rv_zscore(self):
        z = pb.rv_zscore(1.5, [1.0, 1.1, 1.2, 1.3, 1.4])
        assert z > 0


# ---- APIX10: Regulatory ----

class TestAPIX10:
    def test_frtb(self):
        cap = pb.frtb_capital([{"sensitivity": 1e6, "rw": 0.016}])
        assert cap > 0

    def test_mva(self):
        assert pb.mva([500_000, 400_000, 300_000], 0.02, 0.25) > 0


# ---- APIX11: Books ----

class TestAPIX11:
    def test_create_book(self):
        book = pb.create_book("rates", {"swap": 15000, "bond": -3000})
        assert book["total_pv"] == 12000
        assert book["n_trades"] == 2

    def test_book_dv01(self):
        book = pb.create_book("rates", {"swap": 15000})
        assert pb.book_dv01(book, {"swap": 450}) == 450

    def test_book_pnl(self):
        b1 = pb.create_book("r", {"a": 100})
        b2 = pb.create_book("r", {"a": 110})
        assert pb.book_pnl(b2, b1) == 10


# ---- APIX12: Backtesting & Factors ----

class TestAPIX12:
    def test_backtest(self):
        ret = np.random.default_rng(42).standard_normal(200) * 0.01
        sig = np.sign(ret)
        result = pb.backtest(ret, sig)
        assert "sharpe" in result
        assert result["sharpe"] > 0

    def test_risk_parity(self):
        cov = [[0.04, 0.01], [0.01, 0.02]]
        w = pb.risk_parity_weights(cov, ["stocks", "bonds"])
        assert abs(sum(w.values()) - 1.0) < 1e-6
