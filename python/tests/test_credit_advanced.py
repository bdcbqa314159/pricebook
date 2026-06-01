"""Tests for credit spread vol, quanto CDS, credit VaR, index swaption, recovery-locked CDS."""

import pytest
import math
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


def _make_curves(hazard=0.02, rate=0.04):
    from pricebook.core.discount_curve import DiscountCurve
    from pricebook.core.survival_curve import SurvivalCurve
    from pricebook.core.interpolation import InterpolationMethod
    dates = [REF + relativedelta(years=y) for y in range(1, 15)]
    dfs = [math.exp(-rate * y) for y in range(1, 15)]
    survs = [math.exp(-hazard * y) for y in range(1, 15)]
    dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
    sc = SurvivalCurve(REF, dates, survs)
    return dc, sc


# ═══════════════════════════════════════════════════════════════
# Credit Spread Vol Surface (C3)
# ═══════════════════════════════════════════════════════════════

class TestCreditSpreadVol:
    def test_construction(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 2, 5, 10], [3, 5, 10],
                                          [[0.40, 0.38, 0.35],
                                           [0.38, 0.36, 0.33],
                                           [0.35, 0.33, 0.30],
                                           [0.32, 0.30, 0.28]])
        assert surface.vol(1.0, 5.0) == pytest.approx(0.38)

    def test_interpolation(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5, 10], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30], [0.30, 0.25]])
        vol = surface.vol(3.0, 7.0)
        assert 0.25 < vol < 0.40

    def test_bumped(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        bumped = surface.bumped(0.05)
        assert bumped.vol(1.0, 5.0) == pytest.approx(0.45)

    def test_synthetic(self):
        from pricebook.credit.credit_spread_vol import synthetic_credit_vol_surface
        surface = synthetic_credit_vol_surface(0.02, REF)
        vol = surface.vol(5.0, 5.0)
        assert 0.20 < vol < 0.60

    def test_to_dict(self):
        from pricebook.credit.credit_spread_vol import CreditSpreadVolSurface
        surface = CreditSpreadVolSurface(REF, [1, 5], [5, 10],
                                          [[0.40, 0.35], [0.35, 0.30]])
        d = surface.to_dict()
        assert "expiries" in d or "expiry_years" in d or "reference_date" in d


# ═══════════════════════════════════════════════════════════════
# Quanto CDS (C4)
# ═══════════════════════════════════════════════════════════════

class TestQuantoCDS:
    def test_quanto_adjustment_positive_corr(self):
        """Positive FX-credit correlation → quanto spread > foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01  # 100bp
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.30, 5.0)
        assert adjusted > foreign

    def test_quanto_adjustment_negative_corr(self):
        """Negative correlation → quanto spread < foreign."""
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, -0.30, 5.0)
        assert adjusted < foreign

    def test_zero_corr_no_adjustment(self):
        from pricebook.credit.quanto_cds import quanto_cds_spread
        foreign = 0.01
        adjusted = quanto_cds_spread(foreign, 0.10, 0.40, 0.0, 5.0)
        assert adjusted == pytest.approx(foreign)

    def test_price_quanto_cds(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_foreign, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_foreign, sc,
                              fx_spot=1.10, fx_vol=0.08, credit_vol=0.40,
                              correlation=0.25)
        assert r.domestic_spread > 0
        assert r.quanto_adjustment_bp != 0

    def test_to_dict(self):
        from pricebook.credit.quanto_cds import price_quanto_cds
        dc, sc = _make_curves()
        dc_f, _ = _make_curves(rate=0.02)
        r = price_quanto_cds(REF, 5.0, 0.01, dc, dc_f, sc, 1.10, 0.08, 0.40, 0.25)
        d = r.to_dict()
        assert "quanto_adjustment_bp" in d


# ═══════════════════════════════════════════════════════════════
# Credit Portfolio VaR (C9)
# ═══════════════════════════════════════════════════════════════

class TestCreditVaR:
    def test_historical_var(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        spread_changes = {
            "A": rng.normal(0, 0.0005, 250).tolist(),
            "B": rng.normal(0, 0.0003, 250).tolist(),
        }
        result = historical_credit_var(positions, spread_changes, confidence=0.99)
        assert abs(result.var_amount) > 0  # VaR is non-zero
        assert abs(result.es_amount) >= abs(result.var_amount) - 1  # |ES| ≥ |VaR|

    def test_parametric_var(self):
        from pricebook.credit.credit_var import parametric_credit_var
        positions = [
            {"name": "A", "cs01": -50_000},
            {"name": "B", "cs01": -30_000},
        ]
        vols = [0.0005, 0.0003]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = parametric_credit_var(positions, vols, corr, confidence=0.99)
        assert abs(result.var_amount) > 0

    def test_copula_var(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 5_000_000},
            {"name": "C", "notional": 8_000_000},
        ]
        pds = [0.02, 0.03, 0.01]
        lgds = [0.6, 0.6, 0.4]
        result = copula_credit_var(positions, pds, lgds, correlation=0.3, confidence=0.99)
        assert abs(result.var_amount) > 0
        assert result.worst_name is not None

    def test_var_increases_with_correlation(self):
        from pricebook.credit.credit_var import copula_credit_var
        positions = [
            {"name": "A", "notional": 10_000_000},
            {"name": "B", "notional": 10_000_000},
        ]
        pds = [0.02, 0.02]
        lgds = [0.6, 0.6]
        low = copula_credit_var(positions, pds, lgds, correlation=0.1, confidence=0.99, seed=42)
        high = copula_credit_var(positions, pds, lgds, correlation=0.8, confidence=0.99, seed=42)
        assert abs(high.var_amount) >= abs(low.var_amount) * 0.8

    def test_to_dict(self):
        from pricebook.credit.credit_var import historical_credit_var
        rng = np.random.default_rng(42)
        positions = [{"name": "A", "cs01": -50_000}]
        spread_changes = {"A": rng.normal(0, 0.0005, 100).tolist()}
        result = historical_credit_var(positions, spread_changes)
        d = result.to_dict()
        assert "var_amount" in d
        assert "method" in d


# ═══════════════════════════════════════════════════════════════
# Index CDS Swaption (C2)
# ═══════════════════════════════════════════════════════════════

class TestIndexCDSSwaption:
    def _make_index_curves(self, n=5):
        """Create n slightly different survival curves for index constituents."""
        dc, _ = _make_curves()
        scs = []
        for i in range(n):
            hazard = 0.015 + i * 0.005
            _, sc = _make_curves(hazard=hazard)
            scs.append(sc)
        return dc, scs

    def test_payer_positive(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        r = index_cds_swaption_black(0.005, 0.004, 0.40, 1.0, 4.0, 0.95)
        assert r.premium > 0

    def test_receiver_positive(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        r = index_cds_swaption_black(0.004, 0.005, 0.40, 1.0, 4.0, 0.95,
                                      option_type="receiver")
        assert r.premium > 0

    def test_put_call_parity(self):
        """Payer - Receiver = forward value."""
        from pricebook.credit.index_cds_swaption import index_cds_swaption_black
        F, K = 0.005, 0.004
        ann, surv, N = 4.0, 0.95, 10_000_000
        p = index_cds_swaption_black(F, K, 0.40, 1.0, ann, surv, N, "payer")
        r = index_cds_swaption_black(F, K, 0.40, 1.0, ann, surv, N, "receiver")
        forward_value = (F - K) * N * surv * ann
        assert p.premium - r.premium == pytest.approx(forward_value, rel=1e-6)

    def test_bachelier(self):
        from pricebook.credit.index_cds_swaption import index_cds_swaption_bachelier
        r = index_cds_swaption_bachelier(0.005, 0.004, 0.002, 1.0, 4.0, 0.95)
        assert r.premium > 0
        assert r.model == "bachelier"

    def test_forward_index_spread(self):
        from pricebook.credit.index_cds_swaption import index_forward_spread
        dc, scs = self._make_index_curves(5)
        expiry = REF + relativedelta(years=1)
        maturity = REF + relativedelta(years=6)
        fwd = index_forward_spread(dc, scs, expiry, maturity)
        assert fwd.forward_spread > 0
        assert len(fwd.constituent_forwards) == 5
        assert fwd.index_annuity > 0

    def test_greeks(self):
        from pricebook.credit.index_cds_swaption import index_swaption_greeks
        r = index_swaption_greeks(0.005, 0.004, 0.40, 1.0, 4.0, 0.95)
        assert r.delta > 0  # payer delta positive
        assert r.vega > 0   # long vol
        assert r.theta < 0  # time decay

    def test_full_pricing(self):
        from pricebook.credit.index_cds_swaption import price_index_cds_swaption
        dc, scs = self._make_index_curves(5)
        expiry = REF + relativedelta(years=1)
        maturity = REF + relativedelta(years=6)
        r = price_index_cds_swaption(dc, scs, expiry, maturity, 0.005, 0.40)
        assert r.premium > 0
        assert r.delta != 0


# ═══════════════════════════════════════════════════════════════
# Recovery-Locked CDS + LCDS (C5)
# ═══════════════════════════════════════════════════════════════

class TestRecoveryLockedCDS:
    def test_lock_premium_higher_recovery(self):
        """Higher locked recovery → negative premium (less protection)."""
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.60, 0.40)
        assert prem < 0  # locked recovery 60% > market 40%

    def test_lock_premium_lower_recovery(self):
        """Lower locked recovery → positive premium (more protection)."""
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.20, 0.40)
        assert prem > 0

    def test_lock_premium_equal(self):
        from pricebook.credit.recovery_locked_cds import recovery_lock_premium
        prem = recovery_lock_premium(0.01, 0.40, 0.40)
        assert prem == pytest.approx(0.0)

    def test_price_recovery_locked(self):
        from pricebook.credit.recovery_locked_cds import price_recovery_locked_cds
        dc, sc = _make_curves()
        r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.30, dc, sc)
        assert r.par_spread > 0
        assert r.rpv01 > 0
        assert r.locked_recovery == 0.30

    def test_higher_recovery_lower_spread(self):
        """Higher locked recovery → lower par spread."""
        from pricebook.credit.recovery_locked_cds import price_recovery_locked_cds
        dc, sc = _make_curves()
        low_r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.30, dc, sc)
        high_r = price_recovery_locked_cds(REF, 5.0, 0.01, 0.60, dc, sc)
        assert low_r.par_spread > high_r.par_spread


class TestLCDS:
    def test_lcds_higher_recovery(self):
        """LCDS has higher recovery than standard CDS → lower spread."""
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc, recovery=0.70)
        assert r.par_spread > 0
        assert r.par_spread < 0.02  # lower than bond CDS

    def test_prepayment_shortens_maturity(self):
        """Prepayment reduces effective maturity."""
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        no_prepay = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.0)
        with_prepay = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.20)
        assert with_prepay.effective_maturity < no_prepay.effective_maturity

    def test_cancellation_value_positive(self):
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc, prepayment_rate=0.15)
        assert r.cancellation_value > 0  # seller benefits from cancellation

    def test_to_dict(self):
        from pricebook.credit.recovery_locked_cds import price_lcds
        dc, sc = _make_curves()
        r = price_lcds(REF, 5.0, 0.005, dc, sc)
        d = r.to_dict()
        assert "prepayment_rate" in d
        assert "cancellation_value" in d


# ═══════════════════════════════════════════════════════════════
# Index Roll Mechanics (C6)
# ═══════════════════════════════════════════════════════════════

class TestIndexRoll:
    def _make_constituents(self, n=5, base_spread=0.005):
        from pricebook.credit.index_roll import Constituent
        return [Constituent(f"Name_{i}", base_spread + i * 0.001) for i in range(n)]

    def test_series_transition(self):
        from pricebook.credit.index_roll import series_transition, Constituent
        old = self._make_constituents(5)
        additions = [Constituent("NewCo", 0.004)]
        removals = ["Name_4"]
        new = series_transition(old, additions, removals)
        assert len(new) == 5
        names = {c.name for c in new}
        assert "NewCo" in names
        assert "Name_4" not in names

    def test_roll_pnl(self):
        from pricebook.credit.index_roll import index_roll_pnl
        old = self._make_constituents(5, 0.006)
        new = self._make_constituents(5, 0.005)  # tighter
        r = index_roll_pnl(old, new, 41, 42, rpv01=4.5)
        assert r.roll_pnl > 0  # buyer benefits from tighter new series
        assert r.spread_change_bp < 0

    def test_otr_basis(self):
        from pricebook.credit.index_roll import on_the_run_basis
        otr = self._make_constituents(5, 0.004)  # tighter (liquid)
        off = self._make_constituents(5, 0.005)   # wider (off-run)
        r = on_the_run_basis(otr, off, 42, 41)
        assert r.basis_bp < 0  # OTR trades tighter

    def test_series_transition_pnl(self):
        from pricebook.credit.index_roll import series_transition_pnl, Constituent
        old = self._make_constituents(5, 0.006)
        add = [Constituent("NewCo", 0.003)]
        rem = ["Name_4"]  # removing widest name
        r = series_transition_pnl(old, add, rem, 41, rpv01=4.5)
        assert r.new_series == 42
        assert "NewCo" in r.names_added

    def test_to_dict(self):
        from pricebook.credit.index_roll import index_roll_pnl
        old = self._make_constituents(3)
        new = self._make_constituents(3)
        r = index_roll_pnl(old, new, 1, 2, rpv01=4.0)
        d = r.to_dict()
        assert "roll_pnl" in d


# ═══════════════════════════════════════════════════════════════
# Index Replication (C7)
# ═══════════════════════════════════════════════════════════════

class TestIndexReplication:
    def test_full_replication(self):
        from pricebook.credit.index_replication import replicate_index
        rng = np.random.default_rng(42)
        N, T = 10, 250
        spreads = rng.normal(0, 0.0005, (T, N))
        weights_true = np.ones(N) / N
        index = spreads @ weights_true + rng.normal(0, 0.00001, T)
        r = replicate_index(index, spreads)
        assert r.r_squared > 0.9
        assert r.tracking_error < 0.01

    def test_sparse_replication(self):
        from pricebook.credit.index_replication import replicate_index
        rng = np.random.default_rng(42)
        N, T = 20, 250
        spreads = rng.normal(0, 0.0005, (T, N))
        weights_true = np.zeros(N)
        weights_true[:5] = 0.2
        index = spreads @ weights_true
        r = replicate_index(index, spreads, n_select=5)
        assert r.n_active <= 5
        assert r.r_squared > 0.8

    def test_te_decreases_with_more_names(self):
        from pricebook.credit.index_replication import replicate_index
        rng = np.random.default_rng(42)
        N, T = 20, 250
        spreads = rng.normal(0, 0.0005, (T, N))
        index = spreads.mean(axis=1)
        te_5 = replicate_index(index, spreads, n_select=5).tracking_error
        te_15 = replicate_index(index, spreads, n_select=15).tracking_error
        assert te_15 <= te_5 * 1.1  # more names → lower TE

    def test_l1_sparsity(self):
        from pricebook.credit.index_replication import replicate_index
        rng = np.random.default_rng(42)
        N, T = 10, 250
        spreads = rng.normal(0, 0.0005, (T, N))
        index = spreads.mean(axis=1)
        dense = replicate_index(index, spreads, l1_penalty=0.0)
        sparse = replicate_index(index, spreads, l1_penalty=0.01)
        assert sparse.n_active <= dense.n_active

    def test_tracking_error_fn(self):
        from pricebook.credit.index_replication import tracking_error
        rng = np.random.default_rng(42)
        idx = rng.normal(0, 0.001, 250)
        rep = idx + rng.normal(0, 0.0001, 250)
        te = tracking_error(idx, rep)
        assert te > 0
        assert te < 0.01  # small residual


# ═══════════════════════════════════════════════════════════════
# Credit Event Auction (C8)
# ═══════════════════════════════════════════════════════════════

class TestCreditEvent:
    def test_simulate_auction(self):
        from pricebook.credit.credit_event import simulate_auction
        r = simulate_auction(expected_recovery=0.40, seed=42)
        assert 0 <= r.final_price <= 100
        assert r.n_dealers == 14
        assert 0 <= r.recovery_rate <= 1

    def test_auction_near_expected(self):
        from pricebook.credit.credit_event import simulate_auction
        results = [simulate_auction(0.30, seed=i) for i in range(100)]
        avg = np.mean([r.final_price for r in results])
        assert 20 < avg < 40  # near expected 30

    def test_settlement_buyer(self):
        from pricebook.credit.credit_event import settlement_amount
        s = settlement_amount(10_000_000, 35.0, is_protection_buyer=True)
        assert s == pytest.approx(10_000_000 * 0.65)

    def test_settlement_seller(self):
        from pricebook.credit.credit_event import settlement_amount
        s = settlement_amount(10_000_000, 35.0, is_protection_buyer=False)
        assert s < 0

    def test_credit_event_timeline(self):
        from pricebook.credit.credit_event import CreditEvent, CreditEventTimeline, EventType
        event = CreditEvent("WidgetCo", EventType.BANKRUPTCY, date(2024, 6, 1))
        tl = CreditEventTimeline(event)
        dates = tl.timeline()
        assert dates["event_date"] == "2024-06-01"
        assert "auction_date" in dates

    def test_process_credit_event(self):
        from pricebook.credit.credit_event import (
            process_credit_event, CreditEvent, EventType,
        )
        event = CreditEvent("WidgetCo", EventType.FAILURE_TO_PAY, date(2024, 6, 1))
        result = process_credit_event(event, 10_000_000, seed=42)
        assert abs(result["settlement_amount"]) > 0
        assert "auction" in result

    def test_event_types(self):
        from pricebook.credit.credit_event import EventType
        assert len(EventType) >= 6

    def test_to_dict(self):
        from pricebook.credit.credit_event import simulate_auction
        r = simulate_auction(seed=42)
        d = r.to_dict()
        assert "final_price" in d
        assert "recovery_rate" in d


# ═══════════════════════════════════════════════════════════════
# Weighted Portfolio CDS (C10)
# ═══════════════════════════════════════════════════════════════

class TestPortfolioCDS:
    def test_basic_pricing(self):
        from pricebook.credit.portfolio_cds import portfolio_cds_pv, PortfolioPosition
        dc, sc = _make_curves()
        positions = [
            PortfolioPosition("A", 5_000_000, 0.01),
            PortfolioPosition("B", 3_000_000, 0.02),
        ]
        _, sc2 = _make_curves(hazard=0.03)
        r = portfolio_cds_pv(REF, 5.0, positions, dc, [sc, sc2])
        assert r.n_positions == 2
        assert r.total_notional == 8_000_000

    def test_long_short(self):
        from pricebook.credit.portfolio_cds import portfolio_cds_pv, PortfolioPosition
        dc, sc = _make_curves()
        positions = [
            PortfolioPosition("A", 5_000_000, 0.01),   # long protection
            PortfolioPosition("B", -3_000_000, 0.02),   # short protection
        ]
        _, sc2 = _make_curves(hazard=0.03)
        r = portfolio_cds_pv(REF, 5.0, positions, dc, [sc, sc2])
        assert r.net_notional == 2_000_000

    def test_par_spread_positive(self):
        from pricebook.credit.portfolio_cds import portfolio_cds_pv, PortfolioPosition
        dc, sc = _make_curves()
        positions = [PortfolioPosition("A", 10_000_000, 0.01)]
        r = portfolio_cds_pv(REF, 5.0, positions, dc, [sc])
        assert r.par_spread > 0

    def test_constituent_cs01(self):
        from pricebook.credit.portfolio_cds import constituent_cs01, PortfolioPosition
        dc, sc = _make_curves()
        _, sc2 = _make_curves(hazard=0.03)
        positions = [
            PortfolioPosition("A", 5_000_000, 0.01),
            PortfolioPosition("B", 3_000_000, 0.02),
        ]
        result = constituent_cs01(REF, 5.0, positions, dc, [sc, sc2])
        assert len(result) == 2
        assert all(r["cs01"] > 0 for r in result)
        total_pct = sum(r["pct_contribution"] for r in result)
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_to_dict(self):
        from pricebook.credit.portfolio_cds import portfolio_cds_pv, PortfolioPosition
        dc, sc = _make_curves()
        positions = [PortfolioPosition("A", 10_000_000, 0.01)]
        r = portfolio_cds_pv(REF, 5.0, positions, dc, [sc])
        d = r.to_dict()
        assert "gross_cs01" in d


# ═══════════════════════════════════════════════════════════════
# Succession Events (C11)
# ═══════════════════════════════════════════════════════════════

class TestSuccession:
    def test_merger(self):
        from pricebook.credit.succession import (
            apply_succession, SuccessionEvent, SuccessionType,
        )
        event = SuccessionEvent(
            "OldCo", SuccessionType.MERGER, date(2024, 6, 1),
            successors=["NewCo"], weights=[1.0],
            original_notional=10_000_000,
        )
        r = apply_succession(event, 0.01)
        assert len(r.successor_cds) == 1
        assert r.successor_cds[0].notional == pytest.approx(10_000_000)
        assert r.notional_conserved

    def test_spin_off(self):
        from pricebook.credit.succession import (
            apply_succession, SuccessionEvent, SuccessionType,
        )
        event = SuccessionEvent(
            "ParentCo", SuccessionType.SPIN_OFF, date(2024, 6, 1),
            successors=["ParentCo", "SpinCo"], weights=[0.7, 0.3],
            original_notional=10_000_000,
        )
        r = apply_succession(event, 0.01)
        assert len(r.successor_cds) == 2
        assert r.successor_cds[0].notional == pytest.approx(7_000_000)
        assert r.successor_cds[1].notional == pytest.approx(3_000_000)
        assert r.notional_conserved

    def test_three_way_split(self):
        from pricebook.credit.succession import (
            apply_succession, SuccessionEvent, SuccessionType,
        )
        event = SuccessionEvent(
            "BigCo", SuccessionType.SPLIT, date(2024, 6, 1),
            successors=["A", "B", "C"], weights=[0.5, 0.3, 0.2],
        )
        r = apply_succession(event, 0.01, original_notional=10_000_000)
        total = sum(s.notional for s in r.successor_cds)
        assert total == pytest.approx(10_000_000)

    def test_spread_adjustments(self):
        from pricebook.credit.succession import (
            apply_succession, SuccessionEvent, SuccessionType,
        )
        event = SuccessionEvent(
            "Co", SuccessionType.SPIN_OFF, date(2024, 6, 1),
            successors=["Parent", "Child"], weights=[0.6, 0.4],
        )
        r = apply_succession(event, 0.01, 10_000_000,
                              spread_adjustments=[-0.002, 0.005])
        assert r.successor_cds[0].spread == pytest.approx(0.008)
        assert r.successor_cds[1].spread == pytest.approx(0.015)

    def test_to_dict(self):
        from pricebook.credit.succession import (
            apply_succession, SuccessionEvent, SuccessionType,
        )
        event = SuccessionEvent(
            "Co", SuccessionType.MERGER, date(2024, 6, 1),
            successors=["NewCo"], weights=[1.0],
        )
        r = apply_succession(event, 0.01, 10_000_000)
        d = r.to_dict()
        assert "notional_conserved" in d
        assert d["notional_conserved"] is True


# ═══════════════════════════════════════════════════════════════
# Distressed CDS Workflow (C12)
# ═══════════════════════════════════════════════════════════════

class TestDistressedCDS:
    def test_upfront_positive_for_wide_spread(self):
        """Wide spread → positive upfront (buyer pays)."""
        from pricebook.credit.distressed import distressed_cds_upfront
        r = distressed_cds_upfront(0.10, running_coupon=0.05)
        assert r.upfront_pct > 0

    def test_upfront_negative_for_tight_spread(self):
        """Tight spread < running coupon → negative upfront (buyer receives)."""
        from pricebook.credit.distressed import distressed_cds_upfront
        r = distressed_cds_upfront(0.03, running_coupon=0.05)
        assert r.upfront_pct < 0

    def test_implied_cpd(self):
        from pricebook.credit.distressed import distressed_cds_upfront
        r = distressed_cds_upfront(0.10)
        assert 0 < r.implied_cpd < 1
        # Higher spread → higher CPD
        r2 = distressed_cds_upfront(0.20)
        assert r2.implied_cpd > r.implied_cpd

    def test_implied_cpd_from_upfront(self):
        from pricebook.credit.distressed import (
            distressed_cds_upfront, implied_cpd_from_upfront,
        )
        r = distressed_cds_upfront(0.10)
        cpd = implied_cpd_from_upfront(r.upfront_pct)
        assert cpd == pytest.approx(r.implied_cpd, abs=0.02)

    def test_distressed_basis(self):
        from pricebook.credit.distressed import distressed_basis
        # CDS upfront 20% → implied bond price ~80
        # Actual bond at 65 → positive basis (CDS expensive vs bond)
        basis = distressed_basis(0.20, 65.0)
        assert basis > 0

    def test_to_dict(self):
        from pricebook.credit.distressed import distressed_cds_upfront
        r = distressed_cds_upfront(0.10)
        d = r.to_dict()
        assert "upfront_pct" in d
        assert "implied_cpd" in d
