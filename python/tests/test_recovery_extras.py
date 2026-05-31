"""Tests for recovery utilities, seniority waterfall, and bid-ask surface."""

import pytest

from pricebook.credit.recovery_pricing import (
    RecoverySpec, build_recovery_specs, validate_recovery_specs,
    recovery_spec_summary, SeniorityWaterfall, RecoveryBidAsk,
    implied_recovery, recovery_bid_ask_surface,
)


class TestBuildRecoverySpecs:
    def test_from_seniorities(self):
        specs = build_recovery_specs(["1L", "senior", "sub"])
        assert len(specs) == 3
        assert specs[0].mean == pytest.approx(0.77)
        assert specs[2].mean == pytest.approx(0.28)

    def test_correlation_applied(self):
        specs = build_recovery_specs(["senior"], correlation=-0.5)
        assert specs[0].correlation_to_default == -0.5


class TestValidateRecoverySpecs:
    def test_valid(self):
        specs = [RecoverySpec(0.4, 0.1) for _ in range(5)]
        validate_recovery_specs(specs, 5)  # no error

    def test_mismatch_raises(self):
        specs = [RecoverySpec(0.4, 0.1) for _ in range(3)]
        with pytest.raises(ValueError, match="Expected 5"):
            validate_recovery_specs(specs, 5)


class TestRecoverySpecSummary:
    def test_summary(self):
        specs = build_recovery_specs(["1L", "1L", "sub", "sub"])
        s = recovery_spec_summary(specs)
        assert s["n_names"] == 4
        assert 0.28 < s["avg_recovery"] < 0.77
        assert s["min_recovery"] == pytest.approx(0.28)
        assert s["max_recovery"] == pytest.approx(0.77)

    def test_empty(self):
        s = recovery_spec_summary([])
        assert s["n_names"] == 0


class TestSeniorityWaterfall:
    def test_full_recovery(self):
        wf = SeniorityWaterfall([("senior", 60), ("sub", 30), ("equity", 10)])
        dist = wf.distribute(100)
        assert dist["senior"] == 60
        assert dist["sub"] == 30
        assert dist["equity"] == 10

    def test_partial_recovery(self):
        wf = SeniorityWaterfall([("senior", 60), ("sub", 30), ("equity", 10)])
        dist = wf.distribute(70)
        assert dist["senior"] == 60  # full
        assert dist["sub"] == 10    # partial
        assert dist["equity"] == 0   # nothing left

    def test_zero_recovery(self):
        wf = SeniorityWaterfall([("senior", 60), ("sub", 30)])
        dist = wf.distribute(0)
        assert dist["senior"] == 0
        assert dist["sub"] == 0

    def test_recovery_rates(self):
        wf = SeniorityWaterfall([("senior", 60), ("sub", 40)])
        rates = wf.recovery_rates(0.60)  # 60% total recovery
        assert rates["senior"] == 1.0  # fully covered
        assert rates["sub"] == 0.0     # nothing left

    def test_to_recovery_specs(self):
        wf = SeniorityWaterfall([("senior", 60), ("sub", 40)])
        specs = wf.to_recovery_specs(total_recovery_mean=0.50)
        assert len(specs) == 2
        assert all(0 < s.mean < 1 for s in specs)

    def test_to_dict(self):
        wf = SeniorityWaterfall([("senior", 60)])
        d = wf.to_dict()
        assert "tranches" in d


class TestImpliedRecovery:
    def test_standard(self):
        """R = 1 - spread/hazard. With spread=0.012, h=0.02: R=0.4."""
        r = implied_recovery(0.012, 0.02)
        assert r == pytest.approx(0.4)

    def test_zero_hazard(self):
        r = implied_recovery(0.01, 0.0)
        assert r == 0.4  # default

    def test_high_spread_low_recovery(self):
        r = implied_recovery(0.05, 0.03)
        assert r < 0.4  # spread > hazard × 0.6


class TestRecoveryBidAskSurface:
    def test_surface(self):
        spreads = {1.0: 0.005, 3.0: 0.008, 5.0: 0.012}
        hazards = {1.0: 0.01, 3.0: 0.015, 5.0: 0.02}
        surface = recovery_bid_ask_surface(spreads, hazards)
        assert len(surface) == 3
        for rba in surface:
            assert rba.bid <= rba.mid <= rba.ask

    def test_to_dict(self):
        rba = RecoveryBidAsk(5.0, 0.35, 0.45, 0.40)
        d = rba.to_dict()
        assert d["tenor"] == 5.0
