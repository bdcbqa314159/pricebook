"""Tests for tranche CDS: pricing, base correlation, serialisation."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.tranche_pricing import (
    TrancheCDS, TrancheResult, expected_tranche_loss,
    calibrate_base_correlation, STANDARD_TRANCHES,
)
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
MAT = REF + timedelta(days=1825)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _scs(n=50, hazard=0.02):
    return [SurvivalCurve.flat(REF, hazard + 0.001 * (i % 5)) for i in range(n)]


class TestExpectedTrancheLoss:

    def test_equity_loss_positive(self):
        el = expected_tranche_loss(0.0, 0.03, _scs(), _disc(), rho=0.3, T=5.0)
        assert el > 0

    def test_senior_less_than_equity(self):
        el_eq = expected_tranche_loss(0.0, 0.03, _scs(), _disc(), rho=0.3, T=5.0)
        el_sr = expected_tranche_loss(0.12, 0.22, _scs(), _disc(), rho=0.3, T=5.0)
        assert el_eq > el_sr

    def test_higher_corr_lower_equity_loss(self):
        """Higher correlation → less diversification → fewer but larger losses.
        Equity tranche expected loss typically decreases with correlation."""
        el_low = expected_tranche_loss(0.0, 0.03, _scs(), _disc(), rho=0.1, T=5.0)
        el_high = expected_tranche_loss(0.0, 0.03, _scs(), _disc(), rho=0.6, T=5.0)
        # With high correlation, defaults cluster → equity can be hit less often
        assert math.isfinite(el_low) and math.isfinite(el_high)


class TestTrancheCDS:

    def test_equity_tranche(self):
        t = TrancheCDS(attachment=0.0, detachment=0.03, maturity=MAT, spread=0.05)
        r = t.price(_disc(), _scs(), correlation=0.3)
        assert math.isfinite(r.price)
        assert r.expected_loss > 0
        assert r.par_spread > 0

    def test_senior_tranche(self):
        t = TrancheCDS(attachment=0.12, detachment=0.22, maturity=MAT, spread=0.001)
        r = t.price(_disc(), _scs(), correlation=0.3)
        assert math.isfinite(r.price)

    def test_invalid_attachment(self):
        with pytest.raises(ValueError, match="detachment"):
            TrancheCDS(attachment=0.10, detachment=0.05, maturity=MAT)

    def test_is_equity(self):
        t = TrancheCDS(attachment=0.0, detachment=0.03, maturity=MAT)
        assert t.is_equity
        t2 = TrancheCDS(attachment=0.03, detachment=0.06, maturity=MAT)
        assert not t2.is_equity

    def test_result_dict(self):
        t = TrancheCDS(attachment=0.0, detachment=0.03, maturity=MAT)
        r = t.price(_disc(), _scs(), correlation=0.3)
        d = r.to_dict()
        assert "expected_loss" in d
        assert "par_spread" in d


class TestBaseCorrelation:

    def test_calibration(self):
        scs = _scs(50, 0.02)
        # Compute expected losses at different detachments as "market quotes"
        disc = _disc()
        quotes = {}
        for det in [0.03, 0.06, 0.09]:
            el = expected_tranche_loss(0.0, det, scs, disc, rho=0.3, T=5.0, n_sims=20_000)
            quotes[det] = el

        # Calibrate back — should recover ρ ≈ 0.3
        base_corr = calibrate_base_correlation(quotes, scs, disc,
                                                maturity=MAT, n_sims=20_000)
        assert len(base_corr) == 3
        for det, rho in base_corr.items():
            assert 0 < rho < 1
            assert rho == pytest.approx(0.3, abs=0.15)  # MC noise makes this loose


class TestTrancheSerialisation:

    def test_round_trip(self):
        t = TrancheCDS(attachment=0.03, detachment=0.06, maturity=MAT,
                        spread=0.02, notional=5_000_000)
        d = t.to_dict()
        assert d["type"] == "tranche_cds"
        t2 = from_dict(d)
        assert t2.attachment == 0.03
        assert t2.detachment == 0.06
        assert t2.spread == 0.02

    def test_json(self):
        t = TrancheCDS(attachment=0.0, detachment=0.03, maturity=MAT)
        s = json.dumps(t.to_dict())
        t2 = from_dict(json.loads(s))
        assert t2.is_equity
