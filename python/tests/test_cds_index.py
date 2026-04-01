"""Tests for CDS Index and Vanilla CLN."""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.cds_index import CDSIndex, VanillaCLN
from pricebook.cds import CDS
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


def _make_cds(spread=0.01):
    return CDS(REF, END, spread=spread, notional=1_000_000, recovery=0.4)


class TestCDSIndex:
    def test_pv_sum_of_constituents(self):
        """Index PV = weighted sum of constituent PVs."""
        disc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, h) for h in [0.01, 0.02, 0.03]]
        cdss = [_make_cds(0.01) for _ in range(3)]
        index = CDSIndex(cdss, notional=3_000_000)

        index_pv = index.pv(disc, survs)
        manual_sum = sum(cds.pv(disc, sc) for cds, sc in zip(cdss, survs))
        assert index_pv == pytest.approx(manual_sum, rel=0.01)

    def test_intrinsic_spread(self):
        """Intrinsic = average of par spreads."""
        disc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, h) for h in [0.01, 0.02, 0.03]]
        cdss = [_make_cds() for _ in range(3)]
        index = CDSIndex(cdss)

        intrinsic = index.intrinsic_spread(disc, survs)
        avg_par = sum(cds.par_spread(disc, sc) for cds, sc in zip(cdss, survs)) / 3
        assert intrinsic == pytest.approx(avg_par)

    def test_intrinsic_positive(self):
        disc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, 0.02) for _ in range(5)]
        cdss = [_make_cds() for _ in range(5)]
        index = CDSIndex(cdss)
        assert index.intrinsic_spread(disc, survs) > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            CDSIndex([])

    def test_mismatched_curves_raises(self):
        disc = make_flat_curve(REF, 0.04)
        cdss = [_make_cds() for _ in range(3)]
        index = CDSIndex(cdss)
        with pytest.raises(ValueError, match="Expected 3"):
            index.pv(disc, [make_flat_survival(REF, 0.02)])


class TestVanillaCLN:
    def test_price_positive(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln = VanillaCLN(REF, END, coupon_rate=0.06)
        assert cln.dirty_price(disc, surv) > 0

    def test_cln_less_than_risk_free(self):
        """CLN price < risk-free equivalent (credit risk reduces value)."""
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln = VanillaCLN(REF, END, coupon_rate=0.06)
        risky = cln.dirty_price(disc, surv)
        riskfree = cln.risk_free_equivalent_price(disc)
        assert risky < riskfree

    def test_zero_hazard_equals_risk_free(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.0001)
        cln = VanillaCLN(REF, END, coupon_rate=0.06)
        risky = cln.dirty_price(disc, surv)
        riskfree = cln.risk_free_equivalent_price(disc)
        assert risky == pytest.approx(riskfree, rel=0.01)

    def test_credit_spread_positive(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln = VanillaCLN(REF, END, coupon_rate=0.06)
        cs = cln.credit_spread(disc, surv)
        assert cs > 0

    def test_higher_hazard_higher_spread(self):
        disc = make_flat_curve(REF, 0.04)
        surv_low = make_flat_survival(REF, 0.01)
        surv_high = make_flat_survival(REF, 0.05)
        cln = VanillaCLN(REF, END, coupon_rate=0.06)
        cs_low = cln.credit_spread(disc, surv_low)
        cs_high = cln.credit_spread(disc, surv_high)
        assert cs_high > cs_low

    def test_higher_coupon_higher_price(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln_low = VanillaCLN(REF, END, coupon_rate=0.04)
        cln_high = VanillaCLN(REF, END, coupon_rate=0.08)
        assert cln_high.dirty_price(disc, surv) > cln_low.dirty_price(disc, surv)

    def test_higher_recovery_higher_price(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.03)
        cln_low = VanillaCLN(REF, END, coupon_rate=0.06, recovery=0.2)
        cln_high = VanillaCLN(REF, END, coupon_rate=0.06, recovery=0.6)
        assert cln_high.dirty_price(disc, surv) > cln_low.dirty_price(disc, surv)
