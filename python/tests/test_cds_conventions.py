"""Tests for CDS conventions."""

from datetime import date
import pytest

from pricebook.cds_conventions import (
    CDS_SETTLEMENT,
    CDSIndexSpec,
    STANDARD_COUPONS_BPS,
    STANDARD_RECOVERY,
    UpfrontResult,
    cds_index_roll_date,
    get_index_spec,
    next_imm_date,
    par_spread_from_upfront,
    standard_cds_dates,
    upfront_from_par_spread,
)


class TestIMMDates:
    def test_next_imm_on_imm(self):
        assert next_imm_date(date(2026, 3, 20)) == date(2026, 3, 20)

    def test_next_imm_after(self):
        assert next_imm_date(date(2026, 3, 21)) == date(2026, 6, 20)

    def test_next_imm_jan(self):
        assert next_imm_date(date(2026, 1, 15)) == date(2026, 3, 20)

    def test_next_imm_dec(self):
        assert next_imm_date(date(2026, 12, 21)) == date(2027, 3, 20)


class TestStandardCDSDates:
    def test_5y_quarterly(self):
        dates = standard_cds_dates(date(2026, 4, 1), maturity_years=5)
        # Should have ~20 quarterly dates
        assert len(dates) == 20
        # All on 20th
        assert all(d.day == 20 for d in dates)
        # All in IMM months
        assert all(d.month in [3, 6, 9, 12] for d in dates)

    def test_1y(self):
        dates = standard_cds_dates(date(2026, 4, 1), maturity_years=1)
        assert len(dates) == 4

    def test_effective_snaps_to_previous_imm(self):
        """Effective date snaps to previous IMM (20 Mar for Apr trade)."""
        dates = standard_cds_dates(date(2026, 4, 1), maturity_years=1)
        # First payment after Mar 20 snap
        assert dates[0] == date(2026, 6, 20)


class TestUpfrontConversion:
    def test_ig_at_par(self):
        """Par spread = standard coupon → zero upfront."""
        result = upfront_from_par_spread(100, standard_coupon_bps=100, risky_annuity=4.0)
        assert result.upfront_pct == pytest.approx(0.0)

    def test_ig_wider_than_coupon(self):
        """Spread > coupon → protection buyer pays upfront."""
        result = upfront_from_par_spread(150, 100, 4.0)
        assert result.upfront_pct > 0
        # (150-100) × 4.0 / 10000 = 0.02 = 2%
        assert result.upfront_pct == pytest.approx(0.02)

    def test_ig_tighter_than_coupon(self):
        """Spread < coupon → protection buyer receives upfront."""
        result = upfront_from_par_spread(60, 100, 4.0)
        assert result.upfront_pct < 0

    def test_hy_standard(self):
        result = upfront_from_par_spread(350, 500, 4.0)
        assert result.upfront_pct < 0  # spread < coupon → buyer receives

    def test_round_trip(self):
        """upfront → par_spread → upfront should round-trip."""
        original = 150
        uf = upfront_from_par_spread(original, 100, 4.2)
        recovered = par_spread_from_upfront(uf.upfront_pct, 100, 4.2)
        assert recovered == pytest.approx(original)


class TestIndexSpecs:
    def test_cdx_ig(self):
        spec = get_index_spec("CDX.NA.IG")
        assert spec.n_names == 125
        assert spec.standard_coupon_bps == 100
        assert spec.standard_recovery == 0.40

    def test_cdx_hy(self):
        spec = get_index_spec("CDX.NA.HY")
        assert spec.n_names == 100
        assert spec.standard_coupon_bps == 500
        assert spec.standard_recovery == 0.25

    def test_itraxx_europe(self):
        spec = get_index_spec("ITRAXX.EUR.IG")
        assert spec.n_names == 125
        assert spec.region == "EU"

    def test_itraxx_crossover(self):
        spec = get_index_spec("ITRAXX.EUR.XOVER")
        assert spec.n_names == 75
        assert spec.standard_coupon_bps == 500

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_index_spec("NONEXISTENT")

    def test_fuzzy_match(self):
        spec = get_index_spec("CDX.NA.IG")
        assert spec.name == "CDX.NA.IG"


class TestRollDates:
    def test_next_roll_from_jan(self):
        d = cds_index_roll_date("CDX.NA.IG", date(2026, 1, 15))
        assert d == date(2026, 3, 20)

    def test_next_roll_from_apr(self):
        d = cds_index_roll_date("CDX.NA.IG", date(2026, 4, 1))
        assert d == date(2026, 9, 20)

    def test_next_roll_from_oct(self):
        d = cds_index_roll_date("CDX.NA.IG", date(2026, 10, 1))
        assert d == date(2027, 3, 20)


class TestSettlement:
    def test_default_auction(self):
        assert CDS_SETTLEMENT.method == "auction"
        assert CDS_SETTLEMENT.accrued_on_default is True

    def test_standard_coupons(self):
        assert STANDARD_COUPONS_BPS["IG"] == 100
        assert STANDARD_COUPONS_BPS["HY"] == 500

    def test_standard_recovery(self):
        assert STANDARD_RECOVERY["IG"] == 0.40
        assert STANDARD_RECOVERY["HY"] == 0.25
