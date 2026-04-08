"""Tests for bond futures: CF, CTD, implied repo, basis."""

import pytest
from datetime import date

from pricebook.bond_futures import (
    conversion_factor, DeliverableBond, cheapest_to_deliver,
    implied_repo_rate, bond_futures_basis,
)
from pricebook.bond import FixedRateBond


REF = date(2024, 1, 15)


def _bond(coupon=0.05, maturity_year=2034):
    return FixedRateBond(REF, date(maturity_year, 1, 15), coupon)


# ---- Conversion factor ----

class TestConversionFactor:
    def test_at_standard_yield(self):
        """Bond with coupon = standard yield → CF ≈ 1."""
        cf = conversion_factor(0.06, 10.0, yield_standard=0.06)
        assert cf == pytest.approx(1.0, abs=0.01)

    def test_higher_coupon_higher_cf(self):
        cf_low = conversion_factor(0.04, 10.0)
        cf_high = conversion_factor(0.08, 10.0)
        assert cf_high > cf_low

    def test_longer_maturity_lower_cf_below_standard(self):
        """Below-standard coupon: longer maturity → lower CF (more discounting)."""
        cf_5y = conversion_factor(0.04, 5.0)
        cf_20y = conversion_factor(0.04, 20.0)
        assert cf_20y < cf_5y

    def test_positive(self):
        cf = conversion_factor(0.05, 10.0)
        assert cf > 0


# ---- CTD ----

class TestCTD:
    def test_single_bond(self):
        d = DeliverableBond(_bond(), market_price=98.0, conversion_factor=0.95)
        result = cheapest_to_deliver([d], 100.0)
        assert result.ctd_index == 0

    def test_ctd_is_lowest_basis(self):
        d1 = DeliverableBond(_bond(0.04), market_price=92.0, conversion_factor=0.90)
        d2 = DeliverableBond(_bond(0.06), market_price=105.0, conversion_factor=1.05)
        d3 = DeliverableBond(_bond(0.05), market_price=98.0, conversion_factor=0.97)
        result = cheapest_to_deliver([d1, d2, d3], 100.0)
        # CTD = min(price - CF × futures)
        bases = [d.market_price - d.conversion_factor * 100.0 for d in [d1, d2, d3]]
        assert result.ctd_index == bases.index(min(bases))

    def test_all_bases_returned(self):
        d1 = DeliverableBond(_bond(0.04), market_price=92.0, conversion_factor=0.90)
        d2 = DeliverableBond(_bond(0.06), market_price=105.0, conversion_factor=1.05)
        result = cheapest_to_deliver([d1, d2], 100.0)
        assert len(result.all_bases) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cheapest_to_deliver([], 100.0)


# ---- Implied repo ----

class TestImpliedRepo:
    def test_positive_repo(self):
        """Profitable delivery → positive implied repo."""
        repo = implied_repo_rate(
            bond_price=98.0, futures_price=100.0, cf=1.0,
            accrued_at_delivery=1.0, coupon_income=2.5,
            days_to_delivery=90,
        )
        assert repo > 0

    def test_zero_days(self):
        repo = implied_repo_rate(98.0, 100.0, 1.0, 1.0, 0.0, 0)
        assert repo == 0.0

    def test_both_positive(self):
        """Profitable delivery at different horizons → positive repo."""
        repo_90 = implied_repo_rate(98.0, 100.0, 1.0, 1.0, 1.25, 90)
        repo_180 = implied_repo_rate(98.0, 100.0, 1.0, 1.0, 2.50, 180)
        assert repo_90 > 0
        assert repo_180 > 0


# ---- Bond futures basis ----

class TestBondFuturesBasis:
    def test_gross_basis(self):
        result = bond_futures_basis(98.0, 100.0, 0.97, 0.05, 0.03, 90)
        assert result.gross_basis == pytest.approx(98.0 - 0.97 * 100.0)

    def test_net_basis_less_than_gross(self):
        """Positive carry → net basis < gross basis."""
        result = bond_futures_basis(98.0, 100.0, 0.97, 0.06, 0.02, 90)
        # Carry = coupon - financing > 0 when coupon_rate > repo_rate
        assert result.carry > 0
        assert result.net_basis < result.gross_basis

    def test_net_basis_approx_zero_for_ctd(self):
        """For the CTD in a simple case, net basis ≈ 0."""
        # CTD at fair value: gross basis ≈ carry
        result = bond_futures_basis(97.0, 100.0, 0.97, 0.05, 0.05, 90)
        # When coupon = repo, carry ≈ 0, so net ≈ gross
        assert abs(result.net_basis - result.gross_basis) < 1.0

    def test_carry_components(self):
        result = bond_futures_basis(100.0, 100.0, 1.0, 0.06, 0.03, 365)
        # Carry = 0.06*100*1 - 100*0.03*1 = 6 - 3 = 3
        assert result.carry == pytest.approx(3.0)
