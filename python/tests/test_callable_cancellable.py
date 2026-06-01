"""Tests for cancellable swap, extendible swap, callable CDS, callable CLN."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

REF = date(2024, 11, 4)


# ═══════════════════════════════════════════════════════════════
# Cancellable Swap
# ═══════════════════════════════════════════════════════════════

class TestCancellableSwap:
    def test_cancellation_cost_positive(self):
        """Cancellation right has positive value → reduces swap PV."""
        from pricebook.fixed_income.cancellable_swap import cancellable_swap_price
        r = cancellable_swap_price(0.03, 0.01, 0.04, 10.0, 0.04)
        assert r.cancellation_cost >= 0

    def test_cancellable_le_vanilla(self):
        """Cancellable payer PV ≤ vanilla payer PV (option costs the payer)."""
        from pricebook.fixed_income.cancellable_swap import cancellable_swap_price
        r = cancellable_swap_price(0.03, 0.01, 0.04, 10.0, 0.04, is_payer=True)
        assert r.cancellable_pv <= r.vanilla_swap_pv + 0.01

    def test_swaption_value_positive(self):
        from pricebook.fixed_income.cancellable_swap import cancellable_swap_price
        r = cancellable_swap_price(0.03, 0.01, 0.04, 10.0, 0.04)
        assert r.swaption_value >= 0

    def test_par_rate_adjusted(self):
        """Cancellable par rate differs from vanilla par rate."""
        from pricebook.fixed_income.cancellable_swap import cancellable_swap_price
        r = cancellable_swap_price(0.03, 0.01, 0.04, 10.0, 0.04)
        assert r.par_rate_cancellable != r.par_rate_vanilla

    def test_to_dict(self):
        from pricebook.fixed_income.cancellable_swap import cancellable_swap_price
        r = cancellable_swap_price(0.03, 0.01, 0.04, 5.0, 0.04)
        d = r.to_dict()
        assert "swaption_value" in d


# ═══════════════════════════════════════════════════════════════
# Extendible Swap
# ═══════════════════════════════════════════════════════════════

class TestExtendibleSwap:
    def test_extension_option_positive(self):
        from pricebook.fixed_income.extendible import extendible_swap_price
        r = extendible_swap_price(0.03, 0.01, 0.04, 5.0, 10.0, 0.04)
        assert r.extension_option_value >= 0

    def test_extendible_ge_base(self):
        """Extendible payer PV ≥ base payer PV (extension adds value for holder)."""
        from pricebook.fixed_income.extendible import extendible_swap_price
        r = extendible_swap_price(0.03, 0.01, 0.04, 5.0, 10.0, 0.04,
                                   is_payer=True, extension_by="payer")
        assert r.extendible_pv >= r.base_swap_pv - 0.01

    def test_to_dict(self):
        from pricebook.fixed_income.extendible import extendible_swap_price
        r = extendible_swap_price(0.03, 0.01, 0.04, 5.0, 10.0, 0.04)
        d = r.to_dict()
        assert "extension_option_value" in d


# ═══════════════════════════════════════════════════════════════
# Callable CDS
# ═══════════════════════════════════════════════════════════════

class TestCallableCDS:
    def _make_curves(self, hazard=0.02):
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.survival_curve import SurvivalCurve
        from pricebook.core.interpolation import InterpolationMethod
        dates = [REF + relativedelta(years=y) for y in range(1, 11)]
        dfs = [math.exp(-0.04 * y) for y in range(1, 11)]
        survs = [math.exp(-hazard * y) for y in range(1, 11)]
        dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
        sc = SurvivalCurve(REF, dates, survs)
        return dc, sc

    def test_termination_option_positive(self):
        from pricebook.credit.callable_cds import callable_cds_price
        dc, sc = self._make_curves()
        r = callable_cds_price(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        assert r.termination_option >= 0

    def test_callable_pv_le_vanilla(self):
        """Callable CDS PV ≤ vanilla (seller's option caps buyer's value)."""
        from pricebook.credit.callable_cds import callable_cds_price
        dc, sc = self._make_curves()
        r = callable_cds_price(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        assert r.callable_pv <= r.vanilla_pv + 0.01

    def test_callable_spread_ge_vanilla(self):
        """Callable CDS par spread ≥ vanilla (seller wants compensation)."""
        from pricebook.credit.callable_cds import callable_cds_price
        dc, sc = self._make_curves()
        r = callable_cds_price(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        assert r.callable_spread >= r.vanilla_spread - 0.001

    def test_to_dict(self):
        from pricebook.credit.callable_cds import callable_cds_price
        dc, sc = self._make_curves()
        r = callable_cds_price(REF, REF + relativedelta(years=5), 0.01, dc, sc)
        d = r.to_dict()
        assert "termination_option" in d


# ═══════════════════════════════════════════════════════════════
# Callable CLN
# ═══════════════════════════════════════════════════════════════

class TestCallableCLN:
    def _make_curves(self, hazard=0.02):
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.survival_curve import SurvivalCurve
        from pricebook.core.interpolation import InterpolationMethod
        dates = [REF + relativedelta(years=y) for y in range(1, 11)]
        dfs = [math.exp(-0.04 * y) for y in range(1, 11)]
        survs = [math.exp(-hazard * y) for y in range(1, 11)]
        dc = DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)
        sc = SurvivalCurve(REF, dates, survs)
        return dc, sc

    def test_call_option_positive(self):
        """Call option has non-negative value."""
        from pricebook.credit.callable_cln import callable_cln_price
        dc, sc = self._make_curves()
        r = callable_cln_price(REF, REF + relativedelta(years=5), 0.06, dc, sc)
        assert r.call_option_value >= 0

    def test_callable_le_straight(self):
        """Callable CLN price ≤ straight CLN (issuer's call caps upside)."""
        from pricebook.credit.callable_cln import callable_cln_price
        dc, sc = self._make_curves()
        r = callable_cln_price(REF, REF + relativedelta(years=5), 0.06, dc, sc)
        assert r.callable_price <= r.straight_price + 0.1

    def test_high_coupon_more_call_value(self):
        """Higher coupon → CLN trades above par → more call option value."""
        from pricebook.credit.callable_cln import callable_cln_price
        dc, sc = self._make_curves(hazard=0.01)  # good credit
        r_low = callable_cln_price(REF, REF + relativedelta(years=5), 0.03, dc, sc)
        r_high = callable_cln_price(REF, REF + relativedelta(years=5), 0.08, dc, sc)
        assert r_high.call_option_value >= r_low.call_option_value

    def test_straight_price_positive(self):
        from pricebook.credit.callable_cln import callable_cln_price
        dc, sc = self._make_curves()
        r = callable_cln_price(REF, REF + relativedelta(years=5), 0.05, dc, sc)
        assert r.straight_price > 0

    def test_to_dict(self):
        from pricebook.credit.callable_cln import callable_cln_price
        dc, sc = self._make_curves()
        r = callable_cln_price(REF, REF + relativedelta(years=5), 0.05, dc, sc)
        d = r.to_dict()
        assert "call_option_value" in d
        assert "par_spread_callable" in d
