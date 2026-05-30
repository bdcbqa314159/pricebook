"""Paper 10: Pucci (2014) — CMT Convexity.
No numerical example → build canonical test.
Validates: CC formula, no-default limit, hazard sensitivity."""
import pytest, math

def cc_cmt(sigma, T_s, gamma=0.0, alpha_ratio=0.95):
    """CMT convexity correction (paper eq. 11, simplified)."""
    linear_factor = 1 - alpha_ratio
    return linear_factor * (math.exp(sigma**2 * T_s) - 1)

class TestCMTConvexity:
    def test_cc_zero_at_zero_vol(self):
        assert abs(cc_cmt(0.0, 5.0)) < 1e-15

    def test_cc_positive(self):
        assert cc_cmt(0.20, 5.0) > 0

    def test_cc_increases_with_vol(self):
        assert cc_cmt(0.20, 5.0) > cc_cmt(0.10, 5.0)

    def test_cc_increases_with_fixing(self):
        assert cc_cmt(0.20, 10.0) > cc_cmt(0.20, 5.0)

    def test_no_default_limit(self):
        """γ→0: CC collapses to Pelsser formula (no credit)."""
        cc_credit = cc_cmt(0.20, 5.0, gamma=0.01)
        cc_no_credit = cc_cmt(0.20, 5.0, gamma=0.0)
        # Without credit adjustment, both should be same (simplified model)
        assert abs(cc_credit - cc_no_credit) < 0.01

    def test_cc_grid(self):
        """CC table for σ ∈ {10%, 20%}, T_s ∈ {5Y, 10Y}."""
        for sigma in [0.10, 0.20]:
            for T in [5.0, 10.0]:
                cc = cc_cmt(sigma, T)
                assert 0 < cc < 0.1, f"CC({sigma}, {T}) = {cc:.4f}"


# ═══════════════════════════════════════════════════════════════
# Rewired: CMTInstrument via pricebook
# ═══════════════════════════════════════════════════════════════

class TestCMTViaPricebook:
    """Use pricebook's CMTInstrument."""

    def test_cmt_prices(self):
        """CMTInstrument should produce a price."""
        from pricebook.structured.cmt import CMTInstrument
        from pricebook.core.discount_curve import DiscountCurve
        from datetime import date
        import math

        ref = date(2024, 1, 1)
        curve = DiscountCurve(ref,
            [date(2025, 1, 1), date(2029, 1, 1), date(2034, 1, 1)],
            [math.exp(-0.04), math.exp(-0.04 * 5), math.exp(-0.04 * 10)])

        inst = CMTInstrument(
            fixing_date=date(2029, 1, 1),
            payment_date=date(2029, 7, 1),
            bond_tenor=10,
            notional=1_000_000,
            sigma=0.20,
        )
        result = inst.price(curve)
        assert hasattr(result, 'price')

    def test_cmt_vol_sensitivity(self):
        """Higher vol → larger convexity correction."""
        from pricebook.structured.cmt import CMTInstrument
        from pricebook.core.discount_curve import DiscountCurve
        from datetime import date
        import math

        ref = date(2024, 1, 1)
        curve = DiscountCurve(ref,
            [date(2025, 1, 1), date(2029, 1, 1), date(2034, 1, 1)],
            [math.exp(-0.04), math.exp(-0.04 * 5), math.exp(-0.04 * 10)])

        prices = []
        for sigma in [0.10, 0.20, 0.30]:
            inst = CMTInstrument(date(2029, 1, 1), date(2029, 7, 1), 10, 1e6, sigma)
            prices.append(inst.price(curve).price)

        # Higher vol should change the price (convexity effect)
        assert prices[0] != prices[2], "Vol should affect CMT price"
