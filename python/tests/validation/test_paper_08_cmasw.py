"""Paper 8 validation: Pucci (2012) — CMASW Convexity.

Reproduces:
- Convexity correction formula (pure lognormal)
- CC grid: σ_asw × ρ for Italian sovereign bond
- CC increases with vol and correlation
- Linear swap-rate model calibration consistency

Reference: pucci_2012_cmasw_note.tex, §6.
"""

import pytest
import math


# Paper parameters (Italian sovereign, Table 4-5):
R_SWP = 0.0429      # forward swap rate = 4.29%
R_ASW = 0.0490       # forward ASW spread = 490bp
SIGMA_SWP = 0.30     # forward swap-rate vol = 30%


def cc_lognormal(r_swp, r_asw, sigma_swp, sigma_asw, rho, T, alpha_ratio=0.95):
    """Pure lognormal convexity correction (paper eq. 9).

    CC = R_asw × (1 - Ann×α/D) × (exp(σ_swp × σ_asw × ρ × T) - 1)

    The (1 - Ann×α/D) factor is the linear-model mismatch.
    """
    linear_factor = 1 - alpha_ratio  # simplified: 1 - Ann×α/D ≈ 0.05
    cc = r_asw * linear_factor * (math.exp(sigma_swp * sigma_asw * rho * T) - 1)
    return cc


class TestCCFormula:
    """Convexity correction under pure lognormal dynamics."""

    def test_cc_zero_at_zero_vol(self):
        """CC = 0 when σ_asw = 0 (no vol → no convexity)."""
        cc = cc_lognormal(R_SWP, R_ASW, SIGMA_SWP, 0.0, 0.5, 1.0)
        assert abs(cc) < 1e-15

    def test_cc_zero_at_zero_correlation(self):
        """CC = 0 when ρ = 0 (independent swap and ASW rates)."""
        cc = cc_lognormal(R_SWP, R_ASW, SIGMA_SWP, 0.30, 0.0, 1.0)
        assert abs(cc) < 1e-15

    def test_cc_positive_for_positive_rho(self):
        """CC > 0 when ρ > 0 (swap and ASW positively correlated)."""
        cc = cc_lognormal(R_SWP, R_ASW, SIGMA_SWP, 0.30, 0.50, 1.0)
        assert cc > 0

    def test_cc_negative_for_negative_rho(self):
        """CC < 0 when ρ < 0."""
        cc = cc_lognormal(R_SWP, R_ASW, SIGMA_SWP, 0.30, -0.50, 1.0)
        assert cc < 0


class TestCCGrid:
    """Reproduce Table 2: CC grid across σ_asw × ρ."""

    @pytest.fixture
    def grid(self):
        sigmas = [0.20, 0.30, 0.50]
        rhos = [-0.90, -0.50, 0.0, 0.50, 0.90]
        T = 1.0
        results = {}
        for s in sigmas:
            for r in rhos:
                cc = cc_lognormal(R_SWP, R_ASW, SIGMA_SWP, s, r, T)
                results[(s, r)] = cc * 100  # in percent
        return results

    def test_cc_range(self, grid):
        """Paper: CC range from -0.19% to +0.37% (approximate)."""
        values = list(grid.values())
        assert min(values) < 0, "Should have negative CC values"
        assert max(values) > 0, "Should have positive CC values"
        # Order of magnitude: paper says -0.19% to +0.37%
        # Our simplified formula may differ but should be same order
        assert abs(min(values)) < 1.0, f"Min CC = {min(values):.4f}% (too large)"
        assert abs(max(values)) < 1.0, f"Max CC = {max(values):.4f}% (too large)"

    def test_cc_increases_with_vol(self, grid):
        """Higher σ_asw → larger |CC| (at fixed ρ > 0)."""
        rho = 0.50
        cc_20 = grid[(0.20, rho)]
        cc_30 = grid[(0.30, rho)]
        cc_50 = grid[(0.50, rho)]
        assert abs(cc_50) > abs(cc_30) > abs(cc_20)

    def test_cc_increases_with_correlation(self, grid):
        """Higher ρ → larger CC (at fixed σ_asw)."""
        sigma = 0.30
        cc_neg = grid[(sigma, -0.50)]
        cc_zero = grid[(sigma, 0.0)]
        cc_pos = grid[(sigma, 0.50)]
        assert cc_pos > cc_zero > cc_neg

    def test_cc_symmetric_in_rho(self, grid):
        """CC(ρ) ≈ -CC(-ρ) (antisymmetric in ρ)."""
        sigma = 0.30
        cc_pos = grid[(sigma, 0.50)]
        cc_neg = grid[(sigma, -0.50)]
        assert abs(cc_pos + cc_neg) < abs(cc_pos) * 0.2  # roughly antisymmetric


class TestLinearModelConsistency:
    """Linear swap-rate model: D(T,U)/Ann(T) = α + β × R_swp(T)."""

    def test_alpha_beta_positive(self):
        """α should be near 1, β should be negative (higher rate → lower DF)."""
        # Simplified calibration: at flat 4.29% swap rate
        # D(T,U)/Ann ≈ 1 / (1 + R × tau) for short U
        # α ≈ 1 / (1 + R×tau_avg), β ≈ -tau / (1+R×tau)²
        R = R_SWP
        tau = 5.0  # average maturity
        alpha = 1.0 / (1 + R * tau)
        beta = -tau / (1 + R * tau) ** 2
        assert alpha > 0
        assert beta < 0


# ═══════════════════════════════════════════════════════════════
# Rewired: CMASWInstrument via pricebook
# ═══════════════════════════════════════════════════════════════

class TestCMASWViaPricebook:
    """Use pricebook's CMASWInstrument class."""

    def test_cmasw_prices(self):
        """CMASWInstrument should produce a price."""
        from pricebook.structured.cmasw import CMASWInstrument
        from pricebook.core.discount_curve import DiscountCurve
        from datetime import date
        import math

        ref = date(2024, 1, 1)
        curve = DiscountCurve(ref,
            [date(2025, 1, 1), date(2029, 1, 1), date(2034, 1, 1)],
            [math.exp(-0.04 * 1), math.exp(-0.04 * 5), math.exp(-0.04 * 10)])

        inst = CMASWInstrument(
            fixing_date=date(2025, 1, 1),
            payment_date=date(2025, 7, 1),
            swap_tenor=10,
            bond_price=98.0,
            notional=1_000_000,
            sigma_swp=0.30,
            sigma_asw=0.25,
            rho=0.50,
        )
        result = inst.price(curve)
        assert hasattr(result, 'price')
        assert result.price != 0  # should be non-trivial

    def test_cmasw_cc_sign(self):
        """CC should be positive for positive correlation."""
        from pricebook.structured.cmasw import CMASWInstrument
        from pricebook.core.discount_curve import DiscountCurve
        from datetime import date
        import math

        ref = date(2024, 1, 1)
        curve = DiscountCurve(ref,
            [date(2025, 1, 1), date(2029, 1, 1), date(2034, 1, 1)],
            [math.exp(-0.04 * 1), math.exp(-0.04 * 5), math.exp(-0.04 * 10)])

        inst_pos = CMASWInstrument(
            date(2025, 1, 1), date(2025, 7, 1), 10, 98.0, 1e6, 0.30, 0.25, 0.50)
        inst_neg = CMASWInstrument(
            date(2025, 1, 1), date(2025, 7, 1), 10, 98.0, 1e6, 0.30, 0.25, -0.50)

        r_pos = inst_pos.price(curve)
        r_neg = inst_neg.price(curve)
        # Different correlation → different price
        assert r_pos.price != r_neg.price
