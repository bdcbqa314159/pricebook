"""Tests for cross-currency basis."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.xccy_basis import implied_basis_spread, bootstrap_basis_curve
from pricebook.fx_forward import FXForward
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)
SPOT = 1.10


class TestImpliedBasisSpread:

    def test_zero_basis_when_cip_holds(self):
        """If market forward = CIP forward, basis = 0."""
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)
        basis = implied_basis_spread(SPOT, mat, fwd_cip, eur, usd)
        assert basis == pytest.approx(0.0, abs=1e-6)

    def test_positive_basis_when_forward_above_cip(self):
        """Market forward above CIP -> positive basis (quote rate must rise
        to produce a higher forward via F = S * df_base / df_quote)."""
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)
        basis = implied_basis_spread(SPOT, mat, fwd_cip + 0.005, eur, usd)
        assert basis > 0

    def test_negative_basis_when_forward_below_cip(self):
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)
        basis = implied_basis_spread(SPOT, mat, fwd_cip - 0.005, eur, usd)
        assert basis < 0

    def test_basis_magnitude_reasonable(self):
        """Typical xccy basis is single-digit bps to tens of bps."""
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        mat = REF + relativedelta(years=1)
        fwd_cip = FXForward.forward_rate(SPOT, mat, eur, usd)
        # 5 pips off CIP
        basis = implied_basis_spread(SPOT, mat, fwd_cip + 0.0005, eur, usd)
        assert abs(basis) < 0.01  # less than 100bp


class TestBootstrapBasisCurve:

    def _market_forwards(self, eur, usd, basis_bps: float = -15):
        """Generate synthetic market forwards with a flat basis."""
        basis = basis_bps / 10000.0
        tenors = [0.5, 1.0, 2.0, 3.0, 5.0]
        forwards = []
        for t in tenors:
            mat = date.fromordinal(REF.toordinal() + int(t * 365))
            # Adjusted forward: use (usd_rate + basis) for discounting
            df_eur = eur.df(mat)
            usd_z = usd.zero_rate(mat)
            df_usd_adj = math.exp(-(usd_z + basis) * t)
            fwd = SPOT * df_eur / df_usd_adj
            forwards.append((mat, fwd))
        return forwards

    def test_reprices_all_forwards(self):
        """Basis-adjusted curve reprices all market forwards."""
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        forwards = self._market_forwards(eur, usd, basis_bps=-15)

        adj_curve = bootstrap_basis_curve(REF, SPOT, forwards, eur, usd)

        for mat, fwd_market in forwards:
            fwd_repriced = FXForward.forward_rate(SPOT, mat, eur, adj_curve)
            assert fwd_repriced == pytest.approx(fwd_market, rel=1e-6), \
                f"Failed at {mat}: market={fwd_market:.6f}, repriced={fwd_repriced:.6f}"

    def test_adjusted_curve_differs_from_original(self):
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        forwards = self._market_forwards(eur, usd, basis_bps=-15)
        adj = bootstrap_basis_curve(REF, SPOT, forwards, eur, usd)
        mat = date.fromordinal(REF.toordinal() + int(2 * 365))
        assert adj.zero_rate(mat) != pytest.approx(usd.zero_rate(mat), rel=1e-3)

    def test_positive_dfs(self):
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        forwards = self._market_forwards(eur, usd, basis_bps=-15)
        adj = bootstrap_basis_curve(REF, SPOT, forwards, eur, usd)
        for mat, _ in forwards:
            assert adj.df(mat) > 0

    def test_zero_basis_recovers_original(self):
        """With zero basis, the adjusted curve should match the original."""
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        forwards = self._market_forwards(eur, usd, basis_bps=0)
        adj = bootstrap_basis_curve(REF, SPOT, forwards, eur, usd)
        mat = date.fromordinal(REF.toordinal() + int(2 * 365))
        assert adj.zero_rate(mat) == pytest.approx(usd.zero_rate(mat), rel=1e-3)

    def test_unsorted_raises(self):
        bad = [(REF + relativedelta(years=2), 1.12), (REF + relativedelta(years=1), 1.11)]
        eur = make_flat_curve(REF, rate=0.04)
        usd = make_flat_curve(REF, rate=0.05)
        with pytest.raises(ValueError):
            bootstrap_basis_curve(REF, SPOT, bad, eur, usd)
