"""Tests for CDS index depth: factor, CS01, index options, basis decomposition."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cds import CDS
from pricebook.cds_index import CDSIndex
from pricebook.cds_index_product import (
    CDSIndexProduct, IndexResult,
    CDSIndexOption, IndexOptionResult,
    index_basis_decomposition, BasisDecomposition,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END_5Y = REF + relativedelta(years=5)
N = 5  # small index for testing


def _make_index(n=N, hazards=None, market_spread=0.01):
    """Make a small CDSIndexProduct with survival curves."""
    dc = make_flat_curve(REF, 0.04)
    if hazards is None:
        hazards = [0.01 + 0.005 * i for i in range(n)]
    survs = [make_flat_survival(REF, h) for h in hazards]
    product = CDSIndexProduct(
        "TEST", series=1, market_spread=market_spread,
        start=REF, end=END_5Y, notional=10_000_000,
        standard_coupon=0.01, recovery=0.4, n_names=n,
    )
    return product, dc, survs


# ---- Index factor ----

class TestIndexFactor:

    def test_no_defaults(self):
        p = CDSIndexProduct("CDX.NA.IG", n_names=125, n_defaulted=0)
        assert p.factor == 1.0

    def test_with_defaults(self):
        p = CDSIndexProduct("CDX.NA.IG", n_names=125, n_defaulted=5)
        assert p.factor == pytest.approx(120 / 125)

    def test_effective_notional(self):
        p = CDSIndexProduct("CDX.NA.IG", n_names=125, n_defaulted=5,
                             notional=10_000_000)
        assert p.effective_notional == pytest.approx(10_000_000 * 120 / 125)

    def test_factor_in_serialisation(self):
        p = CDSIndexProduct("CDX.NA.IG", n_names=125, n_defaulted=3,
                             market_spread=0.005)
        d = p.to_dict()
        p2 = CDSIndexProduct.from_dict(d)
        assert p2.n_defaulted == 3
        assert p2.factor == pytest.approx(p.factor)


# ---- Index CS01 ----

class TestIndexCS01:

    def test_cs01_nonzero(self):
        product, dc, survs = _make_index()
        cs = product.cs01(dc, survs)
        assert cs != 0

    def test_constituent_cs01_sums(self):
        """Sum of per-name CS01 ≈ index CS01."""
        product, dc, survs = _make_index()
        total = product.cs01(dc, survs)
        per_name = product.constituent_cs01(dc, survs)
        assert sum(per_name) == pytest.approx(total, rel=0.05)

    def test_rec01_negative(self):
        """Higher recovery reduces protection → rec01 < 0 for buyer."""
        product, dc, survs = _make_index()
        r = product.rec01(dc, survs)
        assert r < 0


# ---- Index option ----

class TestIndexOption:

    def test_positive_premium(self):
        product, dc, survs = _make_index()
        opt = CDSIndexOption(
            expiry_date=REF + relativedelta(months=3),
            maturity_date=END_5Y,
            strike_spread=0.01, spread_vol=0.4,
            n_names=N,
        )
        result = opt.price(dc, survs)
        assert result.premium > 0

    def test_payer_vs_receiver(self):
        product, dc, survs = _make_index()
        payer = CDSIndexOption(
            expiry_date=REF + relativedelta(months=3),
            maturity_date=END_5Y,
            strike_spread=0.01, spread_vol=0.4,
            option_type="payer", n_names=N,
        )
        receiver = CDSIndexOption(
            expiry_date=REF + relativedelta(months=3),
            maturity_date=END_5Y,
            strike_spread=0.01, spread_vol=0.4,
            option_type="receiver", n_names=N,
        )
        p = payer.price(dc, survs)
        r = receiver.price(dc, survs)
        assert p.premium > 0
        assert r.premium > 0

    def test_put_call_parity(self):
        """Payer - Receiver ≈ Q × A × (F - K) × notional."""
        product, dc, survs = _make_index()
        K = 0.012
        payer = CDSIndexOption(
            expiry_date=REF + relativedelta(months=6),
            maturity_date=END_5Y,
            strike_spread=K, spread_vol=0.4,
            option_type="payer", n_names=N,
        )
        receiver = CDSIndexOption(
            expiry_date=REF + relativedelta(months=6),
            maturity_date=END_5Y,
            strike_spread=K, spread_vol=0.4,
            option_type="receiver", n_names=N,
        )
        p = payer.price(dc, survs)
        r = receiver.price(dc, survs)
        forward_pv = p.survival_factor * 10_000_000 * p.index_annuity * (p.forward_spread - K)
        diff = p.premium - r.premium
        assert diff == pytest.approx(forward_pv, rel=0.02)

    def test_price_increases_with_vol(self):
        product, dc, survs = _make_index()
        low = CDSIndexOption(
            expiry_date=REF + relativedelta(months=6),
            maturity_date=END_5Y,
            strike_spread=0.01, spread_vol=0.2, n_names=N,
        )
        high = CDSIndexOption(
            expiry_date=REF + relativedelta(months=6),
            maturity_date=END_5Y,
            strike_spread=0.01, spread_vol=0.6, n_names=N,
        )
        p_low = low.price(dc, survs).premium
        p_high = high.price(dc, survs).premium
        assert p_high > p_low

    def test_serialisation(self):
        opt = CDSIndexOption(
            index_name="CDX.NA.IG",
            expiry_date=REF + relativedelta(months=3),
            maturity_date=END_5Y,
            strike_spread=0.005, spread_vol=0.4,
        )
        d = opt.to_dict()
        opt2 = CDSIndexOption.from_dict(d)
        assert opt2.strike_spread == 0.005
        assert opt2.index_name == "CDX.NA.IG"


# ---- Basis decomposition ----

class TestBasisDecomposition:

    def test_components_sum(self):
        """dispersion + liquidity = total basis."""
        product, dc, survs = _make_index()
        bd = index_basis_decomposition(product, dc, survs)
        assert (bd.dispersion_bp + bd.liquidity_bp) == pytest.approx(bd.total_basis_bp, rel=0.1)

    def test_finite(self):
        product, dc, survs = _make_index()
        bd = index_basis_decomposition(product, dc, survs)
        assert math.isfinite(bd.total_basis_bp)
        assert math.isfinite(bd.dispersion_bp)

    def test_to_dict(self):
        product, dc, survs = _make_index()
        bd = index_basis_decomposition(product, dc, survs)
        d = bd.to_dict()
        assert "total_basis_bp" in d
        assert "dispersion_bp" in d


# ---- Improved flat_spread ----

class TestImprovedFlatSpread:

    def test_flat_spread_positive(self):
        dc = make_flat_curve(REF, 0.04)
        survs = [make_flat_survival(REF, 0.01 + 0.005 * i) for i in range(5)]
        constituents = [CDS(REF, END_5Y, spread=0.01, notional=1_000_000) for _ in range(5)]
        idx = CDSIndex(constituents, 5_000_000)
        fs = idx.flat_spread(dc, survs)
        assert fs > 0

    def test_flat_spread_finite(self):
        """Flat spread should be a finite positive number."""
        dc = make_flat_curve(REF, 0.04)
        hazards = [0.01, 0.02, 0.03, 0.04, 0.05]
        survs = [make_flat_survival(REF, h) for h in hazards]
        constituents = [CDS(REF, END_5Y, spread=0.01, notional=1_000_000) for _ in range(5)]
        idx = CDSIndex(constituents, 5_000_000)
        fs = idx.flat_spread(dc, survs)
        intrinsic = idx.intrinsic_spread(dc, survs)
        assert math.isfinite(fs)
        assert fs > 0
        # Flat and intrinsic should be in the same range
        assert abs(fs - intrinsic) / intrinsic < 0.5
