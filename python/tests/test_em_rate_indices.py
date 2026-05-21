"""Tests for EM rate index registry.

Covers: EM overnight RFR, EM term IBOR, registry lookup, currency grouping.
"""

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.rate_index import (
    get_rate_index, all_rate_indices, overnight_indices,
    indices_for_currency, CompoundingMethod, RateIndex,
    # EM overnight
    CDI, KOFR, SORA, HONIA, THOR, DR007, IBR, TPM,
    # EM term
    TIIE_28D, SHIBOR_3M, WIBOR_3M, PRIBOR_3M, BUBOR_3M, JIBAR_3M,
)


class TestEMOvernightIndices:
    def test_cdi(self):
        idx = get_rate_index("CDI")
        assert idx.currency == "BRL"
        assert idx.is_overnight
        assert idx.day_count == DayCountConvention.BUS_252
        assert idx.compounding == CompoundingMethod.COMPOUNDED
        assert idx.administrator == "B3"

    def test_kofr(self):
        idx = get_rate_index("KOFR")
        assert idx.currency == "KRW"
        assert idx.is_overnight

    def test_sora(self):
        idx = get_rate_index("SORA")
        assert idx.currency == "SGD"
        assert idx.is_overnight

    def test_honia(self):
        idx = get_rate_index("HONIA")
        assert idx.currency == "HKD"
        assert idx.is_overnight

    def test_thor(self):
        idx = get_rate_index("THOR")
        assert idx.currency == "THB"
        assert idx.is_overnight

    def test_dr007(self):
        idx = get_rate_index("DR007")
        assert idx.currency == "CNY"
        assert idx.is_overnight
        assert idx.compounding == CompoundingMethod.AVERAGED

    def test_ibr(self):
        idx = get_rate_index("IBR")
        assert idx.currency == "COP"
        assert idx.is_overnight

    def test_tpm(self):
        idx = get_rate_index("TPM")
        assert idx.currency == "CLP"
        assert idx.is_overnight


class TestEMTermIndices:
    def test_tiie_28d(self):
        idx = get_rate_index("TIIE_28D")
        assert idx.currency == "MXN"
        assert not idx.is_overnight
        assert idx.tenor_months == 1
        assert idx.fixing_lag == 1  # T-1 for TIIE

    def test_shibor_3m(self):
        idx = get_rate_index("SHIBOR_3M")
        assert idx.currency == "CNY"
        assert idx.tenor_months == 3

    def test_wibor_3m(self):
        idx = get_rate_index("WIBOR_3M")
        assert idx.currency == "PLN"
        assert idx.tenor_months == 3
        assert idx.fixing_lag == 2

    def test_pribor_3m(self):
        idx = get_rate_index("PRIBOR_3M")
        assert idx.currency == "CZK"
        assert idx.tenor_months == 3

    def test_bubor_3m(self):
        idx = get_rate_index("BUBOR_3M")
        assert idx.currency == "HUF"
        assert idx.tenor_months == 3

    def test_jibar_3m(self):
        idx = get_rate_index("JIBAR_3M")
        assert idx.currency == "ZAR"
        assert idx.tenor_months == 3
        assert idx.day_count == DayCountConvention.ACT_365_FIXED


class TestRegistryCounts:
    def test_total_count(self):
        """11 G10 + 14 EM = 25 total indices."""
        all_idx = all_rate_indices()
        assert len(all_idx) == 25

    def test_overnight_count(self):
        """8 G10 overnight + 8 EM overnight = 16."""
        ovn = overnight_indices()
        assert len(ovn) == 16

    def test_cny_has_two(self):
        """CNY has DR007 (overnight) and SHIBOR_3M (term)."""
        cny = indices_for_currency("CNY")
        assert len(cny) == 2
        names = {i.name for i in cny}
        assert "DR007" in names
        assert "SHIBOR_3M" in names

    def test_brl_indices(self):
        brl = indices_for_currency("BRL")
        assert len(brl) == 1
        assert brl[0].name == "CDI"

    def test_em_currencies_covered(self):
        """All major EM currencies have at least one index."""
        em_ccys = ["BRL", "MXN", "CNY", "KRW", "SGD", "HKD", "THB",
                    "PLN", "CZK", "HUF", "ZAR", "COP", "CLP"]
        for ccy in em_ccys:
            indices = indices_for_currency(ccy)
            assert len(indices) >= 1, f"No index for {ccy}"


class TestToDict:
    def test_to_dict(self):
        d = CDI.to_dict()
        assert d["name"] == "CDI"
        assert d["currency"] == "BRL"
        assert d["is_overnight"] is True

    def test_frozen(self):
        with pytest.raises(Exception):
            CDI.name = "WRONG"
