"""Tests for inflation index registry and linker factory.

Covers: index lookup, convention correctness, linker factory, registry API.
"""

import pytest
from datetime import date

from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.inflation_indices import (
    get_inflation_index, list_inflation_indices, indices_by_currency,
    indices_with_floor, daily_indices, create_inflation_linker,
    InflationIndexDef, IndexInterpolation,
)


REF = date(2024, 1, 15)
MAT = date(2034, 1, 15)


# ═══════════════════════════════════════════════════════════════
# Index lookup
# ═══════════════════════════════════════════════════════════════


class TestIndexLookup:
    def test_cpi_us(self):
        idx = get_inflation_index("CPI_US")
        assert idx.currency == "USD"
        assert idx.publication_lag_months == 3
        assert idx.deflation_floor is True
        assert idx.interpolation == IndexInterpolation.LINEAR
        assert idx.linker_frequency == Frequency.SEMI_ANNUAL

    def test_hicp_xt(self):
        idx = get_inflation_index("HICP_XT")
        assert idx.currency == "EUR"
        assert idx.deflation_floor is False
        assert idx.linker_frequency == Frequency.ANNUAL

    def test_rpi(self):
        """UK RPI: 8-month lag, flat interpolation."""
        idx = get_inflation_index("RPI")
        assert idx.publication_lag_months == 8
        assert idx.interpolation == IndexInterpolation.FLAT

    def test_ipca(self):
        """IPCA: 1-month lag, BUS/252."""
        idx = get_inflation_index("IPCA")
        assert idx.currency == "BRL"
        assert idx.publication_lag_months == 1
        assert idx.linker_day_count == DayCountConvention.BUS_252

    def test_udi_daily(self):
        """UDI: daily publication, zero lag."""
        idx = get_inflation_index("UDI")
        assert idx.publication_lag_months == 0
        assert idx.publication_frequency == "daily"
        assert idx.interpolation == IndexInterpolation.DAILY

    def test_uf_daily(self):
        idx = get_inflation_index("UF")
        assert idx.publication_frequency == "daily"
        assert idx.currency == "CLP"

    def test_uvr_daily(self):
        idx = get_inflation_index("UVR")
        assert idx.publication_frequency == "daily"
        assert idx.currency == "COP"

    def test_cpi_za(self):
        idx = get_inflation_index("CPI_ZA")
        assert idx.currency == "ZAR"
        assert idx.publication_lag_months == 3

    def test_cpi_jp(self):
        idx = get_inflation_index("CPI_JP")
        assert idx.currency == "JPY"
        assert idx.deflation_floor is True

    def test_cpi_in(self):
        """India: 30/360 day count for linkers."""
        idx = get_inflation_index("CPI_IN")
        assert idx.linker_day_count == DayCountConvention.THIRTY_360
        assert idx.deflation_floor is True

    def test_cpi_tr(self):
        idx = get_inflation_index("CPI_TR")
        assert idx.currency == "TRY"

    def test_cpi_kr(self):
        idx = get_inflation_index("CPI_KR")
        assert idx.currency == "KRW"

    def test_cpi_il(self):
        idx = get_inflation_index("CPI_IL")
        assert idx.currency == "ILS"
        assert idx.linker_frequency == Frequency.ANNUAL

    def test_cpi_ca(self):
        idx = get_inflation_index("CPI_CA")
        assert idx.currency == "CAD"
        assert idx.deflation_floor is True

    def test_cpi_au_quarterly(self):
        """Australia: quarterly CPI publication."""
        idx = get_inflation_index("CPI_AU")
        assert idx.linker_frequency == Frequency.QUARTERLY

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown inflation index"):
            get_inflation_index("FAKE")

    def test_case_insensitive(self):
        idx = get_inflation_index("ipca")
        assert idx.name == "IPCA"


# ═══════════════════════════════════════════════════════════════
# Registry API
# ═══════════════════════════════════════════════════════════════


class TestRegistryAPI:
    def test_list_count(self):
        indices = list_inflation_indices()
        assert len(indices) == 18

    def test_list_sorted(self):
        indices = list_inflation_indices()
        assert indices == sorted(indices)

    def test_indices_by_currency_gbp(self):
        """GBP has RPI and CPIH."""
        gbp = indices_by_currency("GBP")
        names = {i.name for i in gbp}
        assert "RPI" in names
        assert "CPIH" in names
        assert len(gbp) == 2

    def test_indices_with_floor(self):
        """TIPS, JGBi, RRBs, India IIBs, KTBi have deflation floor."""
        floored = indices_with_floor()
        names = {i.name for i in floored}
        assert "CPI_US" in names
        assert "CPI_JP" in names
        assert "CPI_CA" in names
        assert "CPI_IN" in names
        assert "CPI_KR" in names
        assert len(floored) == 5

    def test_daily_indices(self):
        """UDI, UF, UVR are daily."""
        daily = daily_indices()
        names = {i.name for i in daily}
        assert names == {"UDI", "UF", "UVR", "CER"}

    def test_no_floor_for_europe(self):
        """European linkers (OATi, BTPi) have no deflation floor."""
        hicp = get_inflation_index("HICP_XT")
        assert hicp.deflation_floor is False
        rpi = get_inflation_index("RPI")
        assert rpi.deflation_floor is False


# ═══════════════════════════════════════════════════════════════
# Linker factory
# ═══════════════════════════════════════════════════════════════


class TestLinkerFactory:
    def test_tips(self):
        params = create_inflation_linker("CPI_US", REF, MAT, 0.02, 300.0)
        assert params["frequency"] == Frequency.SEMI_ANNUAL
        assert params["day_count"] == DayCountConvention.ACT_ACT_ICMA
        assert params["cpi_lag_months"] == 3
        assert params["coupon_rate"] == 0.02
        assert params["base_cpi_value"] == 300.0
        assert params["_deflation_floor"] is True

    def test_ntn_b(self):
        params = create_inflation_linker("IPCA", REF, MAT, 0.06, 6500.0)
        assert params["day_count"] == DayCountConvention.BUS_252
        assert params["cpi_lag_months"] == 1
        assert params["_deflation_floor"] is False

    def test_oatei(self):
        params = create_inflation_linker("HICP_XT", REF, MAT, 0.01, 115.0)
        assert params["frequency"] == Frequency.ANNUAL
        assert params["_deflation_floor"] is False

    def test_uk_ilg(self):
        params = create_inflation_linker("RPI", REF, MAT, 0.0125, 390.0)
        assert params["cpi_lag_months"] == 8
        assert params["_interpolation"] == "flat"

    def test_udibono(self):
        params = create_inflation_linker("UDI", REF, MAT, 0.035, 7.5)
        assert params["cpi_lag_months"] == 0
        assert params["_interpolation"] == "daily"
        assert params["day_count"] == DayCountConvention.ACT_360

    def test_all_indices_produce_params(self):
        for name in list_inflation_indices():
            params = create_inflation_linker(name, REF, MAT, 0.03, 100.0)
            assert "frequency" in params
            assert "day_count" in params
            assert "cpi_lag_months" in params


# ═══════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════


class TestSerialization:
    def test_to_dict(self):
        idx = get_inflation_index("CPI_US")
        d = idx.to_dict()
        assert d["name"] == "CPI_US"
        assert d["interpolation"] == "linear"
        assert d["deflation_floor"] is True

    def test_frozen(self):
        idx = get_inflation_index("CPI_US")
        with pytest.raises(Exception):
            idx.name = "WRONG"
