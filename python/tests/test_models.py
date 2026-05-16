"""Tests for pricebook.models — model abstraction layer."""

from datetime import date
from dateutil.relativedelta import relativedelta

import pytest

from pricebook.curves.bootstrap import bootstrap
from pricebook.models.black76 import OptionType
from pricebook.swaption import Swaption, SwaptionType
from pricebook.capfloor import CapFloor
from pricebook.models.models import (
    Black76Model, BachelierModel, SABRModel, SABRParams,
    HullWhiteModel, BSModel, HestonModel, HestonParams,
    MCEquityModel, IROptionModel, EquityOptionModel,
    price_european,
)


REF = date(2024, 7, 15)


def _curve():
    deposits = [(REF + relativedelta(months=3), 0.05)]
    swaps = [(REF + relativedelta(years=y), r)
             for y, r in [(1, .048), (2, .046), (5, .042), (10, .04)]]
    return bootstrap(REF, deposits, swaps)


def _swaption(stype=SwaptionType.PAYER):
    return Swaption(
        expiry=REF + relativedelta(years=2),
        swap_end=REF + relativedelta(years=7),
        strike=0.04,
        swaption_type=stype,
        notional=10_000_000,
    )


def _capfloor(otype=OptionType.CALL):
    return CapFloor(
        start=REF,
        end=REF + relativedelta(years=5),
        strike=0.04,
        option_type=otype,
        notional=10_000_000,
    )


# ═══════════════════════════════════════════════════════════════
# Protocol checks
# ═══════════════════════════════════════════════════════════════

class TestProtocols:
    def test_black76_is_ir_model(self):
        assert isinstance(Black76Model(0.20), IROptionModel)

    def test_bachelier_is_ir_model(self):
        assert isinstance(BachelierModel(0.005), IROptionModel)

    def test_sabr_is_ir_model(self):
        assert isinstance(SABRModel(SABRParams(0.03, 0.5, -0.2, 0.3)), IROptionModel)

    def test_bs_is_equity_model(self):
        assert isinstance(BSModel(0.25), EquityOptionModel)

    def test_heston_is_equity_model(self):
        assert isinstance(HestonModel(HestonParams(0.04, 2, 0.04, 0.3, -0.7)), EquityOptionModel)

    def test_bs_not_ir_model(self):
        assert not isinstance(BSModel(0.25), IROptionModel)

    def test_black76_not_equity_model(self):
        assert not isinstance(Black76Model(0.20), EquityOptionModel)


# ═══════════════════════════════════════════════════════════════
# Swaption: .price(model, curve)
# ═══════════════════════════════════════════════════════════════

class TestSwaptionBlack76:
    def test_payer_positive(self):
        curve = _curve()
        sw = _swaption(SwaptionType.PAYER)
        assert sw.price(Black76Model(vol=0.20), curve) > 0

    def test_receiver_positive(self):
        curve = _curve()
        sw = _swaption(SwaptionType.RECEIVER)
        assert sw.price(Black76Model(vol=0.20), curve) > 0

    def test_higher_vol_higher_price(self):
        curve = _curve()
        sw = _swaption()
        p1 = sw.price(Black76Model(vol=0.15), curve)
        p2 = sw.price(Black76Model(vol=0.25), curve)
        assert p2 > p1

    def test_positive_price(self):
        curve = _curve()
        sw = _swaption()
        assert sw.price(Black76Model(vol=0.20), curve) > 0


class TestSwaptionBachelier:
    def test_positive_price(self):
        curve = _curve()
        sw = _swaption()
        assert sw.price(BachelierModel(vol_normal=0.005), curve) > 0

    def test_higher_vol_higher_price(self):
        curve = _curve()
        sw = _swaption()
        p1 = sw.price(BachelierModel(vol_normal=0.003), curve)
        p2 = sw.price(BachelierModel(vol_normal=0.008), curve)
        assert p2 > p1

    def test_different_from_black76(self):
        curve = _curve()
        sw = _swaption()
        pb = sw.price(Black76Model(vol=0.20), curve)
        pn = sw.price(BachelierModel(vol_normal=0.005), curve)
        assert abs(pb - pn) > 1  # different models → different prices


class TestSwaptionSABR:
    def test_positive_price(self):
        curve = _curve()
        sw = _swaption()
        params = SABRParams(alpha=0.03, beta=0.5, rho=-0.3, nu=0.4)
        assert sw.price(SABRModel(params), curve) > 0

    def test_beta1_close_to_black76(self):
        """SABR with beta=1, nu=0 degenerates to Black-76."""
        curve = _curve()
        sw = _swaption()
        vol = 0.20
        pb = sw.price(Black76Model(vol=vol), curve)
        # SABR with beta=1, nu=0: alpha = vol, lognormal
        ps = sw.price(SABRModel(SABRParams(alpha=vol, beta=1.0, rho=0.0, nu=0.0)), curve)
        # Should be very close (not exact due to Hagan correction terms)
        assert abs(pb - ps) / pb < 0.01

    def test_from_atm(self):
        curve = _curve()
        sw = _swaption()
        model = SABRModel.from_atm(0.20)
        assert sw.price(model, curve) > 0


class TestSwaptionHullWhite:
    def test_positive_price(self):
        from pricebook.models.hull_white import HullWhite
        curve = _curve()
        hw = HullWhite(a=0.03, sigma=0.01, curve=curve)
        sw = _swaption()
        model = HullWhiteModel(hw)
        assert sw.price(model, curve) > 0

    def test_higher_sigma_higher_price(self):
        from pricebook.models.hull_white import HullWhite
        curve = _curve()
        sw = _swaption()
        p1 = sw.price(HullWhiteModel(HullWhite(a=0.03, sigma=0.005, curve=curve)), curve)
        p2 = sw.price(HullWhiteModel(HullWhite(a=0.03, sigma=0.015, curve=curve)), curve)
        assert p2 > p1


class TestSwaptionGuards:
    def test_incompatible_model_raises(self):
        curve = _curve()
        sw = _swaption()
        with pytest.raises(TypeError, match="does not implement"):
            sw.price(BSModel(vol=0.20), curve)

    def test_incompatible_model_message(self):
        curve = _curve()
        sw = _swaption()
        with pytest.raises(TypeError, match="BSModel"):
            sw.price(BSModel(vol=0.20), curve)


# ═══════════════════════════════════════════════════════════════
# CapFloor: .price(model, curve)
# ═══════════════════════════════════════════════════════════════

class TestCapFloorBlack76:
    def test_cap_positive(self):
        curve = _curve()
        cap = _capfloor(OptionType.CALL)
        assert cap.price(Black76Model(vol=0.20), curve) > 0

    def test_floor_positive(self):
        curve = _curve()
        floor = _capfloor(OptionType.PUT)
        assert floor.price(Black76Model(vol=0.20), curve) > 0

    def test_higher_vol_higher_price(self):
        curve = _curve()
        cap = _capfloor()
        p1 = cap.price(Black76Model(vol=0.15), curve)
        p2 = cap.price(Black76Model(vol=0.25), curve)
        assert p2 > p1

    def test_guard_incompatible(self):
        curve = _curve()
        cap = _capfloor()
        with pytest.raises(TypeError, match="does not implement"):
            cap.price(BSModel(vol=0.20), curve)


class TestCapFloorBachelier:
    def test_cap_positive(self):
        curve = _curve()
        cap = _capfloor()
        assert cap.price(BachelierModel(vol_normal=0.005), curve) > 0

    def test_different_from_black76(self):
        curve = _curve()
        cap = _capfloor()
        pb = cap.price(Black76Model(vol=0.20), curve)
        pn = cap.price(BachelierModel(vol_normal=0.005), curve)
        assert abs(pb - pn) > 1


# ═══════════════════════════════════════════════════════════════
# Equity models
# ═══════════════════════════════════════════════════════════════

class TestBSModel:
    def test_call_positive(self):
        model = BSModel(vol=0.25)
        p = model.price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        assert p > 0

    def test_put_positive(self):
        model = BSModel(vol=0.25)
        p = model.price_european(100, 100, 0.05, 1.0, OptionType.PUT)
        assert p > 0

    def test_put_call_parity(self):
        import math
        model = BSModel(vol=0.25)
        S, K, r, T = 100, 100, 0.05, 1.0
        call = model.price_european(S, K, r, T, OptionType.CALL)
        put = model.price_european(S, K, r, T, OptionType.PUT)
        # C - P = S*e^{-qT} - K*e^{-rT}
        assert abs((call - put) - (S - K * math.exp(-r * T))) < 0.01

    def test_higher_vol_higher_price(self):
        p1 = BSModel(vol=0.15).price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        p2 = BSModel(vol=0.30).price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        assert p2 > p1

    def test_matches_equity_option_price(self):
        from pricebook.equity_option import equity_option_price
        model = BSModel(vol=0.25)
        p1 = model.price_european(100, 105, 0.05, 1.0, OptionType.CALL, 0.02)
        p2 = equity_option_price(100, 105, 0.05, 0.25, 1.0, OptionType.CALL, 0.02)
        assert abs(p1 - p2) < 1e-10


class TestHestonModel:
    def test_call_positive(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        model = HestonModel(params)
        p = model.price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        assert p > 0

    def test_matches_heston_price(self):
        from pricebook.heston import heston_price
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        model = HestonModel(params)
        p1 = model.price_european(100, 105, 0.05, 1.0, OptionType.CALL, 0.02)
        p2 = heston_price(100, 105, 0.05, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                          OptionType.CALL, 0.02)
        assert abs(p1 - p2) < 1e-10

    def test_different_from_bs(self):
        """Heston with vol-of-vol should differ from BS."""
        import math
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        bs_vol = math.sqrt(0.04)  # sqrt(v0)
        p_bs = BSModel(vol=bs_vol).price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        p_heston = HestonModel(params).price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        # With vol-of-vol > 0 and rho != 0, should differ
        assert abs(p_bs - p_heston) > 0.01


class TestPriceEuropean:
    def test_convenience_function(self):
        model = BSModel(vol=0.25)
        p1 = price_european(model, 100, 100, 0.05, 1.0)
        p2 = model.price_european(100, 100, 0.05, 1.0, OptionType.CALL)
        assert abs(p1 - p2) < 1e-10


# ═══════════════════════════════════════════════════════════════
# Params
# ═══════════════════════════════════════════════════════════════

class TestParams:
    def test_sabr_params_frozen(self):
        p = SABRParams(0.03, 0.5, -0.3, 0.4)
        with pytest.raises(AttributeError):
            p.alpha = 0.05

    def test_heston_params_fields(self):
        p = HestonParams(v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
        assert p.v0 == 0.04
        assert p.rho == -0.7

    def test_repr(self):
        assert "0.20" in repr(Black76Model(vol=0.20))
        assert "Bachelier" in repr(BachelierModel(vol_normal=0.005))
        assert "SABR" in repr(SABRModel(SABRParams(0.03, 0.5, -0.3, 0.4)))

    def test_hw_repr(self):
        from pricebook.models.hull_white import HullWhite
        curve = _curve()
        hw = HullWhite(a=0.03, sigma=0.01, curve=curve)
        assert "HullWhite" in repr(HullWhiteModel(hw))
