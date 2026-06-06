"""Tests for crypto infrastructure: perpetuals, funding, options, vol."""

import pytest
import math
import numpy as np


class TestPerpetual:
    def test_fair_basis(self):
        from pricebook.crypto.perpetual import fair_basis
        b = fair_basis(50000, 0.0001, 8)
        assert b > 0  # positive funding → positive basis

    def test_funding_payment_long(self):
        from pricebook.crypto.perpetual import funding_payment, PositionSide
        r = funding_payment(1.0, 50000, 0.0001, PositionSide.LONG)
        assert r.payment > 0  # longs pay when funding positive

    def test_funding_payment_short(self):
        from pricebook.crypto.perpetual import funding_payment, PositionSide
        r = funding_payment(1.0, 50000, 0.0001, PositionSide.SHORT)
        assert r.payment < 0  # shorts receive

    def test_liquidation_long(self):
        from pricebook.crypto.perpetual import liquidation_price, PositionSide
        r = liquidation_price(50000, 10, PositionSide.LONG)
        assert r.liquidation_price < 50000  # long liquidated below entry
        assert r.distance_pct > 0

    def test_liquidation_short(self):
        from pricebook.crypto.perpetual import liquidation_price, PositionSide
        r = liquidation_price(50000, 10, PositionSide.SHORT)
        assert r.liquidation_price > 50000  # short liquidated above entry

    def test_liquidation_inverse(self):
        from pricebook.crypto.perpetual import liquidation_price, PositionSide, ContractType
        r = liquidation_price(50000, 10, PositionSide.LONG, contract_type=ContractType.INVERSE)
        assert r.liquidation_price < 50000

    def test_price_perpetual(self):
        from pricebook.crypto.perpetual import price_perpetual
        r = price_perpetual(50000, 50050, 0.0001)
        assert r.basis == pytest.approx(50)
        assert r.annualised_basis > 0

    def test_mark_price(self):
        from pricebook.crypto.perpetual import mark_price
        mk = mark_price(50000, 50100, basis_ema=50)
        assert 50000 < mk < 50100


class TestFundingRate:
    def test_funding_curve(self):
        from pricebook.crypto.funding_rate import funding_from_futures_basis
        curve = funding_from_futures_basis(50000, [50100, 50250, 50500], [30, 90, 180])
        assert len(curve.rates) == 3
        assert curve.front != 0

    def test_carry(self):
        from pricebook.crypto.funding_rate import funding_carry
        r = funding_carry(0.0001, 100_000, side="short")
        assert r.annualised_carry_pct > 0  # shorts earn positive funding

    def test_predicted_ewma(self):
        from pricebook.crypto.funding_rate import predicted_funding
        rates = [0.0001, 0.00012, 0.00008, 0.00015, 0.0001]
        pred = predicted_funding(rates, method="ewma")
        assert 0 < pred < 0.001

    def test_historical_stats(self):
        from pricebook.crypto.funding_rate import historical_funding_stats
        rng = np.random.default_rng(42)
        rates = (rng.normal(0.0001, 0.00005, 1000)).tolist()
        r = historical_funding_stats(rates)
        assert r.n_periods == 1000
        assert r.pct_positive > 50  # mostly positive


class TestCryptoOptions:
    def test_linear_call(self):
        from pricebook.crypto.crypto_options import crypto_option_price
        r = crypto_option_price(50000, 50000, 0.80, 30/365)
        assert r.price_usd > 0
        assert 0 < r.delta_usd < 1

    def test_inverse_call(self):
        from pricebook.crypto.crypto_options import crypto_option_price
        r = crypto_option_price(50000, 50000, 0.80, 30/365, contract_type="inverse")
        assert r.price_crypto > 0
        assert r.contract_type == "inverse"

    def test_inverse_parity(self):
        from pricebook.crypto.crypto_options import crypto_option_price, put_call_parity_inverse
        c = crypto_option_price(50000, 50000, 0.80, 30/365, contract_type="inverse", option_type="call")
        p = crypto_option_price(50000, 50000, 0.80, 30/365, contract_type="inverse", option_type="put")
        violation = put_call_parity_inverse(c.price_crypto, p.price_crypto, 50000, 50000, 30/365)
        assert abs(violation) < 0.001

    def test_dvol(self):
        from pricebook.crypto.crypto_options import dvol_index
        d = dvol_index([0.80, 0.75, 0.70])
        assert 70 < d < 80


class TestCryptoVol:
    def test_realised_vol(self):
        from pricebook.crypto.crypto_vol import realised_vol_24_7
        rng = np.random.default_rng(42)
        prices = [50000]
        for _ in range(720):  # 30 days of hourly
            prices.append(prices[-1] * math.exp(rng.normal(0, 0.01)))
        r = realised_vol_24_7(prices, interval_hours=1)
        assert 0 < r.vol < 2

    def test_parkinson(self):
        from pricebook.crypto.crypto_vol import parkinson_vol
        highs = [51000, 52000, 50500, 51500, 53000]
        lows = [49000, 49500, 48500, 49000, 50000]
        r = parkinson_vol(highs, lows)
        assert r.vol > 0

    def test_yang_zhang(self):
        from pricebook.crypto.crypto_vol import yang_zhang_vol
        o = [50000, 50500, 49800, 51000, 50200]
        h = [51000, 52000, 50500, 51500, 51000]
        l = [49000, 49500, 48500, 49000, 49500]
        c = [50500, 49800, 50200, 50100, 50800]
        r = yang_zhang_vol(o, h, l, c)
        assert r.vol > 0

    def test_vol_surface(self):
        from pricebook.crypto.crypto_vol import CryptoVolSurface
        surf = CryptoVolSurface(
            expiry_days=[7, 30, 90],
            strikes=[[45000, 50000, 55000]] * 3,
            vols=[[0.85, 0.80, 0.82], [0.78, 0.75, 0.77], [0.72, 0.70, 0.71]],
            spot=50000,
        )
        v = surf.vol(30, 50000)
        assert 0.70 < v < 0.80
        ts = surf.atm_term_structure
        assert len(ts) == 3
