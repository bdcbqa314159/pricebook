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


class TestAMM:
    def test_uniswap_v2(self):
        from pricebook.crypto.amm import uniswap_v2_price
        r = uniswap_v2_price(1_000_000, 1_000_000, 10_000)
        assert r.amount_out > 0
        assert r.amount_out < 10_000
        assert r.price_impact_pct > 0

    def test_curve_stableswap(self):
        from pricebook.crypto.amm import curve_stableswap
        r = curve_stableswap([1_000_000, 1_000_000], 10_000, A=100)
        assert r.amount_out > 9_900
        assert r.price_impact_pct < 1

    def test_lp_return(self):
        from pricebook.crypto.amm import lp_return_v2
        r = lp_return_v2(500, 500, 2.0, fee_income=50)
        assert r.impermanent_loss < 0


class TestImpermanentLoss:
    def test_no_change(self):
        from pricebook.crypto.impermanent_loss import impermanent_loss
        r = impermanent_loss(1.0)
        assert r.il_pct == pytest.approx(0, abs=0.01)

    def test_2x(self):
        from pricebook.crypto.impermanent_loss import impermanent_loss
        r = impermanent_loss(2.0)
        assert r.il_pct == pytest.approx(-5.72, abs=0.1)

    def test_table(self):
        from pricebook.crypto.impermanent_loss import il_table
        assert len(il_table()) > 10


class TestDeFiRates:
    def test_aave(self):
        from pricebook.crypto.defi_rates import aave_rate
        r = aave_rate(0.5)
        assert r.supply_apy < r.borrow_apy

    def test_aave_kink(self):
        from pricebook.crypto.defi_rates import aave_rate
        assert aave_rate(0.95).borrow_apy > aave_rate(0.7).borrow_apy

    def test_liquidation(self):
        from pricebook.crypto.defi_rates import liquidation_threshold
        r = liquidation_threshold(10_000, 7_000, 2_000)
        assert r.health_factor > 1


class TestStaking:
    def test_eth_yield(self):
        from pricebook.crypto.staking import eth_staking_yield
        r = eth_staking_yield()
        assert r.net_yield < r.total_yield

    def test_liquid_staking(self):
        from pricebook.crypto.staking import liquid_staking_premium
        r = liquid_staking_premium(0.99, 1.0, 0.04)
        assert r.premium_pct < 0

    def test_slashing(self):
        from pricebook.crypto.staking import slashing_risk
        assert slashing_risk().expected_loss_pct < 1


class TestBasisArb:
    def test_spot_perp(self):
        from pricebook.crypto.basis_arb import spot_perp_basis
        assert spot_perp_basis(50000, 50050, 0.0001).annualised_yield > 0

    def test_triangular(self):
        from pricebook.crypto.basis_arb import triangular_arb
        r = triangular_arb(50000, 3000, 0.06)
        assert abs(r.profit_pct) < 2

    def test_cross_exchange(self):
        from pricebook.crypto.basis_arb import cross_exchange_arb
        assert cross_exchange_arb(50000, 50100)["gross_spread_bps"] > 0


class TestCryptoRisk:
    def test_var(self):
        from pricebook.crypto.crypto_risk import crypto_var
        rng = np.random.default_rng(42)
        r = crypto_var(rng.normal(0, 0.03, 720).tolist(), interval_hours=1)
        assert r.var_1d > r.var_1h

    def test_tail(self):
        from pricebook.crypto.crypto_risk import tail_risk
        rng = np.random.default_rng(42)
        r = tail_risk(rng.standard_t(3, 1000).tolist())
        assert r.is_heavy_tailed

    def test_exchange_risk(self):
        from pricebook.crypto.crypto_risk import exchange_risk
        r = exchange_risk({"Binance": 80_000, "Coinbase": 15_000, "Kraken": 5_000})
        assert r["concentrated"]
