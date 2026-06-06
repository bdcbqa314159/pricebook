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


class TestCryptoDesk:
    def test_book(self):
        from pricebook.desks.crypto_desk import CryptoBook, CryptoPosition, CryptoInstrument
        book = CryptoBook("test")
        book.add(CryptoPosition("BTC", CryptoInstrument.PERPETUAL, 1.0, 50000, 51000, "Binance", 10))
        book.add(CryptoPosition("ETH", CryptoInstrument.SPOT, 10.0, 3000, 3100, "Coinbase"))
        assert book.total_notional() > 0
        assert book.total_pnl() > 0
        assert len(book.by_symbol()) == 2

    def test_pnl(self):
        from pricebook.desks.crypto_desk import CryptoBook, CryptoPosition, CryptoInstrument, crypto_pnl
        book = CryptoBook()
        book.add(CryptoPosition("BTC", CryptoInstrument.PERPETUAL, 1.0, 50000, 51000, funding_accumulated=50))
        r = crypto_pnl(book)
        assert r.spot_pnl == 1000
        assert r.pnl_btc > 0

    def test_risk_report(self):
        from pricebook.desks.crypto_desk import CryptoBook, CryptoPosition, CryptoInstrument, crypto_risk_report
        book = CryptoBook()
        book.add(CryptoPosition("BTC", CryptoInstrument.PERPETUAL, 1.0, 50000, 51000, "Binance", 10))
        r = crypto_risk_report(book)
        assert r["max_leverage"] == 10


# ═══════════════════════════════════════════════════════════════
# Deep Crypto: CD1–CD12
# ═══════════════════════════════════════════════════════════════

class TestMoveContract:
    def test_move_positive(self):
        from pricebook.crypto.crypto_options import move_contract
        r = move_contract(50000, 0.80, 7/365)
        assert r.price > 0
        assert r.breakeven_move_pct > 0

class TestVarianceSwap:
    def test_var_swap(self):
        from pricebook.crypto.crypto_options import crypto_variance_swap
        r = crypto_variance_swap(0.80, skew_adjustment=0.03)
        assert r.fair_vol > 80

class TestGreeksPnL:
    def test_explain(self):
        from pricebook.crypto.crypto_options import greeks_pnl_explain
        r = greeks_pnl_explain(0.5, 0.001, 200, -50, spot_move=1000, vol_move=0.02, actual_pnl=800)
        assert r.total_pnl == 800
        assert r.delta_pnl == 500

class TestPortfolioGreeks:
    def test_aggregate(self):
        from pricebook.crypto.crypto_options import aggregate_greeks
        pos = [{"delta": 0.5, "gamma": 0.001, "vega": 200, "theta": -50, "quantity": 2}]
        r = aggregate_greeks(pos, 50000)
        assert r.net_delta == 1.0

class TestCrossCorrelation:
    def test_correlation(self):
        from pricebook.crypto.crypto_vol import cross_asset_correlation
        rng = np.random.default_rng(42)
        r = cross_asset_correlation({"BTC": rng.normal(0, 0.03, 100).tolist(),
                                      "ETH": rng.normal(0, 0.04, 100).tolist()})
        assert len(r.assets) == 2

class TestJumpDecomp:
    def test_decomp(self):
        from pricebook.crypto.crypto_vol import jump_decomposition
        rng = np.random.default_rng(42)
        returns = rng.standard_t(3, 500).tolist()
        r = jump_decomposition(returns)
        assert r.jump_fraction > 0
        assert r.n_jumps > 0

class TestMarginAccount:
    def test_cross_margin(self):
        from pricebook.crypto.perpetual import margin_account, MarginMode
        r = margin_account(10000, [{"notional": 50000, "leverage": 5, "unrealised_pnl": 500}])
        assert r.liquidation_risk in ["safe", "warning", "danger"]

class TestPartialLiq:
    def test_partial(self):
        from pricebook.crypto.perpetual import partial_liquidation
        r = partial_liquidation(1.0, 50000, 45000, 10)
        assert r.liquidated_qty > 0

class TestADL:
    def test_ranking(self):
        from pricebook.crypto.perpetual import adl_ranking
        r = adl_ranking([{"unrealised_pnl": 1000, "entry_price": 50000, "quantity": 1, "leverage": 10},
                          {"unrealised_pnl": 500, "entry_price": 50000, "quantity": 1, "leverage": 5}])
        assert r[0].adl_rank == 1

class TestInsuranceFund:
    def test_healthy(self):
        from pricebook.crypto.perpetual import insurance_fund_analysis
        r = insurance_fund_analysis(50_000_000, 10_000_000)
        assert r.depletion_risk in ["healthy", "warning", "depleting"]

class TestTokenomics:
    def test_supply(self):
        from pricebook.crypto.tokenomics import token_supply_schedule
        r = token_supply_schedule(500_000_000, 1_000_000_000, 10_000_000)
        assert r.circulating[-1] > r.circulating[0]

    def test_dcf(self):
        from pricebook.crypto.tokenomics import token_dcf
        r = token_dcf(100_000_000)
        assert r.fair_value_per_token > 0

class TestStablecoin:
    def test_peg_health(self):
        from pricebook.crypto.stablecoin import peg_health
        r = peg_health(0.998)
        assert r.severity == "normal"
        r2 = peg_health(0.95)
        assert r2.severity == "depeg"

    def test_depeg_risk(self):
        from pricebook.crypto.stablecoin import depeg_risk_score, StablecoinType
        r = depeg_risk_score(StablecoinType.FIAT_BACKED, collateral_ratio=1.01)
        assert r.risk_level in ["low", "medium"]

    def test_arb(self):
        from pricebook.crypto.stablecoin import stablecoin_arb
        r = stablecoin_arb(0.995)
        assert r["direction"] == "buy_redeem"

class TestStructuredCrypto:
    def test_dual_investment(self):
        from pricebook.crypto.structured_crypto import dual_investment
        r = dual_investment(50000, 48000, 0.80, 7/365)
        assert r.enhanced_yield_apy > r.base_yield_apy

    def test_shark_fin(self):
        from pricebook.crypto.structured_crypto import crypto_shark_fin
        r = crypto_shark_fin(50000, 0.80, 30/365, 55000)
        assert r.max_return_pct > 0

    def test_snowball(self):
        from pricebook.crypto.structured_crypto import crypto_snowball
        r = crypto_snowball(50000, 0.80, 90/365, n_sims=5000)
        assert 0 < r.price < 2

class TestCascade:
    def test_simulate(self):
        from pricebook.crypto.liquidation_cascade import simulate_cascade, LeveragedPosition
        positions = [LeveragedPosition(100000, 10, 45000, "long") for _ in range(50)]
        r = simulate_cascade(positions, 50000, -0.12)
        assert r.n_liquidations > 0

    def test_risk_score(self):
        from pricebook.crypto.liquidation_cascade import cascade_risk_score, LeveragedPosition
        positions = [LeveragedPosition(100000, 20, 48000, "long") for _ in range(100)]
        r = cascade_risk_score(positions, 50000, 10_000_000, 100_000_000)
        assert r.score > 0

class TestSmartContractRisk:
    def test_contract_score(self):
        from pricebook.crypto.smart_contract_risk import contract_risk_score, AuditStatus
        safe = contract_risk_score(AuditStatus.FORMALLY_VERIFIED, age_days=1000, tvl_usd=1e9,
                                    is_upgradeable=False, multisig_signers=5, multisig_threshold=3,
                                    has_timelock=True, bug_bounty_usd=1_000_000)
        risky = contract_risk_score(AuditStatus.UNAUDITED, age_days=30)
        assert safe.score < risky.score

    def test_composability(self):
        from pricebook.crypto.smart_contract_risk import composability_risk
        r = composability_risk({"Aave": 15, "Curve": 20, "Uniswap": 10},
                                ["Aave", "Curve", "Uniswap"])
        assert r.total_score > 0
        assert r.chain_depth == 3

    def test_oracle_risk(self):
        from pricebook.crypto.smart_contract_risk import oracle_risk
        safe = oracle_risk(n_price_sources=21, update_frequency_seconds=12, uses_twap=True)
        risky = oracle_risk(n_price_sources=1, update_frequency_seconds=7200)
        assert safe.score < risky.score


# ═══════════════════════════════════════════════════════════════
# CD4-CD9 extensions
# ═══════════════════════════════════════════════════════════════

class TestBalancer:
    def test_weighted(self):
        from pricebook.crypto.amm import balancer_weighted_price
        r = balancer_weighted_price([1e6, 1e6], [0.8, 0.2], 10_000)
        assert r.amount_out > 0
        assert r.pool_type == "balancer"

class TestMEV:
    def test_sandwich(self):
        from pricebook.crypto.amm import mev_sandwich_cost
        r = mev_sandwich_cost(100_000, 10_000_000)
        assert r["expected_loss"] > 0
        assert r["safe_trade_size"] > 0

class TestFlashLoan:
    def test_arb(self):
        from pricebook.crypto.defi_rates import flash_loan_arb
        r = flash_loan_arb(1_000_000, 0.005)  # 0.5% arb
        assert r.profitable

class TestYieldRoute:
    def test_route(self):
        from pricebook.crypto.defi_rates import yield_route
        routes = [{"protocol": "Aave", "action": "deposit", "apy": 0.03},
                  {"protocol": "Compound", "action": "borrow", "apy": 0.02},
                  {"protocol": "Curve", "action": "lend", "apy": 0.05}]
        r = yield_route(100_000, routes)
        assert r.net_apy > 0

class TestMultiAssetIL:
    def test_three_token(self):
        from pricebook.crypto.impermanent_loss import multi_asset_il
        r = multi_asset_il([1.5, 0.8, 1.2], [0.5, 0.3, 0.2])
        assert r.il_pct < 0

class TestILHedge:
    def test_perp_hedge(self):
        from pricebook.crypto.impermanent_loss import il_hedge_with_perp
        r = il_hedge_with_perp(-5.0, 100_000)
        assert r.hedge_cost > 0

    def test_options_hedge(self):
        from pricebook.crypto.impermanent_loss import il_hedge_with_options
        r = il_hedge_with_options(-5.0, 100_000)
        assert r.hedge_effectiveness > 0

class TestBinanceFunding:
    def test_formula(self):
        from pricebook.crypto.funding_rate import binance_funding_rate
        rate = binance_funding_rate(0.0005, 0.0001)
        assert rate > 0

class TestCapitalEfficiency:
    def test_basis_trade(self):
        from pricebook.crypto.funding_rate import basis_trade_capital_efficiency
        r = basis_trade_capital_efficiency(50000, 0.0001, leverage=5)
        assert r["return_on_capital_pct"] > 0
