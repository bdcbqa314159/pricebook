"""Basel regulatory capital framework (Basel II / II.5 / III / IV).

Unified regulatory module covering:
- VaR/ES engine with backtesting
- FRTB SA and IMA
- IRC (rating migration MC)
- Credit RWA (SA-CR, F-IRB, A-IRB)
- SA-CCR, CVA, CCP
- Capital framework (output floor, leverage, G-SIB)
- Stress testing, IRRBB, liquidity
- Operational risk
- Basel 2/2.5 legacy

    from pricebook.regulatory import ratings, var_es
    from pricebook.regulatory.ratings import resolve_pd, resolve_rating
    from pricebook.regulatory.var_es import quick_var, backtest_var
"""
