# `pricebook.desks` — desk contract

Uniform API contract for all trading desks. Every desk module `{asset}_desk.py` MUST expose the components and methods below. Canonical implementation: `trs_desk.py`.

The contract is **documentation, not runtime enforcement** — each desk implements its own concrete classes. The shared shape lets `cross_asset_desk.py` aggregate uniformly across all 12 asset classes.

## Required components (9 per desk)

1. `{Asset}RiskMetrics` — per-position Greeks + sensitivities
2. `{Asset}BookEntry` + `{Asset}Book` — position management
3. `{Asset}CarryDecomposition` — prospective carry breakdown
4. `{Asset}DailyPnL` — daily P&L attribution
5. `{Asset}Dashboard` — morning-meeting summary
6. `{Asset}StressResult` — stress scenario output
7. `{Asset}CapitalResult` — SA-CCR / SIMM regulatory capital
8. `{Asset}HedgeRecommendation` — actionable hedge suggestions
9. `{Asset}Lifecycle` — trade events + alerts

## Required method signatures

```
{asset}_risk_metrics(instrument, curve, ...)            → RiskMetrics
{asset}_carry(instrument, curve, ...)                   → CarryDecomposition
{asset}_daily_pnl(instrument, curve_t0, curve_t1, date_t1, ...) → DailyPnL
{asset}_dashboard(book, date, curve)                    → Dashboard
{asset}_stress_suite(book, curve)                       → list[StressResult]
{asset}_scenario_stress(book, ctx, scenarios=None)      → list[ScenarioResult]
{asset}_capital(instrument, curve, ...)                 → CapitalResult
{asset}_hedge_recommendations(book, curve, **limits)    → list[HedgeRecommendation]
```

## Common fields across all desks

| Type             | Fields |
|------------------|--------|
| `RiskMetrics`    | `pv`, `notional`, `to_dict()` |
| `Book`           | `add(entry)`, `entries`, `__len__()`, `total_notional()`, `aggregate_risk(curve)` |
| `CarryDecomp`    | `net_carry`, `to_dict()` |
| `DailyPnL`       | `date`, `total`, `unexplained`, `to_dict()` |
| `Dashboard`      | `date`, `n_positions`, `total_pv`, `total_notional`, `to_dict()` |
| `StressResult`   | `scenario`, `description`, `total_pnl`, `to_dict()` |
| `CapitalResult`  | `ead`, `rwa`, `capital`, `simm_im`, `to_dict()` |
| `HedgeRec`       | `risk_type`, `current`, `limit`, `breach_pct`, `action`, `to_dict()` |
| `Lifecycle`      | `history` (`list[dict]`) |

## Desks implementing this contract

`repo_desk`, `trs_desk`, `cln_desk`, `bond_desk`, `swap_desk`, `cds_desk`,
`futures_desk`, `fx_desk`, `equity_desk`, `commodity_desk`, `inflation_desk`,
`swaption_trading_desk`
