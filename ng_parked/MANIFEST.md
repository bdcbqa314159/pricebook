# ng_parked — parked at v0.54.0 (Topic 0, `slice/ng-parking`)

The entire pre-topic-method ng tree, moved here wholesale so the clean rebuild does not
inherit its structure. **This is a CONTENT source only** (conventions, formulas, edge
cases, oracle reference values) — its *organisation carries no authority* (CLAUDE.md:
mine the quarry/ng_parked for content, never for structure). Modules are re-derived clean
by the topic that owns them, re-based against the oracle noted below.

- **Parked-at version:** v0.54.0 (last version before the rebuild; see CHANGELOG).
- **54 modules** + their tests (`ng_parked/tests_ng/`). Tests are refactored, not
  preserved — ng gets new tests; the old expected values are mined where useful.
- Two quarry modules were already retired before parking and stay ticked:
  `core/numerical_config`, `fixed_income/fixed_leg`.

| parked module | rebuilt by | re-base oracle |
|---|---|---|
| `calibration/discount_curve.py` | Topic 1/3 — curve construction (L3) | inputs reprice to par (self-consistency) + closed-form deposit DFs < … |
| `calibration/hull_white.py` | Topic 1/3 — curve construction (L3) | round-trip sigma recovery + calibrated model reprices the quote < 1e-10 |
| `calibration/survival_curve.py` | Topic 1/3 — curve construction (L3) | each input CDS reprices to zero par spread (self-consistency) < 1e-10 |
| `engine/cds.py` | Topic 5 — engines (L4) | par CDS reprices to zero; matches L1 cds_pv (cds-product slice) |
| `engine/commodity_option.py` | Topic 5 — engines (L4) | put-call parity vs the commodity forward; Black recompute (commodity-… |
| `engine/discounting.py` | Topic 5 — engines (L4) | PV = sum(cf * df) closed form < 1e-12 (S00 single cashflow; S04 bond) |
| `engine/equity_option.py` | Topic 5 — engines (L4) | put-call parity vs the equity forward; Black recompute (equity-option… |
| `engine/fra.py` | Topic 5 — engines (L4) | par (K = forward) -> 0; closed-form off-par; seasoned uses the fixing… |
| `engine/fx_forward.py` | Topic 5 — engines (L4) | par FX forward reprices to zero; PV matches CIP (fx-forward slice) |
| `engine/fx_option.py` | Topic 5 — engines (L4) | put-call parity vs the FX forward; Black recompute (fx-option slice) |
| `engine/inflation.py` | Topic 5 — engines (L4) | par ZCIS reprices to zero (inflation-zcis slice) |
| `engine/ois.py` | Topic 5 — engines (L4) | par OIS -> 0; OIS == vanilla IRS (single-curve) |
| `engine/spot_option.py` | Topic 5 — engines (L4) | put-call parity vs the forward; Black recompute (equity + commodity s… |
| `engine/swap.py` | Topic 5 — engines (L4) | par swap reprices to zero NPV; float leg telescopes (S06) |
| `engine/swaption.py` | Topic 5 — engines (L4) | put-call parity + ATM symmetry + sigma->0 intrinsic (S08) |
| `engine/swaption_mc.py` | Topic 5 — engines (L4) | MC converges to the Jamshidian analytic within a few standard errors … |
| `foundation/black.py` | Topic 0 — foundation (L0) | exercised by the FX (GK) and equity (BS) option oracles |
| `foundation/calendar.py` | Topic 0 — foundation (L0) | hand-computed adjustments over weekends/holidays (exact dates) |
| `foundation/cashflow.py` | Topic 0 — foundation (L0) | N/A (pure value type; exercised by the Slice 0 closed-form oracle) |
| `foundation/distributions.py` | Topic 0 — foundation (L0) | exercised by the Hull-White ZCB-option oracle (S07) |
| `foundation/money.py` | Topic 0 — foundation (L0) | N/A (pure value type; exercised by the Slice 0 closed-form oracle) |
| `foundation/numerical_config.py` | Topic 0 — foundation (L0) | to_dict/from_dict round-trip + schema versioning; fd_bump via the DV0… |
| `foundation/results.py` | Topic 0 — foundation (L0) | N/A (result value types; exercised by the Slice 0 oracle) |
| `foundation/schedule.py` | Topic 0 — foundation (L0) | hand-computed coupon schedules incl. EOM + short front stub (exact) |
| `foundation/solvers.py` | Topic 0 — foundation (L0) | exact roots + known minima (quadratic bowl, Rosenbrock); drives calib… |
| `foundation/time.py` | Topic 0 — foundation (L0) | published ISDA/ICMA year-fraction vectors, exact < 1e-12 |
| `market/discount_curve.py` | Topic 1 — yield curves (L1) | closed-form log-linear df + parallel/pillar `bumped` (greek FD) < 1e-12 |
| `market/keys.py` | Topic 1 — yield curves (L1) | namespacing test (FX "EUR" != equity "EUR"); exercised by every asset… |
| `market/snapshot.py` | Topic 1 — yield curves (L1) | df(t) = exp(-r t) closed form (drives the Slice 0 PV oracle) |
| `market/survival_curve.py` | Topic 1 — yield curves (L1) | closed-form log-linear Q + CDS RPV01/protection/par-spread leg math |
| `models/credit_model.py` | Topic 4 — models (L3) | par CDS reprices to zero through the engine (CDS-as-product slice) |
| `models/discounting_model.py` | Topic 4 — models (L3) | engine-model binding test + unchanged PVs (A1) |
| `models/hull_white.py` | Topic 4 — models (L3) | curve refit P(0,S) + ZCB-option put-call parity + sigma->0 intrinsic;… |
| `products/cds.py` | Topic 2/5/6 — products (L2) | par CDS reprices to zero through the CDSEngine (cds-product slice) |
| `products/commodity_option.py` | Topic 2/5/6 — products (L2) | put-call parity ties to the commodity forward (commodity-option slice) |
| `products/deposit.py` | Topic 2/5/6 — products (L2) | forward par -> 0; spot par -> principal; closed-form off-par; |
| `products/equity_option.py` | Topic 2/5/6 — products (L2) | put-call parity ties to the equity forward (equity-option slice) |
| `products/fixed_cashflow.py` | Topic 2/5/6 — products (L2) | Slice 0 closed form (priced by DiscountingEngine); to_dict/from_dict … |
| `products/fixed_rate_bond.py` | Topic 2/5/6 — products (L2) | closed-form discounted-cashflow PV < 1e-12 (S04); to_dict/from_dict r… |
| `products/fra.py` | Topic 2/5/6 — products (L2) | par FRA (K = forward) reprices to zero; closed-form off-par PV; |
| `products/fx_forward.py` | Topic 2/5/6 — products (L2) | par FX forward reprices to zero (fx-forward slice) |
| `products/fx_option.py` | Topic 2/5/6 — products (L2) | put-call parity ties to the FX forward; Black recompute (fx-option sl… |
| `products/inflation.py` | Topic 2/5/6 — products (L2) | par ZCIS reprices to zero (inflation-zcis slice); to_dict/from_dict r… |
| `products/leg.py` | Topic 2/5/6 — products (L2) | exercised by the bond (S04) and swap (S06) closed-form oracles |
| `products/ois.py` | Topic 2/5/6 — products (L2) | par OIS -> 0; OIS == vanilla IRS (single-curve); |
| `products/swap.py` | Topic 2/5/6 — products (L2) | par swap reprices to zero NPV (S06) |
| `products/swaption.py` | Topic 2/5/6 — products (L2) | put-call parity + ATM symmetry + sigma->0 intrinsic (S08) |
| `risk/exposure.py` | Topic 7 — risk/XVA (L5) | sigma=0 deterministic exposure (exact) + discounted EE == co-terminal… |
| `risk/greeks.py` | Topic 7 — risk/XVA (L5) | dv01/credit01 analytic; spot_delta/vol_vega match FX & equity analytics |
| `risk/priceable.py` | Topic 7 — risk/XVA (L5) | generic dv01 matches analytic across products (L5 greeks slice) |
| `risk/saccr.py` | Topic 7 — risk/XVA (L5) | 10y ATM $100mm IRS EAD ~ 5.5% notional; RC/multiplier-floor limits; |
| `risk/xva.py` | Topic 7 — risk/XVA (L5) | unit-exposure CVA == protection leg; FVA/KVA/MVA == rate·RPV01; BCVA … |
| `shell/booking.py` | Topic 8 — portfolio/lifecycle (L6) | realized (benefit table) + mark reconcile to total economics (A3); di… |
| `shell/xva_report.py` | Topic 8 — portfolio/lifecycle (L6) | single-trade report == standalone CVA/DVA/BCVA/FVA/KVA/MVA; mirror he… |
