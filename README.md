# pricebook

A pricing analytics library where **every number is verified against the book or paper that
defines it.**

## The aim

Products, models, calibration, engines, risk and XVA — built so that each number has a named
reference and an oracle that proves it, not a test that merely re-runs the code.

## How it grows

**Product-first, demand-pulled.** The unit of work is a product worth having: plan what it needs
(this method, this model, these core pieces), check what already exists, build or adapt, **oracle
it against the reference**, and reflect it into the library. A product drags in exactly the core
it requires and nothing it does not. Interest picks the target; the discipline is the constant;
the books and papers are the oracle.

## Scope — analytics only, pluggable not implemented

**In:** products, models, calibration, engines, risk/XVA, and the market-data value types. The
durable public contract is **how a product and a pricing request are represented** — a
serialization format over frozen product data. A pricing server is the first consumer that proves
that representation.

**Out:** GUI, database, broker and market-data feeds. Those are downstream projects that
*consume* the representation; they are never built here.

## State

`v0.0.0` — scaffold only, no pricing code. The foundation tranches are planned before any
library code lands.

## Lineage

This is a clean restart. Its predecessor migrated a large working library into a layered
architecture under an oracle-first discipline, reaching a complete and verified core across rates,
credit and equity. The architecture, the verification standard and the conventions are kept; the
completionist obligation to port every legacy module is not — see
[`docs/handoffs/rulings_reset_to_fresh_repo.md`](docs/handoffs/rulings_reset_to_fresh_repo.md).

The previous tree is archived at **`pricebook-migration`** (tag `migration-archive`) and shelved
locally in a gitignored `_stock/` as reference material — mined by hand for *content*, never for
*structure*, and never imported.
