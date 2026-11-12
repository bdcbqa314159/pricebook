# Tranche 33b — the engine registry dispatches on (product, model)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.40.0`.
**Source:** none external — an infrastructure fix of our own dispatch, pulled by T33's finding (one
product now has two model-pricers). **Pre-registered RED (the coexistence test + bit-identical purity).**
Runs **before T34**; it is not an equity tranche, it is the registry the hybrids will register into.

## 0. The finding, and why now
T33 verified empirically that the engine registry keys on **product type alone**
(`_PRICERS[product_type] = pricer`), so a second `@register(EuropeanEquityOption)` **silently replaces**
T30's Black pricer — a real correctness footgun (Code registered a stub and watched
`price_european_equity_option` vanish from the table). It never bit before because every product had one
engine; the **European equity option is the first with two** (Black T30, Heston T33, local-vol-MC T32),
which is the rule-of-two second consumer that decides the design (§6). The §3 contract already takes the
model — `price(product, model)` — so the model belongs in dispatch.

## 1. The change
- **`_PRICERS` keys on `(product_type, model_type)`.** `@register` declares (or derives from the
  signature) the model type it prices; dispatch in `price(product, model)` looks up
  `(type(product), type(model))`.
- **Every existing registration carries its model type** — a mechanical pass; each current engine has
  exactly one model (`HullWhite`, the LMM, the JY/MM models, the survival/XVA ones), so each becomes
  `(its product, its model)` with **no behaviour change**.
- **`heston_call_price` and the Black equity pricer register distinctly** under
  `(EuropeanEquityOption, HestonModel)` and `(EuropeanEquityOption, <flat-vol/Black model>)` — the two
  that collided, now coexisting.
- **A double registration of the *same* `(product, model)` is an error, not a silent overwrite** — the
  footgun becomes a loud failure at import/registration, the §8 "no debt silenced silently" shape.
- Fix `models/__init__.py`'s missing docstring (imports above it) while here — cosmetic, T33's note.

## 2. The oracle (RULED) — a pure refactor plus the thing that was impossible before
- **Pure refactor, bit-identical (§4).** Every currently-registered product prices **bit-identically**
  through the new `(product, model)` dispatch — the lookup key changed, not an operation. Prove it
  bit-identical (the keys/ops are unchanged, only the dict key widens), not ULP; the whole suite staying
  green is the corroboration, the bit-identity is the proof.
- **The coexistence test — the RED that was impossible before (§4 pre-register).** Register two pricers
  for `EuropeanEquityOption` under **different** models (Black and Heston) and assert **both resolve to
  their own pricer** — the exact thing that silently clobbered before now works. Pre-register it red
  (it cannot pass on the product-only registry).
- **The loud-failure requirement.** Registering the *same* `(product, model)` twice **raises** (asserted);
  an unregistered `(product, model)` pair returns `PricingFailure`/raises a clear error, never a silent
  wrong pricer. **Mutation:** revert the key to product-only (→ the coexistence test fails — the
  clobber returns); drop the double-registration guard (→ the loud-failure test fails). Each must die.
- **No degenerate-fixture risk to assert beyond the above** — the dispatch is exercised by every
  existing tranche's tests, which now all route through the two-tuple key; a failure there would show at
  once.

## 3. Process
Branch `tranche/33b-engine-registry`. `test` (RED: the coexistence test + the double-registration
loud-failure, both failing on the product-only registry) → `feat` (widen the key, update all
registrations) → `docs` → `chore(release): v0.40.0`. **Open the draft PR early.** `source:` none.
`crosscheck: none — a dispatch mechanism, no external library implements our registry`. **Raise an
`Ask:`** only if widening the key surfaces a product that was *relying* on the clobber (it should not —
T33 kept Heston a named function precisely to avoid it), which would be a finding, not a refactor.

## 4. What this tranche is NOT (scope fence)
No new pricing, no new product, no new model — only the dispatch key and the registrations. Not a
plugin/extension system (§6 — no speculative generality); `(product, model)` is exactly the two
arguments §3 already names, no more. The hybrids (T34–T36) register into this cleanly; that is the
point of doing it first.
