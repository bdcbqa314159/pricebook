# Audit log — the periodic whole-tree re-audit

Charter §9: a full re-audit **every 11 tranches, and at each arc close**, run with independence (the
`analysis/` read-only copy + `AUDIT_FINDINGS.md` pointed at current `main`). It targets what the
per-push gates cannot — oracle power (a full-tree mutation sweep), cross-tranche consistency and
duplication, register accuracy (`to_go_deeper`/`OPEN`/`references`/arc plans), and charter staleness.
Findings are ruled by Cowork and become tranches or ledger entries.

This file is the **counter**, so the trigger is mechanical, not remembered.

| re-audit | at version | tranche | findings | notes |
|---|---|---|---|---|
| — | — | — | — | mechanism established 2026-11-07 (cadence 11, Bernardo). No re-audit run yet. |

**Next due:** the **Jäckel arc close** (≈ T22), whichever of {arc close, 11 tranches since the last}
comes first. The first re-audit covers the whole tree T1–T22 (inflation + BGM/J + the Jäckel arc), as
it is the first application.
