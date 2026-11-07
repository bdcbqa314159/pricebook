# Audit log — the periodic whole-tree re-audit

Charter §9: a full re-audit **every 11 tranches, and at each arc close**, run with independence (the
`analysis/` read-only copy + `AUDIT_FINDINGS.md` pointed at current `main`). It targets what the
per-push gates cannot — oracle power (a full-tree mutation sweep), cross-tranche consistency and
duplication, register accuracy (`to_go_deeper`/`OPEN`/`references`/arc plans), and charter staleness.
Findings are ruled by Cowork and become tranches or ledger entries.

This file is the **counter**, so the trigger is mechanical, not remembered.

| re-audit | at version | tranche | findings | notes |
|---|---|---|---|---|
| — | — | — | — | mechanism established 2026-11-07 (cadence 11, Bernardo). |
| #1 (run; remediation open) | v0.27.0 | T22 | **1 HOLE + 1 MED + 3 LOW** | first re-audit, Jäckel arc close, whole tree T1–T22; run independently of the builder from two fresh contexts (Code built all 22), findings → `docs/audit/reaudit_01_findings.md`, ruled by Cowork. See below. |

## Re-audit #1 — result (2026-11-07, v0.27.0, T1–T22)

Two independent mutation passes (fresh contexts; mutants regenerated, not reused — Code's T15–T22
sweeps did not survive its session) plus the three tree-wide judgment passes. Findings and rulings in
`docs/audit/reaudit_01_findings.md`.

- **Oracle power — Jäckel arc (T15–T22): CLEAN.** 26 mutations, 0 holes; 2 proven-equivalent survivors.
- **Oracle power — inflation arc (T2–T14): 1 HOLE.** F-1.2 [MED] — the linker coupon `ψ` accrual
  survives a drop under all 818 tests (every fixture `ψ=1`, the §4 `τ=1` trap); production code
  correct, oracle degenerate; ~8.4% latent mispricing on a semi-annual linker. → **fix-tranche T23**
  (oracle hardening). *Found only because the sweep was extended past the first pass's domain — the
  §4 rule applied to the audit itself.*
- **Duplication & consistency:** 2 LOW (F-2.1 driftless-forward terminal inlined twice → rule-of-two,
  `to_go_deeper`; F-2.2 discount-from-forwards ×3 but different shapes → no action).
- **Register accuracy:** 1 MED + 1 LOW. F-3.1 — `conventions_register.md` C1–C6 describe the archived
  predecessor tree → **fix-tranche T24** (rebuild). F-3.2 — `to_go_deeper` stale lines → folded into T24.
- **Charter staleness: CLEAN.**

**Remediation queue (read by Code, §10):** **T23** (`docs/tranches/23_reaudit_linker_accrual_oracle.md`,
first) → **T24** (`docs/tranches/24_reaudit_register_rebuild.md`). Entry #1 is **complete** when T24
lands; the next-source direction is settled with Bernardo then.

**Next due:** whichever of {next arc close, 11 tranches since this one (T22)} comes first — so the
counter re-arms at **≈ T33**, or the next arc boundary, whichever is first.
