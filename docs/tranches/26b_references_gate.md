# Tranche 26b — the `verify.py references` gate: register status, self-policed

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.32.0`.
**Source:** none external — a governance gate over `docs/references.md`, pulled by the drift Code found
at T26's handoff (thread 0135). **Pre-registered RED (by seeded drift).** A standalone governance
tranche, **run before T27** (the XVA arc resumes after it); it is not part of the XVA arc.

## 0. The finding, and why a gate now
Committing T26, Code cross-checked one word of a `to_go_deeper` route (`CREPEY-2026 (worked-from)`)
against `references.md` and found **three stale statuses** (thread 0135): `CREPEY-2026` and
`CAPINSKI-ZASTAWNIAK` still `in-hand` though T25 landed and three modules cite them; `JAECKEL-2002`
**self-contradictory** — header `in-hand`, body `worked-from`, 18 modules citing it (drift predating
this arc). All three corrected by hand. This is §9's re-audit target (3) — *register accuracy* — but
found **by accident, one word at a time**, between the periodic re-audits that are 11 tranches apart.

§7 already floats this gate: *"A future `verify.py references` gate can assert every `source:` resolves
to an `in-hand`/`worked-from` key here — proposed, not built, until the first drift makes it earn its
place."* **This is that drift** — three of them — so the gate earns its place. Cheap, because the
`provenance` gate already walks every `src/` header for its `source:` line; this reuses that walk.

## 1. The gate (`verify.py references`)
Add `references` to the gate list (§9). It parses `references.md`'s header lines — the mechanical,
parseable form `### \`KEY\` — \`status\`` — for `(key, status)` pairs, and the `source:` keys already
extracted from `src/` module headers by `provenance`, and asserts:

1. **Resolution (§7's original check).** Every `source:` key cited in `src/` resolves to a `KEY` in
   `references.md` whose status is `in-hand` or `worked-from` — never `referenced-by`/`sought` (no
   citation by inference), never absent.
2. **Built ⇒ `worked-from`.** Every key cited by **at least one** `src/` module has status
   `worked-from` (the register's own vocabulary: *"held and built/oracled against"*). This is the check
   that catches the three drifts — a cited key sitting at `in-hand` is the exact stale state found.
3. **No orphan `worked-from`.** Every `worked-from` key has **at least one** citing `src/` module —
   `worked-from` claims something was built against it, so a zero-citation `worked-from` is as wrong as
   an uncited `in-hand` would be right.

**Domain fence.** The gate keys on the **header status field** and the `src/` citations — not the
free-prose body. The JAECKEL header/body contradiction is caught because check (2) forces the header to
`worked-from` (18 citers), which the body already said; the gate does not parse narrative. The author
keeps the body consistent; the gate guarantees the header matches reality.

## 2. The oracle (RULED) — RED by seeded drift
Pre-register a test that **seeds each of the three drift shapes** and asserts the gate rejects it, then
passes on the corrected tree:

- a cited key flipped to `in-hand` (shape 1 — CREPEY/C-Z) → gate **fails**;
- a cited key flipped to `referenced-by`/removed (shape of check 1) → gate **fails**;
- a `worked-from` key with its citations removed (check 3) → gate **fails**;
- the current tree (statuses corrected by Code) → gate **passes**.

This is the mutation-as-proof discipline (§4) applied to a gate: the gate must demonstrably **fail**
the states it exists to forbid, or it is unverified surface. The three real drifts are the natural
fixtures — the gate retro-actively catches exactly what was found by hand.

## 3. Process
Branch `tranche/26b-references-gate`. `test` (RED: the seeded-drift fixtures the gate must reject) →
`feat` (the gate) → `docs` → `chore(release): v0.32.0`. **Updates two docs in the same tranche:**
charter **§7** — the `references` bullet moves from *"proposed, not built"* to built, naming this
tranche and the three drifts as what earned it; and `docs/references.md`'s own parenthetical to match.
Add `references` to the `verify.py` list in §9. **Open as a DRAFT PR early** (§10 step 3, per T26's CI
note — right after the first push). `source:` none. `crosscheck: none — a governance gate over our own
register, no external library implements it`.

## 4. Scope fence
Only status/citation **consistency**. The gate does not: validate that a held PDF actually exists on
disk (it is gitignored and absent in CI — the same reason the stock-path check is local-only, §9); parse
or fact-check the prose body; or verify a formula against its source (that is §4's per-tranche job, not
a gate). It asserts the register's *status field* cannot silently diverge from what `src/` actually
cites — nothing more.
