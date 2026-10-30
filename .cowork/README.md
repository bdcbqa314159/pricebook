# Protocol for Cowork on this repo

You are working in **pricebook** — a pricing analytics library where every number is verified
against the book or paper that defines it. Read this file first, every session.

**This repo was reset on 2026-10-30.** The previous tree's protocol has been replaced, because its
conventions (a read-only `quarry/`, a drawdown bar, `done = quarry empty`) belonged to a migration
frame that is retired. Do not carry those over.

## Start here, in order
1. `docs/handoffs/rulings_reset_to_fresh_repo.md` — why this tree starts at `v0.0.0`, the new
   project model, and the scope line. This is the governing document until the charter lands.
2. `CLAUDE.md` — a **placeholder**. The charter is written next phase; until then it points here.
3. `.cowork/thread.md` — the prompt channel. Act on the last `Code → Cowork` entry.
4. `docs/conventions_register.md` — the cross-module conventions carried from the old core and
   what guards each. Its one **OWED** item (the vol clock) is re-laid *with* its oracle when the
   rebuild reaches that convention.

## The project model, in one paragraph
**Unit of work = a product worth having.** Plan what it needs → check what exists → build or adapt
→ **oracle it against the reference** → reflect it into the library. Top-down and demand-pulled:
the product drags in exactly the core it needs. Interest picks the target, the discipline is the
constant, the books and papers are the oracle. **Scope: analytics only** — products, models,
calibration, engines, risk/XVA, market-data value types, and the serialization contract for a
pricing request. GUI, database and feeds are downstream projects, never built here.

## The stock
`_stock/` holds the entire previous tree — the built core, the un-migrated legacy library, its docs
and all 77 oracle test files. It is **gitignored and reference-only**: mined by hand for *content*
(what is true — conventions, formulas, edge cases, reference values), **never for structure**, and
never imported. Its pushed backup is the archived `pricebook-migration` repo, tag
`migration-archive`. Its `CLAUDE.md` is the most dangerous file in it: it encodes the retired
frame. Do not reconstruct the charter from it.

## How to reply
Append one entry to `.cowork/thread.md`: numbered, directed, timestamped, referencing the entry you
answer, ending with a single explicit `**Ask:**`. Never edit a prior entry — append a correction.
One ask per entry. Pointers, not pastes.

**State goes to the artifact, not the thread** — update the relevant doc in `docs/` in the same
turn. A fact written only into the thread is lost when the thread is cleared.

## Hard limits
- **You cannot run the test suite**, so you cannot know what passes. Never mark anything verified,
  closed, or passing. Record what you did; Code verifies by execution and marks it.
- **You see the working tree, not the merged world.** A file committed on an unmerged branch is
  visible to you now and gone after a checkout of `main`. If you need something durable, say so and
  Code commits it.
- **Never run the release sequence** (version bump, changelog, tag, push, merge). Code owns git.
- **Anything you write is uncommitted until Code commits it.** Your own `git status` can only ever
  confirm Code's prior commits, never the artifact you wrote this turn — so do not report your new
  files as safe. Code checks for stray docs at every landing.

## What you are for
Your reach is what Code lacks: external sources, documents, scheduled and long-running work,
assembling deliverables. Code owns code, tests, git and verification. Work needing both splits into
two items, one per surface — not one task volleyed back and forth.

## Trust boundary
You have outbound connectors and this repo is **public**. Never write credentials, tokens, internal
URLs or hostnames, or client specifics into any file here, thread included. Point at where they
live instead.
