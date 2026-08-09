# Artifact #06 — Release & Versioning Policy (Ratified)

**Status:** Ratified. A companion policy, not one of the five core artifacts.
Governs how the *new tree* versions and how releases are recorded.

---

## What exists today (the quarry's scheme)

- `__version__ = "1.240.0"` in `python/pricebook/__init__.py` (setuptools reads it).
- **693 release entries**, `v0.505.0` (2026-05-12) → `v1.240.0` (2026-07-05): ~11 bumps/day.
- Format: `## vX.Y.0 — DATE — **headline**`, one entry per slice.
- **It is a slice counter, not semver.** MINOR counts slices; PATCH is always `.0`;
  MAJOR flipped 0→1 once. `RELEASE_NOTES.md` is 872KB and grows without bound.

The *discipline* is good — every change logged, with provenance. The *number* carries no
meaning, and the file is unbounded. This policy keeps the discipline and fixes both.

---

## Principle: the version should mean something

For the redesign, the version number tracks **migration progress and API stability**,
not a raw slice count.

### Version line for the new tree — reset to 0.x, climb to 1.0 at quarry-empty
```
0.0.x   Slice 0 landed — walking skeleton proven end-to-end
0.y.z   migration in progress — foundation upward, quarry emptying
1.0.0   THE QUARRY IS EMPTY — every entry migrated or archived; the spine is complete
≥1.0    true semver from here (see below)
```
So `0.x` literally reads "still migrating"; `1.0.0` is the definition-of-done of the
whole redesign. The version becomes a progress bar, and it aligns with the DESIGN.md
gates (G1…G5) — e.g. gate completion can mark `0.<gate>.0` milestones.

### Semver semantics, from 1.0 onward
- **MAJOR** — a breaking change to the public API or a wire/serialisation schema bump
  that isn't back-compatible (ties to the schema-version work in the debt ledger).
- **MINOR** — new capability (an instrument, model, engine, asset class) added
  back-compatibly.
- **PATCH** — bug fix / numerical correction, no API change.

"Public API" = the clean facade from the spine (`pb.price`, `pb.book`, `pb.risk`) plus
the ratified vocabulary types. Internal refactors are PATCH.

---

## Release notes: freeze the quarry history, start a fresh changelog

- **Freeze `RELEASE_NOTES.md`** as the quarry's historical record. It is not deleted (the
  quarry is read-only) and not appended to. It documents the old tree, full stop.
- **Start `CHANGELOG.md`** for the new tree, in *Keep a Changelog* format: grouped
  `Added / Changed / Fixed / Deprecated / Removed` sections per version, newest on top.
  Readable, bounded, and standard.
- **Keep the per-slice discipline:** every landed slice = one version bump + one changelog
  entry. But entries are grouped and human-scannable, not 693 flat lines.
- **Bound the file:** keep the current major line detailed; roll older majors into
  `docs/changelog-archive/` so `CHANGELOG.md` never becomes another 872KB wall.

---

## Tie-in with the migration policy (#05)

A slice's version bump is part of its definition of done, alongside the oracle:

```
… ORACLE green → PROVENANCE recorded → CHANGELOG entry + version bump → MARK landed
```

The changelog entry carries the same provenance the ledger requires (what changed, the
oracle, the quarry path). One discipline, recorded once.

Single source of truth for the number stays `pricebook.__version__`; the build reads it
dynamically (as today). A one-line check can assert `__version__` matches the top
`CHANGELOG.md` entry so they never drift.

---

## Ratified decisions (2026-07, Bernardo)

1. **Version line: reset to 0.x → 1.0 at quarry-empty.** ✅ The new tree starts at
   `0.0.x` (Slice 0), climbs through migration, and reaches **`1.0.0` exactly when the
   quarry is empty**. The version is a migration progress bar; true semver from 1.0 on.
   `pricebook.__version__` remains the single source of truth.
2. **Freeze `RELEASE_NOTES.md`, start a fresh `CHANGELOG.md`.** ✅ The 872KB file becomes
   the read-only quarry record (never appended to, never deleted). The new tree gets a
   clean, bounded `CHANGELOG.md` in Keep-a-Changelog format; per-slice bump discipline
   continues there, grouped and scannable.

Defaults adopted: semver semantics from 1.0 (MAJOR = API/schema break, MINOR = new
capability, PATCH = fix); Keep-a-Changelog format; one version bump + one changelog entry
per landed slice, carrying the same provenance the ledger requires; a CI check asserting
`__version__` matches the top `CHANGELOG.md` entry; older majors rolled into
`docs/changelog-archive/` to keep the file bounded.

## Action items for the build (Claude Code)
- [ ] Set `pricebook.__version__ = "0.0.0"` on the new tree's first commit; bump to
      `0.1.0` (or `0.0.1`) when Slice 0 lands green.
- [ ] Create `CHANGELOG.md` (Keep-a-Changelog); leave `RELEASE_NOTES.md` untouched.
- [ ] Add the `__version__` ↔ changelog drift check to CI.
