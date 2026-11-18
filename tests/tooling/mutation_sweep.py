"""The mutation sweep, as a committed instrument (tranche 52).

§9 makes a whole-tree mutation sweep a standing obligation — every arc close and every eleven
tranches — and until this tranche it was an ad-hoc script rewritten from scratch each re-audit: the
one part of the method with no provenance, and the one that re-introduced the same tooling bugs
twice. This module is that script, versioned and reviewed, with the instrument lessons the charter
has already paid for encoded as **code** rather than re-derived from memory.

**It is the engine, not the audit.** Mutation *selection* stays the re-audit's judgement — where a
transcription error would land, which blind spot to probe (the T43 uniform rescale, the T39 reduction
blindness), oldest-first scoping. A tool that chose its own mutations would sweep where it is easy
rather than where the failure lives, which is §9's can't-fail gate wearing an instrument's clothes.
So this applies and scores a mutation set it is *given*; it never defines the universe of mutations,
and it is not a gate (nothing runs it on a push).

**The three clauses it exists to make unforgettable**, each a regression test in
`test_mutation_sweep.py`:

* **Bytecode invalidation per mutant** (§9, T46). A mutation written inside one filesystem mtime tick
  leaves the `.pyc`'s `(mtime, size)` unchanged, so CPython declines to recompile, the run tests the
  *original* source, and a confident verdict is printed for a mutant that never ran. At T46 that
  produced a false *"killed"* — the dangerous direction, which hides a hole exactly as a false green
  does — and flipped three verdicts once caches were cleared.
* **applied / killed / survived, with the applied count load-bearing** (§9, re-audit #4). A `0`
  survivors is meaningless on its own: a run where every mutation errored reports a clean zero. Here
  `is_clean` is false whenever a mutation failed to apply, so "survivors: 0" from a run that never
  ran is not expressible.
* **A failing run is not automatically a kill** — and this one was found by the instrument's own
  known-answer test, which is the argument for having built it. Re-audit #4's false clean came from
  21 runs that *all errored on the command line*; the applied-count guard does not catch that shape,
  because those mutations **did** reach the source. By return code alone an import error, a usage
  error and a collection failure are indistinguishable from a test failure, so every errored run
  reads as *killed* — the dangerous direction again. A kill is therefore specifically pytest's exit
  code **1** (tests ran and one failed); 2-5 are `errored`, counted apart and never as a kill.
* **The baseline is verified before anything is mutated.** If the chosen tests do not pass on the
  unmutated tree, every mutant "fails" for a reason that has nothing to do with the mutation, and the
  whole sweep is a false clean sheet. The verdict is memoised on a **content digest** of the tree
  rather than re-run per sweep — which is sound rather than a shortcut, because a sweep restores
  every file it touches, so consecutive sweeps face a byte-identical tree; change anything and the
  digest changes and the baseline re-runs. (Deliberately *not* keyed on mtimes: that is the very
  signal the T46 defect exploited.)
* **Positional addressing** (re-audit #5, bitten **twice**). Two textually identical load-bearing
  lines in different functions — `yoy.py:136/:152`, `credit_legs.py:140/:250` — are two real sites. A
  string-replacement sweep refuses them as non-unique and silently leaves them uncounted, so a
  mutation names a **line**, and the fragment is matched within that line only.

A `Mutation` carries both the line and the fragment on purpose: the line makes identical sites
distinct, and the fragment keeps the intent readable and catches a stale line number instead of
silently mutating whatever has moved there.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: What the sweep runs to decide a verdict. `-x` stops at the first failure (a verdict needs one
#: failing test, not all of them), and the cache provider is disabled so a sweep leaves no state
#: behind to influence the next run.
_PYTEST = ("-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider")

#: pytest's exit codes. `0` is "everything passed" and `1` is "tests ran and one failed" — the only
#: code that means *killed*. `2` (interrupted), `3` (internal error), `4` (usage error) and `5` (no
#: tests collected) all mean the run did not deliver a verdict, and counting them as kills is exactly
#: how re-audit #4 reported a clean sheet from 21 runs that had every one errored.
_ALL_PASSED = 0
_TESTS_FAILED = 1


@dataclass(frozen=True)
class Mutation:
    """One planted change: the fragment `old` becomes `new`, on line `line` of `path` and nowhere
    else.

    `line` is 1-based, matching what an editor and a traceback both show.
    """

    path: Path
    line: int
    old: str
    new: str
    label: str

    def __post_init__(self) -> None:
        if self.old == self.new:
            raise ValueError(
                f"the mutation {self.label!r} changes nothing: a no-op would be applied, run, and "
                "scored as a mutant while testing the original, which is a guaranteed false survivor"
            )
        if self.line < 1:
            raise ValueError(f"line numbers are 1-based, got {self.line} for {self.label!r}")


@dataclass(frozen=True)
class SweepResult:
    """What a sweep found — and it cannot be read as a survivor count alone.

    `total` is how many mutations were *asked for*, `applied` how many actually reached the source.
    The two differ exactly when a fragment was absent or a line had moved, and that gap is the
    re-audit #4 defect: a run where nothing applied reports zero survivors and looks clean.
    """

    total: int
    applied: int
    killed: int
    survived: int
    survivors: tuple[str, ...]
    not_applied: tuple[str, ...]
    errored: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """No survivors, every mutation applied, and no run errored — all three, because each alone
        lies. Zero survivors from a sweep where nothing applied, or where every run errored, is the
        re-audit #4 false clean sheet, and it is not expressible here."""
        return (
            self.applied == self.total
            and self.survived == 0
            and not self.errored
            and self.killed == self.total
        )

    def summary(self) -> str:
        """The one line a human reads, which always carries all three counts (§9)."""
        text = f"applied {self.applied}/{self.total}  killed {self.killed}  survived {self.survived}"
        if self.errored:
            text += f"\nERRORED ({len(self.errored)}) — NOT verdicts: " + ", ".join(self.errored)
        if self.not_applied:
            text += f"\nNOT APPLIED ({len(self.not_applied)}): " + ", ".join(self.not_applied)
        if self.survivors:
            text += f"\nSURVIVORS ({len(self.survivors)}): " + ", ".join(self.survivors)
        return text


def _tree_digest(root: Path) -> str:
    """A content hash of every `.py` file under `root` — the baseline's cache key.

    Content, never mtimes: a stale-mtime cache is exactly the T46 defect this instrument exists to
    prevent, and keying its own memo on mtimes would be the same mistake one level up.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _verified_baseline(tests: tuple[str, ...], root: str, digest: str, timeout: float) -> None:
    """Run the chosen tests unmutated and refuse to proceed unless they pass.

    Memoised on the tree's content digest, so repeated sweeps over an unchanged tree pay for this
    once. `digest` is unused in the body on purpose — it is the cache key.
    """
    _invalidate_bytecode(Path(root))
    completed = subprocess.run(
        [sys.executable, *_PYTEST, *tests],
        cwd=root, capture_output=True, text=True, timeout=timeout,
    )
    if completed.returncode != _ALL_PASSED:
        raise ValueError(
            f"the chosen tests do not pass on the UNMUTATED tree (pytest exit "
            f"{completed.returncode}), so every mutant would 'fail' for a reason that is not the "
            f"mutation and the sweep would report a false clean sheet:\n{completed.stdout[-2000:]}"
        )


def _invalidate_bytecode(root: Path) -> None:
    """Remove every `__pycache__` beneath `root` — the T46 clause, as the one line that enforces it."""
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _apply(mutation: Mutation) -> str | None:
    """Mutate one line in place, returning the original text — or `None` if it did not apply.

    Addressed by **position**: the fragment is sought on `mutation.line` and nowhere else, so two
    identical lines in different functions are two distinct sites, and a stale line number is a loud
    *not applied* rather than a silent mutation of whatever moved there.
    """
    original = mutation.path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if mutation.line > len(lines):
        return None
    target = lines[mutation.line - 1]
    if mutation.old not in target:
        return None
    lines[mutation.line - 1] = target.replace(mutation.old, mutation.new, 1)
    mutation.path.write_text("".join(lines), encoding="utf-8")
    return original


def sweep(
    mutations: Sequence[Mutation],
    tests: Sequence[str],
    root: Path,
    timeout: float = 3600.0,
) -> SweepResult:
    """Apply each mutation in turn, run `tests`, and score the verdict.

    `root` is both the working directory for the test run and the tree whose bytecode caches are
    invalidated — caches are cleared *before* each run (so the mutated source is what compiles) and
    *after* (so a mutant's bytecode cannot outlive the mutant).

    The source is restored in a `finally`, so an interrupted or failing sweep never leaves the tree
    mutated — which would be a worse outcome than any verdict.
    """
    if not mutations:
        raise ValueError(
            "a sweep needs at least one mutation: an empty set reports 'applied 0/0, survived 0' "
            "and would be trivially clean, which is the can't-fail shape §9 forbids"
        )

    _verified_baseline(tuple(tests), str(root), _tree_digest(root), timeout)

    killed = survived = 0
    survivors: list[str] = []
    not_applied: list[str] = []
    errored: list[str] = []

    for mutation in mutations:
        original = _apply(mutation)
        if original is None:
            not_applied.append(mutation.label)
            continue
        try:
            _invalidate_bytecode(root)
            completed = subprocess.run(
                [sys.executable, *_PYTEST, *tests],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if completed.returncode == _ALL_PASSED:
                survived += 1
                survivors.append(mutation.label)
            elif completed.returncode == _TESTS_FAILED:
                killed += 1
            else:
                # Not a verdict: an import error, a usage error or an empty collection. Counting
                # this as a kill is re-audit #4's false clean sheet.
                errored.append(
                    f"{mutation.label} (pytest exit {completed.returncode})"
                )
        finally:
            mutation.path.write_text(original, encoding="utf-8")
            _invalidate_bytecode(root)

    return SweepResult(
        total=len(mutations),
        applied=killed + survived + len(errored),
        killed=killed,
        survived=survived,
        survivors=tuple(survivors),
        not_applied=tuple(not_applied),
        errored=tuple(errored),
    )
