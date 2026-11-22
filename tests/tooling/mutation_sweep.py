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

import contextlib
import hashlib
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: What the sweep runs to decide a verdict. `-x` stops at the first failure (a verdict needs one
#: failing test, not all of them), and the cache provider is disabled so a sweep leaves no state
#: behind to influence the next run.
#: The verdict run: `-x` stops at the first failure, because a verdict needs one failing test and
#: nothing is gained by finding the rest.
_PYTEST = ("-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider")
#: The **counting** run, which must NOT stop early. `-x` and a killer count are incompatible by
#: construction: with `-x` pytest reports exactly one `FAILED` line however many tests would have
#: failed, so every kill would read as a single-killer. That is a false-positive machine — the
#: mirror of the false clean sheet this file was built to prevent — and it is why counting is a
#: separate mode rather than always-on. The cost is real (a counting run does not short-circuit),
#: which is the other reason it is opt-in. *(Caught by this file's own known-answer test while
#: building F-7.3's fix, which is the argument for having them.)*
_PYTEST_COUNTING = ("-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider")

#: pytest's exit codes. `0` is "everything passed" and `1` is "tests ran and one failed" — the only
#: code that means *killed*. `2` (interrupted), `3` (internal error), `4` (usage error) and `5` (no
#: tests collected) all mean the run did not deliver a verdict, and counting them as kills is exactly
#: how re-audit #4 reported a clean sheet from 21 runs that had every one errored.
_ALL_PASSED = 0
#: pytest's own `FAILED <test id>` summary line, which `-q` prints. Anchored at the line start so a
#: test whose *name* contains the word cannot be mistaken for a summary line.
_FAILED_LINE = re.compile(r"^FAILED (\S+)", re.MULTILINE)
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
    #: The tests that failed, per mutation label — so §9's **standing single-killer re-run** reads
    #: its counts from the instrument and inherits the verified baseline and the `finally` restore.
    #: Re-audit #7's F-7.3: that obligation was standing but uninstrumented, so it was hand-rolled
    #: beside the sweep five times, and on the fifth a timeout killed the loop between mutating a
    #: source and restoring it — the next run then measured against a corrupted baseline. Nothing
    #: shipped, but it is exactly the failure T52 committed this file to prevent, arriving through
    #: the one obligation T52 left out. An obligation that is standing but uninstrumented will be
    #: re-implemented until it fails.
    killers: Mapping[str, tuple[str, ...]] | None = None

    def single_killers(self) -> tuple[str, ...]:
        """Labels whose mutant died to **exactly one** test — §9's single-point-of-failure oracles.

        A later edit can delete that one test and silently reopen the hole with the suite still
        green, which is the project's signature defect turned on its own test suite. Empty is the
        good answer and is meaningful only beside a full applied count, like every count here.

        **Raises when the sweep did not count**, rather than returning `()`. An empty tuple from a
        verdict-only run would read as "no single-killer oracles" when the truth is "nobody looked"
        — the can't-fail shape §9 forbids, and the same reading error as a `0` survivor count from a
        run where nothing applied.
        """
        if self.killers is None:
            raise ValueError(
                "this sweep did not count killers, so it cannot say whether any oracle is a "
                "single point of failure — re-run with count_killers=True. Returning an empty "
                "tuple here would report 'none found' where the truth is 'nobody looked'"
            )
        return tuple(
            label
            for label, failing in self.killers.items()
            if len(failing) == 1
        )

    def killer_counts(self) -> tuple[tuple[str, int], ...]:
        """`(label, number of failing tests)` for every mutation that produced a verdict.

        The measurement the single-killer table is filled from, so a re-audit records what it
        measured rather than retyping what it remembers. Raises when the sweep did not count, for
        the reason `single_killers` gives.
        """
        if self.killers is None:
            raise ValueError(
                "this sweep did not count killers — re-run with count_killers=True"
            )
        return tuple((label, len(failing)) for label, failing in self.killers.items())

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
        single = self.single_killers() if self.killers is not None else ()
        if single:
            text += (
                f"\nSINGLE-KILLER ({len(single)}) — one test each, a later deletion reopens the "
                "hole silently: " + ", ".join(single)
            )
        return text


def _failing_tests(stdout: str) -> tuple[str, ...]:
    """The test ids pytest reported as failed, from its `FAILED <id>` summary lines.

    Deduplicated and sorted, so a count is a count of distinct tests rather than of report lines —
    a parametrised test failing at several parameters is **one** killer, which is the honest reading
    for §9's purpose: deleting that one test function removes all of them at once.

    `-q` still prints the `FAILED` summary, so this needs no extra pytest flag and cannot change the
    verdict: the exit code decides killed-or-not, and this only says *by whom*.
    """
    return tuple(sorted({m.group(1) for m in _FAILED_LINE.finditer(stdout)}))


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


#: Files mutated right now, mapped to their original text, so a SIGNAL HANDLER can put them back.
#: A mapping rather than one slot because losing an entry is the whole failure being guarded against.
_PENDING: dict[Path, str] = {}


def _restore_pending() -> None:
    """Put every live mutation back and empty the registry. Safe to call twice."""
    for path, original in list(_PENDING.items()):
        try:
            path.write_text(original, encoding="utf-8")
        finally:
            _PENDING.pop(path, None)


@contextlib.contextmanager
def _pending_restores(path: Path, original: str) -> Iterator[None]:
    """Hold `path`'s original text in the registry for as long as it is mutated.

    The `finally` restore below is the normal path; the registry exists so the *abnormal* path — a
    signal — has something to restore from."""
    _PENDING[path] = original
    try:
        yield
    finally:
        try:
            path.write_text(original, encoding="utf-8")
        finally:
            _PENDING.pop(path, None)


@contextlib.contextmanager
def _signal_safe_restore() -> Iterator[None]:
    """Restore any live mutation on SIGTERM/SIGINT, then re-raise so the caller still dies.

    **`finally` does not survive a SIGTERM, and that cost this project twice in one sitting**
    (re-audit #8, F-8.2): a harness timeout and a kill each left a mutation in `src/` — once
    `engine/commodity_asian.py`, once `foundation/sobol.py`'s Gray code, the core of the
    quasi-random spine. A long unattended sweep is exactly when a kill arrives, so the instrument
    must survive one.

    The handler restores, reinstates the previous disposition, and re-raises the same signal — so
    the process still terminates with the signal it was sent, rather than the instrument quietly
    swallowing a kill. If the signals cannot be installed (not the main thread), the `finally`
    restore remains and this is a no-op rather than an error."""
    previous: dict[int, signal._HANDLER] = {}

    def handler(signum: int, frame: object) -> None:
        _restore_pending()
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        signal.raise_signal(signum)

    for number in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[number] = signal.signal(number, handler)
        except (ValueError, OSError):
            continue
    try:
        yield
    finally:
        for number, disposition in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, disposition)


def _assert_tree_restored(root: Path, digest: str) -> None:
    """Refuse to return a verdict from a sweep that left the tree mutated.

    The counterpart to the applied-count and ERRORED guards: those catch a run that did not test
    what it claimed, and this catches a run that did not put back what it changed. A `0 survivors`
    read off a tree that is still mutated is worse than a wrong verdict, because the next commit
    ships it."""
    current = _tree_digest(root)
    assert current == digest, (
        f"the sweep left mutated source behind: the tree digest under {root} changed from "
        f"{digest[:12]}... to {current[:12]}.... Run `git status` and restore before trusting any "
        f"verdict or staging anything — a stranded mutation plus a non-path-by-path `git add` "
        f"commits it with a green suite"
    )


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
    count_killers: bool = False,
) -> SweepResult:
    """Apply each mutation in turn, run `tests`, and score the verdict.

    `root` is both the working directory for the test run and the tree whose bytecode caches are
    invalidated — caches are cleared *before* each run (so the mutated source is what compiles) and
    *after* (so a mutant's bytecode cannot outlive the mutant).

    The source is restored three ways, because losing a mutation into the tree is worse than any
    wrong verdict: a `finally` per mutation (the normal path), a **signal-handled restore** for
    SIGTERM/SIGINT (which `finally` does not survive — F-8.2, which bit twice in one sitting), and a
    **tree-digest assertion** at the end that refuses to return a verdict from a tree left mutated.
    """
    if not mutations:
        raise ValueError(
            "a sweep needs at least one mutation: an empty set reports 'applied 0/0, survived 0' "
            "and would be trivially clean, which is the can't-fail shape §9 forbids"
        )

    baseline_digest = _tree_digest(root)
    _verified_baseline(tuple(tests), str(root), baseline_digest, timeout)

    flags = _PYTEST_COUNTING if count_killers else _PYTEST
    killed = survived = 0
    killers: dict[str, tuple[str, ...]] | None = {} if count_killers else None
    survivors: list[str] = []
    not_applied: list[str] = []
    errored: list[str] = []

    with _signal_safe_restore():
        for mutation in mutations:
            original = _apply(mutation)
            if original is None:
                not_applied.append(mutation.label)
                continue
            with _pending_restores(mutation.path, original):
                _invalidate_bytecode(root)
                completed = subprocess.run(
                    [sys.executable, *flags, *tests],
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
                    if killers is not None:
                        killers[mutation.label] = _failing_tests(completed.stdout)
                else:
                    # Not a verdict: an import error, a usage error or an empty collection.
                    # Counting this as a kill is re-audit #4's false clean sheet.
                    errored.append(
                        f"{mutation.label} (pytest exit {completed.returncode})"
                    )
                _invalidate_bytecode(root)

    _assert_tree_restored(root, baseline_digest)
    return SweepResult(
        total=len(mutations),
        applied=killed + survived + len(errored),
        killed=killed,
        survived=survived,
        survivors=tuple(survivors),
        not_applied=tuple(not_applied),
        errored=tuple(errored),
        killers=killers,
    )
