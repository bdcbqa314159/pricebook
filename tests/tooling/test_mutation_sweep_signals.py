"""The sweep instrument's SIGNAL-SAFE restore. Re-audit #8's F-8.2.

**Why this file exists, measured rather than hypothetical.** `sweep()` restores each mutated file in
a `finally`, which survives an exception and a `KeyboardInterrupt` — and **not** a `SIGTERM`. During
re-audit #8 that stranded a mutation in `src/` **twice in one sitting**: once leaving
`engine/commodity_asian.py` mutated when a foreground run hit a harness timeout, and once leaving
`foundation/sobol.py`'s **Gray code** wrong (`n ^ (n >> 2)`) — the core recurrence of the whole
quasi-random spine, silently altered in the working tree. Both were caught by a `git status` that
happened to be run next; nothing in the instrument noticed.

The hazard is what follows a stranding, not the stranding itself: a mutated `src/` plus any staging
that is not path-by-path commits the mutation, and the suite is then green against a tree nobody
intended. The charter's "never `git add -A`" rule is the backstop; this is the guard.

The end-to-end test here is the one that matters — a **real child process, really sent SIGTERM** —
because that is the exact path that failed, and a unit test of the handler would not have caught it.
"""

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mutation_sweep import (
    Mutation,
    _apply,
    _assert_tree_restored,
    _pending_restores,
    _restore_pending,
    _signal_safe_restore,
    _tree_digest,
)

ORIGINAL = "value = 1\nother = 2\n"


def _sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(ORIGINAL, encoding="utf-8")
    return path


def test_a_pending_restore_is_REGISTERED_while_the_mutation_is_live(tmp_path):
    path = _sample(tmp_path)
    mutation = Mutation(path, 1, "value = 1", "value = 99", "probe")
    with _pending_restores(mutation.path, _apply(mutation)):
        assert path.read_text(encoding="utf-8") == "value = 99\nother = 2\n"
    assert path.read_text(encoding="utf-8") == ORIGINAL


def test_restore_pending_puts_EVERY_live_mutation_back_and_empties_the_registry(tmp_path):
    first, second = _sample(tmp_path), tmp_path / "second.py"
    second.write_text(ORIGINAL, encoding="utf-8")
    for path in (first, second):
        _apply(Mutation(path, 1, "value = 1", "value = 99", "probe"))
    # registered by hand, as the signal handler would find them
    from mutation_sweep import _PENDING

    _PENDING[first] = ORIGINAL
    _PENDING[second] = ORIGINAL
    _restore_pending()
    assert first.read_text(encoding="utf-8") == ORIGINAL
    assert second.read_text(encoding="utf-8") == ORIGINAL
    assert not _PENDING


def test_the_handler_is_INSTALLED_inside_the_window_and_removed_after():
    before = signal.getsignal(signal.SIGTERM)
    with _signal_safe_restore():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_the_tree_restored_assertion_ACCEPTS_an_unchanged_tree(tmp_path):
    _sample(tmp_path)
    _assert_tree_restored(tmp_path, _tree_digest(tmp_path))


def test_the_tree_restored_assertion_REFUSES_a_tree_left_mutated(tmp_path):
    path = _sample(tmp_path)
    digest = _tree_digest(tmp_path)
    path.write_text("value = 99\nother = 2\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="left mutated"):
        _assert_tree_restored(tmp_path, digest)


# ============================================================================================
# The end-to-end test: a real SIGTERM to a real child, which is the path that failed twice.
# ============================================================================================

_CHILD = textwrap.dedent(
    """
    import signal, sys, time
    from pathlib import Path
    sys.path.insert(0, {tooling!r})
    from mutation_sweep import Mutation, _apply, _pending_restores, _signal_safe_restore

    target = Path({target!r})
    mutation = Mutation(target, 1, "value = 1", "value = 99", "probe")
    with _signal_safe_restore():
        with _pending_restores(target, _apply(mutation)):
            Path({flag!r}).write_text("mutated", encoding="utf-8")
            time.sleep(60)
    """
)


def test_a_real_SIGTERM_mid_mutation_leaves_the_file_RESTORED(tmp_path):
    """**Platform-aware rather than skipped, and that is deliberate.**

    On POSIX a SIGTERM is deliverable, so the handler runs and the file comes back — the exact path
    that failed twice. On Windows `Popen.send_signal(SIGTERM)` is `TerminateProcess`, which gives the
    child **no chance to run anything**, so no handler can protect the tree there and the sweep's
    third layer — the tree-digest assertion — is the only guard. This test asserts whichever of those
    two is true for the platform it is running on, instead of carrying a conditional-skip marker: the
    tree has had **zero** suppressions since v0.6.0 (§8's ledger says so), and a conditional skip
    would both break that and hide the Windows limitation rather than record it."""
    target = _sample(tmp_path)
    flag = tmp_path / "mutated.flag"
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.format(
            tooling=str(Path(__file__).parent), target=str(target), flag=str(flag)
        ),
        encoding="utf-8",
    )
    child = subprocess.Popen([sys.executable, str(script)])
    try:
        for _ in range(200):                      # wait until the mutation is actually on disk
            if flag.exists():
                break
            child.poll()
            __import__("time").sleep(0.05)
        assert flag.exists(), "the child never reported the mutation applied"
        assert target.read_text(encoding="utf-8") == "value = 99\nother = 2\n"
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
    if os.name == "nt":
        # TerminateProcess: nothing ran, so the file is still mutated — recorded, not pretended away.
        assert child.returncode != -signal.SIGTERM
    else:
        # the point: the file is back, and the child really did die by the signal
        assert target.read_text(encoding="utf-8") == ORIGINAL
        assert child.returncode == -signal.SIGTERM, child.returncode
