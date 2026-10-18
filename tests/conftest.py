"""Guard rails that only apply while a mutation test runs.

An ordinary ``pytest`` run is untouched: every guard below is gated on
``MUTANT_UNDER_TEST`` holding the name of a mutant, which only mutmut sets, and which it
sets to a fixed set of phase names (``stats``, ``fail``, ...) outside the per-mutant runs.

Two failure modes of the mutated code reach mutmut as a signal on the *process* rather
than as a failing test, and both were mislabelled in the first full run of this suite:

* A mutant that never returns is killed by mutmut's wall-clock limit and recorded as
  ``timeout``. The limit is ``(estimated_test_time + timeout_constant) * timeout_multiplier``
  with the estimate taken from the *instrumented* stats run, which is up to 40x slower than
  a real one -- so the limit lands anywhere between 15 s and 25 minutes, and a single
  non-terminating mutant can hold a worker for that long. Nothing asserts anything in that
  case; the verdict is the harness giving up.
* A mutant that allocates without bound is killed by the kernel. The first full run had two
  of these (a loop counter mutated to step backwards, so every iteration appended another
  simulated strain): each grew to ~57 GB of anonymous memory before the OOM killer took it,
  and it took an unrelated process on the machine with it. mutmut maps the resulting
  ``SIGKILL`` to the same verdict name as ``SIGSEGV`` -- "segfault" -- which reads as a
  memory-safety bug in a C extension rather than what it is.

Both guards below turn those into an ordinary failing test, which mutmut records as
``killed``: a verdict attributable to a test rather than to the harness, reached in seconds
instead of minutes. Every firing is appended to a log file so the count stays visible --
a kill that only happened because the code stopped returning is weaker evidence than one an
assertion made, and folding it silently into the kill count is exactly the score inflation
these guards exist to remove.

A third failure mode needs more than a guard rail, because it reaches no test at all: a worker
forked out of a driver that has already run the suite inherits mutexes held by threads that no
longer exist, and blocks forever on the first one it needs. Neither budget above can fire, since
the block is inside a lock acquisition rather than in bytecode. Disabling tqdm's monitor thread
below removes one such lock but was measured not to be sufficient -- the same mutants still hung
with every Eigen, OpenMP and BLAS thread count pinned to one -- so the general fix is to give the
worker a process that inherited nothing. ``install_if_mutmut_driver`` does that; see
``tests/mutmut_fork_safety.py``.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import pytest
import tqdm

from tests.mutmut_fork_safety import ENV_BASELINE_VMSIZE, install_if_mutmut_driver

try:
    import resource
except ImportError:  # pragma: no cover - Unix-only, and this file is imported by every run
    resource = None  # type: ignore[assignment]

#: Whether this platform can interrupt a running test on a timer at all. `SIGALRM`, `setitimer` and
#: `ITIMER_REAL` are absent on Windows, and this module is imported by every ordinary pytest run --
#: mutation or not -- so neither guard below may be reached through a bare attribute access.
_TIMER_AVAILABLE = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")

# Whether pytest is running under mutmut at all, in any of its phases. Distinct from
# `_active_mutant()` below, which is only true in a worker: this one is also true in the parent,
# and the parent is where the fork happens.
_UNDER_MUTMUT = "MUTANT_UNDER_TEST" in os.environ

if _UNDER_MUTMUT:
    # Stop tqdm starting its monitor thread. This is the difference between a mutation run that
    # finishes and one that stalls: mutmut forks a worker per mutant, and a fork leaves the child
    # with one thread but with every lock in whatever state it was in. tqdm's monitor thread takes
    # tqdm's write lock periodically, so a worker forked at the wrong instant inherits that lock
    # held by a thread that no longer exists, and blocks forever the first time the code under test
    # opens a progress bar -- with no CPU, and past the reach of the per-test budget below, because
    # the block is inside a lock acquisition rather than in bytecode. Observed as workers alive for
    # 25 minutes at 0% CPU with two threads, their captured output ending at a bar stuck on 0%.
    # `monitor_interval = 0` is tqdm's documented way to disable that thread; the bars still work.
    # Set at import so the *parent* never starts one either, since that is the thread that is
    # inherited.
    tqdm.tqdm.monitor_interval = 0

# Replace each forked mutant worker with a fresh interpreter, so no worker runs on inherited
# locks. Called unconditionally rather than under `_UNDER_MUTMUT` above: it does its own, stricter
# check (mutmut actually imported, and this process in a driver phase rather than a worker), and it
# raises if mutmut's internals have moved. A silent no-op here would bring the deadlocks back
# disguised as verdicts, which is the one outcome worth crashing to avoid.
install_if_mutmut_driver()

# mutmut reuses MUTANT_UNDER_TEST for its own phases. None of these run a mutant, so none of
# them should be constrained: the stats phase legitimately runs the whole suite instrumented,
# and the clean and forced-fail runs decide whether the harness itself is working.
_MUTMUT_PHASES = frozenset({"", "stats", "fail", "mutant_generation", "list_all_tests"})

#: Per-test wall-clock budget while a mutant is under test, in seconds. It has to sit between
#: two measured bounds: above the slowest test in the suite (0.9 s uninstrumented, measured
#: with ``--durations``) with enough headroom for a loaded machine, and below mutmut's own
#: smallest process limit (15 s, which is what ``(0 + 1.0) * 15`` comes to for a mutant whose
#: tests are instantaneous) so the failing test is reached before the process is killed.
_TEST_TIMEOUT_S = float(os.environ.get("GWMOCK_MUTATION_TEST_TIMEOUT", "10"))

#: How much *more* address space a mutant's test process may claim than it starts with, in GiB.
#: Relative rather than absolute because a mutmut worker is forked from a parent that has already
#: run the whole suite once, so it inherits a large mapping before it runs a line: the suite peaks
#: at 8.4 GiB of address space (VmPeak, measured over a full run) against 2.2 GiB of resident
#: memory, and an absolute ceiling anywhere near the first number would fail legitimate tests.
#: 8 GiB of growth is several times what any single test needs and a small fraction of the 57 GB
#: the runaway mutants reached before the kernel stepped in.
_MEMORY_HEADROOM_GB = float(os.environ.get("GWMOCK_MUTATION_MEMORY_HEADROOM_GB", "8"))

#: Where firings are recorded. Resolved once, at import, against the working directory mutmut
#: runs pytest in (``mutants/``) -- so the log lands beside the run it describes and *not* in the
#: per-test directory the fixture below moves each test into, which is deleted after a few runs.
_GUARD_LOG = Path(os.environ.get("GWMOCK_MUTATION_GUARD_LOG", "mutation-guard.log")).resolve()

_memory_limit_applied = False


class _NonTermination(BaseException):
    """Raised in the running test when the per-test budget expires.

    Deliberately not an ``Exception``: the code under test catches broad exception groups in
    several places (``retry_with_backoff`` catches ``Exception``, the run-input digest catches
    ``OSError``), and a probe those swallow measures nothing -- the first version of this
    derived from ``TimeoutError``, was caught by the digest's ``except OSError``, and the test
    sailed on into the next unbounded loop.
    """


def _active_mutant() -> str | None:
    """Return the mutant under test, or None when this is not a per-mutant run."""
    name = os.environ.get("MUTANT_UNDER_TEST", "")
    return None if name in _MUTMUT_PHASES else name


def _record(kind: str, mutant: str, nodeid: str) -> None:
    """Append one guard firing to the log, without letting logging break the run."""
    try:
        with _GUARD_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{kind}\t{mutant}\t{nodeid}\n")
    except OSError:  # pragma: no cover - a full or read-only disk must not mask the verdict
        pass


def _inherited_baseline() -> int:
    """The address space this worker would have started with had it been forked, in bytes.

    The ceiling below is *relative*: headroom on top of what the process already has mapped. That
    was written when a worker was a ``fork`` of a driver that had already run the whole suite, so
    "already mapped" meant the suite's peak and the headroom was genuine slack on top. Workers now
    ``exec`` a fresh interpreter instead (see ``tests/mutmut_fork_safety.py``), which maps almost
    nothing before it runs a line, so the same arithmetic would yield a fraction of the intended
    ceiling and abort legitimate tests inside XLA rather than failing them.

    The driver therefore hands its own figure across ``exec``, and the ceiling stays anchored to
    what it was always anchored to. 0 when the variable is absent -- an ordinary ``fork``, or a
    platform without ``/proc`` -- which leaves the original behaviour untouched.
    """
    try:
        return int(os.environ.get(ENV_BASELINE_VMSIZE, "0"))
    except ValueError:  # pragma: no cover - only a corrupted environment gets here
        return 0


def _address_space_in_use() -> int:
    """The process's current virtual size in bytes, or 0 if /proc is not readable."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):  # pragma: no cover - not Linux, or no procfs
        return 0
    return 0


def _apply_memory_limit() -> None:
    global _memory_limit_applied  # noqa: PLW0603
    if resource is None or _memory_limit_applied or _MEMORY_HEADROOM_GB <= 0 or _active_mutant() is None:
        return
    in_use = max(_address_space_in_use(), _inherited_baseline())
    if not in_use:  # pragma: no cover - without a baseline any ceiling would be a guess
        return
    limit = in_use + int(_MEMORY_HEADROOM_GB * 1024**3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    _memory_limit_applied = True


def _on_timeout(signum: int, frame: Any) -> None:
    raise _NonTermination(f"no result within {_TEST_TIMEOUT_S:g}s while testing {_active_mutant()}")


def pytest_configure(config: Any) -> None:
    """Cap the address space of a mutant's test process.

    In a hook rather than at import, because mutmut forks its workers from a parent that has
    already imported this file: module-level code would run once in the parent and never in a
    worker, and the parent must keep its own limits.
    """
    _apply_memory_limit()


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    """Arm the per-test budget. Covers setup and teardown as well as the call."""
    if not _TIMER_AVAILABLE or _TEST_TIMEOUT_S <= 0 or _active_mutant() is None:
        return
    _apply_memory_limit()
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.setitimer(signal.ITIMER_REAL, _TEST_TIMEOUT_S)


def pytest_runtest_logfinish(nodeid: str, location: Any) -> None:
    """Disarm the budget so it cannot fire between tests."""
    if not _TIMER_AVAILABLE or _TEST_TIMEOUT_S <= 0 or _active_mutant() is None:
        return
    signal.setitimer(signal.ITIMER_REAL, 0)


def pytest_exception_interact(node: Any, call: Any, report: Any) -> None:
    """Record the kills that a guard produced rather than an assertion."""
    mutant = _active_mutant()
    if mutant is None or call.excinfo is None:
        return
    if call.excinfo.errisinstance(_NonTermination):
        _record("non-termination", mutant, getattr(node, "nodeid", "?"))
    elif call.excinfo.errisinstance(MemoryError):
        _record("memory-exhaustion", mutant, getattr(node, "nodeid", "?"))


@pytest.fixture(autouse=True)
def _isolated_working_directory(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """Give every test its own working directory.

    Several things the code under test writes are resolved against the working directory rather
    than against a path a test can choose: ``gwmock simulate`` puts its checkpoints in
    ``./.gwmock_checkpoints``, ``download_file`` defaults its output directory to ``Path.cwd()``,
    and a few simulators default their output file names to bare names. Run from the repository
    root, those tests leave files behind; run *concurrently from one directory*, they collide.

    That is not hypothetical. mutmut runs up to one worker per core and every worker's pytest runs
    with the same working directory, so a checkpoint written by one worker made
    ``test_simulate_command_runs_adapter_orchestration`` fail in another with
    ``ForeignCheckpointError: written by a different configuration``. The verdict recorded for the
    mutant under test was then "killed" by an unrelated worker's leftovers, and the same collision
    took the whole run's clean-test check down when a stale checkpoint survived it.

    ``tests/e2e`` is left alone: those tests drive whole example configurations and are not part of
    the unit run this guards, so they keep whatever working directory the caller chose.
    """
    if "e2e" in request.path.parts:
        yield
        return
    previous = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    try:
        yield
    finally:
        os.chdir(previous)


def _installed(package: str) -> bool:
    """Whether ``package`` imports; a broken installation raises rather than reading as absent."""
    try:
        __import__(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        return False
    return True


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip the tests marked ``requires_h5py``, ``requires_gwpy`` or ``requires_lalsuite`` where that package
    is not installed.

    Python 3.15 is the case: h5py, GWpy and lalsuite publish no wheel for it yet, and the dev group and
    gwmock-signal install them only below it. Everything else still runs there.
    """
    for marker, module in (("h5py", "h5py"), ("gwpy", "gwpy"), ("lalsuite", "lalsimulation")):
        if _installed(module):
            continue
        skip = pytest.mark.skip(reason=f"{marker} is not installed")
        for item in items:
            if item.get_closest_marker(f"requires_{marker}") is not None:
                item.add_marker(skip)
