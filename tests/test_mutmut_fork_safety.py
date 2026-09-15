"""Tests for the mutmut fork-safety hook in ``tests/mutmut_fork_safety.py``.

These guard a harness whose failure mode is silent: if the hook stops installing, or
installs into the wrong process, mutant workers go back to deadlocking on locks
inherited across ``fork()`` and mutmut records the result as a verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from tests import mutmut_fork_safety
from tests.mutmut_fork_safety import (
    DISABLED_VALUES,
    DRIVER_PHASES,
    ENV_BASELINE_VMSIZE,
    ENV_CWD,
    ENV_ENABLED,
    ENV_PYTEST_ARGS,
    ENV_ROOT,
    ENV_SYS_PATH,
    PATCHED_MARKER,
    WORKER_BOOTSTRAP,
    WORKER_BOOTSTRAP_FAILURE_EXIT_CODE,
    build_worker_pytest_args,
    driver_address_space_in_use,
    exec_mutant_worker,
    install_fork_safe_mutant_workers,
    install_if_mutmut_driver,
    run_mutant_in_fresh_interpreter,
)

pytestmark = pytest.mark.unit


class _StubRunner:
    """Stands in for mutmut's ``PytestRunner`` with the two members the hook uses."""

    def __init__(self) -> None:
        self._pytest_add_cli_args = ["--override-ini=addopts="]
        self.in_process_calls: list[tuple[str | None, list[str]]] = []

    def _pytest_args_regular_run(self, tests: list[str]) -> list[str]:
        return ["-x", "-q", *tests]

    def run_tests(self, *, mutant_name: str | None, tests: list[str]) -> int:
        self.in_process_calls.append((mutant_name, list(tests)))
        return 0


def _fresh_runner_cls() -> type[_StubRunner]:
    """A new subclass per test, so patching one never leaks into another."""
    return type("StubRunner", (_StubRunner,), {})


class _ExecCalledError(Exception):
    """Raised by the fake ``os.execve`` in place of replacing the process."""

    def __init__(self, path: str, argv: list[str], env: dict[str, str]) -> None:
        super().__init__(path)
        self.path = path
        self.argv = argv
        self.env = env


@pytest.fixture(autouse=True)
def _loaded_mutmut_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the config a real mutmut driver has already loaded.

    ``mutmut``'s ``config()`` resolves ``pyproject.toml`` against the working directory and caches
    it. A driver loads it at startup, so the code under test only ever gets a cache hit -- but a
    test runs in a working directory of its own (see ``tests/conftest.py``), where there is no
    config to find. Stubbing it keeps these tests independent of where they are run from instead
    of relying on some earlier test having warmed the cache.

    Patched by name rather than through an imported reference, because the code under test imports
    ``config`` inside the function that calls it -- so the lookup happens per call, and only a
    patch on the module attribute is seen.
    """
    monkeypatch.setattr("mutmut.configuration.config", lambda: types.SimpleNamespace(debug=False))


@pytest.fixture
def fake_execve(monkeypatch: pytest.MonkeyPatch) -> None:
    def _execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        raise _ExecCalledError(path, list(argv), dict(env))

    monkeypatch.setattr(os, "execve", _execve)


def test_the_hook_is_actually_installed_when_mutmut_is_driving() -> None:
    """End-to-end guard: under mutmut, importing the conftest must have patched it.

    Skipped in an ordinary pytest run. Inside a mutmut driver it is the only check
    that the wiring in ``tests/conftest.py`` still reaches mutmut before the fork
    loop starts -- if it stops reaching it, mutation runs silently go back to
    deadlocking and reporting the deadlock as a verdict.
    """
    mutmut_main = sys.modules.get("mutmut.__main__")
    if mutmut_main is None or os.environ.get("MUTANT_UNDER_TEST", "") not in DRIVER_PHASES:
        pytest.skip("not running inside a mutmut driver")
    if os.environ.get(ENV_ENABLED, "").strip().lower() in DISABLED_VALUES:
        pytest.skip(f"{ENV_ENABLED} turns the hook off on purpose")

    assert getattr(mutmut_main.PytestRunner, PATCHED_MARKER, False)


def test_install_is_a_no_op_outside_a_mutmut_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "mutmut.__main__", raising=False)
    assert install_if_mutmut_driver() is False


def test_install_is_a_no_op_inside_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    runner_cls = _fresh_runner_cls()
    stub_main = types.SimpleNamespace(PytestRunner=runner_cls)
    monkeypatch.setitem(sys.modules, "mutmut.__main__", stub_main)
    # A mutant name in MUTANT_UNDER_TEST means this process *is* the worker.
    monkeypatch.setenv("MUTANT_UNDER_TEST", "gwmock.cli.main.xǁfǁ__mutmut_1")

    assert install_if_mutmut_driver() is False
    assert runner_cls.run_tests is _StubRunner.run_tests


@pytest.mark.parametrize("disabled", sorted(DISABLED_VALUES))
def test_install_can_be_turned_off_from_the_environment(monkeypatch: pytest.MonkeyPatch, disabled: str) -> None:
    runner_cls = _fresh_runner_cls()
    monkeypatch.setitem(sys.modules, "mutmut.__main__", types.SimpleNamespace(PytestRunner=runner_cls))
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.setenv(ENV_ENABLED, f" {disabled.upper()} ")

    assert install_if_mutmut_driver() is False
    assert runner_cls.run_tests is _StubRunner.run_tests


@pytest.mark.parametrize("phase", sorted(DRIVER_PHASES))
def test_install_patches_the_runner_in_every_driver_phase(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    runner_cls = _fresh_runner_cls()
    monkeypatch.setitem(sys.modules, "mutmut.__main__", types.SimpleNamespace(PytestRunner=runner_cls))
    monkeypatch.setenv("MUTANT_UNDER_TEST", phase)

    assert install_if_mutmut_driver() is True
    assert runner_cls.run_tests is not _StubRunner.run_tests


def test_install_is_idempotent() -> None:
    runner_cls = _fresh_runner_cls()

    assert install_fork_safe_mutant_workers(runner_cls) is True
    patched = runner_cls.run_tests
    assert install_fork_safe_mutant_workers(runner_cls) is False
    assert runner_cls.run_tests is patched


def test_install_raises_when_the_runner_has_lost_the_members_it_leans_on() -> None:
    class Renamed:
        run_tests = None

    with pytest.raises(RuntimeError, match="_pytest_args_regular_run"):
        install_fork_safe_mutant_workers(Renamed)


def test_install_raises_when_mutmut_has_no_pytest_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mutmut.__main__", types.SimpleNamespace())
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")

    with pytest.raises(RuntimeError, match="PytestRunner"):
        install_if_mutmut_driver()


def test_driver_side_runs_stay_in_process() -> None:
    runner_cls = _fresh_runner_cls()
    install_fork_safe_mutant_workers(runner_cls)
    runner = runner_cls()

    assert runner.run_tests(mutant_name=None, tests=["tests/test_a.py::test_b"]) == 0
    assert runner.in_process_calls == [(None, ["tests/test_a.py::test_b"])]


def test_mutant_runs_are_handed_to_a_fresh_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    handed_over: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        mutmut_fork_safety,
        "run_mutant_in_fresh_interpreter",
        lambda _runner, *, mutant_name, tests: handed_over.append((mutant_name, list(tests))),
    )
    runner_cls = _fresh_runner_cls()
    install_fork_safe_mutant_workers(runner_cls)
    runner = runner_cls()

    with pytest.raises(AssertionError, match="unreachable"):
        runner.run_tests(mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=["tests/test_a.py::test_b"])

    assert handed_over == [("pkg.mod.xǁfǁ__mutmut_1", ["tests/test_a.py::test_b"])]
    assert runner.in_process_calls == []


def test_exec_mutant_worker_replaces_the_process_with_the_bootstrap(fake_execve: None) -> None:
    with pytest.raises(_ExecCalledError) as excinfo:
        exec_mutant_worker(_StubRunner(), mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=[])

    assert excinfo.value.path == sys.executable
    assert excinfo.value.argv[:3] == [sys.executable, "-c", WORKER_BOOTSTRAP]
    # The mutant name rides along only so `ps` can still identify the worker.
    assert excinfo.value.argv[3] == "pkg.mod.xǁfǁ__mutmut_1"


def test_build_worker_pytest_args_matches_mutmuts_own_composition() -> None:
    runner = _StubRunner()

    args = build_worker_pytest_args(runner, ["tests/test_a.py::test_b"])

    assert args == [
        "--rootdir=.",
        "--tb=native",
        "-x",
        "-q",
        "tests/test_a.py::test_b",
        "--override-ini=addopts=",
    ]


def test_build_worker_pytest_args_honours_mutmut_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    # Overrides the autouse stub above, which reports debug off.
    monkeypatch.setattr("mutmut.configuration.config", lambda: types.SimpleNamespace(debug=True))

    assert build_worker_pytest_args(_StubRunner(), [])[0] == "-vv"


def test_worker_environment_describes_the_run(
    fake_execve: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    # An empty entry means "the current directory", and a relative one moves with it;
    # both would resolve differently once the worker enters `mutants/`.
    monkeypatch.setattr(sys, "path", ["", "relative/dir", "/absolute/dir"])
    monkeypatch.setenv("MUTANT_UNDER_TEST", "pkg.mod.xǁfǁ__mutmut_1")

    with pytest.raises(_ExecCalledError) as excinfo:
        exec_mutant_worker(_StubRunner(), mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=["tests/test_a.py::test_b"])

    env = excinfo.value.env
    assert json.loads(env[ENV_SYS_PATH]) == [
        str(tmp_path / "relative" / "dir"),
        "/absolute/dir",
    ]
    assert env[ENV_ROOT] == str(tmp_path)
    assert env[ENV_CWD] == str(tmp_path / "mutants")
    assert json.loads(env[ENV_PYTEST_ARGS])[:2] == ["--rootdir=.", "--tb=native"]
    # mutmut sets this in the forked child before handing over; the worker needs it to
    # know which mutant to activate.
    assert env["MUTANT_UNDER_TEST"] == "pkg.mod.xǁfǁ__mutmut_1"


def test_the_worker_is_told_the_address_space_the_driver_had(
    fake_execve: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory ceiling in ``tests/conftest.py`` is relative to what is already mapped.

    A forked worker started life holding the driver's mapping; one that ``exec``s a fresh
    interpreter does not, so without this the same arithmetic yields a fraction of the intended
    ceiling and XLA aborts instead of the test failing.

    The reading is stubbed rather than taken live, so this asserts the contract that matters --
    whatever the driver measured reaches the worker -- on every platform, including those with no
    ``/proc`` to measure from.
    """
    monkeypatch.setattr(mutmut_fork_safety, "driver_address_space_in_use", lambda: 3 * 1024**3)

    with pytest.raises(_ExecCalledError) as excinfo:
        exec_mutant_worker(_StubRunner(), mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=[])

    assert int(excinfo.value.env[ENV_BASELINE_VMSIZE]) == 3 * 1024**3


def test_the_baseline_is_omitted_rather_than_sent_as_zero(fake_execve: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Unknown" must not reach the worker looking like "nothing mapped".

    A zero would be indistinguishable from a real reading of zero and would reinstate the tight
    ceiling this exists to avoid; an absent variable makes the worker fall back to its own size.
    """
    monkeypatch.setattr(mutmut_fork_safety, "driver_address_space_in_use", lambda: 0)

    with pytest.raises(_ExecCalledError) as excinfo:
        exec_mutant_worker(_StubRunner(), mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=[])

    assert ENV_BASELINE_VMSIZE not in excinfo.value.env


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="reads procfs, which is Linux-only")
def test_driver_address_space_is_a_real_reading_where_procfs_exists() -> None:
    assert driver_address_space_in_use() > 0


def test_driver_address_space_reports_zero_without_procfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing may raise where there is no ``/proc``: this suite runs on macOS too.

    Zero is the "do not know" answer the caller is written around -- it omits the variable rather
    than sending a figure the worker would read as "nothing mapped".
    """

    def _no_procfs(*_args: object, **_kwargs: object) -> str:
        raise OSError("no /proc on this platform")

    monkeypatch.setattr(Path, "read_text", _no_procfs)

    assert driver_address_space_in_use() == 0


def test_a_worker_that_cannot_exec_does_not_look_like_a_killed_mutant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("no such interpreter")

    exits: list[int] = []

    def _record_exit(code: int) -> None:
        exits.append(code)

    monkeypatch.setattr(os, "execve", _boom)
    monkeypatch.setattr(os, "_exit", _record_exit)

    run_mutant_in_fresh_interpreter(_StubRunner(), mutant_name="pkg.mod.xǁfǁ__mutmut_1", tests=[])

    # 1 would be "killed" and 0 "survived"; this has to be neither.
    assert exits == [WORKER_BOOTSTRAP_FAILURE_EXIT_CODE]


def _write_worker_fixture(tmp_path: Path, *, body: str) -> Path:
    """Lay out the minimum a worker expects: a project root with a `mutants/` copy."""
    mutants = tmp_path / "mutants"
    (mutants / "tests").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["src"]\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (mutants / "src").mkdir()
    (mutants / "pyproject.toml").write_text('[tool.mutmut]\nsource_paths = ["src"]\n', encoding="utf-8")
    (mutants / "tests" / "test_sample.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


def _run_worker_bootstrap(root: Path, *, pytest_args: list[str], sys_path: list[str] | None = None) -> int:
    env = dict(os.environ)
    env[ENV_SYS_PATH] = json.dumps(sys_path if sys_path is not None else [p for p in sys.path if p])
    env[ENV_PYTEST_ARGS] = json.dumps(pytest_args)
    env[ENV_ROOT] = str(root)
    env[ENV_CWD] = str(root / "mutants")
    env.pop("MUTANT_UNDER_TEST", None)
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter, arguments built here
        [sys.executable, "-c", WORKER_BOOTSTRAP, "pkg.mod.xǁfǁ__mutmut_1"],
        env=env,
        capture_output=True,
        check=False,
        timeout=300,
    )
    return completed.returncode


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("def test_ok():\n    assert True\n", 0),
        ("def test_fails():\n    assert False\n", 1),
    ],
)
def test_worker_bootstrap_runs_pytest_and_returns_its_exit_code(tmp_path: Path, body: str, expected: int) -> None:
    root = _write_worker_fixture(tmp_path, body=body)

    assert (
        _run_worker_bootstrap(
            root,
            pytest_args=["--rootdir=.", "--tb=native", "-x", "-q", "-p", "no:cacheprovider", "tests/"],
        )
        == expected
    )


def test_worker_bootstrap_reports_its_own_failure_distinctly(tmp_path: Path) -> None:
    root = _write_worker_fixture(tmp_path, body="def test_ok():\n    assert True\n")

    # An unimportable sys.path is the shape every bootstrap failure takes: pytest is
    # gone, so nothing ran and no verdict was reached.
    assert (
        _run_worker_bootstrap(
            root,
            pytest_args=["-q", "tests/"],
            sys_path=[str(tmp_path / "nowhere")],
        )
        == WORKER_BOOTSTRAP_FAILURE_EXIT_CODE
    )
