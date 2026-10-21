"""Capture, compare, and recreate the software environment of a run.

Reproducing a simulation bit-for-bit requires the same dependency versions.
These helpers record a full environment freeze into metadata and, on request,
rebuild that environment in an isolated, cached uv virtualenv so a reproduction
can re-execute against the exact dependencies the original run used.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from filelock import FileLock

logger = logging.getLogger("gwmock")

#: Set in the child process so a recreated environment never re-isolates.
ISOLATION_ENV_VAR = "GWMOCK_ISOLATED"
#: Overrides the default cache location for recreated environments.
CACHE_ENV_VAR = "GWMOCK_ENV_CACHE"


@functools.lru_cache(maxsize=1)
def capture_environment() -> dict[str, Any]:
    """Return a freeze of the active environment: Python version and every dist.

    Cached because the environment does not change during a run, so per-batch
    metadata writes reuse a single scan of the installed distributions.
    """
    packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        version = dist.version
        # Distributions are keyed case-insensitively; keep the first seen version
        # and normalise the name to its canonical (lower) form for stable diffs.
        packages.setdefault(name.lower(), version)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": dict(sorted(packages.items())),
    }


#: Where Linux reports the processor model. A module attribute so a test can point it elsewhere.
_CPUINFO = Path("/proc/cpuinfo")


@functools.lru_cache(maxsize=1)
def cpu_model() -> str | None:
    """Return the processor's model name, or ``None`` when the platform does not say.

    The marketing name -- ``"AMD Ryzen 7 5800H with Radeon Graphics"`` -- rather than the
    architecture, which the record already carries. It is what tells two machines of one
    architecture apart when a numerical difference between them has to be explained, and it names
    a product rather than a machine, so it belongs in the shareable record.

    Read from ``/proc/cpuinfo`` on Linux (``model name``, with ``Hardware`` and ``Processor`` as the
    spellings ARM kernels use), from ``sysctl machdep.cpu.brand_string`` on macOS, and from
    :func:`platform.processor` on Windows, where that is the processor identifier. Anything else,
    or any failure to read, is ``None``: an explicit "unknown" rather than a guess.
    """
    model: str | None = None
    if sys.platform.startswith("linux"):
        model = _cpu_model_from_cpuinfo(_CPUINFO)
    elif sys.platform == "darwin":
        sysctl = shutil.which("sysctl")
        if sysctl is not None:
            try:
                completed = subprocess.run(  # noqa: S603
                    [sysctl, "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                completed = None
            if completed is not None and completed.returncode == 0:
                model = completed.stdout.strip() or None
    elif sys.platform == "win32":
        model = platform.processor().strip() or None
    return model


def _cpu_model_from_cpuinfo(cpuinfo: Path) -> str | None:
    """Return the processor model named in a ``/proc/cpuinfo``-style file, or ``None``."""
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields.setdefault(key.strip().lower(), value.strip())
    for key in ("model name", "hardware", "processor"):
        value = fields.get(key, "")
        # x86 kernels also have a `processor` line, holding the core number; that is not a model.
        if value and not value.isdigit():
            return value
    return None


def hostname() -> str:
    """Return this machine's network name -- private provenance, never written to a shareable record."""
    return socket.gethostname()


# A recorded pin must be a plain distribution name and version — never an
# option like ``--index-url``. Metadata files are shared for reproduction, so
# their contents are untrusted input that ends up as `uv pip install` arguments.
_DIST_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]*$")


def environment_requirements(snapshot: dict[str, Any]) -> list[str]:
    """Return sorted, validated ``name==version`` requirement strings for a snapshot.

    Every package name and version is checked against strict distribution-name
    and version syntax; a pin that does not match (e.g. an option-shaped key such
    as ``--index-url``) raises ``ValueError`` rather than being forwarded to the
    installer, so a tampered metadata file cannot inject installer options.
    """
    packages = snapshot.get("packages") or {}
    requirements: list[str] = []
    for name, version in sorted(packages.items()):
        if not (isinstance(name, str) and _DIST_NAME_RE.match(name)):
            raise ValueError(f"Refusing to reproduce: invalid package name {name!r} in recorded environment.")
        if not (isinstance(version, str) and _VERSION_RE.match(version)):
            raise ValueError(
                f"Refusing to reproduce: invalid version {version!r} for package {name!r} in recorded environment."
            )
        requirements.append(f"{name}=={version}")
    return requirements


def diff_environment(recorded: dict[str, Any], installed: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Return ``(package, recorded_version, installed_version)`` for every mismatch.

    A package recorded but absent from the installed environment yields an
    installed version of ``None``. Packages only present in the installed
    environment are ignored: they cannot have influenced the recorded run.
    """
    recorded_packages = recorded.get("packages") or {}
    installed_packages = installed.get("packages") or {}
    mismatches: list[tuple[str, str, str | None]] = []
    for name, recorded_version in sorted(recorded_packages.items()):
        installed_version = installed_packages.get(name)
        if installed_version != recorded_version:
            mismatches.append((name, recorded_version, installed_version))
    return mismatches


def environments_match(recorded: dict[str, Any], installed: dict[str, Any]) -> bool:
    """Return whether the installed environment is identical to the recorded freeze.

    Requires the exact same package set — same names and versions, and no extra
    installed packages, since extras can change optional imports, plugin/backend
    resolution, or entry points — plus the same Python ``major.minor``. (The
    laxer :func:`diff_environment`, which ignores installed-only extras, backs
    the advisory drift warning, not this exact-reproduction gate.)
    """
    if (recorded.get("packages") or {}) != (installed.get("packages") or {}):
        return False
    return _python_minor(recorded.get("python")) == _python_minor(installed.get("python"))


def _python_minor(version: str | None) -> str | None:
    """Return the ``major.minor`` prefix of a Python version string."""
    if not version:
        return None
    return ".".join(version.split(".")[:2])


def environment_key(snapshot: dict[str, Any]) -> str:
    """Return a stable short hash identifying a recreatable environment.

    Keyed by the Python ``major.minor`` plus the full sorted requirement set, so
    two runs with the same dependencies reuse one cached virtualenv.
    """
    payload = "\n".join([_python_minor(snapshot.get("python")) or "", *environment_requirements(snapshot)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_cache_root() -> Path:
    """Return the directory under which recreated environments are cached."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "gwmock" / "reproduction-envs"


def build_isolated_environment(
    snapshot: dict[str, Any],
    cache_root: Path | None = None,
    *,
    uv_executable: str | None = None,
) -> Path:
    """Create (or reuse) a cached uv virtualenv pinned to ``snapshot``.

    The env is keyed by :func:`environment_key`, so repeated reproductions of the
    same run reuse it. Returns the path to the environment's Python interpreter.
    Raises ``RuntimeError`` if uv is unavailable and ``CalledProcessError`` if
    provisioning or installing the recorded versions fails (e.g. a dev/editable
    version that is not published to an index).
    """
    cache_root = Path(cache_root) if cache_root is not None else default_cache_root()
    uv_executable = uv_executable or shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError(
            "uv is required to build an isolated reproduction environment. "
            "Install uv (https://docs.astral.sh/uv/) or rerun without --isolate."
        )

    key = environment_key(snapshot)
    env_dir = cache_root / key
    python_bin = env_dir / "bin" / "python"
    marker = env_dir / ".gwmock-ready"
    if marker.exists() and python_bin.exists():
        logger.info("Reusing cached reproduction environment at %s", env_dir)
        return python_bin

    cache_root.mkdir(parents=True, exist_ok=True)
    # Serialize concurrent builds of the same environment (e.g. parallel
    # reproductions of one run) so they cannot rmtree/install over each other.
    with FileLock(str(cache_root / f"{key}.lock")):
        if marker.exists() and python_bin.exists():  # another process built it while we waited
            logger.info("Reusing cached reproduction environment at %s", env_dir)
            return python_bin
        if env_dir.exists():
            shutil.rmtree(env_dir)  # remove a partial build

        venv_command = [uv_executable, "venv", str(env_dir)]
        python_minor = _python_minor(snapshot.get("python"))
        if python_minor:
            venv_command += ["--python", python_minor]
        logger.info("Creating reproduction environment (python %s) at %s", python_minor or "current", env_dir)
        subprocess.run(venv_command, check=True)  # noqa: S603

        requirements = environment_requirements(snapshot)
        if requirements:
            logger.info("Installing %d recorded package versions into the reproduction environment", len(requirements))
            # The trailing "--" makes uv treat every requirement as a positional
            # package spec, never an option, as a second guard against injection.
            subprocess.run(  # noqa: S603
                [uv_executable, "pip", "install", "--python", str(python_bin), "--", *requirements], check=True
            )

        marker.write_text("ready\n")
    return python_bin


def reproduce_in_isolated_environment(
    snapshot: dict[str, Any],
    argv: list[str],
    cache_root: Path | None = None,
) -> int:
    """Recreate the recorded environment and re-run ``gwmock`` inside it.

    ``argv`` is the original ``gwmock`` argument vector (``sys.argv[1:]``); the
    ``--isolate`` flag is stripped so the child does not attempt to isolate
    again, and ``ISOLATION_ENV_VAR`` is set as a second guard. Returns the child
    process's exit code.
    """
    python_bin = build_isolated_environment(snapshot, cache_root)
    child_gwmock = python_bin.parent / "gwmock"
    command = [str(child_gwmock), *[argument for argument in argv if argument != "--isolate"]]
    child_env = dict(os.environ)
    child_env[ISOLATION_ENV_VAR] = "1"
    logger.info("Re-running reproduction in the isolated environment: %s", " ".join(command))
    completed = subprocess.run(command, env=child_env, check=False)  # noqa: S603
    return completed.returncode
