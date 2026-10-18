"""What ``import gwmock`` loads, observed in a fresh interpreter.

The test suite itself imports NumPy long before this runs, so ``sys.modules`` in this process says nothing
about what importing the package costs; each check runs in its own interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _loaded_after(code: str, modules: tuple[str, ...]) -> list[str]:
    """Run ``code`` in a fresh interpreter and return which of ``modules`` it left imported."""
    script = f"{code}\nimport sys\nprint(','.join(name for name in {modules!r} if name in sys.modules))"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)  # noqa: S603
    return [name for name in completed.stdout.strip().split(",") if name]


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
def test_importing_the_package_defers_numpy():
    """From Python 3.15, ``import gwmock`` sets up logging without importing NumPy."""
    assert _loaded_after("import gwmock", ("numpy", "gwmock.utils.random")) == []


def test_the_random_helpers_still_work_after_a_plain_import():
    """Deferred or not, the names ``gwmock.utils`` exports resolve to working functions."""
    loaded = _loaded_after(
        "import gwmock\nfrom gwmock.utils import get_rng, set_seed\nset_seed(1)\nget_rng().random()",
        ("numpy", "gwmock.utils.random"),
    )

    assert loaded == ["numpy", "gwmock.utils.random"]
