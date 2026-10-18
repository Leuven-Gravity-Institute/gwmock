"""What works without h5py and GWpy, which Python 3.15 cannot install yet.

CI runs this module on Python 3.15 with only the package and pytest installed; with everything installed
it checks the same paths never import those packages. Each scenario runs in a fresh interpreter, with an
import hook hiding h5py and GWpy, so the missing-package paths are exercised on every Python.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Makes `import h5py` and `import gwpy` fail as they do where neither is installed.
_HIDE = """
import importlib.abc, sys

class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"h5py", "gwpy"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Hide())
"""


def _run(code: str) -> dict:
    """Run ``code`` with h5py and GWpy hidden; it sets ``result``, printed with what it loaded."""
    script = (
        _HIDE
        + textwrap.dedent(code)
        + textwrap.dedent(
            """
            import json as _json, sys as _sys
            _loaded = sorted({name.split(".")[0] for name in _sys.modules} & {"h5py", "gwpy"})
            print(_json.dumps({"result": result, "loaded": _loaded}))
            """
        )
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_package_its_cli_and_its_adapters_import_without_h5py_or_gwpy():
    """The package, the CLI, the orchestration that runs a simulation and the noise adapter."""
    outcome = _run(
        """
        import gwmock, gwmock.cli.main, gwmock.cli.adapter_orchestration, gwmock.noise.adapter
        import gwmock.strain_schema, gwmock.data, gwmock.cli.utils.hash
        result = None
        """
    )

    assert outcome["loaded"] == []


def test_the_time_series_injects_and_round_trips_without_h5py_or_gwpy():
    """A chunk is injected into a segment, and the segment survives a JSON round trip."""
    outcome = _run(
        """
        import numpy as np
        from gwmock.data.time_series.time_series import TimeSeries

        segment = TimeSeries(np.zeros((2, 64)), start_time=1_400_000_000.0, sampling_frequency=16.0)
        chunk = TimeSeries(np.ones((2, 8)), start_time=1_400_000_001.0, sampling_frequency=16.0)
        segment.inject(chunk)
        back = TimeSeries.from_json_dict(segment.to_json_dict())
        result = [float(np.sum(back[0].data)), float(np.sum(back[1].data)), back == segment]
        """
    )

    assert outcome["loaded"] == []
    assert outcome["result"] == [8.0, 8.0, True]


def test_hdf5_and_frame_files_name_their_extra(tmp_path: Path):
    """A .npy artifact needs neither package; HDF5 and GWF entry points say what to install."""
    npy, h5, gwf = tmp_path / "a.npy", tmp_path / "a.h5", tmp_path / "a.gwf"
    for path in (npy, h5, gwf):
        path.write_bytes(b"\\0" * 16)
    outcome = _run(
        f"""
        from gwmock.cli.utils.hash import compute_content_hash
        from gwmock.strain_schema import declare_strain_schema

        result = {{"npy": declare_strain_schema({str(npy)!r})}}
        for label, call in {{
            "declare": lambda: declare_strain_schema({str(h5)!r}),
            "hash_hdf5": lambda: compute_content_hash({str(h5)!r}),
            "hash_gwf": lambda: compute_content_hash({str(gwf)!r}),
        }}.items():
            try:
                call()
            except ModuleNotFoundError as exc:
                result[label] = str(exc)
        """
    )

    assert outcome["result"]["npy"] is False
    assert "pip install 'gwmock[hdf5]'" in outcome["result"]["declare"]
    assert "pip install 'gwmock[hdf5]'" in outcome["result"]["hash_hdf5"]
    assert "pip install 'gwmock[gwpy]'" in outcome["result"]["hash_gwf"]


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
def test_the_command_line_starts_without_numpy():
    """From Python 3.15, building the command-line app does not import NumPy."""
    outcome = _run(
        """
        import sys
        import gwmock.cli.main
        result = "numpy" in sys.modules
        """
    )

    assert outcome["result"] is False
