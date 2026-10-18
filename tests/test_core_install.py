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


def test_hdf5_strain_artifacts_work_and_frames_name_their_extra(tmp_path: Path):
    """An HDF5 artifact is declared, validated, given its record and hashed with h5py hidden; GWF still needs GWpy."""
    h5, gwf = tmp_path / "H-H1_TEST-100-2.h5", tmp_path / "a.gwf"
    gwf.write_bytes(b"\\0" * 16)
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_data import hdf5
        from gwmock.cli.utils.hash import compute_content_hash
        from gwmock.cli.utils.metadata import embed_metadata_record
        from gwmock.strain_schema import declare_strain_schema, read_run_metadata, require_strain_schema

        hdf5.write(hdf5.Group({{"H1:STRAIN": hdf5.Dataset(np.arange(8.0), {{"t0": 100.0, "dt": 0.25}})}}), {str(h5)!r})
        declared = declare_strain_schema({str(h5)!r})
        embedded = embed_metadata_record({str(h5)!r}, {{"seed": 7}})
        result = {{
            "declared": declared,
            "schema": require_strain_schema({str(h5)!r}).name,
            "record": read_run_metadata({str(h5)!r}),
            "hash": compute_content_hash({str(h5)!r}),
            "embedded": embedded,
            "x0": float(hdf5.read({str(h5)!r}, data=False).members["H1:STRAIN"].attrs["x0"]),
        }}
        try:
            compute_content_hash({str(gwf)!r})
        except ModuleNotFoundError as exc:
            result["hash_gwf"] = str(exc)
        """
    )

    result = outcome["result"]
    assert outcome["loaded"] == []
    assert result["declared"] is True
    assert result["embedded"] is True
    assert result["schema"] == "gwmock-strain"
    assert result["record"] == {"seed": 7}
    assert result["x0"] == 100.0
    assert result["hash"].startswith("sha256:")
    assert "pip install 'gwmock[gwpy]'" in result["hash_gwf"]


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
