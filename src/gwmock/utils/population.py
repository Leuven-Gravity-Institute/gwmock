"""Utility functions for reading population files."""

from __future__ import annotations

from pathlib import Path

import h5py

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    raise ModuleNotFoundError(
        f"{__name__} needs pandas, which is not installed. Install it with: pip install 'gwmock[pycbc]'",
        name=exc.name,
    ) from exc


def read_pycbc_population_file(file_name):
    """
    Read a PyCBC population file (.hdf, .h5) and return a pandas DataFrame.

    Supports:
      - HDF5 population files from pycbc_population
    Parameters
    ----------
    file_name : Path or str
        Path to the PyCBC population file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing population parameters.
    """
    file_path = Path(file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # --- HDF5 (.hdf or .h5) ---
    if suffix in {".hdf", ".h5"}:
        with h5py.File(file_path, "r") as f:
            # PyCBC stores parameters as datasets
            data = {key: value[()] for key, value in f.items()}

            # Collect attributes (Include static parameters in config files)
            attrs = dict(f.attrs.items())

        df = pd.DataFrame(data)
        df.attrs.update(attrs)  # Attach metadata
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df
