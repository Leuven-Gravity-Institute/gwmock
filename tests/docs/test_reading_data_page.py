"""The reading guide's HDF5 example must read a file gwmock writes.

The page is what a user copies. GWpy's HDF5 reader selects a dataset with ``path=`` and rejects
``channel=`` with a ``TypeError``, which the page once showed; this reads a gwmock-written file with
the exact keyword the page uses, so a change on either side is caught here rather than by a user.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "reading-data.md"
_HDF5_READ = re.compile(r'TimeSeries\.read\("filename\.hdf5",\s*(\w+)="([^"]+)"\)')


def test_the_documented_hdf5_read_works_on_a_gwmock_file(tmp_path):
    pytest.importorskip("gwpy", reason="the [gwpy] extra is not installed")
    from gwmock_data import TimeSeries as ChannelSeries
    from gwmock_signal import DetectorStrainStack
    from gwpy.timeseries import TimeSeries

    match = _HDF5_READ.search(_PAGE.read_text(encoding="utf-8"))
    assert match is not None, "the page no longer shows how to read an HDF5 file"
    keyword, channel = match.groups()

    # Written the way the noise and signal writers write a file.
    samples = np.arange(16, dtype=float)
    path = tmp_path / "filename.hdf5"
    DetectorStrainStack.from_mapping([channel], {channel: ChannelSeries(samples, t0=1000.0, sample_rate=4.0)}).write(
        path, format="hdf5"
    )

    data = TimeSeries.read(str(path), **{keyword: channel})

    np.testing.assert_allclose(np.asarray(data.value), samples, rtol=0.0, atol=0.0)
    assert float(data.t0.value) == 1000.0
