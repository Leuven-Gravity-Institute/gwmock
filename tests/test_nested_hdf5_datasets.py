"""A dataset below a group is declared, validated and hashed like a top-level one."""

from __future__ import annotations

import numpy as np
from gwmock_data import hdf5

from gwmock.cli.utils.hash import compute_content_hash
from gwmock.strain_schema import declare_strain_schema, missing_layout_attributes


def _artifact(path, inner_epoch=100.0):
    """A strain file with one dataset at the top and one inside a group."""
    hdf5.write(
        hdf5.Group(
            {
                "H1:STRAIN": hdf5.Dataset(np.arange(4.0), {"t0": 100.0, "dt": 0.25}),
                "aux": hdf5.Group({"L1:STRAIN": hdf5.Dataset(np.arange(4.0), {"t0": inner_epoch, "dt": 0.25})}),
            }
        ),
        path,
    )


def test_a_nested_dataset_is_brought_up_to_the_layout(tmp_path):
    """Its grid is filled in and its channel is its path in the file."""
    path = tmp_path / "H-H1_TEST-100-1.h5"
    _artifact(path)

    assert declare_strain_schema(path) is True

    inner = hdf5.read(path, data=False).members["aux"].members["L1:STRAIN"].attrs
    assert float(inner["x0"]) == 100.0
    assert inner["channel"] == "aux/L1:STRAIN"
    assert missing_layout_attributes(path) == {}


def test_a_nested_dataset_is_part_of_the_content_hash(tmp_path):
    """Moving only the nested dataset in time changes the digest."""
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    _artifact(first)
    _artifact(second, inner_epoch=101.0)

    assert compute_content_hash(first) != compute_content_hash(second)
