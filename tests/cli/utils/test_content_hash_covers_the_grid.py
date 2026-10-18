# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""The content hash must notice a change of grid, in every format a run can write.

The digest folds each channel's samples together with the scalar metadata that says *where* those
samples sit -- epoch, sample interval, length. The GWF branch always recorded those; the HDF5 branch
recorded none of them, so two HDF5 files holding identical samples at different GPS times, or at
different sample rates, produced the same hash. Measured, both collided.

That asymmetry was survivable while HDF5 was the secondary format. It is not now that HDF5 is what a run
writes by default, because the weaker check would become the one every user gets.

These tests are parametrised over both formats deliberately: the point is not that HDF5 works, it is that
the two answer the same question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gwmock_data import hdf5

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:  # Python 3.15: GWpy does not install yet; its tests are skipped
    if exc.name != "gwpy":
        raise
    TimeSeries = None

from gwmock.cli.utils.hash import compute_content_hash

pytestmark = pytest.mark.unit

FORMATS = [".gwf", ".hdf5"]
SAMPLES = np.linspace(0.0, 1.0, 1024)


def _digest(path: Path) -> str:
    """Return the content hash of *path*, refusing a ``None``.

    `compute_content_hash` returns `None` for a file it cannot decode. Two `None`s compare equal and a
    `None` differs from any digest, so a comparison written without this passes whether or not anything
    was actually hashed -- an inequality test survives one side failing to decode, and an equality test
    survives both. Every assertion below is about *which* digest came out, so none of them may be
    satisfied by no digest coming out at all. Returning the value rather than asserting at each call
    site is what stops the next test in this file from being written without the check.
    """
    digest = compute_content_hash(path)
    assert digest is not None, f"nothing was hashed for {path}, so the comparison would prove nothing"
    return digest


def _written(
    directory: Path,
    suffix: str,
    *,
    t0: float = 1000000000.0,
    rate: float = 256.0,
    samples: np.ndarray | None = None,
) -> Path:
    """Write one channel, however it differs from the baseline.

    `name` is passed as well as `channel`: gwpy's HDF5 writer names the dataset from it and raises
    without one, so a series built for a variation has to be constructed the same way as the baseline
    rather than with the minimum that GWF happens to accept.
    """
    series = TimeSeries(
        SAMPLES if samples is None else samples,
        t0=t0,
        sample_rate=rate,
        channel="H1:STRAIN",
        unit="strain",
        name="H1:STRAIN",
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"probe{suffix}"
    series.write(str(path), overwrite=True)
    return path


@pytest.mark.requires_gwpy
@pytest.mark.parametrize("suffix", FORMATS)
def test_the_same_file_hashes_the_same(tmp_path: Path, suffix: str) -> None:
    """The baseline the others are read against: nothing moved, nothing changes."""
    assert _digest(_written(tmp_path / "a", suffix)) == _digest(_written(tmp_path / "b", suffix))


@pytest.mark.requires_gwpy
@pytest.mark.parametrize("suffix", FORMATS)
def test_moving_the_epoch_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """Identical samples at a different GPS time are different data.

    This is the case that collided for HDF5: a segment written for the wrong epoch was indistinguishable
    from the right one.
    """
    original = _digest(_written(tmp_path / "a", suffix))
    moved = _digest(_written(tmp_path / "b", suffix, t0=1000000512.0))
    assert original != moved


@pytest.mark.requires_gwpy
@pytest.mark.parametrize("suffix", FORMATS)
def test_changing_the_sample_rate_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """Same samples, half the span. Also collided for HDF5."""
    original = _digest(_written(tmp_path / "a", suffix))
    resampled = _digest(_written(tmp_path / "b", suffix, rate=512.0))
    assert original != resampled


@pytest.mark.requires_gwpy
@pytest.mark.parametrize("suffix", FORMATS)
def test_changing_a_sample_changes_the_hash(tmp_path: Path, suffix: str) -> None:
    """The property that already held, kept explicit so a metadata-only digest cannot pass this file."""
    original = _digest(_written(tmp_path / "a", suffix))
    altered = SAMPLES.copy()
    altered[17] += 1e-3
    assert original != _digest(_written(tmp_path / "b", suffix, samples=altered))


class TestTheOtherSpellingOfTheGrid:
    """The writer a run's own outputs go through records the grid under different names.

    The tests above write through gwpy, which spells the epoch and the sample interval `x0` and `dx`.
    `gwmock_signal.DetectorStrainStack.write` -- the writer both halves of an orchestrated run use, so
    the one behind almost every artifact a user ends up with -- spells them `t0` and `dt`. The digest
    read only the first spelling, so for those files the grid dropped out of it again and the very
    collision this module exists to prevent came back on the primary path. Measured before the fix: the
    same samples at t0=100.0 and t0=612.0 produced one identical digest.
    """

    @staticmethod
    def _stack_written(path: Path, *, t0: float = 100.0, rate: float = 8.0) -> Path:
        from gwmock_data import TimeSeries as ChannelSeries
        from gwmock_signal import DetectorStrainStack

        series = ChannelSeries(np.arange(8, dtype=float), t0=t0, sample_rate=rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        DetectorStrainStack.from_mapping(["H1:MOCK"], {"H1:MOCK": series}).write(str(path), format="hdf5")
        return path

    def test_the_writer_really_does_use_the_other_spelling(self, tmp_path: Path) -> None:
        """Pinned, because everything below is only meaningful while it is true."""
        attributes = set(hdf5.read(self._stack_written(tmp_path / "probe.hdf5"), data=False).members["H1:MOCK"].attrs)
        assert {"t0", "dt"} <= attributes

    def test_moving_the_epoch_changes_the_hash(self, tmp_path: Path) -> None:
        original = _digest(self._stack_written(tmp_path / "a.hdf5"))
        moved = _digest(self._stack_written(tmp_path / "b.hdf5", t0=612.0))
        assert original != moved

    def test_changing_the_sample_rate_changes_the_hash(self, tmp_path: Path) -> None:
        original = _digest(self._stack_written(tmp_path / "a.hdf5"))
        resampled = _digest(self._stack_written(tmp_path / "b.hdf5", rate=16.0))
        assert original != resampled

    def test_the_two_spellings_of_one_grid_hash_alike(self, tmp_path: Path) -> None:
        """A file carrying both names must not hash differently from one carrying either.

        Only one spelling may contribute to the digest. Folding both in would make an artifact declared
        to the strain schema -- which records `x0`/`dx` alongside the `t0`/`dt` the stack writer leaves
        in place -- hash differently from the same data before it was declared.
        """
        aliased = self._stack_written(tmp_path / "a.hdf5")
        both = self._stack_written(tmp_path / "b.hdf5")
        grid = hdf5.read(both, data=False).members["H1:MOCK"].attrs
        hdf5.update_attrs(both, {"H1:MOCK": {"x0": float(grid["t0"]), "dx": float(grid["dt"])}})

        assert _digest(aliased) == _digest(both)


def test_an_hdf5_dataset_without_grid_attributes_still_hashes(tmp_path: Path) -> None:
    """A file written by something other than gwpy has no `x0`/`dx`, and must not raise.

    The check is only as strong as what the writer recorded. Reading the attributes when they are there
    and hashing the samples alone when they are not is the honest behaviour -- the alternative, demanding
    them, would refuse to hash a perfectly good file somebody else produced.
    """
    path = tmp_path / "bare.hdf5"
    hdf5.write(hdf5.Group({"strain": hdf5.Dataset(SAMPLES)}), path)

    assert compute_content_hash(path) is not None
