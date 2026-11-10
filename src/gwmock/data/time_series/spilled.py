"""A chunk of strain whose samples live in a file on disk rather than in memory.

A signal that crosses a segment boundary is carried into the following segments as a *tail*. A
binary neutron star from a low minimum frequency lasts hours, so at any time a run holds the tails of
every signal still in band, and in memory they grow with the event rate times the square of the
signal duration. :class:`SpilledTimeSeries` holds such a tail in an ``.npy`` file instead: it is a
:class:`~gwmock.data.time_series.time_series.TimeSeries` whose samples are a read-only memory map, so
code that reads a window of it reads only that window from disk.

Each file is written once, when the tail is first spilled, and never changed. Consuming a tail does
not rewrite it: :meth:`SpilledTimeSeries.after` returns a view of the same file starting further in,
and a checkpoint records a tail as the file, the first and last sample of the view, and the series'
time, metadata and channel identities -- a few hundred bytes, where the samples are megabytes to
gigabytes. Deleting files no longer referenced is the run's job; see
:meth:`gwmock.cli.utils.checkpoint.CheckpointManager.remove_unreferenced_tails`.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from astropy.units import Quantity

from gwmock.data.time_series.time_series import TimeSeries

#: The ``__type__`` a checkpoint records a spilled tail under, and the class name the decoder looks up.
SPILLED_TYPE = "SpilledTimeSeries"


def _channel_identities(series: TimeSeries) -> list[dict[str, Any]]:
    """Each channel's name, unit and channel label, which a series built from raw samples lacks."""
    return [
        {"name": series[i].name, "unit": series[i].unit, "channel": series.channels[i]}
        for i in range(series.num_of_channels)
    ]


class SpilledTimeSeries(TimeSeries):
    """A read-only, disk-backed view of a chunk's samples, from sample ``offset`` to ``stop`` of its file.

    Build one with :meth:`spill`, not the constructor. Equal to the chunk it was spilled from in every
    sample, its start time, sampling frequency, metadata and channel identities, so injecting it -- or
    any view of it -- places exactly what injecting the in-memory chunk would.
    """

    _spill_path: Path
    _spill_offset: int
    _spill_stop: int

    @classmethod
    def spill(cls, chunk: TimeSeries, directory: str | Path) -> SpilledTimeSeries:
        """Write *chunk*'s samples to a new file in *directory* and return a view of the whole of it.

        Args:
            chunk: The chunk to spill. Unchanged.
            directory: Where to write the file; created if missing. The file name is random, so two
                spills never collide.

        Returns:
            The spilled chunk.
        """
        # Absolute, because a checkpoint records the path and a resumed run may start elsewhere.
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.npy"
        length = chunk[0].n_samples
        # Written to a temporary name and renamed, so an interrupted write never leaves a partial
        # file under a name a checkpoint could come to reference. Channel by channel, into the file,
        # so spilling a tail does not first stack a second copy of it in memory.
        partial = path.with_suffix(".partial")
        samples = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.result_type(*(chunk[i].data for i in range(chunk.num_of_channels))),
            shape=(chunk.num_of_channels, length),
        )
        for index in range(chunk.num_of_channels):
            samples[index] = np.asarray(chunk[index].data)
        samples.flush()
        del samples
        partial.replace(path)
        return cls._open(
            path,
            0,
            length,
            chunk.start_time,
            chunk.sampling_frequency,
            chunk.metadata,
            _channel_identities(chunk),
        )

    @classmethod
    def _open(
        cls,
        path: Path,
        offset: int,
        stop: int,
        start_time: Quantity,
        sampling_frequency: Quantity,
        metadata: dict[str, Any],
        channels: list[dict[str, Any]],
    ) -> SpilledTimeSeries:
        """Map samples ``offset:stop`` of the file at *path*, starting at *start_time*."""
        samples = np.load(path, mmap_mode="r")[:, offset:stop]
        series = cls(data=samples, start_time=start_time, sampling_frequency=sampling_frequency)
        series._spill_path = Path(path)
        series._spill_offset = int(offset)
        series._spill_stop = int(stop)
        series.metadata.update(copy.deepcopy(metadata))
        for index, channel in enumerate(channels[: series.num_of_channels]):
            series._data[index] = dataclasses.replace(series[index], name=channel.get("name"), unit=channel.get("unit"))
            series.channels[index] = channel.get("channel")
        return series

    @property
    def spill_path(self) -> Path:
        """The file holding this chunk's samples."""
        return self._spill_path

    def _view(self, offset: int, stop: int) -> SpilledTimeSeries:
        """Samples ``offset:stop`` of the same file, with this chunk's metadata and channels."""
        start = self.start_time + Quantity((offset - self._spill_offset) / self.sampling_frequency.value, unit="s")
        return self._open(
            self._spill_path,
            offset,
            stop,
            start.to(self.start_time.unit),
            self.sampling_frequency,
            self.metadata,
            _channel_identities(self),
        )

    def _samples_before(self, time: Quantity) -> int:
        """How many of this chunk's samples lie before *time*, clipped to ``[0, len]``.

        Counted on the chunk's own sample grid, rounding to the nearest sample: the times asked about
        are segment boundaries, which a spilled tail -- already placed on the run's grid -- meets on a
        sample.
        """
        position = (time - self.start_time).to_value("s") * self.sampling_frequency.to_value("Hz")
        return min(max(round(position), 0), self._spill_stop - self._spill_offset)

    def until(self, time: Quantity) -> SpilledTimeSeries:
        """This chunk's samples before *time*, as a view of the same file. Reads nothing."""
        return self._view(self._spill_offset, self._spill_offset + self._samples_before(time))

    def after(self, time: Quantity) -> SpilledTimeSeries | None:
        """This chunk's samples from *time* on, as a view of the same file, or ``None`` if none are left."""
        offset = self._spill_offset + self._samples_before(time)
        if offset >= self._spill_stop:
            return None
        return self._view(offset, self._spill_stop)

    def in_memory(self) -> TimeSeries:
        """A copy of this chunk with its samples read into memory, as an ordinary series."""
        series = TimeSeries(
            data=np.array([np.asarray(self[i].data) for i in range(self.num_of_channels)]),
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
        )
        series.metadata.update(copy.deepcopy(self.metadata))
        for index, channel in enumerate(_channel_identities(self)):
            series._data[index] = dataclasses.replace(series[index], name=channel["name"], unit=channel["unit"])
            series.channels[index] = channel["channel"]
        return series

    def summed_squares(self, block_samples: int = 1 << 20) -> float:
        """The sum of every sample squared, over all channels, read from disk a block at a time.

        The same quantity ``np.sum(np.square(samples))`` gives, without holding a squared copy of the
        whole chunk. Summed per block, so it can differ from the one-pass sum in the last bits.

        Args:
            block_samples: Samples per channel read at once.

        Returns:
            The sum, as a float.
        """
        total = 0.0
        length = self._spill_stop - self._spill_offset
        for start in range(0, length, block_samples):
            block = np.asarray([np.asarray(self[i].data[start : start + block_samples]) for i in range(len(self))])
            total += float(np.sum(np.square(block, dtype=float)))
        return total

    def __deepcopy__(self, memo: dict[int, Any]) -> SpilledTimeSeries:
        """A new view of the same file: the samples are never copied, and the file is never written."""
        return self._view(self._spill_offset, self._spill_stop)

    def to_json_dict(self) -> dict[str, Any]:
        """Describe the chunk by its file and the span of it this view covers, not by its samples.

        Returns:
            A JSON-serializable dictionary :meth:`from_json_dict` reads back.
        """
        return {
            "__type__": SPILLED_TYPE,
            "path": str(self._spill_path),
            "offset": self._spill_offset,
            "stop": self._spill_stop,
            "start_time": self.start_time.value,
            "start_time_unit": str(self.start_time.unit),
            "sampling_frequency": self.sampling_frequency.value,
            "sampling_frequency_unit": str(self.sampling_frequency.unit),
            "metadata": self.metadata,
            "channels": _channel_identities(self),
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, Any]) -> SpilledTimeSeries:
        """Reopen a chunk a checkpoint recorded with :meth:`to_json_dict`.

        Args:
            json_dict: The recorded description.

        Returns:
            The chunk, mapping the same samples of the same file.

        Raises:
            FileNotFoundError: If the file is gone. A checkpoint references only files the run has not
                deleted, so this means something outside the run removed it.
        """
        path = Path(json_dict["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"The checkpoint carries a signal tail spilled to {path}, which no longer exists. "
                "The run removes spilled tails only once no checkpoint needs them, so it was removed "
                "from outside the run; the run cannot resume and has to start again."
            )
        return cls._open(
            path,
            int(json_dict["offset"]),
            int(json_dict["stop"]),
            Quantity(json_dict["start_time"], unit=json_dict["start_time_unit"]),
            Quantity(json_dict["sampling_frequency"], unit=json_dict["sampling_frequency_unit"]),
            json_dict.get("metadata") or {},
            json_dict.get("channels") or [],
        )
