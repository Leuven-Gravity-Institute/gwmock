"""Module for handling time series data for multiple channels."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
from astropy.units import Quantity
from gwmock_data import TimeSeries as ChannelSeries
from gwmock_data import as_timeseries
from gwmock_signal.projection.resampling import resample_uniform_sinc

from gwmock.data.serialize.serializable import JSONSerializable
from gwmock.data.time_series.inject import alignment_tolerance, inject, is_aligned, measure_content_before

logger = logging.getLogger("gwmock")


if TYPE_CHECKING:
    from gwmock.data.time_series.time_series_list import TimeSeriesList


def _in_unit(value: float | Quantity, unit: str) -> float:
    """Return *value* as a float in *unit*; a plain number is taken to be in that unit already."""
    if isinstance(value, Quantity):
        return float(value.to_value(unit))
    if isinstance(value, Number):
        return float(value)
    raise TypeError(f"Expected a number or an astropy Quantity in {unit}; got {type(value).__name__}.")


class TimeSeries(JSONSerializable):
    """Time series data for several channels on one sample grid.

    Each channel is a :class:`gwmock_data.TimeSeries`, the series the gwmock packages share; indexing
    returns one. The channel names used when the data is written (``"H1:STRAIN"`` and the like) are
    kept here, in :attr:`channels`, since the shared series has a name and a unit but no channel.

    Timing is reported as astropy quantities (seconds, hertz), as before.
    """

    __hash__ = None

    def __init__(self, data: np.ndarray, start_time: float | Quantity, sampling_frequency: float | Quantity):
        """Initialize the TimeSeries.

        Args:
            data: 2D numpy array of shape (num_of_channels, num_samples) containing the time series data.
            start_time: Start time of the time series in GPS seconds.
            sampling_frequency: Sampling frequency of the time series in Hz.
        """
        expected_ndim = 2
        if data.ndim != expected_ndim:
            raise ValueError("Data must be a 2D numpy array with shape (num_of_channels, num_samples).")

        t0 = _in_unit(start_time, "s")
        sample_rate = _in_unit(sampling_frequency, "Hz")
        self._data: list[ChannelSeries] = [
            ChannelSeries(data[i], t0=t0, sample_rate=sample_rate) for i in range(data.shape[0])
        ]
        #: The channel each row is written as, or ``None``; one entry per channel.
        self.channels: list[str | None] = [None] * data.shape[0]
        self.num_of_channels = data.shape[0]
        self.dtype = data.dtype
        self.metadata = {}

    def __len__(self) -> int:
        """Get the number of channels in the time series.

        Returns:
            Number of channels in the time series.
        """
        return self.num_of_channels

    def __getitem__(self, index: int) -> ChannelSeries:
        """Get one channel.

        Args:
            index: Index of the channel to retrieve.

        Returns:
            The channel's series.
        """
        return self._data[index]

    def __setitem__(self, index: int, value: ChannelSeries) -> None:
        """Replace one channel.

        Args:
            index: Index of the channel to set.
            value: The new series for the channel: a :class:`gwmock_data.TimeSeries`, or a GWpy
                ``TimeSeries``, which is converted. It must have this series' start time, sampling
                frequency and length.
        """
        value = as_timeseries(value)
        if value.t0 != self[index].t0:
            raise ValueError(
                "Start time of the provided TimeSeries does not match."
                f"The start time of this instance is {self.start_time}, "
                f"while that of the provided TimeSeries is {value.t0} s."
            )
        if value.sample_rate != self[index].sample_rate:
            logger.warning(
                "Sampling frequency mismatch on channel %d. Difference: %.15e Hz.",
                index,
                value.sample_rate - self[index].sample_rate,
            )
            raise ValueError(
                "Sampling frequency of the provided TimeSeries does not match."
                f"The sampling frequency of this instance is {self.sampling_frequency}, "
                f"while that of the provided TimeSeries is {value.sample_rate} Hz."
            )
        if value.n_samples != self[index].n_samples:
            raise ValueError(
                "Duration of the provided TimeSeries does not match."
                f"The duration of this instance is {self.duration}, "
                f"while that of the provided TimeSeries is {value.duration} s."
            )
        self._data[index] = value

    def __iter__(self):
        """Iterate over the channels in the time series.

        Returns:
            Iterator over the channels' series.
        """
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        """Check equality with another TimeSeries object.

        Args:
            other: Another TimeSeries object to compare with.

        Returns:
            True if the two TimeSeries objects are equal, False otherwise.
        """
        if not isinstance(other, TimeSeries):
            return False
        if self.num_of_channels != other.num_of_channels:
            return False
        for mine, theirs in zip(self, other, strict=True):
            if not np.array_equal(mine.data, theirs.data):
                return False
            if mine.t0 != theirs.t0 or mine.sample_rate != theirs.sample_rate:
                return False
        return True

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the time series data.

        Returns:
            Tuple representing the shape of the time series data (num_of_channels, num_samples).
        """
        return (self.num_of_channels, self[0].n_samples)

    @property
    def start_time(self) -> Quantity:
        """Get the start time of the time series.

        Returns:
            Start time of the time series, in seconds.
        """
        return Quantity(self._data[0].t0, unit="s")

    @property
    def duration(self) -> Quantity:
        """Get the duration of the time series.

        Returns:
            Duration of the time series, in seconds: the number of samples times the sample spacing.
        """
        return Quantity(self._data[0].n_samples * (1.0 / self._data[0].sample_rate), unit="s")

    @property
    def end_time(self) -> Quantity:
        """Get the end time of the time series.

        Returns:
            End time of the time series.
        """
        end_time: Quantity = self.start_time + self.duration
        return end_time

    @property
    def sampling_frequency(self) -> Quantity:
        """Get the sampling frequency of the time series.

        Returns:
            Sampling frequency of the time series, in hertz.
        """
        return Quantity(self._data[0].sample_rate, unit="Hz")

    @property
    def time_array(self) -> Quantity:
        """Get the time array of the time series.

        Returns:
            GPS time of every sample, in seconds.
        """
        return Quantity(self[0].times, unit="s")

    def crop(
        self,
        start_time: Quantity | float | None = None,
        end_time: Quantity | float | None = None,
    ) -> TimeSeries:
        """Crop the time series, in place, to the samples at GPS times in ``[start_time, end_time)``.

        A bound that falls on a sample, to within float64 resolution at the GPS epoch, counts as on
        it, so cropping at a sample's own time keeps that sample (see ``gwmock_data.TimeSeries.crop``).

        Args:
            start_time: Start time of the cropped segment in GPS seconds. If None, use the
                original start time.
            end_time: End time of the cropped segment in GPS seconds. If None, use the
                original end time.

        Returns:
            This TimeSeries, cropped.
        """
        start = None if start_time is None else _in_unit(start_time, "s")
        end = None if end_time is None else _in_unit(end_time, "s")
        self._data = [series.crop(start, end).copy() for series in self._data]
        return self

    def inject(self, other: TimeSeries, preceding_gap: tuple[float, float] | None = None) -> TimeSeries | None:
        """Inject another TimeSeries into the current TimeSeries.

        A chunk whose samples fall on this segment's grid is added sample for sample. A chunk that
        sits a fraction of a sample off the grid is first resampled, as a whole, onto this segment's
        grid -- extended past either end of the segment -- with the band-limited windowed-sinc kernel
        ``gwmock_signal.projection.resampling.resample_uniform_sinc``, and a warning is logged. Only
        grid points within the chunk's own sample span are produced; nothing is extrapolated beyond
        its first or last sample.

        Resampling the whole chunk rather than the part inside this segment is what keeps a signal
        continuous across segment boundaries: the kernel needs about half its taps of context on
        either side, which a per-segment resample would cut at every boundary. It also means the
        returned remainder of an off-grid chunk is the *resampled* series, already on this segment's
        grid, so a contiguous next segment adds it without resampling it a second time.

        This replaces linear interpolation, which at 0.8 x Nyquist was off by ~0.63 of the signal's
        peak; the sinc kernel's error there is ~1e-12. Off-grid injections therefore produce
        different samples from earlier releases, most visibly at high frequency.

        Agreement with gwmock-signal: this is the kernel gwmock-signal's projection resamples with,
        but ``gwmock_signal.injection.core.inject_strain`` in gwmock-signal 0.17.3 -- the minimum
        this package requires -- still interpolates cubically. The divergence is deliberate and
        temporary: the kernel module is byte-identical between gwmock-signal 0.17.3 and the pending
        gwmock-signal change that moves ``inject_strain`` onto it, so the two paths become equal
        once that change is released and required here, without any further change on this side.

        Args:
            other: TimeSeries instance to inject.
            preceding_gap: ``(gap_start, gap_end)`` of a configured segment gap immediately before
                this segment, when there is one. Injection places by GPS time either way, so this
                changes no samples; it only tells :meth:`_report_content_before_segment` which of
                two very different losses it is looking at. Content dropped inside a configured gap
                is the run doing what it was asked; content dropped before a segment for any other
                reason is a placement failure.

        Returns:
            What this segment could not place, for the next segment. *other* itself, unchanged, if
            it ends before this segment starts or starts after it ends, in which case nothing is
            injected. Otherwise the part of the chunk past this segment's end, if it extends beyond
            it; for an off-grid chunk that is the resampled series, on this segment's grid, carrying
            *other*'s metadata, channel names and units. ``None`` if the chunk ends within this
            segment, or if it is an off-grid chunk whose sample span contains no grid point at all
            (a single sample between two grid points), in which case nothing is injected and a
            warning is logged.
        """
        if len(other) != len(self):
            raise ValueError(
                f"Number of channels of other ({other.num_of_channels}) must "
                f"match number of channels of self ({self.num_of_channels})."
            )

        # Enforce that other has the same sampling frequency as self
        if not other.sampling_frequency == self.sampling_frequency:
            raise ValueError(
                f"Sampling frequency of chunk ({other.sampling_frequency}) must match "
                f"sampling frequency of segment ({self.sampling_frequency}). "
                "This ensures time grid alignment and avoids rounding errors."
            )

        # Before the early returns below, not after. A chunk lying entirely before this segment is
        # handed back as a remainder and then dropped by `TimeSeriesMixin.simulate`, so it is
        # discarded just as surely as a partially-early one -- and reporting only the overlapping
        # case would leave the larger loss the quieter of the two.
        self._report_content_before_segment(other, preceding_gap)

        if other.end_time < self.start_time:
            logger.warning(
                "The time series to inject ends before the current time series starts. No injection performed."
                "The start time of this segment is %s, while the end time of the other segment is %s",
                self.start_time,
                other.end_time,
            )
            return other

        if other.start_time > self.end_time:
            logger.warning(
                "The time series to inject starts after the current time series ends. No injection performed."
                "The end time of this segment is %s, while the start time of the other segment is %s",
                self.end_time,
                other.start_time,
            )
            return other

        # Kept for the metadata and channel identity the tail carries forward: the interpolation below
        # rebinds `other` to a new series on this segment's lattice, which has neither.
        supplied = other

        idx, aligned, first, last = self._grid_span(other)
        if not aligned:
            logger.warning("Chunk time grid does not align with segment time grid.")
            logger.warning("The chunk will be resampled onto the segment grid with the windowed-sinc kernel.")

            # The whole chunk is resampled once onto this segment's lattice -- extended past either
            # end -- rather than only the part inside the segment. The windowed-sinc kernel needs
            # ~half its taps of context on each side, so resampling segment by segment would cut
            # that context at every boundary: measured errors there reached the signal's own peak,
            # where resampling the whole chunk leaves none. The tail returned below is then already
            # on the lattice, so a contiguous next segment places it without resampling it again.
            #
            # Same kernel, and the same integer-index-minus-one-offset positions, as `inject` and as
            # the pending gwmock-signal change to `inject_strain`. The released `inject_strain` in
            # gwmock-signal 0.17.3 still interpolates cubically, so the two packages agree on this
            # operation only once that change is released and required; the kernel module itself is
            # byte-identical between the two, so nothing here has to change when it is.
            if first > last:
                logger.warning("Chunk spans no sample of the segment grid. No injection performed.")
                return None
            positions = np.arange(first, last + 1) - idx
            other = TimeSeries(
                data=np.array([resample_uniform_sinc(other[i].value, positions) for i in range(len(other))]),
                start_time=Quantity(
                    self.start_time.value + first / self.sampling_frequency.value, unit=self.start_time.unit
                ),
                sampling_frequency=self.sampling_frequency,
            )

        for i in range(self.num_of_channels):
            self[i] = inject(self[i], other[i])

        # The tail is the resampled chunk when interpolation was needed, so it is already on this
        # segment's lattice and the next segment -- which continues that lattice -- places it as is.
        #
        # Cropped from a copy: `crop` rewrites `_data` in place and returns `self`, so cropping the
        # supplied chunk directly would truncate the caller's own object and hand it back as the
        # remainder. `inject_from_list` walks a caller-provided list, so that mutates its elements.
        if other.end_time > self.end_time:
            tail = TimeSeries(
                data=np.array([np.asarray(series.data) for series in other]),
                start_time=other.start_time,
                sampling_frequency=other.sampling_frequency,
            )
            # Carry the wrapper metadata and each channel's identity across. A tail is the same
            # signal continuing into the next segment, so dropping these would strip
            # `injection_parameters` -- and the channel name and unit -- from any injection long
            # enough to cross a boundary, which is precisely the long-inspiral case.
            tail.metadata.update(dict(supplied.metadata))
            for index in range(min(tail.num_of_channels, supplied.num_of_channels)):
                source = supplied[index]
                tail._data[index] = dataclasses.replace(tail[index], name=source.name, unit=source.unit)
                tail.channels[index] = supplied.channels[index]
            return tail.crop(start_time=self.end_time)
        return None

    def _report_content_before_segment(
        self, chunk: TimeSeries, preceding_gap: tuple[float, float] | None = None
    ) -> None:
        """Warn that content starting before this segment is about to be dropped.

        Reaching this is now the exception rather than the rule. Segments claim an event by where its
        waveform *starts*, so an ordinary compact binary is generated early enough for the whole
        buffer to be placed. What still arrives here is the cases that rule cannot cover:

        - a waveform beginning before the run's own start, which has no segment to go in;
        - a backend that cannot say how long before coalescence its buffer starts, or an odd
          catalogue the query cannot be answered for -- both fall back to claiming by ``coa_time``,
          which is the behaviour that crops.

        Still reported rather than fixed, for the same reason as before: placing it would mean
        writing into segments already on disk. The loss is real either way, and a truncated inspiral
        looks like a perfectly ordinary signal, so saying how much went -- per signal -- is what lets
        a run be judged.
        """
        samples, seconds, energy_fraction = measure_content_before(
            float(self.start_time.to(chunk.start_time.unit).value),
            float(self.sampling_frequency.value),
            chunk,
        )
        if samples <= 0:
            return

        # A configured gap is not a placement failure, and the message below would report it as
        # one -- telling a reader to look for a warning about an unavailable pre-coalescence
        # duration that will never appear, for a run behaving exactly as configured. The caller
        # records the loss with the gap it fell into; this only has to stop mislabelling it.
        if preceding_gap is not None and float(chunk.start_time.to(self.start_time.unit).value) >= preceding_gap[0]:
            logger.debug(
                "Dropping %.3f s (%d samples) of a chunk into the configured gap [%s, %s) before this segment.",
                seconds,
                samples,
                preceding_gap[0],
                preceding_gap[1],
            )
            return

        coa_time = (chunk.metadata.get("injection_parameters") or {}).get("coa_time")
        logger.warning(
            "Discarding %.3f s (%d samples, %.2f%% of its unweighted strain-squared energy) of the "
            "signal with coa_time %s: it "
            "starts at %s, before this segment begins at %s. The earlier segments it belongs to are "
            "already written, so this content cannot be placed. Segments normally claim an event by "
            "where its waveform starts, which avoids this; reaching it means either the waveform "
            "begins before the run itself, or the pre-coalescence duration could not be established "
            "and placement fell back to coa_time -- look for a warning saying so.",
            seconds,
            samples,
            100.0 * energy_fraction,
            "unknown" if coa_time is None else coa_time,
            chunk.start_time,
            self.start_time,
        )

    def _grid_span(self, other: TimeSeries) -> tuple[float, bool, int, int]:
        """Locate *other* on this segment's sample grid.

        Shared by :meth:`inject` and :meth:`contributes_samples`, so that what a segment claims to
        receive and what it actually places are the same computation.

        Args:
            other: The chunk to locate.

        Returns:
            ``(offset, aligned, first, last)``: the chunk's first sample as a fractional index on
            this grid (rounded to a whole index when aligned), whether it is aligned within
            :func:`alignment_tolerance`, and the first and last grid indices lying within the
            chunk's sample span -- not clipped to this segment, and ``first > last`` when the span
            holds no grid point.
        """
        unit = self.start_time.unit
        other_start_time = other.start_time.to(unit)
        offset = ((other_start_time - self.start_time) * self.sampling_frequency).value
        tolerance = alignment_tolerance(
            self.start_time.value,
            self.sampling_frequency.value,
            gps_times=(other_start_time.value, other.end_time.to(unit).value),
        )
        aligned = is_aligned(offset, tolerance)
        if aligned:
            offset = float(round(offset))
        return offset, aligned, math.ceil(offset), math.floor(offset + len(other.time_array) - 1)

    def contributes_samples(self, other: TimeSeries) -> bool:
        """Whether injecting *other* into this segment would place at least one sample.

        Provenance needs this, and it cannot be recovered after the fact: injection sums chunks into
        shared channels, so once a segment is built there is no way to ask which signal put samples
        where. It has to be asked before.

        Deliberately a *time* predicate rather than a look at the data. A chunk of genuine zeros --
        a signal whose amplitude is below the sample quantum, or a segment covering only its tapered
        edge -- still means the signal is present in that frame, which is what a provenance record
        is claiming. Reading the samples would call that absent.

        Answered on the sample grid, with the same computation :meth:`inject` places by: a chunk
        contributes when some grid point of this segment lies within its sample span. A chunk ending
        exactly at ``self.start_time`` therefore does not -- its last sample is one interval before
        the segment's first -- and neither does an off-grid chunk overlapping an edge by less than
        a sample, which places nothing here because no grid point of this segment falls inside it.
        This is tested across the boundary rather than argued, in
        ``test_provenance_across_segments.py``, because an off-by-one here silently over- or
        under-reports one frame per signal.

        Args:
            other: The chunk that would be injected.

        Returns:
            ``True`` if injecting the chunk would place at least one sample in this segment.
        """
        _, _, first, last = self._grid_span(other)
        return max(first, 0) <= min(last, len(self.time_array) - 1)

    def inject_from_list(
        self, ts_iterable: Iterable[TimeSeries], preceding_gap: tuple[float, float] | None = None
    ) -> TimeSeriesList:
        """Inject multiple TimeSeries from an iterable into the current TimeSeries.

        Args:
            ts_iterable: Iterable of TimeSeries instances to inject.
            preceding_gap: A configured segment gap immediately before this segment; see
                :meth:`inject`.

        Returns:
            TimeSeriesList of remaining TimeSeries instances that extend beyond the current TimeSeries end time.
        """
        from gwmock.data.time_series.time_series_list import TimeSeriesList  # noqa: PLC0415

        remaining_ts: list[TimeSeries] = []
        for ts in ts_iterable:
            remaining_chunk = self.inject(ts, preceding_gap=preceding_gap)
            if remaining_chunk is not None:
                remaining_ts.append(remaining_chunk)
        return TimeSeriesList(remaining_ts)

    def to_json_dict(self) -> dict:
        """Convert the TimeSeries to a JSON-serializable dictionary.

        Assume the unit

        Returns:
            JSON-serializable dictionary representation of the TimeSeries.
        """
        return {
            "__type__": "TimeSeries",
            # The raw array, not `tolist()`. The encoder base64s an ndarray and writes JSON text
            # numbers for a list, and that is not cosmetic once spillover chunks go into a
            # checkpoint written after every batch. Measured at 100 s x 3 detectors, 4096 Hz:
            # 44.1 MB and 9.02 s as lists (35.9 bytes/sample) against 13.1 MB and 0.43 s as base64
            # (10.7 bytes/sample). At a realistic 1000 s binary-neutron-star tail, measured rather
            # than extrapolated: 131 MB and 1.1 s, which is 1.33x the raw float64 bytes.
            #
            # It also fixes a loss the list form had: `tolist()` went through Python floats and
            # widened float32 to float64 on the way back. Base64 carries the dtype, so a float32
            # chunk returns float32 -- checked for both widths rather than assumed, because the
            # comment here previously claimed the opposite and was wrong.
            "data": np.asarray([np.asarray(self[i].data) for i in range(self.num_of_channels)]),
            # Carried because a serialized chunk is a *spillover tail* waiting in a checkpoint, and
            # the next segment reads exactly these on restore. Omitting them restores the samples
            # while silently dropping `injection_parameters` and `event_id` -- so the run continues
            # with data that no longer says which signal it is -- and drops the channel identity
            # that `inject` copies onto a tail for the same reason.
            "metadata": dict(self.metadata),
            "channels": [
                {
                    "name": self[i].name,
                    "channel": self.channels[i],
                    # An empty string for no unit, as GWpy wrote a dimensionless unit, so records
                    # written before and after read back alike.
                    "unit": "" if self[i].unit is None else str(self[i].unit),
                }
                for i in range(self.num_of_channels)
            ],
            "start_time": self.start_time.value,
            "start_time_unit": str(self.start_time.unit),
            "sampling_frequency": self.sampling_frequency.value,
            "sampling_frequency_unit": str(self.sampling_frequency.unit),
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> TimeSeries:
        """Create a TimeSeries object from a JSON-serializable dictionary.

        Args:
            json_dict: JSON-serializable dictionary representation of the TimeSeries.

        Returns:
            TimeSeries: An instance of the TimeSeries class created from the dictionary.
        """
        data = np.array(json_dict["data"])
        start_time = Quantity(json_dict["start_time"], unit=json_dict["start_time_unit"])
        sampling_frequency = Quantity(json_dict["sampling_frequency"], unit=json_dict["sampling_frequency_unit"])
        series = cls(data=data, start_time=start_time, sampling_frequency=sampling_frequency)

        # Both keys are absent from records written before this was carried, so both default rather
        # than raise: a checkpoint is read by whatever version happens to resume the run, and
        # refusing an older one would turn an upgrade mid-run into a lost run.
        series.metadata.update(json_dict.get("metadata") or {})
        for index, channel in enumerate(json_dict.get("channels") or []):
            if index >= series.num_of_channels:
                break
            unit = channel.get("unit") or None
            series._data[index] = dataclasses.replace(series[index], name=channel.get("name"), unit=unit)
            series.channels[index] = channel.get("channel")
        return series
