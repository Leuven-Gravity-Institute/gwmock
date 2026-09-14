"""Module for handling time series data for multiple channels."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwpy.types.index import Index
from scipy.interpolate import interp1d

from gwmock.data.serialize.serializable import JSONSerializable
from gwmock.data.time_series.inject import alignment_tolerance, inject, is_aligned, measure_content_before

logger = logging.getLogger("gwmock")


if TYPE_CHECKING:
    from gwmock.data.time_series.time_series_list import TimeSeriesList


class TimeSeries(JSONSerializable):
    """Class representing a time series data for multiple channels."""

    __hash__ = None

    def __init__(self, data: np.ndarray, start_time: float | Quantity, sampling_frequency: float | Quantity):
        """Initialize the TimeSeries with a list of GWPy TimeSeries objects.

        Args:
            data: 2D numpy array of shape (num_of_channels, num_samples) containing the time series data.
            start_time: Start time of the time series in GPS seconds.
            sampling_frequency: Sampling frequency of the time series in Hz.
        """
        expected_ndim = 2
        if data.ndim != expected_ndim:
            raise ValueError("Data must be a 2D numpy array with shape (num_of_channels, num_samples).")

        if isinstance(start_time, Number):
            start_time = Quantity(start_time, unit="s")
        if isinstance(sampling_frequency, (int, float)):
            sampling_frequency = Quantity(sampling_frequency, unit="Hz")

        self._data: list[GWpyTimeSeries] = [
            GWpyTimeSeries(
                data=data[i],
                t0=start_time,
                sample_rate=sampling_frequency,
            )
            for i in range(data.shape[0])
        ]
        self.num_of_channels = data.shape[0]
        self.dtype = data.dtype
        self.metadata = {}

    def __len__(self) -> int:
        """Get the number of channels in the time series.

        Returns:
            Number of channels in the time series.
        """
        return self.num_of_channels

    def __getitem__(self, index: int) -> GWpyTimeSeries:
        """Get the GWPy TimeSeries object for a specific channel.

        Args:
            index: Index of the channel to retrieve.

        Returns:
            GWPy TimeSeries object for the specified channel.
        """
        return self._data[index]

    def __setitem__(self, index: int, value: GWpyTimeSeries) -> None:
        """Set the GWPy TimeSeries object for a specific channel.

        Args:
            index: Index of the channel to set.
            value: GWPy TimeSeries object to set for the specified channel.
        """
        # First check whether the start time and sampling frequency match
        if value.t0 != self.start_time:
            raise ValueError(
                "Start time of the provided TimeSeries does not match."
                f"The start time of this instance is {self.start_time}, "
                f"while that of the provided TimeSeries is {value.t0}."
            )

        # Debug: log the sampling frequencies
        logger.debug(
            "Assigning to channel %d: value.sample_rate=%.15f, self.sampling_frequency=%.15f",
            index,
            float(value.sample_rate.value),
            float(self.sampling_frequency.value),
        )

        if value.sample_rate != self.sampling_frequency:
            # Additional debug info
            logger.warning(
                "Sampling frequency mismatch on channel %d. "
                "Difference: %.15e Hz. "
                "Value times: %s to %s (%d samples, dt=%.15f). "
                "Self times span should match.",
                index,
                float(value.sample_rate.value) - float(self.sampling_frequency.value),
                value.times[0],
                value.times[-1],
                len(value),
                float(value.dt.value),
            )
            raise ValueError(
                "Sampling frequency of the provided TimeSeries does not match."
                f"The sampling frequency of this instance is {self.sampling_frequency}, "
                f"while that of the provided TimeSeries is {value.sample_rate}."
            )
        # Check the duration
        if value.duration != self.duration:
            raise ValueError(
                "Duration of the provided TimeSeries does not match."
                f"The duration of this instance is {self.duration}, "
                f"while that of the provided TimeSeries is {value.duration}."
            )

        if not isinstance(value, GWpyTimeSeries):
            raise TypeError(f"Value must be a GWpy TimeSeries instance, got {type(value)}")

        self._data[index] = value

    def __iter__(self):
        """Iterate over the channels in the time series.

        Returns:
            Iterator over the GWPy TimeSeries objects in the time series.
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
        for i in range(self.num_of_channels):
            if not np.array_equal(self[i].value, other[i].value):
                return False
            if self[i].t0 != other[i].t0:
                return False
            if self[i].sample_rate != other[i].sample_rate:
                return False
        return True

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the time series data.

        Returns:
            Tuple representing the shape of the time series data (num_of_channels, num_samples).
        """
        return (self.num_of_channels, self[0].size)

    @property
    def start_time(self) -> Quantity:
        """Get the start time of the time series.

        Returns:
            Start time of the time series.
        """
        return Quantity(self._data[0].t0)

    @property
    def duration(self) -> Quantity:
        """Get the duration of the time series.

        Returns:
            Duration of the time series.
        """
        return Quantity(self._data[0].duration)

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
            Sampling frequency of the time series.
        """
        return Quantity(self._data[0].sample_rate)

    @property
    def time_array(self) -> Index:
        """Get the time array of the time series.

        Returns:
            Time array of the time series.
        """
        return self[0].times

    def crop(
        self,
        start_time: Quantity | None = None,
        end_time: Quantity | None = None,
    ) -> TimeSeries:
        """Crop the time series to the specified start and end times.

        Args:
            start_time: Start time of the cropped segment in GPS seconds. If None, use the
                original start time.
            end_time: End time of the cropped segment in GPS seconds. If None, use the
                original end time.

        Returns:
            Cropped TimeSeries instance.
        """
        for i in range(self.num_of_channels):
            self._data[i] = GWpyTimeSeries(self._data[i].crop(start=start_time, end=end_time, copy=True))
        return self

    def inject(self, other: TimeSeries, preceding_gap: tuple[float, float] | None = None) -> TimeSeries | None:
        """Inject another TimeSeries into the current TimeSeries.

        Args:
            other: TimeSeries instance to inject.
            preceding_gap: ``(gap_start, gap_end)`` of a configured segment gap immediately before
                this segment, when there is one. Injection places by GPS time either way, so this
                changes no samples; it only tells :meth:`_report_content_before_segment` which of
                two very different losses it is looking at. Content dropped inside a configured gap
                is the run doing what it was asked; content dropped before a segment for any other
                reason is a placement failure.

        Returns:
            Remaining TimeSeries instance if the injected TimeSeries extends beyond the current
            TimeSeries end time, otherwise None.
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

        # Kept because the interpolation below rebinds `other` to samples drawn from this segment's
        # own time array, which by construction cannot extend past `self.end_time`. The overflow has
        # to be measured against what the caller actually passed, or a chunk crossing the segment
        # boundary loses its tail -- and `TimeSeriesMixin.simulate` relies on that tail being
        # returned to carry the rest of the signal into the next segment.
        supplied = other

        # Check whether there is any offset in times
        other_start_time = other.start_time.to(self.start_time.unit)
        idx = ((other_start_time - self.start_time) * self.sampling_frequency).value
        tolerance = alignment_tolerance(
            self.start_time.value,
            self.sampling_frequency.value,
            gps_times=(other_start_time.value, other.end_time.to(self.start_time.unit).value),
        )
        if not is_aligned(idx, tolerance):
            logger.warning("Chunk time grid does not align with segment time grid.")
            logger.warning("Interpolation will be used to align the chunk to the segment grid.")

            other_end_time = other.end_time.to(self.start_time.unit)
            other_new_times = self.time_array.value[
                (self.time_array.value >= other_start_time.value) & (self.time_array.value <= other_end_time.value)
            ]

            other = TimeSeries(
                data=np.array(
                    [
                        interp1d(
                            other.time_array.value, other[i].value, kind="linear", bounds_error=False, fill_value=0.0
                        )(other_new_times)
                        for i in range(len(other))
                    ]
                ),
                start_time=Quantity(other_new_times[0], unit=self.start_time.unit),
                sampling_frequency=self.sampling_frequency,
            )

        for i in range(self.num_of_channels):
            self[i] = inject(self[i], other[i])

        # The tail comes from the supplied chunk, unresampled, so the next segment interpolates it
        # against its own grid rather than inheriting this segment's resampling.
        #
        # Cropped from a copy: `crop` rewrites `_data` in place and returns `self`, so cropping the
        # supplied chunk directly would truncate the caller's own object and hand it back as the
        # remainder. `inject_from_list` walks a caller-provided list, so that mutates its elements.
        if supplied.end_time > self.end_time:
            tail = TimeSeries(
                data=np.asarray(supplied).copy(),
                start_time=supplied.start_time,
                sampling_frequency=supplied.sampling_frequency,
            )
            # Carry the wrapper metadata and each channel's identity across. A tail is the same
            # signal continuing into the next segment, so dropping these would strip
            # `injection_parameters` -- and the channel name and unit -- from any injection long
            # enough to cross a boundary, which is precisely the long-inspiral case.
            tail.metadata.update(dict(supplied.metadata))
            for index in range(min(tail.num_of_channels, supplied.num_of_channels)):
                source, target = supplied[index], tail[index]
                target.name = source.name
                target.channel = source.channel
                # `unit` is read-only on a gwpy array; override_unit is the supported way to set it.
                target.override_unit(source.unit)
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

    def contributes_samples(self, other: TimeSeries) -> bool:
        """Whether injecting *other* into this segment would place at least one sample.

        Provenance needs this, and it cannot be recovered after the fact: injection sums chunks into
        shared channels, so once a segment is built there is no way to ask which signal put samples
        where. It has to be asked before.

        Deliberately a *time* predicate rather than a look at the data. A chunk of genuine zeros --
        a signal whose amplitude is below the sample quantum, or a segment covering only its tapered
        edge -- still means the signal is present in that frame, which is what a provenance record
        is claiming. Reading the samples would call that absent.

        The two endpoints are exclusive because a chunk ending exactly at ``self.start_time`` has no
        sample inside this segment: the segment's first sample is *at* that time and the chunk's last
        sample is one interval before it. This is tested across the boundary rather than argued, in
        ``test_provenance_across_segments.py``, because an off-by-one here silently over- or
        under-reports one frame per signal.

        Args:
            other: The chunk that would be injected.

        Returns:
            ``True`` if the chunk overlaps this segment's sampled span.
        """
        unit = self.start_time.unit
        chunk_start = float(other.start_time.to(unit).value)
        chunk_end = float(other.end_time.to(unit).value)
        return chunk_start < float(self.end_time.to(unit).value) and chunk_end > float(self.start_time.value)

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
            "data": np.asarray([self[i].value for i in range(self.num_of_channels)]),
            # Carried because a serialized chunk is a *spillover tail* waiting in a checkpoint, and
            # the next segment reads exactly these on restore. Omitting them restores the samples
            # while silently dropping `injection_parameters` and `event_id` -- so the run continues
            # with data that no longer says which signal it is -- and drops the channel identity
            # that `inject` copies onto a tail for the same reason.
            "metadata": dict(self.metadata),
            "channels": [
                {
                    "name": self[i].name,
                    "channel": None if self[i].channel is None else str(self[i].channel),
                    "unit": str(self[i].unit),
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
            target = series[index]
            target.name = channel.get("name")
            target.channel = channel.get("channel")
            unit = channel.get("unit")
            if unit is not None:
                # `unit` is read-only on a gwpy array; override_unit is the supported way to set it.
                target.override_unit(unit)
        return series
