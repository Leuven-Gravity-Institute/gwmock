"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.units.quantity import Quantity
from gwpy.timeseries import TimeSeries as GWPyTimeSeries

from gwmock.cli.utils.config_resolution import parse_seconds
from gwmock.cli.utils.segment_layout import SegmentLayout, resolve_segment_count
from gwmock.cli.utils.template import expand_template_variables
from gwmock.data.time_series.inject import measure_content_before
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.simulator.state import StateAttribute

logger = logging.getLogger("gwmock")


def _injection_record(chunk: TimeSeries) -> dict[str, Any] | None:
    """Return the injection record a chunk carries, or ``None`` if it carries none.

    Both are stamped at generation and copied onto a tail when a chunk crosses a segment boundary,
    which is what lets a carried-forward chunk still say what it is. A chunk with parameters but no
    ``event_id`` is recorded with ``event_id`` ``None`` rather than dropped: the parameters are still
    the provenance, and dropping it would silently lose a signal from the record.
    """
    parameters = chunk.metadata.get("injection_parameters")
    if parameters is None:
        return None
    event_id = chunk.metadata.get("event_id")
    return {"event_id": event_id, "parameters": dict(parameters)}


def _contributing_injections(segment: TimeSeries, chunks: Iterable[TimeSeries]) -> list[dict[str, Any]]:
    """Return injection records for the chunks that place at least one sample in *segment*."""
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if not segment.contributes_samples(chunk):
            continue
        record = _injection_record(chunk)
        if record is not None:
            records.append(record)
    return records


def _merge_injection_records(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate injection records, keeping the first of each ``event_id`` and order stable.

    Deduplicated because one signal can reach a segment through several chunks -- a multi-detector
    generation emits one chunk per detector, all carrying the same record -- and a provenance list
    naming the same event three times would read as three injections.

    Records whose ``event_id`` is ``None`` are never merged together: without an id there is nothing
    to say two chunks are the same signal, so collapsing them would drop real injections.
    """
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for group in groups:
        for record in group:
            event_id = record.get("event_id")
            if event_id is not None:
                if event_id in seen:
                    continue
                seen.add(event_id)
            merged.append(record)
    return merged


def _gap_discarded_records(
    segment: TimeSeries,
    chunks: Iterable[TimeSeries],
    preceding_gap: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    """Return one record per chunk whose content the gap before *segment* swallows.

    Only content lying inside the gap counts. A chunk reaching further back than the gap -- before
    the previous segment too -- is a different loss with a different cause, already reported by
    :meth:`~gwmock.data.time_series.time_series.TimeSeries._report_content_before_segment`, and is
    left to it rather than relabelled as a gap.

    Args:
        segment: The segment about to be built.
        chunks: The carried-forward chunks about to be injected into it.
        preceding_gap: ``(gap_start, gap_end)`` for the gap immediately before the segment, or
            ``None`` when none precedes it.

    Returns:
        One mapping per affected chunk, carrying its identity, the gap, and how much went.
    """
    if preceding_gap is None:
        return []
    gap_start, gap_end = preceding_gap
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        samples, seconds, energy_fraction = measure_content_before(
            float(segment.start_time.to(chunk.start_time.unit).value),
            float(segment.sampling_frequency.value),
            chunk,
        )
        if samples <= 0:
            continue
        if float(chunk.start_time.to(segment.start_time.unit).value) < gap_start:
            continue
        record = _injection_record(chunk) or {"event_id": chunk.metadata.get("event_id"), "parameters": {}}
        records.append(
            {
                **record,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "discarded_samples": samples,
                "discarded_seconds": seconds,
                "discarded_energy_fraction": energy_fraction,
            }
        )
    return records


class TimeSeriesMixin:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    start_time = StateAttribute(Quantity(0, unit="s"))
    #: Spillover: the part of a chunk that extends past the segment being built, waiting for the
    #: next one.
    #:
    #: **Not a `StateAttribute`, deliberately, and it is still persisted.** It used to be neither,
    #: and a resumed run started with none of it: the tail of any signal crossing the resume point
    #: was never placed and the following segment lost that content silently -- 7.280e-23 to exactly
    #: 0.0, the merger absent, with the frames before it bit-identical.
    #:
    #: The obvious fix, making this stateful, does not work: `state` is serialized into every *batch
    #: metadata record* as well as into the checkpoint, so it would write spillover samples into a
    #: provenance document meant to stay small and readable, and those records are dumped with plain
    #: `json`, which has no encoder for a `TimeSeriesList`. So it travels as its own checkpoint
    #: field, in the entry `simulator_tails` keeps for the simulator that produced it, and is handed
    #: back only to that simulator and only to the batch immediately after the one it came from.
    cached_data_chunks = TimeSeriesList()
    #: Injection records for every signal that reaches the segment currently being built, including
    #: signals generated for an *earlier* segment whose content extends into this one. Rebuilt per
    #: segment by :meth:`simulate`; a subclass writing provenance should union this with whatever it
    #: generated itself. Empty for simulators that do not inject.
    #:
    #: Survives a checkpoint resume, because the chunks carrying these records do: see
    #: ``cached_data_chunks`` above. It did not until the spillover was persisted -- a resumed run
    #: lost both the samples and their provenance at the resume boundary.
    carried_injections: list[dict[str, Any]]
    #: What the gap immediately before the current segment swallowed: one record per carried-forward
    #: chunk whose samples fell inside it. Rebuilt per segment by :meth:`simulate`, and empty for a
    #: contiguous run. The samples themselves are gone -- the run writes no frame covering a gap --
    #: so this is the only place that says a signal lost content there.
    gap_discarded: list[dict[str, Any]]

    def __init__(
        self,
        start_time: int = 0,
        duration: float = 4,
        total_duration: float | str | None = None,
        sampling_frequency: float = 4096,
        num_of_channels: int | None = None,
        dtype: type = np.float64,
        segment_gap: float | str = 0.0,
        **kwargs,
    ):
        """Initialize timing parameters.

        Args:
            start_time: Start time in GPS seconds. Default is 0.
            duration: Duration of simulation in seconds. Default is 4.
            total_duration: The run's GPS **span** -- first segment's start to last segment's end,
                gaps included. See :attr:`total_duration`.
            sampling_frequency: Sampling frequency in Hz. Default is 4096.
            dtype: Data type for the time series data. Default is np.float64.
            segment_gap: Seconds of GPS time between the end of one analysed segment and the start
                of the next. Default is 0, the contiguous layout. See :attr:`segment_gap`.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        # TimeSeriesMixin is the last mixin in the hierarchy, so no super().__init__() call needed
        self.start_time = Quantity(start_time, unit="s")
        #: The epoch of segment 0, which `start_time` no longer is once the run has advanced.
        #:
        #: Configuration rather than state, so it is a plain attribute: every resume re-instantiates
        #: the simulator from the same configured `start-time` and then restores `start_time` and
        #: `counter` over the top, which is exactly the pair this must not be derived from.
        #: :attr:`final_end_time` is measured from here, so the run's end stops moving forward one
        #: segment at a time as the run proceeds.
        self._run_start_time = Quantity(start_time, unit="s")
        # Per instance, not a class attribute: two simulators in one process must not share the
        # list of signals reaching the segment each is building.
        self.carried_injections = []
        #: What the gap preceding the current segment swallowed, rebuilt per segment by
        #: :meth:`simulate`. Empty for a contiguous run and for a segment no signal spilled into.
        self.gap_discarded = []
        self.duration = duration
        # Before `total_duration`, which needs the gap to turn a span into a segment count.
        self.segment_gap = segment_gap
        self.total_duration = total_duration
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype

        # Get the number of channels.
        if num_of_channels is not None:
            self.num_of_channels = num_of_channels
            if (
                "detectors" in kwargs
                and kwargs["detectors"] is not None
                and len(kwargs["detectors"]) != num_of_channels
            ):
                raise ValueError("Number of detectors does not match num_of_channels.")
        elif "detectors" in kwargs and kwargs["detectors"] is not None:
            self.num_of_channels = len(kwargs["detectors"])
        else:
            self.num_of_channels = 1

    @property
    def duration(self) -> Quantity:
        """Get the duration of each simulation segment.

        Returns:
            Duration in seconds.
        """
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the duration of each simulation segment.

        Args:
            value: Duration in seconds.
        """
        if value <= 0:
            raise ValueError("duration must be positive.")
        self._duration = Quantity(value, unit="s")

    @property
    def segment_gap(self) -> Quantity:
        """Seconds of GPS time between the end of one analysed segment and the start of the next.

        Zero -- the default -- gives the contiguous layout, where segment ``k`` starts at
        ``start_time + k * duration``. A positive gap makes segment ``k`` start at
        ``start_time + k * (duration + segment_gap)``, so the frames a run writes are
        **discontiguous in GPS**: the seconds inside a gap appear in no frame.

        **What this does to a stateful noise stream, because it is a real choice and this is
        where it is recorded.** ``gwmock_noise``'s colored-noise simulator interpolates its
        ``psd_schedule`` against the number of samples it has *produced*, not against GPS. gwmock
        therefore draws the gap's samples from the same stream and throws them away, so that the
        schedule's time axis stays locked to GPS: an anchor at ``gps_offset_seconds: 3600`` means
        one hour after the run's start time whether or not gaps fall in between, and a PSD drift is
        a drift in wall-clock time, which is what a drifting instrument does. The same choice keeps
        the glitch injector's Poisson processes running at their configured rate per unit of real
        time across a gap, and keeps the noise in the segment after a gap the continuation of a
        detector that never stopped rather than a fresh draw. The alternative -- skipping the gap,
        so the schedule tracks analysed livetime and drifts away from GPS by the accumulated gap --
        is cheaper and is *not* what this package does. Exactly how much cheaper: a run has no
        trailing gap, so it generates ``(count - 1) * segment_gap`` seconds it never writes against
        ``count * duration`` that it does, a ratio of
        ``(count - 1) * segment_gap / (count * duration)`` that tends to ``segment_gap / duration``
        only as the run grows long.

        Returns:
            The gap in seconds.
        """
        return self._segment_gap

    @segment_gap.setter
    def segment_gap(self, value: float | str) -> None:
        """Set the gap between consecutive analysed segments.

        Args:
            value: Gap in seconds, or a duration string such as ``"4 min"``.

        Raises:
            ValueError: If the gap is negative.
        """
        seconds = parse_seconds(value, "segment-gap")
        if seconds < 0:
            raise ValueError(f"segment-gap must be non-negative; got {seconds:g} s.")
        self._segment_gap = Quantity(seconds, unit="s")

    @property
    def segment_layout(self) -> SegmentLayout:
        """The GPS layout of this run: where every analysed segment starts, and every gap.

        Anchored at the configured start time rather than the current one, so it describes the
        whole run from any point in it -- including a run resumed halfway.

        An **unbounded** run (``max_samples`` infinite, which the orchestrator never produces --
        it always resolves a finite batch count) reports a count of 1. The stride arithmetic --
        :meth:`~gwmock.cli.utils.segment_layout.SegmentLayout.epoch` and
        :meth:`~gwmock.cli.utils.segment_layout.SegmentLayout.gap_before` -- does not depend on the
        count and stays correct; the predicates that *do* bound themselves by the run then answer
        "outside the run" for anything past the first segment, which errs towards claiming a signal
        rather than excluding one, and that is the safe direction.

        Returns:
            The layout of this run's analysed segments.
        """
        max_samples = self.max_samples
        count = 1 if max_samples is None or not np.isfinite(max_samples) else int(max_samples)
        return SegmentLayout(
            start_time=float(self._run_start_time.value),
            duration=float(self.duration.value),
            gap=float(self.segment_gap.value),
            count=count,
        )

    @property
    def sampling_frequency(self) -> Quantity:
        """Get the sampling frequency.

        Returns:
            Sampling frequency in Hz.
        """
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, value: float) -> None:
        """Set the sampling frequency.

        Args:
            value: Sampling frequency in Hz.
        """
        if value <= 0:
            raise ValueError("sampling_frequency must be positive.")
        self._sampling_frequency = Quantity(value, unit="Hz")

    @property
    def total_duration(self) -> Quantity:
        """The run's GPS **span**: the first segment's start to the last segment's end.

        Span, not analysed livetime, and the distinction only exists once ``segment_gap`` is
        non-zero. ``total_duration == count * duration + (count - 1) * segment_gap``: the gaps
        *between* segments are inside the span, and there is no trailing gap, because a run stops
        at the end of data rather than after a hole nobody wrote. :attr:`analysed_livetime` is the
        other quantity -- ``count * duration`` -- and the two are equal exactly when the run is
        contiguous.

        Choosing span is what keeps :attr:`final_end_time` meaning "where the run's data ends",
        which is what the cached-chunk sweep in :meth:`simulate` measures against.

        Returns:
            Total duration in seconds.
        """
        return self._total_duration

    @property
    def analysed_livetime(self) -> Quantity:
        """Seconds of data the run actually writes: ``duration * max_samples``.

        Less than :attr:`total_duration` by ``(max_samples - 1) * segment_gap`` for a gapped run,
        and equal to it for a contiguous one.

        Returns:
            Analysed livetime in seconds.
        """
        return cast(Quantity, self.duration * self.max_samples)

    @total_duration.setter
    def total_duration(self, value: float | str | None) -> None:
        """Set the run's GPS span, and derive the segment count from it.

        Args:
            value: Span in seconds, or a duration string such as ``"1 day"``. ``None`` derives the
                span from ``max_samples`` instead.

        Raises:
            ValueError: If the span is negative, shorter than one segment, or -- for a gapped
                configuration -- does not divide into whole segments.
        """
        if value is not None:
            self._total_duration = Quantity(parse_seconds(value, "total_duration"), unit="s")

            if self.total_duration < 0:
                raise ValueError("total_duration must be non-negative.")

            if self.total_duration < self.duration:
                raise ValueError("total_duration must be greater than or equal to duration.")

            # One arithmetic, in one place: the count comes from the span, and the span is then
            # restated from the count so that `final_end_time` and the epoch list cannot disagree
            # with each other. For a contiguous run this is the round-to-a-multiple-of-duration
            # this setter has always done; for a gapped one a span that does not divide is refused
            # rather than moved.
            num_segments = resolve_segment_count(
                float(self.total_duration.value), float(self.duration.value), float(self.segment_gap.value)
            )
            layout = SegmentLayout(
                start_time=float(self.start_time.value),
                duration=float(self.duration.value),
                gap=float(self.segment_gap.value),
                count=num_segments,
            )
            self._total_duration = Quantity(layout.span, unit="s")

            logger.info("Total duration set to %s seconds.", self.total_duration)

            self.max_samples = num_segments
            logger.info(
                "Setting max_samples to %s based on total_duration, duration and segment_gap.", self.max_samples
            )
        else:
            self._total_duration = Quantity(
                SegmentLayout(
                    start_time=float(self.start_time.value),
                    duration=float(self.duration.value),
                    gap=float(self.segment_gap.value),
                    count=1 if not np.isfinite(self.max_samples) else int(self.max_samples),
                ).span,
                unit="s",
            )
            # total_duration was not passed to this simulator directly. That does
            # not mean the user never set it: the orchestrator resolves a config
            # total_duration into max_samples upstream and forwards only that, so
            # claiming "total_duration not set" here misleads. Report the derived
            # value factually instead.
            logger.info(
                "Resolved total_duration to %s seconds (the span of %s segments of %s s separated by %s s gaps).",
                self.total_duration.value,
                self.max_samples,
                self.duration.value,
                self.segment_gap.value,
            )

    @property
    def end_time(self) -> Quantity:
        """Calculate the end time of the current segment.

        Returns:
            End time in GPS seconds.
        """
        return cast(Quantity, self.start_time + self.duration)

    @property
    def final_end_time(self) -> Quantity:
        """GPS time at which the entire simulation's data ends.

        Measured from the run's *configured* start time, not from the segment currently being
        built. Those are the same thing only for the first segment, and the difference matters
        now that ``total_duration`` is a span that a gapped run does not tile: anchoring it to the
        advancing ``start_time`` would slide the end of the run forward by one stride per segment,
        so "outside the total duration" would name a moving target rather than the end of the data.

        Returns:
            Final end time in GPS seconds.
        """
        return cast(Quantity, self._run_start_time + self.total_duration)

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Generate time series data chunks.

        This method should be implemented by subclasses to generate
        the actual time series data.
        """
        raise NotImplementedError("Subclasses must implement the _simulate method.")

    def simulate(self, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Simulate a segment of time series data.

        Args:
            *args: Positional arguments for the _simulate method.
            **kwargs: Keyword arguments for the _simulate method.

        Returns:
            TimeSeries: Simulated time series segment.
        """
        # First create a new segment
        segment = TimeSeries(
            data=np.zeros(
                (self.num_of_channels, int(self.duration.value * self.sampling_frequency.value)), dtype=self.dtype
            ),
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
        )

        # Which carried-forward chunks reach this segment, recorded *before* injecting them.
        # Injection sums into shared channels, so after this line there is no way to ask which
        # signal contributed to the segment -- and a signal long enough to cross a boundary is
        # exactly the case where the answer is not the segment it was generated in.
        self.carried_injections = _contributing_injections(segment, self.cached_data_chunks)

        # The gap this segment sits behind, if any. A chunk carried across it has the part lying
        # inside the gap cropped away by injection -- correctly, since the gap is written nowhere --
        # and the part beyond it placed at its true GPS sample. Measured before that happens,
        # because afterwards the samples are gone and nothing says they existed.
        preceding_gap = self.segment_layout.gap_before(int(getattr(self, "counter", 0) or 0))
        self.gap_discarded = _gap_discarded_records(segment, self.cached_data_chunks, preceding_gap)
        for record in self.gap_discarded:
            logger.warning(
                "Discarding %.3f s (%d samples, %.2f%% of its unweighted strain-squared energy) of "
                "the signal with event_id %s into the configured segment gap [%s, %s): the run "
                "writes no frame covering that span. The rest of the signal is placed at its true "
                "GPS time in this segment, which starts at %s.",
                record["discarded_seconds"],
                record["discarded_samples"],
                100.0 * record["discarded_energy_fraction"],
                record["event_id"],
                record["gap_start"],
                record["gap_end"],
                self.start_time,
            )

        # Inject cached data chunks into the segment
        self.cached_data_chunks = segment.inject_from_list(self.cached_data_chunks, preceding_gap=preceding_gap)

        # Generate new chunks of data
        new_chunks = self._simulate(*args, **kwargs)

        # Chunks generated for this segment that do not actually reach it are excluded for the same
        # reason the carried ones are included: the record names the frames a signal is *in*.
        self.carried_injections = _merge_injection_records(
            self.carried_injections, _contributing_injections(segment, new_chunks)
        )

        # Add the new chunks to the segment
        remaining_chunks = segment.inject_from_list(new_chunks, preceding_gap=preceding_gap)

        # Add the remaining chunks to the cache
        self.cached_data_chunks.extend(remaining_chunks)

        # Check whether there are chunks that are outside the whole dataset duration
        # Remove the chunks that are outside the total duration
        for i in reversed(range(len(self.cached_data_chunks))):
            chunk = self.cached_data_chunks[i]
            if chunk.start_time >= self.final_end_time:
                logger.info(
                    "Removing cached chunk starting at %s which is outside the total duration ending at %s.",
                    chunk.start_time,
                    self.final_end_time,
                )
                self.cached_data_chunks.pop(i)
            elif chunk.end_time <= self.start_time:
                logger.info(
                    "Removing cached chunk ending at %s which is before the current segment starting at %s.",
                    chunk.end_time,
                    self.start_time,
                )
                self.cached_data_chunks.pop(i)

        return segment

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = {
            "time_series": {
                "arguments": {
                    "start_time": self.start_time,
                    "duration": self.duration,
                    "segment_gap": self.segment_gap,
                    "sampling_frequency": self.sampling_frequency,
                    "num_of_channels": self.num_of_channels,
                    "dtype": str(self.dtype),
                }
            }
        }
        return metadata

    def _save_data(  # pylint: disable=unused-argument
        self,
        data: TimeSeries,
        file_name: str | Path | np.ndarray[Any, np.dtype[np.object_]],
        **kwargs,
    ) -> None:
        """Save time series data to a file.

        Args:
            data: Time series data to save.
            file_name: Path to the output file.
            **kwargs: Additional arguments for the saving function.
        """
        if "channel" in kwargs:
            channel = kwargs.pop("channel")
            channel = expand_template_variables(value=channel, simulator_instance=self)
            if isinstance(channel, str):
                channel = [channel] * data.num_of_channels
            elif isinstance(channel, list):
                if len(channel) != data.num_of_channels:
                    raise ValueError("Length of channel list must match number of channels in data.")
            else:
                raise ValueError("channel must be a string or list of strings.")
        else:
            channel = [None] * data.num_of_channels
        if data.num_of_channels == 1 and isinstance(file_name, (str, Path)):
            self._save_gwf_data(data=data[0], file_name=file_name, channel=channel[0], **kwargs)
        elif (
            data.num_of_channels > 1
            and isinstance(file_name, np.ndarray)
            and len(file_name.shape) == 1
            and file_name.shape[0] == data.num_of_channels
        ):
            for i in range(data.num_of_channels):
                single_file_name = cast(Path, file_name[i])
                single_channel = channel[i]
                self._save_gwf_data(data=data[i], file_name=single_file_name, channel=single_channel, **kwargs)
        else:
            raise ValueError(
                "file_name must be a single path for single-channel data or an array of paths for multi-channel data."
            )

    def _save_gwf_data(  # pylint: disable=unused-argument
        self, data: GWPyTimeSeries, file_name: str | Path, channel: str | None = None, **kwargs
    ) -> None:
        """Save GWPy TimeSeries data to a GWF file.

        Args:
            data: GWPy TimeSeries data to save.
            file_name: Path to the output GWF file.
            channel: Optional channel name to set in the data.
            **kwargs: Additional arguments for the write function.
        """
        if channel is not None:
            data.channel = channel
        data.write(str(file_name))
