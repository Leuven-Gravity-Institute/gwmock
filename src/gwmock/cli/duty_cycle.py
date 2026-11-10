"""The duty cycle of a run: when each interferometer observes, and the products that record it.

A run with ``orchestration.duty-cycle`` draws one realization of gwmock-noise's duty-cycle model over
its whole span, so every frame of the run -- a resumed one included -- sees the same duty cycle. From
each interferometer's observing intervals it takes its **science segments**, the whole GPS seconds
inside each interval, as LVK science segments are published. Every product below is drawn from those
segments, so they agree with one another, and a pipeline that keeps its analysis windows inside them
reads only observing data:

- a **state-vector channel** per interferometer and frame, ``<interferometer>:<channel>``, at the
  configured rate, with gwmock-noise's bits: bits 0 to 2 (``OBSERVING_MASK``, LIGO's ``HOFT_OK``,
  ``OBS_INTENT`` and ``OBS_READY``) on a sample whose whole interval lies in a science segment, and
  bit 16 (``MAINTENANCE_BIT``) on a sample whose time is in planned maintenance. PyCBC's low-latency
  search and gwpy read it. The samples are ``uint32``;
- the run's **segments**, for the pipelines that take segment lists instead (PyCBC's offline search,
  cWB, gwpy): a LIGO_LW XML file with each interferometer's science segments under the flag
  ``<interferometer>:science:1`` -- the name PyCBC's offline search reads from a segment file -- and
  its maintenance windows under ``<interferometer>:maintenance:1``; each interferometer's science
  segments as a two-column ``start end`` text list, the format cWB reads; and one JSON file with the
  exact realization, the science segments, the model and the seed. The first batch writes and records
  them. A batch rewrites them, and records them, when their contents differ from the ones it would
  write and it may overwrite: in a run started with ``--overwrite``, or a resumed run. A resumed run
  with no batch left to run leaves them as they are. Missing ones are batch 0's missing outputs, so a
  resumed run regenerates from the first frame, as for any missing output;
- an **observing flag** on every injected signal with a merger time (for each interferometer, whether
  that time is in a science segment) and on every glitch (for its interferometer, at its peak), in the
  batch metadata.

Unless ``mask-strain`` is off, every strain sample the run writes outside its interferometer's science
segments is NaN, as open data marks time outside observing: a pipeline that reads neither the state
vector nor the segments then meets NaN rather than analysing that time unaware. The noise realization
itself is unchanged; only what is written is masked.

An HDF5 state-vector file declares the gwmock strain schema, like every HDF5 file gwmock writes: the
schema is the file's layout (one dataset per channel, named for it, with its grid in ``x0``/``dx``),
which the state vector follows, and declaring it is what lets the run record be embedded in the file.
Its ``unit`` is empty, as for any dimensionless series: the samples are bit masks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

#: Type of the state-vector samples.
STATE_VECTOR_DTYPE = np.uint32

#: Name and version of the segment-file layout.
SEGMENTS_SCHEMA = "gwmock-duty-cycle-segments"
SEGMENTS_SCHEMA_VERSION = "1.0.0"

#: Name of the run's segment file, in the duty-cycle output directory.
SEGMENTS_FILE = "duty-cycle-segments.json"

#: Last GPS second the LIGO_LW segment file can hold: its time columns are 32-bit signed integers, as
#: gwpy writes them and PyCBC reads them.
LIGOLW_MAX_GPS = 2**31 - 1

#: Name of the run's LIGO_LW XML segment file, in the duty-cycle output directory.
SEGMENTS_XML_FILE = "duty-cycle-segments.xml"

#: Flag names in the LIGO_LW XML segment file. ``science`` is the name PyCBC's offline search reads
#: from a segment file given with ``segments-source = file``, as ``<interferometer>:science``.
OBSERVING_FLAG = "science"
MAINTENANCE_FLAG = "maintenance"


@dataclass
class DutyCycle:
    """One run's duty-cycle realization and how it is written.

    Attributes:
        model: The gwmock-noise ``DutyCycleModel``.
        detectors: The run's interferometers, in order.
        start: GPS start of the run's span.
        span: Length of the span, gaps included, in seconds.
        seed: Seed of the realization.
        sampling_frequency: Samples per second of the state-vector channel.
        mask_strain: Whether strain written outside its interferometer's science segments is NaN.
    """

    model: Any
    detectors: tuple[str, ...]
    start: float
    span: float
    seed: int
    sampling_frequency: float
    mask_strain: bool = True
    _realization: Any = field(default=None, init=False, repr=False)
    _science: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _segment_files: dict[str, str] | None = field(default=None, init=False, repr=False)

    @property
    def realization(self) -> Any:
        """The ``DutyCycleRealization``, drawn once on first use."""
        if self._realization is None:
            from gwmock_noise.duty_cycle import simulate_duty_cycle  # noqa: PLC0415

            self._realization = simulate_duty_cycle(self.model, self.detectors, self.start, self.span, self.seed)
        return self._realization

    @property
    def science(self) -> dict[str, np.ndarray]:
        """Per interferometer, its science segments: the whole GPS seconds inside each observing interval.

        Every product but the JSON record is drawn from these -- the XML and text segment files, the
        state vector, the strain mask and the observing flags -- so they agree with one another. Whole
        seconds because that is how LVK science segments are published, and what PyCBC's offline search
        requires; cWB rounds a segment inward to whole seconds itself.
        """
        if self._science is None:
            science = {}
            for detector in self.detectors:
                intervals = np.asarray(self.realization.observing[detector], dtype=float).reshape(-1, 2)
                starts, ends = np.ceil(intervals[:, 0]), np.floor(intervals[:, 1])
                keep = ends > starts
                science[detector] = np.column_stack([starts[keep], ends[keep]])
            self._science = science
        return self._science

    def _in_science(self, detector: str, times: Any) -> np.ndarray:
        """Whether each GPS time lies in one of ``detector``'s ``[start, end)`` science segments."""
        return _inside(self.science[detector], np.asarray(times, dtype=float))

    def observing(self, detector: str, time: float) -> bool:
        """Whether GPS ``time`` lies in one of ``detector``'s science segments."""
        return bool(self._in_science(detector, [float(time)])[0])

    def n_samples(self, duration: float) -> int:
        """Samples of the state vector in a frame of ``duration`` seconds.

        Raises:
            ValueError: If the frame does not hold a whole number of samples.
        """
        count = duration * self.sampling_frequency
        if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"a {duration} s frame does not hold a whole number of state-vector samples at "
                f"{self.sampling_frequency} Hz."
            )
        return round(count)

    def state_vector(self, detector: str, t0: float, duration: float) -> np.ndarray:
        """The state vector of ``detector`` over ``[t0, t0 + duration)``.

        A sample has the observing bits (``OBSERVING_MASK``) when the whole of its interval
        ``[t, t + 1/f)`` lies in a science segment, so no strain sample a state-valid sample covers is
        masked; it has ``MAINTENANCE_BIT`` when its time is in a maintenance window.
        """
        # gwmock-noise's ``DutyCycleRealization.state_vector`` samples the exact intervals; the product here
        # is drawn from the whole-second science segments instead, so that it agrees with the segment
        # files and the strain mask. Only the bit values are shared, imported so they cannot drift.
        from gwmock_noise.duty_cycle import MAINTENANCE_BIT, OBSERVING_MASK  # noqa: PLC0415

        n_samples = self.n_samples(duration)
        starts = t0 + np.arange(n_samples) / self.sampling_frequency
        stops = t0 + np.arange(1, n_samples + 1) / self.sampling_frequency
        segments = self.science[detector]
        index = np.searchsorted(segments[:, 0], starts, side="right") - 1
        observing = index >= 0
        observing[observing] = stops[observing] <= segments[index[observing], 1]
        maintenance = _inside(np.asarray(self.realization.maintenance[detector], dtype=float), starts)
        state = np.where(observing, OBSERVING_MASK, 0) | np.where(maintenance, MAINTENANCE_BIT, 0)
        return state.astype(STATE_VECTOR_DTYPE)

    def write_frame(
        self,
        paths: dict[str, Path],
        channels: dict[str, str],
        t0: float,
        duration: float,
        output_format: Literal["gwf", "hdf5"],
    ) -> None:
        """Write each interferometer's state vector over one frame to its own file.

        Args:
            paths: Output file per interferometer.
            channels: Channel name per interferometer.
            t0: GPS start of the frame.
            duration: Length of the frame, in seconds.
            output_format: ``gwf`` or ``hdf5``.
        """
        from gwmock_data import TimeSeries  # noqa: PLC0415

        from gwmock.strain_schema import declare_strain_schema  # noqa: PLC0415

        for detector, path in paths.items():
            series = TimeSeries(
                self.state_vector(detector, t0, duration),
                t0=t0,
                sample_rate=self.sampling_frequency,
                name=channels[detector],
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if output_format == "gwf":
                from gwmock._optional import require  # noqa: PLC0415

                require("gwpy.timeseries", "gwpy", "Writing a GWF state vector")
                gwpy_series = series.to_gwpy()
                gwpy_series.channel = channels[detector]
                gwpy_series.write(str(path), format="gwf")
            else:
                from gwmock_data.io import write_hdf5  # noqa: PLC0415

                write_hdf5({channels[detector]: series}, path)
                declare_strain_schema(path)

    def segments_record(self) -> dict[str, Any]:
        """The run's segments: every interferometer's exact observing intervals, science segments and
        maintenance windows."""
        realization = self.realization
        return {
            "schema": SEGMENTS_SCHEMA,
            "schema_version": SEGMENTS_SCHEMA_VERSION,
            "time_convention": (
                "GPS seconds; each interval is [start, end). 'observing' is the realization's exact "
                "intervals; 'science' is the whole seconds inside each, which the other segment files, "
                "the state vector, the strain mask and the observing flags use."
            ),
            "span": [realization.start, realization.end],
            "seed": self.seed,
            "model": asdict(self.model),
            "detectors": {
                detector: {
                    "observing_fraction": realization.observing_fraction(detector),
                    "observing": realization.observing[detector].tolist(),
                    "science": self.science[detector].tolist(),
                    "maintenance": realization.maintenance[detector].tolist(),
                }
                for detector in self.detectors
            },
        }

    def segment_file_contents(self) -> dict[str, str]:
        """The run's segment files, by file name.

        Returns:
            The JSON file, the LIGO_LW XML file, then one ``<interferometer>-observing-segments.txt`` per
            interferometer, each line ``start end`` in GPS seconds.
        """
        if self._segment_files is None:  # a pure function of the realization, rendered once per run
            record = self.segments_record()
            contents = {
                SEGMENTS_FILE: json.dumps(record, indent=2) + "\n",
                SEGMENTS_XML_FILE: self._ligolw_segments(),
            }
            for detector in self.detectors:
                lines = [f"{int(start)} {int(end)}\n" for start, end in self.science[detector]]
                contents[f"{detector}-observing-segments.txt"] = "".join(lines)
            self._segment_files = contents
        return dict(self._segment_files)

    def _ligolw_segments(self) -> str:
        """The LIGO_LW XML segment file: for each interferometer, its observing and maintenance flags.

        The layout is the one gwpy writes and PyCBC's ``SegFile.from_segment_xml`` reads: one
        ``segment_definer`` row per flag, one ``segment_summary`` row giving the span the flag is known
        over, and one ``segment`` row per active interval, each time as integer GPS seconds and
        nanoseconds.
        """
        realization = self.realization
        known = [(realization.start, realization.end)]
        definers, summaries, segments = [], [], []
        definer_id = segment_id = 0
        for detector in self.detectors:
            for name, intervals in (
                (OBSERVING_FLAG, self.science[detector]),
                (MAINTENANCE_FLAG, realization.maintenance[detector]),
            ):
                definers.append(f',,"{_ligolw_string(detector)}",,"{name}",,{definer_id},1')
                for start, end in known:
                    summaries.append(_ligolw_span(start, end, f",,{{end}},,,{definer_id},{len(summaries)},{{start}}"))
                for start, end in intervals:
                    segments.append(_ligolw_span(start, end, f",{{end}},,,{definer_id},{segment_id},{{start}}"))
                    segment_id += 1
                definer_id += 1
        return _LIGOLW_TEMPLATE.format(
            definers=_ligolw_rows(definers),
            summaries=_ligolw_rows(summaries),
            segments=_ligolw_rows(segments),
        )

    def mask(self, detector: str, values: Any, times: Any) -> np.ndarray:
        """``values`` with every sample outside ``detector``'s science segments set to NaN.

        Args:
            detector: The interferometer.
            values: Strain samples.
            times: GPS time of each sample.

        Returns:
            A float array; ``values`` is not changed.
        """
        observing = self._in_science(detector, times)
        values = np.asarray(values, dtype=float)
        if observing.all():
            return values
        return np.where(observing, values, np.nan)

    def write_segment_files(self, directory: Path) -> list[Path]:
        """Write the run's segment files into ``directory``, in the order of :meth:`segment_file_contents`."""
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, text in self.segment_file_contents().items():
            (directory / name).write_text(text, encoding="utf-8")
            paths.append(directory / name)
        return paths

    def metadata(self) -> dict[str, Any]:
        """What a batch's metadata records about the duty cycle."""
        return {
            "model": asdict(self.model),
            "seed": self.seed,
            "detectors": list(self.detectors),
            "span": [self.start, self.start + self.span],
            "sampling_frequency": self.sampling_frequency,
            "strain_masked": self.mask_strain,
            "segments_file": SEGMENTS_FILE,
            "segments_xml_file": SEGMENTS_XML_FILE,
        }


def _inside(intervals: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Whether each time lies in one of the sorted, disjoint ``[start, end)`` intervals.

    gwmock-noise has the same membership test for its realization; this one serves the science segments.
    """
    intervals = intervals.reshape(-1, 2)
    index = np.searchsorted(intervals[:, 0], times, side="right") - 1
    inside = index >= 0
    inside[inside] = times[inside] < intervals[index[inside], 1]
    return inside


def _ligolw_string(value: str) -> str:
    """``value`` as a LIGO_LW ``lstring`` cell: XML-escaped, with quotes and backslashes escaped."""
    from xml.sax.saxutils import escape  # noqa: PLC0415

    return escape(value.replace("\\", "\\\\").replace('"', '\\"'))


_NANOSECONDS_PER_SECOND = 1_000_000_000


def _gps_split(time: float) -> tuple[int, int]:
    """GPS ``time`` as whole seconds and nanoseconds, rounded to the nearest nanosecond."""
    seconds = math.floor(time)
    nanoseconds = round((time - seconds) * _NANOSECONDS_PER_SECOND)  # time - floor(time) is exact
    if nanoseconds == _NANOSECONDS_PER_SECOND:
        seconds, nanoseconds = seconds + 1, 0
    return int(seconds), int(nanoseconds)


def _ligolw_span(start: float, end: float, row: str) -> str:
    """``row`` with ``{start}`` and ``{end}`` replaced by the span's ``seconds,nanoseconds`` cells."""
    start_s, start_ns = _gps_split(start)
    end_s, end_ns = _gps_split(end)
    return row.format(start=f"{start_s},{start_ns}", end=f"{end_s},{end_ns}")


def _ligolw_rows(rows: list[str]) -> str:
    return ",\n".join(f"\t\t\t{row}" for row in rows)


_LIGOLW_TEMPLATE = """<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
\t<Table Name="segment_definer:table">
\t\t<Column Name="comment" Type="lstring"/>
\t\t<Column Name="creator_db" Type="int_4s"/>
\t\t<Column Name="ifos" Type="lstring"/>
\t\t<Column Name="insertion_time" Type="int_4s"/>
\t\t<Column Name="name" Type="lstring"/>
\t\t<Column Name="process:process_id" Type="int_8s"/>
\t\t<Column Name="segment_def_id" Type="int_8s"/>
\t\t<Column Name="version" Type="int_4s"/>
\t\t<Stream Name="segment_definer:table" Delimiter="," Type="Local">
{definers}
\t\t</Stream>
\t</Table>
\t<Table Name="segment_summary:table">
\t\t<Column Name="comment" Type="lstring"/>
\t\t<Column Name="creator_db" Type="int_4s"/>
\t\t<Column Name="end_time" Type="int_4s"/>
\t\t<Column Name="end_time_ns" Type="int_4s"/>
\t\t<Column Name="process:process_id" Type="int_8s"/>
\t\t<Column Name="segment_def_cdb" Type="int_4s"/>
\t\t<Column Name="segment_definer:segment_def_id" Type="int_8s"/>
\t\t<Column Name="segment_sum_id" Type="int_8s"/>
\t\t<Column Name="start_time" Type="int_4s"/>
\t\t<Column Name="start_time_ns" Type="int_4s"/>
\t\t<Stream Name="segment_summary:table" Delimiter="," Type="Local">
{summaries}
\t\t</Stream>
\t</Table>
\t<Table Name="segment:table">
\t\t<Column Name="creator_db" Type="int_4s"/>
\t\t<Column Name="end_time" Type="int_4s"/>
\t\t<Column Name="end_time_ns" Type="int_4s"/>
\t\t<Column Name="process:process_id" Type="int_8s"/>
\t\t<Column Name="segment_def_cdb" Type="int_4s"/>
\t\t<Column Name="segment_definer:segment_def_id" Type="int_8s"/>
\t\t<Column Name="segment_id" Type="int_8s"/>
\t\t<Column Name="start_time" Type="int_4s"/>
\t\t<Column Name="start_time_ns" Type="int_4s"/>
\t\t<Stream Name="segment:table" Delimiter="," Type="Local">
{segments}
\t\t</Stream>
\t</Table>
</LIGO_LW>
"""
