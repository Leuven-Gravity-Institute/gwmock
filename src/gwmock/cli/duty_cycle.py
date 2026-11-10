"""The duty cycle of a run: when each interferometer observes, and the products that record it.

A run with ``orchestration.duty-cycle`` draws one realization of gwmock-noise's duty-cycle model over
its whole span, so every frame of the run -- a resumed one included -- sees the same duty cycle. The
strain is not changed. The realization is recorded three ways:

- a **state-vector channel** per interferometer and frame, ``<interferometer>:<channel>``, at the
  configured rate. Bit 0 (``OBSERVING_BIT``) is set while the interferometer observes and bit 1
  (``MAINTENANCE_BIT``) while it is in planned maintenance. The samples are ``uint32``: LAL, which writes
  GWF, has no unsigned 8-bit type, and LIGO's calibrated state vector is 32-bit too;
- the run's **segments**: every interferometer's observing intervals and maintenance windows in one
  JSON file, and each interferometer's observing intervals as a plain text list. The first batch
  writes and records them. A later batch rewrites them, and records them, only if they hold another
  realization when a run resumes. Missing ones are batch 0's missing outputs, so a resumed run
  regenerates from the first frame, as for any missing output;
- an **observing flag** on every injected signal with a merger time (for each interferometer, at that
  time) and on every glitch (for its interferometer, at its peak), in the batch metadata.

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
    """

    model: Any
    detectors: tuple[str, ...]
    start: float
    span: float
    seed: int
    sampling_frequency: float
    _realization: Any = field(default=None, init=False, repr=False)
    _segment_files: dict[str, str] | None = field(default=None, init=False, repr=False)

    @property
    def realization(self) -> Any:
        """The ``DutyCycleRealization``, drawn once on first use."""
        if self._realization is None:
            from gwmock_noise.duty_cycle import simulate_duty_cycle  # noqa: PLC0415

            self._realization = simulate_duty_cycle(self.model, self.detectors, self.start, self.span, self.seed)
        return self._realization

    def observing(self, detector: str, time: float) -> bool:
        """Whether ``detector`` observed at GPS ``time``."""
        return bool(self.realization.is_observing(detector, np.array([float(time)]))[0])

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
        """The state vector of ``detector`` over ``[t0, t0 + duration)``."""
        state = self.realization.state_vector(detector, t0, self.sampling_frequency, self.n_samples(duration))
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
        """The run's segments, every interferometer's observing intervals and maintenance windows."""
        realization = self.realization
        return {
            "schema": SEGMENTS_SCHEMA,
            "schema_version": SEGMENTS_SCHEMA_VERSION,
            "time_convention": "GPS seconds; each interval is [start, end).",
            "span": [realization.start, realization.end],
            "seed": self.seed,
            "model": asdict(self.model),
            "detectors": {
                detector: {
                    "observing_fraction": realization.observing_fraction(detector),
                    "observing": realization.observing[detector].tolist(),
                    "maintenance": realization.maintenance[detector].tolist(),
                }
                for detector in self.detectors
            },
        }

    def segment_file_contents(self) -> dict[str, str]:
        """The run's segment files, by file name.

        Returns:
            The JSON file, then one ``<interferometer>-observing-segments.txt`` per interferometer,
            each line ``start end`` in GPS seconds.
        """
        if self._segment_files is None:  # a pure function of the realization, rendered once per run
            record = self.segments_record()
            contents = {SEGMENTS_FILE: json.dumps(record, indent=2) + "\n"}
            for detector in self.detectors:
                lines = [f"{start!r} {end!r}\n" for start, end in record["detectors"][detector]["observing"]]
                contents[f"{detector}-observing-segments.txt"] = "".join(lines)
            self._segment_files = contents
        return dict(self._segment_files)

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
            "segments_file": SEGMENTS_FILE,
        }
