"""Adapter from gwmock orchestration to public ``gwmock_noise`` APIs."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import h5py
import numpy as np
from gwmock_noise import (
    AddLines,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    FrameWriter,
    InjectGlitches,
    NoiseComponentConfig,
    NoiseConfig,
    NoiseSimulator,
    OutputConfig,
    SimulationResult,
    SpectralLineSimulator,
)
from gwmock_noise import (
    open_stream as upstream_open_stream,
)
from gwmock_noise.gaussian import normalize_spectral_lines
from gwmock_noise.glitches import normalize_glitch_models
from gwmock_noise.output.frame import compose_frame_name, format_time_token

from gwmock.strain_schema import declare_strain_schema
from gwmock.utils.download import download_file

_SUPPORTED_OUTPUT_FORMATS = {"npy", "gwf", "hdf5"}
_DETECTOR_PAIR_SIZE = 2


def _frame_channel_name(detector: str, channel: str) -> str:
    """Return detector channel name used by gwmock-noise frame outputs.

    Args:
        detector: The detector name.
        channel: The channel name suffix.

    Returns:
        The detector channel name.
    """
    return f"{detector}:{channel}"


def _frame_artifact_name(
    *,
    detector: str,
    channel: str,
    prefix: str,
    gps_start: float,
    duration: float,
) -> str:
    """Return the GWF filename gwmock-noise's ``FrameWriter`` writes for one detector's chunk.

    Delegates rather than composing the name here. This used to hold its own copy of the rule, and the
    copy drifted twice at once: gwmock-noise now drops the channel's ``IFO:`` prefix from the name,
    because a colon made every frame unwritable on Windows, and it refuses a fractional second rather
    than encoding it as ``100p5``, because two epochs that round alike composed one name and the second
    run overwrote the first. Neither change reached this function, so it promised ``H-H1:MOCK_NOISE_...``
    for a file the backend wrote as ``H-H1_MOCK_NOISE_...`` -- a path that never existed.

    The point of the function is to say what the backend will write, so the backend's own composer is
    the only thing that can answer it.

    Args:
        detector: The detector name.
        channel: The full channel name (e.g. ``H1:MOCK_NOISE``).
        prefix: The prefix.
        gps_start: The GPS start time, which must be a whole second.
        duration: The duration, which must be a whole second.

    Returns:
        The GWF filename.

    Raises:
        ValueError: If the epoch or the duration is not a whole number of seconds.
    """
    return compose_frame_name(
        detector=detector,
        channel=channel,
        gps_start=gps_start,
        duration=duration,
        prefix=prefix,
    )


def _hdf5_artifact_name(*, detector: str, prefix: str, gps_start: float, duration: float) -> str:
    """Return the HDF5 filename gwmock-noise writes for one detector's chunk.

    Named for the detector rather than the channel -- ``H-H1_1000000000-4.hdf5`` -- because the dataset
    inside carries the channel, so one detector writes one file. That is gwmock-noise's rule, not a
    choice made here, and `expected_output_paths` has to agree with it or it would promise a path the
    backend does not write.

    The epoch and duration tokens come from gwmock-noise's own ``format_time_token`` for the same reason
    ``_frame_artifact_name`` above delegates: a second expression for one name drifts from the first, and
    the copy this module used to keep had already drifted. Upstream refuses a fractional second there,
    because encoding it to six decimals gave two epochs a microsecond apart the same token and the second
    run silently overwrote the first; calling upstream means this name cannot acquire that bug on its own.

    Only the template is written out here, because gwmock-noise composes the HDF5 name inside its
    simulator with no public function to call.

    Args:
        detector: The detector, whose first character is the site letter.
        prefix: The output prefix, or the empty string.
        gps_start: The epoch of the chunk, which must be a whole second.
        duration: The duration of the chunk, which must be a whole second.

    Returns:
        The HDF5 filename.

    Raises:
        ValueError: If the epoch or the duration is not a whole number of seconds.
    """
    name = f"{detector[0]}-{detector}_{format_time_token(gps_start)}-{format_time_token(duration)}.hdf5"
    return f"{prefix}_{name}" if prefix else name


def _coerce_path(value: str | Path | None) -> Path | None:
    """Normalize a path-like input.

    Args:
        value: The value to coerce.

    Returns:
        The coerced path.
    """
    if value is None:
        return None
    return Path(value)


def _coerce_path_mapping(values: dict[str, str | Path] | None) -> dict[str, Path] | None:
    """Normalize mapping values to ``Path`` objects.

    Args:
        values: The values to coerce.

    Returns:
        The coerced path mapping.
    """
    if values is None:
        return None
    return {key: Path(value) for key, value in values.items()}


def _coerce_path_schedule(values: list[tuple[float, str | Path]] | None) -> list[tuple[float, Path]] | None:
    """Normalize scheduled path values to ``Path`` objects.

    Args:
        values: The values to coerce.

    Returns:
        The coerced path schedule.
    """
    if values is None:
        return None
    return [(offset, Path(path)) for offset, path in values]


def _parse_csd_file_map(csd_files: dict[str, Path] | None) -> dict[tuple[str, str], Path]:
    """Convert ``DET1-DET2`` mapping keys into detector-pair tuples.

    Args:
        csd_files: The CSD files to parse.

    Returns:
        The parsed CSD file map.
    """
    if not csd_files:
        return {}

    parsed: dict[tuple[str, str], Path] = {}
    for pair_key, file_path in csd_files.items():
        detectors = pair_key.split("-")
        if len(detectors) != _DETECTOR_PAIR_SIZE or not all(detectors):
            raise ValueError("csd_files keys must use the 'DET1-DET2' format.")

        detector_a, detector_b = tuple(sorted(detectors))
        if detector_a == detector_b:
            raise ValueError("csd_files keys must reference two distinct detectors.")

        normalized_key = (detector_a, detector_b)
        if normalized_key in parsed:
            raise ValueError(f"Duplicate CSD file mapping for detector pair {detector_a}-{detector_b}.")
        parsed[normalized_key] = Path(file_path)
    return parsed


_REMOTE_GLITCH_FILE_KEYS = ("population_file", "psd_file")


def _is_remote_url(value: Any) -> bool:
    """Return True if ``value`` is an ``http``/``https`` URL string.

    Args:
        value: The candidate file reference.

    Returns:
        Whether the value is a remote URL (named assets and local paths are not).
    """
    return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}


def _glitch_population_cache_directory() -> Path:
    """Return the cache directory for downloaded glitch population/PSD files."""
    return Path(tempfile.gettempdir()) / "gwmock" / "noise_glitches"


def _resolve_glitch_file_urls(glitches: list[Any] | None) -> list[Any] | None:
    """Download URL-valued glitch file references to a local cache.

    gwmock-noise glitch models (e.g. ``GengliBlipGlitch``) read ``population_file``
    and ``psd_file`` with ``Path``/``h5py``, which cannot handle URLs. Resolve any
    ``http(s)`` URL to a cached local path so remote files declared in configs work
    transparently. Named bundled assets (e.g. ``ET_10_full_cryo_psd``) and local
    paths are returned unchanged. The input list and its entries are not mutated.

    Args:
        glitches: The raw glitch-model configuration entries.

    Returns:
        A copy of ``glitches`` with remote file references replaced by local paths.
    """
    if not glitches:
        return glitches

    resolved: list[Any] = []
    for entry in glitches:
        if not isinstance(entry, Mapping):
            resolved.append(entry)
            continue
        updated = dict(entry)
        for key in _REMOTE_GLITCH_FILE_KEYS:
            value = updated.get(key)
            if _is_remote_url(value):
                updated[key] = str(
                    download_file(
                        value,
                        outdir=_glitch_population_cache_directory(),
                        allow_existing=True,
                        dest_path_from_hashed_url=True,
                    )
                )
        resolved.append(updated)
    return resolved


def _build_components(
    *,
    psd_file: Path | None,
    psd_schedule: list[tuple[float, Path]] | None,
    psd_files: dict[str, Path] | None,
    csd_files: dict[str, Path] | None,
    low_frequency_cutoff: float,
    high_frequency_cutoff: float | None,
    spectral_lines: list[Any] | None,
    glitches: list[Any] | None,
) -> list[NoiseComponentConfig]:
    """Translate legacy flat noise fields into the v0.3 component list."""
    components: list[NoiseComponentConfig] = []

    if psd_files is not None or csd_files is not None:
        options: dict[str, Any] = {}
        if psd_files:
            options["psd_files"] = psd_files
        if csd_files:
            options["csd_files"] = csd_files
        options["low_frequency_cutoff"] = low_frequency_cutoff
        if high_frequency_cutoff is not None:
            options["high_frequency_cutoff"] = high_frequency_cutoff
        components.append(NoiseComponentConfig(simulator="correlated", options=options))
    elif psd_file is not None or psd_schedule is not None:
        options = {}
        if psd_file is not None:
            options["psd_file"] = psd_file
        if psd_schedule is not None:
            options["psd_schedule"] = psd_schedule
        options["low_frequency_cutoff"] = low_frequency_cutoff
        if high_frequency_cutoff is not None:
            options["high_frequency_cutoff"] = high_frequency_cutoff
        components.append(NoiseComponentConfig(simulator="colored", options=options))

    if spectral_lines:
        components.append(NoiseComponentConfig(simulator="spectral_lines", options={"lines": spectral_lines}))

    if glitches:
        components.append(NoiseComponentConfig(simulator="glitches", options={"models": glitches}))

    return components


class NoiseAdapter:
    """Bridge gwmock orchestration state to public ``gwmock_noise`` APIs."""

    def __init__(self, *, backend: Any) -> None:
        """Store the resolved public gwmock-noise backend.

        Args:
            backend: The backend to use.
        """
        self._backend = backend
        # Glitch models built for the active stream, retained so their resolved,
        # config-shaped state (e.g. a pinned dataset revision) can be reported
        # for replayable metadata. None until a stream with glitches is opened.
        self._glitch_models: list[Any] | None = None
        # The injector wrapping the active stream, retained so each batch can read
        # the truth catalogue of the chunk it is about to write and stamp it with
        # that chunk's GPS epoch. None until a stream with glitches is opened.
        self._glitch_injector: Any | None = None
        # Memoized resolved_config() payload for the active stream, so per-batch
        # metadata writes across one open_stream() reuse a single resolution
        # (and one pinned revision) instead of re-resolving every batch. Reset
        # whenever the stream is (re)configured.
        self._resolved_config_cache: dict[str, Any] | None = None

    @classmethod
    def from_backend(cls, backend: BaseNoiseSimulator | NoiseSimulator | Any | None = None) -> NoiseAdapter:
        """Build an adapter from a public gwmock-noise backend.

        Args:
            backend: The backend to use.

        Returns:
            A NoiseAdapter instance.
        """
        if backend is None:
            resolved_backend = DefaultNoiseSimulator()
        elif isinstance(backend, (BaseNoiseSimulator, NoiseSimulator)) or callable(getattr(backend, "run", None)):
            resolved_backend = backend
        else:
            raise TypeError("backend must satisfy BaseNoiseSimulator or NoiseSimulator.")
        return cls(backend=resolved_backend)

    @property
    def backend(self) -> Any:
        """Return the public backend used by the adapter.

        Returns:
            The public backend used by the adapter.
        """
        return self._backend

    def run(  # noqa: PLR0913
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        output_directory: str | Path,
        output_prefix: str,
        output_format: Literal["npy", "gwf", "hdf5"],
        gps_start: float,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> SimulationResult:
        """Run one noise batch through the public gwmock-noise boundary.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            output_directory: The output directory.
            output_prefix: The output prefix.
            output_format: The output format.
            gps_start: The GPS start time.
            channel: The channel name suffix assembled as ``{detector}:{channel}``.
            channels: Optional per-detector full channel names, e.g. ``{"H1": "H1:STRAIN_NOISE"}``.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The simulation result.
        """
        config = self.build_config(
            detectors=detectors,
            duration=duration,
            sampling_frequency=sampling_frequency,
            output_directory=output_directory,
            output_prefix=output_prefix,
            output_format=output_format,
            gps_start=gps_start,
            channel=channel,
            channels=channels,
            seed=seed,
            psd_file=psd_file,
            psd_schedule=psd_schedule,
            psd_files=psd_files,
            csd_files=csd_files,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        if callable(getattr(self._backend, "run", None)):
            return self._declare_backend_outputs(self._backend.run(config))

        if not isinstance(self._backend, NoiseSimulator):
            raise TypeError("Noise backend must expose run() or satisfy the gwmock_noise NoiseSimulator protocol.")

        chunk = self._backend.generate(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
        )
        return self.write_chunk(config=config, chunk=chunk)

    def open_stream(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: Sequence[str],
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Open one stateful upstream stream and consume it chunk-by-chunk.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            An iterator over the chunks.
        """
        simulator = self._resolve_stream_backend(
            chunk_duration=chunk_duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
            psd_file=psd_file,
            psd_schedule=psd_schedule,
            psd_files=psd_files,
            csd_files=csd_files,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        return upstream_open_stream(
            simulator,
            chunk_duration=chunk_duration,
            sampling_frequency=sampling_frequency,
            detectors=list(detectors),
            seed=seed,
        )

    def build_config(  # noqa: PLR0913
        self,
        *,
        detectors: Sequence[str],
        duration: float,
        sampling_frequency: float,
        output_directory: str | Path,
        output_prefix: str,
        output_format: Literal["npy", "gwf", "hdf5"],
        gps_start: float,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> NoiseConfig:
        """Construct the public gwmock-noise config model for one output chunk.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            output_directory: The output directory.
            output_prefix: The output prefix.
            output_format: The output format.
            gps_start: The GPS start time.
            channel: The channel name suffix assembled as ``{detector}:{channel}``.
            channels: Optional per-detector full channel names, e.g. ``{"H1": "H1:STRAIN_NOISE"}``.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The noise config.
        """
        components = _build_components(
            psd_file=_coerce_path(psd_file),
            psd_schedule=_coerce_path_schedule(psd_schedule),
            psd_files=_coerce_path_mapping(psd_files),
            csd_files=_coerce_path_mapping(csd_files),
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        kwargs: dict[str, Any] = {
            "detectors": list(detectors),
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "output": OutputConfig(
                directory=Path(output_directory),
                prefix=output_prefix,
                format=output_format,
                gps_start=gps_start,
                channel=channel,
                channels=channels,
            ),
            "seed": seed,
        }
        if components:
            kwargs["components"] = components
        return NoiseConfig(**kwargs)

    def expected_output_paths(self, *, config: NoiseConfig) -> list[Path]:
        """Return the artifact paths gwmock will write for one chunk.

        Args:
            config: The noise config.

        Returns:
            The expected output paths.
        """
        if config.output.format == "npy":
            return [
                config.output.directory
                / (f"{config.output.prefix}_{detector}.npy" if config.output.prefix else f"{detector}.npy")
                for detector in config.detectors
            ]
        if config.output.format == "hdf5":
            return [
                config.output.directory
                / _hdf5_artifact_name(
                    detector=detector,
                    prefix=config.output.prefix,
                    gps_start=config.output.gps_start,
                    duration=config.duration,
                )
                for detector in config.detectors
            ]
        return [
            config.output.directory
            / _frame_artifact_name(
                detector=detector,
                channel=(
                    config.output.channels[detector]
                    if config.output.channels and detector in config.output.channels
                    else _frame_channel_name(detector, config.output.channel)
                ),
                prefix=config.output.prefix,
                gps_start=config.output.gps_start,
                duration=config.duration,
            )
            for detector in config.detectors
        ]

    def write_chunk(self, *, config: NoiseConfig, chunk: Mapping[str, np.ndarray]) -> SimulationResult:
        """Write one chunk returned by ``open_stream`` to gwmock-owned outputs.

        Args:
            config: The noise config.
            chunk: The chunk to write.

        Returns:
            The simulation result.
        """
        chunk_by_detector = self._normalize_chunk(chunk=chunk, detectors=config.detectors)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        if config.output.format == "gwf":
            output_paths = FrameWriter(
                _ChunkNoiseSimulator(chunk_by_detector),
                gps_start=config.output.gps_start,
                output_dir=config.output.directory,
                channel=config.output.channel,
                channels=config.output.channels,
                prefix=config.output.prefix,
            ).write(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=None,
            )
        elif config.output.format == "hdf5":
            output_paths = self._write_hdf5_chunk(config=config, chunk_by_detector=chunk_by_detector)
        else:
            output_paths = {}
            for detector, strain in chunk_by_detector.items():
                file_name = f"{config.output.prefix}_{detector}.npy" if config.output.prefix else f"{detector}.npy"
                output_path = config.output.directory / file_name
                np.save(output_path, strain)
                output_paths[detector] = output_path
        return SimulationResult(output_paths=output_paths, config=config)

    def _write_hdf5_chunk(self, *, config: NoiseConfig, chunk_by_detector: dict[str, np.ndarray]) -> dict[str, Path]:
        """Write one chunk as HDF5, laid out as gwmock-noise's own HDF5 writer lays it out.

        One file per detector, holding one dataset named for the resolved channel, carrying the grid as
        the ``x0``/``dx`` attributes gwpy reads. The layout is copied deliberately rather than delegated:
        this method exists because the chunk is already in hand, so there is no backend run to delegate
        to -- but a file written here and a file written by the backend are the same artifact under the
        same name, so they have to be the same shape inside as well.

        Written uncompressed, as the backend writes it. Strain is Gaussian, so its mantissa is very close
        to incompressible and there is little to buy: measured over a 134 MB segment, gzip level 4 -- the
        level gwpy's own HDF5 writer applies unless told otherwise -- made the file 4.1% smaller and took
        39x as long to write (0.07 s to 2.74 s). Raising the level to 9 changed neither number.

        Args:
            config: The noise config, providing the epoch, the grid and the naming.
            chunk_by_detector: One strain array per detector.

        Returns:
            The path written for each detector.
        """
        output_paths: dict[str, Path] = {}
        for detector, strain in chunk_by_detector.items():
            channel = (
                config.output.channels[detector]
                if config.output.channels and detector in config.output.channels
                else _frame_channel_name(detector, config.output.channel)
            )
            output_path = config.output.directory / _hdf5_artifact_name(
                detector=detector,
                prefix=config.output.prefix,
                gps_start=config.output.gps_start,
                duration=config.duration,
            )
            with h5py.File(output_path, "w") as handle:
                dataset = handle.create_dataset(channel, data=np.asarray(strain, dtype=float))
                dataset.attrs["x0"] = float(config.output.gps_start)
                dataset.attrs["dx"] = 1.0 / float(config.sampling_frequency)
                dataset.attrs["xunit"] = "s"
                dataset.attrs["channel"] = channel
                dataset.attrs["name"] = channel
                dataset.attrs["unit"] = "strain"
            # Declared through the same call every other writer uses, rather than composed inline here:
            # one mechanism is one place for the layout to be decided, and gwmock writes this artifact
            # through three different libraries. See `gwmock.strain_schema`.
            declare_strain_schema(output_path)
            output_paths[detector] = output_path
        return output_paths

    @staticmethod
    def _declare_backend_outputs(result: SimulationResult) -> SimulationResult:
        """Declare the strain schema on artifacts the backend wrote, and return the result unchanged.

        `write_chunk` declares as it writes, but a backend that owns the whole run writes the file itself
        and knows nothing of gwmock's contract -- so the two paths would otherwise disagree about what a
        gwmock artifact carries, and `run()` is the path a real run takes. Formats with no attribute
        space are skipped by `declare_strain_schema` itself.

        Args:
            result: The result returned by the backend.

        Returns:
            The same result.
        """
        for output_path in result.output_paths.values():
            declare_strain_schema(output_path)
        return result

    def _normalize_chunk(self, *, chunk: Mapping[str, np.ndarray], detectors: Sequence[str]) -> dict[str, np.ndarray]:
        """Validate and normalize one upstream chunk.

        Args:
            chunk: The chunk to normalize.
            detectors: The detectors to use.

        Returns:
            The normalized chunk.
        """
        normalized: dict[str, np.ndarray] = {}
        for detector in detectors:
            if detector not in chunk:
                raise ValueError(f"Noise stream did not produce detector '{detector}'.")
            normalized[detector] = np.asarray(chunk[detector])
        return normalized

    def _resolve_stream_backend(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None,
        psd_file: str | Path | None,
        psd_schedule: list[tuple[float, str | Path]] | None,
        psd_files: dict[str, str | Path] | None,
        csd_files: dict[str, str | Path] | None,
        low_frequency_cutoff: float,
        high_frequency_cutoff: float | None,
        spectral_lines: list[Any] | None,
        glitches: list[Any] | None,
    ) -> NoiseSimulator:
        """Return the protocol-compatible backend for ``open_stream``.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The protocol-compatible backend.
        """
        if isinstance(self._backend, DefaultNoiseSimulator):
            protocol_backend = self._configure_default_stream_backend(
                chunk_duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                detectors=detectors,
                seed=seed,
                psd_file=psd_file,
                psd_schedule=psd_schedule,
                psd_files=psd_files,
                csd_files=csd_files,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
                spectral_lines=spectral_lines,
                glitches=glitches,
            )
            if protocol_backend is not None:
                return protocol_backend

        if isinstance(self._backend, NoiseSimulator):
            return self._backend

        raise TypeError("Noise backend must satisfy the gwmock_noise NoiseSimulator protocol to open a stream.")

    def _configure_default_stream_backend(  # noqa: PLR0913
        self,
        *,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None,
        psd_file: str | Path | None,
        psd_schedule: list[tuple[float, str | Path]] | None,
        psd_files: dict[str, str | Path] | None,
        csd_files: dict[str, str | Path] | None,
        low_frequency_cutoff: float,
        high_frequency_cutoff: float | None,
        spectral_lines: list[Any] | None,
        glitches: list[Any] | None,
    ) -> NoiseSimulator | None:
        """Mirror the default gwmock-noise backend selection with protocol simulators.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
            psd_file: The PSD file.
            psd_schedule: The PSD schedule.
            psd_files: The PSD files.
            csd_files: The CSD files.
            low_frequency_cutoff: The low frequency cutoff.
            high_frequency_cutoff: The high frequency cutoff.
            spectral_lines: The spectral lines.
            glitches: The glitches.

        Returns:
            The protocol-compatible backend.
        """
        # Reset any glitch models from a previous stream so resolved_config()
        # never reports stale models when this adapter is reused for a later
        # stream that has no glitches.
        self._glitch_models = None
        self._glitch_injector = None
        self._resolved_config_cache = None

        normalized_psd_files = _coerce_path_mapping(psd_files)
        normalized_csd_files = _coerce_path_mapping(csd_files)
        normalized_psd_schedule = _coerce_path_schedule(psd_schedule)
        normalized_psd_file = _coerce_path(psd_file)

        simulator: NoiseSimulator | None = None
        if normalized_psd_files is not None or normalized_csd_files is not None:
            simulator = CorrelatedNoiseSimulator(
                psd_files=normalized_psd_files or {},
                csd_files=_parse_csd_file_map(normalized_csd_files),
                detectors=detectors,
                duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                seed=seed,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
            )
        elif normalized_psd_file is not None or normalized_psd_schedule is not None:
            simulator = ColoredNoiseSimulator(
                psd_file=normalized_psd_file,
                psd_schedule=normalized_psd_schedule,
                detectors=detectors,
                duration=chunk_duration,
                sampling_frequency=sampling_frequency,
                seed=seed,
                low_frequency_cutoff=low_frequency_cutoff,
                high_frequency_cutoff=high_frequency_cutoff,
            )

        if spectral_lines is not None:
            if not spectral_lines:
                raise ValueError("spectral_lines must contain at least one spectral line.")
            normalized_lines = normalize_spectral_lines(spectral_lines)
            simulator = (
                SpectralLineSimulator(
                    lines=normalized_lines,
                    detectors=detectors,
                    duration=chunk_duration,
                    sampling_frequency=sampling_frequency,
                    seed=seed,
                )
                if simulator is None
                else AddLines(simulator, normalized_lines)
            )

        if glitches is not None:
            if not glitches:
                raise ValueError("glitches must contain at least one glitch model.")
            resolved_glitches = _resolve_glitch_file_urls(glitches)
            if simulator is None:
                simulator = _ZeroNoiseSimulator(
                    detectors=detectors,
                    duration=chunk_duration,
                    sampling_frequency=sampling_frequency,
                    seed=seed,
                )
            glitch_models = normalize_glitch_models(resolved_glitches)
            self._glitch_models = glitch_models
            simulator = InjectGlitches(simulator, glitch_models)
            self._glitch_injector = simulator

        return simulator

    def resolved_config(self) -> dict[str, Any]:
        """Return the runtime-resolved, config-shaped noise arguments.

        Currently reports the glitch models with every external, mutable
        dependency pinned to an immutable id (e.g. a DeepExtractor dataset
        pinned to a concrete Hugging Face commit). Each model is resolved and
        re-serialized, so the returned entries round-trip back through
        ``normalize_glitch_models`` on replay and reproduce the exact resources
        the run used. Returns an empty mapping when the active stream has no
        glitches, so a caller can treat "nothing resolved" uniformly. The result
        is memoized for the active stream so repeated per-batch metadata writes
        do not re-resolve.
        """
        if self._resolved_config_cache is not None:
            return self._resolved_config_cache
        if not self._glitch_models:
            return {}
        for model in self._glitch_models:
            model.resolve()
        self._resolved_config_cache = {"glitches": [model.serialize() for model in self._glitch_models]}
        return self._resolved_config_cache

    def set_segment_gps_start(self, gps_start: float) -> None:
        """Tell the active glitch injector where the next chunk sits in GPS time.

        gwmock owns the epoch of each batch -- it is what the frame's name, its ``t0``
        and its metadata record carry -- so the injector is handed that number rather
        than left to accumulate its own. Threading the one value through is what keeps
        a glitch's recorded time and the frame it was written into describing the same
        instant, including for a resumed run whose batches are not generated in order.

        A no-op when the active stream has no glitches, and when the installed
        ``gwmock-noise`` predates the truth catalogue: the epoch is only consumed by the
        catalogue, so there is nothing to set and nothing to warn about here.

        Args:
            gps_start: GPS time of the first sample of the chunk generated next.
        """
        if self._glitch_injector is None:
            return
        if hasattr(self._glitch_injector, "gps_start"):
            self._glitch_injector.gps_start = float(gps_start)

    def segment_glitch_events(self) -> list[dict[str, Any]]:
        """Return the glitches the most recently generated chunk injected.

        Read per batch rather than accumulated here, because the injector already keeps
        the run's catalogue and a second copy alongside it would be a second thing to
        keep correct. Each row describes a glitch the last chunk *started*, so a batch's
        rows are the events attributable to the frame(s) it is about to write.

        Returns:
            One mapping per injected glitch, in time order. Empty when the active stream
            has no glitches, when the last chunk fired none, or when the installed
            ``gwmock-noise`` is too old to report them -- see
            :meth:`glitch_catalogue_description`, which is what tells the last case apart
            from the first two.
        """
        if self._glitch_injector is None:
            return []
        events = getattr(self._glitch_injector, "segment_glitch_events", None)
        if events is None:
            return []
        return [dict(event) for event in events]

    def glitch_catalogue_description(self) -> dict[str, Any] | None:
        """Return the catalogue's schema version, time convention and column descriptions.

        Recorded beside the rows so a file says what its own columns mean -- in particular
        which of the two time columns is the waveform's start and which its peak, a
        distinction no column name carries on its own.

        Returns:
            The description reported by the active injector, or ``None`` when the stream
            has no glitches or the installed ``gwmock-noise`` predates the catalogue. A
            ``None`` here beside an empty row list is how a consumer tells "this run could
            not record its glitches" from "this run injected none".
        """
        if self._glitch_injector is None:
            return None
        catalogue = (self._glitch_injector.metadata.get("glitches") or {}).get("catalogue")
        if catalogue is None:
            return None
        return {key: value for key, value in catalogue.items() if key != "events"}


class _ChunkNoiseSimulator:
    """Protocol adapter that replays one already-generated chunk."""

    def __init__(self, chunk: Mapping[str, np.ndarray]) -> None:
        """Initialize the chunk noise simulator.

        Args:
            chunk: The chunk to replay.
        """
        self._chunk = {detector: np.asarray(strain) for detector, strain in chunk.items()}
        self.detectors = list(self._chunk)
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.seed = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the stored chunk after validating the requested shape.

        Args:
            duration: The duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            The generated chunk.

        Raises:
            ValueError: If the noise stream did not produce the requested detector.
            ValueError: If the noise chunk for the requested detector has the wrong number of samples.
        """
        _ = seed
        expected_samples = round(duration * sampling_frequency)
        generated: dict[str, np.ndarray] = {}
        for detector in detectors:
            if detector not in self._chunk:
                raise ValueError(f"Noise stream did not produce detector '{detector}'.")
            strain = self._chunk[detector]
            if strain.shape[0] != expected_samples:
                raise ValueError(
                    f"Noise chunk for detector '{detector}' has {strain.shape[0]} samples; expected {expected_samples}."
                )
            generated[detector] = strain
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        return generated

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield the stored chunk once.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            An iterator over the generated chunk.
        """
        yield self.generate(chunk_duration, sampling_frequency, detectors, seed)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata layered on the chunk replay backend.

        Returns:
            The adapter metadata.
        """
        return {"adapter": "chunk-replay"}


class _ZeroNoiseSimulator:
    """Protocol-compatible zero-noise backend used for glitches-only streams."""

    def __init__(
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        seed: int | None,
    ) -> None:
        """Initialize the zero noise simulator.

        Args:
            detectors: The detectors to use.
            duration: The duration.
            sampling_frequency: The sampling frequency.
            seed: The seed.
        """
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return zeros with the requested runtime shape.

        Args:
            duration: The duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.
        """
        _ = seed
        n_samples = round(duration * sampling_frequency)
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield zero-noise chunks lazily.

        Args:
            chunk_duration: The chunk duration.
            sampling_frequency: The sampling frequency.
            detectors: The detectors to use.
            seed: The seed.

        Returns:
            An iterator over the generated chunk.
        """
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata for the zero-noise helper backend.

        Returns:
            The metadata.
        """
        return {"kind": "zero-noise"}
