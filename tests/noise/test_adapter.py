"""Tests for the gwmock-side gwmock_noise adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from gwmock.noise import NoiseAdapter
from gwmock.noise import adapter as noise_adapter
from gwmock.strain_schema import STRAIN_SCHEMA_VERSION, require_strain_schema

TEST_DURATION = 8.0
TEST_SAMPLING_FREQUENCY = 256.0
TEST_GPS_START = 1234.5
TEST_SEED = 11


class FakeNoiseBackend:
    """Minimal run-style gwmock-noise backend for direct adapter tests."""

    def __init__(self) -> None:
        self.run_calls = []

    def run(self, config):
        """Record the config and materialize the declared outputs."""
        self.run_calls.append(config)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        output_paths = {}
        for detector in config.detectors:
            artifact_path = config.output.directory / f"{config.output.prefix}_{detector}.npy"
            artifact_path.write_text(f"{detector}:{config.output.format}")
            output_paths[detector] = artifact_path
        return type("SimulationResultStub", (), {"output_paths": output_paths, "config": config})()


class FakeStreamNoiseBackend:
    """Protocol-style backend that exposes one stateful chunk iterator."""

    def __init__(self) -> None:
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.detectors = ["H1", "L1"]
        self.seed = None
        self.stream_open_calls = []
        self.chunk_index = 0

    def generate(self, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None):
        """Return one deterministic chunk."""
        _ = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.full(n_samples, self.chunk_index, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        """Yield deterministic chunks while recording one stream-open event."""
        self.stream_open_calls.append(
            {
                "chunk_duration": chunk_duration,
                "sampling_frequency": sampling_frequency,
                "detectors": list(detectors),
                "seed": seed,
            }
        )
        while True:
            n_samples = round(chunk_duration * sampling_frequency)
            value = self.chunk_index
            self.chunk_index += 1
            yield {detector: np.full(n_samples, value, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, object]:
        """Return fake backend metadata."""
        return {"kind": "fake-stream-noise"}


def _psd_file(tmp_path: Path) -> Path:
    """Create a simple PSD file for reproducibility tests."""
    freqs = np.linspace(1, 64, 64)
    psd_values = np.ones_like(freqs)
    psd_path = tmp_path / "psd.txt"
    np.savetxt(psd_path, np.column_stack([freqs, psd_values]))
    return psd_path


class TestNoiseAdapter:
    """Tests for direct adapter behavior."""

    def test_run_builds_public_noise_config(self, tmp_path: Path):
        """The adapter should pass gwmock orchestration inputs through NoiseConfig."""
        backend = FakeNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        psd_path = _psd_file(tmp_path)

        result = adapter.run(
            detectors=["H1", "L1"],
            duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            output_directory=tmp_path,
            output_prefix="segment-0",
            output_format="npy",
            gps_start=TEST_GPS_START,
            channel="TEST",
            seed=TEST_SEED,
            psd_file=psd_path,
            low_frequency_cutoff=10.0,
            high_frequency_cutoff=100.0,
        )

        config = backend.run_calls[0]
        assert config.detectors == ["H1", "L1"]
        assert config.duration == TEST_DURATION
        assert config.sampling_frequency == TEST_SAMPLING_FREQUENCY
        assert config.output.directory == tmp_path
        assert config.output.prefix == "segment-0"
        assert config.output.format == "npy"
        assert config.output.gps_start == TEST_GPS_START
        assert config.output.channel == "TEST"
        assert config.seed == TEST_SEED
        assert len(config.components) == 1
        assert config.components[0].simulator == "colored"
        assert config.components[0].options["psd_file"] == psd_path
        assert config.components[0].options["low_frequency_cutoff"] == 10.0
        assert config.components[0].options["high_frequency_cutoff"] == 100.0
        assert result.output_paths["H1"] == tmp_path / "segment-0_H1.npy"

    def test_open_stream_uses_one_upstream_iterator(self):
        """The adapter should open one shared stream and consume it chunk by chunk."""
        backend = FakeStreamNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)

        stream = adapter.open_stream(
            chunk_duration=4.0,
            sampling_frequency=8.0,
            detectors=["H1", "L1"],
            seed=7,
        )

        first_chunk = next(stream)
        second_chunk = next(stream)

        assert backend.stream_open_calls == [
            {
                "chunk_duration": 4.0,
                "sampling_frequency": 8.0,
                "detectors": ["H1", "L1"],
                "seed": 7,
            }
        ]
        assert np.all(first_chunk["H1"] == 0.0)
        assert np.all(second_chunk["L1"] == 1.0)

    def test_write_chunk_persists_numpy_outputs(self, tmp_path: Path):
        """The adapter should let gwmock own NumPy chunk output writing."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="npy",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(
            config=config,
            chunk={
                "H1": np.arange(32, dtype=float),
                "L1": np.arange(32, dtype=float) + 1.0,
            },
        )

        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H1.npy",
            tmp_path / "noise-0_L1.npy",
        ]
        assert np.array_equal(np.load(result.output_paths["H1"]), np.arange(32, dtype=float))
        assert np.array_equal(np.load(result.output_paths["L1"]), np.arange(32, dtype=float) + 1.0)

    def test_expected_output_paths_for_gwf(self, tmp_path: Path):
        """GWF expected paths should match gwmock-noise FrameWriter naming.

        The name carries the channel without its ``IFO:`` prefix. A colon is reserved on NTFS, so a
        frame named with one was unwritable on Windows, and gwmock-noise drops it -- which this
        predicted a path for without following.
        """
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="gwf",
            # A whole second: gwmock-noise rejects a fractional gps_start for gwf, because the
            # artifact name carries the time as an integer and two runs whose times round alike
            # would compose one name.
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H-H1_MOCK_NOISE_100-4.gwf",
            tmp_path / "noise-0_L-L1_MOCK_NOISE_100-4.gwf",
        ]

    def test_multisegment_outputs_match_single_long_run_with_stateful_backend(self, tmp_path: Path):
        """Concatenated chunk outputs should match one long protocol run for the same stream."""
        backend = FakeStreamNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        stream = adapter.open_stream(
            chunk_duration=2.0,
            sampling_frequency=4.0,
            detectors=["H1"],
            seed=13,
        )

        segments = []
        for index in range(5):
            config = adapter.build_config(
                detectors=["H1"],
                duration=2.0,
                sampling_frequency=4.0,
                output_directory=tmp_path,
                output_prefix=f"noise-{index}",
                output_format="npy",
                gps_start=100.0 + index * 2.0,
                channel="MOCK_NOISE",
                seed=13,
            )
            result = adapter.write_chunk(config=config, chunk=next(stream))
            segments.append(np.load(result.output_paths["H1"]))

        concatenated = np.concatenate(segments)
        expected = np.concatenate([np.full(8, value, dtype=float) for value in range(5)])
        assert concatenated.tobytes() == expected.tobytes()

    def test_per_detector_channels_config_and_expected_paths(self, tmp_path: Path):
        """build_config stores channels dict and expected_output_paths uses per-detector names."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="gwf",
            gps_start=100.0,
            channel="STRAIN_NOISE",
            channels={"H1": "H1:STRAIN_NOISE", "L1": "L1:STRAIN_NOISE"},
            seed=7,
        )

        assert config.output.channel == "STRAIN_NOISE"
        assert config.output.channels == {"H1": "H1:STRAIN_NOISE", "L1": "L1:STRAIN_NOISE"}
        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H-H1_STRAIN_NOISE_100-4.gwf",
            tmp_path / "noise-0_L-L1_STRAIN_NOISE_100-4.gwf",
        ]


def _blip_glitch_dict() -> dict[str, object]:
    """Return a valid dict-form blip glitch config."""
    return {
        "kind": "blip",
        "rate": 1.0,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1e-21, "std": 0.0},
        "width": 0.01,
    }


class TestDefaultStreamBackendComponentNormalization:
    """Regression tests for ISS-011.

    The default stream backend builds ``SpectralLineSimulator`` / ``AddLines`` /
    ``InjectGlitches`` directly, so the adapter must normalize dict-form config
    entries into ``SpectralLine`` / ``GlitchModel`` instances first. Before the
    fix these raised ``AttributeError: 'dict' object has no attribute ...`` on the
    first chunk.
    """

    def test_open_stream_accepts_dict_form_spectral_lines(self):
        """A dict-form spectral_lines entry generates a chunk (no AttributeError)."""
        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            spectral_lines=[{"frequency": 60.0, "amplitude": 1e-23}],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))
        assert np.any(chunk["H1"] != 0.0)

    def test_open_stream_accepts_dict_form_glitches(self):
        """A dict-form glitches entry generates a chunk (no AttributeError)."""
        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            glitches=[_blip_glitch_dict()],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))

    def test_open_stream_accepts_dataclass_instances(self):
        """Passing SpectralLine/GlitchModel instances still works (normalize is idempotent)."""
        from gwmock_noise.gaussian import SpectralLine
        from gwmock_noise.glitches.models import BlipGlitch, LogNormalAmplitudeDistribution

        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        stream = adapter.open_stream(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["H1"],
            seed=TEST_SEED,
            spectral_lines=[SpectralLine(frequency=60.0, amplitude=1e-23)],
            glitches=[BlipGlitch(rate=1.0, amplitude_distribution=LogNormalAmplitudeDistribution(mean=1e-21))],
        )

        chunk = next(stream)

        n_samples = round(TEST_DURATION * TEST_SAMPLING_FREQUENCY)
        assert chunk["H1"].shape == (n_samples,)
        assert np.all(np.isfinite(chunk["H1"]))


def _gengli_glitch_dict(population_file: str) -> dict[str, object]:
    """Return a dict-form gengli_blip glitch config with the given population_file."""
    return {
        "kind": "gengli_blip",
        "rate": 1.0,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "population_file": population_file,
        "psd_file": "ET_10_full_cryo_psd",
        "low_frequency_cutoff": 5.0,
    }


class TestGlitchPopulationUrlResolution:
    """Regression tests for ISS-012.

    A glitch ``population_file`` given as an ``http(s)`` URL must be downloaded to a
    local cache and replaced by the local path before it reaches the gwmock-noise
    glitch model (which reads it with ``Path``/``h5py`` and cannot handle URLs).
    Named bundled assets and local paths must pass through untouched.
    """

    def test_resolve_downloads_remote_population_file(self, monkeypatch, tmp_path: Path):
        """A URL population_file is downloaded and replaced with the local path."""
        local = tmp_path / "blip.hdf5"
        local.write_bytes(b"")
        calls = []

        def fake_download(url, **kwargs):
            calls.append((url, kwargs))
            return local

        monkeypatch.setattr(noise_adapter, "download_file", fake_download)

        url = "https://example.org/records/1/files/blip.hdf5"
        glitches = [_gengli_glitch_dict(url)]
        resolved = noise_adapter._resolve_glitch_file_urls(glitches)

        # URL replaced by the local path; named PSD asset untouched.
        assert resolved[0]["population_file"] == str(local)
        assert resolved[0]["psd_file"] == "ET_10_full_cryo_psd"
        # Original input is not mutated.
        assert glitches[0]["population_file"] == url
        # Downloaded exactly once, with caching enabled.
        assert len(calls) == 1
        assert calls[0][0] == url
        assert calls[0][1]["allow_existing"] is True
        assert calls[0][1]["dest_path_from_hashed_url"] is True

    def test_resolve_passes_through_non_url_values(self, monkeypatch, tmp_path: Path):
        """Local paths and named assets are returned unchanged without downloading."""

        def fail_download(*args, **kwargs):
            raise AssertionError("download_file must not be called for non-URL inputs")

        monkeypatch.setattr(noise_adapter, "download_file", fail_download)

        local_population = tmp_path / "local_blip.hdf5"
        glitches = [_gengli_glitch_dict(str(local_population)), _blip_glitch_dict()]
        resolved = noise_adapter._resolve_glitch_file_urls(glitches)

        assert resolved == glitches

    def test_resolve_returns_empty_input_unchanged(self):
        """``None``/empty glitch lists are returned as-is."""
        assert noise_adapter._resolve_glitch_file_urls(None) is None
        assert noise_adapter._resolve_glitch_file_urls([]) == []

    def test_configure_default_stream_backend_resolves_urls(self, monkeypatch, tmp_path: Path):
        """The streaming backend hands the local path (not the URL) to gwmock-noise."""
        local = tmp_path / "blip.hdf5"
        local.write_bytes(b"")
        monkeypatch.setattr(noise_adapter, "download_file", lambda url, **kwargs: local)

        received = {}

        def fake_normalize(glitches):
            received["glitches"] = glitches
            return ["sentinel-model"]

        monkeypatch.setattr(noise_adapter, "normalize_glitch_models", fake_normalize)

        adapter = NoiseAdapter.from_backend()  # DefaultNoiseSimulator
        url = "https://example.org/records/1/files/blip.hdf5"
        adapter._configure_default_stream_backend(
            chunk_duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            detectors=["E1"],
            seed=TEST_SEED,
            psd_file=None,
            psd_schedule=None,
            psd_files=None,
            csd_files=None,
            low_frequency_cutoff=2.0,
            high_frequency_cutoff=None,
            spectral_lines=None,
            glitches=[_gengli_glitch_dict(url)],
        )

        assert received["glitches"][0]["population_file"] == str(local)


TEST_LOW_FREQUENCY_CUTOFF = 7.5
TEST_HIGH_FREQUENCY_CUTOFF = 99.0


class _RecordingSimulator:
    """Base for the recording stand-ins below."""

    def generate(self, duration, sampling_frequency, detectors, seed=None):
        """Return one zero chunk so a recorded simulator can still be streamed."""
        _ = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples) for detector in detectors}


class FakeCorrelatedSimulator(_RecordingSimulator):
    """Stand-in for ``CorrelatedNoiseSimulator`` that records its arguments.

    Every parameter is keyword-only and has no default, so dropping one from the
    call site is a ``TypeError`` rather than a silently different simulator.
    """

    def __init__(
        self,
        *,
        psd_files,
        csd_files,
        detectors,
        duration,
        sampling_frequency,
        seed,
        low_frequency_cutoff,
        high_frequency_cutoff,
    ) -> None:
        self.kwargs = {
            "psd_files": psd_files,
            "csd_files": csd_files,
            "detectors": detectors,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "seed": seed,
            "low_frequency_cutoff": low_frequency_cutoff,
            "high_frequency_cutoff": high_frequency_cutoff,
        }


class FakeColoredSimulator(_RecordingSimulator):
    """Stand-in for ``ColoredNoiseSimulator`` that records its arguments."""

    def __init__(
        self,
        *,
        psd_file,
        psd_schedule,
        detectors,
        duration,
        sampling_frequency,
        seed,
        low_frequency_cutoff,
        high_frequency_cutoff,
    ) -> None:
        self.kwargs = {
            "psd_file": psd_file,
            "psd_schedule": psd_schedule,
            "detectors": detectors,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "seed": seed,
            "low_frequency_cutoff": low_frequency_cutoff,
            "high_frequency_cutoff": high_frequency_cutoff,
        }


class FakeSpectralLineSimulator(_RecordingSimulator):
    """Stand-in for ``SpectralLineSimulator`` that records its arguments."""

    def __init__(self, *, lines, detectors, duration, sampling_frequency, seed) -> None:
        self.kwargs = {
            "lines": lines,
            "detectors": detectors,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "seed": seed,
        }


class FakeZeroNoiseSimulator(_RecordingSimulator):
    """Stand-in for ``_ZeroNoiseSimulator`` that records its arguments."""

    def __init__(self, *, detectors, duration, sampling_frequency, seed) -> None:
        self.kwargs = {
            "detectors": detectors,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
            "seed": seed,
        }


class FakeAddLines(_RecordingSimulator):
    """Stand-in for ``AddLines`` that records the wrapped simulator and lines."""

    def __init__(self, base, lines) -> None:
        self.base = base
        self.lines = lines


class FakeInjectGlitches(_RecordingSimulator):
    """Stand-in for ``InjectGlitches`` that records the wrapped simulator and models."""

    def __init__(self, base, models) -> None:
        self.base = base
        self.models = models


@pytest.fixture
def recorded_simulators(monkeypatch):
    """Replace the gwmock-noise simulators the adapter builds with recording fakes.

    The adapter mirrors gwmock-noise's own backend selection, so what is under test
    here is which class it picks and what it passes -- not what those classes then
    compute. Recording fakes make both observable, and their argument-less-free
    signatures turn a dropped argument into a failure instead of a default.
    """
    monkeypatch.setattr(noise_adapter, "CorrelatedNoiseSimulator", FakeCorrelatedSimulator)
    monkeypatch.setattr(noise_adapter, "ColoredNoiseSimulator", FakeColoredSimulator)
    monkeypatch.setattr(noise_adapter, "SpectralLineSimulator", FakeSpectralLineSimulator)
    monkeypatch.setattr(noise_adapter, "_ZeroNoiseSimulator", FakeZeroNoiseSimulator)
    monkeypatch.setattr(noise_adapter, "AddLines", FakeAddLines)
    monkeypatch.setattr(noise_adapter, "InjectGlitches", FakeInjectGlitches)


def _configure(adapter, **overrides):
    """Call ``_configure_default_stream_backend`` with every argument spelled out."""
    arguments = {
        "chunk_duration": TEST_DURATION,
        "sampling_frequency": TEST_SAMPLING_FREQUENCY,
        "detectors": ["E1", "E2"],
        "seed": TEST_SEED,
        "psd_file": None,
        "psd_schedule": None,
        "psd_files": None,
        "csd_files": None,
        "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
        "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
        "spectral_lines": None,
        "glitches": None,
    }
    arguments.update(overrides)
    return adapter._configure_default_stream_backend(**arguments)


@pytest.mark.usefixtures("recorded_simulators")
class TestDefaultStreamBackendSelection:
    """Which simulator the streaming path builds, and with which arguments.

    This mirrors gwmock-noise's backend selection by hand, so a wrong branch or a
    dropped argument produces a stream that runs and returns plausible noise --
    the wrong noise. Nothing downstream raises, which is why the selection and the
    argument hand-off are pinned here rather than inferred from a generated chunk.
    """

    def test_psd_files_build_a_correlated_simulator_with_every_argument(self):
        """``psd_files`` selects the correlated simulator and forwards each argument."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(
            adapter,
            psd_files={"E1": "e1.txt", "E2": "e2.txt"},
            csd_files={"E2-E1": "e1e2.txt"},
        )

        assert isinstance(simulator, FakeCorrelatedSimulator)
        assert simulator.kwargs == {
            "psd_files": {"E1": Path("e1.txt"), "E2": Path("e2.txt")},
            # Keys are normalized to sorted detector-pair tuples.
            "csd_files": {("E1", "E2"): Path("e1e2.txt")},
            "detectors": ["E1", "E2"],
            "duration": TEST_DURATION,
            "sampling_frequency": TEST_SAMPLING_FREQUENCY,
            "seed": TEST_SEED,
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
            "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
        }

    def test_csd_files_alone_still_build_a_correlated_simulator(self):
        """CSD files without PSD files select the correlated simulator with an empty PSD map."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, csd_files={"E1-E2": "e1e2.txt"})

        assert isinstance(simulator, FakeCorrelatedSimulator)
        assert simulator.kwargs["psd_files"] == {}
        assert simulator.kwargs["csd_files"] == {("E1", "E2"): Path("e1e2.txt")}

    def test_psd_file_builds_a_colored_simulator_with_every_argument(self):
        """A single PSD file selects the colored simulator and forwards each argument."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, psd_file="psd.txt")

        assert isinstance(simulator, FakeColoredSimulator)
        assert simulator.kwargs == {
            "psd_file": Path("psd.txt"),
            "psd_schedule": None,
            "detectors": ["E1", "E2"],
            "duration": TEST_DURATION,
            "sampling_frequency": TEST_SAMPLING_FREQUENCY,
            "seed": TEST_SEED,
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
            "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
        }

    def test_psd_schedule_alone_builds_a_colored_simulator(self):
        """A PSD schedule without a PSD file still selects the colored simulator."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, psd_schedule=[(0.0, "early.txt"), (4.0, "late.txt")])

        assert isinstance(simulator, FakeColoredSimulator)
        assert simulator.kwargs["psd_file"] is None
        assert simulator.kwargs["psd_schedule"] == [(0.0, Path("early.txt")), (4.0, Path("late.txt"))]

    def test_no_noise_inputs_leave_the_upstream_default_in_place(self):
        """With nothing to configure the adapter builds no simulator of its own."""
        adapter = NoiseAdapter.from_backend()

        assert _configure(adapter) is None

    def test_spectral_lines_alone_build_a_spectral_line_simulator(self):
        """Lines without a PSD source select the line-only simulator."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, spectral_lines=[{"frequency": 60.0, "amplitude": 1e-23}])

        assert isinstance(simulator, FakeSpectralLineSimulator)
        assert simulator.kwargs["detectors"] == ["E1", "E2"]
        assert simulator.kwargs["duration"] == TEST_DURATION
        assert simulator.kwargs["sampling_frequency"] == TEST_SAMPLING_FREQUENCY
        assert simulator.kwargs["seed"] == TEST_SEED
        assert [line.frequency for line in simulator.kwargs["lines"]] == [60.0]

    def test_spectral_lines_wrap_an_existing_simulator(self):
        """With a PSD source present the lines are added on top of it, not instead of it."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(
            adapter,
            psd_file="psd.txt",
            spectral_lines=[{"frequency": 60.0, "amplitude": 1e-23}],
        )

        assert isinstance(simulator, FakeAddLines)
        assert isinstance(simulator.base, FakeColoredSimulator)
        assert [line.frequency for line in simulator.lines] == [60.0]

    def test_empty_spectral_lines_are_rejected(self):
        """An empty list is a config mistake, not "no lines"."""
        adapter = NoiseAdapter.from_backend()

        with pytest.raises(ValueError, match=r"^spectral_lines must contain at least one spectral line\.$"):
            _configure(adapter, spectral_lines=[])

    def test_glitches_alone_are_injected_into_zero_noise(self):
        """Glitches without a noise source are injected into a zero-noise stream."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, glitches=[_blip_glitch_dict()])

        assert isinstance(simulator, FakeInjectGlitches)
        assert isinstance(simulator.base, FakeZeroNoiseSimulator)
        assert simulator.base.kwargs == {
            "detectors": ["E1", "E2"],
            "duration": TEST_DURATION,
            "sampling_frequency": TEST_SAMPLING_FREQUENCY,
            "seed": TEST_SEED,
        }
        assert simulator.models == adapter._glitch_models

    def test_glitches_wrap_an_existing_simulator(self):
        """With a PSD source present the glitches are injected into it."""
        adapter = NoiseAdapter.from_backend()

        simulator = _configure(adapter, psd_file="psd.txt", glitches=[_blip_glitch_dict()])

        assert isinstance(simulator, FakeInjectGlitches)
        assert isinstance(simulator.base, FakeColoredSimulator)
        assert len(simulator.models) == 1

    def test_empty_glitches_are_rejected(self):
        """An empty list is a config mistake, not "no glitches"."""
        adapter = NoiseAdapter.from_backend()

        with pytest.raises(ValueError, match=r"^glitches must contain at least one glitch model\.$"):
            _configure(adapter, glitches=[])

    def test_a_reused_adapter_forgets_the_previous_stream_glitch_models(self):
        """A later glitch-free stream must not report the earlier stream's models."""
        adapter = NoiseAdapter.from_backend()
        _configure(adapter, glitches=[_blip_glitch_dict()])
        assert adapter.resolved_config()["glitches"]

        _configure(adapter, psd_file="psd.txt")

        assert adapter._glitch_models is None
        assert adapter.resolved_config() == {}


class TestBuildComponents:
    """The flat-field to component-list translation.

    ``_build_components`` decides which upstream simulator each legacy field maps to
    and under which option name. An option written under the wrong key is dropped
    silently by the upstream config rather than rejected, so the exact option
    mapping is asserted here.
    """

    def test_psd_files_become_a_correlated_component(self):
        """PSD and CSD files map to one correlated component carrying both cutoffs."""
        components = noise_adapter._build_components(
            psd_file=None,
            psd_schedule=None,
            psd_files={"E1": Path("e1.txt")},
            csd_files={"E1-E2": Path("e1e2.txt")},
            low_frequency_cutoff=TEST_LOW_FREQUENCY_CUTOFF,
            high_frequency_cutoff=TEST_HIGH_FREQUENCY_CUTOFF,
            spectral_lines=None,
            glitches=None,
        )

        assert len(components) == 1
        assert components[0].simulator == "correlated"
        assert components[0].options == {
            "psd_files": {"E1": Path("e1.txt")},
            "csd_files": {"E1-E2": Path("e1e2.txt")},
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
            "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
        }

    def test_an_absent_high_frequency_cutoff_is_left_out(self):
        """``None`` means "no cutoff", which upstream expresses as an absent option."""
        components = noise_adapter._build_components(
            psd_file=None,
            psd_schedule=None,
            psd_files={"E1": Path("e1.txt")},
            csd_files=None,
            low_frequency_cutoff=TEST_LOW_FREQUENCY_CUTOFF,
            high_frequency_cutoff=None,
            spectral_lines=None,
            glitches=None,
        )

        assert components[0].options == {
            "psd_files": {"E1": Path("e1.txt")},
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
        }

    def test_a_psd_file_becomes_a_colored_component(self):
        """A single PSD file (or a schedule) maps to one colored component."""
        components = noise_adapter._build_components(
            psd_file=Path("psd.txt"),
            psd_schedule=[(0.0, Path("late.txt"))],
            psd_files=None,
            csd_files=None,
            low_frequency_cutoff=TEST_LOW_FREQUENCY_CUTOFF,
            high_frequency_cutoff=TEST_HIGH_FREQUENCY_CUTOFF,
            spectral_lines=None,
            glitches=None,
        )

        assert len(components) == 1
        assert components[0].simulator == "colored"
        assert components[0].options == {
            "psd_file": Path("psd.txt"),
            "psd_schedule": [(0.0, Path("late.txt"))],
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
            "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
        }

    def test_lines_and_glitches_append_their_own_components(self):
        """Lines and glitches are components in their own right, after the noise source."""
        components = noise_adapter._build_components(
            psd_file=Path("psd.txt"),
            psd_schedule=None,
            psd_files=None,
            csd_files=None,
            low_frequency_cutoff=TEST_LOW_FREQUENCY_CUTOFF,
            high_frequency_cutoff=None,
            spectral_lines=["line"],
            glitches=["glitch"],
        )

        assert [component.simulator for component in components] == ["colored", "spectral_lines", "glitches"]
        assert components[1].options == {"lines": ["line"]}
        assert components[2].options == {"models": ["glitch"]}

    def test_no_noise_fields_produce_no_components(self):
        """Nothing configured means nothing to translate."""
        assert (
            noise_adapter._build_components(
                psd_file=None,
                psd_schedule=None,
                psd_files=None,
                csd_files=None,
                low_frequency_cutoff=TEST_LOW_FREQUENCY_CUTOFF,
                high_frequency_cutoff=None,
                spectral_lines=None,
                glitches=None,
            )
            == []
        )


class FakeFrameWriter:
    """Stand-in for ``FrameWriter`` that records how the adapter drives it."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, simulator, *, gps_start, output_dir, channel, channels, prefix) -> None:
        self.record = {
            "simulator": simulator,
            "gps_start": gps_start,
            "output_dir": output_dir,
            "channel": channel,
            "channels": channels,
            "prefix": prefix,
        }

    def write(self, *, duration, sampling_frequency, detectors, seed):
        """Record the write arguments and return one path per detector."""
        self.record.update(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )
        FakeFrameWriter.calls.append(self.record)
        return {detector: self.record["output_dir"] / f"{detector}.gwf" for detector in detectors}


class TestWriteChunkFrameOutput:
    """How ``write_chunk`` hands a chunk to the frame writer.

    The GWF path is selected by an exact format string and then replays the chunk
    through gwmock-noise's ``FrameWriter``. Getting the duration or the detector
    list wrong there writes frames that are readable but describe the wrong data,
    so the arguments are pinned rather than inferred from the file names.
    """

    def test_gwf_output_drives_the_frame_writer_with_the_config(self, monkeypatch, tmp_path: Path):
        """The frame writer receives the config's duration, rate, detectors and naming."""
        FakeFrameWriter.calls = []
        monkeypatch.setattr(noise_adapter, "FrameWriter", FakeFrameWriter)
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        # A directory that does not exist yet, so a write that fails to create
        # parents fails loudly instead of writing into a stale directory.
        output_directory = tmp_path / "run" / "frames"
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=output_directory,
            output_prefix="noise-0",
            output_format="gwf",
            # A whole second: gwmock-noise rejects a fractional gps_start for gwf, because the
            # artifact name carries the time as an integer and two runs whose times round alike
            # would compose one name.
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(
            config=config,
            chunk={"H1": np.zeros(32), "L1": np.ones(32)},
        )

        assert output_directory.is_dir()
        assert len(FakeFrameWriter.calls) == 1
        call = FakeFrameWriter.calls[0]
        assert call["duration"] == 4.0
        assert call["sampling_frequency"] == 8.0
        assert call["detectors"] == ["H1", "L1"]
        # The chunk is replayed as-is; re-seeding would regenerate different noise.
        assert call["seed"] is None
        assert call["gps_start"] == 100.0
        assert call["output_dir"] == output_directory
        assert call["prefix"] == "noise-0"
        assert result.output_paths == {
            "H1": output_directory / "H1.gwf",
            "L1": output_directory / "L1.gwf",
        }
        # The writer replays the chunk it was given, not a freshly generated one.
        assert np.array_equal(call["simulator"].generate(4.0, 8.0, ["L1"])["L1"], np.ones(32))

    def test_a_non_gwf_format_never_reaches_the_frame_writer(self, monkeypatch, tmp_path: Path):
        """Only the exact string "gwf" selects frames; anything else is written as NumPy."""
        FakeFrameWriter.calls = []
        monkeypatch.setattr(noise_adapter, "FrameWriter", FakeFrameWriter)
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path / "run" / "npy",
            output_prefix="noise-0",
            output_format="npy",
            # A fractional start is fine here: the NumPy artifact name carries no time, so
            # gwmock-noise only requires whole seconds for gwf.
            gps_start=100.5,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(config=config, chunk={"H1": np.arange(32, dtype=float)})

        assert FakeFrameWriter.calls == []
        assert result.output_paths["H1"] == tmp_path / "run" / "npy" / "noise-0_H1.npy"


class TestRunArgumentHandOff:
    """What ``run`` hands to ``build_config``, and which backend path it takes.

    ``run`` forwards eighteen arguments; a dropped one becomes a default (no
    spectral lines, no CSD files, a 2 Hz cutoff) and the batch still runs and still
    writes plausible noise. The config the backend receives is therefore compared
    against an independently built one rather than spot-checked.
    """

    def _run_arguments(self, tmp_path: Path) -> dict:
        """Return one call's worth of arguments with every optional field set."""
        return {
            "detectors": ["H1", "L1"],
            "duration": 4.0,
            "sampling_frequency": 8.0,
            "output_directory": tmp_path,
            "output_prefix": "noise-0",
            "output_format": "npy",
            "gps_start": 100.0,
            "channel": "OTHER_NOISE",
            "channels": {"H1": "H1:STRAIN_NOISE"},
            "seed": 7,
            "psd_file": "psd.txt",
            "psd_schedule": [(0.0, "late.txt")],
            "psd_files": {"H1": "h1.txt"},
            "csd_files": {"H1-L1": "h1l1.txt"},
            "low_frequency_cutoff": TEST_LOW_FREQUENCY_CUTOFF,
            "high_frequency_cutoff": TEST_HIGH_FREQUENCY_CUTOFF,
            "spectral_lines": [{"frequency": 60.0, "amplitude": 1e-23}],
            "glitches": [_blip_glitch_dict()],
        }

    def test_every_argument_reaches_the_backend_config(self, tmp_path: Path):
        """The config the backend runs is the one those arguments describe."""
        backend = FakeNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        arguments = self._run_arguments(tmp_path)

        adapter.run(**arguments)

        assert backend.run_calls == [adapter.build_config(**arguments)]

    def test_channel_and_cutoff_defaults(self, tmp_path: Path):
        """Omitting the channel and the low-frequency cutoff picks the documented defaults."""
        backend = FakeNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        arguments = self._run_arguments(tmp_path)
        del arguments["channel"]
        del arguments["low_frequency_cutoff"]

        adapter.run(**arguments)

        config = backend.run_calls[0]
        assert config.output.channel == "MOCK_NOISE"
        cutoffs = [
            component.options["low_frequency_cutoff"]
            for component in config.components
            if "low_frequency_cutoff" in component.options
        ]
        assert cutoffs == [2.0]

    def test_a_protocol_backend_without_run_generates_and_writes(self, tmp_path: Path):
        """A backend that only satisfies the protocol is driven through generate()."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())

        result = adapter.run(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="npy",
            gps_start=100.0,
            seed=7,
        )

        assert result.output_paths == {
            "H1": tmp_path / "noise-0_H1.npy",
            "L1": tmp_path / "noise-0_L1.npy",
        }
        assert np.load(result.output_paths["H1"]).shape == (32,)

    def test_a_backend_that_is_neither_is_rejected(self, tmp_path: Path):
        """An object with no run() and no protocol support cannot be used."""
        adapter = NoiseAdapter(backend=object())

        with pytest.raises(
            TypeError,
            match=r"^Noise backend must expose run\(\) or satisfy the gwmock_noise NoiseSimulator protocol\.$",
        ):
            adapter.run(
                detectors=["H1"],
                duration=4.0,
                sampling_frequency=8.0,
                output_directory=tmp_path,
                output_prefix="noise-0",
                output_format="npy",
                gps_start=100.0,
            )


class TestTheNameMatchesWhatTheBackendWrites:
    """gwmock predicts each artifact's path before the backend writes it, so the two must agree.

    This replaces a test that pinned the *token formatter* gwmock kept as a mirror of gwmock-noise's.
    Pinning the mirror could only ever confirm the mirror still did what it used to do -- it could not
    notice gwmock-noise doing something else, which is exactly what happened twice over: the channel's
    `IFO:` prefix was dropped from the name, and a fractional second became a refusal rather than a
    `100p5` token. `expected_output_paths` went on predicting `H-H1:MOCK_NOISE_100-4.gwf` for a file
    written as `H-H1_MOCK_NOISE_100-4.gwf`, and no test failed.

    So the assertion is against the real backend rather than against a second statement of its rule.
    """

    @pytest.mark.requires_gwpy
    @pytest.mark.parametrize("output_format", ["gwf", "hdf5"])
    def test_the_predicted_paths_are_the_paths_written(self, tmp_path: Path, output_format: str):
        """Whatever `expected_output_paths` promises, `run` puts a file there."""
        adapter = NoiseAdapter.from_backend(None)
        arguments = {
            "detectors": ["H1", "L1"],
            "duration": 4.0,
            "sampling_frequency": 256.0,
            "output_directory": tmp_path / output_format,
            "output_prefix": "noise-0",
            "output_format": output_format,
            "gps_start": 100.0,
            "channel": "MOCK_NOISE",
            "seed": 5,
        }

        predicted = sorted(adapter.expected_output_paths(config=adapter.build_config(**arguments)))
        written = sorted(adapter.run(**arguments).output_paths.values())

        assert predicted == written
        assert all(path.exists() for path in predicted)


class TestHdf5Output:
    """Writing noise as HDF5, which is what a run's default configuration now asks for.

    gwmock-noise gained an HDF5 writer, and gwmock has to agree with it on two things: the name of the
    file and the shape of what is inside. Both are asserted against the real backend rather than against
    a second copy of the rule written here, because a copy is what already drifted for GWF -- this module
    predicted `H-H1:MOCK_NOISE_100-4.gwf` for a file the backend wrote as `H-H1_MOCK_NOISE_100-4.gwf`.
    """

    def test_expected_output_paths_for_hdf5(self, tmp_path: Path):
        """The HDF5 artifact is named for the detector; the channel lives in the dataset instead."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="noise-0",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        assert adapter.expected_output_paths(config=config) == [
            tmp_path / "noise-0_H-H1_100-4.hdf5",
            tmp_path / "noise-0_L-L1_100-4.hdf5",
        ]

    @pytest.mark.requires_gwpy
    def test_write_chunk_persists_hdf5_outputs(self, tmp_path: Path):
        """A chunk written as HDF5 lands at the promised path and reads back as the samples given.

        Read back through gwpy rather than through h5py: the value of writing HDF5 is that the rest of
        gwmock, and anything downstream, can open it as a time series without a frame library.
        """
        from gwpy.timeseries import TimeSeries

        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(
            config=config,
            chunk={"H1": np.arange(32, dtype=float), "L1": np.arange(32, dtype=float) + 1.0},
        )

        assert sorted(result.output_paths.values()) == sorted(adapter.expected_output_paths(config=config))
        series = TimeSeries.read(str(result.output_paths["H1"]), format="hdf5")
        assert np.array_equal(series.value, np.arange(32, dtype=float))
        assert series.t0.value == 100.0
        assert series.sample_rate.value == 8.0
        assert str(series.channel) == "H1:MOCK_NOISE"

    def test_the_grid_is_recorded_so_a_moved_segment_is_a_different_file(self, tmp_path: Path):
        """The epoch has to reach the file, not just the file name.

        gwmock's content hash reads the grid from the dataset's own attributes, so a writer that omits
        them makes two segments at different GPS times hash alike -- and the run's identity check then
        cannot tell a segment written for the wrong epoch from the right one.
        """
        from gwmock.cli.utils.hash import compute_content_hash

        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        samples = np.arange(32, dtype=float)

        def written(directory: Path, gps_start: float) -> Path:
            directory.mkdir(parents=True, exist_ok=True)
            config = adapter.build_config(
                detectors=["H1"],
                duration=4.0,
                sampling_frequency=8.0,
                output_directory=directory,
                output_prefix="",
                output_format="hdf5",
                gps_start=gps_start,
                channel="MOCK_NOISE",
                seed=7,
            )
            return adapter.write_chunk(config=config, chunk={"H1": samples}).output_paths["H1"]

        original = compute_content_hash(written(tmp_path / "a", 100.0))
        moved = compute_content_hash(written(tmp_path / "b", 612.0))

        assert original is not None
        assert original != moved

    def test_gwmock_writes_the_layout_the_backend_writes(self, tmp_path: Path):
        """gwmock's own HDF5 writer and gwmock-noise's must produce the same artifact.

        `write_chunk` composes the file itself because the samples are already in hand and there is no
        backend run left to delegate to. That makes two writers for one artifact, so this pins them
        together: same name, same dataset, same attributes. Run against the real `DefaultNoiseSimulator`,
        since a fake backend could only repeat whatever this module believes.
        """
        from gwmock_data import hdf5

        real = NoiseAdapter.from_backend(None)
        arguments = {
            "detectors": ["H1"],
            "duration": 4.0,
            "sampling_frequency": 256.0,
            "output_prefix": "",
            "output_format": "hdf5",
            "gps_start": 100.0,
            "channel": "MOCK_NOISE",
            "seed": 11,
        }

        backend_path = next(iter(real.run(output_directory=tmp_path / "backend", **arguments).output_paths.values()))
        config = real.build_config(output_directory=tmp_path / "gwmock", **arguments)
        (tmp_path / "gwmock").mkdir(parents=True, exist_ok=True)
        gwmock_path = real.write_chunk(config=config, chunk={"H1": np.zeros(1024)}).output_paths["H1"]

        assert backend_path.name == gwmock_path.name

        def layout(path: Path) -> tuple[list[str], dict[str, object], dict[str, object]]:
            root = hdf5.read(path, data=False)
            names = sorted(name for name, member in root.members.items() if isinstance(member, hdf5.Dataset))
            dataset = root.members[names[0]]
            return (
                names,
                {key: dataset.attrs[key] for key in sorted(dataset.attrs)},
                {key: root.attrs[key] for key in sorted(root.attrs)},
            )

        # The root attributes are compared too, because that is where the strain schema is declared: a
        # backend-written file that carried no declaration while a gwmock-written one did would make the
        # contract depend on which of the two happened to write the artifact.
        assert layout(backend_path) == layout(gwmock_path)


class TestTheDeclaredStrainSchema:
    """Every HDF5 strain artifact says which contract it meets, whichever writer produced it.

    A consumer reading gwmock output used to have to match the writer's implementation -- this dataset,
    those attributes -- because nothing in the file said what it was. `gwmock.strain_schema` is that
    statement, and these tests pin that it reaches the artifact by both routes: the writer `write_chunk`
    composes itself, and the file a backend writes during `run`.
    """

    def test_a_chunk_written_as_hdf5_declares_the_schema(self, tmp_path: Path):
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(
            config=config,
            chunk={"H1": np.arange(32, dtype=float), "L1": np.arange(32, dtype=float)},
        )

        for path in result.output_paths.values():
            assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    def test_a_file_the_backend_wrote_declares_the_schema(self, tmp_path: Path):
        """`run` hands the writing to gwmock-noise, which knows nothing of gwmock's contract.

        This is the path a real run takes, so a declaration that only `write_chunk` applied would be
        absent from almost every artifact gwmock produces.
        """
        real = NoiseAdapter.from_backend(None)

        result = real.run(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=256.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=11,
        )

        for path in result.output_paths.values():
            assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    @pytest.mark.requires_gwpy
    def test_declaring_it_leaves_the_file_readable(self, tmp_path: Path):
        """The declaration must not cost a consumer the standard reader; see `gwmock.strain_schema`."""
        from gwpy.timeseries import TimeSeries

        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(config=config, chunk={"H1": np.arange(32, dtype=float)})

        series = TimeSeries.read(str(result.output_paths["H1"]), format="hdf5")
        assert np.array_equal(series.value, np.arange(32, dtype=float))

    def test_it_does_not_change_the_content_hash(self, tmp_path: Path):
        """The declaration is provenance, not data: two runs producing the same samples still match.

        The content hash is what a reproduction check compares, so a constant folded into it would make
        every artifact written before the declaration compare unequal to the same data written after.
        """
        import shutil

        from gwmock_data import hdf5

        from gwmock.cli.utils.hash import compute_content_hash

        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format="hdf5",
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )
        written = adapter.write_chunk(config=config, chunk={"H1": np.arange(32, dtype=float)}).output_paths["H1"]
        stamped = compute_content_hash(written)

        undeclared = tmp_path / "undeclared.hdf5"
        shutil.copyfile(written, undeclared)
        stripped = hdf5.read(undeclared)
        stripped.attrs.clear()
        hdf5.write(stripped, undeclared)

        assert stamped == compute_content_hash(undeclared)

    @pytest.mark.requires_gwpy
    @pytest.mark.parametrize("output_format", ["npy", "gwf"])
    def test_the_formats_that_cannot_carry_it_still_write(self, tmp_path: Path, output_format: str):
        """`.npy` has no metadata space and a GWF frame is composed from a fixed set of fields, so the
        declaration is skipped rather than attempted -- but writing them must not break."""
        adapter = NoiseAdapter.from_backend(FakeStreamNoiseBackend())
        config = adapter.build_config(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=8.0,
            output_directory=tmp_path,
            output_prefix="",
            output_format=output_format,
            gps_start=100.0,
            channel="MOCK_NOISE",
            seed=7,
        )

        result = adapter.write_chunk(config=config, chunk={"H1": np.arange(32, dtype=float)})

        assert result.output_paths["H1"].exists()


def test_a_full_channel_name_is_not_prefixed_again():
    """``ET1:STRAIN`` for ``ET1`` is already a full name; prefixing it again doubles the detector."""
    from gwmock.noise.adapter import _frame_channel_name

    assert _frame_channel_name("ET1_SARD", "ET1_SARD:STRAIN") == "ET1_SARD:STRAIN"
    assert _frame_channel_name("ET1_SARD", "STRAIN") == "ET1_SARD:STRAIN"
