"""Focused tests for the adapter-backed CLI orchestration path."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import yaml
from gwmock_signal import DetectorStrainStack
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.simulate import _simulate_impl
from gwmock.cli.utils.config import (
    Config,
    GlobalsConfig,
    NoiseAdapterConfig,
    OrchestrationConfig,
    PopulationConfig,
    SignalConfig,
    SimulatorOutputConfig,
)
from gwmock.cli.utils.simulation_plan import SimulationBatch, create_plan_from_config
from gwmock.simulator.seeds import derive_seed
from gwmock.strain_schema import STRAIN_SCHEMA_VERSION, missing_layout_attributes, require_strain_schema

EXPECTED_BATCHES = 2
FAKE_POPULATION_BACKEND = "tests.cli.test_cli_orchestration:FakePopulationBackend"
FAKE_SIGNAL_BACKEND = "tests.cli.test_cli_orchestration:FakeSignalAdapter"
FAKE_SGWB_SIGNAL_BACKEND = "tests.cli.test_cli_orchestration:FakeSgwbSignalAdapter"
FAKE_NOISE_BACKEND = "tests.cli.test_cli_orchestration:FakeNoiseAdapter"


class FakePopulationBackend:
    """Minimal public-style population backend for orchestration tests."""

    parameter_names: ClassVar[tuple[str, ...]] = ("detector_frame_mass_1", "detector_frame_mass_2", "coa_time")
    metadata: ClassVar[dict[str, object]] = {
        "fetch": {"scheme": "https"},
        "resolved_path": str(Path(tempfile.gettempdir()) / "catalog.h5"),
    }

    def __init__(self, path: str, source_type: str = "bbh") -> None:
        self.path = path
        self.source_type = source_type

    def simulate(self, n_samples: int, **_kwargs):
        if n_samples != EXPECTED_BATCHES:
            raise AssertionError("Unexpected population sample count for test.")
        return {
            "detector_frame_mass_1": np.array([30.0, 31.0]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "coa_time": np.array([100.5, 104.5]),
        }


class FakeSignalAdapter:
    """Minimal signal backend returning deterministic strain stacks."""

    required_params = frozenset({"detector_frame_mass_1", "coa_time"})

    def __init__(self, waveform_model: str = "IMRPhenomXPHM") -> None:
        self.waveform_model = waveform_model

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = background, minimum_frequency, earth_rotation, interpolate_if_offset
        return DetectorStrainStack.from_mapping(
            detector_names,
            {
                detector: GWpyTimeSeries(
                    np.full(4, parameters["detector_frame_mass_1"]),
                    t0=parameters["coa_time"],
                    sample_rate=sampling_frequency,
                )
                for detector in detector_names
            },
        )


class FakeSgwbSignalAdapter:
    """Stationary signal backend used by SGWB orchestration tests."""

    required_params = frozenset({"omega_ref"})

    def __init__(self, duration: float, seed: int | None = None) -> None:
        self.duration = duration
        self.seed = seed
        self.simulate_calls: list[dict[str, object]] = []

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Return one segment-length stationary signal stack."""
        _ = minimum_frequency, earth_rotation, interpolate_if_offset
        names = tuple(detector if isinstance(detector, str) else detector.name for detector in detector_names)
        n_samples = round(self.duration * sampling_frequency)
        self.simulate_calls.append(
            {
                "parameters": dict(parameters),
                "detector_names": names,
                "background": background,
                "sampling_frequency": sampling_frequency,
                "seed": self.seed,
            }
        )
        t0 = 0.0 if background is None else float(next(iter(background.values())).t0.value)
        seed_offset = 0.0 if self.seed is None else float(self.seed % 1000)
        return DetectorStrainStack.from_mapping(
            names,
            {
                detector: GWpyTimeSeries(
                    np.full(n_samples, float(parameters["omega_ref"]) + seed_offset + index),
                    t0=t0,
                    sample_rate=sampling_frequency,
                )
                for index, detector in enumerate(names)
            },
        )


class FakeNoiseAdapter:
    """Minimal noise protocol backend that materializes deterministic arrays."""

    stream_open_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self,
        duration: float = 4.0,
        sampling_frequency: float = 4.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = ["H1"] if detectors is None else detectors
        self.seed = seed
        self._chunk_index = 0

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        _ = duration, sampling_frequency, seed
        return {detector: np.zeros(4) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        type(self).stream_open_calls.append(
            {
                "chunk_duration": chunk_duration,
                "sampling_frequency": sampling_frequency,
                "detectors": list(detectors),
                "seed": seed,
            }
        )
        while True:
            yield {
                detector: np.full(round(chunk_duration * sampling_frequency), self._chunk_index, dtype=float)
                for detector in detectors
            }
            self._chunk_index += 1

    @property
    def metadata(self) -> dict[str, object]:
        return {"kind": "fake-noise"}


def _write_signal_file(self, path, **kwargs):
    """Stand in for `DetectorStrainStack.write`, producing a file its own format can be opened as.

    The stub used to write the bytes `STRAIN` whatever the extension asked for. That was invisible while
    nothing reopened the artifact, and stopped being so once gwmock declares its strain schema on the
    file it just wrote: a `.hdf5` holding six ASCII bytes is not an HDF5 file, and the declaration failed
    on a file no reader could have opened either. A fake writer still has to produce the container the
    name promises.
    """
    _ = kwargs
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() in {".hdf5", ".h5"}:
        import h5py

        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("STRAIN", data=np.zeros(1))
            # The grid, spelled the way the real writer spells it. Without it the stub produces a file
            # carrying no epoch, which is not something that writer can emit and which gwmock refuses to
            # declare a layout over -- so a stub that omitted it would be testing a state that cannot
            # occur.
            dataset.attrs["t0"] = 0.0
            dataset.attrs["dt"] = 1.0
            dataset.attrs["unit"] = "strain"
        return
    Path(path).write_text("STRAIN")


def _fake_orchestration_config(tmp_path: Path, *, source_type: str) -> Config:
    return Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 4,
                "duration": 4,
                "start-time": 100,
                "max-samples": 2,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend=FAKE_POPULATION_BACKEND,
                source_type=source_type,
                n_samples=2,
                arguments={"path": str(tmp_path / "population.h5"), "source_type": source_type},
            ),
            signal=SignalConfig(
                backend=FAKE_SIGNAL_BACKEND,
                waveform_model="IMRPhenomD",
                detectors=["H1"],
                output=SimulatorOutputConfig(
                    file_name="signal-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "H1:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                backend=FAKE_NOISE_BACKEND,
                arguments={"seed": 7, "detectors": ["H1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(
                    file_name="noise-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )


def _fake_sgwb_config(tmp_path: Path, *, detectors: list[str] | None = None) -> Config:
    return Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 8,
                "duration": 2,
                "start-time": 100,
                "max-samples": 2,
                "seed": 11,
            },
        ),
        orchestration=OrchestrationConfig(
            signal=SignalConfig(
                source_type="sgwb",
                backend=FAKE_SGWB_SIGNAL_BACKEND,
                detectors=detectors or ["H1", "L1"],
                parameters={"omega_ref": 3.0},
                minimum_frequency=1.0,
                output=SimulatorOutputConfig(
                    file_name="sgwb-{{ counter }}.hdf5",
                    output_directory="signal",
                ),
            )
        ),
    )


def _assert_noise_outputs_exist(output_directory: Path) -> None:
    for counter in range(EXPECTED_BATCHES):
        assert (output_directory / f"noise-{counter}.npy").exists()


def test_create_plan_from_orchestration_config(tmp_path: Path):
    """Batch planning should respect the new orchestration config surface."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")

    plan = create_plan_from_config(config, tmp_path / "checkpoints")

    assert plan.total_batches == EXPECTED_BATCHES
    assert all(batch.simulator_name == "orchestration" for batch in plan.batches)
    assert all(isinstance(batch.simulator_config, OrchestrationConfig) for batch in plan.batches)


def test_signal_only_sgwb_plan_validates(tmp_path: Path):
    """Signal-only SGWB orchestration does not require a population section."""
    config = _fake_sgwb_config(tmp_path)

    plan = create_plan_from_config(config, tmp_path / "checkpoints")

    assert plan.total_batches == 2
    assert plan.batches[0].simulator_config.signal.source_type == "sgwb"


@pytest.mark.parametrize("source_type", ["bbh", "bns", "nsbh", "gengli"])
def test_simulate_command_runs_adapter_orchestration(monkeypatch, tmp_path: Path, source_type: str):
    """The CLI should execute the adapter-backed orchestration path end to end."""
    FakeNoiseAdapter.stream_open_calls.clear()
    config = _fake_orchestration_config(tmp_path, source_type=source_type)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    assert (tmp_path / "output" / "signal" / "signal-0.gwf").exists()
    assert (tmp_path / "output" / "signal" / "signal-1.gwf").exists()
    _assert_noise_outputs_exist(tmp_path / "output" / "noise")
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    assert metadata["schema_version"] == "1.6.0"
    assert metadata["config"]["orchestration"]["population"]["backend"] == FAKE_POPULATION_BACKEND
    assert metadata["config"]["orchestration"]["signal"]["backend"] == FAKE_SIGNAL_BACKEND
    assert metadata["config"]["orchestration"]["noise"]["backend"] == FAKE_NOISE_BACKEND
    assert metadata["population"]["source_type"] == source_type
    assert metadata["signal"]["detector_network"] == ["H1"]
    # Full environment freeze is recorded for exact-dependency reproduction (#11).
    assert metadata["environment"]["packages"]["gwmock"]
    assert metadata["environment"]["python"].count(".") >= 2
    assert {output["kind"] for output in metadata["outputs"]} == {"signal", "noise"}
    assert metadata["segment_seeds"] == [derive_seed(7, "signal", 0)]
    assert metadata["simulator_config"]["population"]["backend"] == FAKE_POPULATION_BACKEND
    assert metadata["simulator_config"]["signal"]["detectors"] == ["H1"]
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"] == FakePopulationBackend.metadata
    assert metadata["simulator_metadata"]["orchestration"]["population"]["seed"] == derive_seed(7, "population")
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["network_resolution"] == {
        "inputs": ["H1"],
        "detector_names": ["H1"],
        "steps": [{"input": "H1", "resolver": "detector", "detector_names": ["H1"]}],
    }
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["segment_seed"] == derive_seed(7, "signal", 0)
    assert metadata["simulator_metadata"]["orchestration"]["noise"]["stream_seed"] == derive_seed(7, "noise", "stream")
    assert FakeNoiseAdapter.stream_open_calls == [
        {
            "chunk_duration": 4.0,
            "sampling_frequency": 4.0,
            "detectors": ["H1"],
            "seed": derive_seed(7, "noise", "stream"),
        }
    ]


def test_orchestration_records_injections_and_signal_lookup(monkeypatch, tmp_path: Path):
    """End-to-end: source parameters are recorded per frame and are looked up (#12)."""
    from gwmock.cli.utils.signal_lookup import find_signals, parse_param_filter

    FakeNoiseAdapter.stream_open_calls.clear()
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))
    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    metadata_dir = tmp_path / "metadata"

    # Part A: each batch's metadata records the source parameters of its signal(s).
    batch0 = yaml.safe_load((metadata_dir / "orchestration-0.metadata.json").read_text())
    injections0 = batch0["signal"]["injections"]
    assert [inj["event_id"] for inj in injections0] == [0]
    assert injections0[0]["parameters"]["detector_frame_mass_1"] == 30.0
    assert injections0[0]["parameters"]["coa_time"] == 100.5

    batch1 = yaml.safe_load((metadata_dir / "orchestration-1.metadata.json").read_text())
    assert [inj["event_id"] for inj in batch1["signal"]["injections"]] == [1]

    # Part B: signal_index.yaml maps each event id to its containing frame(s).
    index = yaml.safe_load((metadata_dir / "signal_index.yaml").read_text())
    assert set(index) == {"0", "1"}
    assert any("signal-0.gwf" in frame for batch in index["0"]["batches"] for frame in batch["frames"])

    # Lookup by id (fast path) resolves to the right frame.
    by_id = find_signals(metadata_dir, event_id=1)
    assert len(by_id) == 1
    assert any("signal-1.gwf" in frame for frame in by_id[0]["frames"])

    # Lookup by parameter filter scans the recorded injections.
    heavy = find_signals(metadata_dir, param_filters=[parse_param_filter("detector_frame_mass_1>=31")])
    assert [match["event_id"] for match in heavy] == [1]
    assert any("signal-1.gwf" in frame for frame in heavy[0]["frames"])


def _fake_multi_detector_config(working_directory: str) -> Config:
    """Orchestration config whose signal file_name uses a multi-detector template.

    ``{{ detectors }}`` renders to one file per detector, so the metadata sidecar
    records ``file_name`` (and ``channel``) as per-detector lists.
    """
    return Config(
        globals=GlobalsConfig(
            working_directory=working_directory,
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 4,
                "duration": 4,
                "start-time": 100,
                "max-samples": 2,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend=FAKE_POPULATION_BACKEND,
                source_type="bbh",
                n_samples=2,
                arguments={"path": "population.h5", "source_type": "bbh"},
            ),
            signal=SignalConfig(
                backend=FAKE_SIGNAL_BACKEND,
                waveform_model="IMRPhenomD",
                detectors=["H1", "L1"],
                output=SimulatorOutputConfig(
                    file_name="E-{{ detectors }}_BBH-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "{{ detectors }}:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                backend=FAKE_NOISE_BACKEND,
                arguments={"seed": 7, "detectors": ["H1", "L1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(
                    file_name="E-{{ detectors }}_NOISE-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )


def test_simulate_reproduces_metadata_with_multi_detector_signal_template(monkeypatch, tmp_path: Path):
    """Replay: metadata from a multi-detector ``{{ detectors }}`` signal template re-runs cleanly.

    The metadata sidecar stores the rendered per-detector ``file_name`` list rather than
    the template. Regression: the replay path stringified that list into a single bogus
    path (suffix ``.gwf']``), failing the signal output-format check; with a noise section
    the retried batch then collided with its own freshly written noise outputs.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)
    monkeypatch.chdir(run_dir)

    config = _fake_multi_detector_config(".")
    config_file = run_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    expected_signal_names = [
        "E-H1_BBH-0.gwf",
        "E-L1_BBH-0.gwf",
        "E-H1_BBH-1.gwf",
        "E-L1_BBH-1.gwf",
    ]
    for name in expected_signal_names:
        assert (run_dir / "output" / "signal" / name).exists()

    metadata_file = run_dir / "metadata" / "orchestration-0.metadata.json"
    metadata = yaml.safe_load(metadata_file.read_text())
    # Lock in the serialization this regression test relies on: the sidecar carries the
    # rendered per-detector list, not the original template.
    assert metadata["config"]["orchestration"]["signal"]["output"]["file_name"] == [
        "E-H1_BBH-0.gwf",
        "E-L1_BBH-0.gwf",
    ]

    repro_dir = tmp_path / "repro"
    repro_dir.mkdir()
    for metadata_name in ("orchestration-0.metadata.json", "orchestration-1.metadata.json"):
        (repro_dir / metadata_name).write_text((run_dir / "metadata" / metadata_name).read_text())
    monkeypatch.chdir(repro_dir)

    _simulate_impl(
        [str(repro_dir / "orchestration-0.metadata.json"), str(repro_dir / "orchestration-1.metadata.json")],
        metadata=False,
    )

    for name in expected_signal_names:
        assert (repro_dir / "output" / "signal" / name).exists()
    for name in ("E-H1_NOISE-0.npy", "E-L1_NOISE-0.npy", "E-H1_NOISE-1.npy", "E-L1_NOISE-1.npy"):
        assert (repro_dir / "output" / "noise" / name).exists()


def test_a_runs_hdf5_outputs_declare_the_strain_schema(tmp_path: Path):
    """Both halves of a run declare the contract their strain files meet.

    Signal and noise reach disk through different writers -- `DetectorStrainStack.write` from the signal
    side, a second call to it from the noise side -- so a declaration applied to one of them would leave
    a consumer unable to say what a gwmock run produces. The signal writer is the real one here, not the
    stub the tests above use, because what is being checked is the artifact rather than the call.
    """
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.output.file_name = "signal-{{ counter }}.hdf5"
    config.orchestration.noise.output.file_name = "noise-{{ counter }}.hdf5"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    for counter in range(EXPECTED_BATCHES):
        signal_output = tmp_path / "output" / "signal" / f"signal-{counter}.hdf5"
        noise_output = tmp_path / "output" / "noise" / f"noise-{counter}.hdf5"
        # `require_strain_schema` checks the layout, not only the claim: these files are written by
        # gwmock-signal's multichannel writer, which records the grid as `t0`/`dt` and no channel
        # attributes, so passing here is the statement that a run's artifacts genuinely carry the
        # declared 1.0.0 layout rather than merely asserting it.
        assert require_strain_schema(signal_output).version == STRAIN_SCHEMA_VERSION
        assert require_strain_schema(noise_output).version == STRAIN_SCHEMA_VERSION
        assert missing_layout_attributes(signal_output) == {}
        assert missing_layout_attributes(noise_output) == {}
        # Still openable by the standard reader: the declaration lives at the file root precisely so
        # that gwpy does not meet an attribute it would pass to the series constructor.
        assert GWpyTimeSeries.read(str(noise_output), format="hdf5") is not None


def test_simulate_command_runs_signal_only_sgwb_orchestration(monkeypatch, tmp_path: Path):
    """The CLI should generate stationary SGWB segments without a population catalogue."""
    config = _fake_sgwb_config(tmp_path)
    config_file = tmp_path / "sgwb.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    assert (tmp_path / "output" / "signal" / "sgwb-0.hdf5").exists()
    assert (tmp_path / "output" / "signal" / "sgwb-1.hdf5").exists()
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    assert metadata["simulator_metadata"]["orchestration"]["source_type"] == "sgwb"
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["parameters"] == {"omega_ref": 3.0}
    assert metadata["simulator_metadata"]["orchestration"]["signal"]["segment_seed"] == derive_seed(11, "signal", 0)
    assert metadata["outputs"][0]["kind"] == "signal"


def test_orchestrator_restores_noise_stream_from_committed_cursor(tmp_path: Path):
    """Restart should fast-forward noise stream from persisted committed cursor."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    # Simulate restored state where one chunk was already committed.
    orchestrator.noise_stream_committed_count = 1
    orchestrator.counter = 0
    orchestrator._noise_stream = None
    orchestrator._noise_stream_position = 0
    orchestrator._pending_noise_chunk = None

    chunk = orchestrator._next_noise_chunk()
    assert chunk["H1"][0] == 1.0


def test_orchestrator_records_preset_network_resolution(tmp_path: Path):
    """Named detector presets should be resolved once and reflected into metadata."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET-Triangle-Sardinia"]
    config.orchestration.noise.arguments = {"seed": 7, "duration": 4.0, "sampling_frequency": 4.0}

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    signal_metadata = orchestrator.metadata["orchestration"]["signal"]

    assert orchestrator.detectors == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert signal_metadata["network_resolution"] == {
        "inputs": ["ET-Triangle-Sardinia"],
        "detector_names": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
        "steps": [
            {
                "input": "ET-Triangle-Sardinia",
                "resolver": "preset",
                "detector_names": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
            }
        ],
    }


def test_orchestrator_resolves_noise_detector_alias(tmp_path: Path):
    """Detector aliases in noise.arguments.detectors must be resolved to sub-detectors."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET-Triangle-Sardinia"]
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["ET-Triangle-Sardinia"],
    }

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]
    assert orchestrator.detectors == ["ET1_SARD", "ET2_SARD", "ET3_SARD"]


def test_orchestrator_records_single_detector_preset_resolution(tmp_path: Path):
    """Single public ET-detector aliases should resolve via the preset-backed detector catalog."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.signal.detectors = ["ET1_SARD"]
    config.orchestration.noise.arguments = {"seed": 7, "duration": 4.0, "sampling_frequency": 4.0}

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    signal_metadata = orchestrator.metadata["orchestration"]["signal"]

    assert orchestrator.detectors == ["ET1_SARD"]
    assert orchestrator.noise_arguments["detectors"] == ["ET1_SARD"]
    assert signal_metadata["network_resolution"] == {
        "inputs": ["ET1_SARD"],
        "detector_names": ["ET1_SARD"],
        "steps": [
            {
                "input": "ET1_SARD",
                "resolver": "preset-detector",
                "detector_names": ["ET1_SARD"],
            }
        ],
    }


def test_noise_batch_gwf_writes_per_detector_files(monkeypatch, tmp_path: Path):
    """Noise GWF template with {{ detectors }} produces one file per detector at the template-specified path."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["H1", "L1"],
    }
    config.orchestration.noise.output = SimulatorOutputConfig(
        file_name="noise-{{ counter }}-{{ detectors }}.gwf",
        output_directory="noise",
    )

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True

    result = orchestrator._run_noise_batch()

    noise_dir = tmp_path / "noise"
    assert (noise_dir / "noise-0-H1.gwf").exists()
    assert (noise_dir / "noise-0-L1.gwf").exists()
    assert result.output_paths["H1"] == noise_dir / "noise-0-H1.gwf"
    assert result.output_paths["L1"] == noise_dir / "noise-0-L1.gwf"


def test_noise_batch_npy_uses_template_path(tmp_path: Path):
    """Noise NPY files are written to the exact template-expanded path (no detector suffix appended)."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True

    result = orchestrator._run_noise_batch()

    assert (tmp_path / "noise" / "noise-0.npy").exists()
    assert result.output_paths["H1"] == tmp_path / "noise" / "noise-0.npy"


def test_noise_batch_npy_scalar_template_multiple_detectors_raises(tmp_path: Path):
    """Scalar NPY template with multiple detectors raises ValueError to prevent silent overwrites."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["H1", "L1"],
    }

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True

    with pytest.raises(ValueError, match="identical paths for multiple detectors"):
        orchestrator._run_noise_batch()


def test_noise_batch_overwrite_check_uses_template_paths(tmp_path: Path):
    """FileExistsError is raised when a template-expanded noise path already exists."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir(parents=True)
    (noise_dir / "noise-0.npy").write_text("existing")

    orchestrator._active_noise_output_directory = noise_dir
    orchestrator._active_overwrite = False

    with pytest.raises(FileExistsError, match=r"noise-0\.npy"):
        orchestrator._run_noise_batch()


def _make_noise_only_batch(tmp_path: Path, *, batch_index: int, file_name: list[str]) -> SimulationBatch:
    """Build a noise-only orchestration batch with an explicit per-batch file_name list.

    Mirrors a metadata-reproduction batch, where ``file_name`` is an already-expanded literal
    list (one entry per detector) rather than a template.
    """
    return SimulationBatch(
        simulator_name="orchestration",
        simulator_config=OrchestrationConfig(
            noise=NoiseAdapterConfig(
                backend=FAKE_NOISE_BACKEND,
                arguments={"seed": 7, "detectors": ["H1", "L1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(file_name=file_name, output_directory="noise"),
            ),
        ),
        globals_config=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={"sampling-frequency": 4, "duration": 4, "start-time": 100, "max-samples": 6},
        ),
        batch_index=batch_index,
    )


def test_set_batch_context_applies_per_batch_noise_file_name(tmp_path: Path):
    """Reproduction: each batch writes its own per-batch noise file_name, not batch 0's (ISS-016).

    The orchestrator is instantiated once from the first batch's config. When reproducing from
    metadata each batch carries its own already-expanded literal file_name; ``set_batch_context``
    must make ``_run_noise_batch`` use that per-batch file_name rather than the instantiation-time
    config, otherwise every batch reuses the first batch's filenames and the second batch collides.
    """
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["H1", "L1"],
    }
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)

    batch0 = _make_noise_only_batch(tmp_path, batch_index=0, file_name=["E-H1-100.npy", "E-L1-100.npy"])
    batch1 = _make_noise_only_batch(tmp_path, batch_index=1, file_name=["E-H1-104.npy", "E-L1-104.npy"])
    output_directory = tmp_path / "output"

    orchestrator.set_batch_context(batch=batch0, output_directory=output_directory, overwrite=True)
    assert orchestrator._active_noise_file_name == ["E-H1-100.npy", "E-L1-100.npy"]
    result0 = orchestrator._run_noise_batch()
    assert {path.name for path in result0.output_paths.values()} == {"E-H1-100.npy", "E-L1-100.npy"}

    orchestrator.set_batch_context(batch=batch1, output_directory=output_directory, overwrite=True)
    assert orchestrator._active_noise_file_name == ["E-H1-104.npy", "E-L1-104.npy"]
    result1 = orchestrator._run_noise_batch()
    # Second batch must use its own filenames (104), not batch 0's (100) — the collision bug.
    assert {path.name for path in result1.output_paths.values()} == {"E-H1-104.npy", "E-L1-104.npy"}
    assert set(result1.output_paths.values()).isdisjoint(result0.output_paths.values())

    noise_dir = result0.output_paths["H1"].parent
    for name in ("E-H1-100.npy", "E-L1-100.npy", "E-H1-104.npy", "E-L1-104.npy"):
        assert (noise_dir / name).exists()


def test_infer_noise_format_empty_list_raises(tmp_path: Path):
    """_infer_noise_output_format rejects an empty list."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    with pytest.raises(ValueError, match="at least one entry"):
        orchestrator._infer_noise_output_format([])


def test_infer_noise_format_invalid_suffix_in_list_raises(tmp_path: Path):
    """_infer_noise_output_format rejects a list entry with an unsupported extension."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    with pytest.raises(ValueError, match=r"must end with \.npy, \.gwf, \.hdf5, or \.h5"):
        orchestrator._infer_noise_output_format(["noise-H1.wav", "noise-L1.wav"])


def test_infer_noise_format_mixed_suffixes_raises(tmp_path: Path):
    """_infer_noise_output_format rejects a list that mixes .npy and .gwf entries."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    with pytest.raises(ValueError, match="same format"):
        orchestrator._infer_noise_output_format(["noise-H1.npy", "noise-L1.gwf"])


def test_infer_noise_format_accepts_hdf5(tmp_path: Path):
    """.hdf5 is a noise output format, not just a signal one.

    Both halves of a run are configured the same way, by naming the file, so a template the signal side
    accepts and the noise side refuses is a run that cannot be configured coherently.
    """
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    assert orchestrator._infer_noise_output_format("noise-0.hdf5") == "hdf5"


def test_infer_noise_format_folds_h5_into_hdf5(tmp_path: Path):
    """`.h5` is the same format under a shorter name, as it is for signal output."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    assert orchestrator._infer_noise_output_format("noise-0.h5") == "hdf5"
    assert orchestrator._infer_noise_output_format(["noise-H1.h5", "noise-L1.hdf5"]) == "hdf5"


def test_infer_noise_format_accepts_the_shipped_default(tmp_path: Path):
    """The default noise file_name must be a template the noise path can actually run.

    This is the whole reason the item exists: the default moved to `.hdf5` for both halves of a run, but
    only the signal half had ever been taught to read that extension, so an unmodified configuration
    raised "Noise output templates must end with .npy or .gwf" before it wrote anything. A default that
    the code refuses is worse than a wrong default, and nothing failed until a run was attempted.
    """
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    shipped = NoiseAdapterConfig().output.file_name

    assert orchestrator._infer_noise_output_format(shipped) == "hdf5"


def test_expand_noise_output_paths_list_duplicate_paths_raises(tmp_path: Path):
    """_expand_noise_output_paths rejects a pre-resolved list where two entries map to the same path."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["H1", "L1"],
    }
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    with pytest.raises(ValueError, match="identical paths"):
        orchestrator._expand_noise_output_paths(["same-file.npy", "same-file.npy"])


def test_save_data_list_file_name_coerced_to_ndarray(monkeypatch, tmp_path: Path):
    """_save_data accepts list[str] file_name and routes it through the ndarray dispatch path."""
    from gwmock.data.time_series.time_series import TimeSeries as GwmockTimeSeries

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    data = GwmockTimeSeries(np.zeros((1, 4)), start_time=0.0, sampling_frequency=1.0)

    written: list[Path] = []

    def _mock_write(self, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        written.append(path)

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _mock_write)

    out = tmp_path / "signal" / "H1.npy"
    orchestrator._save_data(data, [str(out)])

    assert written == [out]


def test_build_signal_stack_uses_channel_name_as_gwf_channel(tmp_path: Path):
    """_build_signal_stack uses channel_name as the stack name so GWF writes the right channel."""
    from gwmock.data.time_series.time_series import TimeSeries as GwmockTimeSeries

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    data = GwmockTimeSeries(np.zeros((1, 4)), start_time=0.0, sampling_frequency=1.0)

    stack = orchestrator._build_signal_stack(
        data=data,
        channel_names=["H1:STRAIN"],
        detector_names=["H1"],
    )

    assert stack.detector_names == ("H1:STRAIN",)


def test_build_signal_stack_falls_back_to_detector_name_when_channel_is_none(tmp_path: Path):
    """_build_signal_stack uses detector_name when channel_name is None (no regression)."""
    from gwmock.data.time_series.time_series import TimeSeries as GwmockTimeSeries

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    data = GwmockTimeSeries(np.zeros((1, 4)), start_time=0.0, sampling_frequency=1.0)

    stack = orchestrator._build_signal_stack(
        data=data,
        channel_names=[None],
        detector_names=["H1"],
    )

    assert stack.detector_names == ("H1",)


def test_build_signal_stack_disambiguates_duplicate_channel_names(tmp_path: Path):
    """Duplicate effective names are disambiguated so no series silently overwrites another."""
    from gwmock.data.time_series.time_series import TimeSeries as GwmockTimeSeries

    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    data = GwmockTimeSeries(np.zeros((2, 4)), start_time=0.0, sampling_frequency=1.0)

    stack = orchestrator._build_signal_stack(
        data=data,
        channel_names=["STRAIN", "STRAIN"],
        detector_names=["H1", "L1"],
    )

    assert stack.detector_names == ("STRAIN", "STRAIN__L1")


def test_noise_batch_rejects_mismatched_channel_list_length(tmp_path: Path):
    """List-valued channel with wrong length raises ValueError."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    # One detector in noise arguments but two entries in the channel list.
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True
    orchestrator._active_noise_output_arguments = {"channel": ["H1:STRAIN_NOISE", "L1:STRAIN_NOISE"]}

    with pytest.raises(ValueError, match="Noise channel list expanded to 2 entries but there are 1 noise detectors"):
        orchestrator._run_noise_batch()


def test_noise_batch_rejects_channel_prefix_key(tmp_path: Path):
    """_run_noise_batch raises ValueError when 'channel_prefix' is present (renamed to 'channel')."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True
    orchestrator._active_noise_output_arguments = {"channel_prefix": "MOCK"}

    with pytest.raises(ValueError, match="channel_prefix"):
        orchestrator._run_noise_batch()


def test_noise_batch_gwf_per_detector_channel_from_list(monkeypatch, tmp_path: Path):
    """List-valued channel (from {{ detectors }}:STRAIN_NOISE) produces per-detector channels_dict."""
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "duration": 4.0,
        "sampling_frequency": 4.0,
        "detectors": ["H1", "L1"],
    }
    config.orchestration.noise.output = SimulatorOutputConfig(
        file_name="noise-{{ counter }}-{{ detectors }}.gwf",
        output_directory="noise",
    )
    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    orchestrator = AdapterOrchestrator.from_config(config.orchestration, config.globals.simulator_arguments)
    orchestrator._active_noise_output_directory = tmp_path / "noise"
    orchestrator._active_overwrite = True
    orchestrator._active_noise_output_arguments = {"channel": ["H1:STRAIN_NOISE", "L1:STRAIN_NOISE"]}

    result = orchestrator._run_noise_batch()

    assert result.config.output.channel == "STRAIN_NOISE"
    assert result.config.output.channels == {"H1": "H1:STRAIN_NOISE", "L1": "L1:STRAIN_NOISE"}
    assert result.output_paths["H1"] == tmp_path / "noise" / "noise-0-H1.gwf"
    assert result.output_paths["L1"] == tmp_path / "noise" / "noise-0-L1.gwf"


def test_simulate_command_records_per_detector_channel_ids_in_metadata(monkeypatch, tmp_path: Path):
    """Per-detector channel names from {{ detectors }}:X template are recorded correctly in metadata."""
    FakeNoiseAdapter.stream_open_calls.clear()
    config = _fake_orchestration_config(tmp_path, source_type="bbh")
    config.orchestration.noise.arguments = {
        "seed": 7,
        "detectors": ["H1", "L1"],
        "duration": 4.0,
        "sampling_frequency": 4.0,
    }
    config.orchestration.noise.output = SimulatorOutputConfig(
        file_name="noise-{{ counter }}-{{ detectors }}.gwf",
        output_directory="noise",
        arguments={"channel": "{{ detectors }}:STRAIN_NOISE"},
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    noise_outputs = [o for o in metadata["outputs"] if o["kind"] == "noise"]
    channel_ids = {o["channels"][0] for o in noise_outputs}
    assert channel_ids == {"H1:STRAIN_NOISE", "L1:STRAIN_NOISE"}
