"""Regression tests for third-party orchestration backends."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

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
from gwmock.simulator.seeds import derive_seed


def _write_signal_file(self, path, **kwargs):
    _ = self, kwargs
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("STRAIN")


def _install_third_party_backend_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    package_dir = tmp_path / "plugin-src"
    site_packages = tmp_path / "site-packages"
    package_name = "third_party_orchestration_backend_pkg"

    (package_dir / package_name).mkdir(parents=True)
    (package_dir / package_name / "__init__.py").write_text("")
    (package_dir / package_name / "population.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class ThirdPartyPopulationBackend:
                parameter_names = ("detector_frame_mass_1", "detector_frame_mass_2", "coa_time")
                source_type = "gengli"
                metadata = {"kind": "third-party-population"}

                def __init__(self, path: str, source_type: str = "gengli") -> None:
                    self.path = path
                    self.source_type = source_type

                def simulate(self, n_samples: int, **_kwargs):
                    return {
                        "detector_frame_mass_1": np.full(n_samples, 40.0),
                        "detector_frame_mass_2": np.full(n_samples, 30.0),
                        "coa_time": np.linspace(100.0, 101.0, n_samples),
                    }
            """
        )
    )
    (package_dir / package_name / "signal.py").write_text(
        textwrap.dedent(
            """
            import numpy as np
            from gwmock_signal import DetectorStrainStack
            from gwpy.timeseries import TimeSeries as GWpyTimeSeries

            class ThirdPartySignalBackend:
                required_params = frozenset({"detector_frame_mass_1", "coa_time"})

                def __init__(self, waveform_model: str = "IMRPhenomXPHM") -> None:
                    self.waveform_model = waveform_model

                def simulate(
                    self,
                    parameters,
                    detector_names,
                    background=None,
                    *,
                    sampling_frequency: float,
                    minimum_frequency: float,
                    earth_rotation: bool = True,
                    interpolate_if_offset: bool = True,
                ):
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
            """
        )
    )
    (package_dir / package_name / "noise.py").write_text(
        textwrap.dedent(
            """
            import numpy as np
            from gwmock_noise import SimulationResult

            class ThirdPartyNoiseBackend:
                duration = 4.0
                sampling_frequency = 4.0
                detectors = ["H1"]
                seed = None
                metadata = {"kind": "third-party-noise"}

                def __init__(
                    self,
                    duration: float = 4.0,
                    sampling_frequency: float = 4.0,
                    detectors=None,
                    seed=None,
                ) -> None:
                    self.duration = duration
                    self.sampling_frequency = sampling_frequency
                    self.detectors = ["H1"] if detectors is None else detectors
                    self.seed = seed

                def generate(self, duration, sampling_frequency, detectors, seed=None):
                    _ = duration, sampling_frequency, seed
                    return {detector: np.zeros(4) for detector in detectors}

                def generate_stream(self, chunk_duration, sampling_frequency, detectors, seed=None):
                    _ = chunk_duration, sampling_frequency, seed
                    yield {detector: np.zeros(4) for detector in detectors}

                def run(self, config):
                    config.output.directory.mkdir(parents=True, exist_ok=True)
                    output_paths = {}
                    for detector in config.detectors:
                        artifact_path = config.output.directory / f"{config.output.prefix}_{detector}.npy"
                        np.save(artifact_path, np.zeros(round(config.duration * config.sampling_frequency)))
                        output_paths[detector] = artifact_path
                    return SimulationResult(output_paths=output_paths, config=config)
            """
        )
    )
    (package_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "third-party-orchestration-backend-pkg"
            version = "0.0.1"

            [project.entry-points."gwmock.population"]
            third_party_population = "third_party_orchestration_backend_pkg.population:ThirdPartyPopulationBackend"

            [project.entry-points."gwmock.signal"]
            third_party_signal = "third_party_orchestration_backend_pkg.signal:ThirdPartySignalBackend"

            [project.entry-points."gwmock.noise"]
            third_party_noise = "third_party_orchestration_backend_pkg.noise:ThirdPartyNoiseBackend"
            """
        )
    )

    uv_path = shutil.which("uv")
    if uv_path is None:  # pragma: no cover - repository tests run with uv available
        raise AssertionError("uv executable is required for entry-point installation tests.")

    subprocess.run(  # noqa: S603
        [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--quiet",
            "--no-deps",
            "--target",
            str(site_packages),
            str(package_dir),
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(site_packages))

    return {
        "population": "third_party_population",
        "signal": "third_party_signal",
        "noise": "third_party_noise",
    }


@pytest.mark.requires_gwpy
def test_orchestrator_uses_third_party_entry_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The orchestrator should resolve custom entry-point backends without built-in aliases."""
    aliases = _install_third_party_backend_package(tmp_path, monkeypatch)
    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)

    config = Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 4,
                "duration": 4,
                "start-time": 100,
                "max-samples": 1,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend=aliases["population"],
                source_type="gengli",
                n_samples=1,
                arguments={"path": str(tmp_path / "population.h5"), "source_type": "gengli"},
            ),
            signal=SignalConfig(
                backend=aliases["signal"],
                waveform_model="IMRPhenomD",
                detectors=["H1"],
                output=SimulatorOutputConfig(
                    file_name="signal-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "H1:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                backend=aliases["noise"],
                arguments={"seed": 11, "detectors": ["H1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(
                    file_name="noise-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.json").read_text())
    assert (tmp_path / "output" / "signal" / "signal-0.gwf").exists()
    assert (tmp_path / "output" / "noise" / "noise-0.npy").exists()
    assert metadata["config"]["orchestration"]["population"]["backend"] == aliases["population"]
    assert metadata["config"]["orchestration"]["signal"]["backend"] == aliases["signal"]
    assert metadata["config"]["orchestration"]["noise"]["backend"] == aliases["noise"]
    assert metadata["population"]["source_type"] == "gengli"
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"]["kind"] == "third-party-population"
    assert metadata["simulator_metadata"]["orchestration"]["population"]["seed"] == derive_seed(11, "population")
    assert metadata["simulator_metadata"]["orchestration"]["noise"]["stream_seed"] == derive_seed(11, "noise", "stream")
