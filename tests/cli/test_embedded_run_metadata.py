"""What a released HDF5 file says about the run that wrote it, and what it must never say.

A blind mock data challenge is released as the strain files and nothing else: the participants are
asked to find the signals, so the parameters those signals were injected with are the one thing the
release cannot carry. gwmock's provenance record holds them -- that is what it is for -- and the
record lived only in the metadata sidecar, which the producer keeps. Embedding the record in the
HDF5 file makes the data self-describing and, done naively, hands every participant the answer key.

So the embedded copy is the record minus the injection parameters, and a run says explicitly when it
wants them: ``orchestration.include-injection-parameters: true``. The default is the safe one, and
the tests below are written so that they fail if the metadata is not embedded at all -- an assertion
that no injection parameter is present passes trivially against a file carrying no metadata, and
would have gone on passing if the feature had never been wired up.

Three things are checked beyond the flag itself:

* the parameter *values* do not appear anywhere in the embedded document, not merely under the key
  the record spells them with. The orchestrator records its injections three times over -- as
  ``signal.injections``, inside ``signal.metadata`` and again inside ``simulator_metadata`` -- so a
  sanitiser that removes one of them and a test that looks at that same one agree with each other
  and are both wrong. The canary below is a mass no other field in this configuration carries.
* the hashes the sidecar records still match the published bytes. The record is embedded *before*
  the file is hashed, because the other order leaves every ``file_hashes`` entry describing a file
  that no longer exists and makes ``gwmock validate`` fail on every run it wrote.
* the formats with nowhere to put a document -- ``.npy``, ``.gwf`` -- come out byte for byte as
  their writer produced them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
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
from gwmock.cli.utils.hash import compute_file_hash
from tests.cli.test_cli_orchestration import (
    FAKE_NOISE_BACKEND,
    FAKE_SIGNAL_BACKEND,
    FakePopulationBackend,
    _write_signal_file,
)

#: The root attribute a gwmock strain artifact carries its run metadata in.
RUN_METADATA_ATTRIBUTE = "run_metadata"

#: A detector-frame mass no other field of the configuration below holds, so finding it anywhere in
#: an embedded document is proof that an injection parameter reached it -- whatever key it arrived
#: under.
CANARY_MASS = 37.135791
SECOND_MASS = 38.246802

CANARY_POPULATION_BACKEND = "tests.cli.test_embedded_run_metadata:CanaryPopulationBackend"


class CanaryPopulationBackend(FakePopulationBackend):
    """The orchestration test's population, drawn with masses that identify themselves."""

    def simulate(self, n_samples: int, **_kwargs):
        """Return two events whose masses appear nowhere else in the configuration."""
        _ = n_samples
        return {
            "detector_frame_mass_1": np.array([CANARY_MASS, SECOND_MASS]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "coa_time": np.array([100.5, 104.5]),
        }


def _config(tmp_path: Path, *, signal_suffix: str = "hdf5") -> Config:
    """Build the two-batch orchestration used throughout, writing signal in *signal_suffix*."""
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
                backend=CANARY_POPULATION_BACKEND,
                source_type="bbh",
                n_samples=2,
                arguments={"path": str(tmp_path / "population.h5"), "source_type": "bbh"},
            ),
            signal=SignalConfig(
                backend=FAKE_SIGNAL_BACKEND,
                waveform_model="IMRPhenomD",
                detectors=["H1"],
                output=SimulatorOutputConfig(
                    file_name=f"signal-{{{{ counter }}}}.{signal_suffix}",
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


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    include_injection_parameters: bool | None = None,
    signal_suffix: str = "hdf5",
) -> Path:
    """Run the two-batch orchestration and return the configured metadata directory.

    ``include_injection_parameters`` is written into the YAML under the key a user would type,
    rather than passed to the model, so the test exercises the configuration surface itself.
    """
    raw = _config(tmp_path, signal_suffix=signal_suffix).model_dump(by_alias=True, exclude_none=True)
    if include_injection_parameters is not None:
        raw["orchestration"]["include-injection-parameters"] = include_injection_parameters
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw, sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)
    _simulate_impl(str(config_file), overwrite=True, metadata=True)
    return tmp_path / "metadata"


def _embedded_document(path: Path) -> dict[str, Any] | None:
    """Return the run metadata an HDF5 artifact carries, or None if it carries none."""
    import h5py

    with h5py.File(path, "r") as handle:
        if RUN_METADATA_ATTRIBUTE not in handle.attrs:
            return None
        raw = handle.attrs[RUN_METADATA_ATTRIBUTE]
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))


def _embedded_text(path: Path) -> str:
    """Return the embedded document as the text it is stored as, for a value-level leak scan."""
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle.attrs[RUN_METADATA_ATTRIBUTE]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _keys_named(document: Any, name: str) -> list[Any]:
    """Return every value stored under *name*, at any depth of the document."""
    found: list[Any] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key == name:
                found.append(value)
            found.extend(_keys_named(value, name))
    elif isinstance(document, list):
        for item in document:
            found.extend(_keys_named(item, name))
    return found


class TestTheDefault:
    """What a run that says nothing about injection parameters writes into its HDF5 output."""

    def test_the_hdf5_output_carries_the_run_metadata(self, monkeypatch, tmp_path: Path):
        """The point of embedding: the file describes the run that produced it."""
        _run(monkeypatch, tmp_path)

        document = _embedded_document(tmp_path / "output" / "signal" / "signal-0.hdf5")

        assert document is not None
        assert document["schema_version"]
        assert document["config"]["orchestration"]["population"]["backend"] == CANARY_POPULATION_BACKEND
        assert [Path(output["path"]).name for output in document["outputs"]]

    def test_no_injection_parameter_is_embedded(self, monkeypatch, tmp_path: Path):
        """The blind-MDC guarantee, checked by key and by value."""
        _run(monkeypatch, tmp_path)

        signal_output = tmp_path / "output" / "signal" / "signal-0.hdf5"
        document = _embedded_document(signal_output)

        # Fails loudly rather than vacuously if nothing was embedded at all.
        assert document is not None, "no run metadata was embedded, so this assertion proves nothing"
        assert _keys_named(document, "injections") == []
        assert str(CANARY_MASS) not in _embedded_text(signal_output)

    def test_the_sidecar_still_records_the_injection_parameters(self, monkeypatch, tmp_path: Path):
        """The producer's own copy is unchanged; only the copy inside the data file is trimmed."""
        metadata_directory = _run(monkeypatch, tmp_path)

        sidecar = json.loads((metadata_directory / "orchestration-0.metadata.json").read_text())

        injections = sidecar["signal"]["injections"]
        assert [injection["event_id"] for injection in injections] == [0]
        assert injections[0]["parameters"]["detector_frame_mass_1"] == CANARY_MASS


class TestOptingIn:
    """What ``include-injection-parameters: true`` changes, and what it does not."""

    def test_the_injection_parameters_are_embedded(self, monkeypatch, tmp_path: Path):
        """A training set is self-describing down to the parameters that made it."""
        _run(monkeypatch, tmp_path, include_injection_parameters=True)

        signal_output = tmp_path / "output" / "signal" / "signal-0.hdf5"
        document = _embedded_document(signal_output)

        assert document is not None
        injections = document["signal"]["injections"]
        assert [injection["event_id"] for injection in injections] == [0]
        assert injections[0]["parameters"]["detector_frame_mass_1"] == CANARY_MASS

    def test_the_second_batch_carries_its_own_event(self, monkeypatch, tmp_path: Path):
        """Each frame embeds the record of the batch that wrote it, not the run's first one."""
        _run(monkeypatch, tmp_path, include_injection_parameters=True)

        document = _embedded_document(tmp_path / "output" / "signal" / "signal-1.hdf5")

        assert document is not None
        assert [injection["event_id"] for injection in document["signal"]["injections"]] == [1]
        assert document["signal"]["injections"][0]["parameters"]["detector_frame_mass_1"] == SECOND_MASS


class TestReproducingARun:
    """The decision travels in the record, so a rerun from metadata makes the same one."""

    def test_the_choice_is_recorded_and_honoured_on_reproduction(self, monkeypatch, tmp_path: Path):
        """Reproducing an opted-in run must not silently produce files without the parameters --
        the reproduction would then not be one, and the difference would be invisible until someone
        looked for the parameters and found the file bare."""
        metadata_directory = _run(monkeypatch, tmp_path, include_injection_parameters=True)
        signal_output = tmp_path / "output" / "signal" / "signal-0.hdf5"
        signal_output.unlink()

        monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)
        _simulate_impl(str(metadata_directory), overwrite=True)

        document = _embedded_document(signal_output)
        assert document is not None
        assert document["signal"]["injections"][0]["parameters"]["detector_frame_mass_1"] == CANARY_MASS


class TestWhatEmbeddingMustNotBreak:
    """The published bytes, the formats that cannot carry a document, and the sidecar's hashes."""

    def test_the_recorded_hashes_match_the_published_files(self, monkeypatch, tmp_path: Path):
        """Embedding happens before hashing, or every hash the run records is of a stale file."""
        metadata_directory = _run(monkeypatch, tmp_path)

        sidecar = json.loads((metadata_directory / "orchestration-0.metadata.json").read_text())
        working_directory = Path(sidecar["config"]["globals"]["working-directory"])

        assert sidecar["file_hashes"]
        for name, recorded in sidecar["file_hashes"].items():
            match = next(output for output in sidecar["outputs"] if Path(output["path"]).name == name)
            published = working_directory / match["path"]
            assert compute_file_hash(published) == recorded
            assert match["sha256"] == recorded

    def test_a_format_without_metadata_space_is_untouched(self, monkeypatch, tmp_path: Path):
        """``.npy`` is a bare array container; the noise output must survive unchanged."""
        _run(monkeypatch, tmp_path)

        noise_output = tmp_path / "output" / "noise" / "noise-0.npy"

        assert np.load(noise_output).shape == (16,)

    def test_a_gwf_output_is_untouched(self, monkeypatch, tmp_path: Path):
        """A frame is composed from a fixed set of fields, so nothing is written into one."""
        _run(monkeypatch, tmp_path, signal_suffix="gwf")

        # The orchestration fake writes the frame's stand-in bytes; embedding must not append to it.
        assert (tmp_path / "output" / "signal" / "signal-0.gwf").read_text() == "STRAIN"


class TestTheConfigurationFlag:
    """The flag itself: its default, its spelling, and what it refuses."""

    def test_it_defaults_to_excluding_them(self):
        """Safe by default -- a config that says nothing releases nothing."""
        config = OrchestrationConfig(
            signal=SignalConfig(source_type="sgwb", backend=FAKE_SIGNAL_BACKEND, detectors=["H1"]),
        )

        assert config.include_injection_parameters is False

    def test_it_is_spelled_with_hyphens_in_yaml(self):
        """The key a user types, as the published feature request proposes it."""
        config = OrchestrationConfig.model_validate(
            {
                "signal": {"source-type": "sgwb", "backend": FAKE_SIGNAL_BACKEND, "detectors": ["H1"]},
                "include-injection-parameters": True,
            }
        )

        assert config.include_injection_parameters is True
        assert config.model_dump(by_alias=True)["include-injection-parameters"] is True

    def test_it_refuses_a_value_that_is_not_a_flag(self):
        """A typo in a safety switch is an error, not a truthy string."""
        with pytest.raises(ValueError, match="include-injection-parameters"):
            OrchestrationConfig.model_validate(
                {
                    "signal": {"source-type": "sgwb", "backend": FAKE_SIGNAL_BACKEND, "detectors": ["H1"]},
                    "include-injection-parameters": "sometimes",
                }
            )
