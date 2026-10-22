"""
Unit tests for configuration utilities.
"""

from __future__ import annotations

import re
import tempfile
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gwmock.cli.utils.config import (
    Config,
    GlobalsConfig,
    NoiseAdapterConfig,
    OrchestrationConfig,
    PopulationConfig,
    SignalConfig,
    SimulatorConfig,
    SimulatorOutputConfig,
    get_examples_dir,
    get_output_directories,
    load_config,
    merge_parameters,
    resolve_class_path,
    save_config,
    validate_config,
)

# Constants for test assertions to avoid magic values
SEED = 42
SAMPLING_FREQUENCY_2048 = 2048
SAMPLING_FREQUENCY_4096 = 4096
DURATION_4 = 4
DURATION_64 = 64
SAMPLE_RATE_16384 = 16384


def _orchestration_config() -> OrchestrationConfig:
    """Create a minimal orchestration config for tests."""
    return OrchestrationConfig(
        population=PopulationConfig(
            backend="file",
            n_samples=1,
            arguments={"path": "population.h5"},
        ),
        signal=SignalConfig(
            detectors=["H1"],
            output=SimulatorOutputConfig(file_name="signal-{{ counter }}.gwf"),
        ),
        noise=NoiseAdapterConfig(
            output=SimulatorOutputConfig(file_name="noise-{{ counter }}.gwf"),
        ),
    )


class TestSimulatorOutputConfig:
    """Test SimulatorOutputConfig dataclass."""

    def test_valid_output_config(self):
        """Test creating a valid output config."""
        config = SimulatorOutputConfig(file_name="output.gwf")
        assert config.file_name == "output.gwf"
        assert config.arguments == {}
        assert config.output_directory is None
        assert config.metadata_directory is None

    def test_output_config_with_arguments(self):
        """Test output config with arguments."""
        config = SimulatorOutputConfig(
            file_name="{{ detector }}-{{ start_time }}.gwf",
            arguments={"channel": "STRAIN"},
        )
        assert config.arguments == {"channel": "STRAIN"}

    def test_output_config_with_directories(self):
        """Test output config with directory overrides."""
        config = SimulatorOutputConfig(
            file_name="output.gwf",
            output_directory="/custom/output",
            metadata_directory="/custom/metadata",
        )
        assert config.output_directory == "/custom/output"
        assert config.metadata_directory == "/custom/metadata"

    def test_output_config_missing_file_name(self):
        """Test that file_name is required."""
        with pytest.raises(ValidationError):
            SimulatorOutputConfig()


class TestSimulatorConfig:
    """Test SimulatorConfig dataclass."""

    def test_valid_simulator_config(self):
        """Test creating a valid simulator config."""
        config = SimulatorConfig(class_="WhiteNoise")
        assert config.class_ == "WhiteNoise"
        assert config.arguments == {}

    def test_simulator_config_with_arguments(self):
        """Test simulator config with arguments."""
        config = SimulatorConfig(
            class_="WhiteNoise",
            arguments={"seed": 42, "detectors": ["H1", "L1"]},
        )
        assert config.arguments["seed"] == SEED
        assert config.arguments["detectors"] == ["H1", "L1"]

    def test_simulator_config_with_output(self):
        """Test simulator config with output configuration."""
        config = SimulatorConfig(
            class_="WhiteNoise",
            output=SimulatorOutputConfig(file_name="noise.gwf"),
        )
        assert config.output.file_name == "noise.gwf"

    def test_simulator_config_class_alias(self):
        """Test that 'class' alias works (YAML compatibility)."""
        # Using alias directly (as would come from YAML)
        config = SimulatorConfig.model_validate({"class": "WhiteNoise"})
        assert config.class_ == "WhiteNoise"

    def test_simulator_config_missing_class(self):
        """Test that class is required."""
        with pytest.raises(ValidationError):
            SimulatorConfig()


class TestGlobalsConfig:
    """Test GlobalsConfig dataclass."""

    def test_default_globals_config(self):
        """Test GlobalsConfig with default values."""
        config = GlobalsConfig()
        assert config.working_directory == "."
        assert config.output_directory is None
        assert config.metadata_directory is None
        assert config.simulator_arguments == {}
        assert config.output_arguments == {}

    def test_globals_config_with_simulator_arguments(self):
        """Test GlobalsConfig with simulator arguments."""
        config = GlobalsConfig(
            working_directory="/data",
            simulator_arguments={"sampling_frequency": 4096, "duration": 64, "seed": 42},
        )
        assert config.working_directory == "/data"
        assert config.simulator_arguments["sampling_frequency"] == SAMPLING_FREQUENCY_4096
        assert config.simulator_arguments["duration"] == DURATION_64
        assert config.simulator_arguments["seed"] == SEED

    def test_globals_config_with_output_arguments(self):
        """Test GlobalsConfig with output arguments."""
        config = GlobalsConfig(
            output_arguments={"channel": "H1:STRAIN", "sample_rate": 16384},
        )
        assert config.output_arguments["channel"] == "H1:STRAIN"
        assert config.output_arguments["sample_rate"] == SAMPLE_RATE_16384

    def test_globals_config_with_aliases(self):
        """Test that YAML aliases work (snake-case keys)."""
        config = GlobalsConfig.model_validate(
            {
                "working-directory": "/work",
                "output-directory": "/out",
                "metadata-directory": "/meta",
                "simulator-arguments": {"sampling-frequency": 2048, "duration": 4},
                "output-arguments": {"channel": "STRAIN"},
            }
        )
        assert config.working_directory == "/work"
        assert config.output_directory == "/out"
        assert config.metadata_directory == "/meta"
        assert config.simulator_arguments["sampling-frequency"] == SAMPLING_FREQUENCY_2048
        assert config.simulator_arguments["duration"] == DURATION_4
        assert config.output_arguments["channel"] == "STRAIN"

    def test_globals_config_serialization(self):
        """Test YAML serialization with aliases."""
        config = GlobalsConfig(
            working_directory="/data",
            simulator_arguments={"sampling_frequency": 4096},
            output_arguments={"channel": "H1:STRAIN"},
        )
        # Serialize with aliases for YAML export
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert data["working-directory"] == "/data"
        assert data["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_4096
        assert data["output-arguments"]["channel"] == "H1:STRAIN"
        assert "output-directory" not in data  # None values excluded


class TestConfig:
    """Test Config (top-level) dataclass."""

    def test_valid_config(self):
        """Test creating a valid Config."""
        orchestration = _orchestration_config()
        config = Config(orchestration=orchestration)
        assert config.orchestration == orchestration
        assert isinstance(config.globals, GlobalsConfig)

    def test_config_with_globals(self):
        """Test Config with explicit globals."""
        config = Config(
            globals=GlobalsConfig(working_directory="/data"),
            orchestration=_orchestration_config(),
        )
        assert config.globals.working_directory == "/data"

    def test_config_missing_orchestration(self):
        """Test that orchestration field is required."""
        with pytest.raises(ValidationError):
            Config()

    def test_config_rejects_legacy_simulators(self):
        """Legacy simulator configs should raise a validation error."""
        with pytest.raises(ValidationError, match="simulators"):
            Config(
                orchestration=_orchestration_config(),
                simulators={"noise": {"class": "WhiteNoise"}},
            )

    def test_config_serialization(self):
        """Test Config serialization for YAML export."""
        config = Config(
            globals=GlobalsConfig(
                working_directory=".",
                simulator_arguments={"sampling_frequency": 2048},
            ),
            orchestration=_orchestration_config(),
        )
        # Serialize with aliases
        data = config.model_dump(by_alias=True, exclude_none=True)
        assert data["globals"]["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_2048
        assert "orchestration" in data


class TestLoadConfig:
    """Test load_config function."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_yaml = """\
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 2048
  output-arguments: {}
orchestration:
  population:
    backend: file
    n-samples: 1
    arguments:
      path: population.h5
  signal:
    detectors:
      - H1
      - L1
    output:
      file_name: signal-{{ counter }}.gwf
  noise:
    output:
      file_name: noise-{{ counter }}.gwf
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert isinstance(config, Config)
            assert config.globals.simulator_arguments["sampling-frequency"] == SAMPLING_FREQUENCY_2048
            assert config.orchestration.population.backend == "file"
        finally:
            config_path.unlink()

    def test_load_orchestration_config_does_not_warn(self):
        """Fresh orchestration configs should load without warnings."""
        config_yaml = """\
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 64
    duration: 4
    start-time: 1000
    max-samples: 1
orchestration:
  population:
    backend: file
    source-type: bbh
    n-samples: 1
    arguments:
      path: population.h5
  signal:
    detectors:
      - H1
    output:
      file_name: signal-{{ counter }}.gwf
      arguments:
        channel: H1:STRAIN
  noise:
    arguments:
      seed: 7
    output:
      file_name: noise-{{ counter }}.npy
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                config = load_config(config_path)
            assert not record
            assert config.orchestration is not None
            assert config.orchestration.population.backend == "file"
        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to parse YAML"):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_config_missing_orchestration(self):
        """Test loading config without orchestration raises error."""
        config_yaml = "globals:\n  working-directory: .\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestSaveConfig:
    """Test save_config function."""

    def test_save_config_creates_file(self):
        """Test that save_config creates a new file."""
        config = Config(
            globals=GlobalsConfig(working_directory="."),
            orchestration=_orchestration_config(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)
            assert config_path.exists()

    def test_save_config_contains_correct_data(self):
        """Test that saved config contains correct YAML data."""
        config = Config(
            globals=GlobalsConfig(
                working_directory="/data",
                simulator_arguments={"sampling_frequency": 4096},
            ),
            orchestration=_orchestration_config(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)

            # Load and verify
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert data["globals"]["working-directory"] == "/data"
            assert data["globals"]["simulator-arguments"]["sampling_frequency"] == SAMPLING_FREQUENCY_4096
            assert "orchestration" in data

    def test_save_config_file_exists_without_overwrite(self):
        """Test that save_config raises error if file exists and overwrite=False."""
        config = Config(orchestration=_orchestration_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.touch()
            with pytest.raises(FileExistsError):
                save_config(config_path, config, overwrite=False)

    def test_save_config_overwrites_with_flag(self):
        """Test that save_config overwrites when overwrite=True."""
        config = Config(orchestration=_orchestration_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("old content")
            save_config(config_path, config, overwrite=True)
            # Verify new content
            content = config_path.read_text()
            assert "orchestration:" in content
            assert "old content" not in content

    def test_save_config_creates_backup(self):
        """Test that save_config creates backup when overwriting."""
        config = Config(orchestration=_orchestration_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("old content")
            save_config(config_path, config, overwrite=True, backup=True)
            backup_path = config_path.with_suffix(".yaml.backup")
            assert backup_path.exists()
            assert backup_path.read_text() == "old content"


class TestResolveClassPath:
    """Test resolve_class_path function."""

    def test_resolve_class_path_simple_name(self):
        """Test resolving simple class name."""
        path = resolve_class_path("WhiteNoise", "noise")
        assert path == "gwmock.noise.WhiteNoise"

    @pytest.mark.parametrize("class_spec", ["SignalSimulator", "CBCSignalSimulator"])
    def test_resolve_class_path_simple_signal(self, class_spec: str):
        """Removed legacy signal class should raise a migration error."""
        with pytest.raises(
            ValueError, match=rf"Legacy 'simulators\.signal\.class: {class_spec}' is no longer supported"
        ):
            resolve_class_path(class_spec, "signal")

    @pytest.mark.parametrize(
        "class_spec",
        [
            "ColoredNoiseSimulator",
            "gwmock.noise.colored_noise.ColoredNoiseSimulator",
            "CorrelatedNoiseSimulator",
            "StationaryGaussianNoiseSimulator",
        ],
    )
    def test_resolve_class_path_removed_noise(self, class_spec: str):
        """Removed legacy noise classes should raise a migration error."""
        with pytest.raises(
            ValueError, match=rf"Legacy 'simulators\.noise\.class: {class_spec}' is no longer supported"
        ):
            resolve_class_path(class_spec, "noise")

    def test_resolve_class_path_full_path(self):
        """Test resolving full import path."""
        path = resolve_class_path("numpy.random.Generator", "noise")
        assert path == "numpy.random.Generator"

    def test_resolve_class_path_third_party(self):
        """Test resolving third-party class path."""
        path = resolve_class_path("scipy.stats.norm", "noise")
        assert path == "scipy.stats.norm"


class TestMergeParameters:
    """Test merge_parameters function."""

    def test_merge_parameters_empty_global(self):
        """Test merging with empty global config."""
        globals_cfg = GlobalsConfig()
        seed = 42
        sim_args = {"seed": seed}
        merged = merge_parameters(globals_cfg, sim_args)
        assert merged["seed"] == seed
        assert merged["working-directory"] == "."  # Default from globals

    def test_merge_parameters_with_simulator_arguments(self):
        """Test merging with simulator arguments in global config."""
        working_directory = "/data"
        sampling_frequency = 4096
        global_seed = 0
        local_seed = 42

        globals_cfg = GlobalsConfig(
            working_directory=working_directory,
            simulator_arguments={"sampling_frequency": sampling_frequency, "seed": global_seed},
        )
        sim_args = {"seed": local_seed}  # Override seed
        merged = merge_parameters(globals_cfg, sim_args)
        assert merged["seed"] == local_seed  # Simulator overrides
        assert merged["sampling_frequency"] == sampling_frequency  # From globals
        assert merged["working-directory"] == working_directory  # From globals

    def test_merge_parameters_simulator_wins(self):
        """Test that simulator parameters take precedence."""
        global_sampling_frequency = 2048
        global_duration = 64
        local_sampling_frequency = 4096
        globals_cfg = GlobalsConfig(
            simulator_arguments={"sampling_frequency": global_sampling_frequency, "duration": global_duration},
        )
        sim_args = {"sampling_frequency": local_sampling_frequency}
        merged = merge_parameters(globals_cfg, sim_args)
        # Simulator value should win
        assert merged["sampling_frequency"] == local_sampling_frequency
        assert merged["duration"] == global_duration


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_config_valid(self):
        """Test validation of a valid configuration."""
        config = {
            "globals": {"simulator-arguments": {"sampling_frequency": 4096}},
            "orchestration": {
                "population": {"backend": "file", "n-samples": 1, "arguments": {"path": "population.h5"}},
                "signal": {"detectors": ["H1"], "output": {"file_name": "signal.gwf"}},
                "noise": {"output": {"file_name": "noise.gwf"}},
            },
        }
        # Should not raise
        validate_config(config)

    def test_validate_config_missing_orchestration(self):
        """Test validation fails when orchestration section is missing."""
        config = {"globals": {}}
        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config(config)

    def test_validate_config_legacy_simulators(self):
        """Test validation fails when legacy simulators section is present."""
        config = {"simulators": {}, "orchestration": {}}
        with pytest.raises(ValueError, match="simulators"):
            validate_config(config)

    def test_validate_config_invalid_globals_field(self):
        """Test validation fails when globals field is invalid."""
        config = {
            "globals": "invalid",
            "orchestration": {
                "population": {"backend": "file", "n-samples": 1, "arguments": {"path": "population.h5"}},
                "signal": {"detectors": ["H1"], "output": {"file_name": "signal.gwf"}},
                "noise": {"output": {"file_name": "noise.gwf"}},
            },
        }
        with pytest.raises(ValueError, match="'globals' must be a dictionary"):
            validate_config(config)


class TestGetOutputDirectories:
    """Test get_output_directories function."""

    def test_get_output_directories_defaults(self):
        """Test directory resolution with all defaults."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/output/noise")
        assert meta_dir == Path("/work/metadata/noise")

    def test_get_output_directories_global_override(self):
        """Test directory resolution with global override."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
            metadata_directory="/global/metadata",
        )
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/global/output")
        assert meta_dir == Path("/global/metadata")

    def test_get_output_directories_simulator_override(self):
        """Test directory resolution with simulator-specific override."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="/custom/noise",
                metadata_directory="/custom/noise/meta",
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/custom/noise")
        assert meta_dir == Path("/custom/noise/meta")

    def test_get_output_directories_priority_order(self):
        """Test three-level priority: simulator > global > default."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",
        )
        # Simulator only overrides metadata, not output
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                metadata_directory="/custom/metadata",
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/global/output")  # From global
        assert meta_dir == Path("/custom/metadata")  # From simulator

    def test_get_output_directories_relative_simulator_path(self):
        """Test that relative paths in simulator config are prepended with working_dir."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="data/output",  # Relative path
                metadata_directory="data/metadata",  # Relative path
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/data/output")
        assert meta_dir == Path("/work/data/metadata")

    def test_get_output_directories_relative_global_path(self):
        """Test that relative paths in global config are prepended with working_dir."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="output",  # Relative path
            metadata_directory="metadata",  # Relative path
        )
        sim_cfg = SimulatorConfig(class_="Noise")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/output")
        assert meta_dir == Path("/work/metadata")

    def test_get_output_directories_absolute_paths_not_prepended(self):
        """Test that absolute paths are NOT prepended with working_dir."""
        globals_cfg = GlobalsConfig(working_directory="/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="/absolute/output",  # Absolute path
                metadata_directory="/absolute/metadata",  # Absolute path
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/absolute/output")  # Not prepended
        assert meta_dir == Path("/absolute/metadata")  # Not prepended

    def test_get_output_directories_mixed_absolute_relative(self):
        """Test mixed absolute and relative paths."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="./output",  # Relative
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                metadata_directory="/var/data/metadata",  # Absolute
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("/work/./output")  # Relative prepended
        assert meta_dir == Path("/var/data/metadata")  # Absolute not prepended

    def test_get_output_directories_relative_default_working_dir(self):
        """Test relative paths when working_directory itself is relative."""
        globals_cfg = GlobalsConfig(working_directory=".")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="output",  # Relative to working_dir
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        assert out_dir == Path("./output")
        assert meta_dir == Path("./metadata/noise")  # Default falls back to working_dir

    def test_get_output_directories_simulator_relative_overrides_global_absolute(self):
        """Test that simulator relative path is resolved relative to working_dir."""
        globals_cfg = GlobalsConfig(
            working_directory="/work",
            output_directory="/global/output",  # Absolute
        )
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="custom/output",  # Relative - takes precedence
            ),
        )
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise")
        # Simulator relative path is resolved relative to working_dir, not global path
        assert out_dir == Path("/work/custom/output")
        assert meta_dir == Path("/work/metadata/noise")  # Falls back to default

    def test_get_output_directories_custom_working_dir_parameter(self):
        """Test using custom working_directory parameter."""
        globals_cfg = GlobalsConfig(working_directory="/default/work")
        sim_cfg = SimulatorConfig(
            class_="Noise",
            output=SimulatorOutputConfig(
                file_name="noise.gwf",
                output_directory="output",  # Relative path
            ),
        )
        custom_work_dir = Path("/custom/work")
        out_dir, meta_dir = get_output_directories(globals_cfg, sim_cfg, "noise", working_directory=custom_work_dir)
        # Should use custom working_dir, not globals_cfg.working_directory
        assert out_dir == Path("/custom/work/output")
        assert meta_dir == Path("/custom/work/metadata/noise")


class TestConfigRoundTrip:
    """Integration tests for loading and saving configs."""

    def test_round_trip_config(self):
        """Test load -> save -> load round trip."""
        config_yaml = """\
globals:
  working-directory: /data
  simulator-arguments:
    sampling-frequency: 4096
  output-directory: /output
orchestration:
  population:
    backend: file
    n-samples: 1
    arguments:
      path: population.h5
  signal:
    detectors:
      - H1
      - L1
    output:
      file_name: signal-{{ counter }}.gwf
  noise:
    output:
      file_name: noise-{{ counter }}.gwf
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load
            config_path = Path(tmpdir) / "config1.yaml"
            config_path.write_text(config_yaml)
            config1 = load_config(config_path)

            # Save
            config_path2 = Path(tmpdir) / "config2.yaml"
            save_config(config_path2, config1)

            # Load again
            config2 = load_config(config_path2)

            # Verify equivalence
            assert config1.globals.simulator_arguments == config2.globals.simulator_arguments
            assert config1.orchestration == config2.orchestration

    def test_multiple_simulators_round_trip(self):
        """Test round trip with multiple simulators."""
        sampling_frequency = 2048
        config = Config(
            globals=GlobalsConfig(
                working_directory="/data",
                simulator_arguments={"sampling_frequency": sampling_frequency, "duration": 8},
            ),
            orchestration=_orchestration_config(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config_path, config)
            loaded = load_config(config_path)

            assert loaded.orchestration is not None
            assert loaded.globals.simulator_arguments["sampling_frequency"] == sampling_frequency


def test_get_examples_dir_discovers_examples():
    """Test that get_examples_dir() finds the examples directory with YAML files."""
    examples_dir = get_examples_dir()

    # Assert it's a Path object
    assert isinstance(examples_dir, Path)

    # Assert the directory exists
    assert examples_dir.exists(), f"Examples directory not found: {examples_dir}"

    # Assert it's a directory
    assert examples_dir.is_dir(), f"Examples path is not a directory: {examples_dir}"

    # Assert it contains at least one YAML file (to verify discovery works)
    yaml_files = list(examples_dir.rglob("*.yaml"))
    assert len(yaml_files) > 0, f"No YAML files found in examples directory: {examples_dir}"

    # Optional: Check for a known example file (adjust based on your examples/)
    # This ensures the structure is as expected
    expected_files = ["noise/uncorrelated_gaussian/et_triangle_emr/config.yaml"]  # Example; update as needed
    for rel_path in expected_files:
        full_path = examples_dir / rel_path
        assert full_path.exists(), f"Expected example file not found: {full_path}"


class TestSavingConfigsDefensively:
    """The defaults and the failure paths of ``save_config``.

    Every test above passes ``overwrite`` and ``backup`` explicitly, so the values a caller gets
    when it says nothing -- refuse to clobber, and keep a copy when told to clobber -- were not
    pinned by anything.
    """

    @staticmethod
    def _config() -> Config:
        return Config(globals=GlobalsConfig(working_directory="."), orchestration=_orchestration_config())

    def test_an_existing_file_is_refused_by_default(self, tmp_path: Path) -> None:
        """Refusing is the default because the file a user points at is usually one they wrote."""
        path = tmp_path / "config.yaml"
        path.write_text("old content")
        with pytest.raises(FileExistsError, match="Use overwrite=True"):
            save_config(path, self._config())
        assert path.read_text() == "old content"

    def test_the_refusal_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.touch()
        with pytest.raises(FileExistsError, match=re.escape("config.yaml")):
            save_config(path, self._config())

    def test_overwriting_keeps_a_copy_by_default(self, tmp_path: Path) -> None:
        """The backup is the only thing standing between ``--overwrite`` and a lost config."""
        path = tmp_path / "config.yaml"
        path.write_text("old content")
        save_config(path, self._config(), overwrite=True)
        assert (tmp_path / "config.yaml.backup").read_text() == "old content"

    def test_the_copy_can_be_declined(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("old content")
        save_config(path, self._config(), overwrite=True, backup=False)
        assert not (tmp_path / "config.yaml.backup").exists()

    def test_the_backup_sits_beside_the_file_it_copies(self, tmp_path: Path) -> None:
        """Appended to the whole name, so ``config.yaml`` and ``config.yml`` do not share one."""
        path = tmp_path / "config.yaml"
        path.write_text("old")
        save_config(path, self._config(), overwrite=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml", "config.yaml.backup"]

    def test_a_first_save_writes_no_backup(self, tmp_path: Path) -> None:
        """Nothing exists to copy, so asking for a backup must not turn into an error or an empty
        file: all three conditions have to hold before one is written."""
        path = tmp_path / "config.yaml"
        save_config(path, self._config(), overwrite=True, backup=True)
        assert path.exists()
        assert not (tmp_path / "config.yaml.backup").exists()

    def test_nothing_is_left_behind_when_the_write_fails(self, tmp_path: Path, monkeypatch) -> None:
        """The temporary file is the atomic-write staging area; a failed save must not leave it for
        the next reader to find."""
        path = tmp_path / "config.yaml"

        def explode(*_args, **_kwargs):
            raise RuntimeError("no space")

        monkeypatch.setattr("gwmock.cli.utils.config.yaml.safe_dump", explode)

        with pytest.raises(ValueError, match="Failed to save configuration"):
            save_config(path, self._config())

        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_overwrite_leaves_the_original_in_place(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("old content")

        def explode(*_args, **_kwargs):
            raise RuntimeError("no space")

        monkeypatch.setattr("gwmock.cli.utils.config.yaml.safe_dump", explode)

        with pytest.raises(ValueError, match="Failed to save configuration"):
            save_config(path, self._config(), overwrite=True, backup=False)

        assert path.read_text() == "old content"


class TestTheLegacySimulatorClassesAreRefusedWithGuidance:
    """A config written for the removed in-tree simulators must say what to do instead.

    The message is the migration path for anyone with an old config, so the schema keys it names
    are asserted rather than only the fact that something was raised.
    """

    def test_a_removed_signal_simulator_names_the_orchestration_keys(self) -> None:
        with pytest.raises(ValueError, match="no longer supported") as error:
            resolve_class_path("gwmock.signal:CBCSignalSimulator", "signal")
        message = str(error.value)
        assert "orchestration.population" in message
        assert "orchestration.signal" in message
        assert "orchestration.noise" in message

    def test_a_removed_noise_simulator_points_at_the_public_classes(self) -> None:
        with pytest.raises(ValueError, match="no longer supported") as error:
            resolve_class_path("gwmock.noise:StationaryGaussianNoiseSimulator", "noise")
        message = str(error.value)
        assert "orchestration.noise" in message
        assert "gwmock_noise." in message

    def test_the_refusal_quotes_the_class_that_was_asked_for(self) -> None:
        with pytest.raises(ValueError, match=re.escape("gwmock.signal:CBCSignalSimulator")):
            resolve_class_path("gwmock.signal:CBCSignalSimulator", "signal")

    def test_surrounding_whitespace_does_not_get_a_config_past_the_check(self) -> None:
        with pytest.raises(ValueError, match="no longer supported"):
            resolve_class_path("  gwmock.signal:CBCSignalSimulator  ", "signal")

    def test_a_removed_signal_class_is_not_refused_in_the_noise_section(self) -> None:
        """The two lists are per section: only the section that used to hold the class refuses it."""
        assert resolve_class_path("gwmock.signal:CBCSignalSimulator", "noise") == "gwmock.signal:CBCSignalSimulator"

    def test_an_unrelated_class_still_resolves(self) -> None:
        assert resolve_class_path("WhiteNoise", "noise") == "gwmock.noise.WhiteNoise"
        assert resolve_class_path("numpy.random.Generator", "noise") == "numpy.random.Generator"


class TestNoiseArgumentsAreCheckedAtLoadTime:
    """A key the default noise backend refuses must fail ``load_config``, not the run."""

    @staticmethod
    def _write(tmp_path: Path, arguments: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"orchestration": {"noise": {"arguments": {"detectors": ["H1"], **arguments}}}}),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _glitch(**extra) -> dict:
        return {
            "kind": "blip",
            "rate": 0.01,
            "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
            **extra,
        }

    def test_a_top_level_minimum_frequency_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"minimum_frequency.*low_frequency_cutoff"):
            load_config(self._write(tmp_path, {"minimum_frequency": 5.0}))

    def test_a_minimum_frequency_in_a_glitch_model_is_refused(self, tmp_path: Path) -> None:
        """gwmock-noise passes a glitch entry to its model's constructor, which raises a TypeError mid-run."""
        with pytest.raises(ValueError, match=r"glitches\[0\].*minimum_frequency.*low_frequency_cutoff"):
            load_config(self._write(tmp_path, {"glitches": [self._glitch(minimum_frequency=5.0)]}))

    def test_an_unknown_glitch_kind_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"glitches\[0\] has kind 'blips'"):
            load_config(self._write(tmp_path, {"glitches": [{**self._glitch(), "kind": "blips"}]}))

    def test_the_accepted_spellings_load(self, tmp_path: Path) -> None:
        config = load_config(
            self._write(
                tmp_path,
                {"low-frequency-cutoff": 5.0, "glitches": [self._glitch(low_frequency_cutoff=5.0)]},
            )
        )
        assert config.orchestration.noise.arguments["low-frequency-cutoff"] == 5.0

    def test_a_custom_backend_decides_for_itself(self, tmp_path: Path) -> None:
        """A custom backend takes the arguments as constructor keywords; the load does not second-guess it."""
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {"orchestration": {"noise": {"backend": "my.module:Backend", "arguments": {"minimum_frequency": 5}}}}
            ),
            encoding="utf-8",
        )
        assert load_config(path).orchestration.noise.arguments == {"minimum_frequency": 5}
