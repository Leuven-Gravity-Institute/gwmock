"""
Utility functions to load and save configuration files.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger("gwmock")

_LEGACY_SIMULATORS_MIGRATION_URL = "https://leuven-gravity-institute.github.io/gwmock/user-guide/orchestration/"
_LEGACY_SIMULATORS_REMOVAL_VERSION = "v0.5.0"

_REMOVED_SIGNAL_SIMULATOR_CLASS_SPECS = frozenset(
    {
        "SignalSimulator",
        "gwmock.signal.SignalSimulator",
        "gwmock.signal.base.SignalSimulator",
        "gwmock.signal:SignalSimulator",
        "gwmock.signal.base:SignalSimulator",
        "CBCSignalSimulator",
        "gwmock.signal.CBCSignalSimulator",
        "gwmock.signal.cbc.CBCSignalSimulator",
        "gwmock.signal:CBCSignalSimulator",
        "gwmock.signal.cbc:CBCSignalSimulator",
    }
)

_REMOVED_NOISE_SIMULATOR_CLASS_SPECS = frozenset(
    {
        "NoiseSimulator",
        "gwmock.noise.NoiseSimulator",
        "gwmock.noise.base.NoiseSimulator",
        "gwmock.noise:NoiseSimulator",
        "gwmock.noise.base:NoiseSimulator",
        "ColoredNoiseSimulator",
        "gwmock.noise.ColoredNoiseSimulator",
        "gwmock.noise.colored_noise.ColoredNoiseSimulator",
        "gwmock.noise:ColoredNoiseSimulator",
        "gwmock.noise.colored_noise:ColoredNoiseSimulator",
        "CorrelatedNoiseSimulator",
        "gwmock.noise.CorrelatedNoiseSimulator",
        "gwmock.noise.correlated_noise.CorrelatedNoiseSimulator",
        "gwmock.noise:CorrelatedNoiseSimulator",
        "gwmock.noise.correlated_noise:CorrelatedNoiseSimulator",
        "StationaryGaussianNoiseSimulator",
        "gwmock.noise.StationaryGaussianNoiseSimulator",
        "gwmock.noise.stationary_gaussian.StationaryGaussianNoiseSimulator",
        "gwmock.noise:StationaryGaussianNoiseSimulator",
        "gwmock.noise.stationary_gaussian:StationaryGaussianNoiseSimulator",
    }
)


def _raise_removed_signal_simulator_error(class_spec: str) -> None:
    raise ValueError(
        f"Legacy 'simulators.signal.class: {class_spec}' is no longer supported because "
        "the in-tree signal simulator classes have been removed. Migrate this config to the adapter-backed "
        "'orchestration' schema using 'orchestration.population', 'orchestration.signal', and "
        "'orchestration.noise'."
    )


def _raise_removed_noise_simulator_error(class_spec: str) -> None:
    raise ValueError(
        f"Legacy 'simulators.noise.class: {class_spec}' is no longer supported because "
        "the in-tree noise simulator classes have been removed. Use 'orchestration.noise' for new runs, "
        "or point 'simulators.noise.class' at a public 'gwmock_noise.*' class instead."
    )


class SimulatorOutputConfig(BaseModel):
    """Configuration for simulator output handling."""

    file_name: str | list[str] = Field(
        ...,
        description=(
            "Output file name template (supports {{ variable }} placeholders), "
            "or a pre-resolved list of filenames as stored in batch metadata."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Output-specific arguments (e.g., channel name)"
    )
    output_directory: str | None = Field(
        default=None, description="Optional directory override for this simulator's output"
    )
    metadata_directory: str | None = Field(
        default=None, description="Optional directory override for this simulator's metadata"
    )

    # Allow unknown fields
    model_config = ConfigDict(extra="allow")


class SimulatorConfig(BaseModel):
    """Configuration for a single simulator."""

    class_: str = Field(alias="class", description="Simulator class name or full import path")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to simulator constructor")
    output: SimulatorOutputConfig = Field(
        default_factory=lambda: SimulatorOutputConfig(file_name="output-{{counter}}.hdf5"),
        description="Output configuration for this simulator",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Validate class specification is non-empty."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("'class' must be a non-empty string")
        return v


class PopulationConfig(BaseModel):
    """Adapter-backed population configuration."""

    backend: str = Field(..., description="Public gwmock-pop backend or loader name")
    n_samples: int | None = Field(
        default=None,
        alias="n-samples",
        description="Number of population events to draw. Omit to load the full catalogue (file-backed loaders only).",
    )
    source_type: str | None = Field(default=None, alias="source-type", description="Optional explicit source type")
    sort_by: str | None = Field(default="coa_time", alias="sort-by", description="Event ordering key")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the population backend")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_name(cls, v: str) -> str:
        """Validate the backend identifier is non-empty."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("'backend' must be a non-empty string")
        return v

    @field_validator("n_samples")
    @classmethod
    def validate_n_samples(cls, v: int | None) -> int | None:
        """Reject n_samples=0 or negative; None means load the full catalogue."""
        if v is not None and v <= 0:
            raise ValueError("'n-samples' must be a positive integer")
        return v


class SignalConfig(BaseModel):
    """Adapter-backed signal configuration."""

    backend: str | None = Field(
        default=None,
        description="Optional public gwmock-signal backend alias, entry point, or import path",
    )
    source_type: str | None = Field(
        default=None,
        alias="source-type",
        description="Optional source type for signal-only orchestration, e.g. 'sgwb'.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the gwmock-signal backend constructor.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed signal parameters passed to the backend simulate method.",
    )
    waveform_model: str | None = Field(default=None, alias="waveform-model", description="Waveform model name")
    waveform_backend: str | None = Field(
        default=None,
        alias="waveform-backend",
        description=(
            "Which waveform library generates the polarizations: a built-in alias "
            "('lal', 'pycbc', 'ripple', 'gwsignal'), an entry point in 'gwmock.waveform', or a "
            "'module:Class' reference. Defaults to LAL. Note this selects a library, not a "
            "compute device: 'ripple' generates with ripple through the same per-event path."
        ),
    )
    waveform_backend_arguments: dict[str, Any] = Field(
        default_factory=dict,
        alias="waveform-backend-arguments",
        description=(
            "Constructor arguments for the waveform backend, e.g. 'f_ref', 'ringdown_fraction', "
            "'segment_duration', or ripple's 'taper_fraction'. Distinct from 'arguments', which "
            "goes to the simulator rather than to the waveform backend."
        ),
    )
    waveform_arguments: dict[str, Any] = Field(
        default_factory=dict,
        alias="waveform-arguments",
        description="Fixed waveform arguments passed to gwmock-signal",
    )
    waveform_options: dict[str, Any] = Field(
        default_factory=dict,
        alias="waveform-options",
        description=(
            "Extra waveform options (e.g. LAL dictionary entries such as ModeArray) "
            "passed to gwmock-signal as its waveform_arguments parameter"
        ),
    )
    detectors: list[str] = Field(..., description="Detector names or detector config paths")
    minimum_frequency: float = Field(default=5.0, alias="minimum-frequency", description="Minimum waveform frequency")
    earth_rotation: bool = Field(
        default=True,
        alias="earth-rotation",
        description="Whether to project using time-dependent detector response",
    )
    execution: str = Field(
        default="per-event",
        alias="execution",
        description=(
            "How the segment's events are generated: 'per-event' (default) calls the backend once "
            "per event, 'batched' generates them together through gwmock-signal's batched path, "
            "which is what makes GPU execution possible. Distinct from 'waveform-backend', which "
            "chooses the library: 'batched' always generates with ripple, and whether that runs on "
            "a GPU depends on the JAX device available, not on this setting."
        ),
    )
    output: SimulatorOutputConfig = Field(
        default_factory=lambda: SimulatorOutputConfig(
            # HDF5 by default: it is the format gwmock produces, and the one its own tooling reads
            # without a frame library. GWF remains fully supported and is selected the same way any
            # format is, by naming it here -- it exists so other gravitational-wave pipelines can read
            # what gwmock writes, not because a run needs it.
            file_name="signal-{{ detectors }}-{{ start_time }}-{{ duration }}.hdf5",
            output_directory="signal",
            arguments={"channel": "{{ detectors }}:STRAIN"},
        ),
        description="Signal output configuration",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("detectors")
    @classmethod
    def validate_detectors(cls, v: list[str]) -> list[str]:
        """Require at least one detector."""
        if not v:
            raise ValueError("'detectors' must contain at least one detector")
        return v

    @field_validator("execution")
    @classmethod
    def validate_execution(cls, v: str) -> str:
        """Reject an unknown execution mode rather than silently falling back to per-event.

        A typo here would otherwise leave the run on the default path while the author believed
        they had switched it, and the output would look entirely normal.
        """
        allowed = {"per-event", "batched"}
        if v not in allowed:
            raise ValueError(f"'execution' must be one of {sorted(allowed)}, got {v!r}")
        return v


class NoiseAdapterConfig(BaseModel):
    """Adapter-backed noise configuration."""

    backend: str | None = Field(
        default=None,
        description="Optional public gwmock-noise backend alias, entry point, or import path",
    )
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the gwmock-noise adapter")
    output: SimulatorOutputConfig = Field(
        default_factory=lambda: SimulatorOutputConfig(
            # HDF5 by default, as for signal output above.
            file_name="noise-{{ counter }}.hdf5",
            output_directory="noise",
            arguments={"channel": "MOCK_NOISE"},
        ),
        description="Noise output configuration",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OrchestrationConfig(BaseModel):
    """Composite adapter-backed orchestration configuration."""

    noise: NoiseAdapterConfig | None = None
    population: PopulationConfig | None = None
    signal: SignalConfig | None = None
    include_injection_parameters: bool = Field(
        default=False,
        alias="include-injection-parameters",
        description=(
            "Embed the source parameters of the injected signals in the HDF5 output as well as in the "
            "metadata sidecar. False by default: a blind mock data challenge is released as the data "
            "files alone, and those parameters are the answer its participants are asked to find. Set "
            "it for data generated for a different purpose -- a training set, a benchmark, a released "
            "'solved' challenge. It does not change the sidecar, which always records them."
        ),
    )

    @model_validator(mode="after")
    def _validate_combinations(self) -> OrchestrationConfig:
        if self.noise is None and self.population is None and self.signal is None:
            raise ValueError("orchestration must define at least one of: noise, population, signal")
        if self.signal is not None and self.population is None and self.signal.source_type != "sgwb":
            raise ValueError(
                "orchestration.signal without orchestration.population is only supported when "
                "signal.source_type is 'sgwb'; other source types require orchestration.population "
                "and will fail in AdapterOrchestrator._simulate()"
            )
        return self

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class GlobalsConfig(BaseModel):
    """Global configuration applying to all simulators.

    This configuration provides universal directory settings and fallback arguments
    for simulators and output handlers. The simulator_arguments and output_arguments
    are agnostic to simulator type, supporting both time-series and population simulators.
    """

    working_directory: str = Field(
        default=".", alias="working-directory", description="Base working directory for all output"
    )
    output_directory: str | None = Field(
        default=None, alias="output-directory", description="Default output directory (can be overridden per simulator)"
    )
    metadata_directory: str | None = Field(
        default=None,
        alias="metadata-directory",
        description="Default metadata directory (can be overridden per simulator)",
    )
    simulator_arguments: dict[str, Any] = Field(
        default_factory=dict,
        alias="simulator-arguments",
        description="Global default arguments for simulators (e.g., sampling-frequency, duration, seed). "
        "Simulator-specific arguments override these.",
    )
    output_arguments: dict[str, Any] = Field(
        default_factory=dict,
        alias="output-arguments",
        description="Global default arguments for output handlers (e.g., channel names). "
        "Simulator-specific output arguments override these.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ChunkConfig(BaseModel):
    """Configuration for splitting simulations into parallel chunks."""

    enabled: bool = Field(default=False, description="Enable chunking for parallel execution")
    n_chunks: int = Field(default=1, alias="n-chunks", description="Number of chunks to split the simulation into")
    parallel: bool = Field(default=True, description="Run chunks in parallel (local) or submit as array job (SLURM)")

    @field_validator("n_chunks")
    @classmethod
    def validate_n_chunks(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n-chunks must be at least 1")
        return v

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BatchConfig(BaseModel):
    """Batch configuration applying to all simulators."""

    scheduler: str = Field(
        default="slurm", alias="scheduler", description="Name of the scheduler (`slurm` or `htcondor`)"
    )
    job_name: str = Field(default="gwmock_job", alias="job-name", description="Name of the job")
    resources: dict[str, Any] = Field(
        default_factory=dict,
        alias="resources",
        description="Default resources for the simulation (e.g., nodes, ntasks_per_node, cpus_per_task, mem)",
    )
    submit: dict[str, Any] | None = Field(
        default=None,
        alias="submit",
        description="Additional sbatch options (e.g., account, cluster, time, partition)",
    )
    extra_lines: list[str] | None = Field(
        default=None,
        alias="extra_lines",
        description="Custom lines to insert into the submit script before the simulation command (e.g., module loads, conda activate)",  # pylint: disable=line-too-long
    )
    chunks: ChunkConfig | None = Field(default=None, description="Chunking configuration for parallel execution")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Config(BaseModel):
    """Top-level configuration model."""

    globals: GlobalsConfig = Field(default_factory=GlobalsConfig, description="Global configuration")
    orchestration: OrchestrationConfig = Field(..., description="Adapter-backed orchestration configuration")
    batch: BatchConfig | None = Field(default=None, description="Resources and scheduler configuration")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_simulators(cls, data: Any) -> Any:
        """Reject removed legacy simulator configs before model parsing."""
        if isinstance(data, dict) and "simulators" in data:
            raise ValueError(
                "Legacy 'simulators' configurations are no longer supported. Use the adapter-backed 'orchestration' schema."
            )
        return data


def load_config(file_name: Path, encoding: str = "utf-8") -> Config:
    """Load configuration file with validation.

    Args:
        file_name (Path): File name.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        Config: Validated configuration dataclass.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid or cannot be parsed.
    """
    if not file_name.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_name}")
    try:
        with file_name.open(encoding=encoding) as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must be a YAML dictionary")

    # Validate and convert to Config dataclass
    try:
        config = Config(**raw_config)
        logger.info("Configuration loaded and validated")
        return config
    except (ValueError, ValidationError) as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def save_config(
    file_name: Path, config: Config, overwrite: bool = False, encoding: str = "utf-8", backup: bool = True
) -> None:
    """Save configuration to YAML file safely.

    Args:
        file_name: Path to save configuration to
        config: Config dataclass instance
        overwrite: If True, overwrite existing file
        encoding: File encoding (default: utf-8)
        backup: If True and overwriting, create backup

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    if file_name.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {file_name}. Use overwrite=True to overwrite.")

    # Create backup if needed
    if file_name.exists() and overwrite and backup:
        backup_path = file_name.with_suffix(f"{file_name.suffix}.backup")
        logger.info("Creating backup: %s", backup_path)
        backup_path.write_text(file_name.read_text(encoding=encoding), encoding=encoding)

    # Atomic write
    temp_file = file_name.with_suffix(f"{file_name.suffix}.tmp")
    try:
        # Convert to dict, excluding internal fields and defaults
        config_dict = config.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)

        with temp_file.open("w", encoding=encoding) as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        temp_file.replace(file_name)
        logger.info("Configuration saved to: %s", file_name)

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise ValueError(f"Failed to save configuration: {e}") from e


def validate_config(config: dict) -> None:
    """Validate configuration structure and provide helpful error messages.

    Args:
        config (dict): Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid with detailed error message
    """
    has_orchestration = "orchestration" in config

    if "simulators" in config:
        raise ValueError(
            "Invalid configuration: legacy 'simulators' is no longer supported; use 'orchestration' instead"
        )

    if not has_orchestration:
        raise ValueError("Invalid configuration: must define 'orchestration'")

    orchestration = config["orchestration"]
    if not isinstance(orchestration, dict):
        raise ValueError("'orchestration' must be a dictionary")

    known_sections = {"population", "signal", "noise"}
    present_sections = known_sections & set(orchestration.keys())
    if not present_sections:
        raise ValueError("'orchestration' must define at least one of: noise, population, signal")
    for section in present_sections:
        if not isinstance(orchestration[section], dict):
            raise ValueError(f"'orchestration.{section}' must be a dictionary")
    if "signal" in orchestration and "population" not in orchestration:
        raise ValueError("'orchestration.signal' requires 'orchestration.population'")

    # Validate globals section if present
    if "globals" in config:
        globals_config = config["globals"]
        if not isinstance(globals_config, dict):
            raise ValueError("'globals' must be a dictionary")

    logger.info("Configuration validation passed")


def resolve_class_path(class_spec: str, section_name: str | None) -> str:
    """Resolve class specification to full module path.

    Args:
        class_spec: Either 'ClassName' or 'third_party.module.ClassName'
        section_name: Section name (e.g., 'noise', 'signal', 'glitch')

    Returns:
        Full path like 'gwmock.noise.ClassName' or 'third_party.module.ClassName'

    Examples:
        resolve_class_path("WhiteNoise", "noise") -> "gwmock.noise.WhiteNoise"
        resolve_class_path("numpy.random.Generator", "noise") -> "numpy.random.Generator"
    """
    if section_name == "signal" and class_spec.strip() in _REMOVED_SIGNAL_SIMULATOR_CLASS_SPECS:
        _raise_removed_signal_simulator_error(class_spec.strip())
    if section_name == "noise" and class_spec.strip() in _REMOVED_NOISE_SIMULATOR_CLASS_SPECS:
        _raise_removed_noise_simulator_error(class_spec.strip())
    if "." not in class_spec and section_name:
        # Just a class name - use section_name as submodule, class imported in __init__.py
        return f"gwmock.{section_name}.{class_spec}"
    # Contains dots - assume it's a third-party package, use as-is
    return class_spec


def merge_parameters(globals_config: GlobalsConfig, simulator_args: dict[str, Any]) -> dict[str, Any]:
    """Merge global and simulator-specific parameters.

    Flattens simulator_arguments from globals into the result, then applies
    simulator-specific overrides.

    Args:
        globals_config: GlobalsConfig dataclass instance
        simulator_args: Simulator-specific arguments dict

    Returns:
        Merged parameters with simulator args taking precedence

    Note:
        Simulator_arguments from globals_config are flattened into the result.
        Directory settings (working-directory, output-directory, metadata-directory)
        are included. Output_arguments are not included (handled separately).
    """
    # Start with directory settings from globals
    merged = {}
    if globals_config.working_directory:
        merged["working-directory"] = globals_config.working_directory
    if globals_config.output_directory:
        merged["output-directory"] = globals_config.output_directory
    if globals_config.metadata_directory:
        merged["metadata-directory"] = globals_config.metadata_directory

    # Flatten simulator_arguments from globals
    merged.update(globals_config.simulator_arguments)

    # Override with simulator-specific arguments (takes precedence)
    merged.update(simulator_args)

    return merged


def get_output_directories(
    globals_config: GlobalsConfig,
    simulator_config: SimulatorConfig,
    simulator_name: str,
    working_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Get output and metadata directories for a simulator.

    Args:
        globals_config: Global configuration
        simulator_config: Simulator-specific configuration
        simulator_name: Name of the simulator
        working_directory: Override working directory (for testing)

    Returns:
        Tuple of (output_directory, metadata_directory)

    Priority (highest to lowest):
        1. Simulator output.output_directory / output.metadata_directory
        2. Global output-directory / metadata-directory
        3. working-directory / output / {simulator_name}

    Examples:
        >>> globals_cfg = GlobalsConfig(working_directory="/data")
        >>> sim_cfg = SimulatorConfig(class_="Noise")
        >>> get_output_directories(globals_cfg, sim_cfg, "noise")
        (Path("/data/output/noise"), Path("/data/output/noise"))
    """
    working_dir = working_directory or Path(globals_config.working_directory)

    # Simulator-specific overrides
    if simulator_config.output.output_directory:
        output_path = Path(simulator_config.output.output_directory)
        # Prepend working_dir if path is relative
        output_directory = output_path if output_path.is_absolute() else working_dir / output_path
    elif globals_config.output_directory:
        output_path = Path(globals_config.output_directory)
        # Prepend working_dir if path is relative
        output_directory = output_path if output_path.is_absolute() else working_dir / output_path
    else:
        output_directory = working_dir / "output" / simulator_name

    if simulator_config.output.metadata_directory:
        metadata_path = Path(simulator_config.output.metadata_directory)
        # Prepend working_dir if path is relative
        metadata_directory = metadata_path if metadata_path.is_absolute() else working_dir / metadata_path
    elif globals_config.metadata_directory:
        metadata_path = Path(globals_config.metadata_directory)
        # Prepend working_dir if path is relative
        metadata_directory = metadata_path if metadata_path.is_absolute() else working_dir / metadata_path
    else:
        metadata_directory = working_dir / "metadata" / simulator_name

    return output_directory, metadata_directory


def get_examples_dir() -> Path:
    """Get the path to the examples directory.

    Returns:
        Path to the examples directory.
    """
    try:
        examples_resource = importlib.resources.files("gwmock") / "examples"
        examples_path = Path(str(examples_resource))
        if examples_path.exists() and list(examples_path.rglob("*.yaml")):
            return examples_path
    except (TypeError, AttributeError):
        logger.warning("Could not access examples via importlib.resources, falling back to filesystem search.")

    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        examples_path = project_root / "examples"
        if examples_path.exists():
            return examples_path
    except Exception:  # pylint: disable=broad-except
        logger.error("Could not determine project root for examples directory.")
    raise FileNotFoundError("Could not locate the examples directory.")
