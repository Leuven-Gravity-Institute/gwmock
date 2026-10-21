"""Utility functions for creating and managing simulation plans."""

from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gwmock.cli.utils.config import Config, GlobalsConfig, OrchestrationConfig, SimulatorConfig
from gwmock.cli.utils.config_resolution import resolve_max_samples
from gwmock.cli.utils.hash import compute_file_hash
from gwmock.cli.utils.metadata import (
    PRIVATE_REFERENCE_KEY,
    SCHEMA_VERSION,
    MetadataRecord,
    load_metadata_with_external_state,
    with_private_details,
)
from gwmock.utils.log import get_dependency_versions

logger = logging.getLogger("gwmock")


@dataclass
class SimulationBatch:
    """Data class representing a single simulation batch.

    A batch is a unit of work for a particular simulator. For example:
    - A noise simulator might generate multiple batches (segments) of noise data
    - A signal simulator might generate multiple batches of gravitational wave signals

    The batch_index is per-simulator, so batch 0 from noise simulator and batch 0 from
    signal simulator are different batches.

    Metadata can contain two types of information:
    - Configuration metadata: Full config + max_samples (for reproducibility with fresh state)
    - State metadata: Pre-batch state (RNG state, etc.) for exact reproduction of a specific batch
    """

    simulator_name: str
    """Name of the simulator (e.g., 'noise', 'signal', 'glitch')"""

    simulator_config: SimulatorConfig | OrchestrationConfig
    """Configuration for this simulator"""

    globals_config: GlobalsConfig
    """Global configuration (shared across all simulators)"""

    batch_index: int
    """Index of this batch within the simulator (0-based, per-simulator)"""

    # Optional: For metadata-based reproduction
    metadata_file: Path | None = None
    """If reproducing from metadata, path to the metadata file"""

    batch_metadata: dict[str, Any] | None = None
    """Parsed metadata content (if metadata_file is provided)"""

    # State snapshot for exact reproduction
    pre_batch_state: dict[str, Any] | None = None
    """State snapshot taken before this batch was generated.

    Contains simulator-specific state that cannot be known a priori:
    - RNG state (numpy.random.RandomState or similar)
    - Simulator internal state (e.g., filter memory for colored noise)
    - Other stateful components

    If present, use this for exact reproduction. Otherwise, reconstruct from config.
    """

    # For tracking
    source: str = "config"
    """Source of this batch: 'config' (fresh), 'metadata_config' (from saved config),
    or 'metadata_state' (from saved state snapshot)"""

    author: str | None = None
    """Author of this batch (from metadata)"""

    email: str | None = None
    """Email of the author (from metadata)"""

    config_payload: dict[str, Any] | None = None
    """Resolved top-level config snapshot preserved in metadata."""

    config_sha256: str | None = None
    """SHA256 of the original config file when known."""

    def __post_init__(self):
        """Post-initialization checks.

        Raises:
            ValueError: If simulator_name is empty or batch_index is negative.
        """
        if not self.simulator_name:
            raise ValueError("simulator_name must not be empty")
        if self.batch_index < 0:
            raise ValueError("batch_index must be non-negative")

    def is_metadata_based(self) -> bool:
        """Check if this batch is based on saved metadata.

        Returns:
            True if the batch is based on metadata, False otherwise.
        """
        return self.source in ("metadata_config", "metadata_state")

    def has_state_snapshot(self) -> bool:
        """Check if this batch has a pre-batch state snapshot for exact reproduction.

        Returns:
            True if pre_batch_state is available, False otherwise.
        """
        return self.pre_batch_state is not None


@dataclass
class SimulationPlan:
    """Data class representing a simulation plan."""

    batches: list[SimulationBatch] = field(default_factory=list)
    """List of batches to simulate"""

    source_config: Config | None = None
    """Original Config object (if config-based)"""

    checkpoint_directory: Path = Path("checkpoints")
    """Directory for checkpoint files"""

    total_batches: int = 0
    """Total number of batches"""

    def add_batch(self, batch: SimulationBatch) -> None:
        """Add a batch to the plan.

        Args:
            batch: SimulationBatch to add
        """
        self.batches.append(batch)
        self.total_batches = len(self.batches)
        logger.debug(
            "Added batch %d: simulator=%s, source=%s",
            batch.batch_index,
            batch.simulator_name,
            batch.source,
        )

    def get_batches_for_simulator(self, simulator_name: str) -> list[SimulationBatch]:
        """Get all batches for a specific simulator.

        Args:
            simulator_name: Name of the simulator

        Returns:
            List of batches for that simulator, in order
        """
        return [b for b in self.batches if b.simulator_name == simulator_name]


def parse_batch_metadata(metadata_file: Path, metadata_dir: Path | None = None) -> dict[str, Any]:
    """Parse a batch metadata file.

    Args:
        metadata_file: Path to BATCH-*.metadata.json or legacy metadata YAML file
        metadata_dir: Directory containing external state files. If None, uses parent of metadata_file

    Returns:
        Parsed metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If metadata is invalid
    """
    metadata = load_metadata_with_external_state(metadata_file, metadata_dir)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    if "schema_version" in metadata:
        metadata = MetadataRecord.model_validate(metadata).model_dump(mode="python", by_alias=True, exclude_none=True)

    return metadata


_METADATA_JSON_SUFFIX = ".metadata.json"
_METADATA_YAML_SUFFIX = ".metadata.yaml"


def _metadata_file_format_rank(path: Path) -> int:
    """Prefer JSON batch metadata over legacy YAML when both exist for one batch."""
    if path.name.endswith(_METADATA_JSON_SUFFIX):
        return 0
    if path.name.endswith(_METADATA_YAML_SUFFIX):
        return 1
    return 2


def _dedupe_metadata_files_prefer_json(metadata_files: list[Path]) -> list[Path]:
    """Keep one file per (simulator_name, batch_index), preferring ``*.metadata.json``."""
    keyed: dict[tuple[str, int], Path] = {}
    unkeyed: list[Path] = []

    for path in metadata_files:
        name = path.name
        if name.endswith(_METADATA_JSON_SUFFIX):
            stem = name[: -len(_METADATA_JSON_SUFFIX)]
            rank = 0
        elif name.endswith(_METADATA_YAML_SUFFIX):
            stem = name[: -len(_METADATA_YAML_SUFFIX)]
            rank = 1
        else:
            unkeyed.append(path)
            continue

        if "-" not in stem:
            unkeyed.append(path)
            continue

        sim_name, batch_token = stem.rsplit("-", 1)
        try:
            batch_index = int(batch_token)
        except ValueError:
            unkeyed.append(path)
            continue

        key = (sim_name, batch_index)
        existing = keyed.get(key)
        if existing is None or rank < _metadata_file_format_rank(existing):
            keyed[key] = path

    ordered = sorted(keyed.items(), key=lambda item: (item[0][0], item[0][1]))
    deduped = [batch_path for _, batch_path in ordered]
    deduped.extend(sorted(unkeyed))
    return deduped


def create_batch_metadata(  # noqa: PLR0913
    simulator_name: str,
    batch_index: int,
    simulator_config: SimulatorConfig | OrchestrationConfig,
    globals_config: GlobalsConfig,
    simulator_metadata: dict[str, Any] | None = None,
    pre_batch_state: dict[str, Any] | None = None,
    source: str = "config",
    author: str | None = None,
    email: str | None = None,
    timestamp: datetime.datetime | None = None,
    *,
    config_payload: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
    replayable: bool = True,
    config_sha256: str | None = None,
    seed: int | None = None,
    segment_seeds: list[int] | None = None,
    population: dict[str, Any] | None = None,
    signal: dict[str, Any] | None = None,
    noise: dict[str, Any] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    host: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create metadata for a simulation batch.

    This metadata can be used to reproduce a specific batch and validate the
    versioned provenance schema written alongside it.

    Args:
        simulator_name: Name of the simulator
        batch_index: Index of the batch within the simulator
        simulator_config: Configuration for this simulator
        globals_config: Global configuration
        pre_batch_state: Optional state snapshot taken before batch generation
        source: Source of this batch: 'config', 'metadata_config', or 'metadata_state'
        author: Optional author name
        email: Optional author email
        timestamp: Optional timestamp for when the batch was created

    Returns:
        Metadata dictionary suitable for JSON serialization
    """
    if author is None:
        author = getpass.getuser()

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.UTC)

    base_config = config_payload or {
        "globals": globals_config.model_dump(by_alias=True, exclude_none=True),
        "simulators": {
            simulator_name: simulator_config.model_dump(by_alias=True, exclude_none=True),
        },
    }

    derived_config_sha256 = config_sha256
    if derived_config_sha256 is None:
        derived_config_sha256 = hashlib.sha256(
            json.dumps(base_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    dependency_versions = get_dependency_versions()

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gwmock_version": dependency_versions.get("gwmock"),
        "subpackage_versions": {
            "gwmock_signal": dependency_versions.get("gwmock-signal") or dependency_versions.get("gwmock_signal"),
            "gwmock_noise": dependency_versions.get("gwmock-noise") or dependency_versions.get("gwmock_noise"),
            "gwmock_pop": dependency_versions.get("gwmock-pop") or dependency_versions.get("gwmock_pop"),
        },
        "config": base_config,
        # Fully-resolved, replayable config: the input config with every runtime-
        # resolved external value (e.g. a pinned dataset revision) folded in.
        # Replay prefers this over "config" so an unpinned run still reproduces
        # the exact resources it used. None when nothing needed resolving.
        "resolved_config": resolved_config,
        # False when a declared external-mutable input could not be pinned to an
        # immutable id (e.g. an offline dataset with no cache); such a run is not
        # bit-reproducible from its metadata.
        "replayable": replayable,
        "config_sha256": derived_config_sha256,
        "seed": seed,
        "segment_seeds": segment_seeds or [],
        "population": population,
        "signal": signal,
        "noise": noise,
        "outputs": outputs or [],
        "host": host
        or {
            "platform": "unknown",
            "python": "unknown",
            "cpu": "unknown",
            "git_sha": None,
        },
        "environment": environment,
        # Compatibility fields kept while CLI reproduction still consumes them.
        "simulator_name": simulator_name,
        "batch_index": batch_index,
        "simulator_config": simulator_config.model_dump(by_alias=True, exclude_none=True),
        "globals_config": globals_config.model_dump(by_alias=True, exclude_none=True),
        "simulator_metadata": simulator_metadata or {},
        "author": author,
        "email": email,
        "timestamp": timestamp.isoformat(),
        "versions": dependency_versions,
        "source": source,
    }

    if pre_batch_state is not None:
        metadata["pre_batch_state"] = pre_batch_state

    return metadata


def create_plan_from_config(
    config: Config,
    checkpoint_dir: Path,
    author: str | None = None,
    email: str | None = None,
    config_file: Path | None = None,
) -> SimulationPlan:
    """Create a simulation plan from a configuration file.

    This is the standard workflow: start fresh with a config.
    Each simulator can generate multiple batches (e.g., segments or samples).

    Note: State (like RNG state) will be captured during simulation and stored in
    metadata for exact reproduction of individual batches.

    Args:
        config: Parsed Config object
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata
        config_file: Original YAML config path when known

    Returns:
        SimulationPlan with all batches defined across all simulators

    Example:
        >>> from gwmock.cli.utils.config import load_config
        >>> cfg = load_config(Path("config.yaml"))
        >>> plan = create_plan_from_config(cfg, Path("checkpoints"))
        >>> print(f"Total batches: {plan.total_batches}")
        >>> # Get all batches from a specific simulator
        >>> noise_batches = plan.get_batches_for_simulator("noise")
    """
    plan = SimulationPlan(
        source_config=config,
        checkpoint_directory=checkpoint_dir,
    )
    config_payload = config.model_dump(by_alias=True, exclude_none=True)
    config_sha256 = compute_file_hash(config_file) if config_file is not None else None

    global_batch_index = 0
    orchestration_config = config.orchestration

    global_sim_args = {k.replace("-", "_"): v for k, v in config.globals.simulator_arguments.items()}
    max_samples = resolve_max_samples(simulator_args={}, global_args=global_sim_args)
    for global_batch_index in range(max_samples):
        batch = SimulationBatch(
            simulator_name="orchestration",
            simulator_config=orchestration_config,
            globals_config=config.globals,
            batch_index=global_batch_index,
            source="config",
            author=author,
            email=email,
            config_payload=config_payload,
            config_sha256=config_sha256,
        )
        plan.add_batch(batch)

    logger.info("Created simulation plan from config: %d batches", plan.total_batches)
    return plan


def _replay_record(metadata_file: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the record a replay works from: the complete one when its private details are present.

    On the producer's machine the private details sit beside the record, and a replay uses the
    original absolute paths, exactly as from a combined record. Anywhere else the shareable record is
    replayed as it stands: its paths are relative to the working directory, which reads ``.``, so the
    replay writes under the current directory and expects an input that lay outside the original
    working directory -- a population file, say -- by its file name there.

    Args:
        metadata_file: The record's path.
        metadata: The record, as parsed.

    Returns:
        The record to replay.
    """
    complete = with_private_details(metadata_file, metadata)
    if complete is metadata and isinstance(metadata.get(PRIVATE_REFERENCE_KEY), dict):
        logger.info(
            "Replaying %s from its shareable record: paths resolve against the current directory.",
            metadata_file.name,
        )
    return complete


def create_plan_from_metadata_files(
    metadata_files: list[Path],
    checkpoint_dir: Path,
    author: str | None = None,
    email: str | None = None,
) -> SimulationPlan:
    """Create a simulation plan from individual metadata files.

    This allows exact reproduction of specific batches by restoring their pre-batch state.
    Metadata files should follow the naming pattern: SIMULATOR-BATCH_INDEX.metadata.json

    Args:
        metadata_files: List of paths to individual metadata files
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata

    Returns:
        SimulationPlan with batches reconstructed from metadata

    Raises:
        FileNotFoundError: If any metadata file doesn't exist
        ValueError: If metadata files are malformed

    Example:
        >>> files = [Path("signal-0.metadata.json"), Path("signal-1.metadata.json")]
        >>> plan = create_plan_from_metadata_files(files, Path("checkpoints"))
        >>> # Reproduces specific batches with exact state snapshots
    """
    plan = SimulationPlan(checkpoint_directory=checkpoint_dir)

    metadata_files = _dedupe_metadata_files_prefer_json(list(metadata_files))

    for metadata_file in sorted(metadata_files):
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        metadata = _replay_record(metadata_file, parse_batch_metadata(metadata_file))

        # Prefer the fully-resolved config so an unpinned run replays to the exact
        # external resources it used (e.g. a pinned dataset revision); fall back to
        # the raw input config when no resolved layer was recorded.
        resolved_config = metadata.get("resolved_config")
        metadata_config = resolved_config if isinstance(resolved_config, dict) else metadata.get("config")
        if metadata.get("replayable") is False:
            logger.warning(
                "Metadata %s is marked non-replayable (an external-mutable input could not be "
                "pinned to an immutable version); reproduction is not bit-for-bit.",
                metadata_file,
            )

        # Reconstruct configs from metadata
        try:
            if isinstance(metadata_config, dict) and metadata_config.get("orchestration") is not None:
                globals_config = GlobalsConfig(**metadata_config["globals"])
                raw_simulator_config = metadata_config["orchestration"]
            elif isinstance(metadata_config, dict) and metadata_config.get("simulators") is not None:
                globals_config = GlobalsConfig(**metadata_config["globals"])
                simulators_config = metadata_config["simulators"]
                if not isinstance(simulators_config, dict) or len(simulators_config) != 1:
                    raise ValueError("expected exactly one simulator config in metadata.config.simulators")
                raw_simulator_config = next(iter(simulators_config.values()))
            else:
                globals_config = GlobalsConfig(**metadata["globals_config"])
                raw_simulator_config = metadata["simulator_config"]
            if "class" in raw_simulator_config or "class_" in raw_simulator_config:
                simulator_config = SimulatorConfig(**raw_simulator_config)
            elif raw_simulator_config.keys() & {"population", "signal", "noise"}:
                simulator_config = OrchestrationConfig(**raw_simulator_config)
            else:
                raise ValueError("unknown simulator_config shape")
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid metadata in {metadata_file}: missing or malformed config: {e}") from e

        simulator_name = metadata.get("simulator_name")
        batch_index = metadata.get("batch_index")
        pre_batch_state = metadata.get("pre_batch_state")

        if not simulator_name or batch_index is None:
            raise ValueError(f"Invalid metadata in {metadata_file}: missing simulator_name or batch_index")

        batch = SimulationBatch(
            simulator_name=simulator_name,
            simulator_config=simulator_config,
            globals_config=globals_config,
            batch_index=batch_index,
            metadata_file=metadata_file,
            batch_metadata=metadata,
            pre_batch_state=pre_batch_state,
            source="metadata_state" if pre_batch_state else "metadata_config",
            author=author,
            email=email,
            config_payload=metadata_config if isinstance(metadata_config, dict) else None,
            config_sha256=metadata.get("config_sha256"),
        )
        plan.add_batch(batch)

    _warn_version_mismatches(plan.batches, get_dependency_versions())
    _warn_environment_drift(plan.batches)

    logger.info(
        "Created simulation plan from %d metadata files",
        len(metadata_files),
    )
    return plan


def _warn_environment_drift(batches: list[SimulationBatch]) -> None:
    """Warn once if the current environment differs from the recorded freeze.

    Points the user at ``--isolate`` for exact-dependency reproduction. Uses the
    first batch that recorded a full environment; older metadata without one is
    skipped (the coarser gwmock-family check above still applies).
    """
    from gwmock.cli.utils.environment import capture_environment, diff_environment  # noqa: PLC0415

    for batch in batches:
        recorded = (batch.batch_metadata or {}).get("environment")
        if not recorded:
            continue
        mismatches = diff_environment(recorded, capture_environment())
        if mismatches:
            examples = ", ".join(name for name, _, _ in mismatches[:3])
            logger.warning(
                "%d package(s) differ from the environment recorded in the metadata (e.g. %s). "
                "Re-run with 'gwmock simulate --isolate' to reproduce against the exact recorded dependencies.",
                len(mismatches),
                examples,
            )
        return


def _warn_version_mismatches(
    batches: list[SimulationBatch],
    installed: dict[str, str | None],
) -> None:
    """Emit a warning for each gwmock-family package whose stored version differs from installed."""
    field_to_pkg: list[tuple[str | None, str, str]] = [
        # (top-level key or None, subpackage_versions subkey or "", installed pkg key)
        ("gwmock_version", "", "gwmock"),
        (None, "gwmock_noise", "gwmock-noise"),
        (None, "gwmock_pop", "gwmock-pop"),
        (None, "gwmock_signal", "gwmock-signal"),
    ]

    seen: dict[str, set[str]] = {}
    for batch in batches:
        metadata = batch.batch_metadata
        if not metadata:
            continue
        for top_key, sub_key, pkg_key in field_to_pkg:
            if top_key:
                stored = metadata.get(top_key)
            else:
                subpkg = metadata.get("subpackage_versions") or {}
                stored = subpkg.get(sub_key) if isinstance(subpkg, dict) else None
            if stored:
                seen.setdefault(pkg_key, set()).add(stored)

    for pkg_key, stored_versions in seen.items():
        installed_ver = installed.get(pkg_key)
        for stored_ver in stored_versions:
            if stored_ver != installed_ver:
                logger.warning(
                    "Version mismatch for %s: metadata records %s but %s is installed."
                    " Results may not be bit-per-bit reproducible.",
                    pkg_key,
                    stored_ver,
                    installed_ver or "unknown",
                )


def create_plan_from_metadata(
    metadata_dir: Path,
    checkpoint_dir: Path,
    author: str | None = None,
    email: str | None = None,
) -> SimulationPlan:
    """Create a simulation plan from a directory of metadata files.

    This allows exact reproduction of specific batches by restoring their pre-batch state.
    Metadata files should follow the naming pattern: SIMULATOR-BATCH_INDEX.metadata.json

    Args:
        metadata_dir: Directory containing metadata files
        checkpoint_dir: Directory for checkpoints
        author: Optional author name for metadata
        email: Optional author email for metadata

    Returns:
        SimulationPlan with batches reconstructed from metadata

    Raises:
        FileNotFoundError: If metadata_dir doesn't exist
        ValueError: If metadata files are malformed

    Example:
        >>> plan = create_plan_from_metadata(Path("metadata"), Path("checkpoints"))
        >>> # Reproduces batches with exact state snapshots
    """
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    # Find all metadata files in directory
    metadata_files = list(metadata_dir.glob("*.metadata.json")) + list(metadata_dir.glob("*.metadata.yaml"))

    plan = create_plan_from_metadata_files(metadata_files, checkpoint_dir, author=author, email=email)
    logger.info(
        "Created simulation plan from metadata directory: %d batches from %s",
        plan.total_batches,
        metadata_dir,
    )
    return plan


def merge_plans(*plans: SimulationPlan) -> SimulationPlan:
    """Merge multiple simulation plans into one.

    Useful for combining config-based and metadata-based workflows.

    Args:
        *plans: SimulationPlan objects to merge

    Returns:
        Merged SimulationPlan

    Example:
        >>> plan_config = create_plan_from_config(cfg, Path("checkpoints"))
        >>> plan_metadata = create_plan_from_metadata_files([meta1, meta2], Path("checkpoints"))
        >>> combined_plan = merge_plans(plan_config, plan_metadata)
    """
    merged = SimulationPlan()
    batch_index = 0

    for plan in plans:
        for batch in plan.batches:
            # Reassign batch indices to maintain order
            batch.batch_index = batch_index
            merged.add_batch(batch)
            batch_index += 1

    merged.checkpoint_directory = plans[0].checkpoint_directory if plans else Path("checkpoints")
    logger.info("Merged %d plans into one: %d total batches", len(plans), merged.total_batches)
    return merged
