# ruff: noqa PLC0415
"""
A sub-command to handle data generation using simulation plans.
"""

from __future__ import annotations

from typing import Annotated

import typer


def _resolve_recorded_environment(config_file_names_list: list[str]) -> dict | None:
    """Return the single environment freeze recorded across the given metadata.

    Globs JSON and YAML metadata for directories and parses either form. Returns
    ``None`` when no metadata records an environment. Raises ``ValueError`` when
    the metadata span more than one distinct environment, since a single
    isolated environment cannot reproduce them all.
    """
    import yaml  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from gwmock.cli.utils.environment import environment_key  # pylint: disable=import-outside-toplevel

    candidates: list[Path] = []
    for name in config_file_names_list:
        path = Path(name)
        if path.is_dir():
            candidates.extend(sorted(path.glob("*.metadata.json")) + sorted(path.glob("*.metadata.yaml")))
        else:
            candidates.append(path)

    environments: dict[str, dict] = {}
    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle)  # also parses JSON
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(metadata, dict):
            continue
        environment = metadata.get("environment")
        if environment:
            environments[environment_key(environment)] = environment

    if len(environments) > 1:
        raise ValueError(
            "The provided metadata span multiple recorded environments; "
            "reproduce each environment's metadata separately with --isolate."
        )
    return next(iter(environments.values()), None)


def _maybe_isolate(
    config_file_names_list: list[str],
    *,
    isolate: bool,
    output_dir: str | None,
    overwrite: bool,
    ignore_checkpoint: bool = False,
    author: str | None,
    email: str | None,
    dry_run: bool = False,
) -> None:
    """Recreate the recorded environment and re-run inside it when it differs.

    No-op unless ``--isolate`` is set and we are not already the re-executed
    child. Runs in place (with a note) when no environment was recorded or the
    current one already matches. Otherwise builds the isolated environment,
    re-runs the reproduction there, and exits with the child's status.
    """
    import logging  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel

    import typer  # pylint: disable=import-outside-toplevel

    from gwmock.cli.utils.environment import (  # pylint: disable=import-outside-toplevel
        ISOLATION_ENV_VAR,
        capture_environment,
        environments_match,
        reproduce_in_isolated_environment,
    )

    logger = logging.getLogger("gwmock")
    if not isolate or os.environ.get(ISOLATION_ENV_VAR):
        return

    try:
        recorded = _resolve_recorded_environment(config_file_names_list)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if recorded is None:
        logger.warning(
            "--isolate: the metadata records no environment freeze (older run); running in the current environment."
        )
        return
    if environments_match(recorded, capture_environment()):
        logger.info("--isolate: current environment already matches the recorded one; running in place.")
        return

    child_args = ["simulate", *config_file_names_list]
    if output_dir:
        child_args += ["--output-dir", output_dir]
    if overwrite:
        child_args.append("--overwrite")
    if ignore_checkpoint:
        child_args.append("--ignore-checkpoint")
    if author:
        child_args += ["--author", author]
    if email:
        child_args += ["--email", email]
    if dry_run:
        child_args.append("--dry-run")

    logger.info("--isolate: recreating the recorded environment and re-running the reproduction inside it.")
    exit_code = reproduce_in_isolated_environment(recorded, child_args)
    raise typer.Exit(exit_code)


def _simulate_impl(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    config_file_names: str | list[str],
    # Keyword-only: these are seven options of which five are bools, so a positional call binds them
    # by accident rather than by intent. `ignore_checkpoint` sits between two of those bools and
    # skips the checkpoint when truthy, which is the failure this change exists to prevent. Every
    # caller already passes by keyword, so nothing is broken by saying so.
    *,
    output_dir: str | None = None,
    metadata_dir: str | None = None,
    overwrite: bool = False,
    ignore_checkpoint: bool = False,
    metadata: bool = True,
    author: str | None = None,
    email: str | None = None,
    dry_run: bool = False,
    isolate: bool = False,
) -> None:
    """Internal implementation of simulate command that accepts both str and list[str].

    Unified approach: All simulation goes through the same plan execution system.
    The difference is only in how we create the initial plan:
    - Config file → SimulationPlan with pre_batch_state=None (fresh simulation)
    - Metadata files → SimulationPlan with pre_batch_state=<dict> (reproduce)

    Both cases end up calling execute_plan with batches that may or may not have
    state snapshots. The execute_plan function handles both transparently.

    Args:
        config_file_names: Path to YAML config file OR one or more metadata files
        output_dir: Output directory override
        metadata_dir: Metadata directory override (config mode only)
        overwrite: Whether to overwrite existing files
        metadata: Whether to save metadata files (config mode only)
        author: Author name for metadata
        email: Author email for metadata
        dry_run: If True, only validate the plan without executing.

    Returns:
        None
    """
    import logging  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from gwmock.cli.simulate_utils import (  # pylint: disable=import-outside-toplevel
        execute_plan,
        validate_plan,
    )
    from gwmock.cli.utils.config import (  # pylint: disable=import-outside-toplevel
        load_config,
        require_population_path_found,
    )
    from gwmock.cli.utils.simulation_plan import (  # pylint: disable=import-outside-toplevel
        create_plan_from_config,
        create_plan_from_metadata_files,
    )
    from gwmock.monitor.resource import ResourceMonitor  # pylint: disable=import-outside-toplevel

    logger = logging.getLogger("gwmock")
    logger.setLevel(logging.DEBUG)

    checkpoint_dir = Path(".gwmock_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ===== Normalize input: accept both string and list =====
        config_file_names_list = [config_file_names] if isinstance(config_file_names, str) else config_file_names

        # ===== Auto-detect mode: Config file vs Metadata files =====
        if len(config_file_names_list) == 1:
            # Single argument: could be YAML config or metadata file
            single_path = Path(config_file_names_list[0])
            is_metadata = (
                single_path.name.endswith(".metadata.json")
                or single_path.name.endswith(".metadata.yaml")
                or (single_path.is_file() and "metadata" in single_path.name)
                or single_path.is_dir()  # Directory assumed to be metadata dir
            )
        else:
            # Multiple arguments: must be metadata files
            is_metadata = True

        # ===== Create plan (unified approach: both modes create same data structure) =====
        if is_metadata:
            logger.info("Reproduction mode: %d metadata file(s)", len(config_file_names_list))

            # Optionally recreate the recorded environment and re-run inside it
            # before doing any work in the (possibly mismatched) current one.
            _maybe_isolate(
                config_file_names_list,
                isolate=isolate,
                output_dir=output_dir,
                overwrite=overwrite,
                ignore_checkpoint=ignore_checkpoint,
                author=author,
                email=email,
                dry_run=dry_run,
            )

            metadata_paths = [Path(f) for f in config_file_names_list]

            # If single directory, load all metadata files from it
            if len(metadata_paths) == 1 and metadata_paths[0].is_dir():
                metadata_dir_path = metadata_paths[0]
                metadata_files = list(metadata_dir_path.glob("*.metadata.json")) + list(
                    metadata_dir_path.glob("*.metadata.yaml")
                )
                if not metadata_files:
                    raise ValueError(f"No metadata files found in directory: {metadata_dir_path}")
                logger.info("Found %d metadata files in directory: %s", len(metadata_files), metadata_dir_path)
                plan = create_plan_from_metadata_files(metadata_files, checkpoint_dir, author=author, email=email)
            else:
                # Individual metadata files
                plan = create_plan_from_metadata_files(metadata_paths, checkpoint_dir, author=author, email=email)

            logger.info(
                "Created reproduction plan from %d metadata file(s) with %d batches",
                len(config_file_names_list),
                plan.total_batches,
            )
        else:
            logger.info("Config mode: %s", config_file_names_list[0])
            config_path = Path(config_file_names_list[0])
            config = load_config(file_name=config_path)
            logger.debug("Configuration loaded successfully from %s", config_file_names_list[0])
            require_population_path_found(config, config_path)

            plan = create_plan_from_config(config, checkpoint_dir, author=author, email=email, config_file=config_path)
            logger.info("Created simulation plan with %d batches", plan.total_batches)

        # ===== Determine output directories (same logic for both modes) =====
        # Get reference config from first batch
        if not plan.batches:
            raise ValueError("No batches found in simulation plan")

        first_batch = plan.batches[0]
        globals_cfg = first_batch.globals_config
        working_dir = Path(globals_cfg.working_directory or ".")  # pylint: disable=no-member
        output_dir_config = globals_cfg.output_directory or "output"  # pylint: disable=no-member

        # Handle absolute vs relative paths
        config_output_dir = (
            Path(output_dir_config) if Path(output_dir_config).is_absolute() else working_dir / output_dir_config
        )
        final_output_dir = Path(output_dir) if output_dir else config_output_dir

        # Metadata directory only used in config mode (fresh simulation)
        final_metadata_dir = None
        if not is_metadata and metadata:
            metadata_dir_config = globals_cfg.metadata_directory or "metadata"  # pylint: disable=no-member
            config_metadata_dir = (
                Path(metadata_dir_config)
                if Path(metadata_dir_config).is_absolute()
                else working_dir / metadata_dir_config
            )
            final_metadata_dir = Path(metadata_dir) if metadata_dir else config_metadata_dir

        logger.debug("Output directory: %s", final_output_dir)
        if final_metadata_dir:
            logger.debug("Metadata directory: %s", final_metadata_dir)

        # ===== Validate and execute plan =====
        validate_plan(plan)
        logger.info("Simulation plan validation passed")

        if dry_run:
            logger.info("Dry run mode: Simulation plan validated but not executed")
            return

        resource_monitor = ResourceMonitor()

        with resource_monitor.measure():
            execute_plan(
                plan=plan,
                output_directory=final_output_dir,
                metadata_directory=final_metadata_dir or Path("metadata"),
                overwrite=overwrite,
                ignore_checkpoint=ignore_checkpoint,
                max_retries=3,
            )

        resource_monitor.log_summary(logger)

        # Write the resource usage summary to a json file.
        resource_file = working_dir / "resource_usage_summary.json"
        resource_monitor.save_metrics(resource_file, overwrite=True)

        logger.info("Simulation completed successfully. Output written to %s", final_output_dir)

    except Exception as e:
        logger.error("Simulation failed: %s", str(e), exc_info=True)
        raise typer.Exit(code=1) from e


def simulate_command(
    config_file_names: Annotated[
        list[str],
        typer.Argument(help="Configuration file (YAML) or metadata files (can specify multiple metadata files)."),
    ],
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", help="Output directory (overrides config/metadata defaults).")
    ] = None,
    metadata_dir: Annotated[
        str | None,
        typer.Option(
            "--metadata-dir", help="Metadata directory (overrides config/metadata defaults, only for config mode)."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files.")] = False,
    ignore_checkpoint: Annotated[
        bool,
        typer.Option(
            "--ignore-checkpoint",
            help=(
                "Start fresh, ignoring any checkpoint in the directory. Use when a resume is refused "
                "because the checkpoint belongs to a different run and you know it is stale."
            ),
        ),
    ] = False,
    metadata: Annotated[bool, typer.Option("--metadata", help="Generate metadata files (only in config mode).")] = True,
    author: Annotated[str | None, typer.Option("--author", help="Author name for the simulation.")] = None,
    email: Annotated[str | None, typer.Option("--email", help="Author email for the simulation.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate the plan without executing.")] = False,
    isolate: Annotated[
        bool,
        typer.Option(
            "--isolate",
            help="Reproduction mode: recreate the metadata's recorded environment in an "
            "isolated (uv) virtualenv and re-run inside it for exact-dependency reproducibility.",
        ),
    ] = False,
) -> None:
    """Generate gravitational wave simulation data using specified simulators.

    This command can run in two modes, automatically detected from the input:

    1. **Config Mode**: Provide a single YAML configuration file to create a simulation plan.
       Executes all simulators with state tracking for reproducibility.
       Example:

         `gwmock simulate config.yaml`

    2. **Reproduction Mode**: Provide one or more metadata files to reproduce specific batches.
       Each metadata file contains the exact configuration and pre-batch state needed for
       exact reproducibility. Users can distribute individual metadata files, and anyone
       can reproduce those specific batches independently.
       Example:

         `gwmock simulate signal-0.metadata.yaml signal-1.metadata.yaml`

    **Path Overrides:**

    - Use `--output-dir` to specify where output files should be saved
    - Use `--metadata-dir` to specify where metadata should be saved (config mode only)
    - These override paths from the configuration or metadata files

    Args:
        config_file_names: Path to YAML config file OR one or more metadata files
        output_dir: Output directory override
        metadata_dir: Metadata directory override (config mode only)
        overwrite: Whether to overwrite existing files
        metadata: Whether to save metadata files (config mode only)

    Returns:
        None
    """
    _simulate_impl(
        config_file_names=config_file_names,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        overwrite=overwrite,
        ignore_checkpoint=ignore_checkpoint,
        metadata=metadata,
        author=author,
        email=email,
        dry_run=dry_run,
        isolate=isolate,
    )
