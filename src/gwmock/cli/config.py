# ruff: noqa PLC0415

"""
A tool to generate and manage default configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def _redirect_example_data_file(config, *, src_path: Path, dst_path: Path, overwrite: bool) -> tuple[Path, Path] | None:
    """Point a copied example at its own copy of the data file it references.

    An example's population catalogue ships beside it in the package and is referenced by a path
    relative to the example's own directory. That path means nothing once the configuration is
    copied elsewhere, so the path is rewritten to a copy next to the new configuration -- relative
    to the working directory, like every path in a configuration -- and the caller copies the
    catalogue there once the configuration is written. A run from that directory then needs no
    network access and reads exactly the catalogue that shipped.

    Only a relative path that resolves to a file inside the packaged examples is copied. A URL, an
    absolute path, or a path to anything else is left exactly as written.

    Args:
        config: The loaded example configuration, updated in place.
        src_path: The example's configuration file.
        dst_path: Where the copy is being written.
        overwrite: Whether an existing, different file of the same name may be replaced.

    Returns:
        ``(source, destination)`` of the data file to copy, or ``None`` when there is none.

    Raises:
        FileExistsError: If a different file already exists where the data file would be copied
            and *overwrite* is not set.
    """
    import filecmp

    from gwmock.cli.utils.config import get_examples_dir

    orchestration = getattr(config, "orchestration", None)
    population = getattr(orchestration, "population", None)
    arguments = getattr(population, "arguments", None)
    if not isinstance(arguments, dict):
        return None
    path_value = arguments.get("path")
    if not isinstance(path_value, str) or "://" in path_value or Path(path_value).is_absolute():
        return None
    source = (src_path.parent / path_value).resolve()
    if not source.is_file() or get_examples_dir().resolve() not in source.parents:
        return None

    destination = dst_path.parent / source.name
    if destination.exists() and not filecmp.cmp(source, destination, shallow=False) and not overwrite:
        raise FileExistsError(f"File already exists: {destination}. Use --overwrite to overwrite.")
    arguments["path"] = destination.as_posix()
    return source, destination


def _config_command_impl(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    output: Path,
    get: Path | None,
    init: Path | None,
    list_: bool,
    overwrite: bool,
):
    """Internal implementation of the config command

    Args:
        output: File, or existing directory, that --get writes to.
        get: Label of the example configuration to copy.
        init: If provided, the path of the default configuration file to generate.
        list_: If True, prints the names of available example config files.
        overwrite: If True, overwrites existing files without raising an error.

    """
    import logging

    from gwmock.cli.utils.config import (
        get_examples_dir,
        load_config,
        save_config,
    )

    def copy_config_file(src_path: Path, dst_path: Path, overwrite: bool) -> None:
        """Copy a configuration file from src_path to dst_path.

        Args:
            src_path: Source file path.
            dst_path: Destination file path.
            overwrite: Whether to overwrite existing files.
        """
        # Load the configuration file
        config = load_config(src_path)

        # Create parent directories if they don't exist
        dst_path.resolve().parent.mkdir(parents=True, exist_ok=True)

        data_file = _redirect_example_data_file(config, src_path=src_path, dst_path=dst_path, overwrite=overwrite)

        # Save the configuration file to the destination
        try:
            save_config(file_name=dst_path, config=config, overwrite=overwrite, backup=False)
        except FileExistsError as e:
            raise FileExistsError(f"File already exists: {dst_path}. Use --overwrite to overwrite.") from e
        if data_file is not None:
            import shutil

            shutil.copyfile(*data_file)
            logger.info("Copied the example's data file %s to %s", data_file[0].name, data_file[1])

    logger = logging.getLogger("gwmock")

    examples_dir = get_examples_dir()

    # Generate default config file
    if init is not None:
        copy_config_file(src_path=examples_dir / "default_config" / "config.yaml", dst_path=init, overwrite=overwrite)
        logger.info("Generated default configuration file: %s", init)
        return

    if list_:
        available_labels = set()
        for config_file in examples_dir.rglob("config.yaml"):
            # Show the full relative path from examples_dir, excluding the filename
            relative_path = config_file.relative_to(examples_dir)
            available_labels.add(relative_path.parent)
        logger.info("[bold cyan]Available example configuration labels:[/bold cyan]")
        for label in sorted(available_labels):
            logger.info("  - %s", label)

        logger.info("To copy example configuration files, use the [bold green]--get[/bold green] option with a label.")
        logger.info(
            "For example: %s",
            f"[bold green]gwmock config --get {sorted(available_labels)[0]} --output config.yaml[/bold green]",
        )
        return

    # pylint: disable=duplicate-code
    if get is not None:
        src_path = examples_dir / get / "config.yaml"
        if not src_path.exists():
            logger.error("Example configuration '%s' does not exist in examples directory.", get)
            raise typer.Exit(1)
        dst_path = output / "config.yaml" if output.is_dir() else output
        copy_config_file(src_path=src_path, dst_path=dst_path, overwrite=overwrite)
        logger.info("Copied example configuration file: %s to %s", get / "config.yaml", dst_path)
        return

    logger.error(
        "No action specified. Please provide one of [bold green]--init[/bold green], "
        "[bold green]--list[/bold green], [bold green]--get[/bold green], "
        "or [bold green]--interactive[/bold green]."
    )
    logger.error("Use [bold green]--help[/bold green] for more information.")
    raise typer.Exit(1)


def config_command(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="File, or existing directory, that --get writes the copied configuration to.",
        ),
    ] = Path("config.yaml"),
    get: Annotated[Path | None, typer.Option("--get", help="Label of the configuration file to copy")] = None,
    init: Annotated[Path | None, typer.Option("--init", help="Generate a default configuration file")] = None,
    list_: Annotated[bool, typer.Option("--list", help="List the available example configuration files")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Open interactive configuration editor"),
    ] = False,
    load: Annotated[
        Path | None,
        typer.Option("--load", help="Load an existing config file into the interactive editor"),
    ] = None,
) -> None:
    """
    Manage default and example configuration files.

    This command provides utilities to list the available example configurations,
    generate a default configuration, or copy one example configuration.

    Exactly **one** of ``--init``, ``--list``, ``--get`` or ``--interactive`` must be provided.

    **Options:**

    - `--init <PATH>`
          Generate a default configuration file at PATH.

    - `--list`
          Print the labels of all available example configurations and exit.

    - `--get <LABEL>`
          Copy one example configuration. The label must be one ``--list`` prints,
            such as ``noise/uncorrelated_gaussian/quick_start``.

    - `--output <PATH>`
          Where ``--get`` writes the copy: a file name, or an existing directory to
            write ``config.yaml`` into. Parent directories are created.

    - `--overwrite`
          Overwrite an existing file. If not set, the command fails when the target
            file already exists.

    - `--interactive` / `-i`
          Open an interactive TUI editor to build a configuration file
            step-by-step with live previews and dynamic option discovery.

    - `--load <PATH>`
          Load an existing configuration file into the interactive editor.
            Only used with ``--interactive``.

    Examples:
        List available example configurations:

            gwmock config --list

        Generate a default configuration file:

            gwmock config --init config.yaml

        Copy the quick-start example to config.yaml:

            gwmock config --get noise/uncorrelated_gaussian/quick_start --output config.yaml

        Copy an example and overwrite an existing file:

            gwmock config --get noise/uncorrelated_gaussian/quick_start --output config.yaml --overwrite

        Open the interactive configuration editor:

            gwmock config --interactive

        Open the interactive editor with an existing config pre-loaded:

            gwmock config --interactive --load config.yaml

    Args:
        output: File, or existing directory, that --get writes to.
        get: Label of the example configuration to copy.
        init: If provided, the path of the default configuration file to generate.
        list_: If True, prints the labels of the available example configurations.
        overwrite: If True, overwrites existing files without raising an error.
        interactive: If True, opens the interactive configuration editor.
        load: Configuration file to load into the interactive editor.

    Raises:
        typer.Exit: If no action is given, or the label given to --get does not exist.
        FileExistsError: If the target file already exists and --overwrite is not given.
    """
    if interactive:
        from gwmock.cli.utils.config_editor import ConfigEditorApp

        app = ConfigEditorApp(load_path=load)
        app.run()
        return

    _config_command_impl(
        output=output,
        get=get,
        init=init,
        list_=list_,
        overwrite=overwrite,
    )
