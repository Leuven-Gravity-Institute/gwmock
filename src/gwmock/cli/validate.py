# ruff: noqa PLC0415
"""
Validation functions for CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def validate_command(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="Files/directories to validate. Can be output files, metadata files, or directories containing either."
        ),
    ],
    metadata_paths: Annotated[
        list[str] | None, typer.Option("--metadata-paths", help="Additional metadata files or directories")
    ] = None,
    pattern: Annotated[
        str | None, typer.Option("--pattern", help="File pattern to match (e.g., '*noise*' for subset validation)")
    ] = None,
    metadata_pattern: Annotated[
        str | None, typer.Option("--metadata-pattern", help="Metadata file pattern to match")
    ] = "*metadata.*",
) -> None:
    """Validate output files against metadata hashes and other checks.

    This command verifies the integrity of generated simulation files by:

    1. Loading metadata files and extracting expected file hashes
    2. Recomputing hashes for actual output files
    3. Comparing hashes and reporting mismatches
    4. Future: Add sampling rate and continuity checks

    The command automatically detects whether provided paths are:

    - Output files (.gwf, etc.) - will find corresponding metadata
    - Metadata files (.metadata.json / legacy .metadata.yaml) - will validate their output files
    - Directories - will scan for both types of files

    Examples:
        Validate specific output files (finds metadata automatically):

        - `gwmock validate H1-NOISE-123.gwf L1-SIGNAL-456.gwf`

        Validate specific metadata files:

        - `gwmock validate --metadata signal-0.metadata.yaml noise-0.metadata.yaml`

        Validate all files in a directory:
        gwmock validate /path/to/output/

        Validate subset using pattern:
        gwmock validate /path/to/output/ --pattern "*noise*"

        Mix files and directories:
        gwmock validate H1-NOISE-123.gwf /path/to/more/files/

        Override output directory:
        gwmock validate metadata/ --output-dir /custom/output/

    Args:
        paths: Files/directories to validate. Can be output files, metadata files, or directories containing either.
        metadata_paths: Additional metadata files or directories.
        pattern: File pattern to match (e.g., '*noise*' for subset validation).
        metadata_pattern: Metadata file pattern to match.
    """
    import fnmatch
    import logging

    from rich.console import Console
    from rich.table import Table

    from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash
    from gwmock.cli.utils.metadata import load_metadata_with_external_state, with_private_details

    logger = logging.getLogger("gwmock")

    console = Console()

    logger.info("Validating simulation files...")

    # Separate into metadata files and potential output files
    metadata_files: list[Path] = []
    output_files: list[Path] = []
    output_directories: list[Path] = []
    metadata_directories: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            output_directories.append(path)
        elif path.is_file():
            # Assume it's an output file
            output_files.append(path)
        else:
            console.print(f"[red]Error:[/red] Path not found: {path}")

    for path_str in metadata_paths or []:
        path = Path(path_str)
        if path.is_dir():
            metadata_directories.append(path)
        elif path.is_file():
            if path.name.endswith(".metadata.json") or path.name.endswith(".metadata.yaml"):
                metadata_files.append(path)
            else:
                console.print(f"[yellow]Warning:[/yellow] Ignoring non-metadata file: {path}")
        else:
            console.print(f"[red]Error:[/red] Metadata path not found: {path}")

    # Scan directories for files
    for directory in output_directories:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                output_files.append(file_path)

    for directory in metadata_directories:
        for file_path in directory.rglob("*"):
            if "metadata" in file_path.name and file_path.is_file():
                metadata_files.append(file_path)

    # One key per file, however it was reached: a directory scan and a record's own location can
    # name the same file by a relative and an absolute path, and it must be checked -- and counted --
    # once.
    output_files = [_same_file_key(path) for path in output_files]

    # Apply pattern filtering if specified
    if pattern:
        output_files = [f for f in output_files if fnmatch.fnmatch(f.name, pattern)]

    if metadata_pattern:
        metadata_files = [f for f in metadata_files if fnmatch.fnmatch(f.name, metadata_pattern)]

    # Build validation plan: output_file -> metadata_file
    output_to_metadata = {}

    # First, extract output files from provided metadata files
    for metadata_file in metadata_files:
        try:
            metadata = load_metadata_with_external_state(metadata_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error loading metadata %s: %s", metadata_file, e)
            continue
        # Only to *locate* the outputs: the private details, when beside the record, give the
        # original absolute working directory. The hashes compared below are the same either way.
        metadata = with_private_details(metadata_file, metadata)

        outputs = metadata.get("outputs", [])
        # Use the full relative ``path`` recorded in ``outputs`` (e.g.
        # ``output/signal/foo.gwf``) rather than just its basename. Joining only
        # the basename to ``working-directory`` dropped the ``output/<kind>/``
        # sub-directory, so the reconstructed path never existed on disk and the
        # file was reported as "File not found" *in addition to* the real PASS
        # found by the directory scan -- double-counting every output.
        output_files_in_meta = [output["path"] for output in outputs] or metadata.get("output_files", [])
        config = metadata.get("config", {})
        globals_config = config.get("globals", metadata.get("globals_config", {}))
        if outputs:
            output_dir = Path(globals_config.get("working-directory") or globals_config.get("working_directory") or ".")
        else:
            output_dir = Path(globals_config.get("output_directory", "."))

        for filename in output_files_in_meta:
            # Apply pattern filtering
            if pattern and not fnmatch.fnmatch(filename, pattern):
                continue

            output_file = _same_file_key(_locate_output(output_dir / filename, filename, metadata_file, globals_config))
            if output_file not in output_to_metadata:
                output_to_metadata[output_file] = metadata_file

    # Then, for explicitly provided output files, find their metadata
    for output_file in output_files:
        if output_file not in output_to_metadata:
            potential_metadata = None

            # First, check if any already-identified metadata files contain this output file
            for metadata_file in metadata_files:
                try:
                    meta_data = load_metadata_with_external_state(metadata_file)
                    output_names = [Path(output["path"]).name for output in meta_data.get("outputs", [])]
                    if output_file.name in output_names or output_file.name in meta_data.get("output_files", []):
                        potential_metadata = metadata_file
                        break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error reading metadata file %s: %s", metadata_file, e)
                    continue

            # If not found in existing metadata, search in directories
            if not potential_metadata:
                # Look in the same directory or parent directories
                search_dir = output_file.parent
                metadata_dir = search_dir / "metadata"
                if metadata_dir.exists():
                    # Look for metadata files that might match
                    for meta_file in list(metadata_dir.glob("*.metadata.json")) + list(
                        metadata_dir.glob("*.metadata.yaml")
                    ):
                        try:
                            meta_data = load_metadata_with_external_state(meta_file)
                            output_names = [Path(output["path"]).name for output in meta_data.get("outputs", [])]
                            if output_file.name in output_names or output_file.name in meta_data.get(
                                "output_files", []
                            ):
                                potential_metadata = meta_file
                                break
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.error("Error reading metadata file %s: %s", meta_file, e)
                            continue

            if potential_metadata:
                output_to_metadata[output_file] = potential_metadata
            else:
                logger.warning("No metadata found for output file %s", output_file)  # Combine all metadata files
    all_metadata_files = list({v for v in output_to_metadata.values() if v is not None})

    if not all_metadata_files and not metadata_files:
        logger.error("Error: No metadata files found")
        raise typer.Exit(1)

    # Create results table. The authoritative verdict is the *content* hash
    # (decoded data); the raw-file byte hash is a secondary integrity signal.
    # A file whose content matches but whose container bytes differ is reported
    # as "same data, repackaged" -- this is the normal case for re-generated GWF
    # frames, which embed a write-time timestamp and the frame-library version.
    table = Table(title="Validation Results")
    table.add_column("Metadata File", style="cyan")
    table.add_column("Output File", style="magenta")
    table.add_column("Content", style="green")
    table.add_column("Bytes", style="green")
    table.add_column("Status", style="yellow")

    total_files = len(output_to_metadata)
    failed_files = 0
    repackaged_files = 0

    def _expected(metadata: dict, name: str, file_key: str, record_key: str) -> str | None:
        """Look up an expected hash by filename, from the top-level map or outputs[]."""
        top = metadata.get(file_key, {})
        if name in top:
            return top[name]
        records = {Path(o["path"]).name: o for o in metadata.get("outputs", [])}
        return records.get(name, {}).get(record_key)

    # Order output for consistent reporting
    for output_file in sorted(output_to_metadata.keys()):
        metadata_file = output_to_metadata[output_file]
        name = output_file.name
        if metadata_file is None:
            table.add_row("N/A", name, "N/A", "N/A", "[red]No metadata found[/red]")
            failed_files += 1
            continue
        try:
            metadata = load_metadata_with_external_state(metadata_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error loading metadata %s: %s", metadata_file, e)
            table.add_row(str(metadata_file.name), name, "N/A", "N/A", "[red]Error loading metadata[/red]")
            continue

        if not output_file.exists():
            table.add_row(str(metadata_file.name), name, "N/A", "N/A", "[red]File not found[/red]")
            failed_files += 1
            continue

        expected_content = _expected(metadata, name, "content_hashes", "content_sha256")
        expected_bytes = _expected(metadata, name, "file_hashes", "sha256")

        try:
            actual_bytes = compute_file_hash(output_file)
            bytes_ok = None if not expected_bytes else (actual_bytes == expected_bytes)
            bytes_cell = "[green]✓[/green]" if bytes_ok else ("[yellow]≠[/yellow]" if bytes_ok is False else "—")

            if expected_content:
                actual_content = compute_content_hash(output_file)
                if actual_content is None:
                    # Couldn't decode -- fall back to the byte hash as the verdict.
                    if bytes_ok is None:
                        table.add_row(
                            str(metadata_file.name),
                            name,
                            "[yellow]?[/yellow]",
                            bytes_cell,
                            "[yellow]Could not read content[/yellow]",
                        )
                        failed_files += 1
                    elif bytes_ok:
                        table.add_row(str(metadata_file.name), name, "—", bytes_cell, "[green]PASS[/green]")
                    else:
                        table.add_row(str(metadata_file.name), name, "—", bytes_cell, "[red]BYTES MISMATCH[/red]")
                        failed_files += 1
                elif actual_content == expected_content:
                    if bytes_ok is False:
                        table.add_row(
                            str(metadata_file.name),
                            name,
                            "[green]✓[/green]",
                            bytes_cell,
                            "[green]PASS[/green] [dim](same data, repackaged)[/dim]",
                        )
                        repackaged_files += 1
                    else:
                        table.add_row(
                            str(metadata_file.name), name, "[green]✓[/green]", bytes_cell, "[green]PASS[/green]"
                        )
                else:
                    table.add_row(
                        str(metadata_file.name), name, "[red]✗[/red]", bytes_cell, "[red]CONTENT MISMATCH[/red]"
                    )
                    failed_files += 1
            elif expected_bytes:
                # Legacy metadata (no content hash): byte hash is the verdict.
                if bytes_ok:
                    table.add_row(str(metadata_file.name), name, "—", bytes_cell, "[green]PASS[/green]")
                else:
                    table.add_row(str(metadata_file.name), name, "—", bytes_cell, "[red]HASH MISMATCH[/red]")
                    failed_files += 1
            else:
                table.add_row(str(metadata_file.name), name, "N/A", "N/A", "[yellow]No hash in metadata[/yellow]")
                failed_files += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            table.add_row(str(metadata_file.name), name, "N/A", "N/A", f"[red]Error: {e}[/red]")
            failed_files += 1

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {total_files - failed_files}/{total_files} files passed validation")
    if repackaged_files:
        console.print(
            f"[dim]{repackaged_files} file(s): content matches but container bytes differ "
            f"(same data, repackaged) — expected for re-generated GWF frames.[/dim]"
        )

    if failed_files > 0:
        console.print(f"[red]{failed_files} files failed validation[/red]")
        raise typer.Exit(1)
    console.print("[green]All files validated successfully![/green]")


def _locate_output(candidate: Path, recorded: str, metadata_file: Path, globals_config: dict) -> Path:
    """Return where a recorded output is, trying the record's own location when *candidate* is absent.

    A shareable record holds its working directory as ``.``, so *candidate* is relative to wherever
    ``gwmock validate`` was started. When nothing is there, the working directory is inferred from
    where the record itself sits: it was written to ``<working-directory>/<metadata-directory>``, so
    stripping a relative ``metadata-directory`` from the record's directory gives the working
    directory back -- which lets a shared data set be validated from any directory.

    Args:
        candidate: The output resolved against the recorded working directory.
        recorded: The output path as the record holds it.
        metadata_file: The record's path.
        globals_config: The record's ``globals`` section.

    Returns:
        The first location that exists, or *candidate* when neither does.
    """
    if candidate.exists() or Path(recorded).is_absolute():
        return candidate
    metadata_directory = globals_config.get("metadata-directory") or globals_config.get("metadata_directory")
    record_directory = metadata_file.parent.absolute()
    if isinstance(metadata_directory, str) and metadata_directory and not Path(metadata_directory).is_absolute():
        parts = Path(metadata_directory).parts
        if record_directory.parts[-len(parts) :] == parts:
            working_directory = Path(*record_directory.parts[: -len(parts)])
            inferred = working_directory / recorded
            if inferred.exists():
                return inferred
    return candidate


def _same_file_key(path: Path) -> Path:
    """Return one spelling of *path* for an existing file, so two routes to it compare equal."""
    return path.resolve() if path.exists() else path
