# ruff: noqa PLC0415

"""
A tool to generate and manage default configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer


def _redirect_example_data_file(
    config, *, src_path: Path, dst_path: Path, overwrite: bool
) -> tuple[Path, Path, bool] | None:
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
        ``(source, destination, may_replace)`` of the data file to copy, or ``None`` when there is
        nothing to copy -- no data file, or the destination is already that file or holds the same
        bytes. The path is rewritten either way. ``may_replace`` is whether a file was at the
        destination here, with ``--overwrite`` given: only such a file may be replaced; one that
        appears later is not.

    Raises:
        FileExistsError: If a different file already exists where the data file would be copied
            and *overwrite* is not set.
    """
    import filecmp
    import os

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
    # Before anything else, including the skip below: a destination already holding the shipped
    # bytes is exactly what a kill after the install, before the backup was removed, leaves.
    _refuse_leftovers_of_an_interrupted_copy(destination)
    if destination.is_symlink():
        # A link -- dangling or not -- is the user's arrangement: backing it up and putting the
        # catalogue in its place would replace the link rather than what it points to.
        raise FileExistsError(
            f"{destination} is a symbolic link. gwmock config --get does not replace a link with the "
            "example's catalogue; remove it, or write the configuration elsewhere with --output."
        )
    arguments["path"] = destination.as_posix()
    if destination.exists():
        # The same file -- the copy written into the example's own directory -- or one already
        # holding the same bytes needs no copy; copying a file onto itself raises.
        if os.path.samefile(source, destination) or filecmp.cmp(source, destination, shallow=False):
            return None
        if not overwrite:
            raise FileExistsError(f"File already exists: {destination}. Use --overwrite to overwrite.")
        return source, destination, True
    return source, destination, False


#: The prefix of every temporary file ``config --get`` writes beside a catalogue. Reserved for it, so
#: a user's own ``.bbh_population.csv.notes.backup`` is never taken for one.
_TEMPORARY_PREFIX = ".gwmock-config-get."


#: Bytes of randomness in a temporary file's token, written as ``2 *`` that many lowercase hex digits.
_TOKEN_BYTES = 8
#: How many tokens to try before giving up on creating a temporary file.
_TOKEN_ATTEMPTS = 100


def _temporary_token() -> str:
    """Return the random part of a temporary file's name: 16 lowercase hexadecimal digits."""
    import secrets

    return secrets.token_hex(_TOKEN_BYTES)


def _create_temporary_file(destination: Path, suffix: str) -> Path:
    """Create, exclusively, an empty temporary file beside *destination* and return its path.

    See :func:`_open_temporary_file`, which this closes at once.
    """
    import os

    path, descriptor = _open_temporary_file(destination, suffix)
    os.close(descriptor)
    return path


def _open_temporary_file(destination: Path, suffix: str) -> tuple[Path, int]:
    """Create, exclusively, an empty temporary file beside *destination*, and keep it open.

    Named ``.gwmock-config-get.<name>.<token><suffix>`` with the token from
    :func:`_temporary_token`, so the shape is gwmock's own rather than ``tempfile``'s, whose name
    format is not promised. A name already taken is retried with a new token, a bounded number of
    times.

    Args:
        destination: The file the temporary one sits beside.
        suffix: ``".part"`` or ``".backup"``.

    Returns:
        The created file's path and its open, writable descriptor.

    Raises:
        FileExistsError: If every attempted name was taken.
    """
    import os

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _ in range(_TOKEN_ATTEMPTS):
        path = destination.parent / f"{_temporary_prefix(destination)}{_temporary_token()}{suffix}"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        return path, descriptor
    raise FileExistsError(f"Could not create a temporary file beside {destination} after {_TOKEN_ATTEMPTS} attempts.")


def _temporary_prefix(destination: Path) -> str:
    """Return the prefix of the temporary files written beside *destination*."""
    return f"{_TEMPORARY_PREFIX}{destination.name}."


def _leftovers_of_an_interrupted_copy(destination: Path) -> list[Path]:
    """Return the staging and backup files an interrupted ``config --get`` left beside *destination*.

    Matched exactly: the reserved prefix, the catalogue's name, the 16 hexadecimal digits of
    :func:`_temporary_token`, then ``.part`` or ``.backup``.
    """
    import re

    pattern = re.compile(rf"{re.escape(_temporary_prefix(destination))}[0-9a-f]{{{2 * _TOKEN_BYTES}}}\.(part|backup)")
    if not destination.parent.is_dir():
        return []
    return sorted(path for path in destination.parent.iterdir() if pattern.fullmatch(path.name))


def _refuse_leftovers_of_an_interrupted_copy(destination: Path) -> None:
    """Refuse to touch *destination* while an interrupted copy's temporary files sit beside it.

    Raises:
        FileExistsError: Naming each file and how to recover; nothing is changed.
    """
    leftovers = _leftovers_of_an_interrupted_copy(destination)
    if not leftovers:
        return
    backups = [path for path in leftovers if path.suffix == ".backup"]
    parts = [path for path in leftovers if path.suffix == ".part"]
    lines = [f"An earlier gwmock config --get was interrupted beside {destination}. Nothing was changed."]
    if backups:
        lines.append(
            f"{', '.join(map(str, backups))} holds the catalogue that was at {destination} before. To restore "
            f"it, rename it to {destination}, replacing what is there; or, if the catalogue now at "
            f"{destination} is the one you want, delete it."
        )
    if parts:
        moved = "" if backups else f" The catalogue at {destination}, if any, was not moved."
        lines.append(f"{', '.join(map(str, parts))} is an unfinished staged copy and safe to delete.{moved}")
    raise FileExistsError(" ".join(lines))


def _after_staging(staging: Path) -> None:
    """Run between writing the staged copy and checking it; a seam for tests, and a no-op."""
    del staging


class _AppearedError(Exception):
    """Raised inside the install when a file appeared at a destination that was absent."""


def _before_placing(destination: Path) -> None:
    """Run between the preflight and the install's look at the destination; a seam for tests."""
    del destination


def _place_without_replacing(staging: Path, destination: Path) -> None:
    """Move *staging* to *destination*, failing with ``FileExistsError`` if anything is there.

    POSIX: ``os.link`` creates the new name only if it is free, then the staged name is removed.
    Windows: ``os.rename`` refuses an existing destination there.
    """
    import os

    if _closes_before_rename():
        os.rename(staging, destination)
        return
    os.link(staging, destination, follow_symlinks=False)
    _remove_quietly(staging)


class _ForeignEntryError(Exception):
    """Raised inside the install when the moved file is not the one that was staged."""


def _before_move(staging: Path) -> None:
    """Run between the staged copy's check and its move into place; a seam for tests, and a no-op."""
    del staging


def _after_install(destination: Path) -> None:
    """Run right after the new file is moved into place; a seam for tests, and a no-op."""
    del destination


def _close_after_install(descriptor: int, staging: Path) -> None:
    """Close the staged copy's descriptor once it is in place; a failure is logged, not raised.

    The data was fsynced before the move and the file is already installed, so a failing close
    changes nothing on disk, and failing the command over it would roll back a good install.
    """
    import logging
    import os

    try:
        os.close(descriptor)
    except OSError as error:
        logging.getLogger("gwmock").warning("Could not close the installed copy %s: %s", staging, error)


def _closes_before_rename() -> bool:
    """Whether the staged file must be closed before it is renamed -- on Windows, which cannot rename an open file."""
    import os

    return os.name == "nt"


def _identity_from(status: Any) -> tuple[int, int] | None:
    """Return ``(st_dev, st_ino)`` from a stat result, or ``None`` when the file system gives no inode.

    On Windows, Python's ``os.stat`` and ``os.fstat`` fill ``st_ino`` from the file's index (its
    file ID) and ``st_dev`` from the volume serial number; gwmock relies on that, as Python has done
    since 3.5. An NTFS file ID carries a sequence number, so it is not reused for a new file; on
    POSIX the staged file's descriptor stays open until after the move, so its inode cannot be
    reused meanwhile. A file system reporting no index gives ``st_ino == 0``, which identifies
    nothing.
    """
    if not status.st_ino:
        return None
    return (status.st_dev, status.st_ino)


def _witness(descriptor: int) -> tuple[int, int] | None:
    """Return the identity of the open staged file, to recognise it again after it is moved."""
    import os

    return _identity_from(os.fstat(descriptor))


def _names_witness(path: Path, witness: tuple[int, int]) -> bool:
    """Return whether *path* is a regular file, not a link, with the identity *witness*."""
    import os
    import stat

    try:
        entry = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(entry.st_mode) and _identity_from(entry) == witness


def _still_names(path: Path, descriptor: int) -> bool:
    """Return whether *path* is still the regular file open at *descriptor*, not a link or another file."""
    import os
    import stat

    try:
        entry = os.lstat(path)
    except OSError:
        return False
    opened = os.fstat(descriptor)
    return (
        not stat.S_ISLNK(entry.st_mode)
        and stat.S_ISREG(entry.st_mode)
        and (entry.st_dev, entry.st_ino) == (opened.st_dev, opened.st_ino)
    )


def _remove_quietly(path: Path) -> None:
    """Delete *path* if it exists; a failure is logged and never raised, so it cannot stop a restore."""
    import logging

    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        logging.getLogger("gwmock").warning("Could not remove the temporary file %s: %s", path, error)


class _DataFileInstall:
    """Put an example's data file in place so that it can be undone if the configuration fails.

    The file is staged in a temporary ``.gwmock-config-get.<name>.*.part`` beside the destination
    and moved into place with ``os.replace``, so a copy that fails partway never touches the
    destination. A file already there is first moved to a temporary
    ``.gwmock-config-get.<name>.*.backup`` beside it: :meth:`roll_back` restores it -- or, when there
    was none, removes the new file -- and :meth:`commit` discards the backup. Where a file is
    restored, the restore comes before deleting temporary files, which is best effort. A file that
    is not this call's -- one that appeared at the destination, or a foreign entry found there after
    the move -- is never deleted: with a backup of the previous catalogue that backup is put back
    over it, and without one it is left as it is.

    A destination that was absent when the command looked is filled with an operation that fails
    if anything is there by then (``os.link`` on POSIX, ``os.rename`` on Windows), so a file that
    appears meanwhile is not replaced; only ``--overwrite`` on a destination already seen replaces.

    **The guarantee is against exceptions, not against the process being killed.** A kill between
    the steps can leave the catalogue's name missing with its old bytes in the backup, a stray
    ``.part``, or the new catalogue beside the old configuration. ``config --get`` refuses to run
    while such files are present, naming them, rather than guessing how to recover.
    """

    def __init__(self, source: Path, destination: Path, *, may_replace: bool = False) -> None:
        """Install *source* at *destination*.

        Args:
            source: The data file to copy.
            destination: Where it goes.
            may_replace: Whether a file found at *destination* may be backed up and replaced --
                true only when the caller saw one there and ``--overwrite`` was given. Otherwise
                the file is installed with an operation that fails if anything is there, so one
                that appeared meanwhile is never replaced.
        """
        import os
        import shutil

        self._destination = destination
        self._backup: Path | None = None
        staging, descriptor = _open_temporary_file(destination, ".part")
        swapped = False
        foreign = False
        installed = False
        moving = False
        witness: tuple[int, int] | None = None
        existed = os.path.lexists(destination)
        try:
            try:
                # Written through the descriptor the exclusive create returned, never by reopening
                # the path: in a directory another user can write to, the entry could be swapped
                # for a symlink in between, and a copy by path would follow it.
                with source.open("rb") as reader:
                    writer = os.fdopen(descriptor, "wb", closefd=False)
                    shutil.copyfileobj(reader, writer)
                    writer.flush()
                    os.fsync(descriptor)
                witness = _witness(descriptor)
                if witness is None:
                    raise RuntimeError(
                        f"The file system holding {destination.parent} does not report file identity, "
                        "so gwmock config --get cannot install the catalogue there safely: it could not "
                        "tell its own file from one put in its place. Nothing was changed. Copy the "
                        f"catalogue {source} there by hand, or write the configuration to another file "
                        "system with --output."
                    )
                _after_staging(staging)
                if not _still_names(staging, descriptor):
                    swapped = True
                    raise RuntimeError(
                        f"The staged copy {staging} was replaced by another file before it was moved "
                        f"into place, so {destination} was left untouched. Check who can write to "
                        f"{destination.parent}; the replacing entry has been left as it is."
                    )
                if _closes_before_rename():
                    # Windows cannot rename a file that is open; the check above is the last look.
                    os.close(descriptor)
                    descriptor = -1
                _before_placing(destination)
                if not may_replace:
                    if os.path.lexists(destination):
                        raise _AppearedError
                    _before_move(staging)
                    moving = True
                    try:
                        _place_without_replacing(staging, destination)
                    except FileExistsError:
                        moving = False
                        raise _AppearedError from None
                    except OSError as error:
                        moving = False
                        raise RuntimeError(
                            f"The file system holding {destination.parent} cannot place a file there "
                            f"without risking replacing one that appeared meanwhile ({error}), so "
                            "gwmock config --get did not install the catalogue. Nothing was changed. "
                            f"Copy the catalogue {source} there by hand, or write the configuration to "
                            "another file system with --output."
                        ) from error
                else:
                    if destination.is_symlink():
                        raise FileExistsError(
                            f"{destination} became a symbolic link while installing. Nothing was "
                            "replaced; gwmock config --get does not replace a link with the catalogue."
                        )
                    if os.path.lexists(destination):
                        backup_name = _create_temporary_file(destination, ".backup")
                        try:
                            # os.replace renames the directory entry onto backup_name: were that
                            # entry swapped for a symlink meanwhile, the link itself would be
                            # replaced, and its target neither followed nor written.
                            os.replace(destination, backup_name)
                        except BaseException:
                            _remove_quietly(backup_name)
                            raise
                        self._backup = backup_name
                    _before_move(staging)
                    moving = True
                    os.replace(staging, destination)
                if not _names_witness(destination, witness):
                    # The rename moved whatever was at the staged name; if that entry was swapped
                    # after the check above, the destination now holds a file this command did not
                    # write. Handled below, with the same case reached through an exception.
                    raise _ForeignEntryError
                installed = True
                _after_install(destination)
            finally:
                if not installed and moving:
                    # An interrupt can arrive after os.replace has moved the file but before it
                    # returns. The move counts as done only when the destination is the very file
                    # that was staged -- the same (st_dev, st_ino) witness on every platform -- and
                    # never while a destination that was there at the start has not been backed up.
                    # Any other entry there now is foreign: by this point the destination was
                    # either absent or moved to the backup, so whatever is there was put by someone
                    # else, possibly through a swapped staged entry.
                    assert witness is not None
                    installed = not (existed and self._backup is None) and _names_witness(destination, witness)
                    foreign = not installed and os.path.lexists(destination)
                if descriptor >= 0:
                    if installed:
                        _close_after_install(descriptor, staging)
                    else:
                        os.close(descriptor)
        except BaseException as error:
            if isinstance(error, _AppearedError):
                # Not ours: leave it, and remove only our staged copy.
                _remove_quietly(staging)
                raise FileExistsError(
                    f"A file appeared at {destination} while installing the catalogue; nothing was "
                    "replaced. Run the command again with --overwrite to replace it."
                ) from None
            if foreign:
                # The backup is ours and holds the user's previous catalogue: put it back over the
                # foreign entry. With no backup the foreign entry is left alone -- it is not ours.
                backup = self._backup
                if backup is not None:
                    self._restore(error)
                message = (
                    f"{destination} holds a file this command did not write: the staged copy {staging} "
                    "was replaced before it was moved, or another process put a file there meanwhile. "
                    "The configuration was not saved. "
                    + (
                        f"The previous catalogue, from {backup}, has been put back over it."
                        if backup is not None
                        else f"{destination} has been left as it is; check who can write to {destination.parent}."
                    )
                )
                if isinstance(error, _ForeignEntryError):
                    raise RuntimeError(message) from None
                raise RuntimeError(message) from error
            # Restore first, whatever was raised -- an interrupt included -- and whether or not the
            # new file is already in place: the configuration is never saved after this, so the
            # destination must be left as it was.
            if installed:
                self.roll_back(error)
            elif self._backup is not None and not destination.exists():
                self._restore(error)
            if not swapped:
                _remove_quietly(staging)
            raise

    def _restore(self, cause: BaseException | None) -> None:
        """Move the backup back onto the destination, or say exactly where the old file is."""
        import os

        backup = self._backup
        if backup is None:
            return
        try:
            os.replace(backup, self._destination)
        except OSError as error:
            raise RuntimeError(
                f"Could not restore {self._destination} from its backup {backup}: {error}. The previous "
                f"file is intact there; restore it by renaming {backup} to {self._destination}."
            ) from (cause if cause is not None else error)
        self._backup = None

    def commit(self) -> None:
        """Keep the new file and discard the backup of the old one, best effort."""
        if self._backup is not None:
            _remove_quietly(self._backup)
            self._backup = None

    def roll_back(self, cause: BaseException | None = None) -> None:
        """Restore the file that was there before, or remove the new one when there was none.

        Args:
            cause: The error that made the rollback necessary, chained to a failed restore.
        """
        if self._backup is not None:
            self._restore(cause)
        else:
            _remove_quietly(self._destination)


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

        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {dst_path}. Use --overwrite to overwrite.")
        # The catalogue and the configuration are installed as a pair: the catalogue first, so a
        # configuration is never written pointing at one that failed to arrive, and undone if the
        # configuration then cannot be written, so a failure leaves the destination as it was.
        installed = (
            _DataFileInstall(data_file[0], data_file[1], may_replace=data_file[2]) if data_file is not None else None
        )
        try:
            save_config(file_name=dst_path, config=config, overwrite=overwrite, backup=False)
        except BaseException as e:
            if installed is not None:
                installed.roll_back()
            if isinstance(e, FileExistsError):
                raise FileExistsError(f"File already exists: {dst_path}. Use --overwrite to overwrite.") from e
            raise
        if installed is not None:
            installed.commit()
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
            such as ``noise/uncorrelated_gaussian/quick_start``. An example's
            catalogue is copied beside it; a failure puts back whatever was
            there, but a killed process can leave ``.gwmock-config-get.*.backup``
            or ``.part`` files, and the command then refuses until they are dealt with.

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
