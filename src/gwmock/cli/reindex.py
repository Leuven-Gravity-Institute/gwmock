# ruff: noqa: PLC0415
"""CLI command to rebuild the truth indexes from the batch metadata files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


def reindex_command(
    metadata_dir: Annotated[
        Path,
        typer.Option(
            "--metadata-dir",
            help="Directory with batch metadata files, signal_index.yaml and glitch_index.yaml.",
        ),
    ],
) -> None:
    """Rebuild the signal and glitch indexes from the batch metadata files.

    ``signal_index.yaml`` and ``glitch_index.yaml`` are caches of the injections
    recorded in the ``*.metadata.json`` files, which are the source of truth.
    Rebuilding discards whatever an index held and derives it again from those
    files, so an index that lost entries -- concurrent runs sharing this directory
    on a filesystem without working ``flock``, or writers on different hosts -- is
    repaired without rerunning any simulation.

    Both are rebuilt together, and a directory whose runs injected no glitches
    simply gets an empty glitch index; a directory holding no batch metadata files
    at all is refused, because that means the wrong path was given.

    Neither index is replaced until the sources of both have been checked, so a
    metadata file that one index can read and the other cannot stops the command
    with both files untouched rather than half way through. A write that fails
    part-way -- a full disk, say -- cannot be made atomic across two files, and is
    reported naming the index that was replaced.

    The rebuild takes the same exclusive lock a running batch does, and it
    re-baselines the digest recorded beside each index, so a directory whose writes
    were being refused as stale accepts them again afterwards.

    Stop writers on other hosts first: a rebuild indexes the batch metadata files
    it can list, and a shared filesystem may not yet be listing a file another host
    has just written.
    """
    from gwmock.cli.simulate_utils import (
        IndexDigestNotRecordedError,
        IndexRebuildError,
        PartialIndexRebuildError,
        rebuild_truth_indexes,
    )

    try:
        rebuilt_indexes = rebuild_truth_indexes(metadata_dir)
    except (IndexRebuildError, IndexDigestNotRecordedError, PartialIndexRebuildError) as error:
        # Each carries a message written for whoever is holding the terminal -- what stopped, what
        # state that leaves the directory in, and what to do next -- so printing it beats a
        # traceback that buries it. They are not the same outcome: the first means nothing was
        # written, the second that an index is committed and correct while its sidecar is behind
        # it, and the third that one index of the pair was replaced and another was not. Each
        # message says which, and all need the operator, so all exit non-zero.
        #
        # `OSError` is deliberately not caught, though `rebuild_signal_index` documents raising it
        # too. A full disk or an unwritable directory has no repair this command can prescribe, and
        # the traceback's location is worth more than an echo of the OS's own message. The line is
        # whether the exception carries advice, not whether it is documented.
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    for rebuilt in rebuilt_indexes:
        typer.echo(
            f"Rebuilt {rebuilt.index_file} from {rebuilt.batches} batch metadata file(s): {rebuilt.events} event(s)."
        )
