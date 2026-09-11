# ruff: noqa: PLC0415
"""CLI command to look up which frame file(s) contain a given injected glitch."""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated

import typer


def find_glitch_command(
    metadata_dir: Annotated[
        Path,
        typer.Option("--metadata-dir", help="Directory with batch metadata files and glitch_index.yaml."),
    ],
    event_id: Annotated[
        str | None,
        typer.Option("--id", help="Glitch event id, e.g. 'H1-0-3' (detector, model index, ordinal)."),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option("--param", help="Column filter, e.g. 'glitch_class==Blip' (repeatable; combined with AND)."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit results as JSON.")] = False,
) -> None:
    """Find the noise frame file(s) that contain a given injected glitch.

    The glitch counterpart of ``gwmock find-signal``. Look up by ``--id`` (fast, via
    ``glitch_index.yaml``) and/or by one or more ``--param`` predicates matched against
    the truth catalogue recorded in the batch metadata, e.g.
    ``--param glitch_class==Koi_Fish --param realized_snr>=8``.

    A glitch row is flat, so a filter names a catalogue column directly: ``detector``,
    ``kind``, ``glitch_class``, ``target_snr``, ``realized_snr``, ``amplitude``,
    ``gps_start_time``, ``gps_peak_time``, ``duration_seconds``.

    ``gps_start_time`` is where the waveform **starts**, not where it peaks -- for a 2 s
    reconstruction the visible transient is about a second later, and ``gps_peak_time``
    is that instant. A window built around the wrong one of the two misses the glitch.
    """
    from gwmock.cli.utils.signal_lookup import find_glitches, parse_param_filter

    try:
        filters = [parse_param_filter(spec) for spec in (param or [])]
    except ValueError as error:
        # A malformed filter is a mistake in the invocation, so it gets the usage error the
        # rest of the command's input mistakes get. Left to propagate, `parse_param_filter`'s
        # `ValueError` reached the user as a traceback.
        raise typer.BadParameter(str(error), param_hint="--param") from error
    if event_id is None and not filters:
        raise typer.BadParameter("Provide --id and/or at least one --param filter.")

    matches = find_glitches(metadata_dir, event_id=event_id, param_filters=filters)

    if json_output:
        typer.echo(_json.dumps(matches, indent=2, default=str))
        return

    if not matches:
        typer.echo("No matching glitches found.")
        raise typer.Exit(code=1)

    for match in matches:
        frames = ", ".join(match.get("frames") or []) or "(no frame recorded)"
        # The start time is labelled as such in the output, not just in the help: a bare
        # "gps=" beside a glitch reads as the peak, which is the reading that loses the glitch.
        typer.echo(f"glitch {match['event_id']} (waveform starts at {match.get('gps_start_time')}) -> {frames}")
