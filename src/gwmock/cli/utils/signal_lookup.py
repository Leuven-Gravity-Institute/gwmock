"""Look up which frame file(s) contain a given simulated signal or injected glitch.

Both are recorded per batch in the metadata files -- ``signal.injections`` and
``noise.glitch_injections``, the source of truth -- and mirrored into
``signal_index.yaml`` and ``glitch_index.yaml`` for O(1) lookup by ``event_id``.
Parameter-based lookup scans the events in the metadata files.

One implementation over both, parameterised by
:class:`~gwmock.cli.utils.truth_index.IndexSpec`: the two indexes have the same
shape and the same fallback from the id shortcut to the metadata files, and a
second copy of that would be a second place for a fix to be missing from.
"""

from __future__ import annotations

import json
import math
import operator
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from gwmock.cli.utils.truth_index import GLITCH_INDEX, SIGNAL_INDEX, IndexSpec

# Relative tolerance for numeric == / != filters, so representation noise (e.g.
# a float round-tripped through JSON) does not cause a scientifically-equal
# value to be missed. Ordering operators stay exact.
_EQUALITY_REL_TOL = 1.0e-9

# Two-character operators must be tried before one-character ones.
_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}
_FILTER_RE = re.compile(r"^\s*(?P<key>[^<>=!\s]+)\s*(?P<op>>=|<=|!=|==|>|<|=)\s*(?P<value>.+?)\s*$")


def parse_param_filter(spec: str) -> tuple[str, str, Any]:
    """Parse a ``key OP value`` filter such as ``mass_1>=30`` or ``approximant==IMRPhenomXPHM``."""
    match = _FILTER_RE.match(spec)
    if not match:
        raise ValueError(f"Invalid parameter filter '{spec}'; expected key OP value (e.g. mass_1>=30).")
    return match.group("key"), match.group("op"), _coerce(match.group("value"))


def _coerce(raw: str) -> Any:
    """Coerce a filter value to float when possible, otherwise keep it as a string."""
    try:
        return float(raw)
    except ValueError:
        return raw


def _matches(parameters: dict[str, Any], filters: list[tuple[str, str, Any]]) -> bool:
    """Return whether an event's parameters satisfy every filter predicate."""
    for key, op, value in filters:
        if key not in parameters:
            return False
        try:
            if not _compare(parameters[key], op, value):
                return False
        except TypeError:
            # Mismatched types (e.g. numeric op on a string parameter) never match.
            return False
    return True


def _compare(actual: Any, op: str, value: Any) -> bool:
    """Apply one filter operator, using a tolerance for numeric equality tests."""
    if op in ("==", "=", "!=") and isinstance(actual, (int, float)) and isinstance(value, (int, float)):
        close = math.isclose(actual, value, rel_tol=_EQUALITY_REL_TOL, abs_tol=0.0)
        return not close if op == "!=" else close
    return _OPERATORS[op](actual, value)


def _iter_metadata_files(metadata_directory: Path) -> Iterator[Path]:
    yield from sorted(metadata_directory.glob("*.metadata.json"))


def _flatten_batches(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(frames, metadata_files)`` for one index entry, in the order written.

    The index stores contributions per batch rather than a flattened list of frames, because a
    re-run has to be able to withdraw one batch's contribution and a flattened copy alongside it
    would be a second place for the same fact to live. Flattening happens here, once, on read.

    Frames are deduplicated: two batches can legitimately name the same frame when a segment is
    written once and contributed to twice.

    Pre-1.5.0 entries carried ``metadata`` as a string and ``frames`` flat; they are read as a single
    batch rather than rejected, so an index written by an older gwmock still answers.
    """
    batches = entry.get("batches")
    if batches is None:
        metadata = entry.get("metadata")
        return list(entry.get("frames") or []), ([] if metadata is None else [metadata])

    frames: list[str] = []
    seen: set[str] = set()
    metadata_files: list[str] = []
    for batch in batches:
        for frame in batch.get("frames") or []:
            if frame not in seen:
                seen.add(frame)
                frames.append(frame)
        name = batch.get("metadata")
        if name is not None and name not in metadata_files:
            metadata_files.append(name)
    return frames, metadata_files


def find_signals(
    metadata_directory: Path | str,
    *,
    event_id: int | None = None,
    param_filters: list[tuple[str, str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the signals matching an id and/or parameter filters with their frame file(s).

    With only ``event_id`` set, the fast ``signal_index.yaml`` path is used. When
    parameter filters are supplied, the batch metadata files are scanned (their
    ``signal.injections`` are the source of truth). Each result is a mapping with
    ``event_id``, ``frames`` (signal frame paths), ``metadata`` (a *list* of batch
    metadata file names, since a signal long enough to cross a segment boundary is
    recorded by every batch whose frames it reaches), and ``coa_time``;
    parameter-filtered results also carry ``parameters``.
    """
    return find_events(SIGNAL_INDEX, metadata_directory, event_id=event_id, param_filters=param_filters)


def find_glitches(
    metadata_directory: Path | str,
    *,
    event_id: str | int | None = None,
    param_filters: list[tuple[str, str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the injected glitches matching an id and/or filters with their frame file(s).

    The glitch counterpart of :func:`find_signals`, answering "which frame holds glitch X"
    for the other producer. With only ``event_id`` set the fast ``glitch_index.yaml`` path is
    used; filters scan the batch metadata files, whose ``noise.glitch_injections`` are the
    source of truth.

    A glitch row is flat, so a filter names a catalogue column directly --
    ``detector==H1``, ``glitch_class==Koi_Fish``, ``realized_snr>=8``,
    ``gps_start_time>=1256655618`` -- where a signal filter names a source parameter.

    ``gps_start_time`` is the GPS time of the injected waveform's **first sample**, not its
    peak: a filter or a window built around it has to allow for the waveform's own length,
    which the catalogue records as ``duration_seconds``.

    Unlike a signal, a glitch appears in the record of the batch it *starts* in and no other,
    so its ``metadata`` list holds one file even where its samples reach the next frame.
    """
    return find_events(GLITCH_INDEX, metadata_directory, event_id=event_id, param_filters=param_filters)


def find_events(
    spec: IndexSpec,
    metadata_directory: Path | str,
    *,
    event_id: str | int | None = None,
    param_filters: list[tuple[str, str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the events of one truth catalogue matching an id and/or parameter filters.

    Args:
        spec: Which catalogue to search, and which index caches it.
        metadata_directory: Directory holding the batch metadata files and the index.
        event_id: Event id to look up, or ``None`` to match on filters alone.
        param_filters: Parsed ``(key, op, value)`` predicates, combined with AND.

    Returns:
        One mapping per match, carrying ``event_id``, ``frames``, ``metadata`` (a list of
        batch metadata file names) and the catalogue's time column; filtered results also
        carry ``parameters``.
    """
    metadata_directory = Path(metadata_directory)
    param_filters = param_filters or []
    results: list[dict[str, Any]] = []

    if event_id is not None and not param_filters:
        index_file = metadata_directory / spec.index_file_name
        if not index_file.exists():
            return results
        with index_file.open(encoding="utf-8") as f:
            index = yaml.safe_load(f) or {}
        entry = index.get(str(event_id))
        if entry:
            frames, metadata_files = _flatten_batches(entry)
            results.append(
                {
                    "event_id": event_id,
                    "frames": frames,
                    # A list now, because one signal spans the frames of several batches. Kept as a
                    # list even when there is one, so a consumer never has to branch on the type.
                    "metadata": metadata_files,
                    spec.time_key: entry.get(spec.time_key),
                }
            )
        return results

    for meta_path in _iter_metadata_files(metadata_directory):
        try:
            with meta_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        injections = spec.events(metadata) or []
        if not injections:
            continue
        frames = [
            output["path"]
            for output in metadata.get("outputs", [])
            if output.get("kind") == spec.output_kind and "path" in output
        ]
        for injection in injections:
            if event_id is not None and injection.get("event_id") != event_id:
                continue
            parameters = spec.parameters(injection)
            if not _matches(parameters, param_filters):
                continue
            results.append(
                {
                    "event_id": injection.get("event_id"),
                    "frames": frames,
                    # A one-element list, matching the id path's shape. This path yields one result
                    # per batch, so each result genuinely has one metadata file -- but a consumer
                    # reading both paths should not have to know which one it asked.
                    "metadata": [meta_path.name],
                    "parameters": parameters,
                    spec.time_key: parameters.get(spec.time_key),
                }
            )
    return results
