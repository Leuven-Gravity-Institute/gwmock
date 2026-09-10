"""The declared, versioned schema of the strain artifacts gwmock writes.

A pipeline reading gwmock output needs to know what it is holding: which dataset carries the samples,
which attributes carry the grid, and -- when any of that changes -- that it changed. Until this module
existed the answer was only in gwmock's source, so a consumer had to match the writer's implementation
rather than a contract, and a layout change reached it as a wrong result rather than as a refusal.

What the schema declares is written into the artifact itself, at the **file root**:

``schema``
    ``"gwmock-strain"`` -- which contract this is, so a ``schema_version`` written by some other
    producer is not mistaken for this one.

``schema_version``
    ``MAJOR.MINOR.PATCH``. The major moves when a reader written against the previous version would
    misread a file (a dataset renamed, an attribute's meaning changed); the minor moves when something
    is added that such a reader can ignore.

``run_metadata`` (1.1.0, optional)
    The provenance record of the run that wrote the file, as a JSON document -- the same record the
    metadata sidecar carries, so that a file handed to another pipeline describes itself. Optional
    because a file may be written by a path that has no record to embed, and because every file
    written before 1.1.0 has none; a consumer reads it with :func:`read_run_metadata` and falls back
    to the sidecar when it is absent.

    It is a *copy* rather than a second source of truth: one record is built per batch and written to
    both places, and the embedded copy differs only by what it deliberately leaves out. It omits the
    injection parameters unless the run asked for them (``orchestration.include-injection-parameters``
    -- a blind challenge must not release the answer key with the data), and it omits the file hashes,
    which cannot describe the file they are stored in. See
    :func:`gwmock.cli.utils.metadata.embed_metadata_record`.

Version 1.0.0 requires, of every dataset in the file:

* the samples of one channel, and the dataset is **named** for that channel;
* ``x0`` -- the epoch of the first sample, in GPS seconds;
* ``dx`` -- the sample interval, in seconds;
* ``xunit`` -- the unit those two are in, ``"s"``;
* ``channel`` and ``name`` -- the channel the samples belong to.

Anything else on the dataset is permitted and a consumer ignores it. Two such extras are written in
practice: ``unit`` (``"strain"``, where the producer recorded one), and ``t0``/``dt``, which duplicate
``x0``/``dx`` and are how the multichannel writer in ``gwmock-signal`` spells the grid. The duplicate is
kept because that writer's own reader requires it, and it is not a second source of truth: gwmock
derives one pair from the other, and **refuses a file whose two spellings disagree** -- at declaration
and again at validation. Enforcement rather than convention, because the two are read by different
consumers: the content hash takes ``x0``/``dx`` and ``DetectorStrainStack.read`` takes ``t0``/``dt``, so
a file allowed to carry both with different values is one artifact that two readers place at two
different times.

That last point is the reason declaring the schema is not only a matter of writing two root attributes.
gwmock composes strain HDF5 through three different writers -- its own, gwpy's, and gwmock-signal's --
and they did not agree on how to spell the grid. Declaring one contract over three layouts would have
announced a compatibility that two thirds of the artifacts did not have, so
:func:`declare_strain_schema` makes the file match the declaration before making it, and
:func:`require_strain_schema` checks the layout rather than trusting the claim.

Two consequences of the root placement are deliberate:

* **Root, not dataset.** gwpy's HDF5 reader passes every dataset attribute to the series constructor as
  a keyword argument, so an attribute it does not know raises ``TypeError`` and the file becomes
  unreadable through the standard reader. Measured: adding ``schema_version`` to the dataset makes
  ``TimeSeries.read`` fail with ``Array.__new__() got an unexpected keyword argument 'schema_version'``;
  at the root it reads unchanged. The declaration also describes the file rather than one dataset in it,
  so a consumer can check it without first knowing the channel name.
* **Separate from the metadata record's version.** ``gwmock.cli.utils.metadata.SCHEMA_VERSION`` versions
  the *provenance record*, and its releases so far recorded changes to the meaning of
  ``signal.injections`` -- which say nothing about the layout of a strain file. Wiring one constant to
  both would announce a strain-format change on every metadata change and, worse, let a real strain
  change hide inside a metadata bump. They are two contracts and they move independently.

Only HDF5 carries the declaration, because only HDF5 has somewhere to put it: ``.npy`` is a bare array
container with no metadata space, and GWF frames are composed by the frame library from a fixed set of
fields. The same is true of the embedded record above, and for the same reason. For those formats the
run's metadata *sidecar* remains the only description of what was written, and a consumer that wants a
self-describing artifact has to ask for HDF5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

#: The name written to the ``schema`` attribute, identifying which contract the file claims to meet.
STRAIN_SCHEMA = "gwmock-strain"

#: The version of the strain contract this gwmock writes.
#:
#: 1.0.0: the layout described in the module docstring -- one dataset per channel, named for the
#: channel, with the grid in ``x0``/``dx``/``xunit`` and the channel in ``channel``/``name``. This is
#: the layout gwmock's own writer already produced; 1.0.0 declares it, and makes the other two writers
#: meet it rather than widening the contract to cover what each of them happened to emit.
#:
#: 1.1.0: the root may carry ``run_metadata``, the run's provenance record as JSON. Added rather than
#: changed: no dataset moved, nothing already declared means anything different, and a reader written
#: against 1.0.0 goes on reading a 1.1.0 file without noticing the attribute -- which is what the minor
#: component is for.
STRAIN_SCHEMA_VERSION = "1.1.0"

#: Root attribute naming the schema.
SCHEMA_ATTRIBUTE = "schema"

#: Root attribute carrying the schema version.
SCHEMA_VERSION_ATTRIBUTE = "schema_version"

#: Root attribute carrying the run's provenance record, as a JSON document. Written by
#: :func:`gwmock.cli.utils.metadata.embed_metadata_record`, which is the only place that decides what a
#: copy of the record may contain; read back with :func:`read_run_metadata`.
RUN_METADATA_ATTRIBUTE = "run_metadata"

#: The dataset attributes version 1.0.0 requires. A consumer may read these on any declared file.
REQUIRED_DATASET_ATTRIBUTES = ("x0", "dx", "xunit", "channel", "name")

#: How ``gwmock-signal``'s multichannel writer spells the epoch and the sample interval. Read to derive
#: ``x0``/``dx`` when they are absent, checked against them when both are there, and left in place
#: afterwards because that writer's reader needs them: deleting them makes ``DetectorStrainStack.read``
#: fail with ``can't locate attribute: 't0'``.
_GRID_ALIASES = (("x0", "t0"), ("dx", "dt"))

_HDF5_SUFFIXES = frozenset({".hdf5", ".h5"})
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class StrainSchema(NamedTuple):
    """The schema declaration read from a strain artifact."""

    name: str
    version: str

    @property
    def major(self) -> int:
        """Return the major component of the declared version.

        Returns:
            The major version.
        """
        return parse_strain_schema_version(self.version)[0]


def parse_strain_schema_version(value: str) -> tuple[int, int, int]:
    """Split a declared version into its numeric components.

    Args:
        value: The version string.

    Returns:
        The major, minor and patch components.

    Raises:
        ValueError: If the version does not follow ``MAJOR.MINOR.PATCH``.
    """
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Strain schema version '{value}' must follow semantic versioning (MAJOR.MINOR.PATCH).")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def strain_schema_attributes() -> dict[str, str]:
    """Return the root attributes declaring the current strain schema.

    Returns:
        The attribute names and values to write at the root of a strain artifact.
    """
    return {SCHEMA_ATTRIBUTE: STRAIN_SCHEMA, SCHEMA_VERSION_ATTRIBUTE: STRAIN_SCHEMA_VERSION}


def carries_strain_schema(path: str | Path) -> bool:
    """Return whether a path names a format that can carry the declaration.

    Args:
        path: The artifact path.

    Returns:
        True for HDF5 artifacts, False for the formats with no attribute space.
    """
    return Path(path).suffix.lower() in _HDF5_SUFFIXES


def declare_strain_schema(path: str | Path) -> bool:
    """Make an already-written artifact meet the current strain schema, and say that it does.

    Call it once per artifact, after the samples are written and before the file is published under its
    final name -- from every writer, so that one call site is the only thing that decides what a gwmock
    strain artifact looks like. It is a no-op for ``.npy`` and ``.gwf``, which have nowhere to record
    the declaration, so a caller that writes several formats does not have to branch on the format.

    Each dataset is brought up to the 1.0.0 layout before the file claims it: the grid is filled in from
    the ``t0``/``dt`` spelling when ``x0``/``dx`` are absent, and the channel from the dataset's own
    name. Nothing already present is overwritten -- the writer's own value is the authoritative one --
    and no attribute is invented from nothing, so a dataset carrying no grid at all is an error rather
    than a file that declares a layout it does not have.

    A dataset that already carries both spellings of the grid with *different* values is an error for
    the same reason. Not overwriting one of them is right; declaring the file anyway is not, because the
    two are read by different consumers and the artifact would then be one file placing its samples at
    two different times. Refusing it is what "do not overwrite" leaves available.

    **Every** dataset is checked before any of them is written, in a read-only pass, so a refusal leaves
    the artifact exactly as it was found. Checking and completing one dataset at a time reads naturally
    and is wrong for a multichannel file: the datasets before the offending one would already have been
    completed by the time it was reached, leaving a file that is undeclared but no longer what its
    writer produced -- a half-converted artifact, which is worse than either outcome the caller expected.

    Args:
        path: The artifact to declare.

    Returns:
        True if the declaration was written, False if the format cannot carry one.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If a dataset carries no epoch or no sample interval under either spelling, so the
            required layout cannot be completed, or carries both spellings of one with values that
            disagree.
    """
    artifact = Path(path)
    if not carries_strain_schema(artifact):
        return False
    if not artifact.exists():
        raise FileNotFoundError(f"Cannot declare the strain schema on a file that does not exist: {artifact}")

    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    with h5py.File(artifact, "r") as handle:
        for dataset in _datasets(handle):
            _check_dataset(dataset, artifact=artifact)
    with h5py.File(artifact, "a") as handle:
        for dataset in _datasets(handle):
            _complete_dataset(dataset)
        handle.attrs.update(strain_schema_attributes())
    return True


def read_strain_schema(path: str | Path) -> StrainSchema | None:
    """Return the schema an artifact declares, or ``None`` if it declares none.

    ``None`` is the answer for a file written before the declaration existed, one written by another
    producer, and one in a format that cannot carry it -- three different situations that a consumer
    handles the same way: it has no contract and must fall back to the run's metadata record.

    Args:
        path: The artifact to read.

    Returns:
        The declared schema, or None.
    """
    artifact = Path(path)
    if not carries_strain_schema(artifact):
        return None

    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    with h5py.File(artifact, "r") as handle:
        if SCHEMA_ATTRIBUTE not in handle.attrs or SCHEMA_VERSION_ATTRIBUTE not in handle.attrs:
            return None
        return StrainSchema(
            name=_as_text(handle.attrs[SCHEMA_ATTRIBUTE]),
            version=_as_text(handle.attrs[SCHEMA_VERSION_ATTRIBUTE]),
        )


def read_run_metadata(path: str | Path) -> dict[str, Any] | None:
    """Return the provenance record an artifact carries, or ``None`` if it carries none.

    ``None`` covers a file in a format with no metadata space, one written before 1.1.0, one written
    by a path that had no record to embed, and one from another producer -- four situations a consumer
    handles the same way: read the run's metadata sidecar instead.

    What comes back is the *embedded* copy, which is not the whole record. It omits the file hashes,
    and it omits the injection parameters unless the run that wrote it opted in; the sidecar is the
    complete one. See :func:`gwmock.cli.utils.metadata.embed_metadata_record`.

    Args:
        path: The artifact to read.

    Returns:
        The embedded record, or None.

    Raises:
        ValueError: If the attribute is present but is not a JSON document, which no gwmock wrote.
    """
    artifact = Path(path)
    if not carries_strain_schema(artifact):
        return None

    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    with h5py.File(artifact, "r") as handle:
        if RUN_METADATA_ATTRIBUTE not in handle.attrs:
            return None
        document = _as_text(handle.attrs[RUN_METADATA_ATTRIBUTE])
    try:
        return json.loads(document)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact} carries a '{RUN_METADATA_ATTRIBUTE}' attribute that is not JSON: {exc}") from exc


def missing_layout_attributes(path: str | Path) -> dict[str, list[str]]:
    """Return, per dataset, the required attributes the file does not carry.

    Args:
        path: The artifact to inspect.

    Returns:
        A mapping of dataset name to the missing attribute names; empty when the file meets the layout.
    """
    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    missing: dict[str, list[str]] = {}
    with h5py.File(Path(path), "r") as handle:
        for dataset in _datasets(handle):
            absent = [name for name in REQUIRED_DATASET_ATTRIBUTES if name not in dataset.attrs]
            if absent:
                missing[dataset.name.lstrip("/")] = absent
    return missing


def conflicting_grid_attributes(path: str | Path) -> dict[str, list[str]]:
    """Return, per dataset, each grid quantity whose two spellings disagree.

    Args:
        path: The artifact to inspect.

    Returns:
        A mapping of dataset name to a description of each disagreement; empty when the file is
        self-consistent, which includes a file carrying only one spelling.
    """
    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    conflicts: dict[str, list[str]] = {}
    with h5py.File(Path(path), "r") as handle:
        for dataset in _datasets(handle):
            disagreements = _grid_conflicts(dataset)
            if disagreements:
                conflicts[dataset.name.lstrip("/")] = disagreements
    return conflicts


def require_strain_schema(path: str | Path) -> StrainSchema:
    """Return the declared schema, refusing anything this gwmock cannot read.

    This is the consumer-side half of the contract: it turns a file whose layout may have moved under
    the reader into a refusal at open time, rather than a wrong number later. The layout is checked
    rather than taken on trust, because a declaration a writer got wrong is worth exactly as little as
    no declaration at all -- and it is the check that would have caught the multichannel writer emitting
    the grid under a different name than the contract promised.

    Args:
        path: The artifact to check.

    Returns:
        The declared schema.

    Raises:
        ValueError: If the artifact declares no schema, declares a different one, declares a major
            version this gwmock does not know how to read, does not carry the layout it declares, or
            carries a grid that contradicts itself.
    """
    declared = read_strain_schema(path)
    if declared is None:
        raise ValueError(
            f"{Path(path)} declares no strain schema. It predates the declaration or was written by "
            "another producer; read its run metadata to learn its layout."
        )
    if declared.name != STRAIN_SCHEMA:
        raise ValueError(f"{Path(path)} declares schema '{declared.name}', not '{STRAIN_SCHEMA}'.")
    current_major = parse_strain_schema_version(STRAIN_SCHEMA_VERSION)[0]
    if declared.major != current_major:
        raise ValueError(
            f"{Path(path)} declares strain schema major version {declared.major}; this gwmock reads "
            f"major version {current_major} ({STRAIN_SCHEMA_VERSION})."
        )
    missing = missing_layout_attributes(path)
    if missing:
        detail = "; ".join(f"{name} is missing {', '.join(absent)}" for name, absent in sorted(missing.items()))
        raise ValueError(
            f"{Path(path)} declares strain schema {declared.version} but does not carry its layout: {detail}."
        )
    conflicts = conflicting_grid_attributes(path)
    if conflicts:
        detail = "; ".join(f"{name}: {', '.join(items)}" for name, items in sorted(conflicts.items()))
        raise ValueError(
            f"{Path(path)} declares strain schema {declared.version} but its grid contradicts itself: "
            f"{detail}. A file gwmock declared cannot reach this state; it has been edited since."
        )
    return declared


def _datasets(handle: Any) -> list[Any]:
    """Return every dataset in an open HDF5 file, at any depth.

    Args:
        handle: The open file.

    Returns:
        The datasets it holds.
    """
    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    found: list[Any] = []
    handle.visititems(lambda _, obj: found.append(obj) if isinstance(obj, h5py.Dataset) else None)
    return found


def _grid_conflicts(dataset: Any) -> list[str]:
    """Return a description of each grid quantity the dataset records twice, differently.

    Compared exactly rather than within a tolerance. The pair is either derived -- one written from the
    other, so bit-identical -- or written twice by one producer that had a single value in hand. There
    is no third case in which two epochs a hair apart would be the same epoch, so there is no bound to
    pick, and picking one would only decide how large a contradiction may be before it is reported.

    Args:
        dataset: The dataset to inspect.

    Returns:
        One entry per disagreement, empty when the dataset records each quantity once or consistently.
    """
    conflicts: list[str] = []
    for required, alias in _GRID_ALIASES:
        if required not in dataset.attrs or alias not in dataset.attrs:
            continue
        recorded, aliased = float(dataset.attrs[required]), float(dataset.attrs[alias])
        if recorded != aliased:
            conflicts.append(f"{required}={recorded!r} but {alias}={aliased!r}")
    return conflicts


def _check_dataset(dataset: Any, *, artifact: Path) -> None:
    """Refuse a dataset the 1.0.0 layout cannot be completed over, without writing anything.

    Kept apart from :func:`_complete_dataset` so that the caller can ask this of every dataset before
    writing to any of them.

    Args:
        dataset: The dataset to check.
        artifact: The file it belongs to, named in the errors below.

    Raises:
        ValueError: If the epoch or the sample interval is present under both spellings with values that
            disagree, or absent under both, leaving nothing to derive the required grid from.
    """
    disagreements = _grid_conflicts(dataset)
    if disagreements:
        raise ValueError(
            f"Cannot declare the strain schema on {artifact}: dataset '{dataset.name.lstrip('/')}' "
            f"records its grid twice and the two disagree ({'; '.join(disagreements)}). The content "
            "hash reads the first spelling and the multichannel reader the second, so declaring this "
            "file would put one artifact at two different times."
        )
    for required, alias in _GRID_ALIASES:
        if required not in dataset.attrs and alias not in dataset.attrs:
            raise ValueError(
                f"Cannot declare the strain schema on {artifact}: dataset '{dataset.name.lstrip('/')}' "
                f"carries neither '{required}' nor '{alias}', so the grid version "
                f"{STRAIN_SCHEMA_VERSION} requires cannot be recorded."
            )


def _complete_dataset(dataset: Any) -> None:
    """Fill in the attributes version 1.0.0 requires, from what the writer already recorded.

    Writes only: every reason to refuse has been raised by :func:`_check_dataset`, over every dataset in
    the file, before this is called on the first of them.

    Args:
        dataset: The dataset to complete.
    """
    for required, alias in _GRID_ALIASES:
        if required not in dataset.attrs:
            dataset.attrs[required] = float(dataset.attrs[alias])
    if "xunit" not in dataset.attrs:
        dataset.attrs["xunit"] = "s"
    channel = dataset.name.lstrip("/")
    for attribute in ("channel", "name"):
        if attribute not in dataset.attrs:
            dataset.attrs[attribute] = channel


def _as_text(value: object) -> str:
    """Decode an HDF5 attribute that may come back as bytes.

    Args:
        value: The raw attribute value.

    Returns:
        The attribute as text.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
