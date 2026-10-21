"""Utilities for reading and writing versioned metadata records.

A record is written twice: to its sidecar, which the producer keeps, and -- where the output format
has room for it -- into the data file itself, so that a file handed to another pipeline describes
itself. :func:`embed_metadata_record` is the only writer of the second copy, and the two copies come
from one record built once, so they cannot drift into disagreeing about the run.

They are not identical, and the difference is the point. The embedded copy leaves out the injection
parameters unless the run asked for them, because a blind mock data challenge is released as the data
files alone and those parameters are the answer the participants are asked to find; and it leaves out
the file hashes, which a file cannot carry about itself.

Both copies are, by default, the *shareable* record: :func:`split_metadata_record` takes the user
name, the host name, the exact platform string and every absolute path out of the record before it
is embedded or written, and keeps them in a private details file beside the sidecar. Embedding the
complete record instead would put exactly those details into every released data file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import secrets
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_logger = logging.getLogger("gwmock")

#: Two changes to the meaning of ``signal.injections`` ride on this constant, and neither altered the
#: shape of a record -- so a consumer reading an old version parses a new one without noticing, which
#: is exactly why the version has to move.
#:
#: 1.4.0: an event is attributed to the frame its waveform *starts* in rather than the frame holding
#: its coalescence.
#:
#: 1.5.0: ``injections`` lists every event **present** in the frame, including one generated for an
#: earlier segment, so a signal crossing a boundary now appears in each frame it reaches. The list
#: was already a list, so its *cardinality* changed without its type: a consumer that sums
#: ``injections`` across a run to count events will now overcount, and only ``schema_version`` says
#: so. ``signal_index.yaml`` changed shape in the same release.
#:
#: 1.6.0: ``noise.glitch_injections`` records every injected glitch -- its time, detector, class,
#: target and realized SNR and amplitude -- where a run previously recorded only the glitch
#: *configuration* and a per-model count. ``glitch_index.yaml`` arrives in the same release. A
#: record written by an older gwmock has no such key, which is not the same fact as a run that
#: injected no glitches; only ``schema_version`` tells the two apart.
#:
#: 1.7.0: segment epochs may advance by ``duration + segment-gap``, so a run's frames can be
#: discontiguous in GPS. ``simulator_metadata.orchestration.segment_layout`` records the layout
#: each batch's epoch comes from -- including the run's span and its analysed livetime, which
#: differ once gaps exist -- and ``signal.gap_excluded_injections`` and
#: ``signal.gap_discarded_injections`` record what the gaps cost the injected signals. A record
#: written by an older gwmock has none of these keys, which is not the same fact as a contiguous
#: run; only ``schema_version`` tells the two apart.
#:
#: 1.8.0: a record is written as two files by default -- the shareable record under the usual name
#: and a private details file beside it (see :func:`split_metadata_record`). The shareable record
#: has no ``author`` or ``email``, holds every path relative to the run's working directory (or as
#: a file name, for a path outside it), reduces ``host`` to the operating system, architecture,
#: Python version, git SHA and processor model, and names its companion under
#: :data:`PRIVATE_REFERENCE_KEY`. A consumer reading ``author`` or an absolute path from a 1.7.0
#: record finds neither here, and only ``schema_version`` says why. A run that opts out
#: (``globals.metadata.split: false``) writes the 1.7.0 layout under this version number.
SCHEMA_VERSION = "1.8.0"
_SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

#: The key the source parameters of the injected signals are recorded under. Removed from an embedded
#: copy at **every depth**, not only from ``signal.injections``: the orchestrator records the same list
#: three times over -- there, inside ``signal.metadata``, and again inside ``simulator_metadata`` --
#: because each of those blobs is a verbatim copy of what the orchestrator reported. A sanitiser that
#: removed the documented one would leave two intact, and a test written against the same documented
#: one would agree with it.
INJECTION_PARAMETERS_KEY = "injections"

#: The key holding the per-event glitch truth catalogue, dropped from an embedded copy alongside the
#: signal injections and for the same reason: a glitch's time, class and SNR are answers in a blind
#: challenge whose task is to find or veto transients, so a released data file must not carry them
#: unless the run said it is not blind.
#:
#: Named apart from the ``glitches`` key carrying the glitch *configuration* -- the models, rates and
#: target SNRs -- deliberately. That configuration is part of the config an embedded copy already
#: includes, and stripping every key called ``glitches`` at every depth would have taken it too while
#: reading as though it removed only the truth.
GLITCH_INJECTIONS_KEY = "glitch_injections"

#: The keys holding what a run's *gaps* did to the injected signals -- which catalogue events fell
#: wholly inside a gap and were therefore written nowhere, and how much of a signal crossing a gap
#: was discarded into it. Dropped from an embedded copy at **every depth**, alongside
#: :data:`INJECTION_PARAMETERS_KEY`, and for exactly the same reason: both carry the source
#: parameters of real events, and even without them "a signal was lost in the gap at this time" is
#: an answer in a blind challenge. A gapped run is precisely the one whose consumers are counting
#: livetime, so this is the record that would most tempt a release to leak.
GAP_INJECTION_KEYS = ("gap_excluded_injections", "gap_discarded_injections")

#: The key holding the simulator's RNG state, kept for replay. Dropped from an embedded copy at
#: **every depth**, for the same reason as the injections and then one more.
#:
#: The reasons it is not useful there: it is megabytes of array in a record that has to fit in a file
#: attribute, and the sidecar externalises it to ``.npy`` files an embedded copy could not point at.
#:
#: The reason the depth matters: ``gwmock merge`` copies each source's *whole* record under
#: ``source_files``, so a top-level-only drop left every merged artifact carrying its sources' RNG
#: state. Measured before this was fixed. That state, with the configuration beside it, regenerates
#: the run -- which is to say it regenerates the injections that the same document went to the
#: trouble of withholding.
_REPLAY_STATE_KEY = "pre_batch_state"

#: Top-level hash maps an embedded copy drops, and the per-output keys that duplicate them. A file
#: cannot carry its own hash: the digest is taken after the record is embedded, precisely so that what
#: the sidecar records describes the bytes that were published. Left in, they would be the hashes of a
#: file that no longer exists.
#:
#: Top-level only, deliberately, and unlike the two keys above. The nested ones a merge brings in
#: under ``source_files`` are the hashes of the *inputs*, which nothing in this file can invalidate:
#: they say which artifacts this one was made from, and that is provenance worth carrying rather than
#: a self-reference to remove.
_SELF_REFERENTIAL_KEYS = ("file_hashes", "content_hashes")
_SELF_REFERENTIAL_OUTPUT_KEYS = ("sha256", "content_sha256")


class SubpackageVersions(BaseModel):
    """Pinned public subpackage versions used to generate a run."""

    gwmock_signal: str | None = None
    gwmock_noise: str | None = None
    gwmock_pop: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class PopulationSection(BaseModel):
    """Population provenance stored in a metadata record."""

    backend: str
    source_type: str | None = None
    n_events: int | None = None
    parameter_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalSection(BaseModel):
    """Signal provenance stored in a metadata record."""

    backend: str
    waveform_model: str | None = None
    detector_network: list[str] = Field(default_factory=list)
    # Source parameters of every signal **present in** this batch's frame(s), in
    # injection order: [{"event_id": int, "parameters": {...}}]. That includes a signal
    # generated for an earlier segment whose content extends into this one, so one event
    # appears in the record of every frame it reaches. An event is *generated* for the
    # frame its waveform *starts* in -- for a compact binary that
    # is at or before the frame containing its coalescence, because the buffer begins
    # seconds earlier. Empty for stationary/SGWB segments.
    #
    # One gap, and it is a pre-existing data bug rather than a provenance one: spillover
    # chunks are not part of simulator state, so a run resumed from a checkpoint loses the
    # tail itself -- both the samples and the record. See
    # `gwmock/spillover-lost-on-checkpoint-resume`.
    injections: list[dict[str, Any]] = Field(default_factory=list)
    # Catalogue events that fall wholly inside a configured segment gap, so the run writes them to
    # no frame at all: [{"event_id": int, "parameters": {...}, "gap_start": float,
    # "gap_end": float}]. Recorded against the batch that consumed the event. Empty for a
    # contiguous run, and empty for a gapped run whose population happens to miss every gap --
    # which is not the same fact as a run that never checked, and is why the layout recorded under
    # `simulator_metadata.orchestration.segment_layout` says what the gaps were.
    gap_excluded_injections: list[dict[str, Any]] = Field(default_factory=list)
    # How much of a signal crossing into this segment the preceding gap swallowed:
    # [{"event_id": int, "parameters": {...}, "gap_start": float, "gap_end": float,
    # "discarded_samples": int, "discarded_seconds": float, "discarded_energy_fraction": float}].
    # The signal is still present in this frame -- its far-side content is placed at its true GPS
    # time -- so it appears in `injections` too; this says what did not survive the hole.
    gap_discarded_injections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoiseSection(BaseModel):
    """Noise provenance stored in a metadata record."""

    backend: str
    psd: str | None = None
    # Every glitch injected into this batch's noise output(s), in time order:
    # [{"event_id": str, "detector": str, "gps_start_time": float, ...}]. The counterpart of
    # `SignalSection.injections` for the other producer, and what makes a glitch frame set scorable
    # -- a detection-efficiency curve, a classifier's labels and a veto study all need the times and
    # classes, and none of them can be recovered from the strain once a Gaussian background sits on
    # top of it.
    #
    # A row is written for the batch a glitch *starts* in, so a waveform straddling a segment
    # boundary appears once, at its true time, rather than in every frame its samples reach. That is
    # deliberately unlike `SignalSection.injections`, which lists every event *present*: a signal's
    # parameters describe an event an analyst is looking for in this frame, while a glitch row is the
    # record of an injection, and counting rows across a run has to give the number injected.
    #
    # Empty for a run with no glitches -- and also for one whose installed gwmock-noise is too old to
    # report them, which `noise.metadata.glitch_catalogue` tells apart.
    glitch_injections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputRecord(BaseModel):
    """One generated artifact tracked by the metadata record."""

    kind: str
    path: str
    channels: list[str] = Field(default_factory=list)
    t0: float | int | None = None
    duration: float | int | None = None
    sha256: str | None = None
    content_sha256: str | None = None


class HostRecord(BaseModel):
    """Host fingerprint for provenance reporting.

    ``platform``, ``python``, ``cpu`` and ``git_sha`` are the fields every record has carried. In a
    shareable record ``platform`` is reduced to ``<os>-<architecture>`` -- the full string names the
    exact kernel build, which goes to the private details file -- and ``os``, ``architecture`` and
    ``cpu_model`` are added. ``cpu_model`` is written as an explicit ``null`` when the platform does
    not report one, so "unknown" is not confused with a record that predates the field.
    """

    platform: str
    python: str
    cpu: str
    git_sha: str | None = None
    os: str | None = None
    architecture: str | None = None
    cpu_model: str | None = None


class MetadataRecord(BaseModel):
    """Versioned metadata record written for each simulation batch."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    gwmock_version: str | None = None
    subpackage_versions: SubpackageVersions
    config: dict[str, Any]
    config_sha256: str
    seed: int | None = None
    segment_seeds: list[int] = Field(default_factory=list)
    population: PopulationSection | None = None
    signal: SignalSection | None = None
    noise: NoiseSection | None = None
    outputs: list[OutputRecord] = Field(default_factory=list)
    host: HostRecord
    # Full environment freeze (Python version + every installed distribution's
    # version) enabling exact-dependency reproduction into an isolated venv.
    # Null for records written before this field existed.
    environment: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Reject malformed versions and unknown major schema revisions."""
        match = _SCHEMA_VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError("schema_version must follow semantic versioning (MAJOR.MINOR.PATCH).")

        current_major = SCHEMA_VERSION.split(".", maxsplit=1)[0]
        if match.group(1) != current_major:
            raise ValueError(f"Unsupported metadata schema major version {match.group(1)}; expected {current_major}.")
        return value


def _normalize_json_value(value: Any) -> Any:
    """Convert Python objects to JSON-safe values.

    Non-finite floats become ``None``. JSON has no way to write one: ``json.dumps`` emits the bare
    tokens ``Infinity``, ``-Infinity`` and ``NaN``, which Python reads back but which are not JSON,
    so a strict parser rejects the whole document rather than the one field. That was survivable
    while the record only ever went to a sidecar Python read; it is not, now that a copy of it is
    written into the data file for another pipeline to parse.

    ``None`` rather than a string, because these are numeric fields and a consumer doing arithmetic
    on them should get "no value" rather than a type it did not expect. It is also the spelling the
    one reachable case already has on the way in: ``max_samples`` is ``np.inf`` exactly when the run
    asked for no limit, and ``Simulator(max_samples=None)`` is how that limit is *requested*, so the
    round trip lands back on its own input.

    The array branch recurses rather than returning ``tolist()`` directly, so a non-finite sample
    inside an array is normalised too.
    """
    normalized = value
    if isinstance(value, Path):
        normalized = str(value)
    elif hasattr(value, "unit") and hasattr(value, "value"):
        normalized = _normalize_json_value(value.value)
    elif isinstance(value, np.ndarray):
        normalized = _normalize_json_value(value.tolist())
    elif isinstance(value, np.generic):
        normalized = _normalize_json_value(value.item())
    elif isinstance(value, dict):
        normalized = {str(key): _normalize_json_value(val) for key, val in value.items()}
    elif isinstance(value, (list, tuple, set)):
        normalized = [_normalize_json_value(item) for item in value]
    if isinstance(normalized, float) and not math.isfinite(normalized):
        normalized = None
    return normalized


def _extract_external_state(metadata: dict[str, Any], metadata_file: Path, metadata_dir: Path) -> dict[str, Any]:
    """Replace raw numpy arrays in pre_batch_state with external file references."""
    metadata_copy = _normalize_json_value(metadata)
    if "pre_batch_state" not in metadata_copy:
        return metadata_copy

    external_state: dict[str, Any] = {}
    raw_state = metadata.get("pre_batch_state", {})
    if not isinstance(raw_state, dict):
        metadata_copy["pre_batch_state"] = _normalize_json_value(raw_state)
        return metadata_copy

    for key, value in raw_state.items():
        if type(value) is np.ndarray:  # pylint: disable=unidiomatic-typecheck
            state_file = f"{metadata_file.stem}_state_{key}.npy"
            np.save(metadata_dir / state_file, value)
            external_state[key] = {
                "_external_file": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size_bytes": value.nbytes,
                "file": state_file,
            }
        else:
            external_state[key] = _normalize_json_value(value)

    metadata_copy["pre_batch_state"] = external_state
    return metadata_copy


def save_metadata_with_external_state(
    metadata: dict[str, Any],
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Save metadata, extracting large numpy arrays in ``pre_batch_state`` to external ``.npy`` files."""
    metadata_file = Path(metadata_file)
    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_copy = _extract_external_state(metadata, metadata_file, metadata_dir)

    with metadata_file.open("w", encoding=encoding) as handle:
        if metadata_file.suffix.lower() == ".json":
            json.dump(metadata_copy, handle, indent=2, sort_keys=True)
            handle.write("\n")
        else:
            yaml.safe_dump(metadata_copy, handle, default_flow_style=False, sort_keys=False)


def load_metadata_with_external_state(
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Load metadata and reconstruct external numpy arrays from ``.npy`` files."""
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata_dir = metadata_file.parent if metadata_dir is None else Path(metadata_dir)

    try:
        with metadata_file.open("r", encoding=encoding) as handle:
            metadata = json.load(handle) if metadata_file.suffix.lower() == ".json" else yaml.safe_load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse metadata JSON: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse metadata YAML: {exc}") from exc

    if "pre_batch_state" in metadata:
        reconstructed_state = {}
        for key, value in metadata["pre_batch_state"].items():
            if isinstance(value, dict) and value.get("_external_file", False):
                state_file = metadata_dir / value["file"]
                if not state_file.exists():
                    raise FileNotFoundError(f"External state file not found: {state_file}")
                reconstructed_state[key] = np.load(state_file)
            else:
                reconstructed_state[key] = value
        metadata["pre_batch_state"] = reconstructed_state

    return metadata


def save_metadata_record(
    metadata: MetadataRecord | dict[str, Any],
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Validate and persist a versioned metadata record."""
    record = metadata if isinstance(metadata, MetadataRecord) else MetadataRecord.model_validate(metadata)
    document = record.model_dump(mode="python", by_alias=True, exclude_none=True)
    # Unknown is a value here, not an absence: a shareable host says "no model reported" as null.
    if "cpu_model" in record.host.model_fields_set and record.host.cpu_model is None:
        document["host"]["cpu_model"] = None
    save_metadata_with_external_state(
        metadata=_normalize_json_value(document),
        metadata_file=metadata_file,
        metadata_dir=metadata_dir,
        encoding=encoding,
    )


def load_metadata_record(
    metadata_file: Path | str,
    metadata_dir: Path | str | None = None,
    encoding: str = "utf-8",
) -> MetadataRecord:
    """Load and validate a versioned metadata record."""
    metadata = load_metadata_with_external_state(
        metadata_file=metadata_file, metadata_dir=metadata_dir, encoding=encoding
    )
    return MetadataRecord.model_validate(metadata)


def _without_key(document: Any, name: str) -> Any:
    """Return *document* with every entry called *name* removed, at any depth.

    The document is rebuilt rather than edited in place, because the caller's copy is the one that
    goes into the sidecar with everything intact.

    Args:
        document: The record, or any part of one.
        name: The key to remove wherever it appears.

    Returns:
        The same structure without that key.
    """
    if isinstance(document, dict):
        return {key: _without_key(value, name) for key, value in document.items() if key != name}
    if isinstance(document, list):
        return [_without_key(item, name) for item in document]
    return document


def without_injection_parameters(document: Any) -> Any:
    """Return *document* with the injection truth removed, at any depth.

    Both kinds of it: the signals' source parameters and the glitches' per-event catalogue. Which
    transients a released file holds is as much an answer to a blind challenge as the parameters of
    the signals in it, so the two are withheld together and released together.

    Depth matters: see :data:`INJECTION_PARAMETERS_KEY`.

    Args:
        document: The record, or any part of one.

    Returns:
        The same structure without either key.
    """
    document = _without_key(_without_key(document, INJECTION_PARAMETERS_KEY), GLITCH_INJECTIONS_KEY)
    for key in GAP_INJECTION_KEYS:
        document = _without_key(document, key)
    return document


def embeddable_metadata(metadata: dict[str, Any], *, include_injection_parameters: bool = False) -> dict[str, Any]:
    """Return the copy of a metadata record that may be written inside a data file.

    What it drops, and why, is in :data:`_REPLAY_STATE_KEY`, :data:`_SELF_REFERENTIAL_KEYS` and
    :data:`INJECTION_PARAMETERS_KEY`. Everything else the run recorded is kept, so the embedded copy
    still names the configuration, the seeds, the software versions and the outputs -- which is what
    makes the file self-describing, and which is also worth knowing before releasing one: a blind
    challenge whose population is *drawn* rather than loaded from a withheld file is reproducible from
    the configuration and the seed alone, whether or not the parameters themselves are embedded.

    Args:
        metadata: The record, as built for the sidecar.
        include_injection_parameters: Keep the source parameters of the injected signals. False by
            default, so a run has to say that its data is not a blind challenge.

    Returns:
        A JSON-safe copy of the record.
    """
    document = dict(metadata)
    for key in _SELF_REFERENTIAL_KEYS:
        document.pop(key, None)
    document = _without_key(document, _REPLAY_STATE_KEY)
    if not include_injection_parameters:
        document = without_injection_parameters(document)
    document = _normalize_json_value(document)
    outputs = document.get("outputs")
    if isinstance(outputs, list):
        document["outputs"] = [
            {key: value for key, value in record.items() if key not in _SELF_REFERENTIAL_OUTPUT_KEYS}
            if isinstance(record, dict)
            else record
            for record in outputs
        ]
    return document


def embed_metadata_record(
    file_path: Path | str,
    metadata: dict[str, Any],
    *,
    include_injection_parameters: bool = False,
) -> bool:
    """Write a run's metadata record into the data file it describes.

    The only writer of the embedded copy, so that one place decides what a released file may say about
    its run -- and so that a caller cannot embed a record by reaching past the exclusion of the
    injection parameters, which is the default and has to be asked out of.

    Call it after the samples are written and **before** the file is hashed. It changes the container
    bytes, so a hash taken first describes a file that is never published, and ``gwmock validate``
    would then report every output of every run as a byte mismatch.

    It is a no-op for the formats with nowhere to put a document -- ``.npy`` and ``.gwf``, exactly the
    ones :func:`gwmock.strain_schema.declare_strain_schema` skips -- so a caller writing several
    formats does not branch on the format.

    Args:
        file_path: The artifact to write into.
        metadata: The record, as built for the sidecar.
        include_injection_parameters: Embed the source parameters of the injected signals too.

    Returns:
        True if the record was embedded, False if the format cannot carry one.

    Raises:
        FileNotFoundError: If the artifact does not exist. Embedding describes an artifact that was
            already written; creating one here would publish a file holding metadata and no data.
    """
    from gwmock.strain_schema import (  # noqa: PLC0415  # deferred to keep the import graph acyclic
        RUN_METADATA_ATTRIBUTE,
        carries_strain_schema,
    )

    artifact = Path(file_path)
    if not carries_strain_schema(artifact):
        return False
    if not artifact.exists():
        raise FileNotFoundError(f"Cannot embed a metadata record in a file that does not exist: {artifact}")

    document = json.dumps(
        embeddable_metadata(metadata, include_injection_parameters=include_injection_parameters),
        sort_keys=True,
    )

    from gwmock_data import hdf5  # noqa: PLC0415

    hdf5.update_attrs(artifact, {"": {RUN_METADATA_ATTRIBUTE: document}})
    return True


# ---------------------------------------------------------------------------------------------------
# Shareable and private records
# ---------------------------------------------------------------------------------------------------

#: The key a shareable record names its private companion under: ``{"file": <name>, "sha256": <hex>}``.
PRIVATE_REFERENCE_KEY = "private_metadata"

#: The ``kind`` written at the top of every private details file, so a reader can tell one from a
#: record without relying on its name.
PRIVATE_METADATA_KIND = "gwmock-private-metadata"

#: Record fields that identify a person rather than describe a run. Removed from every shareable
#: record -- including the source records a merge nests under :data:`_NESTED_RECORDS_KEY` -- and
#: kept in the private file.
_PERSONAL_KEYS = ("author", "email")

#: The key under which ``gwmock merge`` nests whole source records. Each one is treated as a record in
#: its own right, so a merge of records written before the split, or with it turned off, does not
#: carry their authors and kernel builds into a shareable merged record.
_NESTED_RECORDS_KEY = "source_files"

#: A character that can sit *inside* a path or a word. A path token is only recognised where the
#: character before it is not one of these -- the start of the text, whitespace, or any punctuation
#: (``:``, ``,``, ``=``, a quote, a bracket, ...) -- so ``source:/home/x`` is found while
#: ``strain/Hz`` and ``relative/a.h5`` are not.
_PATH_BOUNDARY = r"(?<![A-Za-z0-9_.~\-/\\])"

#: Where a path token ends: whitespace, quotes, angle brackets and pipes, and the punctuation that
#: separates items in text (``,`` ``;`` ``:`` and closing brackets). A trailing ``.`` is kept in the
#: token and given back afterwards (see :meth:`_PathScrubber.scrub`).
_PATH_END = r"""\s"'<>|,;)\]}"""

#: Every path-like token in free text, in one pass so that each character belongs to one token:
#:
#: - ``uri``: anything ``scheme://...``. A ``file`` URI is reduced (with or without a host); every
#:   other scheme -- ``https`` above all -- is left as it is.
#: - ``unc``: a Windows UNC path, ``\\server\share\...``, reduced to its file name: the server is a
#:   machine name.
#: - ``drive``: a Windows drive path, ``C:\...`` or ``C:/...``.
#: - ``home``: ``~/...`` or ``~user/...``, a home directory, which is never inside the working
#:   directory as far as a shared record can tell, and names a user.
#: - ``posix``: an absolute POSIX path; a lone ``/`` (``1 / Hz``) is not one.
_PATH_TOKEN = re.compile(
    rf"""
    (?P<uri>\b[A-Za-z][A-Za-z0-9+.\-]+://[^\s"'<>]*)
  | {_PATH_BOUNDARY}(?P<unc>\\\\[^\s\\"'<>|]+\\[^{_PATH_END}]*)
  | {_PATH_BOUNDARY}(?P<drive>[A-Za-z]:[\\/][^{_PATH_END}]*)
  | {_PATH_BOUNDARY}(?P<home>~[A-Za-z0-9_.\-]*/[^{_PATH_END}:]*)
  | {_PATH_BOUNDARY}(?P<posix>/[^{_PATH_END}:/][^{_PATH_END}:]*)
    """,
    re.VERBOSE,
)

#: A whole value that is a home path: ``~``, ``~user``, or either followed by ``/...``.
_HOME_PATH = re.compile(r"~[A-Za-z0-9_.\-]*(?:/.*)?")


class SplitRecord(NamedTuple):
    """A metadata record divided into what may be shared and what stays with its producer."""

    shareable: dict[str, Any]
    private: dict[str, Any]


def private_metadata_path(metadata_file: Path | str) -> Path:
    """Return where the private details of the record in *metadata_file* are written.

    The record's own name with its ``.metadata.`` part read as ``.private.`` --
    ``orchestration-0.metadata.json`` becomes ``orchestration-0.private.json`` and a merge's
    ``merged.metadata.yaml`` becomes ``merged.private.yaml`` -- so the companion sits next to its record,
    sorts beside it, and is never matched by the ``*.metadata.json`` globs every reader uses to find
    records.

    Args:
        metadata_file: The shareable record's path.

    Returns:
        The private details file's path, in the same directory and the same format.
    """
    metadata_file = Path(metadata_file)
    name = metadata_file.name
    marker = ".metadata."
    position = name.rfind(marker)
    if position >= 0:
        private_name = f"{name[:position]}.private.{name[position + len(marker) :]}"
    else:
        private_name = f"{metadata_file.stem}.private{metadata_file.suffix}"
    return metadata_file.with_name(private_name)


def shareable_host(host: dict[str, Any], *, describes_this_machine: bool) -> dict[str, Any]:
    """Return the part of a host fingerprint that names a kind of machine rather than a machine.

    Kept: the Python version, the git SHA of the gwmock that ran, the ``cpu`` field as every record
    has carried it, and -- for a record this process is writing -- the operating system, the
    architecture and the processor model, which are what explain a numerical difference between two
    runs. ``platform`` is reduced to ``<os>-<architecture>``: the full string names the exact kernel
    build and C library, which identifies a particular installation.

    Args:
        host: The full fingerprint, as a record carries it.
        describes_this_machine: Whether *host* was taken on the machine running now. Only then can
            the operating system, architecture and processor model be read; a host nested in a merged
            source record keeps what it had, without ``platform``.

    Returns:
        The reduced fingerprint.
    """
    from gwmock.cli.utils.environment import cpu_model  # noqa: PLC0415  # deferred: keeps the import light

    reduced: dict[str, Any] = {
        key: host[key] for key in ("python", "cpu", "git_sha") if key in host and host[key] is not None
    }
    if describes_this_machine:
        system = platform.system() or "unknown"
        machine = platform.machine() or "unknown"
        reduced.update(
            {
                "platform": f"{system}-{machine}",
                "os": system,
                "architecture": machine,
                "cpu_model": cpu_model(),
            }
        )
        reduced.setdefault("python", platform.python_version())
        reduced.setdefault("cpu", machine)
    return reduced


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer:
        return []
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]


def _is_absolute_path(text: str) -> bool:
    """Return whether *text* is, as a whole, an absolute path: POSIX, Windows drive or UNC, or home."""
    if not text or "\n" in text:
        return False
    if text.startswith("/") and len(text) > 1:
        return True
    if _HOME_PATH.fullmatch(text):
        return True
    return PureWindowsPath(text).is_absolute() and bool(PureWindowsPath(text).drive)


class _PathScrubber:
    """Rewrite the absolute paths in a record against one base directory, remembering the originals."""

    def __init__(self, base_directory: Path | str | None) -> None:
        self.bases: list[PurePath] = []
        if base_directory is not None:
            base = Path(base_directory)
            if base.is_absolute():
                self.bases.append(base)
                try:
                    resolved = base.resolve()
                except OSError:
                    resolved = base
                if resolved != base:
                    self.bases.append(resolved)

    def relative(self, text: str) -> str:
        """Return *text*, an absolute path, relative to the base -- or its file name outside it.

        A home path (``~``, ``~user/...``) and a UNC path are always outside: the first names a user
        and the second a machine, and neither can be compared with the working directory.
        """
        if text.startswith("~"):
            name = PurePosixPath(text).name
            return "~" if name.startswith("~") or not name else name
        path: PurePath = PurePosixPath(text) if text.startswith("/") else PureWindowsPath(text)
        if not text.startswith("\\\\"):
            for base in self.bases:
                try:
                    relative = path.relative_to(base)
                except (ValueError, TypeError):
                    continue
                return relative.as_posix() or "."
        return path.name or "."

    def file_uri(self, text: str) -> str:
        """Return the shareable spelling of a ``file`` URI: the path it names, reduced.

        A URI naming a host other than this one (``file://server/...``) is reduced to the file name
        whatever its path, since the host name itself is the thing to withhold.
        """
        parsed = urlparse(text)
        location = unquote(parsed.path)
        if parsed.netloc not in ("", "localhost"):
            return PurePosixPath(location).name or "."
        return self.relative(location) if _is_absolute_path(location) else location

    def scrub(self, text: str) -> str:
        """Return *text* with every absolute path in it made relative (see :meth:`relative`)."""
        if "\n" not in text and text.lower().startswith("file://"):
            return self.file_uri(text)
        if _is_absolute_path(text):
            return self.relative(text)
        if not any(marker in text for marker in ("/", "\\", "~")):
            return text
        return _PATH_TOKEN.sub(self._token, text)

    def _token(self, match: re.Match[str]) -> str:
        kind = match.lastgroup
        token = match.group(kind or 0)
        body = token.rstrip(".")
        trailing = token[len(body) :]
        if kind == "uri":
            if not body.lower().startswith("file://"):
                return token
            return self.file_uri(body) + trailing
        return self.relative(body) + trailing


class _Splitter:
    """One pass over a record, building the shareable copy and the private remainder together."""

    def __init__(self, base_directory: Path | str | None) -> None:
        self.default_base = base_directory
        self.values: dict[str, Any] = {}
        self.renamed_keys: list[dict[str, str]] = []

    def record(self, document: dict[str, Any], pointer: str, scrubber: _PathScrubber, *, root: bool) -> dict[str, Any]:
        """Split one record: the root, or a source record a merge nests inside it."""
        own_base = _record_working_directory(document)
        if own_base is not None and Path(own_base).is_absolute():
            scrubber = _PathScrubber(own_base)
        shareable: dict[str, Any] = {}
        for key, value in document.items():
            if key in _PERSONAL_KEYS:
                self.values[f"{pointer}/{_escape_pointer_token(key)}"] = value
                continue
            if key == "host" and isinstance(value, dict):
                self.values[f"{pointer}/host"] = value
                shareable[key] = shareable_host(value, describes_this_machine=root)
                continue
            child_pointer = f"{pointer}/{_escape_pointer_token(key)}"
            if key == _NESTED_RECORDS_KEY and isinstance(value, dict):
                shareable[key] = self.nested_records(value, child_pointer, scrubber)
                continue
            shareable[key] = self.value(value, child_pointer, scrubber)
        return shareable

    def nested_records(self, records: dict[str, Any], pointer: str, scrubber: _PathScrubber) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, new_key in self.keys(records, pointer, scrubber):
            child_pointer = f"{pointer}/{_escape_pointer_token(new_key)}"
            value = records[key]
            result[new_key] = (
                self.record(value, child_pointer, scrubber, root=False)
                if isinstance(value, dict)
                else self.value(value, child_pointer, scrubber)
            )
        return result

    def keys(self, mapping: dict[Any, Any], pointer: str, scrubber: _PathScrubber) -> list[tuple[Any, Any]]:
        """Pair each key with its shareable spelling, keeping keys distinct if two collapse to one name."""
        taken: set[Any] = set()
        pairs: list[tuple[Any, Any]] = []
        for key in mapping:
            new_key = key
            if isinstance(key, str):
                new_key = _distinct(scrubber.scrub(key), taken)
            taken.add(new_key)
            if new_key != key:
                self.renamed_keys.append({"object": pointer, "key": str(new_key), "original": str(key)})
            pairs.append((key, new_key))
        return pairs

    def value(self, value: Any, pointer: str, scrubber: _PathScrubber) -> Any:
        if isinstance(value, dict):
            return {
                new_key: self.value(value[key], f"{pointer}/{_escape_pointer_token(str(new_key))}", scrubber)
                for key, new_key in self.keys(value, pointer, scrubber)
            }
        if isinstance(value, list):
            return [self.value(item, f"{pointer}/{index}", scrubber) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return [self.value(item, f"{pointer}/{index}", scrubber) for index, item in enumerate(value)]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            scrubbed = scrubber.scrub(value)
            if scrubbed != value:
                self.values[pointer] = value
            return scrubbed
        return value


def _distinct(name: str, taken: set[Any]) -> str:
    """Return *name*, or -- if another key already has it -- the first free ``<stem>~<n><suffix>``.

    Two keys can reduce to one file name (``/home/a/x.h5`` and ``/opt/b/x.h5`` are both ``x.h5``).
    They are told apart by a counter rather than by a longer path, because any path that tells them
    apart is a path out of the working directory -- and would name the directory, perhaps a home
    directory, that the reduction was there to withhold. The originals are in the private file.
    """
    if name not in taken:
        return name
    head, separator, last = name.rpartition("/")
    stem, dot, suffix = last.partition(".") if not last.startswith(".") else (last, "", "")
    counter = 2
    while True:
        candidate = f"{head}{separator}{stem}~{counter}{dot}{suffix}"
        if candidate not in taken:
            return candidate
        counter += 1


def _record_working_directory(document: dict[str, Any]) -> str | None:
    """Return the working directory a record's own configuration names, if it names one."""
    for container in (document.get("config"), {"globals": document.get("globals_config")}):
        if isinstance(container, dict):
            globals_section = container.get("globals")
            if isinstance(globals_section, dict):
                directory = globals_section.get("working-directory") or globals_section.get("working_directory")
                if isinstance(directory, str):
                    return directory
    return None


def split_metadata_record(
    metadata: dict[str, Any],
    *,
    base_directory: Path | str | None,
    record_file_name: str,
) -> SplitRecord:
    """Divide a metadata record into a shareable record and a private details document.

    The shareable record is the one data sets travel with. It differs from the complete record in
    three ways, and in no other:

    - ``author`` and ``email`` are removed, at the top and in every source record a merge nests.
    - Every absolute path -- a value, a mapping key, or a path inside free text -- is rewritten
      relative to *base_directory* (the run's working directory), or to its file name when it lies
      outside it. ``config.globals.working-directory`` therefore reads ``.``. A nested source record
      is rewritten against its own working directory when it names an absolute one.
    - ``host`` is reduced as :func:`shareable_host` describes.

    Everything removed or rewritten is recorded in the private document as the original value at a
    JSON pointer (RFC 6901) into the shareable record, and every renamed mapping key likewise, so
    :func:`combine_metadata_records` rebuilds the complete record exactly. The private document also
    names the author, the hostname, the full platform string and the working directory at its top
    level, for a reader rather than a program. It carries a random ``nonce``, because the shareable
    record publishes the private file's SHA-256 and the private contents -- a user name, a host name,
    a directory layout -- are few enough to guess; the nonce makes that hash useless for confirming a
    guess.

    Hashes are untouched: ``file_hashes`` and ``content_hashes`` are keyed by file name and
    ``outputs[].path`` is already relative, so everything gwmock checks from a record works from the
    shareable one alone.

    Args:
        metadata: The complete record. Not modified.
        base_directory: The absolute working directory paths are made relative to; ``None`` or a
            relative path leaves absolute paths to be reduced to their file names.
        record_file_name: The shareable record's file name, which the private document names.

    Returns:
        The shareable record and the private document.
    """
    splitter = _Splitter(base_directory)
    shareable = splitter.record(metadata, "", _PathScrubber(base_directory), root=True)
    host = metadata.get("host") if isinstance(metadata.get("host"), dict) else {}
    from gwmock.cli.utils.environment import hostname  # noqa: PLC0415

    private: dict[str, Any] = {
        "kind": PRIVATE_METADATA_KIND,
        "schema_version": SCHEMA_VERSION,
        "describes": record_file_name,
        "nonce": secrets.token_hex(16),
        "author": metadata.get("author"),
        "email": metadata.get("email"),
        "hostname": hostname(),
        "platform": host.get("platform"),
        "working_directory": None if base_directory is None else str(base_directory),
        "values": _normalize_json_value(splitter.values),
        "renamed_keys": splitter.renamed_keys,
    }
    return SplitRecord(shareable=shareable, private=private)


def _copy_containers(document: Any) -> Any:
    """Copy the dicts and lists of *document*, sharing everything else (arrays included)."""
    if isinstance(document, dict):
        return {key: _copy_containers(value) for key, value in document.items()}
    if isinstance(document, list):
        return [_copy_containers(item) for item in document]
    return document


def combine_metadata_records(shareable: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the complete record from a shareable record and its private details.

    The inverse of :func:`split_metadata_record`: every value the split removed or rewrote is put
    back at its pointer, every renamed key is given its original name, and the reference to the
    private file is dropped -- so the result has the layout a run writes with the split turned off.

    Args:
        shareable: The shareable record, as read from its file.
        private: Its private details document.

    Returns:
        The complete record. Neither input is modified.

    Raises:
        ValueError: If *private* is not a private details document.
    """
    if not isinstance(private, dict) or private.get("kind") != PRIVATE_METADATA_KIND:
        raise ValueError("Not a gwmock private metadata document.")
    document = _copy_containers(shareable)
    document.pop(PRIVATE_REFERENCE_KEY, None)
    for pointer, value in (private.get("values") or {}).items():
        tokens = _pointer_tokens(pointer)
        if not tokens:
            continue
        container: Any = document
        for token in tokens[:-1]:
            container = container[int(token)] if isinstance(container, list) else container.setdefault(token, {})
        last = tokens[-1]
        if isinstance(container, list):
            container[int(last)] = value
        else:
            container[last] = value
    for rename in reversed(private.get("renamed_keys") or []):
        container = document
        for token in _pointer_tokens(rename["object"]):
            container = container[int(token)] if isinstance(container, list) else container[token]
        if rename["key"] in container:
            container[rename["original"]] = container.pop(rename["key"])
    return document


def _serialise(document: dict[str, Any], path: Path) -> bytes:
    if path.suffix.lower() == ".json":
        return (json.dumps(_normalize_json_value(document), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return yaml.safe_dump(_normalize_json_value(document), default_flow_style=False, sort_keys=False).encode("utf-8")


def write_private_metadata(private: dict[str, Any], private_file: Path | str) -> str:
    """Write a private details document, returning the SHA-256 (hex) of the bytes written.

    JSON or YAML by the file's suffix, as for a record. Written to a temporary name and renamed, so a
    reader never meets a half-written file under the name the shareable record will point at.

    Args:
        private: The document from :func:`split_metadata_record`.
        private_file: Where to write it.

    Returns:
        The hex digest the shareable record records under :data:`PRIVATE_REFERENCE_KEY`.
    """
    private_file = Path(private_file)
    private_file.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialise(private, private_file)
    temporary = private_file.with_name(f".{private_file.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(private_file)
    return hashlib.sha256(payload).hexdigest()


def private_reference(private_file: Path | str, digest: str) -> dict[str, str]:
    """Return the value a shareable record holds under :data:`PRIVATE_REFERENCE_KEY`."""
    return {"file": Path(private_file).name, "sha256": digest}


def load_private_metadata(metadata_file: Path | str, shareable: dict[str, Any]) -> dict[str, Any] | None:
    """Return the private details of a shareable record, when they are beside it and intact.

    ``None`` -- and the caller carries on with the shareable record -- when the record names no
    private file, the file is absent (the usual case for anyone the data was shared with), or its
    SHA-256 is not the one the record names. A mismatch is logged as a warning: a private file that
    is not the one written with this record would put back the wrong paths.

    Args:
        metadata_file: The shareable record's path; the private file is looked for beside it.
        shareable: The shareable record, as read.

    Returns:
        The private document, or ``None``.
    """
    reference = shareable.get(PRIVATE_REFERENCE_KEY) if isinstance(shareable, dict) else None
    name = reference.get("file") if isinstance(reference, dict) else None
    # A bare name only: a record is shared data, and must not be able to point a reader elsewhere.
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    record_directory = Path(metadata_file).parent
    private_file = record_directory / name
    if not private_file.is_file():
        return None
    # A bare name can still be a symbolic link to anywhere; what is read must be in this directory.
    if private_file.resolve().parent != record_directory.resolve():
        _logger.warning("Ignoring %s: it resolves outside the directory of its record.", private_file)
        return None
    payload = private_file.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.get("sha256"):
        _logger.warning(
            "Ignoring %s: its SHA-256 is not the one %s records, so it does not describe this record.",
            private_file,
            Path(metadata_file).name,
        )
        return None
    try:
        document = json.loads(payload) if private_file.suffix.lower() == ".json" else yaml.safe_load(payload)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError):
        document = None
    is_private = isinstance(document, dict) and document.get("kind") == PRIVATE_METADATA_KIND
    return document if is_private else None


def with_private_details(metadata_file: Path | str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the complete record when its private details are available, else *metadata* itself.

    For the readers that act on the producer's own machine -- reproduction and checkpoint resume --
    where the original absolute paths are the ones to use. Every reader works without them.

    Args:
        metadata_file: The record's path.
        metadata: The record, as read.

    Returns:
        The combined record, or *metadata* unchanged.
    """
    private = load_private_metadata(metadata_file, metadata)
    return metadata if private is None else combine_metadata_records(metadata, private)
