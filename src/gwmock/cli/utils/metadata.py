"""Utilities for reading and writing versioned metadata records.

A record is written twice: to its sidecar, which the producer keeps, and -- where the output format
has room for it -- into the data file itself, so that a file handed to another pipeline describes
itself. :func:`embed_metadata_record` is the only writer of the second copy, and the two copies come
from one record built once, so they cannot drift into disagreeing about the run.

They are not identical, and the difference is the point. The embedded copy leaves out the injection
parameters unless the run asked for them, because a blind mock data challenge is released as the data
files alone and those parameters are the answer the participants are asked to find; and it leaves out
the file hashes, which a file cannot carry about itself.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
SCHEMA_VERSION = "1.5.0"
_SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

#: The key the source parameters of the injected signals are recorded under. Removed from an embedded
#: copy at **every depth**, not only from ``signal.injections``: the orchestrator records the same list
#: three times over -- there, inside ``signal.metadata``, and again inside ``simulator_metadata`` --
#: because each of those blobs is a verbatim copy of what the orchestrator reported. A sanitiser that
#: removed the documented one would leave two intact, and a test written against the same documented
#: one would agree with it.
INJECTION_PARAMETERS_KEY = "injections"

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
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoiseSection(BaseModel):
    """Noise provenance stored in a metadata record."""

    backend: str
    psd: str | None = None
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
    """Host fingerprint for provenance reporting."""

    platform: str
    python: str
    cpu: str
    git_sha: str | None = None


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
    save_metadata_with_external_state(
        metadata=_normalize_json_value(record.model_dump(mode="python", by_alias=True, exclude_none=True)),
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
    """Return *document* with every ``injections`` entry removed, at any depth.

    Depth matters: see :data:`INJECTION_PARAMETERS_KEY`.

    Args:
        document: The record, or any part of one.

    Returns:
        The same structure without any ``injections`` key.
    """
    return _without_key(document, INJECTION_PARAMETERS_KEY)


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

    import h5py  # noqa: PLC0415  # deferred so importing the record does not pull in the HDF5 stack

    with h5py.File(artifact, "a") as handle:
        handle.attrs[RUN_METADATA_ATTRIBUTE] = document
    return True
