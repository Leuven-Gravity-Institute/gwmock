"""
Utilities for executing simulation plans via CLI.
"""

from __future__ import annotations

import atexit
import copy
import errno
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from functools import cache
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, NamedTuple, NoReturn, cast

import numpy as np
import yaml
from gwmock_noise import SimulationResult
from tqdm import tqdm

from gwmock.cli.adapter_orchestration import AdapterOrchestrationResult, AdapterOrchestrator
from gwmock.cli.utils.checkpoint import (
    CheckpointManager,
    report_unverified_inputs,
    require_matching_config,
    resume_tail,
    run_fingerprint,
)
from gwmock.cli.utils.config import OrchestrationConfig, SimulatorConfig, resolve_class_path
from gwmock.cli.utils.environment import capture_environment
from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash
from gwmock.cli.utils.metadata import embed_metadata_record, save_metadata_record
from gwmock.cli.utils.simulation_plan import (
    SimulationBatch,
    SimulationPlan,
    create_batch_metadata,
)
from gwmock.cli.utils.template import expand_template_variables
from gwmock.cli.utils.truth_index import GLITCH_INDEX, SIGNAL_INDEX, TRUTH_INDEXES, IndexSpec
from gwmock.cli.utils.utils import handle_signal
from gwmock.simulator.base import Simulator

logger = logging.getLogger("gwmock")

# A full git commit SHA (SHA-1): the only revision form that immutably pins a
# downloaded dataset. Branches, tags, and None can all move upstream.
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
logger.setLevel(logging.DEBUG)


def _backend_path_from_object(obj: Any) -> str:
    """Return a stable ``module:qualname`` identifier for an object or class."""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _flatten_to_strings(value: Any) -> list[str]:
    """Flatten template-expanded values into a simple ordered list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.flatten().tolist()]
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_to_strings(item))
        return flattened
    return [str(value)]


def _to_path_string(path: Path, working_directory: str | None) -> str:
    """Prefer working-directory-relative paths for portable metadata."""
    if working_directory:
        base = Path(working_directory)
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)
    return str(path)


def _to_plain_number(value: Any) -> float | int | None:
    """Convert quantities and numpy scalars to native numbers."""
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else float(value)
    return value


def _get_host_metadata() -> dict[str, Any]:
    """Collect stable host metadata for provenance reporting."""
    git_sha = _get_distribution_git_sha() or _get_source_tree_git_sha()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "git_sha": git_sha,
    }


def _get_distribution_git_sha() -> str | None:
    """Return the gwmock distribution VCS commit hash when explicitly available."""
    try:
        distribution = importlib_metadata.distribution("gwmock")
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None

    for source in (distribution.read_text("direct_url.json"), distribution.metadata.get("Direct-URL")):
        git_sha = _extract_git_sha_from_direct_url(source)
        if git_sha is not None:
            return git_sha
    return None


def _get_source_tree_git_sha() -> str | None:
    """Return the working-tree git commit for a source checkout, else ``None``.

    Complements :func:`_get_distribution_git_sha`: editable/source installs carry no
    PEP 610 VCS metadata, so read the commit directly from the repository that
    contains this module. A ``-dirty`` suffix marks an uncommitted working tree, so a
    downstream lineage system can tell the output was not built from a clean commit.
    Returns ``None`` when git is unavailable or the source is not a repository (e.g. a
    released wheel unpacked into site-packages).
    """
    git_exe = shutil.which("git")
    if git_exe is None:
        return None
    repo_dir = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(  # noqa: S603
            [git_exe, "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return None
        sha = head.stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git_exe, "-C", repo_dir, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            sha = f"{sha}-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_git_sha_from_direct_url(direct_url: str | None) -> str | None:
    """Parse a commit hash from PEP 610 Direct URL metadata content."""
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return None
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    if isinstance(commit_id, str):
        commit_id = commit_id.strip()
        return commit_id or None
    return None


def _build_config_payload(batch: SimulationBatch, simulator: Simulator) -> dict[str, Any]:
    """Build the resolved config snapshot stored in metadata."""
    base_payload = (
        copy.deepcopy(batch.config_payload)
        if batch.config_payload is not None
        else {
            "globals": batch.globals_config.model_dump(by_alias=True, exclude_none=True),
        }
    )

    if isinstance(batch.simulator_config, OrchestrationConfig):
        base_payload["orchestration"] = batch.simulator_config.model_dump(by_alias=True, exclude_none=True)
    else:
        simulators = base_payload.setdefault("simulators", {})
        simulators[batch.simulator_name] = batch.simulator_config.model_dump(by_alias=True, exclude_none=True)

    return cast(dict[str, Any], expand_template_variables(base_payload, simulator))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in place.

    Nested mappings merge key-by-key; every other value (including lists)
    replaces wholesale, so a resolved ``glitches`` list supersedes the input
    one rather than being concatenated with it.
    """
    for key, value in override.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value


def _unresolved_external_inputs(fragment: dict[str, Any]) -> list[str]:
    """Return labels for resolved entries not pinned to an immutable version.

    A glitch entry that carries a ``revision`` key names a dataset-backed model.
    It is only reproducible when that revision is a full commit SHA: ``None``
    means resolution failed (e.g. an offline Hub with no cache), and a symbolic
    ref such as a branch or tag (``"main"``) still moves upstream. Both are
    reported as unresolved so the run is marked non-replayable.
    """
    unresolved: list[str] = []
    noise_arguments = fragment.get("noise", {}).get("arguments", {})
    for entry in noise_arguments.get("glitches", []) or []:
        if isinstance(entry, dict) and "revision" in entry and not _is_pinned_revision(entry["revision"]):
            unresolved.append(f"glitch:{entry.get('kind', 'unknown')}")
    return unresolved


def _is_pinned_revision(revision: Any) -> bool:
    """Return whether a dataset revision is an immutable full commit SHA.

    A 40-character hex string is a git commit SHA and cannot move; anything else
    — ``None`` or a symbolic ref like a branch or tag — can point at different
    content later, so it does not pin the run.
    """
    return isinstance(revision, str) and _COMMIT_SHA_RE.fullmatch(revision) is not None


def _build_resolved_config(
    simulator: Simulator,
    input_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Build the fully-resolved, replayable config for this batch.

    Overlays each adapter's runtime-resolved values (e.g. a pinned dataset
    revision) onto the template-expanded input config. Returns
    ``(resolved_payload, replayable)`` — ``resolved_payload`` is ``None`` when
    nothing needed resolving (a purely parametric run), and ``replayable`` is
    ``False`` when a declared external-mutable input could not be pinned.
    """
    resolved_config_fn = getattr(simulator, "resolved_config", None)
    if not callable(resolved_config_fn):
        return None, True
    fragment = cast(dict[str, Any], resolved_config_fn())
    if not fragment:
        return None, True

    orchestration = input_payload.get("orchestration")
    if not isinstance(orchestration, dict):
        return None, True

    unresolved = _unresolved_external_inputs(fragment)
    if unresolved:
        logger.warning(
            "Could not pin external-mutable input(s) %s to an immutable version; "
            "this run's metadata is marked non-replayable and is not bit-reproducible.",
            ", ".join(unresolved),
        )

    resolved_payload = copy.deepcopy(input_payload)
    _deep_merge(resolved_payload["orchestration"], fragment)
    return resolved_payload, not unresolved


def _resolve_seed(simulator: Simulator, batch: SimulationBatch) -> int | None:
    """Resolve the top-level seed recorded for this batch."""
    if isinstance(simulator, AdapterOrchestrator):
        seed = simulator.noise_arguments.get("seed")
        return int(seed) if seed is not None else None

    seed = getattr(simulator, "seed", None)
    if seed is not None:
        return int(seed)

    global_seed = batch.globals_config.simulator_arguments.get("seed")
    if global_seed is not None:
        return int(global_seed)

    local_seed = getattr(batch.simulator_config, "arguments", {}).get("seed")
    return int(local_seed) if local_seed is not None else None


def _resolve_segment_seeds(simulator: Simulator, batch: SimulationBatch, seed: int | None) -> list[int]:
    """Resolve per-segment seeds for this batch."""
    if seed is None:
        return []
    if isinstance(simulator, AdapterOrchestrator):
        return simulator.segment_seeds()
    return [seed + batch.batch_index]


def _build_population_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the population section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.population is None:
            return None
        return {
            "backend": batch.simulator_config.population.backend,
            "source_type": simulator_metadata["orchestration"]["source_type"],
            "n_events": len(simulator._population_events),
            "parameter_names": list(simulator._population_events[0].keys()) if simulator._population_events else [],
            "metadata": simulator_metadata["orchestration"]["population"]["metadata"],
        }

    signal_metadata = simulator_metadata.get("signal", {}).get("arguments", {})
    source_type = signal_metadata.get("source_type")
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "source_type": source_type,
        "n_events": None,
        "parameter_names": [],
        "metadata": {},
    }


def _build_signal_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the signal section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.signal is None or simulator.signal_adapter is None:
            return None
        return {
            "backend": _backend_path_from_object(simulator.signal_adapter._backend),
            "waveform_model": simulator.waveform_model,
            "detector_network": list(simulator.detectors),
            # Source parameters of the signals that merge in this batch's frame(s),
            # in injection order (empty for stationary/SGWB segments). This makes each
            # frame self-describing and backs the signal->frame lookup.
            "injections": list(simulator_metadata["orchestration"]["signal"].get("injections", [])),
            "metadata": simulator_metadata["orchestration"]["signal"],
        }

    signal_metadata = simulator_metadata.get("signal", {}).get("arguments", {})
    detectors = signal_metadata.get("detectors", getattr(simulator, "detectors", []))
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "waveform_model": signal_metadata.get("waveform_model"),
        "detector_network": [str(detector) for detector in detectors],
        "metadata": simulator_metadata,
    }


def _build_noise_section(simulator: Simulator, batch: SimulationBatch) -> dict[str, Any] | None:
    """Build the noise section for the metadata schema."""
    simulator_metadata = simulator.metadata
    if isinstance(simulator, AdapterOrchestrator):
        if batch.simulator_config.noise is None or simulator.noise_adapter is None:
            return None
        psd_value = simulator.noise_arguments.get("psd_file")
        if psd_value is None and simulator.noise_arguments.get("psd_files"):
            psd_value = "multiple"
        return {
            "backend": _backend_path_from_object(simulator.noise_adapter.backend),
            "psd": None if psd_value is None else str(psd_value),
            # Every glitch this batch's noise output(s) carry, in time order -- the counterpart
            # of `signal.injections`, and what backs the glitch->frame lookup. Empty for a run
            # without glitches.
            "glitch_injections": list(simulator_metadata["orchestration"]["noise"].get("glitch_injections") or []),
            "metadata": simulator_metadata["orchestration"]["noise"],
        }

    noise_metadata = simulator_metadata.get("colored_noise", {}).get("arguments", {})
    return {
        "backend": resolve_class_path(batch.simulator_config.class_, batch.simulator_name),
        "psd": noise_metadata.get("psd_file"),
        "metadata": simulator_metadata,
    }


class _OutputArtifact(NamedTuple):
    """One generated file and the descriptor the metadata record holds it under.

    The two travel together because the descriptor's hashes cannot be filled in when it is built.
    They are taken after the record has been embedded in the artifact -- embedding changes the
    container bytes -- and the path the digest is read from has to be the one the descriptor names,
    not a path re-derived from the string it stores.
    """

    path: Path
    record: dict[str, Any]


def _build_output_records(
    simulator: Simulator,
    batch: SimulationBatch,
    batch_data: object,
    output_files: list[Path],
) -> list[_OutputArtifact]:
    """Build output descriptors for the versioned metadata schema, without their hashes.

    See :func:`_hash_output_records` for why the hashes are absent here.
    """
    working_directory = batch.globals_config.working_directory
    output_records: list[_OutputArtifact] = []

    if isinstance(batch_data, AdapterOrchestrationResult):
        if batch.simulator_config.signal is not None and batch_data.signal_segment is not None:
            signal_files = _resolve_output_paths(
                file_name_template=batch.simulator_config.signal.output.file_name,
                simulator=simulator,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
            )
            signal_channels = _flatten_to_strings(
                expand_template_variables(batch.simulator_config.signal.output.arguments.get("channel"), simulator)
            )
            for index, output_file in enumerate(signal_files):
                output_records.append(
                    _OutputArtifact(
                        output_file,
                        {
                            "kind": "signal",
                            "path": _to_path_string(output_file, working_directory),
                            "channels": signal_channels[index : index + 1] if signal_channels else [],
                            "t0": _to_plain_number(batch_data.signal_segment.start_time),
                            "duration": _to_plain_number(batch_data.signal_segment.duration),
                        },
                    )
                )

        if batch_data.noise_result is not None:
            noise_output_config = batch_data.noise_result.config.output
            for detector, output_path in batch_data.noise_result.output_paths.items():
                if noise_output_config.channels and detector in noise_output_config.channels:
                    channel_id = noise_output_config.channels[detector]
                else:
                    channel_id = f"{detector}:{noise_output_config.channel}"
                output_records.append(
                    _OutputArtifact(
                        output_path,
                        {
                            "kind": "noise",
                            "path": _to_path_string(output_path, working_directory),
                            "channels": [channel_id],
                            "t0": _to_plain_number(simulator.start_time),
                            "duration": _to_plain_number(simulator.duration),
                        },
                    )
                )
        return output_records

    if isinstance(batch_data, SimulationResult):
        channel_prefix = str(getattr(simulator, "_active_channel_prefix", "MOCK"))
        for detector, output_path in batch_data.output_paths.items():
            output_records.append(
                _OutputArtifact(
                    output_path,
                    {
                        "kind": batch.simulator_name,
                        "path": _to_path_string(output_path, working_directory),
                        "channels": [f"{detector}:{channel_prefix}"],
                        "t0": _to_plain_number(getattr(simulator, "start_time", None)),
                        "duration": _to_plain_number(getattr(simulator, "duration", None)),
                    },
                )
            )
        return output_records

    expanded_arguments = expand_template_variables(batch.simulator_config.output.arguments or {}, simulator)
    channels = _flatten_to_strings(expanded_arguments.get("channel"))
    for index, output_file in enumerate(output_files):
        output_records.append(
            _OutputArtifact(
                output_file,
                {
                    "kind": batch.simulator_name,
                    "path": _to_path_string(output_file, working_directory),
                    "channels": channels[index : index + 1] if channels else [],
                    "t0": _to_plain_number(getattr(batch_data, "start_time", getattr(simulator, "start_time", None))),
                    "duration": _to_plain_number(getattr(batch_data, "duration", getattr(simulator, "duration", None))),
                },
            )
        )
    return output_records


def _hash_output_records(output_records: list[_OutputArtifact]) -> None:
    """Fill in each descriptor's hashes, from the artifact as it now stands on disk.

    Separate from :func:`_build_output_records`, and called later, because the run embeds its
    metadata record in the artifacts between the two: hashing first would record the digest of a
    file that is never published, and `gwmock validate` would then report a byte mismatch for every
    output of every run.

    Args:
        output_records: The artifacts and their descriptors, mutated in place.
    """
    for artifact in output_records:
        artifact.record["sha256"] = compute_file_hash(artifact.path)
        artifact.record["content_sha256"] = compute_content_hash(artifact.path)


def _includes_injection_parameters(batch: SimulationBatch) -> bool:
    """Return whether this batch's run asked for its injection parameters in its data files.

    False for anything but an orchestration config, which is the only schema carrying the flag --
    and the only one that has injection parameters to withhold.

    Args:
        batch: The batch being recorded.

    Returns:
        Whether the source parameters may be embedded in the outputs.
    """
    if isinstance(batch.simulator_config, OrchestrationConfig):
        return bool(batch.simulator_config.include_injection_parameters)
    return False


class StaleIndexReadError(RuntimeError):
    """The index this process read is not the one the sidecar says is current.

    Raised instead of writing, because writing is what discards the entries this process cannot
    see. The usual cause is a client cache on a shared filesystem: the lock is held correctly and
    the index is still read from a stale view, since acquiring the lock revalidates the sidecar
    rather than the index.
    """


class IndexRebuildError(RuntimeError):
    """A truth index could not be rebuilt from the batch metadata files.

    Raised instead of writing a partial index. A rebuild's whole value is that its result is
    complete -- it is the repair for an index that lost entries -- so a source file it cannot read
    has to stop it rather than quietly shrink what it produces.
    """


#: The name this error carried while the signal index was the only one. Kept because it is caught
#: by name outside this module.
SignalIndexRebuildError = IndexRebuildError


class PartialIndexRebuildError(RuntimeError):
    """Some truth indexes were rebuilt and at least one was not.

    A run writes one index per catalogue, and two files cannot be replaced as a single atomic
    act. The failure that *can* be removed is removed -- see :func:`rebuild_truth_indexes`,
    which validates every index's sources before publishing any of them -- so what reaches this
    is a write that failed part-way: a full disk, a revoked permission. Raised rather than
    letting the original error through unqualified, because "the command failed" and "the
    command failed after replacing one of the two indexes, and re-recording its digest" call for
    different next steps.

    Attributes:
        rebuilt: The indexes that were replaced, digest recorded, before the failure, in the
            order they were.
        committed_without_digest: The index that was replaced but whose digest could not be
            recorded, or ``None``. Reported apart from ``rebuilt`` because it needs a
            different repair: the index is current and correct, and the sidecar behind it
            refuses every later write until it is re-baselined. Treating it as "not written"
            tells an operator to expect previous contents that are gone.
    """

    def __init__(
        self,
        message: str,
        *,
        rebuilt: list[RebuiltIndex],
        committed_without_digest: Path | None = None,
    ) -> None:
        """Record what had already been written when the rebuild failed."""
        super().__init__(message)
        self.rebuilt = rebuilt
        self.committed_without_digest = committed_without_digest


class IndexDigestNotRecordedError(RuntimeError):
    """The index was committed but its digest could not be recorded.

    Distinct from :class:`StaleIndexReadError`: nothing is stale and no data is lost, but the
    sidecar is now behind the index, and every later write refuses until it is re-baselined.
    """


#: Failures a backoff cannot help. Raised past the retry loop rather than attempted again.
_NOT_WORTH_RETRYING = (StaleIndexReadError, IndexDigestNotRecordedError)


def _is_not_worth_retrying(error: BaseException) -> bool:
    """Return whether this failure, or anything it was raised from, is futile to retry.

    Only ``__cause__`` is followed, never ``__context__``. Explicit chaining --
    ``raise RuntimeError(...) from StaleIndexReadError`` -- says the original failure is the
    reason for this one, so it is still futile to retry. ``__context__`` is set implicitly by
    *any* exception raised while another is being handled, including a cleanup or logging failure
    that is itself the real problem: following it made an ``OSError`` raised while handling a
    stale read non-retryable, which is a different bug from the one this prevents.

    Args:
        error: The exception that ended the attempt.

    Returns:
        Whether to give up immediately.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, _NOT_WORTH_RETRYING):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    state_restore_func: Any = None,
) -> Any:
    """Retry a function with exponential backoff and optional state restoration.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        state_restore_func: Optional callable to restore state before each retry.
                           Called before each retry attempt (not before first attempt).

    Returns:
        Result of function call

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:  # pylint: disable=broad-exception-caught
            if _is_not_worth_retrying(e):
                # Waiting changes nothing for these: a stale index read is a cache state, not a
                # transient fault, and an unrecorded digest needs an operator. Retrying also
                # re-simulates the whole batch each time -- they are raised after the frames and
                # metadata are already written -- so the attempts cost full simulations and fail
                # anyway, with the second onwards tripping over the first's own outputs. Fail
                # now; a resumed run re-runs the batch because its checkpoint was never saved.
                raise
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                    delay,
                    exc_info=e,
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Restore state before retry if function provided
                if state_restore_func is not None:
                    try:
                        state_restore_func()
                        logger.debug("State restored before retry attempt %d", attempt + 2)
                    except Exception as restore_error:
                        logger.error("Failed to restore state before retry: %s", restore_error)
                        raise RuntimeError(f"Cannot retry: failed to restore state: {restore_error}") from restore_error
            else:
                logger.error("All %d attempts failed for batch: %s", max_retries + 1, str(e))

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected retry failure")


def update_metadata_index(
    metadata_directory: Path,
    output_files: list[Path],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update the central metadata index file.

    The index maps data file names to their corresponding metadata files,
    enabling O(1) lookup to find metadata for a given data file.

    Serialised and written atomically, for the reasons its sibling
    :func:`update_signal_index` was: an unlocked read-modify-write let two runs sharing a
    metadata directory lose one side's entries (measured at 4 losses in 20 concurrent trials),
    and the in-place truncating write turned one interrupted dump into the loss of *every* entry,
    because the loader above treats an unparsable index as "create a new index".

    **Deliberately weaker than :func:`update_signal_index`: there is no staleness digest here.**
    That guard's job is to refuse a write, and nothing in production reads ``index.yaml`` today,
    so aborting a run over it would cost more than it protects.

    **Be precise about what that costs, because it is not one entry.** On a shared filesystem a
    client can hold a stale *negative* view of this path -- measured: ``exists()`` answers False
    from a cached dentry, the loader starts from an empty mapping, and the write replaces the
    file, so ``{data-0, data-1}`` became ``{data-2}``. The residue is **whole-index loss**, the
    same catastrophic case the sibling's digest guard exists to catch, not a single dropped row.

    **Revisit when a reader-writer appears from another process**, not merely when a reader
    appears: a second reader alone is harmless, while a second *writer* on another host can
    discard everything recorded so far.

    Args:
        metadata_directory: Directory where metadata files are stored
        output_files: List of output data file Paths
        metadata_file_name: Name of the metadata file (e.g., "signal-0.metadata.yaml")
        encoding: File encoding for reading/writing the index file
    """
    index_file = metadata_directory / "index.yaml"
    with _exclusive_index_lock(index_file):
        # Load existing index or create new
        if index_file.exists():
            try:
                with index_file.open(encoding=encoding) as f:
                    index = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as e:
                logger.warning("Failed to load metadata index: %s. Creating new index.", e)
                index = {}
        else:
            index = {}

        # Add entries for all output files
        for output_file in output_files:
            index[output_file.name] = metadata_file_name
            logger.debug("Index entry: %s -> %s", output_file.name, metadata_file_name)

        # Save updated index
        try:
            _atomically_write_index(index_file, index)
            logger.debug("Updated metadata index: %s", index_file)
        except (OSError, yaml.YAMLError) as e:
            logger.error("Failed to save metadata index: %s", e)
            raise


def _withdraw_batch(spec: IndexSpec, index: dict[str, Any], metadata_file_name: str) -> dict[str, Any]:
    """Return *index* with every contribution from *metadata_file_name* removed.

    Entries left with no contributions are dropped, so a re-run that injects nothing no longer
    leaves an id pointing at frames it did not write.

    Pre-1.5.0 signal entries carried a single ``metadata`` string and a flat ``frames`` list. They
    are migrated in passing rather than rejected: an index is a rebuildable cache, and refusing to
    read one written by an older gwmock would make an upgrade look like data loss.
    """
    migrated: dict[str, Any] = {}
    for event_id, entry in index.items():
        batches = entry.get("batches")
        if batches is None:
            batches = [{"metadata": entry.get("metadata"), "frames": entry.get("frames") or []}]
        kept = [batch for batch in batches if batch.get("metadata") != metadata_file_name]
        if not kept:
            continue
        migrated[event_id] = {"batches": kept, spec.time_key: entry.get(spec.time_key)}
    return migrated


try:  # pragma: no cover - the except branch is unreachable on the supported platforms
    import fcntl
except ImportError:  # pragma: no cover - Windows, which the CI matrix does not cover
    fcntl = None  # type: ignore[assignment]


# `flock` errnos meaning "this filesystem cannot do advisory locks", as opposed to "you may not".
# Only these fall back to an unlocked write; a refusal such as EACCES still raises.
_LOCKING_UNSUPPORTED = frozenset(
    getattr(errno, name) for name in ("EOPNOTSUPP", "ENOLCK", "ENOSYS", "ENOTSUP") if hasattr(errno, name)
)


@cache
def _warn_unlocked_once() -> None:
    """Say once per process that the index is being updated without a lock.

    Cached rather than flagged so the "warns once" claim is enforced by the decorator instead of
    by a global nobody re-checks.
    """
    logger.warning("fcntl unavailable: the truth indexes are updated without a lock, so concurrent runs can race.")


@contextmanager
def _exclusive_index_lock(index_file: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock for the whole read-modify-write of the index.

    A lock is needed rather than a thread lock because the writers are separate *runs* sharing a
    metadata directory, in separate interpreters. The lock lives in a sidecar file rather than on
    the index itself: the index is replaced by :func:`_atomically_write_index`, and a lock held on
    a replaced inode protects nothing once the rename lands.

    The sidecar is created and left in place, never unlinked. Deleting it on release would let a
    second process hold a lock on an unlinked inode while a third creates a fresh one and takes a
    lock nobody else can see -- the classic unlink race. An empty stray file is the cheaper cost.

    **The sidecar is never tighter than the index**, which is not cosmetic: ``flock`` needs a
    writable descriptor, so whoever may write the index must be able to open the sidecar for
    append. Leaving the sidecar at the umask default while the index is group-writable gives a
    second account read access to the index and a ``PermissionError`` at the lock -- the
    multi-account case this locking exists to serve, broken at the gate. Alignment only ever
    widens; see :func:`_align_lock_mode`.

    Where ``fcntl`` is unavailable (Windows, which this project does not test) the body runs
    unlocked and says so once, rather than failing at import: an unsynchronised index is what the
    caller had before, while an ImportError would take out a working single-writer run.

    Args:
        index_file: Path to the index file; the sidecar sits beside it.

    Yields:
        Nothing. The lock is held for the duration of the block.
    """
    # No `# pragma: no cover` any more, and that is the point: this branch broke three times in one
    # day while it was excluded, each time caught by a reviewer reading rather than by a failure.
    # `tests/cli/test_signal_index_without_locking.py` makes it reachable by setting `fcntl` to
    # `None`, and tripwires `os.fchmod` and `os.fchown` so a return of those two names to this path
    # fails a test. It does **not** simulate Windows -- nothing here does, and the real differences
    # that remain are tracked rather than tested.
    if fcntl is None:
        _warn_unlocked_once()
        yield
        return

    lock_file = index_file.with_name(index_file.name + ".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_file.open("a+")
    except PermissionError as error:
        # This, not the chmod below, is where a second account actually fails: `flock` needs a
        # writable descriptor, so a sidecar tighter than the index stops a writer at the gate with
        # a bare Errno 13 that says nothing about locks. Name the cause and the repair.
        raise PermissionError(
            f"Cannot open the signal-index lock at {lock_file} for writing, so this run cannot "
            f"safely update {index_file.name}. Taking the lock needs write access to the sidecar; "
            "have its owner widen it to match the index (the owner's next gwmock run does this "
            "automatically)."
        ) from error
    with handle:
        # After the open, so the very first run aligns its own sidecar rather than leaving it at
        # the umask default for the next one to repair.
        _align_lock_mode(lock_file, index_file)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as error:
            if error.errno not in _LOCKING_UNSUPPORTED:
                raise
            # The filesystem does not implement advisory locks. Failing here would break a write
            # that succeeds -- and did succeed before locking existed -- on exactly the shared
            # filesystems this feature is aimed at. Degrade to the previous unsynchronised
            # behaviour and say so, the same trade as a missing `fcntl` module.
            _warn_unlocked_once()
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _align_lock_mode(lock_file: Path, index_file: Path) -> None:
    """Widen the sidecar so everyone who may write the index can take the lock. Best effort.

    **Widen only.** Matching the index exactly would also *narrow*, and a sidecar is a permission
    surface in its own right: an operator may deliberately open it to an account that takes locks
    without writing the index. Silently pulling that back on the owner's next run would remove a
    capability with no signal and no opt-out, to fix a problem nobody had. Too-tight sidecars are
    the failure this repairs; too-loose ones are somebody's decision.

    Only the file's owner may ``chmod``, so this cannot be an invariant every caller restores.
    The owner's next run repairs a sidecar written before this behaviour existed.

    Args:
        lock_file: The sidecar; may not exist yet.
        index_file: The index whose access the sidecar must not be tighter than.
    """
    desired = _existing_index_mode(index_file)
    current = _existing_index_mode(lock_file)
    if desired is None or current is None:
        return
    widened = current | desired
    if widened == current:
        return
    try:
        os.chmod(lock_file, widened)
    except (FileNotFoundError, PermissionError):
        return
    logger.debug("Widened %s from %s to %s to match the index.", lock_file, oct(current), oct(widened))


def _index_digest(index_file: Path) -> str:
    """Return a digest of the index's bytes, or the marker for "no index".

    Bytes rather than the parsed mapping: the point is to detect that *this read* differs from
    what the last writer committed, and a re-serialisation could mask a difference the reader
    would then act on.

    Args:
        index_file: Path to the index file.

    Returns:
        Hex digest, or ``"absent"`` when the file does not exist.
    """
    try:
        return hashlib.sha256(index_file.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "absent"


def _recorded_digest(lock_file: Path) -> str | None:
    """Return the digest the sidecar records, or ``None`` when it records none.

    ``None`` means an index predating this guard, not a mismatch.

    Args:
        lock_file: The sidecar beside the index.

    Returns:
        The recorded hex digest or ``"absent"`` marker, or ``None`` if unrecorded.
    """
    try:
        recorded = lock_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        # `None` means "no digest yet", which is the permissive legacy path. A sidecar that
        # exists and cannot be read is not that: it is a sidecar whose digest we are unable to
        # check, and treating it as absent would silently disable the guard on a corrupt or
        # unreadable file. Refuse instead.
        raise StaleIndexReadError(
            f"The index sidecar {lock_file} exists but could not be read ({error}), so "
            "this write cannot verify it is reading the current index. Stop all writers against "
            f"this metadata directory, then delete {lock_file.name} to re-baseline the digest; "
            "it holds only the digest and the lock, no data."
        ) from error
    return recorded or None


def _record_digest(lock_file: Path, digest: str) -> None:
    """Write the digest of the index just committed into the sidecar.

    Written in place, deliberately: the sidecar keeps its inode so the lock and this record stay
    on the file the lock made the filesystem revalidate. Replacing it by rename -- as the index
    is -- would reintroduce the cached-dentry problem this exists to detect.

    Written *after* the index, so a failure in between leaves the sidecar behind rather than
    ahead: ahead would promise an index that was never committed, which is silent. Behind is
    loud — but **not self-correcting**, and an earlier version of this docstring wrongly called
    it "recoverable". Nothing advances the recorded digest except this function, which runs only
    after a successful update, which the guard is by then refusing; the directory wedges until
    the sidecar is removed. That is why a failure here is raised rather than logged and dropped:
    the operator learns at the point of the fault, with the index intact, instead of meeting a
    permanent refusal on the next batch.

    Args:
        lock_file: The sidecar beside the index.
        digest: Digest of the index as committed.
    """
    try:
        # "r+" with a create fallback. The sidecar is normally created when the lock is taken,
        # but the `fcntl is None` branch yields without creating it, and plain "r+" would then
        # raise FileNotFoundError on every update -- telling the operator to delete a file that
        # does not exist, on the platform the lock deliberately degrades for.
        #
        # Not "a+": that sets O_APPEND, so writes go to the end whatever the seek position, and
        # the digest would accumulate rather than replace. Caught by the digest tests.
        try:
            handle = lock_file.open("r+", encoding="utf-8")
        except FileNotFoundError:
            handle = lock_file.open("w", encoding="utf-8")
        with handle:
            handle.write(digest)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        # Being able to take the lock does not mean the write will land: fsync can still fail
        # with EIO or ENOSPC. Swallowing it would leave the index committed and the sidecar
        # behind, which is the permanent wedge described above -- discovered by whoever runs the
        # next batch, with a message blaming a cache that is not the cause.
        logger.error("Could not record the index digest in %s: %s", lock_file, error)
        raise IndexDigestNotRecordedError(
            f"The index was committed, but its digest could not be recorded in "
            f"{lock_file.name} ({error}). The index itself is intact and correct. Until the "
            f"digest is re-synced every later write will refuse as stale. Stop every writer "
            f"against this directory, then delete {lock_file.name} to re-baseline it -- it holds "
            "only the digest and the lock, no data. Deleting it while a writer is running lets "
            "two processes lock different inodes and stop being serialised."
        ) from error


class _CommittedIndex(NamedTuple):
    """An index that has been installed, and what is known about the durability of its name."""

    digest: str
    """The sha256 of the bytes committed."""

    directory_flush: _DirectoryFlush
    """Whether the rename that installed those bytes was made durable."""


# The backoff between attempts at the rename that installs the index, in seconds, one entry per
# retry. Windows refuses `MoveFileEx` onto a destination another process holds open, and a *reader* is
# enough: `find_signals` and the CLI's own lookups open the index, so on Windows an update fails with
# `PermissionError` for as long as any consumer has it open. POSIX permits the rename regardless. The
# two ways to close that were to make the writer wait, or to document that consumers must not hold the
# index open; the writer waits, because the second pushes a constraint onto every reader of a file
# whose whole purpose is being read.
#
# Written out rather than derived from a factor and a count so both bounds are readable at once: five
# retries after the first attempt -- six attempts -- and 0.62 s of sleeping if every one of them is
# refused. Both bounds matter. The caller is a per-batch loop under an exclusive lock, so a rename that
# cannot be made to happen must fail soon rather than stall the run holding the lock; 0.62 s covers the
# open-read-close of a lookup, which is the contention this can actually wait out, and deliberately not
# a consumer that keeps the index open across seconds, which is a configuration to fix rather than a
# transient to absorb.
#
# No jitter. Jitter de-correlates contenders that collide on a resource, and the writers here do not
# collide on this one: they are serialised by the index's `flock` well before the rename, and what this
# waits for is a *reader* closing a handle. Adding it would buy nothing, and would cost the schedule
# the property its tests assert against: a bound that can be read off the tuple.
_INDEX_RENAME_BACKOFF_SECONDS: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16, 0.32)


def _replace_index_with_retry(temporary: Path, index_file: Path) -> None:
    """Rename *temporary* onto *index_file*, retrying a refusal a reader can be holding it open for.

    Only ``PermissionError`` is retried, and that is the whole of the transient case: the Windows
    refusals this exists for -- ``ERROR_SHARING_VIOLATION`` from a handle on the destination and
    ``ERROR_ACCESS_DENIED`` -- both reach Python as ``PermissionError``. Every other ``OSError`` is
    raised on the first attempt: a cross-device rename, a full disk or a vanished directory does not
    become possible by being asked again, and spending the backoff window on one would delay a real
    error while looking like resilience.

    **Not gated to Windows**, deliberately. Gating would put the only code path that matters on the one
    platform no host in this project can execute, which is how untested platform code passes review and
    fails in the field -- the same argument :func:`_fsync_directory` records for leaving its Windows
    branch out. The symptom is not exclusively Windows' either: an index on a CIFS/SMB mount carries the
    server's sharing semantics to a POSIX client. What the ungated version costs on POSIX is bounded and
    paid only on failure -- a genuine ``EACCES`` from the directory's permissions is raised after five
    pointless attempts and 0.62 s of sleeping, unchanged in every other respect.

    The atomicity contract is untouched: this retries the *same* rename of the *same* temporary, which
    either happens or does not. A reader sees the old index or the new one at every point.

    Args:
        temporary: The sibling temporary holding the new bytes.
        index_file: Destination the temporary is renamed onto.

    Raises:
        OSError: If the rename fails for any reason other than a refusal, or is still refused after
            every attempt. The error raised is the last attempt's own, unwrapped -- the caller unlinks
            the temporary and records no digest, exactly as it did when there was no retry at all.
    """
    for attempt, delay in enumerate(_INDEX_RENAME_BACKOFF_SECONDS, start=1):
        try:
            os.replace(temporary, index_file)
        except PermissionError as error:
            logger.debug(
                "Attempt %d of %d to rename the new index onto %s was refused (%s); retrying in %.2f s.",
                attempt,
                len(_INDEX_RENAME_BACKOFF_SECONDS) + 1,
                index_file,
                error,
                delay,
            )
            time.sleep(delay)
        else:
            if attempt > 1:
                logger.debug("The new index was renamed onto %s on attempt %d.", index_file, attempt)
            return
    # The last attempt, outside the loop because its failure is the caller's rather than something to
    # sleep after. Kept as a bare `os.replace` so the error the caller sees is the one the filesystem
    # raised, with its own errno and message, and not this function's paraphrase of it.
    try:
        os.replace(temporary, index_file)
    except PermissionError:
        # Warned rather than left to the traceback because the errno alone names the wrong repair. A
        # bare `[WinError 5] Access is denied` reads as a permissions problem, and the fix is to close
        # whatever is reading the index. Only reached once the whole window has been spent, so it is
        # not a per-attempt log, and the update fails immediately after it.
        logger.warning(
            "The new index could not be renamed onto %s: the rename was refused on all %d attempts "
            "over %.2f s. On Windows this is what another process holding %s open looks like -- a "
            "reader is enough, and `gwmock find-signal` and any other consumer of the index hold it "
            "open -- so close those and run the batch again. Elsewhere it is more likely to be the "
            "permissions of the directory the index lives in. The index and its digest are unchanged: "
            "the temporary is removed, nothing was recorded, and the previous index is intact.",
            index_file,
            len(_INDEX_RENAME_BACKOFF_SECONDS) + 1,
            sum(_INDEX_RENAME_BACKOFF_SECONDS),
            index_file.name,
        )
        raise


def _atomically_write_index(index_file: Path, index: dict[str, Any]) -> _CommittedIndex:
    """Write the index so a reader sees either the old file or the new one, never a fragment.

    ``open(path, "w")`` truncates in place, so a crash or a full disk part-way through the dump
    leaves an unparsable file -- and the loader in :func:`update_signal_index` treats an
    unparsable index as "create a new one", which turns one failed write into the silent loss of
    every entry the index already held. Writing a sibling temporary and renaming makes the
    replacement atomic within the directory, so a failure leaves the previous index untouched.

    The temporary is created in the destination directory because ``os.replace`` is only atomic
    within a filesystem, and ``/tmp`` is routinely a different one.

    The rename goes through :func:`_replace_index_with_retry` rather than ``os.replace`` directly,
    because Windows refuses it while any process -- a reader included -- holds the destination open.
    That is a bounded wait on the same rename, not a second way of installing the file: the atomicity
    above is exactly as it was.

    Args:
        index_file: Destination path.
        index: The index mapping to serialise.

    Returns:
        The sha256 of the bytes committed, and whether the rename that installed them was made
        durable. The digest is returned rather than re-read afterwards: the whole defect this guards
        is that reading ``index_file`` can be served from a stale client cache, so digesting a re-read
        could record the digest of this client's *stale view* — and a later client with the same stale
        view would then match it and overwrite silently, which is the bug rather than the fix. The
        flush outcome travels with it so the caller can *report* a rename whose durability is
        unverifiable. It is not a gate on recording the digest: an earlier version of this function made
        it one, and that turned a rare loud failure into a routine silent one -- see the decision comment
        in :func:`update_signal_index`.

    Raises:
        OSError: If the file cannot be written or renamed.
        yaml.YAMLError: If the index cannot be serialised.
    """
    # `os.replace` carries the temporary's mode to the destination, so the temporary must be
    # created the way the destination should end up. `tempfile.mkstemp` is wrong here: it forces
    # 0600 and ignores the umask, which would quietly tighten a 0644 index and lock a second
    # account out of `find-signal` on the shared metadata directory this locking exists for.
    # Opening with an explicit 0o666 lets the kernel apply the umask and any default ACLs exactly
    # as a plain `open(..., "w")` would have -- no process-global umask read, which would race
    # with any other thread creating a file (gwmock runs a resource monitor thread).
    # Serialise first. Doing it after the temporary exists would leak both the file and its
    # descriptor when serialisation fails, because the cleanup below is scoped to the write.
    payload = yaml.safe_dump(index, default_flow_style=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    existing_mode = _existing_index_mode(index_file)
    temporary = index_file.with_name(f"{index_file.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        # Adopt the raw descriptor immediately. Anything that raises between `os.open` and
        # `os.fdopen` -- the chmod below used to sit there -- leaves a descriptor that the
        # cleanup path unlinks the file for but never closes, and this runs once per batch.
        # Binary, so the bytes written are the bytes hashed. A text handle translates newlines
        # on Windows, and the recorded digest would then never match the file -- refusing every
        # update after the first on exactly the platform the lock already degrades for.
        handle = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    try:
        if existing_mode is not None:
            # An index that already exists keeps whatever mode it was given, so a deliberately
            # group-writable index survives an update.
            #
            # `os.chmod` on the path, not `os.fchmod` on the descriptor: `fchmod` does not exist
            # on Windows, and this line runs on every update once an index exists -- it would
            # break the second write on exactly the platform the lock is written to degrade
            # gracefully for. The path is a uuid4 name we just created in a directory we hold,
            # so there is no meaningful window for it to be swapped.
            os.chmod(temporary, existing_mode)
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_index_with_retry(temporary, index_file)
        # The rename itself has to reach the disk before the digest describing it does. `os.replace`
        # is atomic with respect to readers, which is what the comment above is about, but atomicity
        # is not durability: the directory entry can still be in the page cache when
        # `_record_digest` writes the sidecar. A crash in that window leaves the *old* index with the
        # *new* digest, and every later write then refuses as stale against a perfectly good file --
        # loud and recoverable by deleting the sidecar, never a silent blessing of wrong bytes, but
        # it needs an operator.
        flush = _fsync_directory(index_file.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _CommittedIndex(digest, flush)


# What is currently known to be degraded, so the warning is not repeated per batch. Entries are
# `(scope, key)`: a device number for a filesystem that refuses the flush, a directory for one that
# cannot be opened. Both are cleared when a flush there succeeds, so an episode that ends and returns
# is reported again -- a permanent cache went silent after the first episode of an intermittent mount.
#
# The two scopes exist because the two faults have different extents and different repairs. An fsync
# the filesystem rejects is a property of the mount, so keying it per index warned once per metadata
# directory: a run over twenty of them emitted twenty copies of a multi-line warning and kept twenty
# entries for the life of the process. A directory that cannot be *opened* is a property of that
# directory's permissions, and keying it by device hid the second such directory on a filesystem
# entirely -- its own repair never shown. The directory scope keys on the *absolute* path rather than
# the spelling it was given: one relative path used from two working directories is two different
# directories, and suppressing the second one's warning is the failure direction that costs an operator
# a repair. Two spellings of one directory -- through a symlink, or a bind mount -- still warn twice,
# which is the harmless direction and not worth a `resolve()` that can fail on the directory being
# reported. Spelling was never a key for the filesystem scope, where it warned once for what were two
# different mounts.
#
# `st_dev` is the best filesystem identity `os.stat` offers, and it is not perfect: Linux allocates
# device numbers for NFS, tmpfs, FUSE and overlay from a pool and reuses them after unmount, so a
# still-degraded number inherited by a new mount suppresses that mount's first warning. The mount id
# that would distinguish them (`STATX_MNT_ID`) is not exposed through `os.stat`, and clearing on
# success already covers the case where the new mount works. Accepted, and recorded here rather than
# left for the next reader to rediscover.
#
# Not synchronised, and the reason is narrower than this comment used to claim. Callers hold the index's
# `flock` *when locking is available*: `_exclusive_index_lock` deliberately yields **unlocked** when
# `fcntl` is missing, and when `flock` fails with a locking-unsupported errno such as `ENOLCK` or
# `EOPNOTSUPP` -- on those filesystems two hosts reach this set with nothing serialising them. Within one
# process the writer is still a sequential batch loop and the resource-monitor thread never touches it.
#
# The check-then-add is therefore formally racy in more situations than "not at all", and it stays
# unsynchronised anyway: a 64-thread stress probe produced exactly one warning, and the worst case in
# every case is a duplicated or missing *warning*, never a correctness path. Stated properly because the
# false version of it -- "every production caller holds the lock" -- is what a future reader would rely
# on when deciding whether a lock is needed here.
_DEGRADED_TARGETS: set[tuple[str, object]] = set()


def _note_flush_outcome(directory: Path, flush: _DirectoryFlush, index_file: Path, sidecar_name: str) -> None:
    """Warn on entering a degraded episode, once per affected scope rather than once per batch.

    At ``WARNING`` deliberately. The CLI logs at ``INFO`` by default, so the ``logger.debug`` lines
    inside :func:`_fsync_directory` produce nothing an operator sees, and a degradation with no signal
    is indistinguishable from a guarantee. Once per *episode* per process: a scheduler invoking gwmock
    repeatedly against a degraded mount is told every time, which is the right way round for a condition
    an operator can act on.

    ``UNSUPPORTED`` says nothing. It is a permanent property of the platform rather than a fault, and a
    warning on every run about a gap nobody can close is how the ones that can be acted on get ignored.

    Args:
        directory: The directory whose flush was attempted.
        flush: What :func:`_fsync_directory` managed.
        index_file: The index whose rename could not be made durable.
        sidecar_name: File name of the sidecar recording its digest.
    """
    if flush is _DirectoryFlush.UNSUPPORTED:
        return
    try:
        device: object = directory.stat().st_dev
    except OSError:
        # Unidentifiable is its own key rather than a reason to skip: losing the device number must not
        # lose the warning.
        device = None
    if flush is _DirectoryFlush.FLUSHED:
        # Both scopes clear: whichever fault was recorded for this directory, a flush that succeeds here
        # ends its episode.
        _DEGRADED_TARGETS.discard(("filesystem", device))
        _DEGRADED_TARGETS.discard(("directory", os.path.abspath(directory)))
        return
    if flush is _DirectoryFlush.UNAVAILABLE:
        target = ("directory", os.path.abspath(directory))
        message = (
            "%s could not be opened to flush its entries, so the rename that installed this update is "
            "not known to have reached stable storage. The index and its digest are correct and "
            "complete, and the staleness guard is intact. No repair can be named from here -- the "
            "causes range from a directory removed under the run to a process out of file descriptors "
            "-- so the specific error is in the debug log. What is not covered until it is fixed: a "
            "crash before the directory entry reaches the disk can reboot to the previous index while "
            "%s describes this one, and every later write refuses as stale against a good file -- "
            "recovered from by stopping the writers and deleting that sidecar, which holds only the "
            "digest and the lock. Warned once per directory while the condition lasts, and again if it "
            "clears and returns."
        )
    elif flush is _DirectoryFlush.UNREADABLE:
        target = ("directory", os.path.abspath(directory))
        message = (
            "%s could not be opened to flush its entries, so the rename that installed this update is "
            "not known to have reached stable storage. The index and its digest are correct and "
            "complete, and the staleness guard is intact. This is a property of this directory rather "
            "than of the filesystem -- most often its permissions: flushing needs to open it for "
            "reading, which a write-only directory refuses. Repair it by making the directory readable "
            "by the account running gwmock; deleting the sidecar does not help, because the next write "
            "cannot flush it either. What is not covered until then: a crash before the directory entry "
            "reaches the disk can reboot to the previous index while %s describes this one, and every "
            "later write refuses as stale against a good file -- recovered from by stopping the writers "
            "and deleting that sidecar, which holds only the digest and the lock. Warned once per "
            "directory while the condition lasts, and again if it clears and returns."
        )
    else:
        target = ("filesystem", device)
        message = (
            "The filesystem holding %s refused to flush the directory, so the rename that installed "
            "this update is not known to have reached stable storage. The index and its digest are "
            "correct and complete, and the staleness guard is intact. Nothing on this mount repairs "
            "it -- a filesystem that rejects the call offers no way to make a rename durable -- so the "
            "choice is to accept the exposure or to put the metadata directory on a filesystem that "
            "supports the flush. What is not covered meanwhile: a crash before the directory entry "
            "reaches the disk can reboot to the previous index while %s describes this one, and every "
            "later write refuses as stale against a good file -- recovered from by stopping the writers "
            "and deleting that sidecar, which holds only the digest and the lock. On a network mount "
            "that window lasts until the client sends the rename, which is seconds rather than "
            "instants. Warned once per filesystem while the condition lasts, and again if it clears "
            "and returns."
        )
    if target in _DEGRADED_TARGETS:
        return
    _DEGRADED_TARGETS.add(target)
    logger.warning(message, index_file, sidecar_name)


class _DirectoryFlush(Enum):
    """Whether a rename was made durable, and if not, how that is reported.

    Not a gate on recording the digest -- the digest is recorded whatever this says. An earlier version
    of this branch made it one; see the decision comment in :func:`update_signal_index` for why that was
    reversed.
    """

    FLUSHED = "flushed"
    """The directory's entries reached stable storage: the digest describes a durable name."""

    UNSUPPORTED = "unsupported"
    """This platform offers no way to flush a directory here. Durability is unverifiable."""

    REFUSED = "refused"
    """The filesystem rejected the flush itself. Unverifiable, unexpected, and true of the whole mount."""

    UNREADABLE = "unreadable"
    """This directory refused to be opened for reading. Its own permissions, and repairable as such."""

    UNAVAILABLE = "unavailable"
    """This directory could not be opened for some other reason. No repair can be prescribed."""


def _fsync_directory(directory: Path) -> _DirectoryFlush:
    """Flush *directory*'s entries, so a rename inside it survives a crash.

    POSIX only, and deliberately not emulated elsewhere. Windows cannot open a directory with
    ``os.open``, and the equivalent there would be a ``FlushFileBuffers`` call through ``ctypes`` on a
    handle obtained with ``CreateFileW`` -- code no host in this project can execute. A reviewer
    proposed exactly that; it is left out, because untested platform code that looks like protection
    is worse than a documented gap. NTFS journals metadata, so the exposure there is smaller in any
    case.

    Reports its outcome rather than raising. The index is already committed by the time this runs, so
    raising would turn a landed update into an error. But the outcome cannot be *dropped* either: the
    caller records a digest next, and a digest recorded for a rename that never reached the disk is
    precisely the wedge this function exists to prevent. :func:`update_signal_index` decides what to
    do with each outcome; the three failures are kept distinct because they warrant different answers --
    an unreadable directory names a repair, an unsupported platform names none, and a refusing filesystem
    names a different one.

    Args:
        directory: Directory whose entries should be flushed.

    Returns:
        Which of the five outcomes occurred: flushed, unsupported by the platform, refused by the
        filesystem, a directory that refused to be opened, or one that could not be opened for another
        reason.
    """
    # No `pragma: no cover`. Every branch here is reached by
    # `tests/cli/test_index_directory_durability.py` -- this one by deleting the attribute, the two
    # failures by an unopenable directory and a refusing fsync. A pragma claiming an executed branch
    # cannot run is how a break in it still reports green, which this file has already paid for once.
    if not hasattr(os, "O_DIRECTORY"):
        return _DirectoryFlush.UNSUPPORTED
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except PermissionError as error:
        logger.debug("Not permitted to open %s to flush its entries: %s", directory, error)
        return _DirectoryFlush.UNREADABLE
    except OSError as error:
        # Everything else the open can fail with -- `EMFILE`/`ENFILE` from an exhausted descriptor
        # table, `ENOENT` from a directory removed under the run, `EIO` from the device. Kept apart from
        # the permission case because that one's warning names a repair -- widen the mode -- which none
        # of these are fixed by, and a message that prescribes the wrong repair is worse than a generic
        # one. The specific errno reaches the debug log rather than the warning, which stays stable
        # enough to grep for.
        logger.debug("Could not open %s to flush its entries: %s", directory, error)
        return _DirectoryFlush.UNAVAILABLE
    try:
        os.fsync(descriptor)
    except OSError as error:
        logger.debug("Could not flush the entries of %s: %s", directory, error)
        return _DirectoryFlush.REFUSED
    finally:
        os.close(descriptor)
    return _DirectoryFlush.FLUSHED


def _existing_index_mode(index_file: Path) -> int | None:
    """Return the index's current permission bits, or ``None`` when it does not exist yet.

    Args:
        index_file: The index being replaced.

    Returns:
        Permission bits to preserve, or ``None`` to let the umask decide.
    """
    try:
        return stat.S_IMODE(index_file.stat().st_mode)
    except FileNotFoundError:
        return None


def update_signal_index(
    metadata_directory: Path,
    metadata: dict[str, Any],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update ``signal_index.yaml`` with this batch's injected signals.

    See :func:`_update_index`, which this and :func:`update_glitch_index` are both names for.

    Args:
        metadata_directory: Directory where metadata and the index live.
        metadata: The batch metadata record just written.
        metadata_file_name: File name of that batch metadata record.
        encoding: File encoding for reading/writing the index file.
    """
    _update_index(SIGNAL_INDEX, metadata_directory, metadata, metadata_file_name, encoding)


def update_glitch_index(
    metadata_directory: Path,
    metadata: dict[str, Any],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update ``glitch_index.yaml`` with this batch's injected glitches.

    The glitch counterpart of :func:`update_signal_index`, answering "which frame holds glitch X"
    the way ``gwmock find-signal`` already answered it for signals. Same machinery, same failure
    modes, same repair (``gwmock reindex``); see :func:`_update_index`.

    Args:
        metadata_directory: Directory where metadata and the index live.
        metadata: The batch metadata record just written.
        metadata_file_name: File name of that batch metadata record.
        encoding: File encoding for reading/writing the index file.
    """
    _update_index(GLITCH_INDEX, metadata_directory, metadata, metadata_file_name, encoding)


def _update_index(
    spec: IndexSpec,
    metadata_directory: Path,
    metadata: dict[str, Any],
    metadata_file_name: str,
    encoding: str = "utf-8",
) -> None:
    """Update one truth index, mapping each injected event to its frame file(s).

    The index maps an event's ``event_id`` to the frame file(s) that contain it plus the batch
    metadata file, enabling O(1) event->frame lookup by id.

    **Safe against concurrent writers on one host, when the lock is taken**, which it was not before: the
    read-modify-write was unlocked, so two runs sharing a metadata directory lost one side's
    events -- reproduced deterministically with two processes and a barrier, where the loser's
    signals stayed in the frames while vanishing from the id lookup. The whole cycle now runs
    under an exclusive ``flock`` on a sidecar file, and the result is renamed into place rather
    than written over the live index, so a failed or interrupted write leaves the previous index
    intact instead of truncating it -- which mattered because the loader below treats an
    unparsable index as "create a new one", turning one bad write into the loss of every entry.

    .. warning::

        **Writers on different hosts can still lose each other's entries, even where the lock
        works.** Measured on an NFS home shared by two machines (2026-08-07): the lock excluded
        correctly -- the second writer blocked 26.7 s waiting for the first -- and the update was
        lost anyway. Acquiring the lock revalidates the *sidecar*, not ``signal_index.yaml``, so
        the second writer read its client's cached view of the index, found nothing, started from
        an empty mapping and wrote a file holding only its own event. The first writer's data was
        on the server throughout.

        The cause is not the lock. It is that ``_atomically_write_index`` replaces the index by
        rename, giving it a new inode, which a client holding a cached entry for the old path
        does not resolve to. Atomic replacement and cache coherence pull in opposite directions
        here, and the sidecar -- chosen so the lock survives the rename -- is the thing the
        filesystem revalidates instead of the index.

        **Until this is repaired, do not run concurrent writers against one metadata directory
        from more than one host.** Concurrent writers on a single host are safe *when the lock is
        actually taken* — one host has one cache, so the staleness above cannot arise. They are
        **not** safe on the two paths where :func:`_exclusive_index_lock` deliberately proceeds
        unlocked: no ``fcntl`` module, and a filesystem that rejects ``flock``. Both warn once
        and then race exactly as this function did before the lock existed.

        An index that has already lost entries to any of these is repairable without rerunning
        anything: :func:`rebuild_signal_index` (``gwmock reindex``) derives it again from the batch
        metadata files, which no race touches. That is a repair, not a second guarantee -- it fixes
        what was lost rather than stopping the loss.

    Parameter-based lookup reads the events recorded in the batch metadata files (their
    source of truth); this index is only the id shortcut. A batch that injected nothing
    writes nothing.

    Args:
        spec: Which index to update, and where its events come from.
        metadata_directory: Directory where metadata and the index live.
        metadata: The batch metadata record just written.
        metadata_file_name: File name of that batch metadata record.
        encoding: File encoding for reading/writing the index file.
    """
    injections = spec.events(metadata) or []
    index_file = metadata_directory / spec.index_file_name
    lock_file = index_file.with_name(index_file.name + ".lock")
    with _exclusive_index_lock(index_file):
        # Taking the lock for a batch that turns out to have nothing to do is deliberate: the
        # decision below cannot be made without a trustworthy read, and only the lock provides
        # one. Measured at 0.049 s for 1000 no-op batches, and it leaves an empty sidecar in
        # directories that never gain an index -- harmless, since a later real write sees no
        # digest and takes the same permissive path as a fresh directory.
        #
        # Decided inside the lock, deliberately. Nothing read before it can be trusted: a stale
        # negative dentry -- the very fault this guards -- answers `exists()` for the index
        # without contacting the server, and the sidecar is created by the same first write, so a
        # client that probed the directory early caches a negative entry for *both*. An earlier
        # version returned here on an un-revalidated sidecar read, and a withdraw-only batch then
        # skipped its withdrawal exactly as before the fix. Taking the lock revalidates the
        # sidecar, which is the only one of the two whose reading can be believed.
        if not injections and not index_file.exists() and _recorded_digest(lock_file) is None:
            return
        _require_fresh_index_read(index_file, lock_file)
        committed = _update_index_locked(spec, index_file, injections, metadata, metadata_file_name, encoding)
        # Recorded whatever the flush managed, and this is a decision with a history. An earlier
        # version of this branch *withheld* the digest when the directory could not be flushed, on the
        # grounds that a digest describing a possibly-non-durable rename can wedge the directory after
        # a crash. Two reviewers showed that trade is inverted:
        #
        #   - An empty sidecar is the permissive legacy path, so a writer on another host with a stale
        #     cached view is *accepted* and its write discards the entries it could not see. That is
        #     silent loss on exactly the shared mounts this guard was added for, and it needs no crash.
        #   - The next batch would then warn "predates the staleness guard", which is a lie about an
        #     index written seconds earlier, and it fires per batch.
        #   - Withholding cannot even deliver its own invariant: the clear happens after the rename, so
        #     a crash between them leaves the sidecar's old digest against a possibly-new index, which
        #     refuses exactly as the state it was trying to avoid does.
        #
        # So the guard is kept and the durability gap is carried instead. A crash in that window leaves
        # the index and the sidecar disagreeing in one direction or the other, and either way the next
        # write refuses -- loud, and repaired by deleting the sidecar.
        #
        # Not "a rare window", which is what this comment claimed until a reviewer measured the class of
        # mount involved: an NFS client holds the rename until it sends the RPC, so on the filesystems
        # that refuse the flush the exposure is seconds rather than instants. The trade stands anyway --
        # a loud recoverable failure beats a silent one whatever its probability -- but it does not rest
        # on the window being small.
        _record_digest(lock_file, committed.digest)
        _note_flush_outcome(index_file.parent, committed.directory_flush, index_file, lock_file.name)


class RebuiltIndex(NamedTuple):
    """What a rebuild read and what it wrote."""

    index_file: Path
    """The index that was replaced."""

    batches: int
    """Batch metadata files that contributed at least one injection."""

    events: int
    """Distinct ``event_id`` values in the rebuilt index."""


#: The name this result carried while the signal index was the only one.
RebuiltSignalIndex = RebuiltIndex


def rebuild_signal_index(metadata_directory: Path, encoding: str = "utf-8") -> RebuiltIndex:
    """Rebuild ``signal_index.yaml`` from the batch metadata files, discarding what it held.

    See :func:`_rebuild_index`, which this and :func:`rebuild_glitch_index` are both names for.

    Args:
        metadata_directory: Directory holding ``*.metadata.json`` and the index.
        encoding: File encoding for reading the metadata files.

    Returns:
        The index written, and how much went into it.
    """
    return _rebuild_index(SIGNAL_INDEX, metadata_directory, encoding)


def rebuild_glitch_index(metadata_directory: Path, encoding: str = "utf-8") -> RebuiltIndex:
    """Rebuild ``glitch_index.yaml`` from the batch metadata files, discarding what it held.

    See :func:`_rebuild_index`. A directory whose runs injected no glitches rebuilds to an empty
    index rather than being refused: a run without glitches is an ordinary run, unlike a directory
    with no batch metadata files at all, which means the wrong path was given.

    Args:
        metadata_directory: Directory holding ``*.metadata.json`` and the index.
        encoding: File encoding for reading the metadata files.

    Returns:
        The index written, and how much went into it.
    """
    return _rebuild_index(GLITCH_INDEX, metadata_directory, encoding)


def rebuild_truth_indexes(metadata_directory: Path, encoding: str = "utf-8") -> list[RebuiltIndex]:
    """Rebuild every truth index from the batch metadata files, discarding what they held.

    One index per catalogue -- signals and glitches -- so this replaces two files, and two
    files cannot be replaced as a single atomic act. Rebuilding them one after the other
    naively means a failure on the second leaves the pair half rebuilt, with the first
    index replaced, its digest re-recorded, and a message that says only that the command
    failed.

    So the failure that *can* be avoided is avoided. Every index's sources are derived
    first, writing nothing, and a directory that cannot produce all of them is refused
    before any of them is touched. That covers the case a pair is actually exposed to:
    validation is per catalogue -- one index reads ``signal.injections`` and the other
    ``noise.glitch_injections`` -- so a record whose shape one accepts and the other
    refuses would otherwise pass the first rebuild and stop the second.

    The pre-pass is an extra read, not a substitute for the authoritative one. Each
    rebuild still scans the metadata files inside its own lock, which is load-bearing:
    a batch writes its metadata file and only then takes the lock to add its index entry,
    so a scan taken outside would miss a batch mid-update.

    What remains is a write that fails part-way -- a full disk, a revoked permission --
    which is reported rather than hidden: :class:`PartialIndexRebuildError` names the
    indexes that were rebuilt, so an operator knows which half of the pair is current.

    Args:
        metadata_directory: Directory holding ``*.metadata.json`` and the indexes.
        encoding: File encoding for reading the metadata files.

    Returns:
        One result per index, in the order they were rebuilt.

    Raises:
        IndexRebuildError: If the directory holds no batch metadata files, or one of them
            cannot be read, parsed, or decoded into the shape any index reads. Nothing has
            been written.
        PartialIndexRebuildError: If an index was written -- with or without its digest --
            and a later one was not. Its ``rebuilt`` and ``committed_without_digest``
            attributes say which is which, because the two need different repairs.
        IndexDigestNotRecordedError: If the last index was written but its digest could not
            be recorded, in which case every index is current and only a sidecar is behind.
        OSError: If the first index cannot be written.
    """
    for spec in TRUTH_INDEXES:
        # Read-only: builds the mapping each index would hold and throws it away. The point
        # is the validation it performs on the way.
        _index_from_batch_metadata(spec, metadata_directory, encoding)

    rebuilt: list[RebuiltIndex] = []
    for position, spec in enumerate(TRUTH_INDEXES):
        untouched = TRUTH_INDEXES[position + 1 :]
        try:
            rebuilt.append(_rebuild_index(spec, metadata_directory, encoding))
        except IndexDigestNotRecordedError as error:
            # A different outcome from the one below, and telling them apart is the whole
            # point of reporting: `_record_digest` runs *after* the index is committed, so
            # this index has been replaced. Saying it "still holds whatever it held before"
            # would send an operator looking for contents that no longer exist -- which is
            # what the first version of this reporting did say.
            if not untouched:
                # Nothing was skipped, so the error already describes the whole outcome, and
                # its message carries re-baselining advice this one could only restate.
                raise
            raise PartialIndexRebuildError(
                f"Replaced {spec.index_file_name}, but could not record its digest: {error} "
                f"That index is current and correct; its sidecar is behind it and will refuse "
                f"every later write until it is re-baselined, as the message above describes. "
                f"{', '.join(other.index_file_name for other in untouched)} was not rebuilt at "
                f"all and still holds whatever it held before, which may be what needed "
                f"repairing.",
                rebuilt=rebuilt,
                committed_without_digest=metadata_directory / spec.index_file_name,
            ) from error
        except (IndexRebuildError, OSError, yaml.YAMLError) as error:
            if not rebuilt:
                raise
            replaced = ", ".join(str(result.index_file) for result in rebuilt)
            remaining = ", ".join(other.index_file_name for other in TRUTH_INDEXES[position:])
            raise PartialIndexRebuildError(
                f"Rebuilt {replaced}, then could not rebuild {spec.index_file_name}: {error} "
                f"The indexes named first are current and their digests are recorded; "
                f"{remaining} still hold whatever they held before, which may be what needed "
                f"repairing. Fix the cause and run the command again -- a rebuild is idempotent, "
                f"so repeating it on the ones already done costs nothing.",
                rebuilt=rebuilt,
            ) from error
    return rebuilt


def _rebuild_index(spec: IndexSpec, metadata_directory: Path, encoding: str = "utf-8") -> RebuiltIndex:
    """Rebuild one truth index from the batch metadata files, discarding what it held.

    The index is a cache. The batch metadata files are the source of truth -- ``signal.injections``
    and the ``signal`` outputs of each batch -- so an index that has lost entries can be recovered
    exactly, without rerunning anything. That matters because :func:`update_signal_index` does not
    protect every case: it degrades to an unsynchronised write where ``fcntl`` is missing or the
    filesystem refuses ``flock``, and writers on different hosts are refused rather than merged
    (see :class:`StaleIndexReadError`). Each of those leaves a directory whose frames are right and
    whose id lookup is not, and this is what repairs it.

    Rebuilding also re-baselines the sidecar's digest, so the recovery previously written out as
    "delete ``signal_index.yaml.lock`` by hand" is now a command that leaves the directory in a
    state the staleness guard accepts. Hand-editing the index and hand-deleting the sidecar remain
    possible and remain a way to get it wrong.

    Held under the same exclusive lock as an incremental update, so a rebuild and a running batch
    cannot interleave. **The metadata files are read inside the lock, not before it**, and that is
    load-bearing rather than tidy: a batch writes its metadata file and only then takes the lock to
    add its index entry, so a scan taken outside would miss a batch that is mid-update and the
    rebuild would overwrite the entry it went on to write. Scanning under the lock puts that batch
    on one side or the other -- either its file is already listed, or it has yet to touch the index
    and appends to the rebuilt one when it acquires the lock.

    The existing index is **not** read: a rebuild replaces it wholesale, so there is nothing for a
    stale read of it to corrupt -- which is why this works in the wedged states above, where an
    update refuses.

    .. warning::

        A rebuild is only as complete as the directory listing it gets, and the lock does not fix
        that across hosts any more than it does for an update. On a shared filesystem whose client
        caches directory entries, a batch metadata file written seconds ago by another host may not
        be listed yet, and the rebuilt index will not mention it -- silently, because a missing
        file and a directory that never held it look identical from here. Stop writers on other
        hosts and let the listing settle before rebuilding there.

    Batch order follows the metadata file names, sorted. An incremental index instead carries the
    order the runs happened in, so the two can list one event's batches differently while holding
    the same set. Nothing reads that order -- :func:`~gwmock.cli.utils.signal_lookup.find_signals`
    flattens and deduplicates the frames -- but a byte comparison of two indexes is not a valid
    equality test.

    Args:
        spec: Which index to rebuild, and where its events come from.
        metadata_directory: Directory holding ``*.metadata.json`` and the index.
        encoding: File encoding for reading the metadata files.

    Returns:
        The index written, and how much went into it.

    Raises:
        IndexRebuildError: If the directory holds no batch metadata files, or one of them
            cannot be read, cannot be parsed, or does not decode into the shape of a batch
            metadata record.
        IndexDigestNotRecordedError: If the index was written but its digest could not be
            recorded.
        OSError: If the index cannot be written.
    """
    # Checked before the lock as well as inside it, so a mistyped path fails without leaving a
    # stray sidecar in whatever directory it named. The check inside is the one that counts.
    _require_batch_metadata(metadata_directory)

    index_file = metadata_directory / spec.index_file_name
    lock_file = index_file.with_name(index_file.name + ".lock")
    with _exclusive_index_lock(index_file):
        index, contributing = _index_from_batch_metadata(spec, metadata_directory, encoding)
        committed = _atomically_write_index(index_file, index)
        _record_digest(lock_file, committed.digest)
        _note_flush_outcome(index_file.parent, committed.directory_flush, index_file, lock_file.name)
    logger.info("Rebuilt %s from %d batch metadata file(s): %d event(s).", index_file, contributing, len(index))
    return RebuiltIndex(index_file=index_file, batches=contributing, events=len(index))


def _require_batch_metadata(metadata_directory: Path) -> list[Path]:
    """Return the batch metadata files in *metadata_directory*, refusing if there are none.

    Refused rather than treated as "a run that injected nothing". A metadata directory always
    holds the batch records that produced it, so an empty listing means the wrong directory --
    and writing an empty index there would replace a correct one on a mistyped path, which is
    the opposite of what a repair command is for.

    Args:
        metadata_directory: Directory holding ``*.metadata.json``.

    Returns:
        The metadata files, sorted by name.

    Raises:
        IndexRebuildError: If the directory holds none.
    """
    metadata_files = sorted(metadata_directory.glob("*.metadata.json"))
    if not metadata_files:
        raise IndexRebuildError(
            f"No batch metadata files (*.metadata.json) found in {metadata_directory}, so there is "
            "nothing to rebuild the index from. Check the path: rebuilding against the "
            "wrong directory would replace a correct index with an empty one."
        )
    return metadata_files


def _index_from_batch_metadata(spec: IndexSpec, metadata_directory: Path, encoding: str) -> tuple[dict[str, Any], int]:
    """Build the index a directory's batch metadata files describe, without writing it.

    Args:
        spec: Which index to build, and where its events come from.
        metadata_directory: Directory holding ``*.metadata.json``.
        encoding: File encoding for reading them.

    Returns:
        The index mapping, and how many batches contributed at least one injection.

    Raises:
        IndexRebuildError: If the directory holds no batch metadata files, or one of them
            cannot be read, cannot be parsed, or does not decode into the shape of a batch
            metadata record.
    """
    index: dict[str, Any] = {}
    contributing = 0
    for metadata_file in _require_batch_metadata(metadata_directory):
        try:
            decoded = json.loads(metadata_file.read_text(encoding=encoding))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            # Refused, where `find_signals` skips an unreadable file and carries on. The two are
            # answering different questions: a query returns what it can find, while a rebuild
            # replaces the index, so a file skipped here is an event silently deleted from the
            # lookup by the very command run to restore it.
            _refuse_metadata_file(metadata_file, f"could not be read ({error})", error)
        metadata, injections = _validated_batch_metadata(spec, metadata_file, decoded)
        if not injections:
            continue
        _record_batch_in_index(spec, index, injections, metadata, metadata_file.name)
        contributing += 1
    return index, contributing


def _refuse_metadata_file(metadata_file: Path, detail: str, cause: BaseException | None = None) -> NoReturn:
    """Refuse the whole rebuild because one batch metadata file cannot be used.

    One function for both refusals -- the file that will not decode and the file that decodes into
    the wrong shape -- so the reason a rebuild is all-or-nothing is stated once and cannot drift
    between them.

    Args:
        metadata_file: The file that stopped the rebuild.
        detail: What is wrong with it, as a clause completing "<name> ...".
        cause: The underlying error, when there was one.

    Raises:
        SignalIndexRebuildError: Always.
    """
    raise IndexRebuildError(
        f"Cannot rebuild the index: {metadata_file.name} {detail}. Every batch metadata "
        "file has to be usable, because an index built from the rest would be missing exactly the "
        "events this one recorded and would look complete. The existing index has not been "
        "touched."
    ) from cause


def _validated_batch_metadata(
    spec: IndexSpec, metadata_file: Path, decoded: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check that a decoded batch metadata record has the shape the rebuild reads, and return it.

    Parsing as JSON is not the same as being a batch metadata record, and the gap between the two
    is a real file on disk that this command is pointed at precisely when the directory is already
    in a bad state. ``[]``, ``null``, or a ``signal`` that is a string all decode cleanly and then
    fail at ``.get`` with an ``AttributeError`` -- a traceback where the docstring, the CLI help
    and the user guide all promise a refusal naming the file. Found in review.

    Every field the rebuild goes on to touch is checked here, including the nested ones:
    :func:`_record_batch_in_index` indexes into each injection and each output, so validating only
    the top level would move the traceback rather than remove it.

    Not applied to :func:`_update_index`, deliberately. Its record is built in-process by the
    run that is writing it, a few frames up the stack; there is no file, no decode, and nothing to
    distrust. Validating it would be a per-batch cost paid to guard against gwmock having
    constructed its own record wrongly, which is a bug for a test to catch rather than a shape for
    a hot path to re-check.

    ``outputs`` is normalised to a list on the returned record, so an explicit ``null`` is read the
    way an absent key already was rather than becoming a ``TypeError`` in the builder.

    Args:
        spec: Which index is being rebuilt, and which section of the record it reads.
        metadata_file: The file the record came from, named in any refusal.
        decoded: Whatever ``json.loads`` returned.

    Returns:
        The record, and its events (empty when it records none).

    Raises:
        IndexRebuildError: If any part of the record is not the shape the rebuild reads.
    """

    def refuse(field: str, value: Any, expected: str) -> NoReturn:
        _refuse_metadata_file(
            metadata_file,
            f"parsed as JSON but is not a usable batch metadata record: {field} is "
            f"{type(value).__name__}, not {expected}",
        )

    if not isinstance(decoded, dict):
        refuse("the top-level value", decoded, "an object")

    section = decoded.get(spec.section)
    if section is not None and not isinstance(section, dict):
        refuse(f"'{spec.section}'", section, "an object")

    injections = (section or {}).get(spec.events_key)
    if injections is not None and not isinstance(injections, list):
        refuse(f"'{spec.events_path}'", injections, "a list")

    for position, injection in enumerate(injections or []):
        if not isinstance(injection, dict):
            refuse(f"'{spec.events_path}[{position}]'", injection, "an object")
        event_id = injection.get("event_id")
        # `str(event_id)` is the index key, so anything it would stringify into nonsense has to
        # stop here: a mapping or a list would key the entry by its repr, and a float by a
        # spelling (`3.0`) that the integer lookup in `find_signals` never asks for. `bool` is an
        # `int` subclass and would key on `True`, so it is excluded by name.
        if event_id is not None and (isinstance(event_id, bool) or not isinstance(event_id, (int, str))):
            refuse(f"'{spec.events_path}[{position}].event_id'", event_id, "an integer or a string")
        if spec.parameters_key is not None:
            parameters = injection.get(spec.parameters_key)
            if parameters is not None and not isinstance(parameters, dict):
                refuse(f"'{spec.events_path}[{position}].{spec.parameters_key}'", parameters, "an object")

    outputs = decoded.get("outputs")
    if outputs is not None and not isinstance(outputs, list):
        refuse("'outputs'", outputs, "a list")
    for position, output in enumerate(outputs or []):
        if not isinstance(output, dict):
            refuse(f"'outputs[{position}]'", output, "an object")
        # Read off the builder's own inclusion test rather than restated beside it.
        # `_record_batch_in_index` records every matching output whose `path` key is *present*, so
        # every present key has to be a string. An earlier version restated the rule as
        # `path is not None`, which is a different condition: an explicit `"path": null` is present,
        # so the builder took it, and the rebuilt index carried `frames: [null]` -- which
        # `gwmock find-signal --id` then cannot join into its output. Found in review. Whatever the
        # builder includes, this validates; the two conditions are now the same sentence.
        if "path" in output and not isinstance(output["path"], str):
            refuse(f"'outputs[{position}].path'", output["path"], "a string")

    decoded["outputs"] = outputs or []
    # Every element was checked above, so the narrowing is a fact the loop established rather than
    # an assumption this line makes.
    return decoded, cast("list[dict[str, Any]]", injections or [])


def _require_fresh_index_read(index_file: Path, lock_file: Path) -> None:
    """Refuse to proceed when this process cannot see the index the sidecar says is current.

    Holding the lock does not make the read trustworthy across hosts: acquiring it revalidates
    the sidecar, not the index, so a client with a cached view reads an older index -- or none --
    and writing then discards everything it could not see. Measured on a shared NFS home, where
    the lock excluded correctly for 26.7 s and the update was lost anyway.

    The comparison is against the sidecar because that is the file the lock *does* make the
    filesystem revalidate, and it is written in place so it keeps its inode.

    A sidecar with no digest predates this guard and is accepted with a warning: refusing would
    break every in-flight run on upgrade, a certain cost against an uncertain one. The window is
    one unprotected write *per writer that has not yet recorded a digest* -- so on a directory
    shared by several hosts, each gets one pass before any of them records, and the original bug
    is reachable for exactly those writes. That acceptance should become a refusal once no index
    predates the release carrying this.

    Args:
        index_file: Path to the index file.
        lock_file: The sidecar holding the digest of the last committed index.

    Raises:
        StaleIndexReadError: If the index this process reads is not the one last committed.
    """
    recorded = _recorded_digest(lock_file)
    if recorded is None:
        if not index_file.exists():
            # A fresh directory, not a legacy one: the lock creates the sidecar empty, so the
            # first write into any new directory would otherwise warn that an index which does
            # not exist "predates the staleness guard". Warning on every clean run is how the
            # message that matters gets ignored. A directory whose runs have no glitches reaches
            # this on every batch, since the glitch index is never created there.
            return
        logger.warning(
            "The index at %s has no digest recorded in %s, so this write cannot verify it "
            "is reading the current index. Predates the staleness guard; accepted until a digest "
            "is recorded -- on a directory shared by several hosts each unprotected writer gets "
            "one such pass, so the first post-upgrade write is the one to run from a single host.",
            index_file,
            lock_file.name,
        )
        return
    seen = _index_digest(index_file)
    if seen == recorded:
        return
    raise StaleIndexReadError(
        f"This process reads {index_file} as {seen}, but {lock_file.name} records the last "
        f"committed index as {recorded}, so the read is stale and writing would discard entries "
        "this process cannot see. Causes, in the order worth checking: (1) a filesystem cache on "
        "a shared mount, where the lock is held correctly and the index is still read from an "
        "out-of-date view -- retry, and if it persists the writers are on different hosts and "
        "must be serialised outside gwmock; (2) the index was deleted or rebuilt by hand, which "
        "is legitimate -- it is a rebuildable cache -- but leaves the sidecar describing an index "
        f"that no longer exists; (3) a previous run could not record its digest. For (2) and (3) "
        f"the repair is `gwmock reindex --metadata-dir {index_file.parent}`, which rebuilds the "
        "index from the batch metadata files -- their injections are the source of truth -- and "
        "re-records the digest to match, so nothing is left resting on whichever index happens to "
        f"be on disk. Failing that, stop every writer against this directory and delete "
        f"{lock_file.name}, which re-baselines the digest on the next write against the index as "
        "it stands: that discards nothing the sidecar holds, but it also accepts an index that "
        "has already lost entries, which the rebuild does not. **Stop the writers first** -- "
        "removing it while another process holds the old inode lets a third create and lock a new "
        "one, so two writers hold different locks and are no longer serialised."
    )


def _record_batch_in_index(
    spec: IndexSpec,
    index: dict[str, Any],
    injections: list[dict[str, Any]],
    metadata: dict[str, Any],
    metadata_file_name: str,
) -> None:
    """Add one batch's contribution to *index*, in place.

    Shared by the incremental update and by the rebuild, deliberately. The two have to agree on
    what an entry looks like -- a rebuild that produced a subtly different shape would be a repair
    that corrupts -- and one implementation is the only way to say that which cannot drift. The
    rebuild test asserts the two paths produce the same mapping, so a change here that only suits
    one of them fails.

    Args:
        spec: Which index is being built, and how its entries are shaped.
        index: The index being built or updated; mutated in place.
        injections: The batch's events -- injected signals, or injected glitches.
        metadata: The batch metadata record.
        metadata_file_name: File name of that batch metadata record.
    """
    frames = [
        output["path"]
        for output in metadata.get("outputs", [])
        if output.get("kind") == spec.output_kind and "path" in output
    ]
    for injection in injections:
        event_id = injection.get("event_id")
        if event_id is None:
            continue
        # Appended, not assigned. A signal reaches every frame its samples land in, and each of those
        # frames belongs to a different batch writing this index in turn, so the previous
        # `index[event_id] = ...` kept whichever batch happened to write last. For a continuous wave
        # that is one frame out of every frame in the run; for a 48 s inspiral across 32 s segments it
        # was one of three, and not the one holding the merger.
        entry = index.setdefault(
            str(event_id),
            {"batches": [], spec.time_key: spec.event_time(injection)},
        )
        entry["batches"].append({"metadata": metadata_file_name, "frames": frames})


def _update_index_locked(
    spec: IndexSpec,
    index_file: Path,
    injections: list[dict[str, Any]],
    metadata: dict[str, Any],
    metadata_file_name: str,
    encoding: str,
) -> _CommittedIndex:
    """Do the read-modify-write, with the caller holding the index lock.

    Split out so the critical section is a single named span: the read, the withdrawal and the
    write must all be inside one lock, and a reader of :func:`_update_index` should not have
    to trace an indented block to confirm it.

    Args:
        spec: Which index is being updated.
        index_file: Path to the index file.
        injections: The batch's events.
        metadata: The batch metadata record just written.
        metadata_file_name: File name of that batch metadata record.
        encoding: File encoding for reading the index file.

    Returns:
        The committed index's digest and the durability of the rename that installed it, for the
        caller to record and, when the durability is unverifiable, to warn about.
    """
    if index_file.exists():
        try:
            with index_file.open(encoding=encoding) as f:
                index = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load %s: %s. Creating new index.", index_file.name, e)
            index = {}
    else:
        index = {}

    # Withdraw this batch's previous contribution, per event, so a re-run or overwrite (which may
    # now inject different or no events) cannot leave stale id -> frame rows the fast path would
    # trust. This used to drop whole entries whose `metadata` matched, which was equivalent only
    # while an entry belonged to exactly one batch -- and that is the assumption being removed here.
    index = _withdraw_batch(spec, index, metadata_file_name)

    _record_batch_in_index(spec, index, injections, metadata, metadata_file_name)

    try:
        return _atomically_write_index(index_file, index)
    except (OSError, yaml.YAMLError) as e:
        logger.error("Failed to save %s: %s", index_file.name, e)
        raise


def instantiate_simulator(
    simulator_config: SimulatorConfig | OrchestrationConfig,
    simulator_name: str | None = None,
    global_simulator_arguments: dict[str, Any] | None = None,
) -> Simulator:
    """Instantiate a simulator from configuration.

    Creates a single simulator instance that will be reused across multiple batches.
    The simulator maintains state (RNG, counters, etc.) across iterations.

    Global simulator arguments are merged with simulator-specific arguments,
    with simulator-specific arguments taking precedence.

    Args:
        simulator_config: Configuration for this simulator
        simulator_name: Name of the simulator (used for class path resolution)
        global_simulator_arguments: Global fallback arguments for the simulator

    Returns:
        Instantiated Simulator

    Raises:
        ImportError: If simulator class cannot be imported
        TypeError: If simulator instantiation fails
    """
    if isinstance(simulator_config, OrchestrationConfig):
        simulator = AdapterOrchestrator.from_config(
            orchestration_config=simulator_config,
            global_simulator_arguments=global_simulator_arguments,
        )
        logger.info("Instantiated adapter-backed orchestration path")
        return simulator

    class_spec = simulator_config.class_

    # Resolve short class names to full paths
    class_spec = resolve_class_path(class_spec, simulator_name)

    module_name, class_name = class_spec.rsplit(".", 1)
    simulator_module = import_module(module_name)
    simulator_cls = getattr(simulator_module, class_name)

    # Merge global and simulator-specific arguments
    # Simulator-specific arguments override global defaults
    if global_simulator_arguments:
        merged_arguments = {**global_simulator_arguments, **simulator_config.arguments}
    else:
        merged_arguments = simulator_config.arguments

    # Normalize keys: convert hyphens to underscores (YAML uses hyphens, Python uses underscores)
    normalized_arguments = {k.replace("-", "_"): v for k, v in merged_arguments.items()}

    simulator = simulator_cls(**normalized_arguments)

    logger.info("Instantiated simulator from class %s", class_spec)
    return simulator


def _checkpoint_state_belongs_to(
    batch: SimulationBatch,
    last_simulator_state: dict[str, Any],
    last_state_batch_index: int | None,
) -> bool:
    """Whether a checkpoint's state describes the simulator as it stood before *batch*.

    **Preferred rule: which batch produced it.** A state saved after batch N is the entry state of
    batch N+1 and of nothing else, so the caller passing ``last_state_batch_index`` gets that
    comparison.

    **Fallback: the counter.** Without that provenance the only thing to compare is
    ``state["counter"]`` against the batch index, which is a *proxy* -- it holds when a simulator's
    counter and its batch numbering share an origin, and silently fails when they do not. A plan
    numbering batches across several simulators (alpha 0-2, then beta 3-4) is exactly that case:
    beta's counter after its first batch is 1 while the batch about to run is index 4, so the proxy
    refuses a tail that was perfectly valid and the segment regenerates from scratch, losing the
    spillover with it -- measured at ``beta-0`` where a continued run gives ``beta-1``.

    The proxy is kept for callers that have no provenance to offer rather than removed, because
    dropping the check entirely would apply whatever state a caller happened to hold to whatever
    batch was in hand -- and restoring the *wrong* state is worse than restoring none.

    Args:
        batch: The batch about to run.
        last_simulator_state: The state from the checkpoint.
        last_state_batch_index: The batch this state was saved after, or ``None`` if unknown.

    Returns:
        Whether the state may be restored onto this batch.
    """
    if last_state_batch_index is not None:
        return last_state_batch_index == batch.batch_index - 1
    return batch.batch_index == last_simulator_state.get("counter")


def restore_batch_state(
    simulator: Simulator,
    batch: SimulationBatch,
    last_simulator_state: dict[str, Any] | None = None,
    last_simulator_spillover: Any = None,
    last_state_batch_index: int | None = None,
) -> None:
    """Restore simulator state from batch metadata or checkpoint file if available.

    This is used when reproducing a specific batch. It restores the RNG state,
    filter memory, and other stateful components that existed before this batch
    was generated.

    Both checkpoint arguments belong to *this* simulator: the checkpoint keeps one tail per
    simulator and the caller looks up the entry for the one about to run (:func:`resume_tail`).
    Passing another simulator's tail would resume this segment behind an RNG stream that never
    produced it, and place a signal tail that belongs to different data.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch potentially containing state snapshot
        last_simulator_state (optional): State this simulator left in the checkpoint, or None if
            unavailable. Applied only when it belongs to this batch -- see
            :func:`_checkpoint_state_belongs_to`.
        last_simulator_spillover (optional): Chunks that simulator's previous batch left for this
            one, or None. The caller has already checked they belong to this batch.
        last_state_batch_index (optional): The batch ``last_simulator_state`` was saved after. Pass
            it whenever it is known: without it the check falls back to a counter proxy that refuses
            a valid tail whenever a simulator's counter and the plan's batch numbering disagree.

    Raises:
        ValueError: If state restoration fails
    """
    if batch.has_state_snapshot() and batch.pre_batch_state is not None:
        logger.debug(
            "[RESTORE] Batch %d: Restoring state from snapshot - state_keys=%s",
            batch.batch_index,
            list(batch.pre_batch_state.keys()),
        )
        try:
            logger.debug(
                "[RESTORE] Batch %d: Setting state dict - counter=%s",
                batch.batch_index,
                batch.pre_batch_state.get("counter"),
            )
            simulator.state = batch.pre_batch_state
            logger.debug(
                "[RESTORE] Batch %d: State restored successfully - new_counter=%s",
                batch.batch_index,
                simulator.counter,
            )
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    elif last_simulator_state is not None and _checkpoint_state_belongs_to(
        batch, last_simulator_state, last_state_batch_index
    ):
        logger.debug(
            "[RESTORE] Batch %d: Restoring state from checkpoint last state - state_keys=%s",
            batch.batch_index,
            list(last_simulator_state.keys()),
        )
        try:
            logger.debug(
                "[RESTORE] Batch %d: Setting state dict - counter=%s",
                batch.batch_index,
                last_simulator_state.get("counter"),
            )
            simulator.state = last_simulator_state
            # Restored only on this branch -- the one that resumes from the checkpoint's *last*
            # state. The branch above restores from a batch metadata record, which by design does
            # not carry samples, so there is no spillover to restore there and a run resumed that
            # way still loses the tail. Stated rather than silently half-handled.
            if last_simulator_spillover is not None:
                simulator.cached_data_chunks = last_simulator_spillover
            logger.debug(
                "[RESTORE] Batch %d: State restored successfully - new_counter=%s",
                batch.batch_index,
                simulator.counter,
            )
        except Exception as e:
            logger.error("Failed to restore batch state: %s", e)
            raise ValueError(f"Failed to restore state for batch {batch.batch_index}") from e
    else:
        logger.debug(
            "[RESTORE] Batch %d: No pre-batch state snapshot available (fresh generation)",
            batch.batch_index,
        )


def save_batch_metadata(
    simulator: Simulator,
    batch: SimulationBatch,
    metadata_directory: Path,
    batch_data: object,
    output_files: list[Path],
    pre_batch_state: dict[str, Any] | None = None,
) -> None:
    """Save batch metadata including pre-batch simulator state and all output files.

    The metadata file uses batch-indexed naming ({simulator_name}-{batch_index}.metadata.yaml)
    to provide a single source of truth for all outputs from that batch. This handles
    cases where a single batch generates multiple output files (e.g., one per detector).

    An index file is also maintained to enable quick lookup of metadata for a given data file.

    Args:
        simulator: Simulator instance
        batch: SimulationBatch
        metadata_directory: Directory to save metadata
        batch_data: Generated batch artifact used to derive output provenance
        output_files: List of Path objects for all output files generated by this batch
        pre_batch_state: State of simulator before batch generation (for reproducibility).
                        If None, uses current simulator state.
    """
    metadata_directory.mkdir(parents=True, exist_ok=True)

    # Use provided pre_batch_state or current simulator state
    state_to_save = pre_batch_state if pre_batch_state is not None else simulator.state

    seed = _resolve_seed(simulator, batch)
    config_payload = _build_config_payload(batch, simulator)
    resolved_config, replayable = _build_resolved_config(simulator, config_payload)
    output_records = _build_output_records(simulator, batch, batch_data, output_files)
    metadata = create_batch_metadata(
        simulator_name=batch.simulator_name,
        batch_index=batch.batch_index,
        simulator_config=batch.simulator_config,
        globals_config=batch.globals_config,
        simulator_metadata=simulator.metadata,
        pre_batch_state=state_to_save,
        source=batch.source,
        author=batch.author,
        email=batch.email,
        config_payload=config_payload,
        resolved_config=resolved_config,
        replayable=replayable,
        config_sha256=batch.config_sha256,
        seed=seed,
        segment_seeds=_resolve_segment_seeds(simulator, batch, seed),
        population=_build_population_section(simulator, batch),
        signal=_build_signal_section(simulator, batch),
        noise=_build_noise_section(simulator, batch),
        outputs=[artifact.record for artifact in output_records],
        host=_get_host_metadata(),
        environment=capture_environment(),
    )

    # Add output files to metadata for easy discovery
    # Store just the file names, not full paths
    metadata["output_files"] = [f.name for f in output_files]

    # Embed the record in the artifacts it describes, so a file handed to another pipeline says
    # which run wrote it. One record, written to both places, rather than two constructions that
    # can disagree -- and the embedded copy withholds the injection parameters unless this run
    # asked for them. Before the hashes below, because it changes the container bytes.
    include_injection_parameters = _includes_injection_parameters(batch)
    for output_file in output_files:
        embed_metadata_record(
            output_file,
            metadata,
            include_injection_parameters=include_injection_parameters,
        )

    # Compute and add file hashes for integrity checking. Two hashes are kept:
    #   * file_hashes    -- raw container bytes (exact-file integrity)
    #   * content_hashes -- decoded scientific content, stable across write-time
    #                       and frame-library version (reproducibility check)
    _hash_output_records(output_records)
    metadata["outputs"] = [artifact.record for artifact in output_records]
    file_hashes = {}
    content_hashes = {}
    for output_file in output_files:
        try:
            file_hash = compute_file_hash(output_file)
            file_hashes[output_file.name] = file_hash
            logger.debug("Compute hash for %s: %s", output_file.name, file_hash)
        except OSError as e:
            logger.warning("Failed to compute hash for %s: %s", output_file.name, e)
            # Continue without failing - metadata is still useful
        content_hash = compute_content_hash(output_file)
        if content_hash is not None:
            content_hashes[output_file.name] = content_hash

    metadata["file_hashes"] = file_hashes
    metadata["content_hashes"] = content_hashes

    metadata_file_name = f"{batch.simulator_name}-{batch.batch_index}.metadata.json"
    metadata_file = metadata_directory / metadata_file_name
    logger.debug("Saving batch metadata to %s with %d output files", metadata_file, len(output_files))

    save_metadata_record(metadata=metadata, metadata_file=metadata_file)

    # Update the metadata index for quick lookup
    update_metadata_index(metadata_directory, output_files, metadata_file_name)

    # Update the truth indexes (event id -> containing frame file(s)) for event->frame lookup.
    # One per producer: signals into `signal_index.yaml`, glitches into `glitch_index.yaml`. A
    # batch that injected neither writes neither.
    update_signal_index(metadata_directory, metadata, metadata_file_name)
    update_glitch_index(metadata_directory, metadata, metadata_file_name)


def process_batch(
    simulator: Simulator,
    batch_data: object,
    batch: SimulationBatch,
    output_directory: Path,
    overwrite: bool,
) -> list[Path]:
    """Process and save a single batch of generated data.

    A single batch may generate multiple output files (e.g., one per detector).
    This function handles both single and multiple output files.

    Args:
        simulator: Simulator instance
        batch_data: Generated batch data (may contain multiple outputs)
        batch: SimulationBatch metadata
        output_directory: Directory for output files
        overwrite: Whether to overwrite existing files

    Returns:
        List of Path objects for all generated output files
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(batch_data, AdapterOrchestrationResult):
        if not isinstance(batch.simulator_config, OrchestrationConfig):
            raise TypeError("Adapter orchestration results require an OrchestrationConfig batch.")

        signal_output_files: list[Path] = []
        if batch.simulator_config.signal is not None and batch_data.signal_segment is not None:
            signal_output = batch.simulator_config.signal.output
            logger.debug(
                "[PROCESS] Batch %s: Saving signal output - counter=%s, template=%s",
                batch.batch_index,
                simulator.counter,
                signal_output.file_name,
            )
            signal_output_files = _resolve_output_paths(
                file_name_template=signal_output.file_name,
                simulator=simulator,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
            )
            simulator.save_data(
                data=batch_data.signal_segment,
                file_name=signal_output.file_name,
                output_directory=cast(AdapterOrchestrator, simulator).signal_output_directory(),
                overwrite=overwrite,
                **cast(AdapterOrchestrator, simulator).signal_output_arguments(),
            )

        noise_output_files: list[Path] = []
        if batch_data.noise_result is not None:
            noise_output_files = list(batch_data.noise_result.output_paths.values())
            missing_noise_outputs = [path for path in noise_output_files if not path.exists()]
            if missing_noise_outputs:
                raise FileNotFoundError(
                    "Noise adapter reported output files that do not exist: "
                    + ", ".join(str(path) for path in missing_noise_outputs)
                )

        logger.debug(
            "[PROCESS] Batch %s: adapter outputs - signal=%d files, noise=%d files",
            batch.batch_index,
            len(signal_output_files),
            len(noise_output_files),
        )
        return [*signal_output_files, *noise_output_files]

    if isinstance(batch_data, SimulationResult):
        output_files_list = list(batch_data.output_paths.values())
        missing_outputs = [path for path in output_files_list if not path.exists()]
        if missing_outputs:
            raise FileNotFoundError(
                "Noise adapter reported output files that do not exist: "
                + ", ".join(str(path) for path in missing_outputs)
            )
        logger.debug(
            "[PROCESS] Batch %s: Using upstream-written outputs - %s",
            batch.batch_index,
            [str(path.name) for path in output_files_list],
        )
        return output_files_list

    # Build output configuration
    output_config = batch.simulator_config.output
    logger.debug(
        "[PROCESS] Batch %s: Saving data - counter=%s, file_template=%s",
        batch.batch_index,
        simulator.counter,
        output_config.file_name,
    )
    file_name_template = output_config.file_name
    output_args = output_config.arguments.copy() if output_config.arguments else {}

    # Save data with output directory
    logger.debug(
        "Saving batch data for %s batch %d",
        batch.simulator_name,
        batch.batch_index,
    )

    # Resolve the output file names (may be multiple if template contains arrays)
    output_files = expand_template_variables(value=file_name_template, simulator_instance=simulator)

    # Normalize to list of Paths
    if isinstance(output_files, str):
        output_files_list = [output_directory / Path(output_files)]
    else:
        # If it's an array (multiple detectors), flatten it
        output_files_list = [output_directory / Path(str(f)) for f in np.array(output_files).flatten()]

    logger.debug(
        "[PROCESS] Batch %s: Resolved filenames - %s", batch.batch_index, [str(f.name) for f in output_files_list]
    )

    simulator.save_data(
        data=batch_data,
        file_name=file_name_template,
        output_directory=output_directory,
        overwrite=overwrite,
        **output_args,
    )

    logger.debug("[PROCESS] Batch %s: Data saved - counter=%s", batch.batch_index, simulator.counter)

    return output_files_list


def _resolve_output_paths(file_name_template: str, simulator: Simulator, output_directory: Path) -> list[Path]:
    """Resolve one or more concrete output paths for a template."""
    output_files = expand_template_variables(value=file_name_template, simulator_instance=simulator)
    if isinstance(output_files, str):
        return [output_directory / Path(output_files)]
    return [output_directory / Path(str(f)) for f in np.array(output_files).flatten()]


def setup_signal_handlers(checkpoint_dirs: list[Path]) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        checkpoint_dirs: List of checkpoint directories to clean up
    """

    def cleanup_checkpoints():
        """Clean up temporary checkpoint files."""
        for checkpoint_dir in checkpoint_dirs:
            for backup_file in checkpoint_dir.glob("*.bak"):
                try:
                    backup_file.unlink()
                    logger.debug("Cleaned up backup file: %s", backup_file)
                except OSError as e:
                    logger.warning("Failed to clean up backup file %s: %s", backup_file, e)

    atexit.register(cleanup_checkpoints)
    signal.signal(signal.SIGINT, handle_signal(cleanup_checkpoints))
    signal.signal(signal.SIGTERM, handle_signal(cleanup_checkpoints))


def validate_plan(plan: SimulationPlan) -> None:
    """Validate simulation plan before execution.

    Args:
        plan: SimulationPlan to validate

    Raises:
        ValueError: If plan validation fails
    """
    logger.info("Validating simulation plan with %d batches", plan.total_batches)

    if plan.total_batches == 0:
        raise ValueError("Simulation plan contains no batches")

    # Validate each batch
    for batch in plan.batches:
        if not batch.simulator_name:
            raise ValueError("Batch has empty simulator name")
        if batch.batch_index < 0:
            raise ValueError(f"Batch {batch.batch_index} has invalid index")

        if isinstance(batch.simulator_config, OrchestrationConfig):
            if batch.simulator_config.signal is not None and not batch.simulator_config.signal.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing signal output file_name")
            if batch.simulator_config.noise is not None and not batch.simulator_config.noise.output.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing noise output file_name")
        else:
            output_config = batch.simulator_config.output
            if not output_config.file_name:
                raise ValueError(f"Batch {batch.simulator_name}-{batch.batch_index} missing file_name")

    logger.info("Simulation plan validation completed successfully")


def _resolve_recorded_output_paths(metadata: dict[str, Any], working_directory: str | None) -> list[Path]:
    """Resolve the output file paths recorded in a batch's metadata.

    Paths in ``outputs[].path`` are stored relative to the working directory (see
    ``_to_path_string``); resolve them back against it so existence can be checked.
    """
    base = Path(working_directory) if working_directory else None
    paths: list[Path] = []
    for record in metadata.get("outputs", []) or []:
        raw = record.get("path") if isinstance(record, dict) else None
        if not raw:
            continue
        path = Path(raw)
        if base is not None and not path.is_absolute():
            path = base / path
        paths.append(path)
    return paths


def _referenced_population_files(plan: SimulationPlan) -> list[str]:
    """Every population source the plan's batches name, for the run's identity.

    Read from each batch's own config rather than the first: a plan assembled from several metadata
    records can name more than one catalogue, and taking one of them would let the rest change
    unnoticed -- the same reasoning as hashing every batch's config rather than the first.

    Only the population is collected. Other paths a config can reference are out of scope here, and
    :func:`run_fingerprint` says so rather than implying they are covered.
    """
    references: list[str] = []
    for batch in plan.batches:
        population = getattr(batch.simulator_config, "population", None)
        if population is None:
            continue
        source = (getattr(population, "arguments", None) or {}).get("path")
        if source:
            references.append(str(source))
    return references


def _batch_outputs_present(batch: SimulationBatch, metadata_directory: Path) -> bool | None:
    """Return whether the outputs recorded for ``batch`` all exist on disk.

    Returns:
        ``True`` if the batch metadata exists and every recorded output is present,
        ``False`` if the metadata exists but one or more recorded outputs are missing,
        ``None`` if the batch metadata file itself is missing.
    """
    metadata_file = metadata_directory / f"{batch.simulator_name}-{batch.batch_index}.metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with metadata_file.open("r") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as error:
        logger.warning(
            "Failed to read metadata for batch %d during checkpoint reconciliation: %s", batch.batch_index, error
        )
        return None
    working_directory = getattr(batch.globals_config, "working_directory", None)
    recorded = _resolve_recorded_output_paths(metadata, working_directory)
    if not recorded:
        # No output paths recorded to verify; trust the checkpoint for this batch.
        return True
    missing = [path for path in recorded if not path.exists()]
    if missing:
        logger.warning(
            "Checkpoint marks batch %d complete but output(s) are missing: %s",
            batch.batch_index,
            ", ".join(str(path) for path in missing),
        )
        return False
    return True


def reconcile_completed_batches(
    plan: SimulationPlan,
    metadata_directory: Path,
    completed_batch_indices: set[int],
) -> set[int]:
    """Prune checkpointed batches whose outputs are missing from disk.

    Resume restores each simulator's state from the last snapshot it left and assumes
    the completed batches form a contiguous prefix (the orchestration noise stream is
    sequential/stateful). So the completed set is validated in order and truncated at
    the FIRST batch whose recorded outputs (or metadata) are missing: that batch and
    every later batch are dropped so they get re-run.

    Returns the validated contiguous prefix of ``completed_batch_indices``.
    """
    if not completed_batch_indices:
        return set()
    batch_by_index = {batch.batch_index: batch for batch in plan.batches}
    valid: set[int] = set()
    for index in sorted(completed_batch_indices):
        batch = batch_by_index.get(index)
        if batch is None:
            logger.warning("Checkpoint references unknown batch %d; it and later batches will be re-run.", index)
            break
        present = _batch_outputs_present(batch, metadata_directory)
        if present is None:
            logger.warning(
                "Checkpoint marks batch %d complete but its metadata is missing; it and later batches will be re-run.",
                index,
            )
            break
        if not present:
            break
        valid.add(index)
    return valid


def execute_plan(  # noqa: PLR0915
    plan: SimulationPlan,
    output_directory: Path,
    metadata_directory: Path,
    overwrite: bool,
    # Keyword-only from here. Inserting `ignore_checkpoint` before `max_retries` made a positional
    # call bind the retry count to it -- any non-zero count is truthy and would silently skip the
    # checkpoint, which is the failure this whole change exists to prevent.
    *,
    ignore_checkpoint: bool = False,
    max_retries: int = 3,
) -> None:
    """Execute a complete simulation plan.

    The key insight: Simulators are stateful objects. Each simulator is instantiated
    once and then generates multiple batches by calling next() repeatedly. State
    (RNG, counters, filters) accumulates across batches.

    Checkpoint behavior:
    1. After each successfully completed batch, save checkpoint with updated state
    2. Checkpoint contains: completed batch indices, simulator state
    3. On next run, already-completed batches are skipped (resumption)
    4. On successful completion of all batches, checkpoint is cleaned up

    Workflow:
    1. Group batches by simulator name
    2. Load checkpoint to find already-completed batches
    3. For each simulator:
       a. Create ONE simulator instance
       b. For each batch of that simulator:
          - Skip if already completed (from checkpoint)
          - Restore state if reproducing from metadata
          - Call next(simulator) to generate batch (increments state)
          - Save batch output and metadata
          - Save checkpoint with updated state (for resumption)

    Args:
        plan: SimulationPlan to execute
        output_directory: Directory for output files
        metadata_directory: Directory for metadata files
        overwrite: Whether to overwrite existing files
        ignore_checkpoint: Discard any checkpoint in the directory and start fresh. The way past
            a refused resume for a caller that cannot delete the file by hand.
        max_retries: Maximum retries per batch
    """
    logger.info("Executing simulation plan: %d batches", plan.total_batches)

    validate_plan(plan)
    setup_signal_handlers([plan.checkpoint_directory] if plan.checkpoint_directory else [])

    # Initialize checkpoint manager for resumption support
    checkpoint_manager = CheckpointManager(plan.checkpoint_directory)
    # One decode for the whole setup. The file now carries the spillover -- 131 MB of base64 for a
    # 1000 s tail -- and every `load_checkpoint` decodes all of it, so each convenience getter used
    # here would pay that again before the run started.
    # `--ignore-checkpoint` discards it here rather than deleting the file: the refusal below is a
    # dead end for anything that cannot answer a prompt -- an automated campaign would fail on a
    # stale file with no way forward but manual intervention -- and deleting on the user's behalf is
    # the one action that cannot be undone.
    checkpoint = {} if ignore_checkpoint else (checkpoint_manager.load_checkpoint() or {})
    if ignore_checkpoint:
        logger.warning("Ignoring any checkpoint in %s: --ignore-checkpoint was given.", plan.checkpoint_directory)
    # Checked before anything is read from it. A checkpoint another configuration wrote will
    # otherwise be believed: the batches it records as complete are skipped and their outputs never
    # produced, with no warning and exit code 0.
    # Not `batch.config_sha256` on its own: that hashes the config *file*, so the same file run with
    # a different `--output-dir` fingerprints identically and the guard waves it through -- measured
    # at 2 frames where a clean run writes 3. The identity has to include where the outputs go.
    # The population file's *content*, not only its path: a config names its catalogue by name, so
    # swapping that file's bytes left this identity unchanged and a resume mixed two catalogues into one
    # run -- measured at batch 0 holding the old catalogue's event while batches 1 and 2 held the new
    # one's, exit code 0.
    referenced_populations = _referenced_population_files(plan)
    plan_sha256 = run_fingerprint(
        [batch.config_sha256 for batch in plan.batches],
        output_directory,
        metadata_directory,
        referenced_populations,
    )
    if checkpoint:
        require_matching_config(checkpoint.get("config_sha256"), plan_sha256, checkpoint_manager.checkpoint_file)
    # A set, matching what `get_completed_batch_indices` returned: it is compared against
    # `reconcile_completed_batches`'s output below, and a list never equals a set, which silently
    # sends every resume down the "outputs are missing" branch.
    loaded_batch_indices = set(checkpoint.get("completed_batch_indices") or [])
    resuming = bool(loaded_batch_indices)
    # Said on a resume, because that is when an unverifiable population can silently mix two catalogues:
    # the fingerprints match whatever the bytes were, so this check is the only signal the operator gets.
    report_unverified_inputs(referenced_populations, resuming)

    # Reconcile the checkpoint against the filesystem: a batch may be recorded as
    # completed while its output is missing (partial write at interrupt, an external
    # move/backup, fs hiccup). Skipping such a batch would silently drop a file.
    completed_batch_indices = reconcile_completed_batches(plan, metadata_directory, loaded_batch_indices)

    # One tail per simulator -- state and spillover -- carried across the whole run and written
    # back with every checkpoint. A single tail would be overwritten by whichever simulator finished
    # most recently, taking an earlier one's RNG state and the tail of any signal crossing its final
    # boundary with it.
    simulator_tails: dict[str, dict[str, Any]] = {}
    if not resuming:
        logger.debug("No checkpoint found or no batches completed yet")
    elif completed_batch_indices == loaded_batch_indices:
        logger.info("Loaded checkpoint: %d batches already completed", len(completed_batch_indices))
        # From the single decode above, already normalised by `load_checkpoint`. The per-batch
        # scoping the getter would apply is done at the restore call instead, from these values.
        simulator_tails = checkpoint.get("simulator_tails") or {}
    else:
        # One or more checkpointed batches are missing their outputs. The checkpoint
        # only holds each simulator's *last* state, so an interior batch cannot be
        # regenerated in isolation; discard the checkpoint and regenerate from the
        # first batch to keep the simulator/noise-stream/RNG state consistent.
        logger.warning(
            "Checkpoint listed %d completed batch(es) but only the first %d still have all outputs "
            "on disk; regenerating the simulation from the start to restore the missing output(s).",
            len(loaded_batch_indices),
            len(completed_batch_indices),
        )
        completed_batch_indices = set()
        # Dropped rather than carried: every batch is regenerated, so no simulator resumes from a
        # tail, and writing these back out would describe a run that no longer exists.
        simulator_tails = {}

    # Group batches by simulator name to execute sequentially per simulator
    simulator_batches: dict[str, list[SimulationBatch]] = {}
    for batch in plan.batches:
        if batch.simulator_name not in simulator_batches:
            simulator_batches[batch.simulator_name] = []
        simulator_batches[batch.simulator_name].append(batch)

    logger.info("Executing %d simulators", len(simulator_batches))

    with tqdm(total=plan.total_batches, desc="Executing simulation plan") as p_bar:
        for simulator_name, batches in simulator_batches.items():
            logger.info("Starting simulator: %s with %d batches", simulator_name, len(batches))

            # Create ONE simulator instance for all batches of this simulator
            # Extract global simulator arguments from the first batch's global config
            global_sim_args = batches[0].globals_config.simulator_arguments if batches else {}
            simulator = instantiate_simulator(batches[0].simulator_config, simulator_name, global_sim_args)

            # Process batches sequentially, maintaining state across them
            for batch_idx, batch in enumerate(batches):
                # Skip batches that were already completed AND whose outputs were verified
                # on disk during reconciliation (for resumption after interrupt).
                if batch.batch_index in completed_batch_indices:
                    logger.info(
                        "Skipping batch %d (already completed from checkpoint)",
                        batch.batch_index,
                    )
                    continue

                # On resume, any output present for a batch we are about to run is an
                # unverified leftover (orphan/partial) from the interrupted attempt, not
                # user data, so allow overwriting it even without --overwrite.
                batch_overwrite = overwrite or resuming

                try:
                    logger.debug(
                        "Executing batch %d/%d for simulator %s",
                        batch_idx + 1,
                        len(batches),
                        simulator_name,
                    )

                    # Capture pre-batch state first for potential retries
                    logger.debug(
                        "[EXECUTE] Batch %s: Before restore - counter=%s, has_state_snapshot=%s",
                        batch.batch_index,
                        simulator.counter,
                        batch.has_state_snapshot(),
                    )
                    # This simulator's own tail, never another's. A plan can execute several, and
                    # an unscoped hand-off puts one simulator's RNG state behind another's segment
                    # and one simulator's spillover into another's data -- real strain of the right
                    # shape, in the wrong place. The spillover is also only valid for the batch
                    # immediately after the one that produced it.
                    state_for_batch, spillover_for_batch = resume_tail(
                        simulator_tails, batch.simulator_name, batch.batch_index
                    )
                    # Which batch left that state, so the restore compares batches rather than
                    # falling back to the counter proxy -- which refuses a valid tail as soon as a
                    # simulator's counter and the plan's batch numbering disagree, as they do for
                    # every simulator after the first.
                    tail_batch_index = (simulator_tails.get(batch.simulator_name) or {}).get("batch_index")
                    restore_batch_state(simulator, batch, state_for_batch, spillover_for_batch, tail_batch_index)
                    logger.debug("[EXECUTE] Batch %s: After restore - counter=%s", batch.batch_index, simulator.counter)
                    pre_batch_state = copy.deepcopy(simulator.state)
                    # Spillover too, and separately, because it is not part of `state`. `simulate`
                    # consumes `cached_data_chunks` and replaces it with the new tail, so a retry
                    # after a failed write would re-run against consumed chunks and produce
                    # different data than the first attempt -- silently, since a retry that
                    # succeeds looks like a success.
                    pre_batch_spillover = copy.deepcopy(getattr(simulator, "cached_data_chunks", None))
                    logger.debug(
                        "[EXECUTE] Batch %s: Captured pre_batch_state - keys=%s",
                        batch.batch_index,
                        list(pre_batch_state.keys()),
                    )

                    def execute_batch(
                        _simulator=simulator,
                        _batch=batch,
                        _output_directory=output_directory,
                        _pre_batch_state=pre_batch_state,
                        _overwrite=batch_overwrite,
                    ):
                        """Execute a single batch with state management."""
                        set_batch_context = getattr(_simulator, "set_batch_context", None)
                        if callable(set_batch_context):
                            set_batch_context(
                                batch=_batch,
                                output_directory=_output_directory,
                                overwrite=_overwrite,
                            )

                        # Generate data by calling next() - this advances simulator state
                        logger.debug("[BATCH] %s: Before next() - counter=%s", _batch.batch_index, _simulator.counter)
                        batch_data = _simulator.simulate()
                        logger.debug("[BATCH] %s: After next() - counter=%s", _batch.batch_index, _simulator.counter)

                        # Save the generated data and get all output file paths
                        output_files = process_batch(
                            simulator=_simulator,
                            batch_data=batch_data,
                            batch=_batch,
                            output_directory=_output_directory,
                            overwrite=_overwrite,
                        )

                        # Only save metadata if data save succeeded
                        # This ensures metadata only exists for successfully saved data
                        save_batch_metadata(
                            _simulator,
                            _batch,
                            metadata_directory,
                            batch_data,
                            output_files,
                            pre_batch_state=_pre_batch_state,
                        )
                        # Update the state after successful save
                        _simulator.update_state()

                    def restore_state_for_retry(
                        _simulator=simulator,
                        _pre_batch_state=pre_batch_state,
                        _pre_batch_spillover=pre_batch_spillover,
                    ):
                        """Restore simulator state to pre-batch state before retry."""
                        _simulator.state = copy.deepcopy(_pre_batch_state)
                        if _pre_batch_spillover is not None:
                            _simulator.cached_data_chunks = copy.deepcopy(_pre_batch_spillover)

                    # Execute batch with retry mechanism that restores state on failure
                    retry_with_backoff(
                        execute_batch,
                        max_retries=max_retries,
                        state_restore_func=restore_state_for_retry,
                    )

                    # After successful completion, save checkpoint with updated state
                    # At this point, state has been incremented by next() -> update_state()
                    # Save checkpoint to enable resumption if interrupted before next batch
                    completed_batch_indices.add(batch.batch_index)
                    state_snapshot = copy.deepcopy(simulator.state)
                    # Beside the state, not inside it: `state` also goes into every batch metadata
                    # record, and spillover is raw samples. See `save_checkpoint`.
                    spillover_snapshot = copy.deepcopy(getattr(simulator, "cached_data_chunks", None))
                    # Replaces only this simulator's entry, so the tails of the simulators that ran
                    # before it are still in the checkpoint after this one completes.
                    simulator_tails[simulator_name] = {
                        "batch_index": batch.batch_index,
                        "state": state_snapshot,
                        "spillover": spillover_snapshot,
                    }
                    checkpoint_manager.save_checkpoint(
                        completed_batch_indices=sorted(completed_batch_indices),
                        last_simulator_name=simulator_name,
                        last_completed_batch_index=batch.batch_index,
                        last_simulator_state=state_snapshot,
                        last_simulator_spillover=spillover_snapshot,
                        config_sha256=plan_sha256,
                        simulator_tails=simulator_tails,
                    )
                    logger.debug(
                        "Checkpoint saved after batch %d - state counter=%s",
                        batch.batch_index,
                        simulator.counter,
                    )
                    p_bar.update(1)

                except Exception as e:
                    logger.error(
                        "Failed to execute batch %d for simulator %s after %s: %s",
                        batch.batch_index,
                        simulator_name,
                        # Not every failure is retried -- a stale index read and an unrecorded
                        # digest are raised straight past the loop -- and claiming retries that
                        # never happened misleads exactly when someone is reading logs in anger.
                        "no retries (not retryable)" if _is_not_worth_retrying(e) else f"{max_retries} retries",
                        e,
                    )
                    raise

    # All batches completed successfully - clean up checkpoint files
    checkpoint_manager.cleanup()
    logger.info("All batches completed successfully. Checkpoint files cleaned up.")
