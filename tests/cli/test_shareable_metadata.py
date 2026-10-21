"""The record that travels with a data set names no person, no machine and no directory.

A run's metadata record used to carry the user name of whoever ran it, the absolute working
directory (and through it, as often as not, a home directory), the paths of the inputs, and the exact
kernel build of the machine -- in the sidecar, in the indexes beside it, and in the copy embedded in
every HDF5 output. A data set handed to a collaboration carried all of it.

By default a record is now written as two files: the shareable record under its usual name, holding
nothing of the kind, and a private details file beside it holding exactly what was taken out. These
tests run a real orchestration (with stand-in backends) and check the guarantees from the outside:

* every key and every string value of the shareable record, of the embedded copy and of the indexes
  is scanned for an absolute path, the user name, the host name and the full platform string;
* the private file holds those, and the shareable record names it with its SHA-256;
* everything gwmock reads back -- ``merge``, ``validate``, the index rebuild, replay and checkpoint
  resume -- works with the private files deleted;
* ``globals.metadata.split: false`` writes the combined record, with the keys it always had.

The user name and host name are replaced with canaries, so that a scan for them cannot match by
coincidence, and the working directory is a subdirectory of the test's temporary directory with the
population file outside it, so both kinds of path -- inside the working directory and outside it --
are exercised.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from gwmock.cli.simulate import _simulate_impl
from tests.cli.test_cli_orchestration import _write_signal_file
from tests.cli.test_embedded_run_metadata import _config, _embedded_document

CANARY_USER = "canary-user-5d1e"
CANARY_HOST = "canary-host-8b27"

#: The top-level keys of the record v0.18.2 wrote for this configuration, measured on that release.
#: The opt-out must write exactly these.
COMBINED_TOP_LEVEL_KEYS = {
    "author",
    "batch_index",
    "config",
    "config_sha256",
    "content_hashes",
    "environment",
    "file_hashes",
    "globals_config",
    "gwmock_version",
    "host",
    "noise",
    "output_files",
    "outputs",
    "population",
    "pre_batch_state",
    "replayable",
    "schema_version",
    "seed",
    "segment_seeds",
    "signal",
    "simulator_config",
    "simulator_metadata",
    "simulator_name",
    "source",
    "subpackage_versions",
    "timestamp",
    "versions",
}
#: ``git_sha`` is left out when it cannot be read (gwmock not installed from a repository).
COMBINED_HOST_KEYS = {"platform", "python", "cpu", "git_sha"}


def _has_combined_host_keys(host: dict[str, Any]) -> bool:
    return set(host) in (COMBINED_HOST_KEYS, COMBINED_HOST_KEYS - {"git_sha"})


def _strings(document: Any, pointer: str = ""):
    """Yield ``(pointer, text)`` for every key and every string value in *document*."""
    if isinstance(document, dict):
        for key, value in document.items():
            yield f"{pointer}/{key} (key)", str(key)
            yield from _strings(value, f"{pointer}/{key}")
    elif isinstance(document, list):
        for index, item in enumerate(document):
            yield from _strings(item, f"{pointer}/{index}")
    elif isinstance(document, str):
        yield pointer, document


def _leaks(document: Any, tmp_path: Path) -> list[str]:
    """Return every place *document* names a path, the user, the host or the kernel build."""
    forbidden = [CANARY_USER, CANARY_HOST, str(tmp_path), platform.platform()]
    found = []
    for pointer, text in _strings(document):
        if os.path.isabs(text) or text.startswith("file://"):
            found.append(f"{pointer}: absolute path {text!r}")
        found.extend(f"{pointer}: contains {needle!r}" for needle in forbidden if needle in text)
    return found


def _private_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.private.*"))


@pytest.fixture
def canaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the user name and host name distinctive, so a scan for them means something."""
    monkeypatch.setattr("getpass.getuser", lambda: CANARY_USER)
    monkeypatch.setattr("socket.gethostname", lambda: CANARY_HOST)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    split: bool | None = None,
    signal_suffix: str = "hdf5",
) -> Path:
    """Run the two-batch orchestration in ``tmp_path/run`` and return its metadata directory.

    The population file sits in ``tmp_path``, outside the working directory, so its path cannot be
    made relative to it. ``split`` is written into the YAML under the key a user types.
    """
    working_directory = tmp_path / "run"
    working_directory.mkdir()
    raw = _config(working_directory, signal_suffix=signal_suffix).model_dump(by_alias=True, exclude_none=True)
    raw["orchestration"]["population"]["arguments"]["path"] = str(tmp_path / "population.h5")
    if split is not None:
        raw["globals"]["metadata"] = {"split": split}
    config_file = working_directory / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw, sort_keys=False))

    monkeypatch.setattr("gwmock.cli.adapter_orchestration.DetectorStrainStack.write", _write_signal_file)
    _simulate_impl(str(config_file), overwrite=True, metadata=True)
    return working_directory / "metadata"


def _record(metadata_directory: Path, batch: int = 0) -> dict[str, Any]:
    return json.loads((metadata_directory / f"orchestration-{batch}.metadata.json").read_text())


class TestTheShareableRecord:
    """What the sidecar under the usual name holds."""

    def test_it_names_no_path_user_host_or_kernel_build(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path)

        for batch in (0, 1):
            record = _record(metadata_directory, batch)
            assert record["config"], "an empty record would pass this scan trivially"
            assert _leaks(record, tmp_path) == []

    def test_its_paths_are_relative_to_the_working_directory(self, monkeypatch, tmp_path: Path, canaries) -> None:
        """Inside the working directory a path keeps its layout; outside it, its file name."""
        record = _record(_run(monkeypatch, tmp_path))

        assert record["config"]["globals"]["working-directory"] == "."
        assert record["globals_config"]["working-directory"] == "."
        assert record["config"]["orchestration"]["population"]["arguments"]["path"] == "population.h5"
        assert {output["path"] for output in record["outputs"]} == {
            "output/signal/signal-0.hdf5",
            "output/noise/noise-0.npy",
        }

    def test_it_has_no_author(self, monkeypatch, tmp_path: Path, canaries) -> None:
        record = _record(_run(monkeypatch, tmp_path))

        assert "author" not in record
        assert "email" not in record

    def test_its_host_describes_a_kind_of_machine(self, monkeypatch, tmp_path: Path, canaries) -> None:
        """Operating system, architecture, Python, git SHA and processor model -- not the kernel build."""
        host = _record(_run(monkeypatch, tmp_path))["host"]

        assert {"os", "architecture", "python", "cpu_model"} <= set(host)
        assert host["os"] == platform.system()
        assert host["architecture"] == platform.machine()
        assert host["platform"] == f"{platform.system()}-{platform.machine()}"
        assert host["python"] == platform.python_version()
        assert host["cpu_model"] is None or isinstance(host["cpu_model"], str)

    def test_it_names_its_private_file_and_that_files_digest(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path)
        reference = _record(metadata_directory)["private_metadata"]

        private_file = metadata_directory / reference["file"]
        assert private_file.name == "orchestration-0.private.json"
        assert hashlib.sha256(private_file.read_bytes()).hexdigest() == reference["sha256"]


class TestWhereElseTheRecordGoes:
    """The copy inside each output and the indexes are built from the shareable record."""

    def test_the_embedded_copy_is_the_shareable_one(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path)

        embedded = _embedded_document(metadata_directory.parent / "output" / "signal" / "signal-0.hdf5")

        assert embedded is not None, "nothing was embedded, so the scan below would prove nothing"
        assert embedded["config"]["globals"]["working-directory"] == "."
        assert "author" not in embedded
        assert _leaks(embedded, tmp_path) == []

    def test_the_indexes_name_nothing_private(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path)

        indexes = sorted(metadata_directory.glob("*index.yaml"))
        assert indexes
        for index in indexes:
            assert _leaks(yaml.safe_load(index.read_text()), tmp_path) == [], index.name


class TestThePrivateFile:
    """What was taken out of the shareable record is kept, beside it."""

    def test_it_holds_the_author_host_platform_and_paths(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path)
        private = json.loads((metadata_directory / "orchestration-0.private.json").read_text())

        assert private["author"] == CANARY_USER
        assert private["hostname"] == CANARY_HOST
        assert private["platform"] == platform.platform()
        assert private["working_directory"] == str(tmp_path / "run")
        assert private["values"]["/config/globals/working-directory"] == str(tmp_path / "run")
        assert private["values"]["/config/orchestration/population/arguments/path"] == str(tmp_path / "population.h5")
        assert private["values"]["/host"]["platform"] == platform.platform()

    def test_it_is_not_taken_for_a_record(self, monkeypatch, tmp_path: Path, canaries) -> None:
        """Every reader finds records by ``*.metadata.json``; the private file must not match."""
        metadata_directory = _run(monkeypatch, tmp_path)

        assert sorted(path.name for path in metadata_directory.glob("*.metadata.json")) == [
            "orchestration-0.metadata.json",
            "orchestration-1.metadata.json",
        ]
        assert [path.name for path in _private_files(metadata_directory)] == [
            "orchestration-0.private.json",
            "orchestration-1.private.json",
        ]

    def test_with_the_record_it_rebuilds_the_combined_layout(self, monkeypatch, tmp_path: Path, canaries) -> None:
        """Shareable plus private is the record the opt-out writes: the same keys, the same paths."""
        from gwmock.cli.utils.metadata import combine_metadata_records

        metadata_directory = _run(monkeypatch, tmp_path)
        shareable = _record(metadata_directory)
        private = json.loads((metadata_directory / "orchestration-0.private.json").read_text())

        combined = combine_metadata_records(shareable, private)

        assert {key for key, value in combined.items() if value is not None} == COMBINED_TOP_LEVEL_KEYS
        assert _has_combined_host_keys(combined["host"])
        assert combined["host"]["platform"] == platform.platform()
        assert combined["author"] == CANARY_USER
        assert combined["config"]["globals"]["working-directory"] == str(tmp_path / "run")


class TestTheOptOut:
    """``globals.metadata.split: false`` writes one combined record, as before the split existed."""

    def test_it_writes_the_combined_record_with_its_original_keys(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path, split=False)
        record = _record(metadata_directory)

        assert set(record) == COMBINED_TOP_LEVEL_KEYS
        assert _has_combined_host_keys(record["host"])
        assert record["host"]["platform"] == platform.platform()
        assert record["author"] == CANARY_USER
        assert record["config"]["globals"]["working-directory"] == str(tmp_path / "run")
        assert _private_files(metadata_directory) == []

    def test_it_embeds_the_combined_record(self, monkeypatch, tmp_path: Path, canaries) -> None:
        metadata_directory = _run(monkeypatch, tmp_path, split=False)

        embedded = _embedded_document(metadata_directory.parent / "output" / "signal" / "signal-0.hdf5")

        assert embedded is not None
        assert embedded["author"] == CANARY_USER
        assert embedded["config"]["globals"]["working-directory"] == str(tmp_path / "run")

    @pytest.mark.parametrize("value", [{"split": "sometimes"}, {"splitt": False}, "off"])
    def test_a_malformed_setting_is_refused(self, tmp_path: Path, value) -> None:
        """A misspelt key must not leave a run silently writing the layout its author turned off."""
        from pydantic import ValidationError

        from gwmock.cli.utils.config import Config

        raw = _config(tmp_path).model_dump(by_alias=True, exclude_none=True)
        raw["globals"]["metadata"] = value

        with pytest.raises(ValidationError):
            Config.model_validate(raw)


class TestEverythingGwmockReadsWorksWithoutThePrivateFiles:
    """The shareable records are what a recipient has; gwmock's own readers must need nothing else."""

    @staticmethod
    def _shared(monkeypatch, tmp_path: Path) -> Path:
        """Run, then move the data set elsewhere without its private files -- as a recipient gets it."""
        metadata_directory = _run(monkeypatch, tmp_path)
        received = tmp_path / "received"
        shutil.copytree(metadata_directory.parent, received)
        for private_file in _private_files(received / "metadata"):
            private_file.unlink()
        shutil.rmtree(metadata_directory.parent)
        return received

    def test_validate(self, monkeypatch, tmp_path: Path, canaries, capsys) -> None:
        from gwmock.cli.validate import validate_command

        received = self._shared(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)  # deliberately not the data set's directory

        validate_command([str(received / "output")], metadata_paths=[str(received / "metadata")])

        # Each output once: found through its record and through the directory scan, it is one file.
        assert "4/4 files passed validation" in capsys.readouterr().out

    def test_the_index_rebuild_matches_the_index_the_run_wrote(self, monkeypatch, tmp_path: Path, canaries) -> None:
        from gwmock.cli.simulate_utils import rebuild_truth_indexes

        received = self._shared(monkeypatch, tmp_path)
        written = (received / "metadata" / "signal_index.yaml").read_text()

        rebuild_truth_indexes(received / "metadata")

        assert yaml.safe_load((received / "metadata" / "signal_index.yaml").read_text()) == yaml.safe_load(written)

    def test_replay_plans_from_the_shareable_record(self, monkeypatch, tmp_path: Path, canaries) -> None:
        from gwmock.cli.utils.simulation_plan import create_plan_from_metadata_files

        received = self._shared(monkeypatch, tmp_path)

        plan = create_plan_from_metadata_files(
            [received / "metadata" / "orchestration-1.metadata.json"], tmp_path / "checkpoints"
        )

        (batch,) = plan.batches
        assert batch.batch_index == 1
        assert batch.globals_config.working_directory == "."
        assert batch.pre_batch_state, "replay needs the recorded state, which the shareable record keeps"

    def test_checkpoint_resume_finds_the_outputs(self, monkeypatch, tmp_path: Path, canaries) -> None:
        from gwmock.cli.simulate_utils import _batch_outputs_present
        from gwmock.cli.utils.simulation_plan import create_plan_from_metadata_files

        received = self._shared(monkeypatch, tmp_path)
        monkeypatch.chdir(received)
        plan = create_plan_from_metadata_files(
            [received / "metadata" / "orchestration-0.metadata.json"], tmp_path / "checkpoints"
        )

        assert _batch_outputs_present(plan.batches[0], received / "metadata") is True

    def test_merge_checks_its_inputs_against_shareable_records(self, tmp_path: Path, canaries) -> None:
        """``file_hashes`` is keyed by file name, so the check needs no path from the private file."""
        pytest.importorskip("gwpy")
        from gwmock.cli.merge import merge_command
        from tests.cli.test_merge import _frame

        sources = [_frame(tmp_path / "a.gwf", [1.0] * 16), _frame(tmp_path / "b.gwf", [2.0] * 16)]
        records = []
        for source in sources:
            record_file = tmp_path / f"{source.stem}.metadata.yaml"
            _write_shareable_record(source, record_file)
            records.append(str(record_file))
        assert _private_files(tmp_path) == []

        merge_command(sources, output=str(tmp_path / "merged.gwf"), metadata=records)

        merged = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert sorted(merged["source_files"]) == ["a.gwf", "b.gwf"]
        assert "author" not in merged
        assert _leaks(merged, tmp_path) == []
        private = yaml.safe_load((tmp_path / "merged.private.yaml").read_text())
        assert private["author"] == CANARY_USER


class TestAMergeOfCombinedRecords:
    """Sources written with the split off still yield a shareable merged record."""

    def test_the_sources_authors_hosts_and_paths_stay_private(self, tmp_path: Path, canaries) -> None:
        pytest.importorskip("gwpy")
        from gwmock.cli.merge import merge_command
        from gwmock.cli.utils.hash import compute_file_hash
        from tests.cli.test_merge import _frame

        sources = [_frame(tmp_path / "a.gwf", [1.0] * 16), _frame(tmp_path / "b.gwf", [2.0] * 16)]
        records = []
        for source in sources:
            record_file = tmp_path / f"{source.stem}.metadata.yaml"
            record_file.write_text(yaml.safe_dump(_combined_record(source, compute_file_hash(source))))
            records.append(str(record_file))

        merge_command(sources, output=str(tmp_path / "merged.gwf"), metadata=records)

        merged = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert merged["source_files"]["a.gwf"]["config"]["globals"]["working-directory"] == "."
        assert "author" not in merged["source_files"]["a.gwf"]
        assert _leaks(merged, tmp_path) == []


def _combined_record(source: Path, digest: str) -> dict[str, Any]:
    """A source record in the layout a run with the split off writes."""
    return {
        "author": CANARY_USER,
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu": "x86_64"},
        "config": {"globals": {"working-directory": str(source.parent)}},
        "outputs": [{"kind": "signal", "path": str(source)}],
        "file_hashes": {source.name: digest},
    }


def _write_shareable_record(source: Path, record_file: Path) -> None:
    """Write *source*'s record in the shareable layout, naming a private file that is not there."""
    from gwmock.cli.utils.hash import compute_file_hash

    record_file.write_text(
        yaml.safe_dump(
            {
                "config": {"globals": {"working-directory": "."}},
                "host": {"platform": f"{platform.system()}-{platform.machine()}", "python": "3", "cpu": "x"},
                "outputs": [{"kind": "signal", "path": source.name}],
                "file_hashes": {source.name: compute_file_hash(source)},
                "private_metadata": {"file": f"{source.stem}.private.yaml", "sha256": "0" * 64},
            }
        )
    )
