"""Unit tests for metadata utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwmock.cli.utils.hash import compute_content_hash, compute_file_hash
from gwmock.cli.utils.metadata import (
    embed_metadata_record,
    embeddable_metadata,
    load_metadata_with_external_state,
    save_metadata_with_external_state,
    without_injection_parameters,
)
from gwmock.strain_schema import read_run_metadata


class TestSaveMetadataWithExternalState:
    """Test save_metadata_with_external_state function."""

    def test_save_without_numpy_arrays(self, tmp_path):
        """Test saving metadata without numpy arrays."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_seed": 42, "counter": 10},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check no external files created
        assert not any(metadata_dir.glob("*.npy"))

        # Check YAML content
        with metadata_file.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded == metadata

    def test_save_with_numpy_arrays(self, tmp_path):
        """Test saving metadata with numpy arrays."""
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.random.rand(10, 10)
        rng_seed = 42
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {
                "rng_seed": rng_seed,
                "rng_state": array1,
                "filter_state": array2,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check external files created
        npy_files = list(metadata_dir.glob("*.npy"))
        expected_num_files = 2
        assert len(npy_files) == expected_num_files

        # Check YAML content has references
        with metadata_file.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded["simulator_name"] == "test_sim"
        assert loaded["batch_index"] == 0
        assert loaded["pre_batch_state"]["rng_seed"] == rng_seed
        assert loaded["pre_batch_state"]["rng_state"]["_external_file"] is True
        assert loaded["pre_batch_state"]["filter_state"]["_external_file"] is True

    def test_save_with_empty_pre_batch_state(self, tmp_path):
        """Test saving metadata with empty pre_batch_state."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Check YAML file exists
        assert metadata_file.exists()

        # Check no external files
        assert not any(metadata_dir.glob("*.npy"))

    def test_save_creates_metadata_dir(self, tmp_path):
        """Test that metadata directory is created if it doesn't exist."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_state": np.array([1, 2, 3])},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "new_metadata_dir"

        assert not metadata_dir.exists()
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)
        assert metadata_dir.exists()
        assert metadata_file.exists()


class TestLoadMetadataWithExternalState:
    """Test load_metadata_with_external_state function."""

    def test_load_without_numpy_arrays(self, tmp_path):
        """Test loading metadata without numpy arrays."""
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {"rng_seed": 42, "counter": 10},
        }
        metadata_file = tmp_path / "metadata.yaml"

        # Save directly
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        # Load
        loaded = load_metadata_with_external_state(metadata_file)

        assert loaded == metadata

    def test_load_with_numpy_arrays(self, tmp_path):
        """Test loading metadata with numpy arrays."""
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.random.rand(3, 3)
        rng_seed = 42
        metadata = {
            "simulator_name": "test_sim",
            "batch_index": 0,
            "pre_batch_state": {
                "rng_seed": rng_seed,
                "rng_state": array1,
                "filter_state": array2,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        # Save with external state
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Load
        loaded = load_metadata_with_external_state(metadata_file, metadata_dir)

        assert loaded["simulator_name"] == "test_sim"
        assert loaded["batch_index"] == 0
        assert loaded["pre_batch_state"]["rng_seed"] == rng_seed
        np.testing.assert_array_equal(loaded["pre_batch_state"]["rng_state"], array1)
        np.testing.assert_array_equal(loaded["pre_batch_state"]["filter_state"], array2)

    def test_load_missing_metadata_file(self, tmp_path):
        """Test loading with missing metadata file raises FileNotFoundError."""
        metadata_file = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            load_metadata_with_external_state(metadata_file)

    def test_load_missing_external_file(self, tmp_path):
        """Test loading with missing external file raises FileNotFoundError."""
        # Create metadata with external reference
        metadata = {
            "simulator_name": "test_sim",
            "pre_batch_state": {
                "rng_state": {
                    "_external_file": True,
                    "dtype": "int64",
                    "shape": [5],
                    "size_bytes": 40,
                    "file": "missing_state_rng_state.npy",
                }
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        with metadata_file.open("w") as f:
            yaml.safe_dump(metadata, f)

        with pytest.raises(FileNotFoundError, match="External state file not found"):
            load_metadata_with_external_state(metadata_file)

    def test_round_trip_save_load(self, tmp_path):
        """Test round-trip: save then load should match original."""
        original_array = np.random.rand(5, 5)
        original_metadata = {
            "simulator_name": "test_sim",
            "batch_index": 1,
            "pre_batch_state": {
                "rng_seed": 123,
                "rng_state": original_array,
                "counter": 5,
            },
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "metadata"

        # Save
        save_metadata_with_external_state(original_metadata, metadata_file, metadata_dir)

        # Load
        loaded_metadata = load_metadata_with_external_state(metadata_file, metadata_dir)

        # Check non-array fields
        assert loaded_metadata["simulator_name"] == original_metadata["simulator_name"]
        assert loaded_metadata["batch_index"] == original_metadata["batch_index"]
        assert loaded_metadata["pre_batch_state"]["rng_seed"] == original_metadata["pre_batch_state"]["rng_seed"]
        assert loaded_metadata["pre_batch_state"]["counter"] == original_metadata["pre_batch_state"]["counter"]

        # Check array
        np.testing.assert_array_equal(
            loaded_metadata["pre_batch_state"]["rng_state"], original_metadata["pre_batch_state"]["rng_state"]
        )

    def test_load_with_custom_metadata_dir(self, tmp_path):
        """Test loading with custom metadata directory."""
        array = np.array([10, 20, 30])
        metadata = {
            "simulator_name": "test_sim",
            "pre_batch_state": {"array": array},
        }
        metadata_file = tmp_path / "metadata.yaml"
        metadata_dir = tmp_path / "custom_metadata"

        # Save
        save_metadata_with_external_state(metadata, metadata_file, metadata_dir)

        # Load with custom dir
        loaded = load_metadata_with_external_state(metadata_file, metadata_dir)

        np.testing.assert_array_equal(loaded["pre_batch_state"]["array"], array)


def _refuse_non_standard(token: str):
    """`json.loads` hook that rejects the tokens JSON does not have, which Python accepts anyway."""
    raise AssertionError(f"the document is not valid JSON: it contains the bare token {token}")


class TestWithoutInjectionParameters:
    """The sanitiser: what it removes, where it looks, and what it leaves alone."""

    def test_it_removes_the_documented_copy(self):
        document = {"signal": {"backend": "b", "injections": [{"event_id": 0}]}}

        assert without_injection_parameters(document) == {"signal": {"backend": "b"}}

    def test_it_removes_every_copy_at_any_depth(self):
        """The orchestrator writes the same list into three places; removing one is removing none."""
        document = {
            "signal": {"injections": [{"event_id": 0}], "metadata": {"injections": [{"event_id": 0}]}},
            "simulator_metadata": {"orchestration": {"signal": {"injections": [{"event_id": 0}]}}},
        }

        sanitised = without_injection_parameters(document)

        assert json.dumps(sanitised).count("event_id") == 0

    def test_it_reaches_inside_lists(self):
        document = {"batches": [{"injections": [1]}, {"injections": [2]}]}

        assert without_injection_parameters(document) == {"batches": [{}, {}]}

    def test_it_leaves_the_callers_document_alone(self):
        """The caller's copy is the one that goes into the sidecar, parameters and all."""
        document = {"signal": {"injections": [{"event_id": 0}]}}

        without_injection_parameters(document)

        assert document["signal"]["injections"] == [{"event_id": 0}]


class TestEmbeddableMetadata:
    """The copy that may go inside a data file, and the four things it drops."""

    @staticmethod
    def _record():
        return {
            "schema_version": "1.5.0",
            "config": {"globals": {"working-directory": "."}},
            "pre_batch_state": {"counter": 3},
            "signal": {"injections": [{"event_id": 0, "parameters": {"detector_frame_mass_1": 30.0}}]},
            "outputs": [{"kind": "signal", "path": "a.hdf5", "sha256": "x", "content_sha256": "y"}],
            "file_hashes": {"a.hdf5": "x"},
            "content_hashes": {"a.hdf5": "y"},
        }

    def test_the_injection_parameters_are_dropped_by_default(self):
        assert "injections" not in embeddable_metadata(self._record())["signal"]

    def test_the_injection_parameters_are_kept_on_request(self):
        document = embeddable_metadata(self._record(), include_injection_parameters=True)

        assert document["signal"]["injections"][0]["parameters"]["detector_frame_mass_1"] == 30.0

    def test_the_replay_state_is_dropped(self):
        """It is arrays the sidecar externalises to `.npy` files an embedded copy cannot point at."""
        assert "pre_batch_state" not in embeddable_metadata(self._record())

    def test_the_replay_state_is_dropped_at_every_depth(self):
        """`gwmock merge` nests each source's whole record under `source_files`, so a top-level-only
        drop put every source's RNG state into the merged artifact -- and RNG state beside the
        configuration regenerates the run, which is to say the injections the same document is at
        pains to withhold."""
        record = {"type": "merged", "source_files": {"a.hdf5": self._record()}}

        document = embeddable_metadata(record)

        assert "pre_batch_state" not in json.dumps(document)

    def test_a_sources_own_hashes_are_kept(self):
        """The counterpart, and the reason the hash maps are *not* stripped at depth: a nested one
        is the digest of an input, which says what this artifact was made from. Nothing in this file
        can invalidate it, so it is provenance to carry rather than a self-reference to remove."""
        record = {"type": "merged", "source_files": {"a.hdf5": self._record()}}

        document = embeddable_metadata(record)

        assert document["source_files"]["a.hdf5"]["file_hashes"] == {"a.hdf5": "x"}
        assert "file_hashes" not in document

    @pytest.mark.parametrize(
        ("value", "why"),
        [
            (float("inf"), "unlimited max_samples is stored as np.inf"),
            (float("-inf"), "the mirror case"),
            (float("nan"), "a backend may report one from a failed fit"),
        ],
    )
    def test_a_non_finite_float_does_not_produce_invalid_json(self, value, why):
        """`json.dumps` writes `Infinity`/`NaN`, which Python reads back and a strict parser refuses
        -- and the point of the embedded copy is that something other than gwmock parses it."""
        _ = why

        document = embeddable_metadata({"simulator_metadata": {"max_samples": value}})

        text = json.dumps(document)
        assert json.loads(text, parse_constant=_refuse_non_standard) == {"simulator_metadata": {"max_samples": None}}

    def test_a_non_finite_value_inside_an_array_is_normalised_too(self):
        """The array branch has to recurse, or the same token appears one level down."""
        document = embeddable_metadata({"signal": {"metadata": {"grid": np.array([1.0, np.inf])}}})

        text = json.dumps(document)
        assert json.loads(text, parse_constant=_refuse_non_standard)["signal"]["metadata"]["grid"] == [1.0, None]

    def test_the_self_referential_hashes_are_dropped(self):
        """A file cannot carry its own digest: the digest is taken after this copy is written."""
        document = embeddable_metadata(self._record())

        assert "file_hashes" not in document
        assert "content_hashes" not in document
        assert set(document["outputs"][0]) == {"kind", "path"}

    def test_the_rest_of_the_record_is_kept(self):
        """Self-describing is the point; only what must go is removed."""
        document = embeddable_metadata(self._record())

        assert document["schema_version"] == "1.5.0"
        assert document["config"]["globals"]["working-directory"] == "."

    def test_it_is_json_safe(self):
        """A record holds numpy scalars, arrays and paths, none of which `json.dumps` accepts."""
        record = {"host": {"cpu": np.int64(8)}, "grid": np.arange(3), "path": Path("out") / "a.hdf5"}

        assert json.loads(json.dumps(embeddable_metadata(record))) == {
            "host": {"cpu": 8},
            "grid": [0, 1, 2],
            "path": str(Path("out") / "a.hdf5"),
        }

    def test_the_callers_record_is_left_alone(self):
        """It is the sidecar's, and the sidecar keeps everything."""
        record = self._record()

        embeddable_metadata(record)

        assert record["signal"]["injections"]
        assert record["file_hashes"] == {"a.hdf5": "x"}
        assert record["outputs"][0]["sha256"] == "x"


class TestEmbedMetadataRecord:
    """Writing the copy into the artifact, and declining to for the formats with nowhere to put it."""

    @staticmethod
    def _hdf5(path):
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset("H1:STRAIN", data=np.zeros(4))
        return path

    def test_it_writes_the_record_at_the_root(self, tmp_path):
        path = self._hdf5(tmp_path / "strain.hdf5")

        assert embed_metadata_record(path, {"schema_version": "1.5.0"}) is True
        assert read_run_metadata(path) == {"schema_version": "1.5.0"}

    def test_a_second_call_replaces_rather_than_accumulates(self, tmp_path):
        """A record is rewritten when a batch is regenerated over its own output."""
        path = self._hdf5(tmp_path / "strain.hdf5")

        embed_metadata_record(path, {"batch_index": 0})
        embed_metadata_record(path, {"batch_index": 1})

        assert read_run_metadata(path) == {"batch_index": 1}

    @pytest.mark.parametrize("suffix", [".npy", ".gwf", ".txt"])
    def test_a_format_with_no_metadata_space_is_left_untouched(self, tmp_path, suffix):
        path = tmp_path / f"strain{suffix}"
        path.write_bytes(b"not-hdf5")

        assert embed_metadata_record(path, {"schema_version": "1.5.0"}) is False
        assert path.read_bytes() == b"not-hdf5"

    def test_a_missing_artifact_is_refused_rather_than_created(self, tmp_path):
        """Embedding describes a file that was already written; creating one here would publish a
        file holding metadata and no data."""
        path = tmp_path / "absent.hdf5"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            embed_metadata_record(path, {"schema_version": "1.5.0"})

        assert not path.exists()

    def test_the_content_hash_is_unchanged_by_embedding(self, tmp_path):
        """The digest of the *scientific content* must not move, or every stored reference value and
        every reproducibility check in the suite would have to be regenerated for a change that
        touched no sample. The record goes at the root, which the content hash does not read; the
        raw-byte hash does move, which is why it is taken afterwards.
        """
        path = self._hdf5(tmp_path / "strain.hdf5")
        before_content = compute_content_hash(path)
        before_bytes = compute_file_hash(path)

        embed_metadata_record(path, {"schema_version": "1.5.0"})

        assert compute_content_hash(path) == before_content
        assert compute_file_hash(path) != before_bytes

    def test_a_record_far_larger_than_an_object_header_round_trips(self, tmp_path):
        """A real record carries the environment freeze -- every installed distribution and its
        version -- and HDF5 stores an attribute that outgrows the object header densely rather than
        refusing it. Measured here rather than assumed, since the classic limit is 64 KiB and a
        freeze can pass it.
        """
        path = self._hdf5(tmp_path / "strain.hdf5")
        record = {"environment": {"packages": {f"package-{index}": "1.2.3" for index in range(20000)}}}

        embed_metadata_record(path, record)

        assert read_run_metadata(path) == record
