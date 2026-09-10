"""``gwmock merge``: combining per-detector frame files into one, and vouching for the inputs.

The command had no test at all. What it promises is worth pinning: it refuses to merge files it
cannot verify against their metadata, it refuses to add series that do not describe the same
stretch of time, the sum it writes is the sum of its inputs, and the merged file appears whole or
not at all.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from gwpy.timeseries import TimeSeries

from gwmock.cli.merge import merge_command
from gwmock.cli.utils.hash import compute_file_hash
from gwmock.strain_schema import (
    STRAIN_SCHEMA_VERSION,
    read_run_metadata,
    read_strain_schema,
    require_strain_schema,
)

pytestmark = pytest.mark.unit

CHANNEL = "STRAIN"
START = 1000000000
RATE = 16


def _frame(path: Path, values, *, start: float = START, rate: float = RATE) -> Path:
    """Write one frame file holding *values* on ``CHANNEL``."""
    series = TimeSeries(np.asarray(values, dtype=float), sample_rate=rate, t0=start, channel=CHANNEL, name=CHANNEL)
    series.write(path)
    return path


def _metadata_for(path: Path, metadata_path: Path, *, digest: str | None = None) -> Path:
    """Write the metadata file the command validates *path* against."""
    hashes = {} if digest == "" else {path.name: digest or compute_file_hash(path)}
    metadata_path.write_text(yaml.safe_dump({"file_hashes": hashes}))
    return metadata_path


@pytest.fixture
def two_frames(tmp_path: Path):
    first = _frame(tmp_path / "a.gwf", np.ones(RATE))
    second = _frame(tmp_path / "b.gwf", 2 * np.ones(RATE))
    return first, second


class TestTheInputsAreVouchedFor:
    def test_merging_without_metadata_is_refused(self, tmp_path: Path, two_frames) -> None:
        """The default is to refuse: a merged frame with no provenance is what ``--force`` is for."""
        with pytest.raises(ValueError, match="Metadata files must be provided unless --force is used"):
            merge_command(list(two_frames), output=str(tmp_path / "merged.gwf"))

    def test_a_metadata_file_per_frame_is_required(self, tmp_path: Path, two_frames) -> None:
        first, _ = two_frames
        with pytest.raises(ValueError, match="number of metadata files must match"):
            merge_command(
                list(two_frames),
                output=str(tmp_path / "merged.gwf"),
                metadata=[str(_metadata_for(first, tmp_path / "a.yaml"))],
            )

    def test_a_frame_absent_from_its_metadata_is_refused(self, tmp_path: Path, two_frames) -> None:
        first, second = two_frames
        metadata = [
            str(_metadata_for(first, tmp_path / "a.yaml", digest="")),
            str(_metadata_for(second, tmp_path / "b.yaml")),
        ]
        with pytest.raises(ValueError, match=r"No hash found in metadata for file a\.gwf"):
            merge_command(list(two_frames), output=str(tmp_path / "merged.gwf"), metadata=metadata)

    def test_a_frame_that_does_not_match_its_recorded_hash_is_refused(self, tmp_path: Path, two_frames) -> None:
        """The point of the check: a file edited or regenerated since the run that recorded it must
        not be merged as if it were the original."""
        first, second = two_frames
        metadata = [
            str(_metadata_for(first, tmp_path / "a.yaml", digest="sha256:" + "0" * 64)),
            str(_metadata_for(second, tmp_path / "b.yaml")),
        ]
        with pytest.raises(ValueError, match=r"Hash mismatch for file a\.gwf"):
            merge_command(list(two_frames), output=str(tmp_path / "merged.gwf"), metadata=metadata)

    def test_nothing_is_written_when_a_frame_fails_verification(self, tmp_path: Path, two_frames) -> None:
        first, second = two_frames
        output = tmp_path / "merged.gwf"
        metadata = [
            str(_metadata_for(first, tmp_path / "a.yaml", digest="sha256:" + "0" * 64)),
            str(_metadata_for(second, tmp_path / "b.yaml")),
        ]
        with pytest.raises(ValueError, match="Hash mismatch"):
            merge_command(list(two_frames), output=str(output), metadata=metadata)
        assert not output.exists()

    def test_force_merges_without_any_metadata(self, tmp_path: Path, two_frames) -> None:
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), force=True)
        assert output.exists()

    def test_force_skips_the_hash_check_it_would_otherwise_fail(self, tmp_path: Path, two_frames) -> None:
        """``--force`` is documented as bypassing the metadata requirement, so a wrong hash must not
        stop it either -- otherwise the flag cannot rescue a run whose metadata was lost."""
        first, second = two_frames
        metadata = [
            str(_metadata_for(first, tmp_path / "a.yaml", digest="sha256:" + "0" * 64)),
            str(_metadata_for(second, tmp_path / "b.yaml")),
        ]
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), metadata=metadata, force=True)
        assert output.exists()


class TestTheSeriesMustLineUp:
    def test_a_different_start_time_is_refused(self, tmp_path: Path) -> None:
        """Injecting a series that starts elsewhere would silently pad or shift the data."""
        first = _frame(tmp_path / "a.gwf", np.ones(RATE))
        second = _frame(tmp_path / "b.gwf", np.ones(RATE), start=START + 8)
        with pytest.raises(ValueError, match="Start time mismatch"):
            merge_command([first, second], output=str(tmp_path / "merged.gwf"), force=True)

    def test_a_different_duration_is_refused(self, tmp_path: Path) -> None:
        first = _frame(tmp_path / "a.gwf", np.ones(RATE))
        second = _frame(tmp_path / "b.gwf", np.ones(RATE // 2))
        with pytest.raises(ValueError, match="Duration mismatch"):
            merge_command([first, second], output=str(tmp_path / "merged.gwf"), force=True)

    def test_a_different_sampling_frequency_is_refused(self, tmp_path: Path) -> None:
        first = _frame(tmp_path / "a.gwf", np.ones(RATE))
        second = _frame(tmp_path / "b.gwf", np.ones(2 * RATE), rate=2 * RATE)
        with pytest.raises(ValueError, match="Sampling frequency mismatch"):
            merge_command([first, second], output=str(tmp_path / "merged.gwf"), force=True)

    def test_the_first_file_sets_what_the_others_are_checked_against(self, tmp_path: Path) -> None:
        """Three files, and it is the third that disagrees: the loop has to keep comparing against
        the first rather than against its predecessor."""
        first = _frame(tmp_path / "a.gwf", np.ones(RATE))
        second = _frame(tmp_path / "b.gwf", np.ones(RATE))
        third = _frame(tmp_path / "c.gwf", np.ones(RATE), start=START + 8)
        with pytest.raises(ValueError, match="Start time mismatch"):
            merge_command([first, second, third], output=str(tmp_path / "merged.gwf"), force=True)


class TestWhatComesOut:
    def test_the_merged_series_is_the_sum_of_the_inputs(self, tmp_path: Path, two_frames) -> None:
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), force=True)
        merged = TimeSeries.read(output, CHANNEL)
        assert np.allclose(merged.value, 3.0, atol=0.0)

    def test_three_files_are_all_added(self, tmp_path: Path) -> None:
        files = [
            _frame(tmp_path / "a.gwf", np.ones(RATE)),
            _frame(tmp_path / "b.gwf", 2 * np.ones(RATE)),
            _frame(tmp_path / "c.gwf", 4 * np.ones(RATE)),
        ]
        output = tmp_path / "merged.gwf"
        merge_command(files, output=str(output), force=True)
        assert np.allclose(TimeSeries.read(output, CHANNEL).value, 7.0, atol=0.0)

    def test_a_single_file_is_copied_through_unchanged(self, tmp_path: Path) -> None:
        first = _frame(tmp_path / "a.gwf", np.arange(RATE))
        output = tmp_path / "merged.gwf"
        merge_command([first], output=str(output), force=True)
        assert np.allclose(TimeSeries.read(output, CHANNEL).value, np.arange(RATE), atol=0.0)

    def test_the_time_axis_is_preserved(self, tmp_path: Path, two_frames) -> None:
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), force=True)
        merged = TimeSeries.read(output, CHANNEL)
        assert merged.epoch.value == START
        assert merged.sample_rate.value == RATE

    def test_the_output_channel_can_be_renamed(self, tmp_path: Path, two_frames) -> None:
        """A merged network stream is not the single-detector channel it came from."""
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), output_channel="NETWORK_STRAIN", force=True)
        assert TimeSeries.read(output, "NETWORK_STRAIN") is not None

    def test_the_channel_is_left_alone_by_default(self, tmp_path: Path, two_frames) -> None:
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), force=True)
        assert TimeSeries.read(output, CHANNEL) is not None

    def test_no_temporary_file_is_left_behind(self, tmp_path: Path, two_frames) -> None:
        """Written to ``merged.tmp.gwf`` and renamed, so a crash mid-write cannot leave a file that
        looks like a finished merge."""
        output = tmp_path / "merged.gwf"
        merge_command(list(two_frames), output=str(output), force=True)
        assert not (tmp_path / "merged.tmp.gwf").exists()
        assert sorted(p.name for p in tmp_path.glob("merged*")) == ["merged.gwf"]

    def test_the_named_channel_is_the_one_that_is_read(self, tmp_path: Path) -> None:
        """Two channels in one file: the merge must take the one it was asked for."""
        path = tmp_path / "a.gwf"
        wanted = TimeSeries(np.ones(RATE), sample_rate=RATE, t0=START, channel="WANTED", name="WANTED")
        other = TimeSeries(9 * np.ones(RATE), sample_rate=RATE, t0=START, channel="OTHER", name="OTHER")
        from gwpy.timeseries import TimeSeriesDict

        TimeSeriesDict({"WANTED": wanted, "OTHER": other}).write(path)

        output = tmp_path / "merged.gwf"
        merge_command([path], channel="WANTED", output=str(output), force=True)
        assert np.allclose(TimeSeries.read(output, "WANTED").value, 1.0, atol=0.0)


class TestTheStrainSchemaIsDeclared:
    """A merged file is the artifact gwmock most often hands to another pipeline, so it says what it is.

    The declaration rides in the file itself rather than only in the metadata sidecar, because a merge
    can be forced past metadata entirely and the merged frame then travels alone.
    """

    @staticmethod
    def _hdf5(path: Path, values) -> Path:
        series = TimeSeries(np.asarray(values, dtype=float), sample_rate=RATE, t0=START, channel=CHANNEL, name=CHANNEL)
        series.write(path)
        return path

    def test_an_hdf5_merge_declares_the_schema(self, tmp_path: Path) -> None:
        inputs = [self._hdf5(tmp_path / "a.hdf5", np.ones(RATE)), self._hdf5(tmp_path / "b.hdf5", 2 * np.ones(RATE))]
        output = tmp_path / "merged.hdf5"

        merge_command(inputs, output=str(output), force=True)

        assert require_strain_schema(output).version == STRAIN_SCHEMA_VERSION

    def test_the_merged_file_is_still_readable(self, tmp_path: Path) -> None:
        """Declaring the schema must not cost the consumer the standard reader."""
        inputs = [self._hdf5(tmp_path / "a.hdf5", np.ones(RATE)), self._hdf5(tmp_path / "b.hdf5", 2 * np.ones(RATE))]
        output = tmp_path / "merged.hdf5"

        merge_command(inputs, output=str(output), force=True)

        assert np.allclose(TimeSeries.read(output, CHANNEL).value, 3.0, atol=0.0)

    def test_the_declaration_is_in_place_before_the_file_appears(self, tmp_path: Path) -> None:
        """Stamped on the temporary file, so the merged path never exists in an undeclared state -- the
        same reason the samples are written to a temporary file and renamed."""
        inputs = [self._hdf5(tmp_path / "a.hdf5", np.ones(RATE))]
        output = tmp_path / "merged.hdf5"

        merge_command(inputs, output=str(output), force=True)

        assert not (tmp_path / "merged.tmp.hdf5").exists()
        assert require_strain_schema(output) is not None

    def test_a_gwf_merge_is_written_without_one(self, tmp_path: Path, two_frames) -> None:
        """A frame has no attribute space for the declaration, so the merge must simply not attempt it."""
        output = tmp_path / "merged.gwf"

        merge_command(list(two_frames), output=str(output), force=True)

        assert output.exists()
        assert read_strain_schema(output) is None


class TestTheMergedMetadata:
    @staticmethod
    def _merge_with_metadata(tmp_path: Path, frames) -> Path:
        metadata = [str(_metadata_for(path, tmp_path / f"{path.stem}.yaml")) for path in frames]
        output = tmp_path / "merged.gwf"
        merge_command(list(frames), output=str(output), metadata=metadata)
        return output

    def test_a_metadata_file_is_written_beside_the_merged_frame(self, tmp_path: Path, two_frames) -> None:
        self._merge_with_metadata(tmp_path, two_frames)
        assert (tmp_path / "merged.metadata.yaml").exists()

    def test_it_records_every_source_file(self, tmp_path: Path, two_frames) -> None:
        self._merge_with_metadata(tmp_path, two_frames)
        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert record["type"] == "merged"
        assert sorted(Path(name).name for name in record["source_files"]) == ["a.gwf", "b.gwf"]

    def test_it_records_the_hash_of_what_it_wrote(self, tmp_path: Path, two_frames) -> None:
        """The merged file's own digest, so the next consumer can verify it the way this command
        verified its inputs."""
        output = self._merge_with_metadata(tmp_path, two_frames)
        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert record["output_files"] == [str(output)]
        assert record["file_hashes"][str(output)] == compute_file_hash(output)

    def test_the_author_defaults_to_the_user_running_it(self, tmp_path: Path, two_frames) -> None:
        self._merge_with_metadata(tmp_path, two_frames)
        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert record["author"] == getpass.getuser()

    def test_a_given_author_and_email_are_recorded(self, tmp_path: Path, two_frames) -> None:
        metadata = [str(_metadata_for(path, tmp_path / f"{path.stem}.yaml")) for path in two_frames]
        output = tmp_path / "merged.gwf"
        merge_command(
            list(two_frames),
            output=str(output),
            metadata=metadata,
            author="A Person",
            email="person@example.invalid",
        )
        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert record["author"] == "A Person"
        assert record["email"] == "person@example.invalid"

    def test_it_records_when_and_with_what_versions(self, tmp_path: Path, two_frames) -> None:
        self._merge_with_metadata(tmp_path, two_frames)
        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())
        assert record["timestamp"].endswith("+00:00"), "the timestamp has to carry its time zone"
        assert record["versions"], "the dependency versions are what make the merge reproducible"

    def test_no_metadata_file_is_written_for_a_forced_merge(self, tmp_path: Path, two_frames) -> None:
        merge_command(list(two_frames), output=str(tmp_path / "merged.gwf"), force=True)
        assert not (tmp_path / "merged.metadata.yaml").exists()

    def test_no_temporary_metadata_file_is_left_behind(self, tmp_path: Path, two_frames) -> None:
        self._merge_with_metadata(tmp_path, two_frames)
        assert not (tmp_path / "merged.metadata.tmp").exists()


class TestTheRecordInsideTheMergedFile:
    """A merge is the artifact gwmock hands on, so it carries the record -- minus the answer key.

    The sources' metadata files are copied into the merged record whole, and a run's sidecar always
    records its injection parameters whatever it embedded in its own outputs. So a merge that wrote
    the assembled record into the file would re-publish, in one file, exactly what every run in the
    merge had withheld from its own. It goes through the same withholding, with the same default.
    """

    CANARY_MASS = 41.2468

    @staticmethod
    def _hdf5(path: Path, values) -> Path:
        series = TimeSeries(np.asarray(values, dtype=float), sample_rate=RATE, t0=START, channel=CHANNEL, name=CHANNEL)
        series.write(path)
        return path

    @classmethod
    def _source_metadata(cls, source: Path, metadata_path: Path) -> Path:
        """Write a source record shaped like a run's: hashes to verify, and injections to protect."""
        metadata_path.write_text(
            yaml.safe_dump(
                {
                    "file_hashes": {source.name: compute_file_hash(source)},
                    "signal": {
                        "injections": [{"event_id": 0, "parameters": {"detector_frame_mass_1": cls.CANARY_MASS}}],
                    },
                }
            )
        )
        return metadata_path

    def _merge(self, tmp_path: Path, **kwargs) -> Path:
        sources = [self._hdf5(tmp_path / "a.hdf5", np.ones(RATE)), self._hdf5(tmp_path / "b.hdf5", 2 * np.ones(RATE))]
        metadata = [str(self._source_metadata(path, tmp_path / f"{path.stem}.yaml")) for path in sources]
        output = tmp_path / "merged.hdf5"
        merge_command(sources, output=str(output), metadata=metadata, **kwargs)
        return output

    def test_the_merged_file_carries_the_merged_record(self, tmp_path: Path) -> None:
        output = self._merge(tmp_path)

        embedded = read_run_metadata(output)

        assert embedded is not None
        assert embedded["type"] == "merged"
        assert sorted(Path(name).name for name in embedded["source_files"]) == ["a.hdf5", "b.hdf5"]

    def test_it_withholds_the_injection_parameters_by_default(self, tmp_path: Path) -> None:
        output = self._merge(tmp_path)

        embedded = read_run_metadata(output)

        assert embedded is not None, "nothing was embedded, so this assertion proves nothing"
        assert str(self.CANARY_MASS) not in json.dumps(embedded)

    def test_it_keeps_them_when_the_merge_asks_for_them(self, tmp_path: Path) -> None:
        output = self._merge(tmp_path, include_injection_parameters=True)

        embedded = read_run_metadata(output)

        assert embedded is not None
        source = embedded["source_files"][str(tmp_path / "a.hdf5")]
        assert source["signal"]["injections"][0]["parameters"]["detector_frame_mass_1"] == self.CANARY_MASS

    def test_the_sidecar_keeps_them_either_way(self, tmp_path: Path) -> None:
        """The producer's copy is untouched: only what goes inside the released file is trimmed."""
        self._merge(tmp_path)

        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())

        source = record["source_files"][str(tmp_path / "a.hdf5")]
        assert source["signal"]["injections"][0]["parameters"]["detector_frame_mass_1"] == self.CANARY_MASS

    def test_the_recorded_hash_is_of_the_file_that_carries_the_record(self, tmp_path: Path) -> None:
        """Embedding changes the container bytes, so it happens before the digest is taken."""
        output = self._merge(tmp_path)

        record = yaml.safe_load((tmp_path / "merged.metadata.yaml").read_text())

        assert record["file_hashes"][str(output)] == compute_file_hash(output)

    def test_a_forced_merge_embeds_nothing(self, tmp_path: Path) -> None:
        """With no source metadata there is no record to carry, and none is invented."""
        sources = [self._hdf5(tmp_path / "a.hdf5", np.ones(RATE))]
        output = tmp_path / "merged.hdf5"

        merge_command(sources, output=str(output), force=True)

        assert read_run_metadata(output) is None
