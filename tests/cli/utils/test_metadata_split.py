"""Unit tests for dividing a metadata record into its shareable and private parts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from gwmock.cli.utils.environment import _cpu_model_from_cpuinfo
from gwmock.cli.utils.metadata import (
    PRIVATE_METADATA_KIND,
    PRIVATE_REFERENCE_KEY,
    combine_metadata_records,
    load_private_metadata,
    private_metadata_path,
    private_reference,
    save_metadata_record,
    split_metadata_record,
    with_private_details,
    write_private_metadata,
)

BASE = "/data/alice/run"


def _record() -> dict:
    return {
        "author": "alice",
        "email": "alice@example.invalid",
        "host": {"platform": "Linux-6.8.0-1-generic-x86_64-with-glibc2.39", "python": "3.13.1", "cpu": "x86_64"},
        "config": {
            "globals": {"working-directory": BASE, "output-directory": f"{BASE}/output", "metadata-directory": "m"},
            "orchestration": {"population": {"arguments": {"path": "/shared/catalogues/pop.h5"}}},
        },
        "outputs": [{"kind": "signal", "path": "output/signal/a.hdf5", "t0": 0}],
        "file_hashes": {"a.hdf5": "sha256:00"},
        "note": f"loaded from {BASE}/inputs/x.txt and file:///home/alice/y.txt",
        "uri": "file:///shared/z.txt",
        "windows": "C:\\Users\\alice\\psd.txt",
        "by_path": {f"{BASE}/a": 1, "/elsewhere/a": 2},
        "url": "https://example.invalid/data/pop.h5",
        "ratio": "1 / Hz",
        "channel": "H1:STRAIN",
    }


def _all_strings(document):
    if isinstance(document, dict):
        for key, value in document.items():
            yield str(key)
            yield from _all_strings(value)
    elif isinstance(document, list):
        for item in document:
            yield from _all_strings(item)
    elif isinstance(document, str):
        yield document


class TestPrivateFileName:
    @pytest.mark.parametrize(
        ("record", "private"),
        [
            ("orchestration-0.metadata.json", "orchestration-0.private.json"),
            ("merged.metadata.yaml", "merged.private.yaml"),
            ("my.metadata.run-3.metadata.json", "my.metadata.run-3.private.json"),
            ("odd.json", "odd.private.json"),
        ],
    )
    def test_it_sits_beside_the_record(self, record: str, private: str) -> None:
        assert private_metadata_path(Path("/x") / record) == Path("/x") / private


class TestTheSplit:
    def test_the_shareable_part_names_no_one_and_no_absolute_path(self) -> None:
        shareable = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").shareable

        assert "author" not in shareable
        assert "email" not in shareable
        for text in _all_strings(shareable):
            assert not os.path.isabs(text), text
            assert "alice" not in text, text
            assert not text.startswith("file://"), text
            assert "6.8.0" not in text, text

    def test_paths_inside_the_working_directory_keep_their_layout(self) -> None:
        shareable = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").shareable

        assert shareable["config"]["globals"]["working-directory"] == "."
        assert shareable["config"]["globals"]["output-directory"] == "output"
        assert shareable["config"]["orchestration"]["population"]["arguments"]["path"] == "pop.h5"
        assert shareable["note"] == "loaded from inputs/x.txt and y.txt"
        assert shareable["uri"] == "z.txt"
        assert shareable["windows"] == "psd.txt"

    def test_what_is_not_a_path_is_left_alone(self) -> None:
        shareable = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").shareable

        for key in ("url", "ratio", "channel"):
            assert shareable[key] == _record()[key]
        assert shareable["outputs"] == _record()["outputs"]
        assert shareable["file_hashes"] == _record()["file_hashes"]

    def test_path_keys_stay_distinct_when_their_file_names_collide(self) -> None:
        shareable = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").shareable

        assert shareable["by_path"] == {"a": 1, "../../../elsewhere/a": 2}

    def test_the_private_part_holds_what_was_taken(self) -> None:
        private = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private

        assert private["kind"] == PRIVATE_METADATA_KIND
        assert private["describes"] == "r.json"
        assert private["author"] == "alice"
        assert private["email"] == "alice@example.invalid"
        assert private["platform"] == _record()["host"]["platform"]
        assert private["working_directory"] == BASE
        assert isinstance(private["hostname"], str)
        assert private["values"]["/config/globals/working-directory"] == BASE

    def test_the_two_parts_rebuild_the_record_exactly(self) -> None:
        parts = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json")

        assert combine_metadata_records(parts.shareable, parts.private) == _record()

    def test_the_rebuild_survives_a_round_trip_through_json(self) -> None:
        parts = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json")
        shareable = json.loads(json.dumps(parts.shareable))
        private = json.loads(json.dumps(parts.private))

        assert combine_metadata_records(shareable, private) == _record()

    def test_a_nested_source_record_is_split_against_its_own_working_directory(self) -> None:
        merged = {
            "type": "merged",
            "author": "bob",
            "source_files": {"/home/bob/frames/a.gwf": _record(), "frames/b.gwf": _record()},
        }

        parts = split_metadata_record(merged, base_directory="/home/bob", record_file_name="m.yaml")

        assert sorted(parts.shareable["source_files"]) == ["frames/a.gwf", "frames/b.gwf"]
        source = parts.shareable["source_files"]["frames/a.gwf"]
        assert "author" not in source
        assert source["config"]["globals"]["output-directory"] == "output"
        assert "platform" not in source["host"], "a nested host is not this machine's"
        assert combine_metadata_records(parts.shareable, parts.private) == merged

    def test_the_private_digest_cannot_confirm_a_guess(self, tmp_path: Path) -> None:
        """Two splits of one record give private files with different digests (the nonce)."""
        first = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private
        second = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private

        assert write_private_metadata(first, tmp_path / "1.json") != write_private_metadata(second, tmp_path / "2.json")

    def test_combining_with_something_else_is_refused(self) -> None:
        with pytest.raises(ValueError, match="private metadata"):
            combine_metadata_records({}, {"author": "x"})


class TestReadingThePrivateFile:
    @staticmethod
    def _written(tmp_path: Path) -> tuple[Path, dict]:
        record_file = tmp_path / "r.metadata.json"
        parts = split_metadata_record(_record(), base_directory=BASE, record_file_name=record_file.name)
        private_file = private_metadata_path(record_file)
        shareable = {**parts.shareable}
        shareable[PRIVATE_REFERENCE_KEY] = private_reference(
            private_file, write_private_metadata(parts.private, private_file)
        )
        return record_file, shareable

    def test_it_is_written_with_the_digest_the_record_names(self, tmp_path: Path) -> None:
        record_file, shareable = self._written(tmp_path)

        private_file = tmp_path / shareable[PRIVATE_REFERENCE_KEY]["file"]
        assert hashlib.sha256(private_file.read_bytes()).hexdigest() == shareable[PRIVATE_REFERENCE_KEY]["sha256"]
        assert with_private_details(record_file, shareable) == _record()

    def test_without_it_the_shareable_record_is_used(self, tmp_path: Path) -> None:
        record_file, shareable = self._written(tmp_path)
        private_metadata_path(record_file).unlink()

        assert with_private_details(record_file, shareable) is shareable

    def test_a_private_file_that_does_not_match_is_ignored(self, tmp_path: Path, caplog) -> None:
        record_file, shareable = self._written(tmp_path)
        private_file = private_metadata_path(record_file)
        private_file.write_text(private_file.read_text().replace("alice@", "mallory@"))

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            assert load_private_metadata(record_file, shareable) is None
        assert "SHA-256" in caplog.text

    def test_a_record_cannot_point_outside_its_directory(self, tmp_path: Path) -> None:
        record_file, shareable = self._written(tmp_path)
        # The same file, reached by a path rather than a name: refused all the same.
        shareable[PRIVATE_REFERENCE_KEY]["file"] = f"../{tmp_path.name}/{shareable[PRIVATE_REFERENCE_KEY]['file']}"

        assert load_private_metadata(record_file, shareable) is None


class TestTheHost:
    def test_an_unknown_processor_model_is_written_as_null(self, tmp_path: Path) -> None:
        record = {
            "subpackage_versions": {},
            "config": {},
            "config_sha256": "0",
            "host": {"platform": "Linux-x86_64", "python": "3.13", "cpu": "x86_64", "cpu_model": None},
        }

        save_metadata_record(record, tmp_path / "r.metadata.json")

        host = json.loads((tmp_path / "r.metadata.json").read_text())["host"]
        assert "cpu_model" in host
        assert host["cpu_model"] is None
        assert "git_sha" not in host, "only the processor model is spelled out as null"

    @pytest.mark.parametrize(
        ("text", "model"),
        [
            (
                "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 7763 64-Core Processor\n",
                "AMD EPYC 7763 64-Core Processor",
            ),
            ("processor\t: 0\nBogoMIPS\t: 50.00\n\nHardware\t: BCM2835\n", "BCM2835"),
            ("Processor\t: ARMv7 Processor rev 4 (v7l)\nprocessor\t: 0\n", "ARMv7 Processor rev 4 (v7l)"),
            ("processor\t: 0\nprocessor\t: 1\n", None),
        ],
    )
    def test_the_model_is_read_from_cpuinfo(self, tmp_path: Path, text: str, model: str | None) -> None:
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(text)

        assert _cpu_model_from_cpuinfo(cpuinfo) == model

    def test_an_unreadable_cpuinfo_is_unknown(self, tmp_path: Path) -> None:
        assert _cpu_model_from_cpuinfo(tmp_path / "absent") is None
