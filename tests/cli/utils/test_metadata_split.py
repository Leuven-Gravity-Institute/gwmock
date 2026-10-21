"""Unit tests for dividing a metadata record into its shareable and private parts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
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

#: The accounts the tests pretend exist on the writing machine. ``~name`` is reduced only for an
#: account, so the lookup is fixed rather than left to whoever runs the suite.
ACCOUNTS = {"alice"}


@pytest.fixture(autouse=True)
def _accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gwmock.cli.utils.metadata._is_account", lambda name: name in ACCOUNTS, raising=False)


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

        assert shareable["by_path"] == {"a": 1, "a~2": 2}

    def test_colliding_keys_outside_the_working_directory_name_no_directory(self) -> None:
        """A second key with a taken file name is told apart by a suffix, never by a path out of the
        working directory -- which would spell the home directory it came from."""
        record = {
            "source_files": {
                f"{BASE}/a.h5": {},
                "/home/alice/catalogues/a.h5": {},
                "/opt/other/a.h5": {},
            }
        }

        parts = split_metadata_record(record, base_directory=BASE, record_file_name="r.json")

        assert list(parts.shareable["source_files"]) == ["a.h5", "a~2.h5", "a~3.h5"]
        assert combine_metadata_records(parts.shareable, parts.private) == record

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

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_the_private_file_is_readable_by_its_owner_only(self, tmp_path: Path) -> None:
        """It names a user, a host and directories, so a shared filesystem must not expose it."""
        private = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private
        previous = os.umask(0o022)
        try:
            write_private_metadata(private, tmp_path / "r.private.json")
            write_private_metadata(private, tmp_path / "r.private.json")  # replacing one keeps the mode
        finally:
            os.umask(previous)

        assert stat.S_IMODE((tmp_path / "r.private.json").stat().st_mode) == 0o600
        assert sorted(path.name for path in tmp_path.iterdir()) == ["r.private.json"], "no temporary left"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_a_stale_temporary_file_does_not_block_the_write(self, tmp_path: Path) -> None:
        private = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private
        (tmp_path / ".r.private.json.tmp").write_text("left by an interrupted write")

        write_private_metadata(private, tmp_path / "r.private.json")

        assert json.loads((tmp_path / "r.private.json").read_text())["kind"] == PRIVATE_METADATA_KIND
        assert stat.S_IMODE((tmp_path / "r.private.json").stat().st_mode) == 0o600

    def test_the_private_digest_cannot_confirm_a_guess(self, tmp_path: Path) -> None:
        """Two splits of one record give private files with different digests (the nonce)."""
        first = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private
        second = split_metadata_record(_record(), base_directory=BASE, record_file_name="r.json").private

        assert write_private_metadata(first, tmp_path / "1.json") != write_private_metadata(second, tmp_path / "2.json")

    def test_combining_with_something_else_is_refused(self) -> None:
        with pytest.raises(ValueError, match="private metadata"):
            combine_metadata_records({}, {"author": "x"})


#: Free text in which a path, a home directory, a host or a server must not survive, and what each
#: becomes. ``None`` means "unchanged".
_FREE_TEXT = [
    ("source:/home/alice/input.h5", "source:input.h5"),
    ("inputs=/home/alice/a.h5,/home/alice/b.h5", "inputs=a.h5,b.h5"),
    ("paths: /home/alice/a.h5; /opt/b.h5.", "paths: a.h5; b.h5."),
    ("(read from /home/alice/x.txt)", "(read from x.txt)"),
    ('{"file": "/home/alice/x.txt"}', '{"file": "x.txt"}'),
    ("PATH=/usr/alice/bin:/home/alice/bin", "PATH=bin:bin"),
    (f"under {BASE}/output/a.hdf5, done", "under output/a.hdf5, done"),
    ("copied from file://canary-host/home/alice/y.txt today", "copied from y.txt today"),
    ("copied from file:///home/alice/y.txt today", "copied from y.txt today"),
    ("share \\\\fileserver\\share\\alice\\psd.txt here", "share psd.txt here"),
    ("drive:C:\\Users\\alice\\psd.txt,", "drive:psd.txt,"),
    ("(C:/Users/alice/psd.txt)", "(psd.txt)"),
    ("~alice/catalogues/pop.h5", "pop.h5"),
    ("from ~alice/catalogues/pop.h5 and ~/x.h5", "from pop.h5 and x.h5"),
    ("~/x.h5", "x.h5"),
    ("\\\\fileserver\\share\\alice\\psd.txt", "psd.txt"),
    ("file://canary-host/home/alice/y.txt", "y.txt"),
    ("see //fileserver/share/alice/psd.txt now", "see psd.txt now"),
    ("see \\\\fileserver/share/alice/psd.txt now", "see psd.txt now"),
    ("//fileserver/share/alice/psd.txt", "psd.txt"),
    ("from ~alice\\catalogues\\pop.h5, then", "from pop.h5, then"),
    ("~alice\\catalogues\\pop.h5", "pop.h5"),
    ("generated by ~alice", "generated by ~"),
    ("see ~alice.", "see ~."),
    ("(~alice, ~bob)", "(~, ~bob)"),
    ("~alice/x", "x"),
    ("~alice\\x", "x"),
    ("~alice", "~"),
    ("C:\\Users\\alice\\psd.txt", "psd.txt"),
    ("C:/Users/alice/psd.txt", "psd.txt"),
    ("at C:/Users/alice/psd.txt", "at psd.txt"),
    ("{/home/alice/x.h5}", "{x.h5}"),
    ("<C:\\Users\\alice\\x.h5>", "<x.h5>"),
    # Not paths: left exactly as they are.
    ("x://host/share/f.h5", None),
    ("see x://host/share/f.h5, then", None),
    # `~name` that is not an account on the writing machine.
    ("~bob", None),
    ("~bob/x", None),
    ("~bob\\x", None),
    # Ordinary gravitational-wave notation.
    ("h_+/h_x", None),
    ("F_+/F_x", None),
    ("h+/hx", None),
    ("h_{+}/h_{x}", None),
    ("(h_+)/(h_x)", None),
    ("~pi/2", None),
    ("~M_\\odot", None),
    ("mass ~M_\\odot", None),
    ("~Msun", None),
    ("chirp mass ~Msun", None),
    ("distance ~Gpc", None),
    ("frequency ~mHz", None),
    ("~kpc", None),
    ("~nHz", None),
    ("~O(v/c)", None),
    ("phase ~O(v^5)", None),
    ("~f^{-7/3}", None),
    ("x^/y", None),
    ("a-/b", None),
    ("a\\/b", None),
    ("[1]/[2]", None),
    ("~5 Hz", None),
    ("~0.3", None),
    ("x ~ y", None),
    ("a // b", None),
    ("https://host.example/share/data/x.h5", None),
    ("see https://host.example//share/x.h5, then", None),
    ("https://example.invalid/data/pop.h5", None),
    ("see https://example.invalid/a/b, then go", None),
    ("1 / Hz", None),
    ("strain/Hz", None),
    ("relative/inputs/a.h5", None),
    ("H1:STRAIN", None),
    ("a~2.h5", None),
]


class TestFreeText:
    @pytest.mark.parametrize(("text", "expected"), _FREE_TEXT)
    def test_paths_in_text_are_reduced(self, text: str, expected: str | None) -> None:
        shareable = split_metadata_record({"note": text}, base_directory=BASE, record_file_name="r").shareable

        assert shareable["note"] == (text if expected is None else expected)

    def test_a_corpus_leaves_nothing_identifying_and_rebuilds_exactly(self) -> None:
        """Every tricky string, as a value, inside lists and as a mapping key, at several depths."""
        texts = [text for text, _ in _FREE_TEXT]
        record = {
            "notes": texts,
            "nested": {"deeper": [{"text": text, "pair": [text, text]} for text in texts]},
            "by_text": {text: index for index, text in enumerate(texts)},
            "source_files": {text: {"author": "alice", "note": text} for text in texts},
        }

        parts = split_metadata_record(record, base_directory=BASE, record_file_name="r.json")

        for text in _all_strings(parts.shareable):
            assert not os.path.isabs(text), text
            assert "alice" not in text, text
            assert "canary-host" not in text, text
            assert "fileserver" not in text, text
            assert "file://" not in text, text
            assert not re.search(r"~[\\/]", text), text  # a home path; `~5 Hz` is not one
            assert "\\\\" not in text, text
            assert not re.search(r"(?:^|[^A-Za-z0-9:])//[^/\s]", text), text
            assert not re.search(r"~(?:alice|canary)", text), text
            assert not re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z]:(?:\\|/(?!/))", text), text
        assert combine_metadata_records(parts.shareable, parts.private) == record
        assert (
            combine_metadata_records(json.loads(json.dumps(parts.shareable)), json.loads(json.dumps(parts.private)))
            == record
        )


class TestWhoIsAnAccount:
    """The real lookup, without the fixture: the current user is one, a made-up name is not."""

    def test_the_current_user_is_an_account(self, monkeypatch) -> None:
        import getpass

        from gwmock.cli.utils import metadata

        monkeypatch.undo()
        assert metadata._is_account(getpass.getuser())

    def test_a_name_no_one_has_is_not(self, monkeypatch) -> None:
        from gwmock.cli.utils import metadata

        monkeypatch.undo()
        assert not metadata._is_account("no-such-account-7f3e9c")
        assert not metadata._is_account("Msun")

    def test_an_account_is_reduced_and_a_word_is_not(self, monkeypatch) -> None:
        monkeypatch.setattr("gwmock.cli.utils.metadata._is_account", lambda name: name == "Msun", raising=False)

        shareable = split_metadata_record(
            {"a": "~Msun/x.h5", "b": "~alice/x.h5"}, base_directory=BASE, record_file_name="r"
        ).shareable

        assert shareable == {"a": "x.h5", "b": "~alice/x.h5"}


class TestConcurrentWriters:
    def test_the_file_is_one_writers_whole_payload(self, tmp_path: Path) -> None:
        """Every writer gets its own temporary file, so what lands is one complete write."""
        import threading

        target = tmp_path / "r.private.json"
        for trial in range(12):
            documents = [
                {"kind": PRIVATE_METADATA_KIND, "writer": index, "pad": "x" * (5000 * index)} for index in range(8)
            ]
            digests: list[str] = []
            errors: list[BaseException] = []
            barrier = threading.Barrier(len(documents))

            def write(document, digests=digests, errors=errors, barrier=barrier) -> None:
                barrier.wait()
                try:
                    digests.append(write_private_metadata(document, target))
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=write, args=(document,)) for document in documents]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert errors == [], f"trial {trial}"
            assert hashlib.sha256(target.read_bytes()).hexdigest() in digests, f"trial {trial}"
            assert sorted(path.name for path in tmp_path.iterdir()) == [target.name], f"trial {trial}"


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

    def test_a_companion_that_links_out_of_the_directory_is_refused(self, tmp_path: Path) -> None:
        """A bare name that is a symbolic link to a file elsewhere -- even one with the right digest."""
        record_directory = tmp_path / "records"
        record_directory.mkdir()
        record_file, shareable = self._written(record_directory)
        companion = private_metadata_path(record_file)
        outside = tmp_path / "elsewhere.json"
        outside.write_bytes(companion.read_bytes())
        companion.unlink()
        companion.symlink_to(outside)

        assert hashlib.sha256(companion.read_bytes()).hexdigest() == shareable[PRIVATE_REFERENCE_KEY]["sha256"]
        assert load_private_metadata(record_file, shareable) is None

    def test_a_companion_linked_within_the_directory_is_read(self, tmp_path: Path) -> None:
        record_file, shareable = self._written(tmp_path)
        companion = private_metadata_path(record_file)
        target = tmp_path / "actual.json"
        companion.rename(target)
        companion.symlink_to(target.name)

        assert load_private_metadata(record_file, shareable) is not None


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
