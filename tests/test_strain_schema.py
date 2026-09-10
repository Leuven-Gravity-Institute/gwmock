"""The declared, versioned strain schema: what gwmock stamps on a strain file, and what it refuses.

Before this contract existed a pipeline reading gwmock output matched the writer's implementation --
which dataset it happened to create, which attributes it happened to set -- and a change to that layout
reached the reader as a wrong number rather than as a refusal. These tests pin the two halves: the
declaration a writer puts on the artifact, and the check a consumer runs before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock.cli.utils.metadata import embed_metadata_record
from gwmock.strain_schema import (
    REQUIRED_DATASET_ATTRIBUTES,
    RUN_METADATA_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    STRAIN_SCHEMA,
    STRAIN_SCHEMA_VERSION,
    StrainSchema,
    carries_strain_schema,
    conflicting_grid_attributes,
    declare_strain_schema,
    missing_layout_attributes,
    parse_strain_schema_version,
    read_run_metadata,
    read_strain_schema,
    require_strain_schema,
    strain_schema_attributes,
)

pytestmark = pytest.mark.unit

RATE = 8.0
START = 100.0


def _attribute_snapshot(path: Path) -> dict[str, dict[str, object]]:
    """Return every dataset's attributes, for comparing a file against itself across a call."""
    with h5py.File(path, "r") as handle:
        return {
            name: {key: handle[name].attrs[key].item() for key in sorted(handle[name].attrs)} for name in sorted(handle)
        }


def _hdf5_strain(path: Path, *, channel: str = "H1:MOCK_NOISE") -> Path:
    """Write a strain file laid out the way gwmock lays one out, with nothing declared on it yet."""
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(channel, data=np.arange(16, dtype=float))
        dataset.attrs["x0"] = START
        dataset.attrs["dx"] = 1.0 / RATE
        dataset.attrs["xunit"] = "s"
        dataset.attrs["channel"] = channel
        dataset.attrs["name"] = channel
        dataset.attrs["unit"] = "strain"
    return path


class TestWhatTheDeclarationSays:
    """The constants are the contract, so their shape is part of it."""

    def test_the_version_is_a_semantic_version(self) -> None:
        assert parse_strain_schema_version(STRAIN_SCHEMA_VERSION) == (1, 1, 0)

    def test_the_minor_moved_for_the_embedded_record(self) -> None:
        """1.1.0 adds `run_metadata` at the root and changes nothing a 1.0.0 reader reads, so the major
        must not have moved with it: a bumped major would make every such reader refuse a file it can
        still read perfectly well."""
        assert parse_strain_schema_version(STRAIN_SCHEMA_VERSION)[0] == 1

    def test_the_attributes_name_the_schema_and_its_version(self) -> None:
        assert strain_schema_attributes() == {
            SCHEMA_ATTRIBUTE: STRAIN_SCHEMA,
            SCHEMA_VERSION_ATTRIBUTE: STRAIN_SCHEMA_VERSION,
        }

    def test_the_layout_the_version_requires(self) -> None:
        """The five attributes a consumer may read on any declared file, pinned so widening the contract
        is a deliberate edit rather than a side effect."""
        assert REQUIRED_DATASET_ATTRIBUTES == ("x0", "dx", "xunit", "channel", "name")

    @pytest.mark.parametrize("value", ["1", "1.0", "1.0.0.0", "v1.0.0", "1.0.0-rc1", "01.0.0", "", "one.0.0"])
    def test_a_version_that_is_not_one_is_refused(self, value: str) -> None:
        """A reader compares majors, so a version it cannot split is not a version it can compare."""
        with pytest.raises(ValueError, match="semantic versioning"):
            parse_strain_schema_version(value)


class TestStamping:
    """What `declare_strain_schema` writes, and where."""

    def test_an_hdf5_artifact_is_stamped(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        assert declare_strain_schema(path) is True
        assert read_strain_schema(path) == StrainSchema(name=STRAIN_SCHEMA, version=STRAIN_SCHEMA_VERSION)

    def test_the_h5_spelling_is_the_same_format(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.h5")

        assert declare_strain_schema(path) is True
        assert read_strain_schema(path) is not None

    def test_the_declaration_is_written_at_the_root(self, tmp_path: Path) -> None:
        """Where it goes is the whole design, not a detail -- see the sibling test below for why."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            assert handle.attrs[SCHEMA_ATTRIBUTE] == STRAIN_SCHEMA
            assert handle.attrs[SCHEMA_VERSION_ATTRIBUTE] == STRAIN_SCHEMA_VERSION
            assert SCHEMA_VERSION_ATTRIBUTE not in handle["H1:MOCK_NOISE"].attrs

    def test_a_stamped_file_still_reads_through_gwpy(self, tmp_path: Path) -> None:
        """The reason the declaration is not a dataset attribute.

        gwpy hands every dataset attribute to the series constructor as a keyword argument, so an
        attribute it does not know raises `TypeError: Array.__new__() got an unexpected keyword argument
        'schema_version'` and the file becomes unreadable through the standard reader. Declaring the
        schema must not cost a consumer the ability to open the file.
        """
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        series = TimeSeries.read(str(path), format="hdf5")
        assert series.t0.value == START
        assert series.sample_rate.value == RATE
        assert np.array_equal(series.value, np.arange(16, dtype=float))

    def test_the_samples_are_untouched(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            dataset = handle["H1:MOCK_NOISE"]
            assert np.array_equal(dataset[()], np.arange(16, dtype=float))
            assert dict(dataset.attrs)["x0"] == START

    def test_stamping_twice_leaves_one_declaration(self, tmp_path: Path) -> None:
        """A file may be re-stamped -- a merge of merges, a re-run over an existing path."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        declare_strain_schema(path)
        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            assert sorted(handle.attrs) == sorted([SCHEMA_ATTRIBUTE, SCHEMA_VERSION_ATTRIBUTE])

    @pytest.mark.parametrize("name", ["strain.npy", "strain.gwf", "strain.txt"])
    def test_a_format_with_no_attribute_space_is_skipped(self, tmp_path: Path, name: str) -> None:
        """`.npy` is a bare array container and a GWF frame is composed from a fixed set of fields, so
        neither can carry the declaration. A caller writing several formats says so once, here, rather
        than branching on the format at every write site."""
        path = tmp_path / name
        path.write_bytes(b"not hdf5")

        assert declare_strain_schema(path) is False
        assert path.read_bytes() == b"not hdf5"

    def test_a_missing_hdf5_artifact_is_an_error(self, tmp_path: Path) -> None:
        """It is called straight after a write, so an absent file means the write did not happen."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            declare_strain_schema(tmp_path / "never-written.hdf5")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("a.hdf5", True), ("a.h5", True), ("A.HDF5", True), ("a.gwf", False), ("a.npy", False), ("a", False)],
    )
    def test_which_formats_can_carry_it(self, name: str, expected: bool) -> None:
        assert carries_strain_schema(Path(name)) is expected


class TestBringingAFileUpToTheLayout:
    """gwmock composes strain HDF5 through three libraries, and they did not agree on the layout.

    `gwmock_signal.DetectorStrainStack.write` -- the writer behind both halves of an orchestrated run,
    so behind almost every artifact a user ends up with -- records the grid as `t0`/`dt` and writes no
    channel attributes at all. Declaring 1.0.0 over that file announced a layout it did not have. The
    declaration now completes the file first, from what its writer already recorded.
    """

    @staticmethod
    def _aliased(path: Path, *, t0: float = START, dt: float = 1.0 / RATE) -> Path:
        """A dataset in the other spelling, as the multichannel writer leaves it."""
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(16, dtype=float))
            dataset.attrs["t0"] = t0
            dataset.attrs["dt"] = dt
            dataset.attrs["unit"] = "strain"
        return path

    def test_the_grid_is_filled_in_from_the_other_spelling(self, tmp_path: Path) -> None:
        path = self._aliased(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            attributes = handle["H1:MOCK_NOISE"].attrs
            assert attributes["x0"] == START
            assert attributes["dx"] == 1.0 / RATE
            assert attributes["xunit"] == "s"

    def test_the_derived_grid_is_the_one_the_writer_recorded(self, tmp_path: Path) -> None:
        """Derived rather than assumed: a file at another epoch must not acquire the default one."""
        path = self._aliased(tmp_path / "strain.hdf5", t0=612.0, dt=0.5)

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            attributes = handle["H1:MOCK_NOISE"].attrs
            assert attributes["x0"] == 612.0
            assert attributes["dx"] == 0.5

    def test_the_other_spelling_is_left_in_place(self, tmp_path: Path) -> None:
        """Deleting it breaks the writer's own reader: `DetectorStrainStack.read` looks for `t0`."""
        path = self._aliased(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            assert handle["H1:MOCK_NOISE"].attrs["t0"] == START
            assert handle["H1:MOCK_NOISE"].attrs["dt"] == 1.0 / RATE

    def test_the_channel_is_taken_from_the_dataset_name(self, tmp_path: Path) -> None:
        path = self._aliased(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            attributes = handle["H1:MOCK_NOISE"].attrs
            assert attributes["channel"] == "H1:MOCK_NOISE"
            assert attributes["name"] == "H1:MOCK_NOISE"

    def test_what_the_writer_recorded_is_not_overwritten(self, tmp_path: Path) -> None:
        """The writer's own value is the authoritative one; the completion only fills gaps.

        The channel carries this rather than the grid: a dataset whose two grid spellings differ is not
        a question of which to keep but an inconsistent file, and is refused -- see
        `TestAGridThatContradictsItself`.
        """
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(4, dtype=float))
            dataset.attrs["t0"] = 9.0
            dataset.attrs["dt"] = 0.25
            dataset.attrs["channel"] = "SOMETHING:ELSE"
            dataset.attrs["xunit"] = "ms"

        declare_strain_schema(path)

        with h5py.File(path, "r") as handle:
            attributes = handle["H1:MOCK_NOISE"].attrs
            assert attributes["channel"] == "SOMETHING:ELSE"
            assert attributes["xunit"] == "ms"
            assert attributes["name"] == "H1:MOCK_NOISE", "the gap is still filled"
            assert attributes["x0"] == 9.0, "and the grid is still derived"

    def test_every_dataset_in_the_file_is_completed(self, tmp_path: Path) -> None:
        """A multichannel file holds one dataset per channel, and the contract is about all of them."""
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            for channel in ("H1:MOCK", "L1:MOCK"):
                dataset = handle.create_dataset(channel, data=np.arange(4, dtype=float))
                dataset.attrs["t0"] = START
                dataset.attrs["dt"] = 1.0 / RATE

        declare_strain_schema(path)

        assert missing_layout_attributes(path) == {}

    def test_a_dataset_with_no_grid_at_all_is_refused(self, tmp_path: Path) -> None:
        """Nothing is invented from nothing: the alternative is a file declaring a layout it lacks."""
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("H1:MOCK_NOISE", data=np.arange(4, dtype=float))

        with pytest.raises(ValueError, match="carries neither 'x0' nor 't0'"):
            declare_strain_schema(path)

    def test_a_file_it_refuses_is_left_undeclared(self, tmp_path: Path) -> None:
        """The root attributes go on last, so a refusal cannot leave a half-declared artifact."""
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("H1:MOCK_NOISE", data=np.arange(4, dtype=float))

        with pytest.raises(ValueError, match="carries neither"):
            declare_strain_schema(path)

        assert read_strain_schema(path) is None

    def test_a_completed_file_meets_the_layout(self, tmp_path: Path) -> None:
        path = self._aliased(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        assert missing_layout_attributes(path) == {}
        assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    def test_a_completed_file_still_reads_through_gwpy(self, tmp_path: Path) -> None:
        """Both spellings of the grid now sit on the dataset, and gwpy passes every one of them to the
        series constructor -- so this is the check that the pair does not collide there."""
        path = self._aliased(tmp_path / "strain.hdf5")

        declare_strain_schema(path)

        series = TimeSeries.read(str(path), format="hdf5")
        assert series.t0.value == START
        assert series.sample_rate.value == RATE


class TestAGridThatContradictsItself:
    """A file recording the same quantity twice, differently, is refused at both ends.

    The two spellings are read by different consumers -- the content hash takes `x0`/`dx`, and
    `DetectorStrainStack.read` takes `t0`/`dt` -- so a declared file allowed to carry both with
    different values is one artifact that two readers place at two different times, or sample at two
    different rates. Not overwriting the writer's value is right; declaring the file anyway is not, and
    refusing it is what "do not overwrite" leaves available.
    """

    @staticmethod
    def _written(path: Path, **attributes: float) -> Path:
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(8, dtype=float))
            dataset.attrs["unit"] = "strain"
            for name, value in attributes.items():
                dataset.attrs[name] = value
        return path

    @staticmethod
    def _declared_then_edited(path: Path, **edits: float) -> Path:
        """A file gwmock declared cleanly and something changed afterwards -- the only way a declared
        artifact can reach this state, and the reason the consumer checks rather than trusting."""
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(8, dtype=float))
            dataset.attrs["t0"] = START
            dataset.attrs["dt"] = 1.0 / RATE
            dataset.attrs["unit"] = "strain"
        declare_strain_schema(path)
        with h5py.File(path, "a") as handle:
            for name, value in edits.items():
                handle["H1:MOCK_NOISE"].attrs[name] = value
        return path

    def test_an_epoch_conflict_is_refused_at_declaration(self, tmp_path: Path) -> None:
        path = self._written(tmp_path / "strain.hdf5", x0=100.0, t0=200.0, dx=0.5, dt=0.5)

        with pytest.raises(ValueError, match=r"x0=100\.0 but t0=200\.0"):
            declare_strain_schema(path)

    def test_an_interval_conflict_is_refused_at_declaration(self, tmp_path: Path) -> None:
        path = self._written(tmp_path / "strain.hdf5", x0=100.0, t0=100.0, dx=0.5, dt=0.25)

        with pytest.raises(ValueError, match=r"dx=0\.5 but dt=0\.25"):
            declare_strain_schema(path)

    def test_a_refused_file_is_left_undeclared(self, tmp_path: Path) -> None:
        """The check runs before any attribute is written, so a refusal changes nothing."""
        path = self._written(tmp_path / "strain.hdf5", x0=100.0, t0=200.0, dx=0.5, dt=0.5)

        with pytest.raises(ValueError, match="disagree"):
            declare_strain_schema(path)

        assert read_strain_schema(path) is None
        with h5py.File(path, "r") as handle:
            assert "xunit" not in handle["H1:MOCK_NOISE"].attrs

    def test_an_epoch_conflict_is_refused_at_require_time(self, tmp_path: Path) -> None:
        path = self._declared_then_edited(tmp_path / "strain.hdf5", t0=612.0)

        assert conflicting_grid_attributes(path) == {"H1:MOCK_NOISE": ["x0=100.0 but t0=612.0"]}
        with pytest.raises(ValueError, match="grid contradicts itself"):
            require_strain_schema(path)

    def test_an_interval_conflict_is_refused_at_require_time(self, tmp_path: Path) -> None:
        path = self._declared_then_edited(tmp_path / "strain.hdf5", dt=0.5)

        assert conflicting_grid_attributes(path) == {"H1:MOCK_NOISE": ["dx=0.125 but dt=0.5"]}
        with pytest.raises(ValueError, match="grid contradicts itself"):
            require_strain_schema(path)

    def test_a_consistent_duplicate_is_not_a_conflict(self, tmp_path: Path) -> None:
        """The ordinary case: the pair gwmock itself derives, and the pair a writer records twice."""
        path = self._written(tmp_path / "strain.hdf5", x0=100.0, t0=100.0, dx=0.5, dt=0.5)

        declare_strain_schema(path)

        assert conflicting_grid_attributes(path) == {}
        assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    def test_one_spelling_alone_is_not_a_conflict(self, tmp_path: Path) -> None:
        """Nothing to disagree with. Both single-spelling shapes go through the ordinary path."""
        aliased = self._written(tmp_path / "aliased.hdf5", t0=100.0, dt=0.125)
        canonical = self._written(tmp_path / "canonical.hdf5", x0=100.0, dx=0.125)

        for path in (aliased, canonical):
            declare_strain_schema(path)
            assert conflicting_grid_attributes(path) == {}

    def test_a_refusal_does_not_half_convert_a_multichannel_file(self, tmp_path: Path) -> None:
        """The datasets before the offending one must not be written either.

        Checking and completing one dataset at a time reads naturally and is wrong here: a file whose
        first channel is derivable and whose second contradicts itself would be refused *after* the
        first had been completed, leaving an artifact that is undeclared but no longer what its writer
        produced. Both orderings are covered because the property is "nothing is written", not "the
        first dataset happens to be safe".
        """
        h1_first, l1_second = ({"t0": START, "dt": 1.0 / RATE}, {"x0": 100.0, "t0": 200.0})
        for name, (h1, l1) in (
            ("derivable-then-conflicting", (h1_first, l1_second)),
            ("conflicting-then-derivable", (l1_second, h1_first)),
        ):
            path = tmp_path / f"{name}.hdf5"
            with h5py.File(path, "w") as handle:
                for channel, attributes in (("H1:MOCK", h1), ("L1:MOCK", l1)):
                    dataset = handle.create_dataset(channel, data=np.arange(4, dtype=float))
                    for attribute, value in attributes.items():
                        dataset.attrs[attribute] = value
            before = _attribute_snapshot(path)
            before_bytes = path.read_bytes()

            with pytest.raises(ValueError, match="disagree"):
                declare_strain_schema(path)

            assert _attribute_snapshot(path) == before, f"{name}: a dataset was written before the refusal"
            assert path.read_bytes() == before_bytes, f"{name}: the file was modified by a refusal"
            assert read_strain_schema(path) is None

    def test_a_missing_grid_in_a_later_dataset_writes_nothing_either(self, tmp_path: Path) -> None:
        """The other reason to refuse takes the same path, so it gets the same guarantee."""
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            derivable = handle.create_dataset("H1:MOCK", data=np.arange(4, dtype=float))
            derivable.attrs["t0"] = START
            derivable.attrs["dt"] = 1.0 / RATE
            handle.create_dataset("L1:MOCK", data=np.arange(4, dtype=float))
        before = _attribute_snapshot(path)

        with pytest.raises(ValueError, match="carries neither"):
            declare_strain_schema(path)

        assert _attribute_snapshot(path) == before
        assert read_strain_schema(path) is None

    def test_the_conflict_is_reported_for_the_dataset_that_has_it(self, tmp_path: Path) -> None:
        """A multichannel file: one bad dataset must not be hidden by its well-formed neighbours."""
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            for channel in ("H1:MOCK", "L1:MOCK"):
                dataset = handle.create_dataset(channel, data=np.arange(4, dtype=float))
                dataset.attrs["x0"] = START
                dataset.attrs["t0"] = START
                dataset.attrs["dx"] = 1.0 / RATE
                dataset.attrs["dt"] = 1.0 / RATE
            handle["L1:MOCK"].attrs["t0"] = 612.0

        assert sorted(conflicting_grid_attributes(path)) == ["L1:MOCK"]
        with pytest.raises(ValueError, match="L1:MOCK"):
            declare_strain_schema(path)


class TestReadingTheDeclaration:
    """What a consumer sees when it looks."""

    def test_an_undeclared_file_reads_as_none(self, tmp_path: Path) -> None:
        """A file written before the declaration existed, or by another producer."""
        assert read_strain_schema(_hdf5_strain(tmp_path / "strain.hdf5")) is None

    def test_a_half_declared_file_reads_as_none(self, tmp_path: Path) -> None:
        """A version with no schema name says nothing: `schema_version` is a common attribute name and
        another producer's is not this contract."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.0.0"

        assert read_strain_schema(path) is None

    def test_a_format_that_cannot_carry_it_reads_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "strain.npy"
        np.save(path, np.arange(4, dtype=float))

        assert read_strain_schema(path) is None

    def test_a_byte_string_attribute_is_decoded(self, tmp_path: Path) -> None:
        """h5py hands back `bytes` for an attribute written by a writer that used a fixed-length string
        type, which is what a non-Python producer typically writes."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs.create(SCHEMA_ATTRIBUTE, np.bytes_(STRAIN_SCHEMA))
            handle.attrs.create(SCHEMA_VERSION_ATTRIBUTE, np.bytes_(STRAIN_SCHEMA_VERSION))

        assert read_strain_schema(path) == StrainSchema(name=STRAIN_SCHEMA, version=STRAIN_SCHEMA_VERSION)


class TestRequiringTheDeclaration:
    """The consumer-side half: what gets refused, and with what said about it."""

    def test_a_stamped_file_is_accepted(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        declare_strain_schema(path)

        assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    def test_an_undeclared_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="declares no strain schema"):
            require_strain_schema(_hdf5_strain(tmp_path / "strain.hdf5"))

    def test_another_producers_schema_is_refused(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = "someone-elses-strain"
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.0.0"

        with pytest.raises(ValueError, match="declares schema 'someone-elses-strain'"):
            require_strain_schema(path)

    def test_a_future_major_version_is_refused(self, tmp_path: Path) -> None:
        """The point of the version: a layout this reader was not written against is a refusal at open
        time rather than a wrong number later."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "2.0.0"

        with pytest.raises(ValueError, match="major version 2"):
            require_strain_schema(path)

    def test_a_later_minor_version_is_accepted(self, tmp_path: Path) -> None:
        """A minor release adds what an existing reader can ignore, so refusing it would be wrong."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.7.3"

        assert require_strain_schema(path).version == "1.7.3"

    def test_a_declared_file_that_does_not_carry_the_layout_is_refused(self, tmp_path: Path) -> None:
        """A declaration a writer got wrong is worth exactly as little as no declaration at all.

        This is the check that would have caught the multichannel writer emitting the grid under a name
        the contract did not promise, instead of a consumer discovering it as a missing attribute.
        """
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(4, dtype=float))
            dataset.attrs["t0"] = START
            dataset.attrs["dt"] = 1.0 / RATE
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = STRAIN_SCHEMA_VERSION

        with pytest.raises(ValueError, match="does not carry its layout"):
            require_strain_schema(path)

    def test_the_refusal_names_what_is_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "strain.hdf5"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("H1:MOCK_NOISE", data=np.arange(4, dtype=float))
            dataset.attrs["x0"] = START
            dataset.attrs["dx"] = 1.0 / RATE
            dataset.attrs["xunit"] = "s"
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = STRAIN_SCHEMA_VERSION

        assert missing_layout_attributes(path) == {"H1:MOCK_NOISE": ["channel", "name"]}
        with pytest.raises(ValueError, match=r"H1:MOCK_NOISE is missing channel, name"):
            require_strain_schema(path)

    def test_a_malformed_version_is_refused(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "not-a-version"

        with pytest.raises(ValueError, match="semantic versioning"):
            require_strain_schema(path)


class TestReadingTheEmbeddedRecord:
    """The consumer side of `run_metadata`: a file that describes its own run, or says it cannot."""

    def test_a_record_written_into_a_file_reads_back(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        embed_metadata_record(path, {"schema_version": "1.5.0", "seed": 7})

        assert read_run_metadata(path) == {"schema_version": "1.5.0", "seed": 7}

    def test_a_file_carrying_none_reads_as_none(self, tmp_path: Path) -> None:
        """Everything written before 1.1.0, and everything another producer wrote: read the sidecar."""
        assert read_run_metadata(_hdf5_strain(tmp_path / "strain.hdf5")) is None

    def test_a_format_that_cannot_carry_one_reads_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "strain.npy"
        np.save(path, np.arange(4, dtype=float))

        assert read_run_metadata(path) is None

    def test_an_attribute_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        """No gwmock wrote it, so it is another producer's attribute of the same name -- and returning
        None would file that under 'no record' rather than under 'this file is not what it claims'."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[RUN_METADATA_ATTRIBUTE] = "not json"

        with pytest.raises(ValueError, match="is not JSON"):
            read_run_metadata(path)

    @pytest.mark.parametrize(
        ("document", "described"),
        [("null", "NoneType"), ("[1, 2]", "list"), ('"a string"', "str"), ("7", "int")],
    )
    def test_json_that_is_not_an_object_is_refused(self, tmp_path: Path, document: str, described: str) -> None:
        """A record is an object, and `null` is the case that would otherwise pass silently: it
        decodes to None, which is this function's way of saying "no record, read the sidecar", so a
        consumer would be sent to the sidecar rather than told the file carries something unusable."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[RUN_METADATA_ATTRIBUTE] = document

        with pytest.raises(ValueError, match=f"JSON {described}, not an object"):
            read_run_metadata(path)

    def test_the_record_does_not_cost_the_consumer_the_standard_reader(self, tmp_path: Path) -> None:
        """The same reason the declaration goes at the root: gwpy hands every *dataset* attribute to
        the series constructor, so a record written there would make the file unreadable."""
        path = _hdf5_strain(tmp_path / "strain.hdf5", channel="H1:MOCK_NOISE")
        declare_strain_schema(path)
        embed_metadata_record(path, {"schema_version": "1.5.0"})

        assert np.allclose(TimeSeries.read(path, "H1:MOCK_NOISE").value, np.arange(16, dtype=float), atol=0.0)

    def test_the_record_does_not_disturb_the_declaration(self, tmp_path: Path) -> None:
        """Two root attributes written by two calls, in either order, and both survive."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        embed_metadata_record(path, {"schema_version": "1.5.0"})
        declare_strain_schema(path)

        assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION
        assert read_run_metadata(path) == {"schema_version": "1.5.0"}
