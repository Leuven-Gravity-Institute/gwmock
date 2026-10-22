"""An example copied with ``gwmock config --get`` must run without network access.

The quick-start used to read its one-event population from a URL on the default branch, and the
wheel shipped only the YAML files, so an offline machine could not run the quick-start at all. The
catalogues now ship with the package; ``--get`` copies the one an example references next to the
configuration it writes and points the copy at it.

Nothing here touches the network: a URL left in a copied configuration fails the first assertion.
"""

from __future__ import annotations

import filecmp
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.utils.config import get_examples_dir, load_config

_EXAMPLES = get_examples_dir()
_RUNNER = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _get(label: str, output: str, *extra: str):
    return _RUNNER.invoke(app, ["config", "--get", label, "--output", output, *extra])


@pytest.mark.parametrize(
    ("label", "catalogue"),
    [
        ("noise/uncorrelated_gaussian/quick_start", "signal/bbh_population.csv"),
        ("signal/waveform_backend/ripple", "signal/bbh_population.csv"),
        ("signal/execution/batched", "signal/bbh_population.csv"),
        ("signal/cw/et_triangle_sardinia", "signal/cw_population.csv"),
        ("signal/bbh/ligo_hl", "signal/bbh_population.csv"),
    ],
)
def test_the_copy_reads_the_shipped_catalogue_from_beside_it(tmp_path: Path, monkeypatch, label, catalogue):
    monkeypatch.chdir(tmp_path)

    result = _get(label, "config.yaml")

    assert result.exit_code == 0, result.output
    path = load_config(Path("config.yaml")).orchestration.population.arguments["path"]
    assert "://" not in path, f"the copied configuration still fetches its population: {path}"
    assert not Path(path).is_absolute()
    assert Path(path).is_file(), f"{path} does not exist beside the copied configuration"
    assert filecmp.cmp(path, _EXAMPLES / catalogue, shallow=False)


def test_a_copy_written_elsewhere_names_its_catalogue_from_the_working_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = _get("noise/uncorrelated_gaussian/quick_start", os.path.join("run", "config.yaml"))

    assert result.exit_code == 0, result.output
    path = load_config(Path("run/config.yaml")).orchestration.population.arguments["path"]
    assert path == "run/bbh_population.csv"
    assert Path(path).is_file()


def test_a_different_file_in_the_way_is_not_replaced_silently(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("bbh_population.csv").write_text("my own catalogue\n", encoding="utf-8")

    result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

    assert result.exit_code != 0
    assert Path("bbh_population.csv").read_text(encoding="utf-8") == "my own catalogue\n"
    assert not Path("config.yaml").exists(), "the configuration was written pointing at the wrong catalogue"


def test_every_shipped_example_names_no_population_on_a_branch():
    """A URL on a branch can change under an example; a remote population, if any, is pinned."""
    for config_file in _EXAMPLES.rglob("config.yaml"):
        assert "/main/" not in config_file.read_text(encoding="utf-8"), config_file


class TestRunningAnExampleInPlace:
    """Relative paths are read from the working directory; an example run from elsewhere says so."""

    _QUICK_START = _EXAMPLES / "noise" / "uncorrelated_gaussian" / "quick_start" / "config.yaml"

    def test_a_catalogue_beside_the_file_but_not_here_names_where_to_run(self, tmp_path: Path, monkeypatch):
        from gwmock.cli.utils.config import require_population_path_found
        from gwmock.utils.errors import DeterministicRefusalError

        monkeypatch.chdir(tmp_path)
        with pytest.raises(DeterministicRefusalError, match=r"Run gwmock from .*quick_start.*gwmock config --get"):
            require_population_path_found(load_config(self._QUICK_START), self._QUICK_START)

    def test_run_from_the_example_directory_the_path_is_found(self, monkeypatch):
        from gwmock.cli.utils.config import require_population_path_found

        monkeypatch.chdir(self._QUICK_START.parent)
        require_population_path_found(load_config(self._QUICK_START), self._QUICK_START)

    def test_gwmock_simulate_stops_before_anything_starts(self, tmp_path: Path, monkeypatch):
        """Through the CLI, the dry run included: the plan is never built."""
        monkeypatch.chdir(tmp_path)

        result = _RUNNER.invoke(app, ["simulate", str(self._QUICK_START), "--dry-run"])

        assert result.exit_code != 0
        # The message reaches the output through the log handler, which wraps it to the console
        # width and colours it under a CI service; compare it with the layout taken out.
        shown = _ANSI_ESCAPE.sub("", str(result.exception) + result.output)
        assert "Run gwmock from" in " ".join(shown.split())
        assert not (tmp_path / "output").exists()


class TestWritingBesideTheCatalogue:
    """``--output`` into the directory the catalogue already lives in must not copy it onto itself."""

    def test_a_copy_written_into_the_catalogues_own_directory(self, tmp_path: Path, monkeypatch):
        import shutil

        examples = tmp_path / "examples"
        shutil.copytree(_EXAMPLES, examples)
        monkeypatch.setattr("gwmock.cli.utils.config.get_examples_dir", lambda: examples)
        catalogue_directory = examples / "signal"
        monkeypatch.chdir(tmp_path)

        result = _get("signal/bbh/ligo_hl", str(catalogue_directory / "ligo.yaml"))

        assert result.exit_code == 0, result.output
        path = load_config(catalogue_directory / "ligo.yaml").orchestration.population.arguments["path"]
        assert Path(path).resolve() == (catalogue_directory / "bbh_population.csv").resolve()

    def test_an_identical_catalogue_already_there_is_kept(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bbh_population.csv").write_bytes((_EXAMPLES / "signal" / "bbh_population.csv").read_bytes())

        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code == 0, result.output

    def test_a_configuration_that_cannot_be_written_leaves_no_catalogue_behind(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("gwmock.cli.utils.config.save_config", fail)
        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code != 0
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_a_catalogue_that_cannot_be_copied_leaves_no_configuration_behind(self, tmp_path: Path, monkeypatch):
        """Written first, the configuration would point at a catalogue that never arrived."""
        monkeypatch.chdir(tmp_path)

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("shutil.copyfileobj", fail)
        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code != 0
        assert not (tmp_path / "config.yaml").exists()


class TestTheCatalogueAndConfigurationAreInstalledTogether:
    """A failed ``config --get`` leaves the destination as it was, and leaves no temporary file."""

    _LABEL = "noise/uncorrelated_gaussian/quick_start"
    _OLD_CATALOGUE = b"my own catalogue\n"
    _OLD_CONFIG = "globals: {}\n# the user's own configuration\n"

    @staticmethod
    def _partial_copy(source, destination, *args, **kwargs):
        del source, args, kwargs
        destination.write(b"partial")
        destination.flush()
        raise OSError("disk full partway")

    def test_a_partial_copy_leaves_an_existing_catalogue_byte_identical(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bbh_population.csv").write_bytes(self._OLD_CATALOGUE)
        monkeypatch.setattr("shutil.copyfileobj", self._partial_copy)

        result = _get(self._LABEL, "config.yaml", "--overwrite")

        assert result.exit_code != 0
        assert (tmp_path / "bbh_population.csv").read_bytes() == self._OLD_CATALOGUE
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bbh_population.csv"]

    def test_a_partial_copy_creates_nothing_when_nothing_was_there(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("shutil.copyfileobj", self._partial_copy)

        result = _get(self._LABEL, "config.yaml")

        assert result.exit_code != 0
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_save_restores_the_catalogue_it_overwrote(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bbh_population.csv").write_bytes(self._OLD_CATALOGUE)
        (tmp_path / "config.yaml").write_text(self._OLD_CONFIG)

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("gwmock.cli.utils.config.save_config", fail)
        result = _get(self._LABEL, "config.yaml", "--overwrite")

        assert result.exit_code != 0
        assert (tmp_path / "bbh_population.csv").read_bytes() == self._OLD_CATALOGUE
        assert (tmp_path / "config.yaml").read_text() == self._OLD_CONFIG
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bbh_population.csv", "config.yaml"]

    def test_an_overwrite_that_succeeds_replaces_both_and_leaves_nothing_else(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bbh_population.csv").write_bytes(self._OLD_CATALOGUE)
        (tmp_path / "config.yaml").write_text(self._OLD_CONFIG)

        result = _get(self._LABEL, "config.yaml", "--overwrite")

        assert result.exit_code == 0, result.output
        assert filecmp.cmp(tmp_path / "bbh_population.csv", _EXAMPLES / "signal" / "bbh_population.csv", shallow=False)
        assert load_config(tmp_path / "config.yaml").orchestration.population.arguments["path"] == "bbh_population.csv"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bbh_population.csv", "config.yaml"]

    def test_a_first_copy_that_succeeds_leaves_nothing_else(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        result = _get(self._LABEL, "config.yaml")

        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bbh_population.csv", "config.yaml"]


class TestRestoringComesBeforeCleaningUp:
    """A failure deleting a temporary file must not stop the old catalogue being put back."""

    def test_a_failing_unlink_of_the_staged_file_still_restores_the_catalogue(self, tmp_path: Path, monkeypatch):
        import os

        from gwmock.cli.config import _DataFileInstall

        destination = tmp_path / "bbh_population.csv"
        destination.write_bytes(b"old catalogue\n")
        real_replace, real_unlink = os.replace, Path.unlink

        def replace(source, target):
            if str(source).endswith(".part"):
                raise OSError("rename refused")
            return real_replace(source, target)

        def unlink(self, *args, **kwargs):
            if self.name.endswith(".part"):
                raise OSError("unlink refused")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr("os.replace", replace)
        monkeypatch.setattr(Path, "unlink", unlink)

        with pytest.raises(OSError, match="rename refused"):
            _DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)

        assert destination.read_bytes() == b"old catalogue\n"

    def test_a_restore_that_fails_names_the_backup(self, tmp_path: Path, monkeypatch):
        import os

        from gwmock.cli.config import _DataFileInstall

        destination = tmp_path / "bbh_population.csv"
        destination.write_bytes(b"old catalogue\n")
        install = _DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)
        backup = install._backup
        real_replace = os.replace

        def replace(source, target):
            if str(source).endswith(".backup"):
                raise OSError("rename refused")
            return real_replace(source, target)

        monkeypatch.setattr("os.replace", replace)
        with pytest.raises(RuntimeError, match=re.escape(str(backup))):
            install.roll_back()
        assert backup.read_bytes() == b"old catalogue\n"


class TestLeftoversOfAnInterruptedCopy:
    _BACKUP = ".gwmock-config-get.bbh_population.csv.0123456789abcdef.backup"
    _PART = ".gwmock-config-get.bbh_population.csv.fedcba9876543210.part"

    @pytest.mark.parametrize("leftover", [_BACKUP, _PART])
    def test_they_are_named_and_nothing_is_changed(self, tmp_path: Path, monkeypatch, leftover):
        monkeypatch.chdir(tmp_path)
        (tmp_path / leftover).write_bytes(b"old catalogue\n")

        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code != 0
        assert leftover in str(result.exception)
        assert sorted(p.name for p in tmp_path.iterdir()) == [leftover]
        assert (tmp_path / leftover).read_bytes() == b"old catalogue\n"

    def test_a_backup_is_said_to_hold_the_previous_catalogue(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / self._BACKUP).write_bytes(b"old catalogue\n")

        message = str(_get("noise/uncorrelated_gaussian/quick_start", "config.yaml").exception)

        assert "holds the catalogue that was at" in message
        assert "To restore it, rename it to" in message

    def test_a_part_alone_is_said_to_be_safe_to_delete(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / self._PART).write_bytes(b"partial")

        message = str(_get("noise/uncorrelated_gaussian/quick_start", "config.yaml").exception)

        assert "unfinished staged copy and safe to delete" in message
        assert "was not moved" in message
        assert "rename" not in message

    def test_a_users_own_backup_is_ignored(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".bbh_population.csv.notes.backup").write_text("my notes\n")

        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code == 0, result.output
        assert (tmp_path / ".bbh_population.csv.notes.backup").read_text() == "my notes\n"

    @pytest.mark.parametrize("overwrite", [(), ("--overwrite",)])
    def test_they_are_found_when_the_catalogue_already_holds_the_shipped_bytes(
        self, tmp_path: Path, monkeypatch, overwrite
    ):
        """What a kill after the install, before the backup was removed, leaves: no copy is needed, yet the backup is named."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bbh_population.csv").write_bytes((_EXAMPLES / "signal" / "bbh_population.csv").read_bytes())
        (tmp_path / self._BACKUP).write_bytes(b"old catalogue\n")

        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml", *overwrite)

        assert result.exit_code != 0
        assert self._BACKUP in str(result.exception)
        assert not (tmp_path / "config.yaml").exists()

    def test_the_temporary_files_carry_the_reserved_prefix(self, tmp_path: Path) -> None:
        from gwmock.cli.config import _DataFileInstall, _leftovers_of_an_interrupted_copy

        destination = tmp_path / "bbh_population.csv"
        destination.write_bytes(b"old catalogue\n")
        install = _DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)
        assert _leftovers_of_an_interrupted_copy(destination) == [install._backup]
        install.commit()
        assert _leftovers_of_an_interrupted_copy(destination) == []

    def test_an_mkstemp_shaped_name_with_the_prefix_is_not_taken_for_one(self, tmp_path: Path) -> None:
        from gwmock.cli.config import _leftovers_of_an_interrupted_copy

        destination = tmp_path / "bbh_population.csv"
        (tmp_path / ".gwmock-config-get.bbh_population.csv.abc_1234.backup").write_text("x")
        assert _leftovers_of_an_interrupted_copy(destination) == []

    def test_a_taken_name_is_retried_with_a_new_token(self, tmp_path: Path, monkeypatch) -> None:
        from gwmock.cli import config as config_module

        destination = tmp_path / "bbh_population.csv"
        taken = tmp_path / ".gwmock-config-get.bbh_population.csv.0000000000000000.part"
        taken.write_bytes(b"someone else's")
        tokens = iter(["0000000000000000", "1111111111111111"])
        monkeypatch.setattr(config_module, "_temporary_token", lambda: next(tokens))

        created = config_module._create_temporary_file(destination, ".part")

        assert created.name == ".gwmock-config-get.bbh_population.csv.1111111111111111.part"
        assert taken.read_bytes() == b"someone else's"


class TestTheStagedCopyIsWrittenThroughItsOwnDescriptor:
    """The staged ``.part`` is never reopened by path, and a swapped entry is refused, not followed."""

    def test_the_copy_does_not_reopen_the_staged_file(self, tmp_path: Path, monkeypatch):
        def reopen(*_args, **_kwargs):
            raise AssertionError("the staged file was reopened by path")

        monkeypatch.setattr("shutil.copyfile", reopen)
        monkeypatch.chdir(tmp_path)

        result = _get("noise/uncorrelated_gaussian/quick_start", "config.yaml")

        assert result.exit_code == 0, result.output
        assert filecmp.cmp(tmp_path / "bbh_population.csv", _EXAMPLES / "signal" / "bbh_population.csv", shallow=False)

    def test_a_staged_file_swapped_for_a_symlink_is_refused(self, tmp_path: Path, monkeypatch):
        import os

        from gwmock.cli import config as config_module

        destination = tmp_path / "run" / "bbh_population.csv"
        destination.parent.mkdir()
        destination.write_bytes(b"old catalogue\n")
        target = tmp_path / "victim.txt"
        target.write_bytes(b"not yours\n")

        def swap(staging: Path) -> None:
            staging.unlink()
            try:
                os.symlink(target, staging)
            except (OSError, NotImplementedError) as error:
                pytest.skip(f"symlinks are unavailable here: {error}")

        monkeypatch.setattr(config_module, "_after_staging", swap)

        with pytest.raises(RuntimeError, match="was replaced by another file"):
            config_module._DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)

        assert target.read_bytes() == b"not yours\n"
        assert destination.read_bytes() == b"old catalogue\n"
        assert [p for p in destination.parent.iterdir() if p.is_symlink()], "the swapped entry was removed"


class TestAFailureAfterTheMoveIntoPlace:
    """Once the new catalogue is in place, a later failure must still put the old one back.

    The failures are injected by wrapping ``os.replace``: the real move of the staged copy into place
    runs, and the failure follows it -- the moment between the move and the end of the install.
    """

    _OLD = b"old catalogue\n"

    def _destination(self, tmp_path: Path) -> Path:
        destination = tmp_path / "run" / "bbh_population.csv"
        destination.parent.mkdir()
        destination.write_bytes(self._OLD)
        return destination

    @staticmethod
    def _after_the_move(monkeypatch, action) -> None:
        import os

        real_replace = os.replace

        def replace(source, target):
            real_replace(source, target)
            if str(source).endswith(".part"):
                action()

        monkeypatch.setattr("os.replace", replace)

    def test_a_failing_close_after_the_move_is_not_a_failure(self, tmp_path: Path, monkeypatch, caplog):
        """The copy was fsynced and moved, so the install stands; the pair is completed as usual."""
        import os

        from gwmock.cli import config as config_module

        destination = self._destination(tmp_path)
        real_close = os.close
        state = {"moved": False}
        self._after_the_move(monkeypatch, lambda: state.update(moved=True))

        def close(descriptor):
            real_close(descriptor)
            if state["moved"]:
                raise OSError("close refused")

        monkeypatch.setattr("os.close", close)
        with caplog.at_level("WARNING", logger="gwmock"):
            install = config_module._DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)
        monkeypatch.setattr("os.close", real_close)
        install.commit()

        assert filecmp.cmp(destination, _EXAMPLES / "signal" / "bbh_population.csv", shallow=False)
        assert "Could not close" in caplog.text
        assert sorted(p.name for p in destination.parent.iterdir()) == ["bbh_population.csv"]

    @pytest.mark.parametrize("interruption", [RuntimeError("late failure"), KeyboardInterrupt()])
    def test_an_interruption_after_the_move_restores_the_old_catalogue(self, tmp_path: Path, monkeypatch, interruption):
        from gwmock.cli import config as config_module

        destination = self._destination(tmp_path)

        def interrupt():
            raise interruption

        self._after_the_move(monkeypatch, interrupt)

        with pytest.raises(type(interruption)):
            config_module._DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)

        assert destination.read_bytes() == self._OLD
        assert sorted(p.name for p in destination.parent.iterdir()) == ["bbh_population.csv"]

    def test_an_interruption_after_a_first_install_leaves_nothing(self, tmp_path: Path, monkeypatch):
        from gwmock.cli import config as config_module

        destination = tmp_path / "bbh_population.csv"

        def interrupt():
            raise KeyboardInterrupt

        self._after_the_move(monkeypatch, interrupt)

        with pytest.raises(KeyboardInterrupt):
            config_module._DataFileInstall(_EXAMPLES / "signal" / "bbh_population.csv", destination)

        assert list(tmp_path.iterdir()) == []


class TestWhetherTheMoveHappenedIsDecidedByIdentity:
    """Only the very file that was staged counts as installed; anything else is left alone."""

    _SOURCE = _EXAMPLES / "signal" / "bbh_population.csv"

    @pytest.fixture(params=[False, True], ids=["posix-order", "windows-order"])
    def closes_before_rename(self, request, monkeypatch):
        from gwmock.cli import config as config_module

        monkeypatch.setattr(config_module, "_closes_before_rename", lambda: request.param)
        return request.param

    def test_an_unrelated_destination_is_not_taken_for_the_install(
        self, tmp_path: Path, monkeypatch, closes_before_rename
    ):
        """The staged copy vanished and another process created the destination: neither is ours to delete."""
        import os

        from gwmock.cli import config as config_module

        destination = tmp_path / "bbh_population.csv"
        real_replace = os.replace

        def replace(source, target):
            if str(source).endswith(".part"):
                # The other file first: an inode freed by the unlink could otherwise be reused for it
                # on this test's file system, which NTFS file IDs, carrying a sequence number, are not.
                Path(target).write_bytes(b"someone else's catalogue\n")
                Path(source).unlink()
                raise OSError("interrupted")
            return real_replace(source, target)

        monkeypatch.setattr("os.replace", replace)
        with pytest.raises(OSError, match="interrupted"):
            config_module._DataFileInstall(self._SOURCE, destination)

        assert destination.read_bytes() == b"someone else's catalogue\n"

    def test_the_previous_catalogue_is_untouched_when_no_backup_was_taken(
        self, tmp_path: Path, monkeypatch, closes_before_rename
    ):
        from gwmock.cli import config as config_module

        destination = tmp_path / "bbh_population.csv"
        destination.write_bytes(b"old catalogue\n")
        real_create = config_module._create_temporary_file

        def create(path, suffix):
            if suffix == ".backup":
                for staged in tmp_path.glob("*.part"):
                    staged.unlink()
                raise OSError("no room for a backup")
            return real_create(path, suffix)

        monkeypatch.setattr(config_module, "_create_temporary_file", create)
        with pytest.raises(OSError, match="no room"):
            config_module._DataFileInstall(self._SOURCE, destination)

        assert destination.read_bytes() == b"old catalogue\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bbh_population.csv"]

    def test_without_an_identity_nothing_is_undone_and_the_paths_are_named(self, tmp_path: Path, monkeypatch):
        import os

        from gwmock.cli import config as config_module

        destination = tmp_path / "bbh_population.csv"
        destination.write_bytes(b"old catalogue\n")
        monkeypatch.setattr(
            config_module, "_witness", lambda descriptor: config_module._identity_from(_NoInode(os.fstat(descriptor)))
        )
        real_replace = os.replace

        def replace(source, target):
            real_replace(source, target)
            if str(source).endswith(".part"):
                raise KeyboardInterrupt

        monkeypatch.setattr("os.replace", replace)
        with pytest.raises(RuntimeError, match="cannot be established") as excinfo:
            config_module._DataFileInstall(self._SOURCE, destination)

        backups = sorted(tmp_path.glob("*.backup"))
        assert len(backups) == 1
        assert str(backups[0]) in str(excinfo.value)
        assert backups[0].read_bytes() == b"old catalogue\n"
        assert filecmp.cmp(destination, self._SOURCE, shallow=False)


class _NoInode:
    """A stat result whose file system reports no inode."""

    def __init__(self, status) -> None:
        self.st_dev, self.st_mode, self.st_ino = status.st_dev, status.st_mode, 0
