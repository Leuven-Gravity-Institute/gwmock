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
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.utils.config import get_examples_dir, load_config

_EXAMPLES = get_examples_dir()
_RUNNER = CliRunner()


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
        assert "Run gwmock from" in str(result.exception) + result.output
        assert not (tmp_path / "output").exists()
