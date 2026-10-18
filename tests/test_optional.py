"""``require``: an optional package that is missing is reported with the extra that installs it."""

from __future__ import annotations

import importlib
import types

import pytest

from gwmock._optional import MissingExtraError, require

pytestmark = pytest.mark.unit


def _failing_import(missing: str):
    """An ``import_module`` stand-in that fails as if ``missing`` were not installed."""

    def fake(name: str) -> types.ModuleType:
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    return fake


def test_an_installed_module_is_returned():
    """An importable module comes back as the module itself."""
    assert require("json", "unused", "Anything") is importlib.import_module("json")


def test_a_missing_package_names_the_extra(monkeypatch):
    """The message names the feature, the package and the install command; ``name`` is kept."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("h5py"))

    with pytest.raises(ModuleNotFoundError, match=r"HDF5 output needs h5py") as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert "pip install 'gwmock[hdf5]'" in str(caught.value)
    assert caught.value.name == "h5py"
    assert isinstance(caught.value, MissingExtraError)


def test_a_missing_enclosing_package_names_the_extra(monkeypatch):
    """Asking for ``gwpy.timeseries`` when ``gwpy`` itself is absent."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("gwpy"))

    with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock\[gwpy\]'"):
        require("gwpy.timeseries", "gwpy", "GWpy output")


def test_a_broken_installation_is_not_reported_as_missing(monkeypatch):
    """A module inside an installed package failing to import is re-raised unchanged."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("h5py._internal"))

    with pytest.raises(ModuleNotFoundError) as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert caught.value.name == "h5py._internal"
    assert "pip install" not in str(caught.value)


def test_a_missing_dependency_of_the_package_is_not_reported_as_the_package(monkeypatch):
    """A module the package needs, rather than the package, is re-raised unchanged."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("numpy"))

    with pytest.raises(ModuleNotFoundError) as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert caught.value.name == "numpy"
    assert "pip install" not in str(caught.value)


def test_a_content_hash_does_not_fall_back_when_an_extra_is_missing(monkeypatch, tmp_path):
    """Hashing an HDF5 file without h5py raises, rather than recording the raw-file hash instead."""
    from gwmock.cli.utils import hash as hash_module

    def missing(module, extra, feature):
        raise MissingExtraError(f"{feature} needs h5py. Install it with: pip install 'gwmock[{extra}]'", name="h5py")

    monkeypatch.setattr(hash_module, "require", missing)
    path = tmp_path / "artifact.h5"
    path.write_bytes(b"\0" * 16)

    with pytest.raises(MissingExtraError, match=r"gwmock\[hdf5\]"):
        hash_module.compute_content_hash(path)
