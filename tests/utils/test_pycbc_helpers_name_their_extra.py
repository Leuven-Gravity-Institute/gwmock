"""The PyCBC helpers in ``gwmock.utils`` say which extra installs what they are missing."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.requires_h5py
@pytest.mark.parametrize(
    ("module", "missing"),
    [
        ("gwmock.utils.population", "pandas"),
        ("gwmock.utils.et_2l_geometry", "pycbc"),
        ("gwmock.utils.triangular_et_geometry", "pycbc"),
    ],
)
def test_a_missing_package_names_the_extra(monkeypatch, module, missing):
    """Importing the helper without its package raises naming ``gwmock[pycbc]``, not a bare import error."""
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock\[pycbc\]'") as raised:
        importlib.import_module(module)
    # The import that fails is `pycbc.detector`, so its name may be that submodule.
    assert raised.value.name.split(".")[0] == missing


class _FailsInside:
    """A finder that makes importing *package* fail on one of its own modules, as a broken install does."""

    def __init__(self, package: str) -> None:
        self.package = package

    def find_spec(self, name, path=None, target=None):
        if name == self.package:
            raise ModuleNotFoundError(f"No module named '{name}._internal'", name=f"{name}._internal")


@pytest.mark.requires_h5py
@pytest.mark.parametrize(
    ("module", "package"),
    [
        ("gwmock.utils.population", "pandas"),
        ("gwmock.utils.et_2l_geometry", "pycbc"),
        ("gwmock.utils.triangular_et_geometry", "pycbc"),
    ],
)
def test_a_broken_installation_is_not_reported_as_missing(monkeypatch, module, package):
    """A module missing from inside an installed package is re-raised as it is, without an install hint."""
    for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FailsInside(package), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError) as raised:
        importlib.import_module(module)
    assert raised.value.name == f"{package}._internal"
    assert "pip install" not in str(raised.value)
