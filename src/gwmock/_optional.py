"""Import an optional package, naming the extra that installs it when it is missing."""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtraError(ModuleNotFoundError):
    """An optional package a feature needs is not installed; the message names the extra to install."""


def require(module: str, extra: str, feature: str) -> ModuleType:
    """Return ``module``, importing it; say which extra installs it if it is missing.

    Only ``module`` itself missing, or a package it sits in, is reported that way. Any other
    missing module -- one it needs, or a part of an installed package that fails to import -- is
    re-raised unchanged, since installing the extra would not help.

    Args:
        module: Dotted module name, such as ``"h5py"``.
        extra: The ``gwmock`` extra that installs it, such as ``"hdf5"``.
        feature: What needs it, for the message, such as ``"HDF5 output"``.

    Raises:
        MissingExtraError: If the package is not installed. It is a ``ModuleNotFoundError``.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        parts = module.split(".")
        enclosing = {".".join(parts[: index + 1]) for index in range(len(parts))}
        if exc.name not in enclosing:
            raise
        root = parts[0]
        raise MissingExtraError(
            f"{feature} needs {root}, which is not installed. Install it with: pip install 'gwmock[{extra}]'",
            name=exc.name,
        ) from exc
