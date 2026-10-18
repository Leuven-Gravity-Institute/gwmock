"""Utility functions."""

from __future__ import annotations

# Python 3.15 (PEP 810): `import gwmock` loads this package to set up logging, and this keeps that from
# also importing NumPy, which only the random helpers need. Earlier versions ignore it.
__lazy_modules__ = ["gwmock.utils.random"]

from .random import generate_seeds, get_rng, set_seed

__all__ = [
    "generate_seeds",
    "get_rng",
    "set_seed",
]
