"""The waveform library gwmock names when a config gives backend arguments but no backend."""

from __future__ import annotations

import logging

import pytest
from gwmock_signal.waveform import factory
from gwmock_signal.waveform.backends import RippleBackend

from gwmock.cli import adapter_orchestration

pytestmark = pytest.mark.unit


class _StubRipple(RippleBackend):
    """A ripple backend that needs no JAX to construct."""

    def __init__(self, **kwargs):
        pass


class _StubLAL:
    """Stands in for the LAL backend, which needs lalsuite to construct."""


def test_where_lalsuite_is_installed_it_is_lal(monkeypatch, caplog):
    """gwmock-signal's default is LAL there, so gwmock names LAL, and nothing is logged."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: True)
    monkeypatch.setattr(factory, "LALSimulationBackend", _StubLAL)

    with caplog.at_level(logging.WARNING):
        assert adapter_orchestration._default_waveform_backend_name() == "lal"
    assert caplog.records == []


def test_where_lalsuite_is_not_it_is_ripple_with_gwmock_signals_warning(monkeypatch, caplog):
    """The fallback is gwmock-signal's, warning included, not a copy of its rule."""
    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", _StubRipple)

    with caplog.at_level(logging.WARNING, logger="gwmock_signal.waveform"):
        assert adapter_orchestration._default_waveform_backend_name() == "ripple"
    assert "ripple (JAX) backend rather than the default LAL one" in caplog.text


def test_without_either_the_error_names_both_remedies(monkeypatch):
    """The run stops with gwmock-signal's combined message, as it does when no arguments are given."""

    def unavailable(**kwargs):
        raise ImportError("ripple (rippleGW) is not installed. Run: pip install 'gwmock-signal[jax]'")

    monkeypatch.setattr(factory, "_lalsuite_installed", lambda: False)
    monkeypatch.setattr(factory, "RippleBackend", unavailable)

    with pytest.raises(ImportError, match="requires lalsuite") as raised:
        adapter_orchestration._default_waveform_backend_name()
    assert "pip install 'gwmock-signal[jax]'" in str(raised.value)
