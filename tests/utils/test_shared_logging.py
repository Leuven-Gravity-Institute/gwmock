"""Messages from the gwmock-data code gwmock re-exports reach gwmock's own log output."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gwmock.cli.main import LoggingLevel, setup_logging
from gwmock.data import TimeSeries
from gwmock.utils.log import setup_logger


@pytest.fixture
def clean_loggers():
    """Restore the gwmock and gwmock-data loggers after each test."""
    saved = {}
    for name in ("gwmock", "gwmock_data"):
        log = logging.getLogger(name)
        saved[name] = (log.level, log.propagate, list(log.handlers))
    yield
    for name, (level, propagate, handlers) in saved.items():
        log = logging.getLogger(name)
        for handler in log.handlers:
            if handler not in handlers:
                handler.close()
        log.setLevel(level)
        log.propagate = propagate
        log.handlers = handlers


class _Collect(logging.Handler):
    """Keeps every record it is handed."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def test_a_series_warning_reaches_gwmock_handlers(clean_loggers):
    """A warning raised inside the shared series is handled by whatever gwmock's logger writes to."""
    collected = _Collect()
    logging.getLogger("gwmock").addHandler(collected)
    setup_logger(log_level="INFO")
    series = TimeSeries(np.zeros((1, 8)), start_time=0.0, sampling_frequency=4.0)

    # A channel at another rate is refused, after a warning logged by the shared series.
    with pytest.raises(ValueError, match="Sampling frequency"):
        series[0] = type(series[0])(np.zeros(8), t0=0.0, sample_rate=8.0)

    assert [record.getMessage() for record in collected.records if record.name == "gwmock_data"] == [
        "Sampling frequency mismatch on channel 0. Difference: 4.000000000000000e+00 Hz."
    ]


def test_the_command_line_setup_covers_the_shared_logger(clean_loggers):
    """The CLI's verbosity and console handler apply to the shared logger too."""
    setup_logging(LoggingLevel.DEBUG)

    gwmock_logger, shared = logging.getLogger("gwmock"), logging.getLogger("gwmock_data")
    assert shared.level == logging.DEBUG
    assert shared.handlers == gwmock_logger.handlers
    assert shared.propagate is False
