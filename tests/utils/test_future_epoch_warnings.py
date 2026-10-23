"""Dates past astropy's tables must not flood a run's log, and nothing else may be hidden.

The examples are placed in 2030, past the leap-second table and the measured Earth-orientation
data, so every UTC or UT1 conversion warned -- about twenty lines per run, drowning the warnings a
user has to act on.
"""

from __future__ import annotations

import warnings

from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from erfa import ErfaWarning
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.utils.log import ignore_future_epoch_warnings

_GPS_2030 = 1577491218.0


def _convert_a_2030_time() -> None:
    time = Time(_GPS_2030, format="gps")
    _ = time.utc.iso
    _ = time.sidereal_time("mean", longitude=0)


def _recorded(function) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function()
    return caught


def test_a_2030_conversion_warns_without_the_filter():
    """Guards the test below: if astropy stopped warning, it would pass for the wrong reason."""
    assert any(issubclass(w.category, ErfaWarning) for w in _recorded(_convert_a_2030_time))


def test_the_filter_silences_the_future_epoch_warnings():
    def run():
        ignore_future_epoch_warnings()
        _convert_a_2030_time()

    caught = _recorded(run)

    assert not [w for w in caught if issubclass(w.category, (ErfaWarning, AstropyWarning))], [
        str(w.message) for w in caught
    ]


def test_other_astropy_and_erfa_warnings_stay_visible():
    def run():
        ignore_future_epoch_warnings()
        warnings.warn('ERFA function "gd2gc" yielded 1 of "illegal case"', ErfaWarning, stacklevel=1)
        warnings.warn("something astropy wants you to know", AstropyWarning, stacklevel=1)

    messages = {str(w.message) for w in _recorded(run)}

    assert 'ERFA function "gd2gc" yielded 1 of "illegal case"' in messages
    assert "something astropy wants you to know" in messages


def test_the_command_line_installs_the_filter():
    def run():
        result = CliRunner().invoke(app, ["config", "--list"])
        assert result.exit_code == 0, result.output
        _convert_a_2030_time()

    assert not [w for w in _recorded(run) if issubclass(w.category, ErfaWarning)]
