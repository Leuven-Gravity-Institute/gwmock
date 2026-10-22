"""A column restating something the waveform reads is left out only when the waveform's copy agrees.

Dropping such a column is exact only when the waveform reads the same information from another
column -- otherwise the signal changes: spins given only as magnitude and tilt would generate a
non-spinning binary, ``theta_jn`` without ``inclination`` would generate at the default inclination,
and spins defined at one reference frequency would be generated at another.
"""

from __future__ import annotations

import logging
import math

import pytest

from gwmock.signal.population_columns import require_redundant_columns_agree


def _check(event: dict, *, reference_frequency: float | None = 20.0, ignored=()) -> None:
    columns = {
        name
        for name in event
        if name
        in {
            "spin_1",
            "spin_2",
            "tilt_1",
            "tilt_2",
            "theta_jn",
            "f_ref",
            "source_frame_mass_1",
            "source_frame_mass_2",
            "eccentricity",
        }
    }
    require_redundant_columns_agree(
        columns, ignored=ignored, reader="the test backend", events=[event], reference_frequency=reference_frequency
    )


_SPIN = {"spin_1x": 0.3, "spin_1y": 0.0, "spin_1z": 0.4}  # |chi| = 0.5, tilt = arccos(0.8)


class TestSpinMagnitudeAndTilt:
    def test_without_the_components_they_are_refused(self) -> None:
        with pytest.raises(ValueError, match="spin_1, tilt_1 are given without all of spin_1x"):
            _check({"spin_1": 0.5, "tilt_1": 0.6})

    def test_consistent_with_the_components_they_are_left_out(self) -> None:
        _check({**_SPIN, "spin_1": 0.5, "tilt_1": math.acos(0.8)})

    def test_the_backend_aliases_count_as_the_components(self) -> None:
        _check({"spin1x": 0.3, "spin1y": 0.0, "spin1z": 0.4, "spin_1": 0.5})

    def test_a_magnitude_that_is_not_the_norm_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"spin_1 = 0\.6 is not the norm"):
            _check({**_SPIN, "spin_1": 0.6})

    def test_a_tilt_that_is_not_the_tilt_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"tilt_1 = 0\.1 is not the tilt"):
            _check({**_SPIN, "tilt_1": 0.1})

    def test_a_tilt_at_zero_spin_is_not_checked(self) -> None:
        _check({"spin_1x": 0.0, "spin_1y": 0.0, "spin_1z": 0.0, "spin_1": 0.0, "tilt_1": 1.0})


class TestThetaJN:
    def test_without_the_inclination_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="theta_jn is given without inclination"):
            _check({"theta_jn": 0.4})

    def test_without_in_plane_spin_it_must_equal_the_inclination(self) -> None:
        with pytest.raises(ValueError, match="differs from inclination"):
            _check({"theta_jn": 0.4, "inclination": 0.5, "spin_1z": 0.3})

    def test_equal_to_the_inclination_it_is_left_out(self) -> None:
        _check({"theta_jn": 0.5, "inclination": 0.5})

    def test_with_in_plane_spin_it_may_differ(self) -> None:
        _check({"theta_jn": 0.4, "inclination": 0.5, **_SPIN})


class TestReferenceFrequency:
    def test_one_the_backend_does_not_use_is_refused_naming_both(self) -> None:
        with pytest.raises(ValueError, match=r"f_ref is 20\.0 Hz.*5\.0 Hz.*waveform-backend-arguments"):
            _check({"f_ref": 20.0}, reference_frequency=5.0)

    def test_the_one_the_backend_uses_is_left_out(self) -> None:
        _check({"f_ref": 20.0}, reference_frequency=20.0)


class TestSourceFrameMasses:
    def test_without_the_detector_frame_mass_they_are_refused(self) -> None:
        with pytest.raises(ValueError, match="source_frame_mass_1 is given without detector_frame_mass_1"):
            _check({"source_frame_mass_1": 30.0, "redshift": 0.1})

    def test_they_must_agree_through_the_redshift(self) -> None:
        with pytest.raises(ValueError, match="is not source_frame_mass_1"):
            _check({"source_frame_mass_1": 30.0, "detector_frame_mass_1": 30.0, "redshift": 0.1})

    def test_consistent_they_are_left_out(self) -> None:
        _check({"source_frame_mass_1": 30.0, "detector_frame_mass_1": 33.0, "redshift": 0.1})

    def test_the_mass1_alias_counts(self) -> None:
        _check({"source_frame_mass_1": 30.0, "mass1": 33.0, "redshift": 0.1})


def test_a_declared_column_is_not_checked() -> None:
    _check({"theta_jn": 0.4}, ignored=("theta_jn",))


def test_checking_many_events_logs_nothing_per_event(caplog) -> None:
    events = [{**_SPIN, "spin_1": 0.5, "coa_time": float(i)} for i in range(50)]
    with caplog.at_level(logging.DEBUG, logger="gwmock"):
        require_redundant_columns_agree(
            {"spin_1"}, ignored=(), reader="the test backend", events=events, reference_frequency=None
        )
    assert caplog.records == []
