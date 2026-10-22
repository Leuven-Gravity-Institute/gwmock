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

from gwmock.signal.population_columns import (
    DIRECTION_BAND,
    orbit_aligned_angular_momentum_side,
    require_redundant_columns_agree,
)


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


#: An equal-mass binary without spin: J is L, so theta_jn must be the inclination.
_NO_SPIN = {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 30.0, "spin_1z": 0.0, "spin_2z": 0.0}


def _heavy_primary(chi1z: float) -> dict:
    """100 + 5 Msun at 5 Hz: |L| ~ 2.5e3 Msun**2 against S_z = 1e4 chi1z Msun**2."""
    return {"detector_frame_mass_1": 100.0, "detector_frame_mass_2": 5.0, "spin_1z": chi1z, "spin_2z": 0.0}


class TestThetaJN:
    def test_without_the_inclination_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="theta_jn is given without inclination"):
            _check({"theta_jn": 0.4})

    def test_without_spin_it_must_equal_the_inclination(self) -> None:
        _check({**_NO_SPIN, "theta_jn": 0.5, "inclination": 0.5})

    def test_without_spin_pi_minus_the_inclination_is_refused(self) -> None:
        """With no spin J is L: the anti-parallel value would describe a different orientation."""
        with pytest.raises(ValueError, match="J is parallel to L"):
            _check({**_NO_SPIN, "theta_jn": math.pi - 0.5, "inclination": 0.5})

    def test_an_unrelated_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not inclination"):
            _check({**_NO_SPIN, "theta_jn": 0.4, "inclination": 0.5})

    def test_clearly_anti_parallel_it_must_be_pi_minus_the_inclination(self) -> None:
        """A strongly anti-aligned heavy primary outweighs L: J points the other way."""
        event = {**_heavy_primary(-0.9), "inclination": 0.5}
        _check({**event, "theta_jn": math.pi - 0.5}, reference_frequency=5.0)
        with pytest.raises(ValueError, match="J is anti-parallel"):
            _check({**event, "theta_jn": 0.5}, reference_frequency=5.0)

    def test_where_the_sign_is_not_resolved_either_value_is_accepted(self) -> None:
        event = {**_heavy_primary(-0.25), "inclination": 0.5}
        _check({**event, "theta_jn": 0.5}, reference_frequency=5.0)
        _check({**event, "theta_jn": math.pi - 0.5}, reference_frequency=5.0)
        with pytest.raises(ValueError, match="is neither inclination"):
            _check({**event, "theta_jn": math.pi - 0.5 + 1e-5}, reference_frequency=5.0)

    def test_without_the_masses_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="theta_jn is given without detector_frame_mass_1"):
            _check({"spin_1z": 0.0, "spin_2z": 0.0, "theta_jn": 0.5, "inclination": 0.5})

    def test_without_the_aligned_spins_it_is_refused(self) -> None:
        event = {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 30.0, "theta_jn": 0.5, "inclination": 0.5}
        with pytest.raises(ValueError, match="without spin_1z, spin_2z"):
            _check(event)

    def test_without_a_reference_frequency_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reference frequency"):
            _check({**_NO_SPIN, "theta_jn": 0.5, "inclination": 0.5}, reference_frequency=None)

    def test_with_in_plane_spin_it_may_differ(self) -> None:
        _check({"theta_jn": 0.4, "inclination": 0.5, **_SPIN})


class TestTheDirectionOfJ:
    def test_without_spin_j_is_parallel(self) -> None:
        assert orbit_aligned_angular_momentum_side((30.0, 30.0), (0.0, 0.0), 20.0) == 1

    def test_the_band_is_relative_to_l(self) -> None:
        """J_z just outside the band is resolved; just inside, it is not."""
        m1, m2, f = 100.0, 5.0, 5.0
        total = m1 + m2
        v = (math.pi * total * 4.925490947641267e-06 * f) ** (1 / 3)
        orbital = m1 * m2 / v
        for fraction, side in ((DIRECTION_BAND * 0.99, 0), (DIRECTION_BAND * 1.01, 1)):
            chi = (fraction - 1.0) * orbital / m1**2
            assert orbit_aligned_angular_momentum_side((m1, m2), (chi, 0.0), f) == side

    def test_past_the_innermost_stable_orbit_it_is_not_decided(self) -> None:
        with pytest.raises(ValueError, match="innermost stable circular orbit"):
            orbit_aligned_angular_momentum_side((500.0, 500.0), (0.0, 0.0), 20.0)


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
