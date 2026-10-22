"""Which population columns may be left out of a waveform call, and which must stop a run.

A population catalogue carries more than the waveform reads: a redshift, source-frame masses, a
class label, an index. Those are left out of the waveform call quietly -- they are still recorded in
the injection parameters. Any *other* column the waveform does not read stops the run before
anything is generated: a misspelt spin (``spin1_z`` for ``spin_1z``) or a column meant as a waveform
setting would otherwise vanish, and the run would generate a different signal than the catalogue
describes while reporting success. A user who knows a column is not a waveform input declares it
under ``signal.ignored-population-columns``.

This module is the single source of truth for that decision, for the per-event path (where the
waveform backend names the columns it rejects) and the batched path (which reads a fixed set and
ignores the rest).

gwmock-pop exposes the names it writes as parameters (``gwmock_pop.constants.CBC_PARAMETER_NAMES``)
but no list of which of them are population-only, so the classification below is gwmock's own.
``tests/signal/test_population_columns.py`` checks it against what gwmock-pop's packaged presets
and catalogue writers actually produce.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from gwmock.utils.errors import DeterministicRefusalError

logger = logging.getLogger("gwmock")

#: The configuration key a user declares extra columns under, named in every refusal.
IGNORED_COLUMNS_OPTION = "signal.ignored-population-columns"

#: Columns a population carries that no waveform reads and that nothing the waveform reads
#: duplicates, each with the reason. Left out of the waveform call without a check.
POPULATION_ONLY_COLUMNS: Mapping[str, str] = {
    # Cosmology: the waveform takes the detector-frame masses and the luminosity distance, which
    # already include it.
    "redshift": "cosmological redshift; already folded into detector-frame masses and distance",
    # Labels and indices.
    "event_id": "catalogue index",
    "source_type": "class label",
    "source_class": "class label",
    "class": "class label",
}

#: Columns that restate, in another form, something the waveform reads -- or a setting it takes
#: once per run. Each is left out only when, for the event at hand, the waveform's own copy is
#: present and agrees; see :func:`require_redundant_columns_agree` for the rules and tolerances.
#: Otherwise dropping it would change the signal: a catalogue carrying spins only as magnitude and
#: tilt would generate non-spinning binaries, one carrying only ``theta_jn`` would generate at the
#: default inclination.
REDUNDANT_COLUMNS: Mapping[str, str] = {
    "source_frame_mass_1": "source-frame mass; needs the detector-frame mass, = m_src (1 + z)",
    "source_frame_mass_2": "source-frame mass; needs the detector-frame mass, = m_src (1 + z)",
    "spin_1": "spin magnitude; needs the Cartesian components it is the norm of",
    "spin_2": "spin magnitude; needs the Cartesian components it is the norm of",
    "tilt_1": "spin tilt; needs the Cartesian components it is the polar angle of",
    "tilt_2": "spin tilt; needs the Cartesian components it is the polar angle of",
    "theta_jn": "orientation of J; needs the inclination, and without in-plane spin the side J is on",
    "f_ref": "reference frequency of the spins; must equal the one the waveform backend uses",
}

#: Orbital parameters every gwmock-pop CBC draw carries, which the quasi-circular waveforms gwmock
#: generates with do not read. Leaving one out is exact only where its value is zero, the
#: quasi-circular value; a nonzero value stops the run, because the waveform would not apply it.
ZERO_ONLY_COLUMNS: frozenset[str] = frozenset({"eccentricity", "long_asc_node", "mean_per_ano"})

#: Every column that may be left out at all, under its condition.
KNOWN_COLUMNS: frozenset[str] = frozenset(POPULATION_ONLY_COLUMNS) | frozenset(REDUNDANT_COLUMNS) | ZERO_ONLY_COLUMNS

#: Absolute tolerance on a spin magnitude and on an angle, in radians: well above the rounding of a
#: catalogue written to text, well below any physical difference.
ABSOLUTE_TOLERANCE = 1e-6
#: Relative tolerance on a mass and on a frequency.
RELATIVE_TOLERANCE = 1e-6

#: The waveform's own name for a quantity, then the aliases the backends accept for it.
_NAMES: Mapping[str, tuple[str, ...]] = {
    "detector_frame_mass_1": ("detector_frame_mass_1", "mass1"),
    "detector_frame_mass_2": ("detector_frame_mass_2", "mass2"),
    **{f"spin_{k}{c}": (f"spin_{k}{c}", f"spin{k}{c}") for k in (1, 2) for c in "xyz"},
    "inclination": ("inclination",),
    "redshift": ("redshift",),
}


def unread_columns_to_refuse(
    columns: Collection[str],
    ignored: Collection[str],
) -> list[str]:
    """Return the columns in *columns* that are not known and not declared ignored.

    Args:
        columns: Population columns the waveform does not read.
        ignored: Columns the configuration declares ignored.

    Returns:
        The columns that must stop the run, sorted.
    """
    return sorted(set(columns) - KNOWN_COLUMNS - set(ignored))


def require_population_columns_read(
    columns: Collection[str],
    *,
    ignored: Collection[str],
    reader: str,
    events: Collection[Mapping[str, Any]] = (),
    reference_frequency: float | None = None,
) -> None:
    """Raise unless every population column the waveform does not read may be left out.

    Args:
        columns: Population columns *reader* does not read.
        ignored: Columns the configuration declares ignored.
        reader: What rejected them, for the message -- for example ``"the LAL waveform backend"``.
        events: Events whose conditional columns are checked; may be empty.
        reference_frequency: The reference frequency the waveform backend uses for this run, in Hz.

    Raises:
        ValueError: If a column is neither known nor declared ignored, or a known column's
            condition fails for any of *events*.
    """
    refused = unread_columns_to_refuse(columns, ignored)
    if refused:
        raise DeterministicRefusalError(
            f"Population column(s) {', '.join(refused)} are not read by {reader}, so they would be "
            "dropped and the run would generate a waveform other than the catalogue describes. Check "
            "the spelling against the waveform's parameter names (for example spin_1z, not spin1_z). "
            f"If a column is not meant as a waveform input, list it under {IGNORED_COLUMNS_OPTION}."
        )
    require_redundant_columns_agree(
        columns, ignored=ignored, reader=reader, events=events, reference_frequency=reference_frequency
    )
    quiet = sorted(set(columns))
    if quiet:
        logger.debug("Population columns left out of the waveform call by %s: %s", reader, ", ".join(quiet))


def require_redundant_columns_agree(  # noqa: PLR0912
    columns: Collection[str],
    *,
    ignored: Collection[str],
    reader: str,
    events: Collection[Mapping[str, Any]],
    reference_frequency: float | None,
) -> None:
    """Raise if, for any event, a conditional column left out by *reader* may not be left out.

    The rules, each per event and each skipped for a column declared ignored:

    - ``eccentricity``, ``long_asc_node``, ``mean_per_ano``: zero.
    - ``source_frame_mass_k``: the detector-frame mass is present, and with ``redshift`` present
      ``m_det = m_src (1 + z)`` to a relative 1e-6.
    - ``spin_k``, ``tilt_k``: all three Cartesian components are present, the magnitude is their
      norm and the tilt is ``arccos(z / |chi|)``, each to an absolute 1e-6 (the tilt only where
      ``|chi| > 1e-6``, since it is undefined at zero spin).
    - ``theta_jn``: the inclination is present. Where both bodies' in-plane spin components are
      zero (or absent) the binary has no precession and J lies along L, on the side
      :func:`orbit_aligned_angular_momentum_side` decides from the event's masses and aligned
      spins: ``theta_jn`` must equal the inclination when J is parallel to L, pi minus it when
      anti-parallel, and either within the band where the sign is not resolved -- each to 1e-6 rad.
      An event lacking a mass, an aligned spin or a reference frequency to decide that is refused.
      With in-plane spin the two differ in general, so nothing more is required.
    - ``f_ref``: equal, to a relative 1e-6, to *reference_frequency* -- the frequency the waveform
      backend references the spins to for this run.

    These run for every event but log nothing, so a long catalogue costs no log lines.

    Args:
        columns: Population columns *reader* does not read.
        ignored: Columns the configuration declares ignored; those are not checked.
        reader: What does not read them, for the message.
        events: The events to check.
        reference_frequency: The backend's reference frequency in Hz, or ``None`` when unknown,
            which refuses a per-event ``f_ref``.

    Raises:
        ValueError: On the first event, and column, whose condition fails.
    """
    checked = (set(columns) & (set(REDUNDANT_COLUMNS) | ZERO_ONLY_COLUMNS)) - set(ignored)
    if not checked:
        return
    for event in events:
        problems: list[str] = []

        def value(name: str, event: Mapping[str, Any] = event) -> float | None:
            for alias in _NAMES.get(name, (name,)):
                if alias in event and event[alias] is not None:
                    return float(np.asarray(event[alias], dtype=float))
            return None

        for name in sorted(checked & ZERO_ONLY_COLUMNS):
            if name in event and value(name) not in (None, 0.0):
                problems.append(
                    f"{name} is {value(name)!r}, and {reader} generates a quasi-circular waveform that "
                    "would not apply it; use a waveform that reads it, or set it to zero"
                )
        for k in (1, 2):
            source = f"source_frame_mass_{k}"
            if source in checked and source in event:
                detector, z = value(f"detector_frame_mass_{k}"), value("redshift")
                if detector is None:
                    problems.append(f"{source} is given without detector_frame_mass_{k}, the mass the waveform reads")
                elif z is not None and not np.isclose(
                    detector, value(source) * (1.0 + z), rtol=RELATIVE_TOLERANCE, atol=0.0
                ):
                    problems.append(
                        f"detector_frame_mass_{k} = {detector!r} is not {source} (1 + redshift) = "
                        f"{value(source) * (1.0 + z)!r}"
                    )
            spin_columns = [name for name in (f"spin_{k}", f"tilt_{k}") if name in checked and name in event]
            if spin_columns:
                components = [value(f"spin_{k}{c}") for c in "xyz"]
                if any(component is None for component in components):
                    problems.append(
                        f"{', '.join(spin_columns)} are given without all of spin_{k}x, spin_{k}y, spin_{k}z, "
                        "the components the waveform reads, so the spin would be dropped"
                    )
                else:
                    magnitude = float(np.linalg.norm(components))
                    if f"spin_{k}" in spin_columns and abs(magnitude - value(f"spin_{k}")) > ABSOLUTE_TOLERANCE:
                        problems.append(
                            f"spin_{k} = {value(f'spin_{k}')!r} is not the norm of its components, {magnitude!r}"
                        )
                    if f"tilt_{k}" in spin_columns and magnitude > ABSOLUTE_TOLERANCE:
                        tilt = float(np.arccos(np.clip(components[2] / magnitude, -1.0, 1.0)))
                        if abs(tilt - value(f"tilt_{k}")) > ABSOLUTE_TOLERANCE:
                            problems.append(
                                f"tilt_{k} = {value(f'tilt_{k}')!r} is not the tilt of its components, {tilt!r}"
                            )
        if "theta_jn" in checked and "theta_jn" in event:
            inclination = value("inclination")
            in_plane = [value(f"spin_{k}{c}") for k in (1, 2) for c in "xy"]
            if inclination is None:
                problems.append("theta_jn is given without inclination, so the inclination would default")
            elif all(component in (None, 0.0) for component in in_plane):
                problems.extend(
                    _theta_jn_problems(
                        value("theta_jn"),
                        inclination,
                        masses=(value("detector_frame_mass_1"), value("detector_frame_mass_2")),
                        aligned_spins=(value("spin_1z"), value("spin_2z")),
                        reference_frequency=reference_frequency,
                    )
                )
        if "f_ref" in checked and "f_ref" in event:
            if reference_frequency is None:
                problems.append("f_ref is given per event, and the waveform backend's reference frequency is unknown")
            elif not np.isclose(value("f_ref"), reference_frequency, rtol=RELATIVE_TOLERANCE, atol=0.0):
                problems.append(
                    f"f_ref is {value('f_ref')!r} Hz, but the waveform backend references the spins to "
                    f"{reference_frequency!r} Hz; set f_ref under signal.waveform-backend-arguments to the "
                    "catalogue's value"
                )
        if problems:
            raise DeterministicRefusalError(
                f"A population column {reader} does not read cannot be left out of the event with "
                f"coa_time={event.get('coa_time')!r}: {'; '.join(problems)}. To leave such a column out "
                f"deliberately, list it under {IGNORED_COLUMNS_OPTION}."
            )


#: Solar mass in seconds, G M_sun / c^3 (IAU 2015 nominal, as LAL uses).
_SOLAR_MASS_SECONDS = 4.925490947641267e-06

#: Orbital velocity at the innermost stable circular orbit of a test mass, ``v = 6**-0.5``; past it
#: the post-Newtonian orbital angular momentum below does not describe a binary.
_ISCO_VELOCITY = 6.0**-0.5

#: Half-width, relative to the Newtonian orbital angular momentum, of the band where the sign of
#: ``J_z = L + S_z`` is not resolved.
#:
#: **Measured.** The relative difference between the 2PN and the Newtonian orbital angular
#: momentum of a circular non-spinning binary, ``(3/2 + nu/6) x + (27/8 - 19 nu/8 + nu**2/24) x**2``
#: with ``x = v**2``, grows with ``v`` and is largest at the edge of the range checked here,
#: ``v = v_ISCO``: 0.344 for ``nu -> 0`` and 0.334 for ``nu = 1/4``. At the reference frequencies
#: typical of ground-based detectors it is far smaller -- 0.006 for a 3 Msun binary at 5 Hz, 0.12 for
#: 60 Msun at 20 Hz. 0.5 leaves margin over the largest value. **Not anchored**: the spin-orbit
#: (1.5PN and higher) and spin-spin corrections to L are not included in that measurement; the
#: margin is meant to absorb them but is not shown to.
DIRECTION_BAND = 0.5


def orbit_aligned_angular_momentum_side(
    masses: tuple[float, float],
    aligned_spins: tuple[float, float],
    reference_frequency: float,
) -> int:
    """Return on which side of the orbital angular momentum L the total J lies, for an aligned-spin binary.

    With no in-plane spin, J = L + S lies along L: ``J_z = |L| + S_z`` with
    ``S_z = m1**2 chi1z + m2**2 chi2z``. The orbital angular momentum is taken at Newtonian order,
    ``|L| = nu M**2 / v`` with ``v = (pi M f_ref)**(1/3)`` in geometric units and ``M`` the
    detector-frame total mass. Post-Newtonian corrections change ``|L|``, so a ``J_z`` within
    :data:`DIRECTION_BAND` of zero, relative to ``|L|``, is reported as unresolved.

    The post-Newtonian expression the band is measured against is the circular-orbit angular
    momentum of a non-spinning binary to 2PN, ``J = (G m**2 nu / c x**0.5) [1 + (3/2 + nu/6) x +
    (27/8 - 19 nu/8 + nu**2/24) x**2]`` (L. Blanchet, Living Rev. Relativ. 17, 2 (2014)); its
    coefficients were checked against the 2PN binding energy through the first law of binary
    mechanics, ``dE = omega dJ``, which they satisfy exactly to that order.

    Args:
        masses: The detector-frame component masses, in solar masses.
        aligned_spins: The dimensionless spin components along L, ``chi1z`` and ``chi2z``.
        reference_frequency: The gravitational-wave frequency the spins are referenced to, in Hz.

    Returns:
        ``1`` when J is parallel to L, ``-1`` when anti-parallel, ``0`` when unresolved.

    Raises:
        ValueError: If the reference frequency puts the orbit past the innermost stable circular
            orbit, where the orbital angular momentum above does not apply.
    """
    m1, m2 = masses
    total = m1 + m2
    velocity = (np.pi * total * _SOLAR_MASS_SECONDS * reference_frequency) ** (1.0 / 3.0)
    if velocity >= _ISCO_VELOCITY:
        raise ValueError(
            f"at the reference frequency {reference_frequency!r} Hz the orbit of a {total!r} Msun binary is past "
            "the innermost stable circular orbit"
        )
    orbital = (m1 * m2 / total**2) * total**2 / velocity
    total_z = orbital + m1**2 * aligned_spins[0] + m2**2 * aligned_spins[1]
    if abs(total_z) <= DIRECTION_BAND * orbital:
        return 0
    return 1 if total_z > 0 else -1


def _theta_jn_problems(
    theta_jn: float,
    inclination: float,
    *,
    masses: tuple[float | None, float | None],
    aligned_spins: tuple[float | None, float | None],
    reference_frequency: float | None,
) -> list[str]:
    """Return what is wrong with leaving out *theta_jn* for an event without in-plane spin."""
    missing = [
        name
        for name, given in (
            ("detector_frame_mass_1", masses[0]),
            ("detector_frame_mass_2", masses[1]),
            ("spin_1z", aligned_spins[0]),
            ("spin_2z", aligned_spins[1]),
        )
        if given is None
    ]
    if missing or reference_frequency is None:
        needed = ", ".join(missing) if missing else "the waveform backend's reference frequency"
        return [f"theta_jn is given without {needed}, which decide whether it must equal the inclination"]
    try:
        side = orbit_aligned_angular_momentum_side(
            (float(masses[0]), float(masses[1])),  # type: ignore[arg-type]
            (float(aligned_spins[0]), float(aligned_spins[1])),  # type: ignore[arg-type]
            reference_frequency,
        )
    except ValueError as error:
        return [f"theta_jn cannot be checked: {error}"]
    parallel = abs(theta_jn - inclination) <= ABSOLUTE_TOLERANCE
    anti_parallel = abs(theta_jn - (np.pi - inclination)) <= ABSOLUTE_TOLERANCE
    if side > 0 and not parallel:
        return [
            f"theta_jn = {theta_jn!r} is not inclination = {inclination!r}, although J is parallel to L for this "
            "event's masses and aligned spins"
        ]
    if side < 0 and not anti_parallel:
        return [
            f"theta_jn = {theta_jn!r} is not pi - inclination = {np.pi - inclination!r}, although the anti-aligned "
            "spin outweighs L and J is anti-parallel to it for this event"
        ]
    if side == 0 and not (parallel or anti_parallel):
        return [f"theta_jn = {theta_jn!r} is neither inclination = {inclination!r} nor pi minus it"]
    return []
