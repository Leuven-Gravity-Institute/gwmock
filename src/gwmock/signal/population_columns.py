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
    "theta_jn": "orientation of J; needs the inclination, and equals it without in-plane spin",
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
    - ``theta_jn``: the inclination is present; where both bodies' in-plane spin components are
      zero (or absent) the binary has no precession, J is parallel to L, and ``theta_jn`` must equal
      the inclination to 1e-6 rad. With in-plane spin the two differ in general, so nothing more
      is required.
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
    import numpy as np  # noqa: PLC0415

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
            elif all(component in (None, 0.0) for component in in_plane) and (
                abs(value("theta_jn") - inclination) > ABSOLUTE_TOLERANCE
            ):
                problems.append(
                    f"theta_jn = {value('theta_jn')!r} differs from inclination = {inclination!r}, although "
                    "without in-plane spin the binary has no precession and the two are the same angle"
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
