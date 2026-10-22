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

logger = logging.getLogger("gwmock")

#: The configuration key a user declares extra columns under, named in every refusal.
IGNORED_COLUMNS_OPTION = "signal.ignored-population-columns"

#: Columns a population carries that no waveform reads, each with the reason. Left out of the
#: waveform call without a warning.
POPULATION_ONLY_COLUMNS: Mapping[str, str] = {
    # Cosmology: the waveform takes the detector-frame masses and the luminosity distance, which
    # already include it.
    "redshift": "cosmological redshift; already folded into detector-frame masses and distance",
    "source_frame_mass_1": "source-frame mass; the waveform takes the detector-frame mass",
    "source_frame_mass_2": "source-frame mass; the waveform takes the detector-frame mass",
    # Spin magnitude and tilt, which gwmock-pop's MDC lists write beside the Cartesian components
    # the waveform reads.
    "spin_1": "spin magnitude; the waveform reads the Cartesian components",
    "spin_2": "spin magnitude; the waveform reads the Cartesian components",
    "tilt_1": "spin tilt; the waveform reads the Cartesian components",
    "tilt_2": "spin tilt; the waveform reads the Cartesian components",
    # gwmock-pop draws record the angle between total angular momentum and line of sight beside
    # the inclination the waveform reads.
    "theta_jn": "orientation recorded beside the inclination the waveform reads",
    # gwmock-pop draws record the reference frequency their spins refer to. A waveform backend takes
    # its reference frequency from signal.waveform-backend-arguments, never per event.
    "f_ref": "per-event reference frequency; backends take f_ref from waveform-backend-arguments",
    # Labels and indices.
    "event_id": "catalogue index",
    "source_type": "class label",
    "source_class": "class label",
    "class": "class label",
}

#: Orbital parameters every gwmock-pop CBC draw carries, which the quasi-circular waveforms gwmock
#: generates with do not read. Leaving one out is exact only where its value is zero, the
#: quasi-circular value; a nonzero value stops the run, because the waveform would not apply it.
ZERO_ONLY_COLUMNS: frozenset[str] = frozenset({"eccentricity", "long_asc_node", "mean_per_ano"})


def unread_columns_to_refuse(
    columns: Collection[str],
    ignored: Collection[str],
) -> list[str]:
    """Return the columns in *columns* that are neither population-only, zero-only nor declared.

    Args:
        columns: Population columns the waveform does not read.
        ignored: Columns the configuration declares ignored.

    Returns:
        The columns that must stop the run, sorted.
    """
    return sorted(set(columns) - set(POPULATION_ONLY_COLUMNS) - ZERO_ONLY_COLUMNS - set(ignored))


def require_population_columns_read(
    columns: Collection[str],
    *,
    ignored: Collection[str],
    reader: str,
    events: Collection[Mapping[str, Any]] = (),
) -> None:
    """Raise unless every population column the waveform does not read may be left out.

    Args:
        columns: Population columns *reader* does not read.
        ignored: Columns the configuration declares ignored.
        reader: What rejected them, for the message -- for example ``"the LAL waveform backend"``.
        events: Events whose zero-only columns are checked for a nonzero value; may be empty.

    Raises:
        ValueError: If a column is neither population-only nor declared ignored, or a zero-only
            column is nonzero in any of *events*.
    """
    refused = unread_columns_to_refuse(columns, ignored)
    if refused:
        raise ValueError(
            f"Population column(s) {', '.join(refused)} are not read by {reader}, so they would be "
            "dropped and the run would generate a waveform other than the catalogue describes. Check "
            "the spelling against the waveform's parameter names (for example spin_1z, not spin1_z). "
            f"If a column is not meant as a waveform input, list it under {IGNORED_COLUMNS_OPTION}."
        )
    require_zero_only_columns_zero(columns, ignored=ignored, reader=reader, events=events)
    quiet = sorted(set(columns))
    if quiet:
        logger.debug("Population columns left out of the waveform call by %s: %s", reader, ", ".join(quiet))


def require_zero_only_columns_zero(
    columns: Collection[str],
    *,
    ignored: Collection[str],
    reader: str,
    events: Collection[Mapping[str, Any]],
) -> None:
    """Raise if a zero-only column left out by *reader* is nonzero in any of *events*.

    Args:
        columns: Population columns *reader* does not read.
        ignored: Columns the configuration declares ignored; those are not checked.
        reader: What does not read them, for the message.
        events: The events to check.

    Raises:
        ValueError: If a zero-only, undeclared column has a nonzero value.
    """
    checked = (set(columns) & ZERO_ONLY_COLUMNS) - set(ignored)
    nonzero = sorted(name for name in checked if any(_is_nonzero(event.get(name)) for event in events if name in event))
    if nonzero:
        raise ValueError(
            f"Population column(s) {', '.join(nonzero)} are nonzero, and {reader} does not read them: it "
            "generates a quasi-circular waveform, so the eccentric orbit the catalogue describes would "
            "be dropped. Use a waveform that reads them, or set them to zero. To leave them out "
            f"deliberately, list them under {IGNORED_COLUMNS_OPTION}."
        )


def _is_nonzero(value: Any) -> bool:
    import numpy as np  # noqa: PLC0415

    if value is None:
        return False
    return bool(np.any(np.asarray(value, dtype=float) != 0.0))
