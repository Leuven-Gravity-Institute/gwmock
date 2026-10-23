"""Turn gwmock-signal's batched device output into gwmock chunks.

The batched entry point returns ``strain`` of shape ``(n_events, n_detectors, n_samples)`` --
per-event and *unsuperposed*, with placement left to the caller. gwmock already has an assembler for
exactly that: ``TimeSeries.inject_from_list`` places chunks by absolute time, caches whatever
overflows a segment, and carries it into the next one. So the device does generation and projection,
and gwmock keeps the incremental, checkpointed assembly it already owns.

That split is deliberate rather than a compromise. Generation was measured at 9.67e-3 s per second
of data against 3.66e-4 for assembly, so the expensive 96% moves to the device while the part that
owns spill-over, resumability and memory bounds stays where it works.

Peak host memory is one full copy of the batch. ``np.asarray(batch.strain)`` materialises
``(n_events, n_detectors, n_samples)`` at once, and gwmock-signal's ``chunk_size`` bounds the work
done per generation call without bounding the size of the result handed back. That is acceptable
here because the intended caller is ``_simulate``, which runs once per segment and so passes only
the events belonging to that segment -- but it is not a safe path for a whole catalogue in one call,
and nothing in this module enforces that. A caller batching more than one segment at a time needs
its own bound.

Placement is required to be **on the output lattice**. When ``simulate_cbc_batch`` is given an
``output_grid``, each buffer starts exactly on a sample of that grid, so injection is an
integer-offset add. Without it, buffers begin at an arbitrary time and gwmock has to interpolate
every chunk onto the segment grid -- which costs a resample per chunk and is, in gwmock-signal's own
words, "accurate only for heavily oversampled strain". This module refuses that case rather than
letting it degrade quietly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock.data.time_series.inject import alignment_tolerance, is_aligned
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.utils.errors import DeterministicRefusalError

#: Dimensions of a batched strain array: event, detector, sample.
_BATCHED_STRAIN_DIMENSIONS = 3

#: Alias to canonical parameter name, mirroring what the per-event backends accept.
#:
#: The per-event path resolves these inside gwmock-signal (``_pop_alias`` in each backend); the
#: batched entry point reads canonical names straight from the struct-of-arrays and does not. So a
#: population using ``distance`` -- as the bundled BBH catalogue does -- works per-event and fails
#: batched, which would make switching execution mode change whether a config runs at all. Taken
#: from gwmock-signal's backends rather than invented, so the two agree on what an alias is.
#:
#: Scope: this covers the aliases LAL and ripple share, plus PyCBC's ``lambda1``/``lambda2``. It is a
#: fourth copy of a mapping gwmock-signal already holds three times, one per backend, and the right
#: home for it is a canonicalisation helper there. Until that exists, a config using an alias no
#: backend shares would still be refused by the batched path with a missing-parameter error naming
#: the canonical name.
_PARAMETER_ALIASES: dict[str, str] = {
    "mass1": "detector_frame_mass_1",
    "mass2": "detector_frame_mass_2",
    "distance": "luminosity_distance",
    "tidal_1": "lambda_1",
    "tidal_2": "lambda_2",
    "lambda1": "lambda_1",
    "lambda2": "lambda_2",
    "spin1x": "spin_1x",
    "spin1y": "spin_1y",
    "spin1z": "spin_1z",
    "spin2x": "spin_2x",
    "spin2y": "spin_2y",
    "spin2z": "spin_2z",
}


#: Canonical parameters the batched path actually consumes.
#:
#: Projection reads the first four; ripple's waveform generation reads the rest. Anything else
#: handed to ``simulate_cbc_batch`` is ignored without complaint, which for a *fixed* argument the
#: user wrote in their config is a silent drop -- the third of that kind found in this work. Listed
#: here so it can be rejected instead.
#:
#: Taken from ``gwmock_signal.jax_batch`` and ``waveform.backends.ripple``. It will drift if
#: gwmock-signal grows a parameter; the cost of that is a spurious rejection, which is visible,
#: rather than a silent omission, which is not.
#:
#: The in-plane spins are here because ripple's batch resolver reads all six components for the
#: precessing approximants (``IMRPhenomPv2``, ``IMRPhenomXP``, ``IMRPhenomXPHM``). Omitting them
#: rejected every precessing configuration -- the opposite failure to a silent drop, and the reason
#: this list is checked against the library rather than written from memory.
#:
#: ``f_ref`` is deliberately *absent*. It is a backend option, not a per-event parameter:
#: gwmock-signal's own ``_RESERVED_WAVEFORM_ARGUMENTS`` says "f_ref is configured on the backend,
#: not through waveform_arguments". Passing it in this mapping does nothing at all. The
#: orchestrator forwards it to the ripple backend instead, so the configuration key still works.
BATCHED_PARAMETERS: frozenset[str] = frozenset(
    {
        "coa_time",
        "declination",
        "polarization_angle",
        "right_ascension",
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "luminosity_distance",
        "coa_phase",
        "inclination",
        "spin_1x",
        "spin_1y",
        "spin_1z",
        "spin_2x",
        "spin_2y",
        "spin_2z",
        "lambda_1",
        "lambda_2",
    }
)

#: Waveform arguments that configure the ripple *backend* rather than an individual event.
#:
#: They are accepted in ``waveform-arguments`` because that is where the per-event path takes them,
#: and a key that changes the waveform in one execution mode must not be inert in the other. The
#: orchestrator routes them to the backend constructor.
BATCHED_BACKEND_ARGUMENTS: frozenset[str] = frozenset({"f_ref"})


def require_batched_parameters_supported(waveform_arguments: dict[str, Any]) -> None:
    """Raise if a fixed waveform argument would be ignored by the batched path.

    Args:
        waveform_arguments: The ``waveform-arguments`` mapping from the configuration, already
            canonicalised.

    Raises:
        ValueError: If any key is not one the batched entry point reads.
    """
    unsupported = sorted(set(waveform_arguments) - BATCHED_PARAMETERS - BATCHED_BACKEND_ARGUMENTS)
    if unsupported:
        raise DeterministicRefusalError(
            f"execution: batched cannot apply these waveform-arguments: {unsupported}. The batched "
            f"entry point reads only {sorted(BATCHED_PARAMETERS | BATCHED_BACKEND_ARGUMENTS)}, and "
            f"would ignore the rest without complaint. Use execution: per-event, or remove them."
        )


def canonicalise_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return *parameters* with known aliases renamed to their canonical names.

    Args:
        parameters: Struct-of-arrays as the population loader produced it.

    Returns:
        A new mapping; *parameters* is not modified.

    Raises:
        ValueError: If both an alias and its canonical name are present with different values.
            Guessing which the caller meant would silently pick one physics over another.
    """
    renamed = dict(parameters)
    for alias, canonical in _PARAMETER_ALIASES.items():
        if alias not in renamed:
            continue
        if canonical in renamed and not np.array_equal(np.asarray(renamed[canonical]), np.asarray(renamed[alias])):
            raise DeterministicRefusalError(
                f"Parameters contain both '{canonical}' and its alias '{alias}' with different "
                f"values. Remove one; picking either would silently choose which physics to use."
            )
        renamed[canonical] = renamed.pop(alias)
    return renamed


if TYPE_CHECKING:  # pragma: no cover - typing only
    from gwmock_signal import BatchedDetectorStrain


def batched_strain_to_chunks(
    batch: BatchedDetectorStrain,
    *,
    expected_detector_names: tuple[str, ...] | None = None,
) -> TimeSeriesList:
    """Convert one batched device result into per-event chunks ready for injection.

    Args:
        batch: Result of ``gwmock_signal.simulate_cbc_batch``, generated with an ``output_grid``.
        expected_detector_names: Detector order the caller intends to write. Checked rather than
            assumed, because the chunk's channel *k* becomes detector *k* downstream and a
            reordering would silently attribute each signal to the wrong interferometer.

    Returns:
        One chunk per event, each of shape ``(n_detectors, n_samples)``, in event order.

    Raises:
        ValueError: If the batch was generated without an output grid, if any buffer does not start
            on that grid, or if the detector names disagree with *expected_detector_names*.
    """
    if batch.grid is None or batch.start_index is None:
        raise DeterministicRefusalError(
            "The batched strain was generated without an output grid, so each buffer starts at an "
            "arbitrary time and every chunk would have to be resampled onto the segment grid. Pass "
            "output_grid=SamplingGrid(segment_start, sampling_frequency) to simulate_cbc_batch."
        )

    produced_names = tuple(batch.detector_names)
    if expected_detector_names is not None and produced_names != tuple(expected_detector_names):
        raise DeterministicRefusalError(
            f"The device returned detectors {produced_names} but the caller expects "
            f"{tuple(expected_detector_names)}. Channel order carries detector identity downstream, "
            f"so this would attribute each signal to the wrong interferometer."
        )

    # One host transfer for the whole batch. Converting per event would copy the same device buffer
    # repeatedly, and the device-to-host path was measured to be allocator-bound rather than
    # bus-bound, so the number of transfers is what costs.
    strain = np.asarray(batch.strain)
    if strain.ndim != _BATCHED_STRAIN_DIMENSIONS:
        raise DeterministicRefusalError(
            f"Expected strain of shape (n_events, n_detectors, n_samples), got {strain.shape}."
        )

    start_indices = np.atleast_1d(np.asarray(batch.start_index))
    if start_indices.size != strain.shape[0]:
        raise DeterministicRefusalError(
            f"Got {start_indices.size} start indices for {strain.shape[0]} events; each event needs "
            f"its own buffer position."
        )
    if strain.shape[1] != len(produced_names):
        raise DeterministicRefusalError(
            f"Strain has {strain.shape[1]} detector rows but {len(produced_names)} detector names."
        )

    grid = batch.grid

    # The producer enforces this today, so it is not reachable through `simulate_cbc_batch`. Checked
    # anyway because this is a public conversion entry point: start times come from the grid's rate
    # while the chunks are built at the batch's, so a disagreement produces chunks whose timestamps
    # and sample spacing describe different things -- placed without complaint, wrong by a stretch
    # factor.
    if float(grid.sampling_frequency) != float(batch.sampling_frequency):
        raise DeterministicRefusalError(
            f"The output grid is at {grid.sampling_frequency!r} Hz but the batch is at "
            f"{batch.sampling_frequency!r} Hz. Start times are taken from the grid and the samples "
            f"from the batch, so the chunks would be timestamped at one rate and spaced at another."
        )

    # Integer-valued, not merely int-able. `int()` truncates, so a malformed 10.5 would place the
    # buffer half a sample early and look entirely normal afterwards.
    non_integral = [value for value in start_indices.tolist() if float(value) != int(float(value))]
    if non_integral:
        raise DeterministicRefusalError(
            f"Buffer start indices must be whole samples, got {non_integral}. Truncating them would "
            f"displace the affected events by a fraction of a sample without any later check noticing."
        )

    start_times = np.array([float(grid.time_of(int(index))) for index in start_indices], dtype=float)

    # Checked against *both* predicates, because agreeing with the producer is not the same as
    # agreeing with the consumer. gwmock-signal's `is_on_lattice` says the buffer sits on the grid it
    # was generated for; `is_aligned` is what `TimeSeries.inject` will actually apply when placing
    # it. Those can disagree -- at GPS 1577491296, 4000 Hz, sample 10, the representation error is
    # 2.29e-4 samples, which `is_on_lattice` accepts and a relative-tolerance `inject` rejected,
    # silently interpolating a chunk this module promised would be placed verbatim. Asserting only
    # the producer's view is how that went unnoticed.
    grid_offsets = (start_times - float(grid.epoch)) * float(grid.sampling_frequency)
    on_lattice = np.atleast_1d(grid.is_on_lattice(start_times))
    tolerance = alignment_tolerance(float(grid.epoch), float(grid.sampling_frequency), gps_times=start_times)
    placeable = np.array([is_aligned(float(offset), tolerance) for offset in grid_offsets], dtype=bool)

    rejected = ~(on_lattice & placeable)
    if np.any(rejected):
        first = int(np.argmax(rejected))
        offset = float(grid_offsets[first])
        raise DeterministicRefusalError(
            f"Event {first}'s buffer starts at GPS {start_times[first]!r}, which is {offset - round(offset):+.3e} "
            f"samples off the output grid (epoch {grid.epoch!r}, {grid.sampling_frequency!r} Hz). "
            f"gwmock-signal considers it on-lattice: {bool(on_lattice[first])}; gwmock considers it "
            f"placeable within {tolerance:.3e} samples: {bool(placeable[first])}. Injecting it would "
            f"resample the chunk instead of placing it exactly."
        )

    sampling_frequency = float(batch.sampling_frequency)
    return TimeSeriesList(
        [
            TimeSeries(
                data=np.ascontiguousarray(strain[event]),
                start_time=float(start_times[event]),
                sampling_frequency=sampling_frequency,
            )
            for event in range(strain.shape[0])
        ]
    )
