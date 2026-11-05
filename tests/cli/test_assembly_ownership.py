"""Two implementations could place signals into segments; this pins that they agree.

``gwmock_signal.jax_batch.assemble_segments`` scatters a batch into a fixed tiling of segments in
memory, given every segment start up front. gwmock assembles differently by necessity: it streams,
holding one segment at a time, claiming events for it and carrying the overflow forward as a tail.

gwmock owns assembly and does not call ``assemble_segments`` -- the batched path is a different way
to *generate*, not a different way to assemble. That decision is recorded in
:meth:`~gwmock.cli.adapter_orchestration.AdapterOrchestrator._simulate_batched_segment`, and it is
the reason this file exists rather than a deduplication: two assemblers stay in the ecosystem, so
the risk is that they drift while both keep producing plausible segments. Nothing about a wrong
segment looks wrong.

So the guard is an equivalence measurement rather than a shared implementation. Same events, both
assemblers, compared sample by sample.

**The two do not agree bit-for-bit, and the reason is generation rather than assembly.** Each path
generates its own batch -- gwmock per segment, the scatter once for the catalogue -- and a JAX batch
is padded to its longest event, so the same event is computed at different batch shapes and the FFTs
differ in their last bits. That makes the residue catalogue-dependent, which is why a mixed-mass case
is tested explicitly rather than assumed to behave like the uniform one.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock.cli.utils.config import Config

_FS = 1024.0
_SEGMENT = 16.0
_SEGMENTS = 3
_START = 1577491296.0
_DETECTOR = "ET-Triangle-Sardinia"

#: One event mid-segment, one just past a boundary so the claiming rule has to pull it earlier, and
#: one in the last segment. The middle event is the interesting one: gwmock claims it for the
#: *previous* segment because its waveform starts there, while ``assemble_segments`` simply scatters
#: it into both. If the two disagree anywhere, it is on an event like that.
_EVENTS: list[dict[str, Any]] = [
    {
        "detector_frame_mass_1": 30.0,
        "detector_frame_mass_2": 25.0,
        "distance": 400.0,
        "right_ascension": 1.0,
        "declination": 0.5,
        "polarization_angle": 0.2,
        "inclination": 0.3,
        "coa_time": _START + offset,
    }
    for offset in (8.0, _SEGMENT + 1.0, 2 * _SEGMENT + 8.0)
]


def _population_file(directory: Path, events: list[dict[str, Any]]) -> Path:
    """Write *events* as a catalogue CSV, since the loader reads from disk."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "population.csv"
    columns = list(events[0])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for event in events:
            writer.writerow([event[column] for column in columns])
    return path


def _orchestrator(
    directory: Path,
    events: list[dict[str, Any]] | None = None,
    duration: float = _SEGMENT,
):
    """Return an orchestrator on the batched path, which is the one that shares generation."""
    events = _EVENTS if events is None else events
    from gwmock.cli.adapter_orchestration import AdapterOrchestrator

    raw = {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _FS,
                "duration": duration,
                "total-duration": duration * _SEGMENTS,
                "start-time": _START,
                "seed": 20260804,
            },
            "working-directory": str(directory),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "bbh",
                "n-samples": len(events),
                "arguments": {"path": str(_population_file(directory, events))},
            },
            "signal": {
                "source-type": "bbh",
                "waveform-model": "IMRPhenomD",
                # The batched entry point is ripple-only, and both paths must generate with the
                # same library or this compares waveforms rather than assemblers.
                "waveform-backend": "ripple",
                "execution": "batched",
                "minimum-frequency": 20,
                "detectors": [_DETECTOR],
                "output": {
                    "output_directory": "signal",
                    "file_name": "signal.gwf",
                    "arguments": {"channel": "X:STRAIN"},
                },
            },
        },
    }
    config = Config.model_validate(raw)
    return AdapterOrchestrator.from_config(
        config.orchestration,
        global_simulator_arguments=dict(config.globals.simulator_arguments),
    )


def _streamed_segments(orchestrator) -> list[np.ndarray]:
    """Assemble with gwmock: one segment at a time, tails carried forward."""
    segments = []
    for _ in range(_SEGMENTS):
        segment = orchestrator.simulate().signal_segment
        segments.append(np.atleast_2d(np.asarray(segment, dtype=float)).copy())
        orchestrator.update_state()
    return segments


def _scattered_segments(orchestrator, events: list[dict[str, Any]] | None = None) -> list[np.ndarray]:
    """Assemble with gwmock-signal: one batch on the run grid, scattered into the tiling."""
    events = _EVENTS if events is None else events
    from gwmock_signal import SamplingGrid, simulate_cbc_batch
    from gwmock_signal.jax_batch import assemble_segments

    from gwmock.signal.adapter import SignalAdapter
    from gwmock.signal.device_chunks import canonicalise_parameters

    batch = simulate_cbc_batch(
        orchestrator.signal_adapter.device_approximant(),
        list(orchestrator._signal_network.detector_names),
        sampling_frequency=_FS,
        minimum_frequency=20.0,
        parameters=canonicalise_parameters(SignalAdapter.events_to_struct_of_arrays(events)),
        backend=orchestrator._batched_waveform_backend(),
        effects=orchestrator.effects,
        output_grid=SamplingGrid(_START, _FS),
    )
    stacks = assemble_segments(
        batch,
        segment_duration=_SEGMENT,
        segment_start_times=[_START + index * _SEGMENT for index in range(_SEGMENTS)],
    )
    return [np.atleast_2d(np.asarray(stack.data, dtype=float)).copy() for stack in stacks]


def _generated_energy_in_run(orchestrator, events: list[dict[str, Any]] | None = None) -> float:
    """Return the strain energy the *generator* produced inside the run window.

    An oracle independent of both assemblers, so a comparison against it says which one is wrong and
    by how much, rather than only that they differ.

    **Scope, stated because it is narrower than "no energy is lost" sounds.** Only samples inside
    ``[start, start + segments * duration)`` are counted. Content generated *outside* that window --
    a buffer beginning before the run, most of all -- is excluded here, so this oracle cannot detect
    its loss: no segment covers that time, both assemblers truncate it identically, and all three
    numbers agree while the data is genuinely missing it. That loss is real but it is not an assembly
    defect, and it is reported elsewhere, by
    ``TimeSeries._report_content_before_segment`` per signal.
    """
    events = _EVENTS if events is None else events
    from gwmock_signal import SamplingGrid, simulate_cbc_batch

    from gwmock.signal.adapter import SignalAdapter
    from gwmock.signal.device_chunks import canonicalise_parameters

    grid = SamplingGrid(_START, _FS)
    batch = simulate_cbc_batch(
        orchestrator.signal_adapter.device_approximant(),
        list(orchestrator._signal_network.detector_names),
        sampling_frequency=_FS,
        minimum_frequency=20.0,
        parameters=canonicalise_parameters(SignalAdapter.events_to_struct_of_arrays(events)),
        backend=orchestrator._batched_waveform_backend(),
        effects=orchestrator.effects,
        output_grid=grid,
    )
    strain = np.asarray(batch.strain, dtype=float)
    starts = np.asarray(batch.grid.time_of(batch.start_index), dtype=float)
    run_end = _START + _SEGMENTS * _SEGMENT
    total = 0.0
    for event in range(strain.shape[0]):
        times = starts[event] + np.arange(strain.shape[2]) / _FS
        inside = (times >= _START - 0.5 / _FS) & (times < run_end - 0.5 / _FS)
        total += float(np.sum(strain[event][:, inside] ** 2))
    return total


@pytest.fixture(scope="module")
def assembled(tmp_path_factory) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Assemble the same events both ways, and measure what generation produced.

    A separate orchestrator per path: the streaming one is advanced through the run with
    ``update_state``, so sharing it would leave the scatter reading mutated state and make these
    tests order-sensitive as soon as either helper starts consulting it.
    """
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    directory = tmp_path_factory.mktemp("assembly")
    streamed = _streamed_segments(_orchestrator(directory))
    scatter_orchestrator = _orchestrator(directory)
    scattered = _scattered_segments(scatter_orchestrator)
    return streamed, scattered, _generated_energy_in_run(scatter_orchestrator)


def test_both_assemblers_produce_the_same_strain(assembled):
    """The whole point: same events in, same samples out, whichever assembler ran.

    **What the comparison actually is**, stated precisely because a looser description invites the
    wrong assumption. It is the largest absolute sample difference in a segment, normalised by that
    segment's peak. It is deliberately *not* element-wise exact equality -- the two paths generate
    separately and the difference is genuinely nonzero -- and it is not element-wise relative either:
    a spurious value at a sample where the reference is zero passes, provided it stays under
    ``peak * 1e-9``. That is the intended contract, since the noise being admitted scales with the
    signal in the segment rather than with each sample.

    An absolute tolerance would not work here at all: strain is ~1e-21, so any fixed threshold is
    either meaningless or unfailable depending on which side of 1e-21 it falls.

    The tolerance admits generation-side noise and nothing more, and what produces that noise is
    worth stating because the obvious explanation is wrong. It is **not** the grid anchor: generating
    the same event on a grid at the run start and on one at the third segment's start is
    bit-identical, measured. It is **batch composition**. gwmock generates each segment's events in
    their own batch while the scatter generates the whole catalogue in one, and a batch is padded to
    its longest event -- 4,608 samples for a 30+25 binary alone against 236,196 once a 1.4+1.4 binary
    joins it. JAX returns last-bit-different FFT results at different batch shapes.

    So the residue depends on the catalogue, which is why the bound here is 1e-9 rather than tighter:
    3.3e-13 of peak for the same-mass catalogue below, and 2.0e-11 with a binary neutron star mixed
    in. Eight orders of magnitude below the 17.7% a real placement regression produces, so the
    sensitivity that matters is intact.
    """
    streamed, scattered, _generated = assembled

    assert len(streamed) == len(scattered) == _SEGMENTS
    for index, (mine, theirs) in enumerate(zip(streamed, scattered, strict=True)):
        assert mine.shape == theirs.shape, f"segment {index}: shape differs"
        peak = float(np.max(np.abs(theirs)))
        assert peak > 0.0, f"segment {index} is silent, so this comparison proves nothing"
        difference = float(np.max(np.abs(mine - theirs)))
        assert difference / peak < 1e-9, (
            f"segment {index}: assemblers differ by {difference:.3e}, {difference / peak:.2e} of "
            f"peak {peak:.3e} -- more than float64 accumulation, so one of them has drifted"
        )


def test_both_assemblers_place_everything_the_generator_produced(assembled):
    """Conservation, against generation rather than against each other.

    Comparing the two assembled totals only says they agree. Against an independent reference this
    says *which* assembler is short and by how much: reverting gwmock's claiming rule makes this
    report "streamed assembly holds 5.337782e-40 of the 5.402430e-40 the generator produced inside the
    run (-1.1967%)", where a two-way comparison could only report a difference.

    A one-sided drop does also fail the per-sample comparison, at 17.7% of peak for a cropped
    inspiral and 100% for a whole event, so this is not the only thing that would notice. What it adds
    is a *partial* loss shared by neither assembler's samples in an obvious way: a shared partial
    truncation inside the run leaves the per-sample comparison passing while this reports the energy
    gap directly.

    What it does *not* cover is loss the two share, because the reference is windowed to the run --
    see :func:`_generated_energy_in_run`. An event whose buffer begins before the run is truncated
    identically by both and by the oracle, so all three agree. That is deliberate: no segment covers
    that time, so it is not an assembly question.
    """
    streamed, scattered, generated = assembled

    mine = sum(float(np.sum(segment**2)) for segment in streamed)
    theirs = sum(float(np.sum(segment**2)) for segment in scattered)

    assert generated > 0.0, "the oracle measured no signal, so this proves nothing"
    for name, total in (("streamed", mine), ("scattered", theirs)):
        assert total == pytest.approx(generated, rel=1e-6, abs=0.0), (
            f"{name} assembly holds {total:.6e} of the {generated:.6e} the generator produced inside "
            f"the run ({100.0 * (total / generated - 1.0):+.4f}%): content is being dropped or "
            f"counted twice"
        )


def test_gwmock_does_not_call_the_other_assembler(tmp_path, monkeypatch):
    """Ownership, observed at run time rather than grepped for.

    If a future change wires ``assemble_segments`` into gwmock, the equivalence tests above keep
    passing -- they would be comparing that function against itself, which is the failure mode most
    worth catching and the one hardest to notice.

    Instrumenting the real function is what makes this hold: an earlier version scanned two module
    files for the literal string, which an alias, a third module or a ``getattr`` would walk straight
    past.
    """
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    from gwmock_signal import jax_batch

    calls: list[int] = []
    original = jax_batch.assemble_segments

    def counting(*args: Any, **kwargs: Any):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(jax_batch, "assemble_segments", counting)

    # Positive control first: a guard that watches the wrong name would report "not called" forever.
    # The scatter helper does call it, so this proves the instrumentation can see a call at all.
    _scattered_segments(_orchestrator(tmp_path))
    assert calls, "the instrumentation cannot observe a call, so its silence below would mean nothing"

    calls.clear()
    _streamed_segments(_orchestrator(tmp_path))

    assert not calls, (
        "gwmock called gwmock_signal.jax_batch.assemble_segments. gwmock owns assembly; if that is "
        "changing deliberately, the equivalence tests in this module become circular and must be "
        "rewritten to compare against something else"
    )


def test_a_segment_duration_off_the_sample_grid_is_a_known_boundary(assembled, tmp_path):
    """The two are *not* equivalent here, and pinning that is better than avoiding the case.

    gwmock sizes a segment with ``int(duration * sampling_frequency)``, which truncates silently. The
    scatter requires a whole number of samples for a grid-aligned batch and refuses otherwise. So a
    duration of 16.0006 s at 1024 Hz -- 16384.6144 samples -- is a configuration where equivalence
    cannot hold by construction: one rounds down to 16.0 s of data, the other raises.

    Asserted so this is documented behaviour rather than an untested gap, and so that a change making
    the scatter round silently instead of refusing does not slip through: that would turn a loud
    configuration error into two subtly different datasets.
    """
    _streamed, scattered, _generated = assembled
    from gwmock_signal import SamplingGrid, simulate_cbc_batch
    from gwmock_signal.jax_batch import assemble_segments

    from gwmock.signal.adapter import SignalAdapter
    from gwmock.signal.device_chunks import canonicalise_parameters

    off_grid = _SEGMENT + 0.0006
    exact = off_grid * _FS
    assert abs(exact - round(exact)) > 1e-6, "this duration must not land on a whole sample"

    orchestrator = _orchestrator(tmp_path)
    batch = simulate_cbc_batch(
        orchestrator.signal_adapter.device_approximant(),
        list(orchestrator._signal_network.detector_names),
        sampling_frequency=_FS,
        minimum_frequency=20.0,
        parameters=canonicalise_parameters(SignalAdapter.events_to_struct_of_arrays(_EVENTS[:1])),
        backend=orchestrator._batched_waveform_backend(),
        effects=orchestrator.effects,
        output_grid=SamplingGrid(_START, _FS),
    )

    with pytest.raises(ValueError, match="whole number of samples"):
        assemble_segments(batch, segment_duration=off_grid, segment_start_times=[_START])

    # gwmock's own sizing truncates rather than refusing. Run it at that duration instead of
    # restating the arithmetic: if gwmock starts rejecting or rounding, the asymmetry recorded here
    # has changed and this must fail rather than keep passing on a formula.
    truncated = _streamed_segments(_orchestrator(tmp_path / "offgrid", _EVENTS[:1], duration=off_grid))
    assert truncated[0].shape[1] == int(off_grid * _FS), (
        f"gwmock sized the segment as {truncated[0].shape[1]} samples; the recorded asymmetry is that "
        f"it truncates {off_grid * _FS} to {int(off_grid * _FS)}"
    )
    assert truncated[0].shape[1] == round(_SEGMENT * _FS), (
        "the truncation lands on exactly the on-grid sample count, which is what makes the scatter's "
        "refusal and gwmock's silence differ in kind rather than in degree"
    )
    assert scattered[0].shape[1] == int(_SEGMENT * _FS), "the on-grid case is unaffected"


def test_a_mixed_mass_catalogue_still_agrees(tmp_path):
    """The uniform catalogue above is the easy case; a mass spectrum is the realistic one.

    All three events above are 30+25, so both paths pad their batches to the same length and the
    residue stays at 3.3e-13 of peak. Add a binary neutron star and the scatter's single batch is
    padded to *its* buffer -- 236,196 samples against 4,608 -- while gwmock still generates the black
    hole binary in a batch of its own. If batch composition were going to break the agreement, this is
    where it would show.

    Measured at 2.0e-11 of peak, sixty times the uniform case and still eight orders below a real
    placement regression. Recorded because "the assemblers agree" is otherwise tested only on a
    catalogue where every buffer happens to be the same length.
    """
    pytest.importorskip("ripplegw", reason="the [jax] extra is not installed")
    mixed = [
        {**_EVENTS[0], "coa_time": _START + 8.0},
        {
            **_EVENTS[0],
            "detector_frame_mass_1": 1.4,
            "detector_frame_mass_2": 1.4,
            "coa_time": _START + 2 * _SEGMENT + 8.0,
        },
    ]
    streamed = _streamed_segments(_orchestrator(tmp_path / "streamed", mixed))
    scattered = _scattered_segments(_orchestrator(tmp_path / "scattered", mixed), mixed)

    for index, (mine, theirs) in enumerate(zip(streamed, scattered, strict=True)):
        peak = float(np.max(np.abs(theirs)))
        if peak == 0.0:
            continue
        difference = float(np.max(np.abs(mine - theirs)))
        assert difference / peak < 1e-9, (
            f"segment {index}: a mixed-mass catalogue makes the assemblers differ by "
            f"{difference / peak:.2e} of peak, beyond the batch-composition noise this admits"
        )
