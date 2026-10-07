"""Reporting the signal content that injection drops off the front of a segment.

``inject`` crops a chunk to the target's span. Forward overflow is returned as a tail and carried
into the next segment; backward overflow is discarded, because the segments it belongs to have
already been written. A compact-binary waveform starts before its ``coa_time`` while the segment
claiming an event is the one ``coa_time`` falls in, so the discard is reached by ordinary
configurations rather than by anything exotic.

These tests pin the measurement and the warning. They do not assert that the content is kept --
it is not, and keeping it is a change to how events are scheduled, not to how they are injected.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
from astropy.units import Quantity

from gwmock.data.time_series.inject import measure_content_before
from gwmock.data.time_series.time_series import TimeSeries

_FS = 1024.0
# A GPS-scale epoch on purpose: the float64 spacing here is ~2.4e-7 s, which is where a naive
# boundary comparison starts reporting phantom dropped samples.
_EPOCH = 1577491296.0


def _series(start_time: float, n_samples: int, amplitude: float = 1.0, n_channels: int = 1) -> TimeSeries:
    return TimeSeries(
        data=np.full((n_channels, n_samples), amplitude, dtype=float),
        start_time=Quantity(start_time, unit="s"),
        sampling_frequency=Quantity(_FS, unit="Hz"),
    )


class TestMeasurement:
    """The numbers themselves, without the logging."""

    def test_a_chunk_starting_inside_the_segment_loses_nothing(self):
        chunk = _series(_EPOCH + 1.0, 1024)

        assert measure_content_before(_EPOCH, _FS, chunk) == (0, 0.0, 0.0)

    def test_a_tail_landing_exactly_on_the_boundary_loses_nothing(self):
        """The carry-forward case, which happens every segment and must stay silent."""
        chunk = _series(_EPOCH, 1024)

        assert measure_content_before(_EPOCH, _FS, chunk) == (0, 0.0, 0.0)

    def test_a_boundary_off_by_a_float_ulp_still_loses_nothing(self):
        """Without the half-sample slack this reports a dropped sample on every carried tail.

        ``np.spacing`` at a GPS epoch is ~2.4e-7 s against a 9.8e-4 s sample period, so the error is
        four thousand times smaller than a sample and must not round up to one.
        """
        chunk = _series(_EPOCH - np.spacing(_EPOCH), 1024)

        samples, _, _ = measure_content_before(_EPOCH, _FS, chunk)

        assert samples == 0

    def test_the_dropped_sample_count_and_duration_are_exact(self):
        chunk = _series(_EPOCH - 0.5, 2048)

        samples, seconds, _ = measure_content_before(_EPOCH, _FS, chunk)

        assert samples == 512
        assert seconds == pytest.approx(0.5)

    @pytest.mark.parametrize("gap", [0.5, 2.0, 10.0])
    def test_the_duration_always_matches_the_sample_count(self, gap: float):
        """The warning prints the two side by side, so they have to describe the same span."""
        samples, seconds, _ = measure_content_before(_EPOCH, _FS, _series(_EPOCH - gap, 512))

        assert seconds == pytest.approx(samples / _FS)

    def test_the_energy_fraction_is_the_share_of_summed_squares(self):
        """Not the share of samples -- an SNR responds to energy, and the two differ a lot.

        Half the samples here carry a quarter of the amplitude, hence a sixteenth of the energy
        each, so the dropped fraction is 1/17 rather than 1/2.
        """
        data = np.concatenate([np.full(512, 0.25), np.full(512, 1.0)])
        chunk = TimeSeries(
            data=data.reshape(1, -1),
            start_time=Quantity(_EPOCH - 0.5, unit="s"),
            sampling_frequency=Quantity(_FS, unit="Hz"),
        )

        _, _, fraction = measure_content_before(_EPOCH, _FS, chunk)

        assert fraction == pytest.approx((512 * 0.0625) / (512 * 0.0625 + 512 * 1.0))

    def test_a_silent_chunk_reports_no_energy_rather_than_dividing_by_zero(self):
        chunk = _series(_EPOCH - 0.5, 2048, amplitude=0.0)

        samples, _, fraction = measure_content_before(_EPOCH, _FS, chunk)

        assert samples == 512
        assert fraction == 0.0

    def test_every_channel_counts_towards_the_energy(self):
        """A per-detector chunk drops content in all of its channels at once."""
        one = measure_content_before(_EPOCH, _FS, _series(_EPOCH - 0.5, 2048, n_channels=1))
        three = measure_content_before(_EPOCH, _FS, _series(_EPOCH - 0.5, 2048, n_channels=3))

        assert one[2] == pytest.approx(three[2]), "the fraction is per chunk, so channels cancel out"

    def test_a_sample_exactly_on_the_threshold_is_not_dropped(self):
        """The rounding tie, pinned rather than left to `searchsorted`'s default.

        A sample exactly half a sample before the boundary is ambiguous by construction. It resolves
        towards the boundary -- not counted -- which is what the slack exists to do. Untested, a
        change of `side` would flip it silently.
        """
        chunk = _series(_EPOCH - 0.5 / _FS, 1024)

        assert measure_content_before(_EPOCH, _FS, chunk) == (0, 0.0, 0.0)

    def test_a_sample_just_past_the_threshold_is_dropped(self):
        """The other side of the tie, so the test above pins a boundary rather than a blanket."""
        chunk = _series(_EPOCH - 0.5 / _FS - 1.0 / _FS, 1024)

        samples, _, _ = measure_content_before(_EPOCH, _FS, chunk)

        assert samples == 1

    def test_an_empty_chunk_is_handled(self):
        assert measure_content_before(_EPOCH, _FS, _series(_EPOCH, 0)) == (0, 0.0, 0.0)


class TestTheWarning:
    """What a run actually sees. Silence here is the defect this exists to remove."""

    @staticmethod
    def _segment() -> TimeSeries:
        return TimeSeries(
            data=np.zeros((1, int(4 * _FS))),
            start_time=Quantity(_EPOCH, unit="s"),
            sampling_frequency=Quantity(_FS, unit="Hz"),
        )

    def test_injecting_a_chunk_that_starts_early_warns(self, caplog):
        chunk = _series(_EPOCH - 0.5, 2048)
        chunk.metadata["injection_parameters"] = {"coa_time": _EPOCH + 1.0}

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(chunk)

        assert "Discarding" in caplog.text
        assert "0.500 s" in caplog.text, "the message must say how much time went"
        assert "512 samples" in caplog.text
        assert str(_EPOCH + 1.0) in caplog.text, "the message must identify which signal"

    def test_the_warning_reports_energy_not_sample_count(self, caplog):
        """A message quoting 50% of samples for 6% of the energy would misdirect the reader."""
        data = np.concatenate([np.full(512, 0.25), np.full(512, 1.0)])
        chunk = TimeSeries(
            data=data.reshape(1, -1),
            start_time=Quantity(_EPOCH - 0.5, unit="s"),
            sampling_frequency=Quantity(_FS, unit="Hz"),
        )

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(chunk)

        assert "5.88% of its unweighted strain-squared energy" in caplog.text

    def test_an_ordinary_injection_is_silent(self, caplog):
        """Most injections are fine, and a warning on every one would be ignored."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(_series(_EPOCH + 1.0, 1024))

        assert caplog.text == ""

    def test_a_chunk_without_injection_parameters_still_warns(self, caplog):
        """The loss is worth reporting even when the chunk cannot say which signal it is."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(_series(_EPOCH - 0.5, 2048))

        assert "Discarding" in caplog.text
        assert "unknown" in caplog.text

    def test_a_real_run_warns_when_the_pre_coalescence_duration_is_unavailable(self, tmp_path, caplog):
        """Synthetic arrays cannot show that a real configuration reaches this.

        A 30+25 Msun binary at 20 Hz is conditioned into a buffer that starts seconds before its
        merger, so a ``coa_time`` 0.5 s past a segment boundary leaves all but the last 0.5 s of
        that lead in the previous segment -- which is already written. How long the lead is belongs
        to the waveform backend's conditioning and changes with it (3.6 s in one gwmock-signal
        release, 7.2 s in the next), so the expected loss is taken from the backend's own answer.

        Segments now claim an event by where its waveform starts, so this configuration keeps that
        inspiral -- ``tests/cli/test_inspiral_segment_placement.py`` measures that it survives, by
        energy rather than by the absence of this warning. What is exercised here is the fallback:
        with the pre-coalescence duration unavailable -- an older gwmock-signal, a backend that
        cannot say -- placement reverts to ``coa_time`` and the loss returns. That is the path this
        reporting exists for, and the one that must keep saying how much went.
        """
        from gwmock.cli.adapter_orchestration import AdapterOrchestrator
        from gwmock.cli.utils.config import Config

        epoch, segment = 1577491296.0, 16.0
        population = tmp_path / "population.csv"
        population.write_text(
            "detector_frame_mass_1,detector_frame_mass_2,coa_time,distance,declination,"
            "right_ascension,polarization_angle,inclination\n"
            f"30.0,25.0,{epoch + segment + 0.5},400.0,0.3,1.1,0.2,0.5\n"
        )
        config = Config.model_validate(
            {
                "globals": {
                    "simulator-arguments": {
                        "sampling-frequency": _FS,
                        "duration": segment,
                        "total-duration": segment * 2,
                        "start-time": epoch,
                        "seed": 1,
                    },
                    "working-directory": str(tmp_path),
                    "output-directory": "output",
                    "metadata-directory": "metadata",
                },
                "orchestration": {
                    "population": {
                        "backend": "FilePopulationLoader",
                        "source-type": "bbh",
                        "n-samples": 1,
                        "arguments": {"path": str(population)},
                    },
                    "signal": {
                        "source-type": "bbh",
                        "waveform-model": "IMRPhenomD",
                        "minimum-frequency": 20,
                        "detectors": ["H1"],
                        "output": {
                            "output_directory": "signal",
                            "file_name": "s.gwf",
                            "arguments": {"channel": "H1:S"},
                        },
                    },
                },
            }
        )
        orchestrator = AdapterOrchestrator.from_config(
            config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
        )

        # Force the fallback rather than simulate an old install: the query is made through the
        # adapter, so replacing its answer with "unknown" is the whole difference. The real answer
        # is still recorded, with the arguments placement asked it with, because it is what the
        # loss should be measured against.
        real_query = orchestrator.signal_adapter.pre_coalescence_duration
        leads = []

        def unavailable(*args, **kwargs):
            leads.append(real_query(*args, **kwargs))

        orchestrator.signal_adapter.pre_coalescence_duration = unavailable

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            orchestrator.simulate()  # first segment: nothing to claim yet
            orchestrator.update_state()
            orchestrator.simulate()  # the segment holding coa_time

        assert "Discarding" in caplog.text, "the fallback lost part of an inspiral without saying so"
        # Exact to a sample, against a figure the reporting does not compute: everything before the
        # boundary is the backend's lead less the 0.5 s by which coa_time follows the boundary. A
        # fixed number here would encode one release's conditioning and fail on the next while
        # saying nothing about the reporting. This still fails if the measurement were the whole
        # buffer (which also holds the post-merger tail), a single sample, or zero.
        assert len(set(leads)) == 1, f"the backend gave inconsistent leads for one event: {leads}"
        assert leads[0] is not None, "the real query must answer here, or the expected loss is unknown"
        assert leads[0] > 1.0, f"expected a real inspiral lead, got {leads[0]}"
        dropped = float(re.search(r"Discarding ([\d.]+) s", caplog.text).group(1))
        assert dropped == pytest.approx(leads[0] - 0.5, abs=1.0 / _FS), (
            f"the backend leads coalescence by {leads[0]} s, so {leads[0] - 0.5} s precede the boundary; "
            f"the warning reported {dropped}"
        )

    def test_a_chunk_lying_entirely_before_the_segment_is_reported(self, caplog):
        """The larger loss must not be the quieter one.

        ``inject`` returns early for a chunk with no overlap, and ``TimeSeriesMixin.simulate`` then
        drops it from the cache. Reporting only the partially-early case would mean 100% of a
        waveform vanishing with less said about it than 30% of one.
        """
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(_series(_EPOCH - 10.0, 1024))

        assert "Discarding" in caplog.text
        assert "100.00% of its unweighted strain-squared energy" in caplog.text
        # 1024 samples at 1024 Hz span 1 s, however far before the segment they sit. Reporting the
        # gap to the boundary instead would say 10 s here, while the message pairs the figure with a
        # sample count as though the two described the same span.
        assert "1.000 s" in caplog.text

    def test_an_off_grid_chunk_is_measured_before_it_is_resampled(self, caplog):
        """The interpolation branch rebinds the chunk onto the segment's own grid.

        Measuring after that would report what survived resampling rather than what was supplied,
        so the loss has to be taken from the chunk as handed in.
        """
        off_grid = _series(_EPOCH - 0.5 + 0.5 / _FS, 2048)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            self._segment().inject(off_grid)

        assert "Discarding" in caplog.text
        dropped = float(re.search(r"Discarding ([\d.]+) s", caplog.text).group(1))
        assert dropped == pytest.approx(0.5 - 0.5 / _FS, abs=1.0 / _FS)

    def test_the_content_is_still_dropped(self, caplog):
        """Pins the behaviour this PR does *not* change, so a later fix has to update it knowingly."""
        segment = self._segment()
        chunk = _series(_EPOCH - 0.5, 2048)

        with caplog.at_level(logging.WARNING, logger="gwmock"):
            segment.inject(chunk)

        written = np.asarray(segment[0]).astype(float)
        assert np.count_nonzero(written) == 1536, "only the part inside the segment is placed"
        assert "Discarding" in caplog.text
