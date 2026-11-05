"""Continuous waves through the orchestrator, against the real waveform backend.

What is here is only what a real backend can establish: that the phase joins up across a segment
boundary, and that two pulsars compose into their sum. Both are physical claims about generated
strain, and a fake backend asserting either would be asserting its own arithmetic.

The load-bearing test is :meth:`TestPhaseCoherence.test_segments_join_up_through_the_orchestrator`.
The simulator guarantees coherence only if ``reference_time_ssb`` reaches it as one value for the
whole run; plumbing that derived it per segment would produce frames that look entirely normal and
are useless to a coherent search.

This module needs JAX, on which the continuous-wave simulator projects, and the LALPulsar ephemeris
tables, and it **does** run in CI: the `test-jax` job installs the extra, and caches the tables, which
gwmock-signal verifies against the digests it records -- the ones ``tests/data/ephemeris.sha256``
lists. It used to skip when they were absent, which on a fresh runner was
always, so it ran nowhere while the suite reported green. There is no skip gate now -- absent
tables fail.

Everything that does *not* need a waveform lives in ``test_continuous_wave_wiring.py``, which runs
in the default job too: routing, the refusals, the ordering exemption and its source-type gate,
provenance, and that the population index never advances. Put a wiring-level test there rather than
here, so it is covered without the optional extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax", reason="the [jax] extra is not installed")

from gwmock.cli.adapter_orchestration import AdapterOrchestrator
from gwmock.cli.utils.config import Config

_EPOCH = 1577491218.0
_FS = 64.0
_DETECTORS = ["H1", "L1"]


#: The tables by their standard names, which gwmock-signal looks up in its cache (``GWMOCK_SIGNAL_CACHE_DIR``
#: moves it), downloads there when absent, and refuses unless their content is the one it records. A
#: path with a directory would be read as the caller's own table and used unverified.
_EARTH_EPHEMERIS = "earth00-40-DE405.dat.gz"
_SUN_EPHEMERIS = "sun00-40-DE405.dat.gz"

_PULSARS = (
    "right_ascension,declination,frequency,initial_phase,amplitude_plus,amplitude_cross,polarization_angle\n"
    "1.1,0.3,20.0,0.4,1.0e-24,7.0e-25,0.2\n"
    "2.7,-0.6,15.0,1.9,6.0e-25,4.0e-25,1.1\n"
)


# No skip gate. These tests used to skip when the tables were not cached, which meant they never
# ran in CI at all: the ephemeris is fetched on demand, so "not cached" was the normal state on a
# fresh runner, and the module reported skipped rather than absent. A test that stops running
# silently is worse than one that fails. CI now caches the tables, has gwmock-signal verify them, and
# fails the job if it cannot obtain them; locally, gwmock-signal fetches them on first use.


def _config(tmp_path: Path, *, duration: float, total: float, n_pulsars: int = 2) -> dict[str, Any]:
    catalogue = tmp_path / "pulsars.csv"
    rows = _PULSARS.splitlines()
    catalogue.write_text("\n".join([rows[0], *rows[1 : 1 + n_pulsars]]) + "\n")
    return {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": _FS,
                "duration": duration,
                "total-duration": total,
                "start-time": _EPOCH,
                "seed": 7,
            },
            "working-directory": str(tmp_path),
            "output-directory": "output",
            "metadata-directory": "metadata",
        },
        "orchestration": {
            "population": {
                "backend": "FilePopulationLoader",
                "source-type": "cw",
                "n-samples": n_pulsars,
                "arguments": {"path": str(catalogue)},
            },
            "signal": {
                "source-type": "cw",
                "detectors": list(_DETECTORS),
                "minimum-frequency": 5,
                "effects": {"earth-rotation": True},
                "arguments": {
                    "earth_ephemeris": _EARTH_EPHEMERIS,
                    "sun_ephemeris": _SUN_EPHEMERIS,
                    "reference_time_ssb": _EPOCH,
                },
                "output": {
                    "output_directory": "signal",
                    "file_name": "cw.gwf",
                    "arguments": {"channel": "{{ detectors }}:STRAIN"},
                },
            },
        },
    }


def _orchestrator(tmp_path: Path, **kwargs: Any) -> AdapterOrchestrator:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = Config.model_validate(_config(tmp_path, **kwargs))
    return AdapterOrchestrator.from_config(
        config.orchestration, global_simulator_arguments=dict(config.globals.simulator_arguments)
    )


def _segment(orchestrator: AdapterOrchestrator) -> np.ndarray:
    chunks = orchestrator._simulate()
    assert len(chunks) == 1, f"expected one chunk spanning the segment, got {len(chunks)}"
    return np.asarray(chunks[0][0], dtype=float)


class TestPhaseCoherence:
    """The property the whole design protects, checked through the orchestrator."""

    def test_segments_join_up_through_the_orchestrator(self, tmp_path):
        """Two consecutive segments must reconstruct the same span generated in one go.

        The simulator only guarantees this if ``reference_time_ssb`` arrives as a single value for
        the run. Plumbing that derived it per segment -- from the segment start, say -- would leave
        each segment individually perfect and the join incoherent, with nothing in the frames
        showing it. The tolerance clears the reassociation floor between differently-sized FFTs and
        is nine orders below the ~1 of peak a phase reset produces.
        """
        stitched_source = _orchestrator(tmp_path / "segmented", duration=64, total=128)
        first = _segment(stitched_source)
        stitched_source.update_state()
        second = _segment(stitched_source)
        stitched = np.concatenate([first, second])

        whole = _segment(_orchestrator(tmp_path / "whole", duration=128, total=128))

        assert stitched.shape == whole.shape
        peak = float(np.max(np.abs(whole)))
        worst = float(np.max(np.abs(stitched - whole))) / peak
        assert worst < 1e-9, f"segments disagree with the continuous run by {worst:.3e} of peak"


class TestTheCatalogueReachesEverySegment:
    """Every pulsar contributes to every segment, unlike an event that is consumed once."""

    def test_two_pulsars_give_the_sum_of_the_two_alone(self, tmp_path):
        """The catalogue must be *summed*, and this checks the sum rather than merely a difference.

        Detecting that a second source changed the output would also pass for an implementation
        that overwrote, scaled, or otherwise distorted the total. Comparing against the two sources
        generated separately and added pins the actual composition.
        """
        first_only = _segment(_orchestrator(tmp_path / "first", duration=64, total=64, n_pulsars=1))
        both = _segment(_orchestrator(tmp_path / "both", duration=64, total=64, n_pulsars=2))

        # The second pulsar alone, obtained by removing the first from the catalogue.
        second_dir = tmp_path / "second"
        second_dir.mkdir(parents=True, exist_ok=True)
        rows = _PULSARS.splitlines()
        (second_dir / "only.csv").write_text(rows[0] + "\n" + rows[2] + "\n")
        config = _config(second_dir, duration=64, total=64, n_pulsars=1)
        config["orchestration"]["population"]["arguments"]["path"] = str(second_dir / "only.csv")
        parsed = Config.model_validate(config)
        second_only = _segment(
            AdapterOrchestrator.from_config(
                parsed.orchestration, global_simulator_arguments=dict(parsed.globals.simulator_arguments)
            )
        )

        expected = first_only + second_only
        peak = float(np.max(np.abs(expected)))
        worst = float(np.max(np.abs(both - expected))) / peak
        assert worst < 1e-9, f"two pulsars together differ from their sum by {worst:.3e} of peak"
