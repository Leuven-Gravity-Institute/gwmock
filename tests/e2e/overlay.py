"""Test-time overrides that make an example configuration cheap to run.

The examples are sized to be realistic -- most generate a day of data in 4096-second segments.
The suite needs seconds. Rather than shorten the examples themselves, which would make them
worse at their actual job, each is loaded and a small declarative overlay is merged over it.

Two consequences worth knowing:

* Changing an example changes what the suite runs. That is deliberate -- it is how an example
  that stops working becomes a test failure.
* An overlay is *not* allowed to change which code path the config takes. Shortening a duration
  or repointing an input is fine; switching the waveform model or the detector network would
  make the test stop covering the thing its matrix entry claims.

Inputs are repointed at in-repo files because most example configs fetch their population over
the network -- two of them from ``sandbox.zenodo.org``, whose records are purged. A suite that
depends on those is a suite that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIRECTORY = Path(__file__).resolve().parent
EXAMPLES_DIR = _REPO_ROOT / "examples"

#: Seed pinned for every run, so two runs of one entry are comparable.
#:
#: Written to ``noise.arguments.seed`` as well as the global, because the orchestrator copies the
#: global seed with ``setdefault`` -- an example that pins its own (several do, at 42) keeps it
#: and the global is ignored. Setting only the global looks like it controls the seed while the
#: example silently does; verified by changing each and seeing which moved the output.
_TEST_SEED = 20260731

#: In-repo CBC population, used in place of every remote population file. Small and fixed, so a
#: run is reproducible and needs no network.
#:
#: Test-owned rather than the example's copy, because its coalescence time has to sit at the epoch
#: below and the examples deliberately use 2030. Identical to
#: ``examples/signal/bbh_population.csv`` apart from that time.
POPULATION_FIXTURE = _THIS_DIRECTORY / "data" / "bbh_population.csv"

#: In-repo pulsar catalogue for the continuous-wave entry.
#:
#: The example points at this file with a path relative to the examples tree, which does not
#: survive being copied into a test working directory -- the run failed with
#: `FileNotFoundError: ../../cw_population.csv`. Repointed absolutely for the same reason every
#: other entry's population is: the overlay owns where inputs come from.
CW_POPULATION_FIXTURE = EXAMPLES_DIR / "signal" / "cw_population.csv"


def _fixture_event_gps() -> float:
    """Return the single event's coalescence time, read from the fixture rather than restated.

    Duplicating it let the CSV and this constant drift: the bracket check below would keep passing
    while the event sat outside every segment, and only the slower end-to-end nonzero assertion
    would have caught it.
    """
    rows = POPULATION_FIXTURE.read_text(encoding="utf-8").strip().splitlines()
    columns = rows[0].split(",")
    return float(rows[1].split(",")[columns.index("coa_time")])


#: The single event in :data:`POPULATION_FIXTURE`, read from the file itself.
_FIXTURE_EVENT_GPS = _fixture_event_gps()

#: Start time placing that event inside a short segment. A run whose span misses the population
#: still succeeds and writes only zeros, so this is not a cosmetic choice -- see the
#: ``contains_signal`` assertions in the end-to-end tests.
#:
#: GPS 1419724816, which is 2024-12-31 23:59:58 UTC -- chosen for 16-second alignment, so the
#: fixture event four seconds later falls at 2025-01-01 00:00:02. The choice of *year* is
#: load-bearing; the seconds are only alignment. The shipped examples run in 2030 to match
#: the Einstein Telescope era, which is roughly 885 days beyond the end of the IERS Earth-orientation
#: table Astropy ships. Astropy clamps UT1-UTC to the table's final value there, so every weekly
#: `astropy-iers-data` release moved that clamped value and with it the generated strain -- measured
#: at 1.569e-06 of peak, above this suite's 1e-06 drift tolerance, so the reference comparison failed
#: roughly weekly on something that was not a defect.
#:
#: Inside the *finalised* part of the table, nothing moves. Measured across the 0.2026.7.27 and
#: 0.2026.8.3 releases: UT1-UTC at 2024-01-01, 2025-01-01 and 2026-01-01 is bit-identical, while
#: 2030-01-01 stepped by 2.740 ms. Finalised data ended 2026-07-23 when this was chosen, so this
#: epoch keeps about eighteen months of margin before the prediction boundary, and the margin only
#: grows as the table extends.
#:
#: Not an indefinite bit-stability guarantee, though. IERS does occasionally reprocess historical
#: spans when the C04 solution's reference frame changes -- 2021-2024 data has been revised that
#: way -- so a future release could still move this epoch. What the move buys is removing a
#: *weekly* certainty, not every possibility; a rare reprocessing is what
#: `astropy-iers-data` staying in the recorded package list is for.
#:
#: The cost, stated because it is a real one, and narrower than it first looks: the *reference
#: comparison* no longer runs at the epoch the examples document, so an error that depends on being
#: far outside the IERS table would not show up in a reference diff. The far-future regime is still
#: exercised by unit tests that stay at 2030 -- the continuous-wave orchestration, device-chunk and
#: content-before-segment modules among them -- and those do not churn, because they assert
#: properties rather than compare against stored numbers. So what is lost is reference-level
#: coverage of the clamped regime, not all coverage of it.
_ALIGNED_START = 1419724816.0

#: Entries that cannot be run without fetching from the network, with the reason. These are
#: skipped rather than run against a remote URL. Removing an entry from here requires adding a
#: local fixture for whatever it downloads.
NOT_HERMETIC: dict[str, str] = {
    "noise/glitches/gengli/et_triangle_sardinia/e1": (
        "needs a local blip-glitch population fixture; the example reads one from "
        "sandbox.zenodo.org, whose records are purged"
    ),
}

#: Per-example overrides, deep-merged over the example's own configuration.
#:
#: Only durations, rates, sample counts, seeds and input paths appear here. Anything that would
#: change the code path under test does not belong in an overlay.
_OVERLAYS: dict[str, dict[str, Any]] = {
    "default_config": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 8,
                "total-duration": 8,
            }
        }
    },
    "noise/uncorrelated_gaussian/quick_start": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Kept multi-segment on purpose: this entry exists to cover chunking and per-segment seeds,
    # so collapsing it to one segment would silently retire what it tests.
    "noise/uncorrelated_gaussian/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 4,
                "total-duration": 12,
            }
        }
    },
    "signal/bbh/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Already 16 s at 512 Hz; the seed is pinned centrally, so there is nothing to override.
    "signal/sgwb/et_triangle_sardinia": {},
    # The example pins its population to a commit, so the URL is stable -- but it is still a
    # network fetch, and the local fixture is the same data. `waveform-backend: ripple` is left
    # alone: it is the entire point of this entry.
    #
    # ripple JIT-compiles on first use, so this entry costs seconds where the others cost
    # fractions of one. IMRPhenomD is what the example already selects, and is the cheap case --
    # the precessing models take roughly seven times longer to compile.
    "signal/waveform_backend/ripple": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # A continuous wave is on for the whole run, so there is no event to bracket and nothing to
    # align a start time to -- the only thing to shorten is the span. Kept multi-segment on
    # purpose: this entry exists to cover the branch where a population is never consumed and
    # every source contributes to every segment, and one segment could not distinguish that from
    # the per-event path.
    #
    # The ephemeris the example names is fetched by ripple on first use; CI caches it and verifies
    # it against `tests/data/ephemeris.sha256`. That is why this entry is no longer in
    # NOT_HERMETIC: the tables are obtained once and their content is pinned, rather than being
    # re-downloaded per run or trusted unchecked.
    "signal/cw/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 256,
                "duration": 8,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        # The SSB reference moves with the data. It could stay in 2030 -- it is only the epoch the
        # spin parameters refer to -- but then the phase would be accumulated across five years for
        # no reason, and the run would be describing a catalogue referenced to a time it never
        # covers.
        #
        # Nested under `orchestration`, which is where the configuration reads it. The first
        # version of this put `signal` at the root: the deep merge accepted the stray key without
        # complaint, the effective value stayed at the example's 2030 epoch, and both the override
        # and this comment were false while every test passed.
        # One `orchestration` key, not two. Written as two at first, and Python keeps only the
        # last -- so the reference-epoch override vanished silently and nothing failed.
        "orchestration": {
            "population": {"arguments": {"path": str(CW_POPULATION_FIXTURE)}},
            "signal": {"arguments": {"reference_time_ssb": _ALIGNED_START}},
        },
    },
    # Deliberately the same overlay as the ripple entry above: the two configs differ only by
    # `execution`, and holding the span, rate and population identical is what makes their stored
    # references comparable. The batched path JIT-compiles like the per-event ripple one.
    "signal/execution/batched": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 16,
                "total-duration": 16,
                "start-time": _ALIGNED_START,
            }
        },
        "orchestration": {"population": {"arguments": {"path": str(POPULATION_FIXTURE)}}},
    },
    # Two segments rather than one, so a glitch drawn near a boundary is placed by the same code
    # a full run uses. There is no event to bracket, so the example's 2030 start time is left
    # alone -- glitch generation reads no Earth-orientation table, which is what the epoch choice
    # elsewhere in this module is about.
    #
    # The rate is not overridden here; it is *scaled*, by _GLITCH_RATE_SCALES below. At the
    # example's own O3 rates an 8-second segment expects 0.065 glitches per interferometer, and
    # the whole span 0.13, so nearly every output file would be zeros and the entry would assert
    # nothing about injection at all.
    "noise/glitches/deepextractor/et_triangle_sardinia": {
        "globals": {
            "simulator-arguments": {
                "sampling-frequency": 1024,
                "duration": 8,
                "total-duration": 16,
            }
        }
    },
}

#: Factor applied to an entry's configured glitch rates, by label.
#:
#: Separate from :data:`_OVERLAYS` because a glitch model lives inside a *list* under
#: ``noise.arguments.glitches``, which the deep merge replaces wholesale rather than merging into.
#: An overlay entry that changed the rate would therefore have to restate the whole model --
#: every class name, SNR, PSD and revision -- and would then be free to drift from the example it
#: is supposed to be shortening. Multiplying what the example declares duplicates none of it.
#:
#: Scaling also preserves the *shape* of the configuration, which the overlay contract requires:
#: a per-class mapping stays a per-class mapping with its relative weights intact, so the
#: proportional class draw this entry claims to cover still runs. Rewriting it as a scalar would
#: switch the backend to a uniform draw and quietly retire that coverage.
#:
#: 60 puts the total at 0.486 Hz, about 3.9 glitches per interferometer per 8-second segment --
#: high enough that no output file is plausibly empty, low enough that events remain mostly
#: separated rather than piling into continuous noise.
_GLITCH_RATE_SCALES: dict[str, float] = {
    "noise/glitches/deepextractor/et_triangle_sardinia": 60.0,
}


def _scale_glitch_rates(config: dict[str, Any], scale: float) -> None:
    """Multiply every glitch model's ``rate`` in *config*, in place.

    Handles both forms the backends accept: a scalar total rate and a per-class mapping.
    """
    glitches = config.get("orchestration", {}).get("noise", {}).get("arguments", {}).get("glitches")
    if not isinstance(glitches, list):
        raise KeyError(
            "a glitch rate scale is declared for this entry, but its configuration has no "
            "orchestration.noise.arguments.glitches list to scale."
        )
    for model in glitches:
        rate = model["rate"]
        model["rate"] = (
            {name: value * scale for name, value in rate.items()} if isinstance(rate, dict) else rate * scale
        )


#: Entries whose span is expected to contain a gravitational-wave signal, so an all-zero output
#: is a failure rather than a valid result. Noise-only runs are excluded because their content
#: is noise, and the SGWB run produces a background rather than a located event.
CONTAINS_SIGNAL: frozenset[str] = frozenset(
    {
        "noise/uncorrelated_gaussian/quick_start",
        "signal/bbh/et_triangle_sardinia",
        "signal/waveform_backend/ripple",
        "signal/execution/batched",
        # Every pulsar is present in every segment, so any span contains signal -- unlike the
        # transient entries, whose span has to be aligned to an event.
        "signal/cw/et_triangle_sardinia",
    }
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return *base* with *override* merged into it, recursing into nested mappings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overlay(config: dict[str, Any], label: str, working_directory: Path) -> dict[str, Any]:
    """Return *config* shortened for test use and rooted at *working_directory*.

    Args:
        config: The example configuration, as loaded from its ``config.yaml``.
        label: The example label, used to select the overlay.
        working_directory: Directory the run should write into.

    Returns:
        A new configuration; *config* is not modified.

    Raises:
        KeyError: If no overlay is defined for *label*. Deliberate -- a new matrix entry has to
            state how it is shortened, rather than silently running at full example size.
    """
    if label not in _OVERLAYS:
        raise KeyError(
            f"no test overlay is defined for '{label}'. Add one to tests/e2e/overlay.py; running "
            f"an example at its full size would take minutes to hours."
        )
    merged = _deep_merge(config, _OVERLAYS[label])
    if label in _GLITCH_RATE_SCALES:
        _scale_glitch_rates(merged, _GLITCH_RATE_SCALES[label])
    merged.setdefault("globals", {})["working-directory"] = str(working_directory)
    merged["globals"].setdefault("simulator-arguments", {})["seed"] = _TEST_SEED

    # Override the noise seed rather than defaulting it: the orchestrator's own copy of the
    # global seed is a `setdefault`, so an example that pins its own would otherwise keep it and
    # this would have no effect. See _TEST_SEED.
    noise = merged.get("orchestration", {}).get("noise")
    if isinstance(noise, dict):
        noise.setdefault("arguments", {})["seed"] = _TEST_SEED
    return merged
