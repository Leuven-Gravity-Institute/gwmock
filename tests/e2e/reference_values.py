"""Reference values for the end-to-end matrix: building them, and comparing against them.

Each matrix entry has a small JSON file under ``references/`` recording, per output file, a content
hash and a handful of summary statistics. **The statistics are what the tests assert on**; the hash
is recorded and reported but not compared, for the reason set out under portability below.

The statistics also make a failure *diagnosable* rather than merely red. A bare hash tells you
something changed and nothing about what, so every failure becomes a bisect. With the statistics
stored alongside, a mismatch reports which files moved and by how much, and a drift from a library
upgrade looks obviously different from a waveform that shifted in time or changed amplitude.

What this does **not** detect: a change that preserves every gated statistic. Reordering samples, or
altering the shape between the peak and the edges, can leave ``n``, ``nonzero``, ``argmax``,
``peak``, ``rms`` and ``signed_peak`` all intact. Sign inversion is covered, by ``signed_peak``;
arbitrary shape change is not. Closing that needs a quantised digest of the samples, which is future
work rather than something this claims today.

A mismatch is not automatically a bug. When a dependency bump changes the last bits of a
waveform, the right response is to look at the reported differences, decide the change is
harmless, and regenerate. When it changes a peak by a percent, it is not harmless. The stored
environment makes the first case quick to recognise -- it says which versions produced the
reference.

Regenerate with::

    GWMOCK_WRITE_E2E_REFERENCES=1 uv run pytest -m e2e --no-cov

Review the resulting diff before committing it. The references are small JSON in the repository
rather than an opaque archive precisely so that an update shows up as a reviewable change.

**Regenerate on the platform the references already name**, which the ``fingerprint`` block in each
file records. Rewriting them from an architecture whose LAL differs does not fix anything -- it moves
the same 4e-06 onto the machines that were green before, and the diff looks like a real change to
every entry that generates through LAL. The macOS section below is what that is about.

**Portability: measured, and it does not hold.** An earlier run suggested these hashes reproduced
bit-for-bit between a local Linux/x86_64 machine and GitHub's ubuntu-latest runner. That was wrong.
With identical package versions -- numpy, scipy, lalsuite, jax and ripplegw all unchanged -- CI and
local now disagree in the last bits. The observable difference is confined to ``mean``, at 1e-16 to
1e-12 relative, while ``peak``, ``rms``, ``argmax`` and ``nonzero`` are identical: floating-point
reassociation, not a change in the data.

The fingerprint cannot separate those two environments, because both are Linux/x86_64 on the same
Python. Widening it to CPU or BLAS identity is not something this can do reliably.

So the exact hash is no longer the assertion. It is recorded and reported, but what *fails* a run is
a statistic moving beyond :data:`STATISTIC_TOLERANCE` -- integer statistics exactly, float
magnitudes within tolerance. That is weaker than a bit-for-bit check and the weakening is real: a
change of one sample in one channel now passes. What it buys is a check that means the same thing on
every machine, rather than one that is green only where the references happened to be generated.

``mean`` is excluded from the gate. It is a sum, so its last bits depend on summation order rather
than on the data -- the reason sums were left out of the statistics in the first place, before it was
added back to catch sign inversions. It stays as a diagnostic, where being sensitive is useful.

**One reference, both devices -- and how that was established.** These references are written on a CPU,
and the entries that generate through JAX (`execution: batched`, the ripple backend, the CW branch) run
on a GPU wherever one is present. That makes "does the GPU still match?" a real question, and it is
answered by replaying the matrix on a CUDA host rather than by argument:

    # on a host with a CUDA-capable GPU and the `cuda` extra installed
    uv run pytest -m e2e --no-cov tests/e2e/test_reference_values.py

Done on 2026-08-09 against an RTX 5060 Ti (compute capability 12.0): **all eight stored references
matched**, `argmax` included -- which is compared exactly, so the device difference moved no peak off its
sample. (The run reported "9 passed, 1 skipped" at the time: eight reference comparisons, plus the separate
test that every runnable entry has a reference, with the gengli entry skipped. An earlier draft of this
paragraph read that as nine entries, which a reviewer corrected. A second non-hermetic entry has been
added since, so the same run now reports two skips and still eight comparisons.) An earlier measurement on an RTX 2080 Ti agreed. The delta
itself was characterised separately as a global time shift of 2.3e-16 s, 3.16e-13 relative end to end.

**Where the gate actually sits.** Two effects were measured against it, and it falls between them:

===========================================================  ==========  =====================
difference                                                   relative    against a 1e-06 gate
===========================================================  ==========  =====================
CPU to GPU, one Linux host                                   3.16e-13    passes, 3e6 to spare
Linux/x86_64 to Darwin/arm64, LAL entries                    4.17e-06    **fails, by 4x**
Linux/x86_64 to Darwin/arm64, every other entry              8.0e-13     passes, 1e6 to spare
py3.12 to py3.13, either platform                            0           bit-identical
===========================================================  ==========  =====================

So the tolerance is not simply loose: it admits the device difference and rejects a platform change. What
it does *not* do is catch a GPU-specific regression smaller than about 1e-06, which is seven orders above
the device difference -- passing the GPU replay says these references survive the device, not that the
device is tightly watched.

**What the macOS row actually is.** It was recorded here as a whole-platform gap -- "this suite cannot
be run green on an Apple machine" -- and that was wrong in the way that matters: it is confined to the
two entries whose samples come out of LALSimulation, and it is not the Python version at all. Measured
by replaying the matrix on both platforms at both Python versions, and by dumping the pipeline stage by
stage:

* **The Python version contributes nothing.** Linux/x86_64 py3.12 against py3.13 is *bit-identical* on
  all 37 output files, and so is Darwin/arm64 py3.12 against py3.13. The whole 4.17e-06 is the platform.
* **Six of the eight entries reproduce across the platforms**, at 8.0e-13 and below -- the noise-only
  entries and the stochastic one at 1.9e-16, about one ulp. Only
  ``noise/uncorrelated_gaussian/quick_start`` (3.90e-06) and ``signal/bbh/et_triangle_sardinia``
  (4.17e-06) fail, and those are exactly the two that generate through LAL.
* **The difference is already in LAL's own output**, before numpy touches it: the raw ``IMRPhenomXPHM``
  frequency-domain polarizations differ by 2.0e-07 rms (5.1e-07 peak) between the two builds. Conditioning
  to the time domain conserves that rms and merely redistributes it, concentrating the *maximum* to 1.3e-05.
  Sidereal time over the same span -- astropy and pyerfa, no LAL -- differs by 1.6e-14.
* **The control that isolates it**: the ripple backend is an independent implementation of the same
  physics running through the *same* projection, resampling and writer code, and it reproduces across the
  platforms at 1.5e-14. Same downstream code, different waveform generator, eight orders of magnitude
  better agreement. Neither build is thereby shown *correct* -- this compares two builds of one version
  against each other, not against an external reference -- but it locates where they part.
* Every integer statistic survives, on all 37 files. ``argmax`` is compared exactly and did not move, so
  no peak changed sample.

So the rule this suite applies is not "skip macOS". Entries flagged
:attr:`~tests.e2e.matrix.MatrixEntry.lal_waveform` are compared at :data:`FOREIGN_PLATFORM_TOLERANCE`
when the reference is *established* to have been written on a different system or machine -- see
:func:`comparison_bound`, which decides that and is the only place the choice is made -- and at the
full :data:`STATISTIC_TOLERANCE` on the platform that wrote it, or whenever the platforms cannot be
compared at all; every other entry is gated at 1e-06 everywhere. That keeps the six portable entries under the tight gate on an Apple machine instead of
discarding coverage that demonstrably works, and keeps the LAL pair checked at a bound that still
rejects a waveform regression while admitting a difference that is the library's, not the project's.

**Measured on one Mac.** The 4.17e-06 comes from a single machine -- macOS 26.5.2, arm64, the PyPI
``lalsuite`` 7.26.15 ``macosx_12_0_arm64`` wheel. Whether an x86_64 Mac, another macOS release, or a
Linux/aarch64 host lands at the same figure is *not* measured, and :data:`FOREIGN_PLATFORM_TOLERANCE`
carries margin for that reason rather than because a spread was observed.

This check also cannot run in CI: GitHub-hosted runners have no GPU. Repeat it when the waveform path or
the JAX dependency changes, rather than expecting it to guard every pull request.

Note what the fingerprint deliberately does *not* include: package versions. A dependency bump
changing a waveform is precisely what these references exist to surface, so putting versions in the
gate would silently downgrade the very comparison that is wanted. Versions are recorded, and
reported when a comparison fails, but they do not excuse a difference.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from gwmock.cli.utils.hash import compute_content_hash

REFERENCES_DIR = Path(__file__).resolve().parent / "references"

#: Environment variable that switches the tests from comparing to writing.
WRITE_ENVIRONMENT_VARIABLE = "GWMOCK_WRITE_E2E_REFERENCES"

#: Packages whose versions are recorded with a reference. These are the ones whose output could
#: plausibly change a waveform or a noise realisation, so they are the first things to look at
#: when a comparison fails.
_RECORDED_PACKAGES = (
    "gwmock",
    "gwmock-signal",
    "gwmock-noise",
    "gwmock-pop",
    "numpy",
    "scipy",
    "lalsuite",
    "ripplegw",
    "jax",
    # Astropy supplies sidereal time to the projection, and `astropy-iers-data` is the Earth
    # orientation table behind it. These are the *most frequent* reason this comparison moves, so
    # they belong here: the table is republished weekly.
    #
    # Added after a lock-file bump moved a BBH peak by 1.569e-06 and the report said "no recorded
    # package version changed, so a dependency bump does not explain this" -- pointing the next
    # reader at a code regression that did not exist.
    #
    # What used to move, and why it no longer does. The shipped examples run at GPS 1577491296 --
    # 2030-01-01, about 885 days *beyond* the end of the packaged IERS table, where Astropy clamps
    # UT1-UTC to the final tabulated value. What changed each week was not a revised measurement
    # for 2030 but the clamped edge following the table's new
    # end date: -0.044955 s to -0.047695 s, a step of 2.740 ms. That rotates GMST by 1.9976e-07
    # rad, matching 2.740e-3 s x 7.2921e-5 rad/s to 0.02%.
    #
    # Because the epoch rides the table edge, this churn is structural rather than incidental: a
    # test epoch *inside* the table would move only when a finalised value is revised. That is a
    # design question, not something to fix by widening a tolerance.
    "astropy",
    "astropy-iers-data",
    # The C library behind Astropy's time and coordinate transforms; a change here moves sidereal
    # time for the same reason astropy itself does.
    "pyerfa",
    # The implementation actually executing JAX computations on the device path.
    "jaxlib",
)

#: Why this is a curated list rather than the full environment.
#:
#: `gwmock.cli.utils.environment.capture_environment` records *every* installed distribution, and a
#: reference could store that instead -- which would make this diagnostic exhaustive by
#: construction and remove the class of false negative that motivated adding astropy above.
#:
#: The cost is churn: every reference file would be rewritten whenever any development tool moved,
#: which buries a real numerical change in noise and trains readers to skim the diff. So the list
#: is deliberately confined to packages whose output can reach generated data. When a comparison
#: fails and nothing here explains it, the run's own metadata carries the complete environment.
_RECORDED_PACKAGES_ARE_CURATED = True


#: Relative tolerance applied to ``peak`` and ``rms`` when the exact comparison is not available.
#:
#: **This number is not measured.** It is chosen to sit far below any physically meaningful change
#: -- a different waveform, a rescaled PSD, a lost taper all move these by percent or more -- and
#: far above the last-bit differences a different BLAS or libm produces for the same algorithm.
#: Nobody has measured how far apart two platforms actually land, so treat it as a bound on what
#: this check can detect rather than as a statement about numerical agreement.
#:
#: A weekly `astropy-iers-data` release used to exceed this bound, reaching 1.569e-06 on one
#: detector, because the examples ran at a 2030 epoch beyond the end of the Earth-orientation
#: table where Astropy clamps UT1-UTC to the final tabulated value -- so every release moved that
#: clamp and the generated strain with it. That is fixed at the cause: the suite now runs inside
#: the finalised part of the table (see ``_ALIGNED_START`` in ``overlay.py``), where the value is
#: bit-identical across releases.
#:
#: Left at 1e-06 rather than widened, and the distinction matters. Raising it to sit above routine
#: Earth-orientation movement would also have stopped it detecting a projection regression of the
#: same size, which is what it exists to catch.
#:
#: What can still move output through this route is a *revised finalised* value, which IERS does
#: occasionally publish. That is rare and small rather than weekly, and it is why
#: `astropy-iers-data` stays in :data:`_RECORDED_PACKAGES`: the churn is gone, the dependency is
#: not.
STATISTIC_TOLERANCE = 1e-6

#: Relative tolerance for an entry flagged :attr:`~tests.e2e.matrix.MatrixEntry.lal_waveform`
#: when the stored reference was written on a different system or machine.
#:
#: **Measured, then given margin.** Replaying the matrix on Darwin/arm64 against these
#: Linux/x86_64 references moves the LAL entries by 4.17e-06 and 3.90e-06 of peak, while every
#: entry that does not generate through LAL stays at 8.0e-13 or below and the Python version
#: moves nothing at all. The difference is present in LAL's own frequency-domain output before
#: any of this project's code runs, so it is the library's and not something a gate here can
#: catch or fix. The macOS section of the module docstring carries the full measurement.
#:
#: Set at 1e-04, which is 24x the largest figure observed. The margin is *not* an observed
#: spread: it was measured on one Mac, and nothing here establishes what an x86_64 Mac, another
#: macOS release, or a Linux/aarch64 host would give. It is sized so that a plausible variation
#: between LAL builds does not turn into a red suite, while a real waveform regression -- which
#: moves these by percent, two orders above this -- still fails.
#:
#: This is deliberately *not* applied to every entry. Widening the gate globally would cost the
#: six entries that reproduce across platforms at 1e-13 and better the sensitivity they
#: currently have, to buy portability for two; and :data:`STATISTIC_TOLERANCE` was already held
#: at 1e-06 on purpose, so that it would keep catching a projection regression of the same size
#: as the Earth-orientation churn that used to trip it.
FOREIGN_PLATFORM_TOLERANCE = 1e-4


def writing_references() -> bool:
    """Return whether this run should write references instead of comparing against them.

    Raises:
        RuntimeError: If write mode is requested while ``CI`` is set. Regenerating references is a
            deliberate local act; a continuous-integration run that quietly rewrote them would
            overwrite the very thing it is supposed to be checking against and then report
            success.
    """
    requested = os.environ.get(WRITE_ENVIRONMENT_VARIABLE, "").strip().lower() in {"1", "true", "yes"}
    # Presence, not value: CI systems spell it CI=true, CI=1, and occasionally CI=. Failing closed
    # on any setting is right for a mode whose accidental use would overwrite the comparison.
    if requested and "CI" in os.environ:
        raise RuntimeError(
            f"{WRITE_ENVIRONMENT_VARIABLE} is set while CI is set. Reference values must not be "
            f"regenerated by a CI run: it would overwrite the comparison and then pass. Generate "
            f"them locally, review the diff, and commit it."
        )
    return requested


def fingerprint() -> dict[str, str]:
    """Return what distinguishes this numerical environment from another.

    Exact content hashes are only a fair comparison between runs on the *same* numerical stack.
    Package versions do not capture that: the same numpy and LAL built against a different BLAS,
    or on a different architecture, can produce different last bits for identical inputs. So the
    reference records a fingerprint, and the comparison uses it to decide whether an exact match
    is a reasonable thing to demand.
    """
    import platform
    import sys

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(str(part) for part in sys.version_info[:2]),
        # The compute device, because two runs on one machine can differ by it alone. `execution:
        # batched` generates through JAX, so the same host produces different last bits on its CPU and
        # its GPU -- measured end to end at 3.16e-13 relative, a global time shift of 2.3e-16 s, which is
        # benign and far inside `STATISTIC_TOLERANCE`. Without this the two runs fingerprint identically
        # and the bit-mismatch note below blames "references written somewhere subtly different" for what
        # is simply the other device.
        "device": _jax_device(),
    }


def _jax_device() -> str:
    """Return the device this run generates on.

    The device, not JAX's availability -- a distinction a reviewer had to draw. With no JAX installed the
    numerics still run on the CPU, on the same device the references were written on, and the entries
    that need `ripplegw` are skipped rather than run differently. Reporting "none" there described the
    library instead of the hardware, so CI's no-`jax` cell read as a foreign environment for every
    reference and silenced the bit-mismatch note on entries that never touch JAX.

    ``"unavailable"`` stays its own state, because that environment is *broken* rather than CPU-only: a
    CUDA plugin that raises has JAX-using entries failing, not falling back.
    """
    try:
        import jax
    except ImportError:
        return "cpu"
    try:
        return str(jax.default_backend())
    except Exception:  # pragma: no cover - a broken backend must not fail the comparison
        # A CUDA plugin that cannot see a supported GPU raises rather than falling back, and that is
        # worth recording as its own state instead of crashing a test run that would otherwise pass.
        return "unavailable"


def same_environment(stored: dict[str, str] | None, produced: dict[str, str] | None) -> bool:
    """Whether two fingerprints describe the same numerical environment.

    Every key, in both records. An earlier version compared only the keys the *stored* record carried, so
    that references written before a key existed would keep behaving as they had -- and that defeated the
    point of adding one: a CPU reference replayed on a GPU still counted as the same environment, and the
    bit-mismatch note still blamed "references written somewhere subtly different" for the device. Both
    reviewers caught it. A note that misattributes is worse than a note that is missing, and the
    references now carry `device` anyway, so there is nothing to be compatible with.

    Args:
        stored: The fingerprint recorded with the reference, if any.
        produced: The fingerprint of this run, if any.

    Returns:
        Whether the two describe the same environment. A reference with no fingerprint at all is not the
        same environment as anything, since there is nothing to compare.
    """
    if not stored or not produced:
        return False
    return stored == produced


#: The fingerprint keys that decide whether two runs are on the same platform.
#:
#: Narrower than what :func:`same_environment` compares, and the narrowing is the measurement:
#: ``python`` and ``device`` do not belong here. Both Python versions produce bit-identical output
#: on either platform, and the device difference is 3.16e-13 -- so letting either answer this
#: question would relax the LAL entries' bound for a reason known not to move them, which is
#: exactly the false excuse a gate must not accept.
_PLATFORM_KEYS = ("system", "machine")


def known_different_platform(stored: dict[str, str] | None, produced: dict[str, str] | None) -> bool:
    """Whether two fingerprints *positively establish* that the platforms differ.

    Phrased as the positive claim, and defaulting to ``False``, because this is what widens a
    tolerance: the loose bound should need evidence that the platform really changed, not merely
    the absence of evidence that it did not. An earlier version asked the opposite question --
    "are these the same platform?" -- and took the loose branch on ``not same_platform(...)``,
    which handed every flagged entry the wider bound whenever a fingerprint was missing or
    incomplete, *including on the machine that wrote the reference*. Every reference in the tree
    carries a full fingerprint today, so nothing was actually widened; it was a latent widening
    waiting for the first reference that omitted one, and a reviewer caught it.

    So anything short of a complete comparison answers ``False`` and the caller keeps the tight
    gate. A reference that cannot say where it was written gets held to the strict bound, and a
    strict bound failing on a foreign platform is a visible prompt to look -- which is the
    direction this should fail in.

    Args:
        stored: The fingerprint recorded with the reference, if any.
        produced: The fingerprint of this run, if any.

    Returns:
        Whether both fingerprints name a ``system`` and a ``machine`` and at least one differs.
    """
    if not stored or not produced:
        return False
    if not all(stored.get(key) and produced.get(key) for key in _PLATFORM_KEYS):
        return False
    return any(stored[key] != produced[key] for key in _PLATFORM_KEYS)


def comparison_bound(
    lal_waveform: bool, stored: dict[str, str] | None, produced: dict[str, str] | None
) -> tuple[float, str | None]:
    """Return the tolerance this comparison runs at, and why, if it is not the usual one.

    One function rather than a condition spelled out at the call site, for two reasons a review
    drew out. It makes the conjunction -- a flagged entry *and* an established platform change,
    both required -- directly testable without a generation run. And it hands back the
    explanation alongside the number, so the caller cannot report the relaxed bound without the
    reason for it, or report the reason in a place a failure would skip past.

    Args:
        lal_waveform: Whether the entry generates through LALSimulation.
        stored: The fingerprint recorded with the reference, if any.
        produced: The fingerprint of this run, if any.

    Returns:
        The relative tolerance to apply, and a phrase naming why it was relaxed -- or ``None``
        when it was not, which is every case except a flagged entry on an established different
        platform.
    """
    if not (lal_waveform and known_different_platform(stored, produced)):
        return STATISTIC_TOLERANCE, None
    return FOREIGN_PLATFORM_TOLERANCE, (
        f"it generates through LAL, whose output does not reproduce across architectures, and "
        f"this run is on {produced['system']}/{produced['machine']} against a reference written "
        f"on {stored['system']}/{stored['machine']}"
    )


def _environment() -> dict[str, str]:
    """Return the installed versions of the packages that could change the data."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in _RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def summarise(samples: np.ndarray) -> dict[str, Any]:
    """Return diagnostic statistics for one file's samples.

    Chosen so that the *kind* of a change is often readable from the numbers: ``peak`` and ``rms``
    move when amplitudes change, ``argmax`` when something shifts in time, ``nonzero`` when a
    signal appears or vanishes, and ``signed_peak`` when signs invert.

    These triage a difference; they do not explain every one. A phase change that preserves the
    amplitude envelope, or a permutation of channels within a file, can leave all of them
    untouched -- the hash still catches such a change, but the report will say only that the
    recorded statistics are unchanged. That is a limit of the explanation, not of the detection.
    """
    finite = np.asarray(samples, dtype=float)
    return {
        "n": int(finite.size),
        "nonzero": int(np.count_nonzero(finite)),
        "peak": float(np.max(np.abs(finite))) if finite.size else 0.0,
        "rms": float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0,
        "argmax": int(np.argmax(np.abs(finite))) if finite.size else 0,
        # Signed, so an inversion shows up: `peak` and `rms` are magnitudes and would not move at
        # all if every sample changed sign. Taken at the peak rather than as a mean because it is
        # then O(peak) instead of near zero -- an inversion moves it by twice the peak -- and
        # because it is a single sample rather than a sum, so it carries no summation-order noise.
        "signed_peak": float(finite[int(np.argmax(np.abs(finite)))]) if finite.size else 0.0,
        # Kept for diagnosis only. It is a sum, so its last bits track summation order rather than
        # the data, which is why it is not part of the gate below.
        "mean": float(np.mean(finite)) if finite.size else 0.0,
    }


def build_reference(label: str, working_directory: Path, files: list[Path]) -> dict[str, Any]:
    """Return a reference document describing what a run produced.

    Args:
        label: The matrix entry's label.
        working_directory: The run directory, used to make paths relative.
        files: The data files the run wrote.

    Returns:
        A JSON-serialisable reference document.
    """
    from .runner import samples

    outputs: dict[str, Any] = {}
    for path in sorted(files):
        outputs[str(path.relative_to(working_directory))] = {
            "content_hash": compute_content_hash(path),
            **summarise(samples(path)),
        }
    return {
        "label": label,
        "environment": _environment(),
        "fingerprint": fingerprint(),
        "outputs": outputs,
    }


def reference_path(label: str) -> Path:
    """Return the file storing *label*'s reference, with slashes flattened for a filename."""
    return REFERENCES_DIR / f"{label.replace('/', '__')}.json"


def write_reference(document: dict[str, Any]) -> Path:
    """Write *document* and return where it went."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = reference_path(document["label"])
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_reference(label: str) -> dict[str, Any] | None:
    """Return *label*'s stored reference, or ``None`` if there is not one yet."""
    path = reference_path(label)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def statistic_differences(
    stored: dict[str, Any], produced: dict[str, Any], tolerance: float = STATISTIC_TOLERANCE
) -> list[str]:
    """Return the statistics that differ beyond what floating-point drift explains.

    Used instead of the hash when the two runs are not on the same numerical stack, where an exact
    match is not a fair thing to require. Integer statistics -- sample count, occupancy, the peak's
    position -- must match exactly, because no amount of last-bit drift moves them, and they are
    held exactly whatever *tolerance* says: the measured platform difference moved none of them.
    The float magnitudes are compared with *tolerance*.

    Args:
        stored: The reference's per-file statistics.
        produced: This run's per-file statistics.
        tolerance: Relative bound on ``peak``, ``rms`` and ``signed_peak``. Defaults to
            :data:`STATISTIC_TOLERANCE`; the caller passes :data:`FOREIGN_PLATFORM_TOLERANCE`
            for a LAL-generated entry compared off the platform that wrote its reference.
    """
    differences: list[str] = []
    for name in sorted(set(stored) & set(produced)):
        before, after = stored[name], produced[name]
        for key in ("n", "nonzero", "argmax"):
            if before.get(key) != after.get(key):
                differences.append(f"{name}: {key} {before.get(key)!r} -> {after.get(key)!r}")
        # `mean` is deliberately absent: it is a sum, so its last bits track summation order rather
        # than the data, and it is the statistic that moved between machines. `signed_peak` covers
        # what `mean` was added for -- a full sign inversion -- without that noise, since it is a
        # single sample of magnitude `peak` rather than a near-zero sum.
        for key in ("peak", "rms", "signed_peak"):
            old, new = float(before.get(key, 0.0)), float(after.get(key, 0.0))
            # A NaN makes every comparison below false, so it would be read as "no difference".
            # Non-finite values are the difference.
            if not (math.isfinite(old) and math.isfinite(new)):
                differences.append(f"{name}: {key} {old!r} -> {new!r} (not finite)")
                continue
            scale = max(abs(old), abs(new))
            if scale and abs(new - old) / scale > tolerance:
                differences.append(f"{name}: {key} {old!r} -> {new!r} (relative {abs(new - old) / scale:.3e})")
    return differences


def describe_difference(stored: dict[str, Any], produced: dict[str, Any]) -> str:
    """Return a human-readable account of how a run differs from its reference.

    Written to answer the question a failure actually raises -- *is this harmless?* -- so it
    reports the statistics side by side with a relative change, and the environment differences,
    rather than only saying that two hashes are unequal.
    """
    lines: list[str] = []

    stored_outputs, produced_outputs = stored.get("outputs", {}), produced.get("outputs", {})
    missing = sorted(set(stored_outputs) - set(produced_outputs))
    extra = sorted(set(produced_outputs) - set(stored_outputs))
    if missing:
        lines.append(f"  files in the reference but not produced: {missing}")
    if extra:
        lines.append(f"  files produced but not in the reference: {extra}")

    for name in sorted(set(stored_outputs) & set(produced_outputs)):
        before, after = stored_outputs[name], produced_outputs[name]
        if before.get("content_hash") == after.get("content_hash"):
            continue
        lines.append(f"  {name}:")
        for key in ("n", "nonzero", "argmax", "peak", "rms", "signed_peak", "mean"):
            old, new = before.get(key), after.get(key)
            if old == new:
                continue
            if isinstance(old, float) and isinstance(new, float) and old != 0.0:
                relative = abs(new - old) / abs(old)
                lines.append(f"    {key}: {old!r} -> {new!r}  (relative change {relative:.3e})")
            else:
                lines.append(f"    {key}: {old!r} -> {new!r}")
        if all(
            before.get(key) == after.get(key)
            for key in ("n", "nonzero", "argmax", "peak", "rms", "signed_peak", "mean")
        ):
            lines.append("    every recorded statistic is unchanged; the difference is below them")

    changed_environment = {
        name: (stored.get("environment", {}).get(name), produced.get("environment", {}).get(name))
        for name in sorted(set(stored.get("environment", {})) | set(produced.get("environment", {})))
        if stored.get("environment", {}).get(name) != produced.get("environment", {}).get(name)
    }
    if changed_environment:
        lines.append("  versions that differ from when the reference was written:")
        lines.extend(f"    {name}: {old} -> {new}" for name, (old, new) in changed_environment.items())
    else:
        lines.append("  no recorded package version changed, so a dependency bump does not explain this")

    return "\n".join(lines) if lines else "  no differences found"
