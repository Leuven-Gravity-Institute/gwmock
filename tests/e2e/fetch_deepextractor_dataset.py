"""Download the DeepExtractor glitch dataset the end-to-end matrix draws from.

Run by CI before the end-to-end suite, and usable by hand:

    uv run python tests/e2e/fetch_deepextractor_dataset.py

The dataset is 2.3 GB and ``DeepExtractorGlitch`` fetches it lazily, on the first glitch it is
asked for. Left to the test run, a Hub outage therefore surfaces from inside a test as
huggingface_hub's ``LocalEntryNotFoundError`` -- a message about a missing cache entry, attributed
to whichever assertion happened to trigger the load. Fetching it up front turns that into a fetch
failure with retries, said plainly.

Which revision is fetched comes from the example configuration rather than from a constant here.
The example is what the suite runs, so a constant would be a second place for the pin to live and
therefore a place for it to drift: CI would warm the cache for one commit of the dataset and the
run would then fetch another.

Prints the resolved commit SHA on success, and exits non-zero with the reason otherwise.
"""

from __future__ import annotations

import pathlib
import sys

import yaml

#: The matrix entry whose dataset this is. Only one entry uses DeepExtractor, so naming it here is
#: not a duplication of the matrix -- but the label has to match, which the test module asserts.
ENTRY_LABEL = "noise/glitches/deepextractor/et_triangle_sardinia"

_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def glitch_model_configuration(label: str = ENTRY_LABEL) -> dict[str, object]:
    """Return the first glitch model declared by one example configuration."""
    config = yaml.safe_load((_EXAMPLES / label / "config.yaml").read_text(encoding="utf-8"))
    glitches = config["orchestration"]["noise"]["arguments"]["glitches"]
    if not glitches:
        raise ValueError(f"'{label}' declares no glitch models, so there is no dataset to fetch.")
    return glitches[0]


def pinned_revision(label: str = ENTRY_LABEL) -> str:
    """Return the dataset revision the example pins, refusing to fetch an unpinned one.

    An unpinned fetch would warm the cache with whatever the Hub serves now, which is not
    necessarily what the run resolves later -- so it is a configuration error here rather than
    something to paper over with the repository default.
    """
    revision = glitch_model_configuration(label).get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            f"'{label}' does not pin `revision`, so there is no specific dataset version to "
            f"fetch. Pin it in the example, which is also what makes its runs reproducible."
        )
    return revision


def main() -> int:
    """Fetch the dataset and print the commit SHA it resolved to."""
    from gwmock_noise.glitches import DeepExtractorGlitch
    from gwmock_noise.glitches.models import LogNormalAmplitudeDistribution

    model_configuration = glitch_model_configuration()
    # Constructed through the model rather than by calling `hf_hub_download` directly, so the
    # filenames and repository come from the backend that will read them. Only `resolve()` is
    # called, which fetches the samples file -- all but 2 MB of the 2.3 GB; the remaining two come
    # down on the first waveform. `rate` and `snr` are required to construct one and play no part
    # in the download, so they are taken from the example too rather than invented.
    model = DeepExtractorGlitch(
        rate=model_configuration["rate"],
        amplitude_distribution=LogNormalAmplitudeDistribution(),
        psd_file=model_configuration["psd_file"],
        snr=model_configuration["snr"],
        revision=pinned_revision(),
    )
    resolved = model.resolve()
    print(f"DeepExtractor dataset resolved to {resolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
