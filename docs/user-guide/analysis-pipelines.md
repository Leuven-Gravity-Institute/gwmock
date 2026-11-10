# Analysing Data with a Duty Cycle

A run with [`orchestration.duty-cycle`](configuration.md#duty-cycle) marks when
each interferometer observes, in the forms the common analysis pipelines read.
This page shows how each pipeline reads them. Every example on it is run by
gwmock's tests against a gwmock run and against the pipeline's current release,
so a change on either side is caught there before it reaches a user.

## The rule

Each interferometer's **science segments** are the whole GPS seconds inside its
observing intervals: up to a second at each end of an interval is outside them,
and an interval shorter than a second gives none. The strain is NaN outside them
(unless `mask-strain` is off). NaN spreads through filtering: an FFT-based step,
such as whitening or a PSD estimate, over a stretch with one NaN sample gives
NaN throughout, and a time-domain filter spreads it over its length. So every
window a pipeline reads, including its padding and the data it estimates a PSD
from, has to lie inside one science segment. This is the rule open data follows
too. A pipeline that ignores the segments meets NaN and stops, instead of
analysing time the duty cycle marks as not analysable. The exception is a
pipeline that filters continuously and selects time with the state vector, such
as PyCBC's low-latency search: it needs the strain in full (below).

## The files

The examples read a run whose duty-cycle files are in `output/duty-cycle/` and
whose strain is merged with the state vector into one frame file per
interferometer and frame:

```bash
mkdir -p merged
for i in 0 1 2; do                                   # the run's three frames
    t0=$((1000000000 + 1024 * i))
    for ifo in ET1_EMR ET2_EMR ET3_EMR; do
        gwmock merge "output/noise/E-${ifo}_STRAIN-${t0}-1024.gwf" \
            "output/duty-cycle/E-${ifo}_STATE-${t0}-1024.gwf" \
            --metadata "metadata/orchestration-${i}.metadata.json" \
            --channel "${ifo}:STRAIN" --copy-channel "${ifo}:STATE_VECTOR" \
            --output "merged/E-${ifo}_STRAIN-${t0}-1024.gwf"
    done
done
```

Add the signal file of each frame to the merge when the run injects signals. The
noise and duty-cycle outputs are named as in this configuration:

```yaml
orchestration:
    noise:
        output:
            output_directory: noise
            file_name:
                E-{{ detectors }}_STRAIN-{{ start_time }}-{{ duration }}.gwf
            arguments:
                channel: '{{ detectors }}:STRAIN'
    duty-cycle:
        output:
            output_directory: duty-cycle
            file_name:
                E-{{ detectors }}_STATE-{{ start_time }}-{{ duration }}.gwf
```

| File                                             | What it holds                                         | Read by                   |
| ------------------------------------------------ | ----------------------------------------------------- | ------------------------- |
| `output/duty-cycle/duty-cycle-segments.xml`      | LIGO_LW XML: `<ifo>:science:1`, `<ifo>:maintenance:1` | PyCBC (offline), gwpy     |
| `output/duty-cycle/<ifo>-observing-segments.txt` | the science segments, `start end` per line            | cWB, gwpy, bilby          |
| `<ifo>:STATE_VECTOR` in `merged/`                | bits 0-2 set in the science segments                  | PyCBC (low latency), gwpy |
| `<ifo>:STRAIN` in `merged/`                      | the strain, NaN outside the science segments          | every pipeline            |

## gwpy

The segments, three ways, all giving the same science segments; then the strain
of each segment, read and whitened:

<!-- example: gwpy -->

```python
from pathlib import Path

import numpy as np
from gwpy.segments import DataQualityDict, SegmentList
from gwpy.timeseries import StateVector, TimeSeries

ifo = "ET1_EMR"
frames = sorted(str(path) for path in Path("merged").glob(f"E-{ifo}_STRAIN-*.gwf"))

# From the XML file (needs igwn-ligolw), the text file, and the state vector.
from_xml = DataQualityDict.read("output/duty-cycle/duty-cycle-segments.xml", format="ligolw")
science = from_xml[f"{ifo}:science:1"].active
from_text = SegmentList.read(f"output/duty-cycle/{ifo}-observing-segments.txt", format="segwizard")
state = StateVector.read(frames, f"{ifo}:STATE_VECTOR", bits=["HOFT_OK", "OBS_INTENT", "OBS_READY"])
flags = state.to_dqflags()
from_state = (flags["HOFT_OK"] & flags["OBS_INTENT"]).active

assert science == from_text == from_state

for start, end in science:
    if end - start < 64:  # too short to whiten with 4 s FFTs
        continue
    strain = TimeSeries.read(frames, f"{ifo}:STRAIN", start=start, end=end)
    whitened = strain.whiten(4, 2)
    assert np.isfinite(whitened.value).all()
```

## PyCBC, offline

`pycbc_make_offline_search_workflow` reads the science segments from the XML
file, as `<ifo>:science`, when the workflow configuration says so. Put the two
options in a `[workflow-segments-science]` section: in `[workflow-segments]`
they would apply to the veto and data-quality segments too, which would then
need files of their own. The workflow moves the file into its own output
directory, so give it a copy.

<!-- example: pycbc-offline-ini -->

```ini
[workflow-segments-science]
segments-source = file
segments-science-file = /path/to/a/copy/of/duty-cycle-segments.xml
```

The segments are in whole seconds, as the workflow requires, and each job reads
its data, padding included, inside one of them. What a job then does with its
data:

<!-- example: pycbc-offline -->

```python
from pathlib import Path

import numpy as np
from pycbc.frame import read_frame
from pycbc.workflow.segment import SegFile

ifo = "ET1_EMR"
frames = sorted(str(path) for path in Path("merged").glob(f"E-{ifo}_STRAIN-*.gwf"))

segments = SegFile.from_segment_xml("output/duty-cycle/duty-cycle-segments.xml").segment_dict
for segment in segments[f"{ifo}:science"]:
    start, end = int(segment[0]), int(segment[1])
    assert (start, end) == (segment[0], segment[1])  # whole seconds
    if end - start < 64:
        continue
    strain = read_frame(frames, f"{ifo}:STRAIN", start_time=start, end_time=end)
    whitened = strain.highpass_fir(15, 512).whiten(4, 4)
    assert np.isfinite(whitened.numpy()).all()
```

## PyCBC, low latency

`pycbc_live` reads the state vector and analyses a block only when the flags it
is given are all set. It filters continuously, though: the filter and the PSD of
the first block after a gap reach back into the gap, so NaN there would reach
the analysed data and, through the PSD, every block after it. So generate the
data for it with the strain in full, `mask-strain: false` in
`orchestration.duty-cycle`, as LIGO's low-latency frames carry it: the state
vector then decides what is analysed, and a block whose state is not valid is
replaced by zeros and never analysed. Give it the merged frames and:

<!-- example: pycbc-live -->

```bash
pycbc_live --frame-src 'merged/*.gwf' --channel-name ET1_EMR:STRAIN \
    --state-channel ET1_EMR:STATE_VECTOR \
    --analyze-flags ET1_EMR:HOFT_OK,SCIENCE_INTENT
```

It needs both `--state-channel` and `--analyze-flags`: with only one of them it
reads no state vector. The flags of one interferometer go in one comma-separated
value; repeating the interferometer keeps only the last of them.

## cWB

cWB reads two-column segment lists. Give each interferometer a data quality file
of category `CWB_CAT0`, not inverted, in two columns (the last field, `false`),
in `user_parameters.C`:

<!-- example: cwb -->

```cpp
nDQF = 3;
dqfile dqf[3] = {
    {"ET1_EMR", "output/duty-cycle/ET1_EMR-observing-segments.txt", CWB_CAT0, 0., false, false},
    {"ET2_EMR", "output/duty-cycle/ET2_EMR-observing-segments.txt", CWB_CAT0, 0., false, false},
    {"ET3_EMR", "output/duty-cycle/ET3_EMR-observing-segments.txt", CWB_CAT0, 0., false, false},
};
for (int n = 0; n < nDQF; n++) DQF[n] = dqf[n];
```

cWB intersects the interferometers' lists, rounds each segment inward to whole
seconds and builds its jobs with their edges inside it (`CWB::Toolbox`'s
`readSegList`, `mergeSegLists` and `getJobList`, in its O4 release), so a job
reads only observing data. It stops on an empty file, which is what an
interferometer that never observes in the run gets; leave it out of the network.
cWB is not published on PyPI, so gwmock's tests check this with a
reimplementation of its segment reader and job builder rather than with cWB
itself.

## bilby

bilby reads neither the state vector nor the segments: choose each analysis time
so that its data, and the data its PSD is estimated from, lie inside one science
segment of every interferometer used. With `bilby_pipe`, leave
`ignore-gwpy-data-quality-check` at its default, true: the check asks a segment
server about `<ifo>:DMT-SCIENCE:1`, which knows nothing about these data.

<!-- example: bilby -->

```python
from pathlib import Path

import bilby
import numpy as np
from gwpy.segments import SegmentList
from gwpy.timeseries import TimeSeries

ifo = "ET1_EMR"
frames = sorted(str(path) for path in Path("merged").glob(f"E-{ifo}_STRAIN-*.gwf"))
science = SegmentList.read(f"output/duty-cycle/{ifo}-observing-segments.txt", format="segwizard")

duration, post_trigger, psd_duration = 4, 2, 32 * 4
segment = next(s for s in science if abs(s) >= psd_duration + duration)
start = segment[0] + psd_duration  # the analysed data
trigger_time = start + duration - post_trigger

strain = TimeSeries.read(frames, f"{ifo}:STRAIN", start=start, end=start + duration)
psd_data = TimeSeries.read(frames, f"{ifo}:STRAIN", start=start - psd_duration, end=start)
psd = psd_data.psd(fftlength=duration, overlap=0, window=("tukey", 0.4), method="median")

interferometer = bilby.gw.detector.InterferometerList(["ET"])[0]
interferometer.set_strain_data_from_gwpy_timeseries(strain)
interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=psd.frequencies.value, psd_array=psd.value
)
assert np.isfinite(interferometer.strain_data.frequency_domain_strain).all()
assert np.isfinite(interferometer.power_spectral_density_array[1:]).all()
```
