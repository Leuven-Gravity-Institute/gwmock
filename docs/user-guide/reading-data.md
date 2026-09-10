# Reading Data

This guide explains how to read and work with the data gwmock generates.

We use the [GWpy](https://gwpy.github.io/) Python package for these examples.
For more details, refer to the
[GWpy documentation](https://gwpy.github.io/docs/stable/).

## Which format you get

**HDF5 is gwmock's primary output format**, and what a run writes unless you ask
for something else. **GWF (frame files) remain fully supported**, and exist so
that other gravitational-wave pipelines can read what gwmock produces.

The format is chosen by the **file extension** in your configuration, not by a
separate setting:

```yaml
orchestration:
    signal:
        output:
            # HDF5, the default
            file_name:
                'signal-{{ detectors }}-{{ start_time }}-{{ duration }}.hdf5'
            # ... or GWF, for a pipeline that reads frames
            # file_name: 'E-{{ detectors }}_STRAIN-{{ start_time }}-{{ duration }}.gwf'
```

Everything below works for either: GWpy reads both, and the channel is named the
same way in each.

## What the file says it is

Every HDF5 file gwmock writes -- noise, signal, and the output of `gwmock merge`
-- declares the contract it meets, in two attributes at the **root of the
file**:

| Attribute        | Value               | Meaning                                      |
| ---------------- | ------------------- | -------------------------------------------- |
| `schema`         | `gwmock-strain`     | Which contract this is                       |
| `schema_version` | `MAJOR.MINOR.PATCH` | Which revision of it the file was written to |

Since `1.1.0` the root may carry a third attribute, `run_metadata`: the run's
provenance record as JSON. It is optional -- a file written before `1.1.0`, or
by a path with no record to embed, carries none -- so a consumer falls back to
the metadata sidecar when it is absent. See
[The record inside the file](#the-record-inside-the-file) below.

Version `1.0.0` requires, of **every** dataset in the file: the samples of one
channel, in a dataset named for that channel, plus these attributes.

| Attribute          | Meaning                                   |
| ------------------ | ----------------------------------------- |
| `x0`               | Epoch of the first sample, in GPS seconds |
| `dx`               | Sample interval, in seconds               |
| `xunit`            | The unit those two are in, `s`            |
| `channel` / `name` | The channel the samples belong to         |

Anything else on the dataset is an extra, and a consumer ignores it. Two extras
appear in practice: `unit` (`strain`, where the producer recorded one), and
`t0`/`dt`, which duplicate `x0`/`dx` and are how the multichannel writer inside
`gwmock-signal` spells the grid. When only one of the two pairs is present,
gwmock derives the other as it declares the file. When both are present and
disagree, gwmock **refuses** the file -- at declaration, and again at validation
-- so a declared file never carries two grids that contradict each other.

The **major** version moves when a reader written against the previous version
would misread a file; the **minor** moves when something is added that such a
reader can safely ignore. So a consumer checks the major and refuses anything
else, rather than discovering the change as a wrong number:

```python
from gwmock.strain_schema import read_strain_schema, require_strain_schema

# Raises if the file declares no schema, another schema, a major version this
# gwmock does not know how to read, or a layout that does not match what it
# declares -- the claim is checked, not taken on trust.
require_strain_schema("filename.hdf5")

# Or look without refusing: None means the file predates the declaration or
# came from another producer.
declared = read_strain_schema("filename.hdf5")
```

The declaration sits at the file root rather than on the dataset so that GWpy
still reads the file: GWpy passes every _dataset_ attribute to the series
constructor, and one it does not recognise makes the file unreadable through
`TimeSeries.read`.

**GWF and `.npy` carry no declaration.** A frame is composed from a fixed set of
fields and `.npy` is a bare array container, so neither has anywhere to put it;
for those formats the run's metadata sidecar remains the description of what was
written.

## The record inside the file

An HDF5 file usually also carries the provenance record of the run that wrote
it, so a file that reaches you without its sidecar still says where it came
from. Read it, and fall back to the sidecar when there is none — a file written
before this existed carries no record, and neither does one from
`gwmock merge --force`, which is merging files it was given no metadata for:

```python
from gwmock.strain_schema import read_run_metadata

record = read_run_metadata("filename.hdf5")
if record is None:
    ...  # no embedded record: read the run's metadata sidecar instead
else:
    print(record["gwmock_version"], record["config"], record["outputs"])
```

**Use the sidecar whenever you have one.** It is the complete record, and it is
the only description at all for `.npy` and `.gwf` — and for the forced merge
above, which writes neither.

What the file carries is the same record the sidecar holds, with three
deliberate omissions:

- **The injection parameters**, unless the run that wrote the file set
  `orchestration.include-injection-parameters: true`. The default excludes them
  so that a blind mock data challenge can be released as the data files
  themselves. `record["signal"]["injections"]` is simply absent, rather than
  empty -- an empty list would be indistinguishable from a segment that holds no
  signal.
- **The file hashes.** A file cannot carry its own digest: `file_hashes`,
  `content_hashes` and the `sha256`/`content_sha256` of each entry in `outputs`
  are recorded in the sidecar, which is taken after the record is written into
  the file.

- **The simulator's replay state.** `pre_batch_state` is stored as separate
  `.npy` files an embedded copy could not point at. It is removed wherever it
  appears, including inside the source records a merged file carries: that state
  and the configuration beside it regenerate the run, and so regenerate the
  injections the same document withheld.

## Reading a file

```python
from gwpy.timeseries import TimeSeries

# HDF5, what a run writes by default
data = TimeSeries.read("filename.hdf5", channel="ET1_EMR:STRAIN")

# GWF, if you configured frame output
data = TimeSeries.read("filename.gwf", channel="ET1_EMR:STRAIN")
```

**Parameters:**

- `filename`: Path to the file, HDF5 or GWF
- `channel`: Channel name to read (common format: `DETECTOR:CHANNEL_NAME`)

### Example

```python
from gwpy.timeseries import TimeSeries

# Read ET1_EMR strain data
e1_data = TimeSeries.read("E-ET1_EMR_STRAIN_NOISE-1577491218-4096.gwf", channel="ET1_EMR:STRAIN")

# Check properties
print(f"Duration: {e1_data.duration}")
print(f"Sampling frequency: {e1_data.sample_rate}")
print(f"Start time: {e1_data.t0}")
```

## Merging Frame Files

Frame files generated by gwmock may contain different types of content (noise,
signals, glitches). To obtain a realistic data stream, merge multiple files:

```python
from gwpy.timeseries import TimeSeries

# Read noise and signal data
noise_data = TimeSeries.read("filename_noise.gwf", channel="ET1_EMR:STRAIN")
signal_data = TimeSeries.read("filename_signal.gwf", channel="ET1_EMR:STRAIN")

# Combine them
combined_data = noise_data.inject(signal_data)
```

You can also merge files directly using the CLI:

```bash
gwmock merge filename_noise.gwf filename_signal.gwf \
    --metadata noise/metadata/orchestration-0.metadata.json \
    --metadata signal/metadata/orchestration-0.metadata.json \
    --channel ET1_EMR:STRAIN \
    --output-channel ET1_EMR:STRAIN
```

This produces a merged frame file and a merged metadata file documenting all
input files and merge details.

### Merging Multiple Files

To merge a sequence of files:

```python
from gwpy.timeseries import TimeSeries

files = [
    "E-ET1_EMR_STRAIN_NOISE-1000000000-1024.gwf",
    "E-ET1_EMR_STRAIN_NOISE-1000001024-1024.gwf",
    "E-ET1_EMR_STRAIN_NOISE-1000002048-1024.gwf",
]

# Read all files
data_list = [TimeSeries.read(f, channel="ET1_EMR:STRAIN") for f in files]

# Concatenate
combined = data_list[0]
for data in data_list[1:]:
    combined = combined.append(data)
```

<!-- prettier-ignore-start -->

!!! warning
    Two time series can only be combined if:

    1. **Time properties match**: Same start time, sampling frequency, and continuous coverage
    2. **Units match**: Both must have the same physical units (e.g., strain)

    If units differ, override them before combining:

    ```python
    from astropy.units import Unit

    noise_data.override_unit(Unit(""))
    signal_data.override_unit(Unit(""))
    ```

<!-- prettier-ignore-end -->

## Accessing Metadata

gwmock automatically generates metadata files for each simulation. Access them
with:

```python
import json

# Read a JSON metadata record
with open("metadata/orchestration-0.metadata.json", "r") as f:
    metadata = json.load(f)

print(metadata["schema_version"])  # e.g. "1.0.0"
print(metadata["gwmock_version"])  # e.g. "0.5.0"
print(metadata["config"])  # resolved config snapshot
print(metadata["outputs"])  # list of generated files with hashes
```

**Metadata fields:**

| Field                 | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `schema_version`      | Provenance format version                                      |
| `gwmock_version`      | Package version used to generate the data                      |
| `subpackage_versions` | Versions of `gwmock_signal`, `gwmock_noise`, `gwmock_pop`      |
| `config`              | Resolved configuration snapshot for this run                   |
| `config_sha256`       | SHA-256 hash of the resolved config                            |
| `seed`                | Top-level RNG seed                                             |
| `segment_seeds`       | Per-segment deterministic seeds                                |
| `population`          | Population backend and provenance                              |
| `signal`              | Signal backend, waveform model, detector network               |
| `noise`               | Noise backend and PSD                                          |
| `outputs`             | List of generated files (path, channels, t0, duration, sha256) |
| `host`                | Platform, Python version, CPU, git SHA                         |

For a quick guide on how to inspect and reuse metadata files to reproduce a
dataset, see the [Metadata Files](metadata.md) page.

## Working with Multiple Detectors

Process data from multiple detectors:

```python
from gwpy.timeseries import TimeSeries

detectors = ["ET1_EMR", "ET2_EMR", "ET3_EMR"]

# Read data for each detector
detector_data = {}
for detector in detectors:
    channel = f"{detector}:STRAIN"
    filename = f"E-{detector}_STRAIN_NOISE-1000000000-1024.gwf"
    detector_data[detector] = TimeSeries.read(filename, channel=channel)

# Process or analyze each
for detector, data in detector_data.items():
    print(f"{detector}: {data.duration.to('minute')} of data")
```

## Plotting Data

Visualize the data using GWpy's plotting utilities:

```python
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# Read data
data = TimeSeries.read("E-ET1_EMR_STRAIN_NOISE-1000000000-1024.gwf", channel="ET1_EMR:STRAIN")

# Plot time series
plot = data.plot(title="Strain Data")
plot.show()

# Plot power spectral density
spectrum = data.psd()
plot = spectrum.plot()
plot.show()
```

## Best Practices

1. **Always specify the channel**: Use full channel name format
   `DETECTOR:CHANNEL_NAME`
2. **Check continuity**: Verify time properties before combining files
3. **Preserve units**: Don't remove or override units unless necessary
4. **Use metadata**: Reference metadata files to understand generation
   parameters
5. **Handle large files**: Use streaming/windowing for files larger than
   available RAM

## Troubleshooting

**"Channel not found" error**

Check available channels in the file:

```python
from gwpy.io import gwf

# List all channels
channels = gwf.get_channel_names("filename.gwf")
print(channels)
```

**Units mismatch**

Ensure both time series have compatible units:

```python
# Check units
print(data1.unit)
print(data2.unit)

# Convert if needed
data2_converted = data2.to("strain")
```

**Time alignment issues**

Verify time properties before merging:

```python
print(f"Data 1: {data1.t0} to {data1.tf}")
print(f"Data 2: {data2.t0} to {data2.tf}")
```
