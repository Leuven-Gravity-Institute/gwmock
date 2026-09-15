# Generating Data

This guide shows how to use `gwmock` to create realistic mock data for
gravitational-wave detectors.

It uses the Einstein Telescope (ET) triangular configuration located in the
Meuse-Rhine Euregion as an example.

For an overview of all example configuration files for ET simulations, see the
[Examples](examples.md) page. For a quick guide on reading and working with the
output GWF files, see the [Reading Data](reading-data.md) page.

## Generating Detector Noise

Detector noise can be generated using configuration files in the
[`examples/noise`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise)
directory. An example configuration for producing one day of ET noise data is
provided in
[`uncorrelated_gaussian/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/blob/main/examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml):

```yaml
--8<-- "examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml"
```

This configuration uses the `ET-Triangle-EMR` network alias, which expands to
its three interferometers (`ET1_EMR`, `ET2_EMR`, `ET3_EMR`), generating one day
of noise data per interferometer. Each frame file covers 4096 seconds, resulting
in 21 frame files per interferometer, starting on 1 January 2030.

Noise is simulated using the
[ET_10_full_cryo_psd](https://github.com/Leuven-Gravity-Institute/gwmock/blob/main/src/gwmock/detector/noise_curves/ET_10_full_cryo_psd.txt)
sensitivity curve from the
[CoBA Science Study](https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068)
and publicly available. A low-frequency cutoff of 3 Hz is used.

To generate the ET noise data, run:

```bash
# Create working directory
mkdir noise_et_triangle_emr
cd noise_et_triangle_emr

# Copy configuration file to your working directory
gwmock config --get noise/uncorrelated_gaussian/et_triangle_emr --output config.yaml

# Run simulation
gwmock simulate config.yaml
```

### Storage Requirements

Each GWF file is approximately 123 MB. For three detectors with 21 files each:

- **Data files**: ~7.6 GB
- **Metadata**: ~52.5 KB
- **Total**: ~7.6 GB

## Generating CBC Signals

Compact Binary Coalescence (CBC) signals can be generated using configuration
files in the
[`examples/signal/bbh`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh)
directory.

### Binary Black Hole (BBH) Signals

An example configuration for producing one day of ET data containing BBH signals
is provided in
[`signal/bbh/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_triangle_emr/config.yaml):

```yaml
--8<-- "examples/signal/bbh/et_triangle_emr/config.yaml"
```

As with the noise example, this configuration file produces one day of data per
interferometer, with each frame file lasting 4096 seconds (for a total of 21
frame files), starting at GPS 1000000540 (14 September 2011).

BBH signals are injected into zero noise. The population is loaded with the
`FilePopulationLoader` from the MDC1 BBH catalogue hosted on Zenodo
(`mdc1_bbh.h5`); because `n-samples` is omitted, the full catalogue is used. The
[IMRPhenomXPHM](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.104056)
waveform model is used, with a low-frequency cutoff of 10 Hz and the
time-dependent detector response enabled (`earth-rotation: true`).

To generate the ET data with BBH signals, run:

```bash
# Create working directory
mkdir bbh_et_triangle_emr
cd bbh_et_triangle_emr

# Copy configuration file to your working directory
gwmock config --get signal/bbh/et_triangle_emr --output config.yaml

# Run simulation
gwmock simulate config.yaml
```

### Binary Neutron Star (BNS) Signals

BNS datasets are produced the same way, using the `signal/bns/*` examples. These
set `population.source-type: bns`, load the MDC1 BNS catalogue from Zenodo
(`mdc1_bns.h5`), and use the tidal
[IMRPhenomPv2_NRTidalv2](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.044003)
waveform model with a low-frequency cutoff of 20 Hz:

```yaml
--8<-- "examples/signal/bns/et_triangle_emr/config.yaml"
```

To generate the ET data with BNS signals, run:

```bash
gwmock config --get signal/bns/et_triangle_emr --output config.yaml
gwmock simulate config.yaml
```

## Generating Transient Noise Artifacts (Glitches)

Glitches can be generated using configuration files in the
[`examples/noise/glitches`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli)
directory. These examples attach
[`gwmock-noise`](https://github.com/Leuven-Gravity-Institute/gwmock-noise)
glitch models through `orchestration.noise.arguments.glitches`, so glitches are
treated as detector artifacts in the protocol-only pipeline. The bundled gengli
integration currently supports only _blip_ glitches.

An example configuration for producing one day of ET data for the E1 detector
containing blip glitches from a realistic population is provided in
[`noise/glitches/gengli/et_triangle_emr/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e1/config.yaml):

```yaml
--8<-- "examples/noise/glitches/gengli/et_triangle_emr/e1/config.yaml"
```

This configuration file generates one day of data for the E1 detector, divided
into 4096-second frame files (for a total of 21 frames), starting on 1
January 2030.

Blip glitches are injected into zero noise from the
[blip_glitch_population_E1.hdf5](https://sandbox.zenodo.org/records/514722)
population file, which can be generated from GravitySpy tables with
`gwmock-noise build-blip-glitch-table`. These glitches are modeled on LIGO blip
glitches observed during the O3 observing run and recolored to match the ET
sensitivity.

To generate the ET data for detector E1 with glitches, run:

```bash
# Create working directory
mkdir -p glitch_et_triangle_emr/e1
cd glitch_et_triangle_emr/e1

# Copy configuration file to your working directory for glitch simulation
gwmock config --get noise/glitches/gengli/et_triangle_emr/e1 --output config.yaml

# Run simulation
gwmock simulate config.yaml
```

<!-- prettier-ignore -->
!!! note
    The configuration file automatically downloads the glitch population file from a
    [Zenodo repository](https://sandbox.zenodo.org/records/514722).
    The file is saved in a cache directory (by default, `~/.gwmock/population/`).
    When the same population file is needed again, gwmock uses the cached copy to avoid re-downloading.

## Using Different Detector Configurations

`gwmock` includes several pre-configured Einstein Telescope detector geometries,
available in
[`gwmock/detector/detectors`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/src/gwmock/detector/detectors):

Triangular Configuration (Meuse-Rhine Euregion)

- `E1_triangle_emr`
- `E2_triangle_emr`
- `E3_triangle_emr`

Triangular Configuration (Sardinia)

- `E1_triangle_sardinia`
- `E2_triangle_sardinia`
- `E3_triangle_sardinia`

2L Aligned Configuration

- `E1_2L_aligned_sardinia`
- `E2_2L_aligned_emr`

2L Misaligned Configuration

- `E1_2L_misaligned_sardinia`
- `E2_2L_misaligned_emr`

To use a specific configuration, update the `detectors` list in your
configuration file:

```yaml
detectors:
    - E1_2L_aligned_sardinia
    - E2_2L_aligned_emr
```

You don't need to include all detectors. For example, to generate only E1 data:

```yaml
detectors:
    - E1_2L_aligned_sardinia
```

## Using Different Sensitivity Curves

Multiple Einstein Telescope sensitivity curves (PSD files) are available in
[`gwmock/detector/noise_curves/`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/src/gwmock/detector/noise_curves).
These correspond to those used in the CoBA study.

To use a specific sensitivity curve, set `psd_file` in the noise arguments:

```yaml
orchestration:
    noise:
        arguments:
            psd_file: ET_15_HF_psd.txt
```

<!-- prettier-ignore -->
!!! note
    The detector geometries assume 10 km arms for triangular configurations and 15 km arms for 2L configurations.
    Choose sensitivity curves accordingly.

## Adjusting Dataset Duration

The length of a dataset is controlled by:

```yaml
start-time: # GPS start time of the dataset
duration: # Duration per frame file (seconds)
total-duration: # GPS span of the dataset
segment-gap: # GPS seconds between consecutive segments (default 0)
```

To change the dataset duration, simply adjust these parameters in your
configuration file.

You can also change the sampling frequency of your dataset (the number of
samples per second, measured in Hz), using the `sampling-frequency` argument.

**Total number of frame files:**

`total-duration` is the dataset's GPS **span** — from the first segment's start
to the last segment's end. With the default `segment-gap: 0` the segments tile
that span, so the number of frame files is the span divided by the duration,
rounded to the nearest integer:

```python
max_samples = round(total_duration / duration)
```

For example, a one-day dataset (86400 s) in 4096-second frames yields
`round(86400 / 4096) = 21` frame files per interferometer. Note that 86400 is
not a multiple of 4096: the rounding is kept for compatibility, and gwmock warns
when it happens, naming the span it actually used (86016 s here).

With a non-zero `segment-gap` the segments no longer tile the span, and the
count follows from the stride instead:

```python
max_samples = (total_duration + segment_gap) / (duration + segment_gap)
```

A **gapped** span that does not divide exactly is **refused**, naming the span,
the duration and the gap, rather than rounded — rounding would move the run's
last epoch away from the layout the configuration describes.

## Gapped segments (discontiguous data)

Real instruments are not on all the time. `segment-gap` makes a run write
segments that are **discontiguous in GPS**: the seconds inside a gap appear in
no frame at all, so the released data carries the gaps rather than relying on a
downstream consumer to read a contiguous stream selectively.

```yaml
globals:
    simulator-arguments:
        sampling-frequency: 4096
        duration: 1024 # analysed segment length
        segment-gap: 256 # GPS seconds skipped between segments
        total-duration: 40704 # span = 32 * 1024 + 31 * 256
        start-time: 1577491218
```

Segment epochs then advance by `duration + segment-gap`:

```text
1577491218, 1577492498, 1577493778, 1577495058, ...
```

Two durations describe such a run and they are not equal:

- **span** (`total-duration`) — `count * duration + (count - 1) * segment-gap`.
  There is no trailing gap: a run stops at the end of data.
- **analysed livetime** — `count * duration`, which is what the frames actually
  cover.

Both are recorded per batch under
`simulator_metadata.orchestration.segment_layout` in the metadata record, so a
released frame set says what its own discontinuity is.

### What the gaps do to the signals

- A waveform that **crosses** a gap keeps its far-side content **at its true GPS
  sample**: the part inside the gap is discarded and the rest is placed where it
  belongs, not shifted forward to close the hole. What the gap swallowed is
  recorded under `signal.gap_discarded_injections`.
- A signal lying **entirely inside** a gap is written to no frame. It is not
  dropped silently: it is recorded under `signal.gap_excluded_injections` and
  warned about.

Both lists carry source parameters, so — like `signal.injections` — they are
withheld from the copy of the record embedded in a released data file unless
`orchestration.include-injection-parameters` is set.

### What the gaps do to the noise

The noise stream is stateful, and `gwmock_noise` interpolates a `psd_schedule`
against the number of samples the stream has **produced** rather than against
GPS. gwmock therefore **generates the gap's samples and throws them away**, so
that:

- a `psd_schedule` anchor at `gps_offset_seconds: 3600` means one hour after the
  run's start time whether or not gaps fall in between — the schedule's time
  axis stays locked to GPS, and a drifting instrument drifts in wall-clock time;
- a glitch model's Poisson process keeps its configured rate per unit of
  **real** time across a gap;
- the noise in the segment after a gap continues a detector that never stopped,
  rather than being a fresh draw.

The cost is the gaps' own generation. A run has no trailing gap, so for `count`
segments it discards `(count - 1) * segment-gap` seconds while writing
`count * duration`, and the extra-generation ratio is exactly

```text
(count - 1) * segment-gap / (count * duration)
```

which approaches `segment-gap / duration` for a long run — 24.2 % for the 32
segments above, against the 25 % the limit suggests. The alternative — skipping
the gap, so the schedule's axis tracks analysed livetime and drifts away from
GPS by the accumulated gap — is cheaper and is **not** what gwmock does.

<!-- prettier-ignore-start -->

!!! note
    The `total-duration` argument can be passed as a `float` in seconds, or as a `str` specifying the time unit
    (`"1 day"`, `"5 days"`, `"2 weeks"`, `"2 months"`, etc.).
    The supported time units are:

    - `second`
    - `minute`
    - `hour`
    - `day`
    - `week`
    - `month` (30 days)
    - `year` (365 days).

    Singular and plural forms are both accepted (e.g., `"1 day"` and `"2 days"`).

<!-- prettier-ignore-end -->

<!-- prettier-ignore -->
!!! tip
    A [UTC/GPS time converter](https://gwosc.org/gps/) is available at the Gravitational Wave Open Science Center.

<!-- prettier-ignore-start -->

!!! tip
    Sampling frequencies are often powers of 2 for efficiency. Common choices:

    - 4096 Hz (standard for GW data analysis)
    - 2048 Hz
    - 16384 Hz (high-frequency instruments)

    Lowering sampling frequency reduces computation time but also reduces the highest resolvable frequency (Nyquist limit = sampling_frequency / 2).

<!-- prettier-ignore-end -->

## Generate Multi-Detector Correlated Noise

You can generate multi-detector correlated noise by specifying a cross-power
spectral density (CSD) file via the `orchestration.noise` backend. Pass
`csd_file` as a noise argument:

<!-- prettier-ignore -->
!!! warning
    Correlated noise generation is experimental and not fully tested. Use at your own risk.

```yaml
globals:
    simulator-arguments:
        sampling-frequency: 4096
        duration: 4096
        total-duration: '1 day'
        start-time: 1577491218
    working-directory: .
    output-directory: output
    metadata-directory: metadata

orchestration:
    noise:
        arguments:
            psd_file: ET_10_full_cryo_psd
            csd_file: path_to_csd_file.txt
            detectors:
                - ET-Triangle-EMR
            minimum_frequency: 3
            seed: 42
        output:
            output_directory: noise
            file_name:
                'E-{{ detectors }}_STRAIN_CORRELATED-NOISE-{{ start_time }}-{{
                duration }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

`gwmock` uses a windowing approach to generate long-duration datasets. If the
input CSD varies rapidly with frequency, this windowing can introduce artifacts
in the resulting frame files.

A diagnostic tool to check whether your CSD file is susceptible to such issues
will be provided soon.

## Resume Interrupted Simulations

If a simulation is interrupted, resume it by running the same command:

```bash
# Start
gwmock simulate config.yaml

# If interrupted, resume
gwmock simulate config.yaml
```

gwmock automatically detects and continues from the last checkpoint, as long as
the checkpoint belongs to the command being run — see
[Checkpointing](configuration.md#checkpointing) for what it refuses and why.

## Combining Data Types

To create realistic mock data, you may generate noise, signals, and glitches
separately, then combine them:

```bash
gwmock simulate noise_config.yaml
gwmock simulate signal_config.yaml
gwmock simulate glitch_config.yaml
```

Then merge the files using GWpy (see [Reading Data](reading-data.md) for
details).
