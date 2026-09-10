# Examples

This page provides an overview of example configuration files available for ET
simulations.

## Overview

Most of the ET noise, BBH, and BNS example configurations in the
[`examples/`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory generate one day of data per interferometer, divided into 4096-second
frames (21 frame files per interferometer), sampled at 4096 Hz. The noise
examples start at GPS 1577491218 (1 January 2030); the BBH/BNS signal examples
start at GPS 1000000540 (14 September 2011). A few examples differ on purpose:
`quick_start` is a short single-segment (1024 s) demonstration, and the SGWB
examples generate a short stochastic-background segment.

For guidance on changing dataset duration or simulation properties, see the
[Generating Data](generating-data.md) page. For a more complete guide to writing
your own configuration files, see the [Configuration Files](configuration.md)
page.

To list all the available example configuration files:

```bash
gwmock config --list
```

To run any of the following configuration files:

```bash
# Copy configuration file to working directory
gwmock config --get <label> --output config.yaml

# Run simulation
gwmock simulate config.yaml
```

## Blank Starting Config

The `default_config` label produces a minimal, hand-commented noise-only
orchestration template for the ET Sardinia triangle. Use it as a starting point
when writing a custom configuration from scratch:

```bash
gwmock config --get default_config --output config.yaml
```

It runs as-is (noise only). To generate signals as well, add `population` and
`signal` sections to the `orchestration:` block — see the
[Configuration Files](configuration.md) and [Orchestration](orchestration.md)
guides, or copy one of the signal examples below.

## Noise Generation

Example configurations for generating detector noise with various configurations
and sensitivities.

### Einstein Telescope - Triangular

- EMR location:
  [`noise/uncorrelated_gaussian/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml)
- Sardinia location:
  [`noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml)

### Einstein Telescope - 2L

- Aligned configuration:
  [`noise/uncorrelated_gaussian/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_2l_aligned/config.yaml)
- Misaligned configuration:
  [`noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml)

## CBC Signals Generation

Example configurations for generating detector data with compact-binary
coalescence (BBH and BNS) signals for various detector configurations.

### Binary Black Hole (BBH)

- EMR (triangular):
  [`signal/bbh/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_triangle_emr/config.yaml)
- Sardinia (triangular):
  [`signal/bbh/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_triangle_sardinia/config.yaml)
- 2L aligned:
  [`signal/bbh/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_2l_aligned/config.yaml)
- 2L misaligned:
  [`signal/bbh/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_2l_misaligned/config.yaml)

### Binary Neutron Star (BNS)

- EMR (triangular):
  [`signal/bns/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bns/et_triangle_emr/config.yaml)
- Sardinia (triangular):
  [`signal/bns/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bns/et_triangle_sardinia/config.yaml)
- 2L aligned:
  [`signal/bns/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bns/et_2l_aligned/config.yaml)
- 2L misaligned:
  [`signal/bns/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bns/et_2l_misaligned/config.yaml)

## Stochastic Background (SGWB) Generation

Example configurations for generating an isotropic stochastic gravitational-wave
background. These set `signal.source-type: sgwb` and require no population:

- Sardinia (triangular):
  [`signal/sgwb/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/sgwb/et_triangle_sardinia/config.yaml)
- 2L aligned:
  [`signal/sgwb/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/sgwb/et_2l_aligned/config.yaml)

## Glitch Generation

Example configurations for generating detector glitches with various
configurations and sensitivities.

### gengli blip glitches

One configuration **per interferometer**, because each draws from its own glitch
population file. Output is HDF5. These need `gengli`, which is not a declared
extra of gwmock — install it yourself.

#### Einstein Telescope - Triangular

- EMR location
    - E1:
      [`noise/glitches/gengli/et_triangle_emr/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_triangle_emr/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e2/config.yaml)
    - E3:
      [`noise/glitches/gengli/et_triangle_emr/e3/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e3/config.yaml)
- Sardinia location
    - E1:
      [`noise/glitches/gengli/et_triangle_sardinia/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_triangle_sardinia/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e2/config.yaml)
    - E3:
      [`noise/glitches/gengli/et_triangle_sardinia/e3/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e3/config.yaml)

#### Einstein Telescope - 2L

- Aligned configuration
    - E1:
      [`noise/glitches/gengli/et_2l_aligned/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_aligned/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_2l_aligned/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_aligned/e2/config.yaml)
- Misaligned configuration
    - E1:
      [`noise/glitches/gengli/et_2l_misaligned/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_misaligned/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_2l_misaligned/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_misaligned/e2/config.yaml)

### DeepExtractor glitches

One configuration **per detector network**: `detectors` names a network preset
and the noise side resolves it into that geometry's interferometers, each with
its own glitch realisation. Real O3 reconstructions in seven Gravity Spy
classes, at their measured O3 rates and median SNRs, written as GWF frames.
These need `gwmock[deepextractor]` and download a 2.3 GB HuggingFace dataset on
first use. See [Configuration Files](configuration.md) for what each argument
does.

- Triangle, Sardinia location:
  [`noise/glitches/deepextractor/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/deepextractor/et_triangle_sardinia/config.yaml)
- Triangle, EMR location:
  [`noise/glitches/deepextractor/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/deepextractor/et_triangle_emr/config.yaml)
- 2L, aligned configuration:
  [`noise/glitches/deepextractor/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/deepextractor/et_2l_aligned/config.yaml)
- 2L, misaligned configuration:
  [`noise/glitches/deepextractor/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/deepextractor/et_2l_misaligned/config.yaml)

## Storage Estimates

For reference, typical storage requirements:

- **Noise**: ~123 MB per file, ~7.6 GB for 3 detectors, 24 hours
- **Signals**: Variable depending on waveform complexity, typically similar to
  noise
- **Glitches**: Variable depending on number and duration
