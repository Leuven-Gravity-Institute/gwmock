# Configuration Files

This guide explains how to use and write configuration files to generate
datasets tailored to your needs.

## Verbosity

All gwmock commands accept a top-level `--verbose` / `-v` flag to control log
output:

```bash
gwmock --verbose DEBUG simulate config.yaml   # detailed debug output
gwmock --verbose WARNING simulate config.yaml # only warnings and errors
```

Supported levels: `NOTSET`, `DEBUG`, `INFO` (default), `WARNING`, `ERROR`,
`CRITICAL`.

## Command-Line Options

### Command [`simulate`](/reference/gwmock/cli/simulate/)

```bash
gwmock simulate config.yaml
```

This is the primary command used to generate mock data. It takes a `.yaml`
configuration file as input, which defines the simulation parameters.

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing output files. If a file already
exists, the tool will raise an error and halt execution. To force overwriting of
existing files, use the `--overwrite` flag:

```bash
gwmock simulate config.yaml --overwrite
```

#### Flag `--dry-run` (optional)

Test your configuration without generating data:

```bash
gwmock simulate config.yaml --dry-run
```

This validates the configuration and shows what would be generated without
actually creating files.

#### Flag `--output-dir` (optional)

Override the output directory from the command line without editing the config:

```bash
gwmock simulate config.yaml --output-dir /scratch/my_run/data
```

#### Flag `--metadata-dir` (optional)

Override the metadata directory from the command line (config mode only):

```bash
gwmock simulate config.yaml --metadata-dir /scratch/my_run/metadata
```

#### Flag `--metadata` (optional)

Generate metadata files along with the data (automatically enabled by default):

```bash
gwmock simulate config.yaml --metadata
```

Metadata files contain complete provenance information including:

- Simulator configuration
- Random number generator state
- Output file names
- Version information

#### Flags `--author` and `--email` (optional)

Include author information in the metadata files:

```bash
gwmock simulate config.yaml --author <your-name> --email <your-email>
```

Without `--author`, the user name of the account running gwmock is recorded.
Either way the author and email go to the private details file, not to the
shareable record that travels with the data, unless the split is turned off; see
[`metadata`](#metadata-shareable-and-private-records) below.

### Command [`config`](/reference/gwmock/cli/config/)

```bash
gwmock config <flag>
```

This command is used to manage default and example configuration files. Exactly
one of the flags `--list`, `--get`, or `--init` must be provided.

#### Flag `--list`

List all the available example configuration files stored in the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory (see the [Examples](examples.md) page).

```bash
gwmock config --list
```

#### Flag `--get`

Copy one of the available example configuration files from the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory into the working directory. The `<example_label>` must be one of the
example names listed by the `gwmock config --list` command.

```bash
gwmock config --get <example_label>
```

#### Flag `--init`

Creates a default configuration file and saves it to the working directory.

```bash
gwmock config --init config.yaml
```

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing configuration files. If a file
already exists, the tool will raise an error and halt execution. To force
overwriting of existing files, use the `--overwrite` flag together with `--get`
or `--init`:

```bash
gwmock config --get noise/uncorrelated_gaussian/quick_start --overwrite
gwmock config --init config.yaml --overwrite
```

With `--get`, `--overwrite` also replaces a different catalogue file that an
example copies beside the configuration, such as `bbh_population.csv`. It does
not replace a symbolic link at that path: a link there, dangling or not, is
refused, so that the link is not replaced by a copy of the catalogue.

#### Flag `--output` (optional)

Specifies the directory where the configuration file will be saved. This flag
must be used together with `--get` or `--init`. If not provided, the working
directory is used by default.

```bash
gwmock config --get <label of the configuration file> --output <directory or file>
```

#### Flag `--interactive` (optional)

Launch an interactive terminal-based configuration editor with a live preview,
autocomplete, and guided workflows:

```bash
gwmock config --interactive
```

The interactive editor provides:

- **Live configuration preview**: See your configuration update in real-time as
  you build it
- **Autocomplete suggestions**: Type `/` to see available commands, then use Tab
  to complete
- **Command history**: Use Up/Down arrows to navigate through previously entered
  commands
- **Validation feedback**: Get immediate feedback on invalid values (e.g.,
  negative seeds, invalid chunk counts)
- **Templates**: Start with common configurations using `/template` commands
- **Script generation**: Generate SLURM job scripts or local execution scripts
  with `/generate-script`

**Common commands:**

| Command                             | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `/template <type>`                  | Load a preset (e.g., `noise`, `signal+noise`, `glitch`) |
| `/psds`                             | List available power spectral densities                 |
| `/geometries`                       | List available detector geometries                      |
| `/noise psd <value>`                | Set noise PSD                                           |
| `/noise detectors <list>`           | Set detector network                                    |
| `/batch chunks-enabled true`        | Enable chunking for parallel execution                  |
| `/batch chunks-n-chunks <n>`        | Set number of chunks                                    |
| `/save <filename>`                  | Save configuration to file                              |
| `/generate-script slurm <filename>` | Generate SLURM job script                               |
| `/generate-script local <filename>` | Generate local execution script                         |
| `/help`                             | Show all available commands                             |

**Example workflow:**

```bash
# Start interactive editor
gwmock config --interactive

# Inside the editor:
/template noise
/noise psd ET_10_full_cryo_psd
/noise detectors ET-Triangle-EMR
/batch chunks-enabled true
/batch chunks-n-chunks 4
/save my_config.yaml
/generate-script slurm submit.sh
```

#### Flag `--load` (optional)

Load an existing configuration file into the interactive editor for
modification:

```bash
gwmock config --interactive --load existing_config.yaml
```

This is useful for:

- Modifying existing configurations without manually editing YAML
- Exploring what settings are in a configuration file
- Generating job scripts for existing configurations

## Configuration File Structure

The configuration file uses YAML format. It consists of a shared `globals`
section plus the adapter-backed [orchestration](orchestration.md) schema.

### Globals

Top-level shared parameters used across all simulators:

```yaml
globals:
    working-directory: .
    output-directory: output
    metadata-directory: metadata
    simulator-arguments:
        sampling-frequency:
        duration:
        start-time:
        total-duration:
        segment-gap:
    output-arguments: {}
    metadata:
        split: true
```

**Key parameters:**

- `working-directory`: Base directory for operations
- `output-directory`: Where to save generated data files
- `metadata-directory`: Where to save metadata files
- `sampling-frequency`: Sample rate in Hz
- `duration`: Duration of each analysed segment in seconds
- `start-time`: GPS start time of the first segment
- `total-duration`: The run's GPS **span** — first segment's start to last
  segment's end, gaps included
- `segment-gap`: Seconds of GPS time between consecutive segments, so epochs
  advance by `duration + segment-gap`. Defaults to `0`, which is the contiguous
  layout. See
  [Gapped segments](generating-data.md#gapped-segments-discontiguous-data)
- `output-arguments`: Additional global arguments passed to the file writer
- `metadata.split`: Write each metadata record as a shareable record plus a
  private details file (`true`, the default), or as one combined record
  (`false`). See [`metadata`](#metadata-shareable-and-private-records)

#### `metadata`: shareable and private records

```yaml
globals:
    metadata:
        split: true # the default
```

With `split: true` every record a run writes is two files in the metadata
directory: the shareable record under its usual name
(`orchestration-<k>.metadata.json`), with no user name, host name or absolute
path, and a private details file beside it (`orchestration-<k>.private.json`)
holding exactly what was taken out. The copy embedded in each HDF5 output and
the signal and glitch indexes are built from the shareable record. What goes
where is listed in
[Shareable and private metadata](metadata.md#shareable-and-private-metadata).

`split: false` writes one combined record per batch, with the author, the
absolute paths and the full platform string in it -- the layout gwmock wrote
before the split existed, and the one that is embedded in the outputs. No
private file is written.

`metadata` accepts only `split`, and `split` only a boolean: a misspelt key or
value is refused when the configuration is loaded rather than ignored, so a run
cannot silently write the layout its author meant to turn off. `gwmock merge`
takes the same decision through `--split-metadata` / `--no-split-metadata`.

### Orchestration

The `orchestration:` section is required and must contain at least one of
`population`, `signal`, or `noise`. CBC signal generation uses `population` plus
`signal`. SGWB signal generation can use `signal` without `population` when
`signal.source-type` is set.

```yaml
orchestration:
    # Whether the HDF5 output files may carry the parameters the signals were
    # injected with. False by default -- see "Injection parameters in the data
    # files" below.
    include-injection-parameters: false

    population:
        backend: FilePopulationLoader # or any registered backend alias
        source-type: bbh
        n-samples: 128 # optional; omit to load the full catalogue
        arguments:
            path: population.h5

    signal:
        waveform-model: IMRPhenomXPHM
        minimum-frequency: 10
        detectors:
            - ET-Triangle-EMR
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_BBH-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'

    noise:
        arguments:
            psd_file: ET_10_full_cryo_psd
            seed: 42
            detectors:
                - ET-Triangle-EMR
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_NOISE-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

### Noise arguments

With the default noise backend (no `noise.backend`), `noise.arguments` accepts
exactly these keys, with hyphens and underscores interchangeable:

| Key                     | Meaning                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `detectors`             | Detector network; required when there is no `signal` section, else defaults to its list |
| `seed`                  | Root seed; defaults to the global `seed` when that is set                               |
| `psd_file`              | One PSD for every detector                                                              |
| `psd_files`             | One PSD per detector                                                                    |
| `csd_files`             | Cross-spectral densities for correlated noise                                           |
| `psd_schedule`          | A PSD that changes over the run                                                         |
| `low_frequency_cutoff`  | Lowest frequency the noise is generated from, in Hz (default `2.0`)                     |
| `high_frequency_cutoff` | Highest frequency the noise is generated to, in Hz                                      |
| `spectral_lines`        | Narrow spectral lines added to the noise                                                |
| `glitches`              | Transient glitch models, described below                                                |

Any other key is an error naming the key, rather than being ignored: a
misspelled or misplaced setting would otherwise leave the run on the default
while looking entirely normal. In particular the noise floor is
`low_frequency_cutoff`, not `minimum-frequency`, which belongs to `signal`. The
keys of each `glitches` entry are checked the same way against the glitch model
its `kind` names, which also calls its floor `low_frequency_cutoff`. A
`population` entry is checked by `gwmock-noise` itself, including its pinned
`digest`. Both checks run when the configuration is loaded, so `gwmock simulate`
stops before anything is generated. A custom `noise.backend` receives
`noise.arguments` as its constructor's keyword arguments instead, so that
backend decides which keys it accepts.

### Injection parameters in the data files

Every HDF5 file a run writes carries the run's metadata record inside it, at the
file root, so that a file handed to another pipeline says which run produced it
without its sidecar. `.npy` and `.gwf` have nowhere to put a document, so for
those formats the sidecar remains the only description, and so does a file
written by `gwmock merge --force`, which was given no metadata to carry. A
consumer reads the sidecar whenever there is one — see
[Reading data](reading-data.md) for how to read the embedded record back.

The embedded copy leaves out the source parameters of the injected signals, and
the per-event glitch truth beside them:

```yaml
orchestration:
    include-injection-parameters: false # the default
```

A blind mock data challenge is released as the strain files alone, and those
parameters are the answer its participants are asked to find, so excluding them
is the default and including them has to be asked for. Which transients a file
holds — a glitch's time, class and SNR, recorded under `noise.glitch_injections`
— is as much an answer, for a challenge whose task is to find or veto them, so
the flag governs both together. Set the flag to `true` for data generated for a
different purpose — a training or inference set, a benchmark, a released
"solved" challenge.

It changes the data files only. The metadata sidecar always records the
injection parameters, whatever the flag says; it is the producer's copy and is
not part of a release. `gwmock merge` takes the same decision through
`--include-injection-parameters`, with the same default, because the sidecars it
reads carry the parameters even when the files it merges do not.

Two things the flag does **not** do, and both matter before releasing a blind
challenge:

- The embedded record still carries the run's configuration, its seeds and its
  software versions. If the population is _drawn_ from a distribution rather
  than loaded from a file you withhold, the configuration and the seed
  regenerate the injections whether or not their values were embedded.
- It does not touch data already written. Files from an earlier run keep
  whatever they were written with.

One further consequence: two identical runs no longer write byte-identical HDF5
files, because the record embedded in them carries a timestamp, the host and the
environment freeze. What reproducibility is checked against is the _content_
hash — the decoded samples and their timing — which the record does not affect,
and which `gwmock validate` reports separately from the byte hash.

### Choosing the waveform library

`signal.waveform-backend` selects which library generates the polarizations —
`lal`, `pycbc`, `ripple`, or `gwsignal`. The default is `lal` where lalsuite is
installed and `ripple` where it is not (Python 3.15, for now; it needs the `jax`
extra, and a run without either stops with an error saying how to install them).
It also accepts an entry point in the `gwmock.waveform` group or a
`module:Class` reference, so a third-party backend can be plugged in the same
way. Such a backend is matched by its public surface — `available_approximants`
and `generate_td_waveform` — and does not have to subclass gwmock-signal's
`WaveformBackend`.

Constructor arguments for that backend go under
`signal.waveform-backend-arguments`. These are distinct from `signal.arguments`,
which is passed to the _simulator_ rather than to the waveform backend:

```yaml
signal:
    waveform-model: IMRPhenomD
    waveform-backend: ripple
    waveform-backend-arguments:
        taper_fraction: 0.05 # ripple-specific
```

Keep them apart from `signal.waveform-arguments`, which are merged into every
event's parameters and passed to each waveform call. A key there that the
backend does not accept stops the run with the backend's message rather than
being dropped, because dropping it would generate a waveform other than the one
configured. A constructor setting such as `f_ref` belongs under
`waveform-backend-arguments`.

### Population columns the waveform does not read

A population catalogue usually carries more columns than the waveform reads.
Each is recorded in the injection parameters either way; what happens in the
waveform call depends on the column:

- **Population-only columns are left out quietly**: `redshift` and labels and
  indices (`event_id`, `source_type`, `source_class`, `class`).
- **Columns that restate what the waveform reads are left out only when, for
  each event, the waveform's own copy is present and agrees**; otherwise the run
  stops, naming the event. Every value a rule reads must be finite — a NaN would
  make each comparison false and let any value through — and masses must be
  positive. A column declared ignored is never read, even by the rule of a
  column it accompanies:
    - `source_frame_mass_1`/`_2`: the detector-frame mass is present, and with
      `redshift` present (and not declared ignored) equals `m_src (1 + z)` to a
      relative 1e-6.
    - `spin_1`, `spin_2`, `tilt_1`, `tilt_2`: all three Cartesian components of
      that spin are present, and the magnitude is their norm and the tilt
      `arccos(z / |chi|)`, to 1e-6 (the tilt is not checked at zero spin). A
      catalogue giving spins only as magnitude and tilt would otherwise generate
      non-spinning binaries.
    - `theta_jn`: `inclination` is present. Where neither body has an in-plane
      spin component the binary has no precession and the total angular momentum
      J lies along the orbital one L: `J_z = |L| + S_z`, with
      `S_z = m1² χ1z + m2² χ2z` from the detector-frame masses and aligned
      spins. Where `S_z ≥ 0` — a binary without spin included — `J_z` is
      positive whatever `|L|` is, so J is parallel to L exactly and `theta_jn`
      must equal `inclination`. Where `S_z < 0`, anti-aligned spin may outweigh
      `|L|` and turn J around; which happens depends on `|L|` at the reference
      frequency, which no truncated post-Newtonian series pins down near
      `J = 0`, so `inclination` and `π − inclination` are both accepted.
      Anything else is refused, to 1e-6 rad. An event without both
      detector-frame masses and both aligned spin components is refused. With
      in-plane spin the two angles differ in general, and only the presence of
      `inclination` is required.
    - A per-event `f_ref`: equals, to a relative 1e-6, the reference frequency
      the waveform backend uses: `f_ref` under `waveform-backend-arguments`, or
      the `minimum-frequency` when that is unset. A catalogue whose spins refer
      to 20 Hz needs `f_ref: 20` there.
- **`eccentricity`, `long_asc_node` and `mean_per_ano` are left out only while
  they are zero.** gwmock generates quasi-circular waveforms, which are exact at
  zero eccentricity; a nonzero value stops the run.
- **Any other column stops the run before anything is generated**, naming every
  such column and what does not read it: the waveform backend on the per-event
  path, or the batched path with `execution: batched`. A misspelt column —
  `spin1_z` for `spin_1z` — would otherwise vanish and the run would generate a
  different signal than the catalogue describes.

<!-- prettier-ignore -->
!!! note "Catalogues written by gwmock-pop before 0.18.0"
    gwmock requires gwmock-pop 0.18.0 or later, whose packaged presets write
    neither `theta_jn` nor `f_ref`, so their draws pass these checks. A
    catalogue _file_ drawn from the presets `bbh_flat`, `bbh_gwtc4`,
    `bns_flat` or `nsbh_flat` with an older gwmock-pop still carries both:
    `theta_jn` drawn independently of `inclination`, although the presets draw
    no in-plane spin (`bbh_gwtc4` draws aligned spin components; the others
    draw none), and `f_ref = 20`. A run on such a file is refused before
    anything is generated, naming `theta_jn` (and `f_ref`, unless `f_ref: 20` is
    set under `waveform-backend-arguments`). Draw the catalogue again with the
    current gwmock-pop rather than declaring the columns ignored, since the
    inclination the waveform reads and the `theta_jn` the file records
    disagree.

If a column is not meant as a waveform input, declare it:

```yaml
orchestration:
    signal:
        ignored-population-columns:
            - snr
            - catalogue_weight
```

A declared column is left out of the waveform call without a refusal. Naming a
column the waveform reads, such as `spin_1z`, is an error when the configuration
is loaded, since such a column is never left out.

Two things to be aware of:

- This selects a **library, not a compute device.** `ripple` is JAX-based, but
  on its own it runs through the same per-event path as LAL. The batched
  on-device entry point is selected separately, with
  [`signal.execution`](#choosing-the-execution-mode).
- The same approximant from two libraries agrees closely but not exactly, so the
  choice changes the data. It is recorded in the run metadata as
  `orchestration.signal.waveform_backend` for that reason.

`ripple` requires the extra: `pip install 'gwmock[jax]'`. It also JIT-compiles
each waveform model on first use — measured on a single 8-second segment,
`IMRPhenomD` took ~10 s end to end against ~0.7 s for LAL, and the precessing
`IMRPhenomXPHM` ~72 s. The cost is paid once per process, so it amortises over a
long run.

### Choosing the projection effects

`signal.effects` says which physical effects the projection of the polarizations
onto the detectors includes. It is gwmock-signal's
[`ProjectionEffects`](https://leuven-gravity-institute.github.io/gwmock-signal/user_guide/detector-projection/#projection-effects),
one key per effect. **Every effect is on by default**, and each can be turned
off on its own:

| Key                 | Default | What it is                                                                                                                                                    |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `earth-rotation`    | `true`  | The antenna pattern and delays follow the Earth's rotation sample by sample. Off, they are held at the span's midpoint, which suits only a signal of seconds. |
| `finite-arm-length` | `true`  | Each arm responds to the strain averaged along the light's round trip to its end mirror and back, rather than to the strain at the vertex.                    |
| `source-precession` | `true`  | The J2000 source direction is precessed to the epoch of date before the Earth's orientation is applied, as LALPulsar's barycentring does.                     |
| `roemer-delay`      | `true`  | The light-travel time between the geocentre and the solar-system barycentre, from the Earth's orbit.                                                          |
| `einstein-delay`    | `true`  | The relativistic time dilation of the Earth's clock, TDB minus TT.                                                                                            |
| `shapiro-delay`     | `true`  | The Sun's gravitational delay.                                                                                                                                |

```yaml
orchestration:
    signal:
        effects:
            earth-rotation: true
            finite-arm-length: true
            source-precession: true
            roemer-delay: true
            einstein-delay: true
            shapiro-delay: true
```

A key left out keeps its default, so the block above is what an omitted
`effects` means. A key that is not one of the six is refused when the
configuration is loaded: a misspelled effect would otherwise stay on while the
file reads as if it were off.

**Matching LAL, Bilby and PyCBC.** Their default projection is the
long-wavelength one: a point detector at its vertex, a geocentre at rest and a
J2000 direction with sidereal time of date. To produce data in that convention,
turn every effect off — or every effect but `earth-rotation`, for a signal long
enough to need it:

```yaml
orchestration:
    signal:
        effects:
            earth-rotation: true
            finite-arm-length: false
            source-precession: false
            roemer-delay: false
            einstein-delay: false
            shapiro-delay: false
```

An analysis in that convention recovers a right ascension offset from an
injection made with `source-precession` on (about 0.43 degrees in 2030), and a
slightly different waveform for a long or loud signal; turn the effects off to
match it. One difference remains: gwmock-signal evaluates sidereal time from
UT1, the Earth's measured rotation, while LAL's `GreenwichMeanSiderealTime`,
which PyCBC and Bilby use, evaluates it from UTC. Such an analysis therefore
recovers the right ascension offset by −(UT1 − UTC) × 7.29e-5 rad/s, about 5
arcsec at GPS 1e9 and at most 6.6e-5 rad, with the waveform and the SNR
unchanged (gwmock-signal's
[scientific notes](https://leuven-gravity-institute.github.io/gwmock-signal/user_guide/detector-projection/#scientific-notes)).
The block above is also what compact-binary and burst signals got from gwmock
before `effects` existed, when gwmock-signal projected them without the other
five effects.

For a transient, the three solar-system delays enter only through how they
change across the signal: their value and rate at the signal's reference time (a
compact binary's coalescence, a burst's peak) are absorbed, so `coa_time` keeps
its meaning at the geocentre and the detector-frame masses theirs. A continuous
wave, whose parameters are defined at the barycentre, takes them in full. A
continuous wave also needs `earth-rotation`: the simulator refuses `false` for a
signal lasting months. The stochastic background projects nothing and ignores
the effects; its detector response is its overlap reduction function.

**What the defaults need.**

- **An arm length for every detector**, for `finite-arm-length`. The built-in
  LIGO and Virgo codes and the Einstein Telescope presets carry one; a custom
  detector needs `arm_length_m`. A detector without one stops the run with a
  message naming it — give it its arm length, or turn `finite-arm-length` off.
- **An epoch between 2000 and 2040**, for the solar-system delays, which read
  LALPulsar's JPL DE405 tables (`earth00-40-DE405.dat.gz`,
  `sun00-40-DE405.dat.gz`) covering those years. A signal outside them stops the
  run; turn `roemer-delay`, `einstein-delay` and `shapiro-delay` off to place
  one there.
- **The ephemeris tables, downloaded once.** The first run that needs them
  downloads the two tables from the LALSuite repository and caches them in
  `~/.cache/gwmock-signal/ephemeris` (`$XDG_CACHE_HOME/gwmock-signal/ephemeris`
  when that is set; `~/Library/Caches/gwmock-signal/ephemeris` on macOS). Set
  `GWMOCK_SIGNAL_CACHE_DIR` to cache them elsewhere, in
  `$GWMOCK_SIGNAL_CACHE_DIR/ephemeris`. On a machine without network access,
  such as a cluster's compute nodes, copy the two files there before the run.
  gwmock-signal checks both against their recorded SHA-256 and refuses a corrupt
  copy; deleting it lets it download again.

**The top-level `earth-rotation` key is deprecated.** It chose the Earth's
rotation before the other effects existed. It is still read: alone, it sets
`effects.earth-rotation`, leaves the other five at their defaults, and warns.
Given together with `effects`, it is refused; move its value into `effects`.

The effects are recorded, all six, in the run metadata as
`orchestration.signal.effects`, and they are part of the run's identity for
[checkpointing](#checkpointing): a checkpoint written with different effects is
not resumed, including one written before `effects` existed.

### Choosing the projection implementation

`signal.projection-backend` selects **which implementation projects** the
polarizations onto the detectors, independently of which library generated them:

| Value     | Behaviour                                                              |
| --------- | ---------------------------------------------------------------------- |
| _omitted_ | Leave the choice to the backend. **This is the default.**              |
| `numpy`   | Project on the host, asking Astropy for sidereal time at every sample. |
| `jax`     | The same algorithm, compiled into one fused kernel.                    |

```yaml
orchestration:
    signal:
        projection-backend: jax
        effects:
            earth-rotation: true
```

**Omitting the key is not the same as writing `numpy`.** Omitted, each
gwmock-signal backend keeps its own choice — the host path for compact binaries,
so a CBC run behaves exactly as it did before this key existed, and the _device_
path for continuous waves, where projection is 99% of a segment. Writing `numpy`
explicitly overrides that, which for a continuous-wave run means giving up the
faster path. Set it only when you mean to choose.

**What it buys.** Projection is where a long segment spends its time. Measured
at 1024 s and 8192 Hz across five ET detectors, a single-event `gwmock simulate`
run cost 620.3 CPU s, of which the projection alone was 604.3 s — **97%** — and
the same projection on the device path took 224.5 s. That is **2.7x off the
generation cost of the whole run**, which for a large dataset is most of the
compute bill. A short segment will show much less, because the one-off
compilation is then a larger share of the total.

**It is CPU or GPU depending on the JAX you installed**, exactly as for
[`execution`](#choosing-the-execution-mode):

- `pip install 'gwmock[jax]'` — the device path, on the CPU. This is where the
  2.7x above was measured.
- `pip install 'gwmock[cuda]'` — on a GPU when a compatible device and driver
  are present, and silently on the CPU when they are not.

Unlike `execution: batched`, this needs **only JAX** — not `ripple`. The two
keys are unrelated: `batched` chooses the batched _waveform_ entry point (which
projects on device unconditionally, and therefore refuses this key), while this
one changes nothing about how the waveforms are generated.

**It is not a different answer.** The two implementations agree to ~1e-10 of
peak — 2.5e-10 worst case across five ET detectors at the configuration above,
and 8.0e-13 through a 32 s segment at 256 Hz. The difference is floating-point
reassociation, so this is a substitution rather than a change of model.

Three things are refused when the configuration is loaded rather than part-way
through a run:

- `projection-backend: jax` with `effects.earth-rotation: false`. The
  constant-pattern branch is a single frequency-domain phase shift with no
  device implementation — and it is already the cheap branch, being the one that
  skips the resampler.
- `projection-backend: jax` without JAX installed, or with JAX unable to run in
  64-bit mode. In 32-bit mode the GPS times and sidereal angles lose the
  precision the delays depend on, and the projection is wrong by of order a
  percent of peak while still looking like strain, so it refuses rather than
  degrading. gwmock turns 64-bit mode on for you; it only fails if something in
  the environment forces it off.
- A `globals.simulator-arguments.duration` longer than 86400 s. The device path
  extrapolates sidereal time linearly from one Astropy anchor and is validated
  to a day. A run of any total length is unaffected — each segment re-anchors —
  so the fix is shorter segments. Note this check is the segment, not the whole
  condition: a compact binary's waveform buffer starts well before its
  coalescence and can be longer than the segment it lands in, and that case is
  still reported by gwmock-signal at generation time.

One more is refused a moment later, when the signal backend is built: a backend
whose constructor does not **name** a `projection_backend` parameter. That
covers a gwmock-signal older than the release which added it, and it covers a
custom backend of your own — including one whose constructor ends in `**kwargs`.
Taking arbitrary keywords is not evidence of using them: such a backend would
accept `projection-backend: jax`, discard it, and produce host output from a run
whose configuration and metadata both record `jax`. Naming the parameter is how
a backend says it honours the setting, so that is what is required.

### Choosing the execution mode

`signal.execution` selects **how** a segment's events are computed,
independently of which library computes them:

| Value                 | Behaviour                                                                  |
| --------------------- | -------------------------------------------------------------------------- |
| `per-event` (default) | Loop over the segment's events, one waveform at a time.                    |
| `batched`             | Hand the whole segment to gwmock-signal's batched entry point in one call. |

```yaml
orchestration:
    signal:
        execution: batched
        waveform-model: IMRPhenomD
        waveform-backend: ripple
```

Batched is the **GPU-capable** path, but this key does not choose a device.
Whether it runs on a GPU depends only on the installed JAX backend:

- `pip install 'gwmock[jax]'` — batched, on the CPU.
- `pip install 'gwmock[cuda]'` — installs the CUDA backend (Linux x86_64, CUDA
  12); runs on a GPU when a compatible device and driver are present, and
  silently falls back to the CPU when they are not.

Nothing in the output distinguishes the two and no warning is raised for the CPU
case. Check with `python -c "import jax; print(jax.devices())"`.

Two constraints. The batched path always generates with `ripple` whatever
`waveform-backend` names — a different library is refused rather than silently
substituted. And it refuses any signal setting it cannot apply
(`waveform-options`, `signal.arguments`, `signal.parameters`), so a
configuration that reaches the generator unchanged is the only one that runs.

GPU and CPU results are **not** bit-identical. Measured on an RTX 2080 Ti, they
agree to ~4e-13 of peak — a sub-sample time shift, not an accuracy difference.
Neither is known to be more correct.

See `examples/signal/execution/batched` for a runnable configuration.

For SGWB studies, use `signal.source-type: sgwb`. Constructor options for the
SGWB backend belong under `signal.arguments`, while spectrum parameters passed
to `simulate(...)` belong under `signal.parameters`:

```yaml
orchestration:
    signal:
        source-type: sgwb
        detectors:
            - ET-Triangle-Sardinia
        minimum-frequency: 5
        parameters:
            omega_ref: 1.0e-9
            spectral_index: 0.0
            reference_frequency: 25.0
        output:
            file_name: sgwb-{{ counter }}.hdf5
```

For the full schema and backend registration options, see the
[Orchestration](orchestration.md) guide.

Transient glitches are configured on the noise side under
`orchestration.noise.arguments.glitches` using public `gwmock-noise` glitch
models. For example:

```yaml
orchestration:
    noise:
        arguments:
            glitches:
                - kind: gengli_blip
                  rate: 0.0011111111111111111
                  amplitude_distribution:
                      distribution: lognormal
                      mean: 1.0
                      std: 0.0
                  population_file: glitches.hdf5
                  psd_file: https://example.org/ET_10_full_cryo_psd.txt
```

`kind: deepextractor` injects real O3 glitch reconstructions, in seven Gravity
Spy classes, coloured against a target PSD and rescaled to a target SNR. It
takes two arguments the parametric models leave optional: `psd_file` and `snr`
are **required** here, where `blip` and `scattered_light` default both to `None`
and emit an uncoloured, unscaled waveform. There is no default to fall back on,
because a reconstruction arrives whitened and amplitude-normalized — the PSD is
what turns it into strain and the SNR is what sets its size.

```yaml
orchestration:
    noise:
        arguments:
            detectors:
                - ET-Triangle-Sardinia
            glitches:
                - kind: deepextractor
                  # Required. Total Poisson rate in Hz, or one rate per class.
                  rate:
                      Blip: 0.0003536
                      Fast_Scattering: 0.001955
                      Koi_Fish: 0.0006158
                      Low_Frequency_Burst: 0.0006542
                      Scattered_Light: 0.002902
                      Tomte: 0.001239
                      Whistle: 0.0003792
                  # Required. Target optimal SNR, scalar or one per class.
                  snr:
                      Blip: 13.95
                      Fast_Scattering: 9.004
                      Koi_Fish: 113.9
                      Low_Frequency_Burst: 12.23
                      Scattered_Light: 11.47
                      Tomte: 13.96
                      Whistle: 10.73
                  # Required. The coloring reference.
                  psd_file: ET_10_full_cryo_psd
                  low_frequency_cutoff: 5.0
                  # Pin the dataset, so the run is reproducible.
                  revision: 144f56880c6e7aa8def31c537ca843b8c9e5bdda
```

| Argument                 | Required | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rate`                   | yes      | Poisson rate in Hz, per interferometer. A number is the total rate, with each event's class drawn uniformly; a mapping gives one rate per class, the total being their sum and each event's class drawn in proportion to its own rate                                                                                                                                                                                             |
| `snr`                    | **yes**  | Target optimal SNR against `psd_file`. A number applies to every class; a mapping gives one target per class. A target may also be a distribution, drawn per event: `{distribution: power_law, minimum, alpha, maximum}` (`maximum` optional) or `{distribution: empirical, samples: [...]}` / `{..., file: path}`. Given in place of the number it applies to every class; given per class, fixed and sampled targets mix freely |
| `psd_file`               | **yes**  | The PSD the whitened reconstructions are coloured with. A bundled name (`ET_10_full_cryo_psd`), a local path, or an http(s) URL to a two-column `.txt`                                                                                                                                                                                                                                                                            |
| `detectors`              | no       | The interferometers this model injects into: one name or a list. Unset applies it to every interferometer of the run. Names are interferometers (`ET1_SARD`), not network presets (`ET-Triangle-Sardinia`)                                                                                                                                                                                                                        |
| `glitch_classes`         | no       | Which of the seven classes to draw from. Defaults to all seven                                                                                                                                                                                                                                                                                                                                                                    |
| `revision`               | no       | Pins the HuggingFace dataset to a branch, tag, or commit SHA. Unset tracks the repository default                                                                                                                                                                                                                                                                                                                                 |
| `low_frequency_cutoff`   | no       | Lower edge of the band the SNR is computed over, in Hz. Defaults to 2.0                                                                                                                                                                                                                                                                                                                                                           |
| `high_frequency_cutoff`  | no       | Upper edge, in Hz. Defaults to Nyquist                                                                                                                                                                                                                                                                                                                                                                                            |
| `amplitude_distribution` | yes      | Multiplier applied on top of the SNR calibration, required of every glitch model. `mean: 1.0, std: 0.0` for no spread                                                                                                                                                                                                                                                                                                             |
| `local_files_only`       | no       | Read the cached dataset without contacting the Hub at all. Defaults to `false`                                                                                                                                                                                                                                                                                                                                                    |
| `repo_id`                | no       | The dataset to draw from. Defaults to `tomdooney/deepextractor-glitch-reconstructions`                                                                                                                                                                                                                                                                                                                                            |

Two of those arguments reach past a single fixed target and a single network:

- **`detectors` is what lets one file describe two designs.** The 10 km triangle
  and the 15 km 2L do not share a noise curve, so they cannot share a
  `psd_file`; give each design its own model, scoped to its interferometers. A
  configuration that does not say what every interferometer gets is refused
  before any strain is generated: an interferometer no model claims, two models'
  PSDs claiming the same interferometer, or a selector naming an interferometer
  the run does not have each raise an error naming the interferometers involved.
  Where an interferometer should carry no glitches, say so with a model at
  `rate: 0.0`. A model without `detectors` applies to every interferometer, as
  before. See
  [Scoping a model to some interferometers](https://leuven-gravity-institute.github.io/gwmock-noise/user_guide/noise_simulation/#scoping-a-model-to-some-interferometers)
  in the gwmock-noise guide.
- **A distribution-valued `snr` keeps a class's tail.** Measured glitch SNRs are
  heavy-tailed, and one fixed target per class discards the tail. `alpha` is the
  exponent of the survival function — the convention a Hill or
  maximum-likelihood tail index is quoted in — so a measured index goes in as
  measured, with `minimum` set to the threshold it was measured above. Setting
  `maximum` renormalizes the law onto `[minimum, maximum]`; leave it unset only
  where `alpha` is comfortably above 1, since below that the mean is infinite
  and a long run eventually draws an SNR no detector could record. An
  `empirical` file holds one SNR per line, or is an HDF5 file with an `snr`
  dataset. Each event's `target_snr` in the batch metadata is the value it was
  drawn with (see
  [Finding which frame contains a glitch](reproducibility.md#finding-which-frame-contains-a-glitch)).
  See
  [Sampled SNR distributions](https://leuven-gravity-institute.github.io/gwmock-noise/user_guide/noise_simulation/#sampled-snr-distributions)
  for the survival functions and what an SNR measured by another pipeline means
  as a target here.

Four things about it are worth knowing before a production run:

- **A `rate` mapping's keys must match `glitch_classes` exactly.** Not a subset
  and not a superset — a missing or unconfigured class is an error, not a
  silently-zero rate. The same holds for an `snr` mapping. Configuring fewer
  classes therefore means narrowing both.
- **`revision` is the difference between a reproducible run and one that tracks
  whatever the Hub served that day.** Pin it to a commit SHA and a regenerated
  dataset holds bit-identical samples for a fixed (version, configuration, seed)
  — compare the content hash, not the container bytes, since each HDF5 file
  embeds its own run's record. Whatever you pin, the run records the concrete
  commit it resolved to, so replaying a run through its metadata fetches that
  commit even as the upstream dataset moves.
- **The dataset is 2.3 GB, fetched lazily on first use** and cached by
  `huggingface_hub`. Later runs reuse the cache after an ETag check; if the Hub
  is unreachable the check is skipped with a warning and the cache is used
  anyway. With nothing cached and no network, the first glitch raises.
- **Below 4096 Hz the backend resamples by linear interpolation, with no
  anti-aliasing filter.** 4096 Hz is the dataset's native rate; under it,
  high-frequency glitch content aliases. The SNR calibration is unaffected,
  being computed after resampling — so what a lower `sampling-frequency` costs
  is the morphology, not the amplitude.

The extra is required: `pip install 'gwmock[deepextractor]'`. Its waveforms are
reconstructions fetched from a HuggingFace dataset rather than generated, so
without it the model raises the moment it first reaches for the dataset.

See `examples/noise/glitches/deepextractor/<network>` for runnable
configurations — one per detector network, each covering that geometry's
interferometers in a single file and writing frames — and
`examples/noise/glitches/deepextractor/et_triangle_and_2l` for both designs in
one file, with per-class power-law SNR tails.

### Registered glitch populations

A `glitches` entry can name a glitch population registered in `gwmock-noise`
instead of describing one model. It expands to the population's class models.
Its other keys are the population's run settings, such as `psd_file`, and
`digest` pins the population those settings build:

```yaml
orchestration:
    noise:
        arguments:
            glitches:
                - population: deepextractor-o3-v1
                  psd_file: ET2026a_10km_psd
                  digest: sha256:bad0306a212c9d810c087763cbc4ff8fa652d50837ccdecef2f5393d6c10c7b9
```

`deepextractor-o3-v1` is the seven DeepExtractor classes at their measured O3
rates and SNR tails; `gwmock-noise`'s documentation describes it. When the
rates, tails or settings no longer build the pinned digest, the configuration is
refused when it is loaded. Print the digest to pin with
`gwmock_noise.populations.get_glitch_population(name, **settings).digest()`.
Name the noise curve by a bundled preset. A path in `psd_file` is part of the
digest, so the pin would differ between machines. A URL is passed to
`gwmock-noise` as written, so the digest stays the same everywhere, but
`gwmock-noise` downloads it for each class each time the configuration is loaded
or run, and the run is recorded as not replayable: nothing records the bytes it
read. A `population` entry takes no `kind`, and can sit beside ordinary model
entries. A population of DeepExtractor classes needs the `deepextractor` extra,
described above.

### Duty cycle

`orchestration.duty-cycle` records when each interferometer observes. The model
is `gwmock-noise`'s (see its "Duty Cycle" guide): unplanned unlocks and relocks
as a Markov chain, with correlated unlocks that take every observing
interferometer down together, and periodic planned maintenance. The strain is
not changed; the run records the duty cycle beside it.

```yaml
orchestration:
    noise:
        arguments:
            psd_file: ET2026a_10km_psd
            seed: 42
            detectors:
                - ET-Triangle-EMR
    duty-cycle:
        mean-uptime: 19440 # s; 5.4 h
        mean-downtime: 5760 # s; 1.6 h
        correlated-downtime-fraction: 0.02
        maintenance:
            strategy: rotating # or coincident, or none
            period: 604800 # s; weekly
            duration: 60480 # s; 10 % of the time
            offset: 198000 # s after GPS 0; Tuesdays at 07:00 UTC
        sampling-frequency: 16
```

Every key is optional; the values above are `gwmock-noise`'s defaults, from
Negri et al. (arXiv:2606.19201) and LVK maintenance practice. An invalid value
is refused when the configuration is loaded. What depends on the network or the
frame length is refused before the first frame is written: a `rotating` schedule
whose windows do not fit one after another in a period, a correlated share the
network cannot reach, or a `sampling-frequency` that does not give each frame a
whole number of samples.

The realization is drawn when the run starts. At the default rates it takes well
under a second for a year of three interferometers; its cost, and the size of
the segment files, grow with the number of unlocks, so a `mean-uptime` of
minutes makes both large.

One realization covers the run's whole span, so every frame sees the same duty
cycle, and a resumed run sees the one it started with. Its seed is `seed`, or is
derived from the noise seed when `seed` is omitted; with neither, the run is
refused. The duty cycle is drawn for the noise's interferometers, or for the
signal's when the run has no noise.

The run records it three ways:

- **A state-vector channel**, `<interferometer>:STATE_VECTOR`, one file per
  interferometer and frame, in `duty-cycle/` beside the other outputs. Bit 0 is
  set while the interferometer observes and bit 1 while it is in planned
  maintenance. The samples are `uint32` at `sampling-frequency`, which must give
  each frame a whole number of samples. `duty-cycle.output` has the shape of
  `noise.output`: `file_name` (required whenever the block is given; its
  extension selects GWF or HDF5), `output_directory`, and `arguments`, which
  takes `channel` only. Templates are expanded as for noise. An existing
  state-vector file is refused without `--overwrite`, as a noise file is.
- **Segment files**, written and recorded with the first frame:
  `duty-cycle-segments.json` holds every interferometer's observing intervals
  and maintenance windows, the model and the seed, and
  `<interferometer>-observing-segments.txt` lists each interferometer's
  observing intervals, one `start end` line each, in GPS seconds. Intervals are
  `[start, end)`. A segment file on disk whose contents differ from the ones
  this run would write is refused without `--overwrite` when a run starts, and
  is replaced, and recorded again, by the next frame the run writes with
  `--overwrite` or after resuming. A resumed run with no frames left to write
  leaves it as it is. Missing ones are missing outputs of the first frame, so
  resuming regenerates the run from its first frame. The segment files have
  fixed names, so without `--overwrite` two runs can write to one duty-cycle
  `output_directory` only if they write identical segment files and their
  state-vector file names differ.
- **Observing flags** in the batch metadata: every injected signal with a merger
  time (`coa_time`) gets `observing: {<interferometer>: true | false}` at that
  time, for each interferometer of the duty cycle, and every glitch gets
  `observing: true | false` for its own interferometer at its peak. A signal
  with no merger time, such as a stochastic background, gets no flag.

To carry the state vector into the merged strain file, pass it to `gwmock merge`
with `--copy-channel`; see "Reading Data".

## Template Variables

You can use Jinja2-style templates in configuration values such as file names
and channel names:

```yaml
orchestration:
    noise:
        arguments:
            detectors:
                - E1_triangle_emr
                - E2_triangle_emr
                - E3_triangle_emr
        output:
            file_name:
                'E-{{ detectors }}_STRAIN_NOISE-{{ start_time }}-{{ duration
                }}.hdf5'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

In this example, `file_name` is automatically expanded for each detector being
processed.

**Common variables:**

- `{{ start_time }}`: GPS start time from globals
- `{{ duration }}`: Segment duration from globals
- `{{ detectors }}`: Current detector being processed. A network alias such as
  `ET-Triangle-EMR` expands to one file/channel per interferometer, with
  `{{ detectors }}` resolving to the per-interferometer token (`ET1_EMR`,
  `ET2_EMR`, `ET3_EMR`)

## Checkpointing

gwmock automatically creates checkpoints during long simulations. If a process
is interrupted:

1. A `.gwmock_checkpoint/simulation.checkpoint.json` file is saved in the
   working directory
2. Rerun the same command to resume from the last checkpoint
3. The tool automatically detects and continues from where it left off

```bash
# Start simulation
gwmock simulate config.yaml

# If interrupted (Ctrl+C, crash, etc.), resume with same command
gwmock simulate config.yaml
```

The checkpoint contains:

- Simulator state
- Progress information
- Already-generated file tracking
- The fingerprint of the run that wrote it — which configuration, which output
  and metadata directories, and, for a population file on local disk, that
  file's contents (a population fetched from a URL contributes its address only,
  since identifying a run does not download the catalogue again)

A resume only continues from a checkpoint it can attribute to the command being
run. It refuses when the fingerprints differ, and when the checkpoint carries no
fingerprint at all because it was written by a version from before the field
existed (`gwmock` 0.13.0 and earlier). Without that check, a second
configuration run from the same working directory resumes from the first's
checkpoint and skips the batches it recorded, so the outputs those batches would
have produced are never written and the run still exits successfully.

Both refusals name the checkpoint file, so it can be moved or deleted to start
fresh. To start fresh without touching it — for an automated run that cannot
answer a prompt — pass `--ignore-checkpoint`:

```bash
gwmock simulate config.yaml --ignore-checkpoint
```

## Resource Usage Summary

After every successful simulation, gwmock writes a `resource_usage_summary.json`
file to the working directory. This file records CPU time, peak memory usage,
and wall time for the run. It is always written (overwriting any previous
summary) and is not controlled by a flag.

## Best Practices

1. **Use templates**: Leverage Jinja2 templates for dynamic configuration
2. **Set seeds**: Always set `seed` for reproducibility
3. **Check space**: Ensure sufficient disk space before long runs
4. **Use dry-run**: Test configurations with `--dry-run` before full simulation
5. **Organize outputs**: Use descriptive `output-directory` and
   `metadata-directory` names
