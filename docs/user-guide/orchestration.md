# Orchestration

`gwmock` now expects adapter-backed `orchestration:` configs for new runs. This
surface replaces the legacy `simulators:` layout and splits configuration into
three explicit sections:

- `orchestration.population`
- `orchestration.signal`
- `orchestration.noise`

One key sits beside those three rather than inside them, because it applies to
everything the run writes: `orchestration.include-injection-parameters`.

## Migration from `simulators:`

The legacy `simulators:` schema has been removed. Configs that use a top-level
`simulators:` key are now rejected at load time with a clear error message. All
configurations must use the `orchestration:` schema described below. For the
full configuration shape and examples, see the
[Configuration Files](configuration.md) guide.

The `orchestration:` section may contain `population`, `signal`, `noise`, or a
combination of them. CBC-style transient signal generation uses `population`
plus `signal`. Stationary SGWB generation can use `signal` without `population`
by setting `signal.source-type: sgwb`.

```yaml
orchestration:
    population:
        backend: file
        n-samples: 1
        arguments:
            path: population.h5
    signal:
        detectors:
            - H1
        output:
            file_name: signal-{{ counter }}.hdf5
            arguments:
                channel: H1:STRAIN
    noise:
        arguments:
            seed: 7
        output:
            file_name: noise-{{ counter }}.hdf5
```

Signal-only SGWB example:

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

## Injection parameters in the released files

Each HDF5 file a run writes carries the run's metadata record inside it, so the
data describes itself. The injection parameters are excluded from that embedded
copy unless the run asks for them:

```yaml
orchestration:
    include-injection-parameters: false # default
    population: ...
    signal: ...
```

The default keeps a blind mock data challenge blind: it is released as the
strain files alone, and the parameters the signals were injected with are the
answer its participants are asked to find. Set it to `true` when the data is for
a training set, a benchmark, or a challenge released with its solutions.

- It applies to the whole run — every signal and noise output of every batch —
  not to one section.
- It never changes the metadata sidecar, which always records the injection
  parameters. The sidecar is the producer's copy; the flag governs the files
  that leave.
- `.npy` and `.gwf` outputs carry no record either way, having nowhere to put
  one.
- No adapter has to do anything to honour it. gwmock writes the record into the
  artifact after the backend has produced it, in the one place that also stamps
  the [strain schema](reading-data.md), so a third-party population, signal or
  noise backend gets the behaviour without implementing any of it.

`gwmock merge` carries the same switch as `--include-injection-parameters`, with
the same default: the sidecars it reads hold the parameters even when the files
it merges do not, so a merge would otherwise re-publish what each run withheld.

See
[Configuration Files](configuration.md#injection-parameters-in-the-data-files)
for what the flag does not cover, and [Reading data](reading-data.md) for how a
consumer reads the embedded record.

For protocol details and third-party backend integration, see
[Protocol Contracts](protocols.md) and [Extensibility](extensibility.md).
