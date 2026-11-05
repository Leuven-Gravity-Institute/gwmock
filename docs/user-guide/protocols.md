# Protocol Contracts

This page defines the stable public interfaces that `gwmock` consumes from the
split-out `gwmock-pop`, `gwmock-signal`, and `gwmock-noise` packages. If a
third-party package implements the same public protocol, `gwmock` can consume it
without depending on that package's internal classes.

For packaging and registration guidance, see [Extensibility](extensibility.md).

## Population protocol: `GWPopSimulator`

Upstream definition:
[`gwmock_pop.protocols.simulator.GWPopSimulator`](https://github.com/Leuven-Gravity-Institute/gwmock-pop/blob/main/src/gwmock_pop/protocols/simulator.py)

`GWPopSimulator` is a `@runtime_checkable` `typing.Protocol`.

| Member                          | Type                                                        |
| ------------------------------- | ----------------------------------------------------------- |
| `parameter_names`               | `Sequence[str]` property; stable across calls               |
| `source_type`                   | `str` property; must be non-empty                           |
| `simulate(n_samples, **kwargs)` | `Mapping[str, ArrayLike]`; each value is length `n_samples` |

`gwmock` validates the backend with `isinstance(obj, GWPopSimulator)`. The
returned mapping must use the same key order as `parameter_names`, and every
array-like value must have the same leading dimension so `gwmock` can slice the
batch into per-event parameter dictionaries.

## Signal protocol: `GWSimulator`

Upstream definition:
[`gwmock_signal.simulator.GWSimulator`](https://github.com/Leuven-Gravity-Institute/gwmock-signal/blob/main/src/gwmock_signal/simulator.py)

| Member                                  | Type                                                        |
| --------------------------------------- | ----------------------------------------------------------- |
| `required_params`                       | `frozenset[str]` property                                   |
| `simulate(params, detector_names, ...)` | Returns `DetectorStrainStack`                               |
| `register_waveform_model` _(optional)_  | `(name: str, factory: Callable \| WaveformBackend) -> None` |

For third-party packages, the preferred approach is to subclass
`gwmock_signal.GWSimulator` directly. A class with the same public
`required_params` and `simulate(...)` surface is also compatible with the
consumer-side contract.

The stable boundary is `simulate(...)`. `gwmock` expects a `DetectorStrainStack`
and should not need backend-specific helpers such as `generate_polarizations()`.
`register_waveform_model(...)` is only relevant for backends that accept
callable waveform overrides.

`gwmock` calls `simulate(params, detector_names, ...)` with the keyword
arguments `background`, `sampling_frequency`, `minimum_frequency` and `effects`.
`effects` is a `gwmock_signal.projection.ProjectionEffects`: the projection
effects the run's configuration chooses (`orchestration.signal.effects`, see
[Configuration](configuration.md)). gwmock-signal's own simulators accept it
from 0.22.0. A third-party backend must accept it in its own `simulate`
signature, whether or not it subclasses `GWSimulator`: `simulate` is abstract
there, so upgrading gwmock-signal does not change a subclass's override.
`gwmock` no longer passes the `earth_rotation` keyword that releases before
`gwmock-signal` 0.22.0 took, and a backend that does not accept `effects` fails
with a `TypeError`, which `gwmock` reports as a refusal. To migrate such a
backend, accept `effects` and read `effects.earth_rotation` where
`earth_rotation` was read; apply the other effects, or refuse the ones it does
not implement.

## Noise protocol: `NoiseSimulator`

Upstream definitions:
[`gwmock_noise.simulators.protocol.NoiseSimulator`](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/protocol.py)
and
[`gwmock_noise.open_stream(...)`](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/src/gwmock_noise/simulators/streaming.py)

`NoiseSimulator` is a `@runtime_checkable` `typing.Protocol`.

| Member                                                | Type                                    |
| ----------------------------------------------------- | --------------------------------------- |
| `duration`, `sampling_frequency`, `detectors`, `seed` | Required attributes                     |
| `generate(...)`                                       | `dict[str, np.ndarray]` one-shot output |
| `generate_stream(chunk_duration, ...)`                | `Iterator[dict[str, np.ndarray]]`       |
| `metadata`                                            | `dict` property                         |

`gwmock` consumes the streaming form through `gwmock_noise.open_stream(...)`.
The contract is stateful continuation across yielded chunks: chunk `N+1` must
continue the filter state of chunk `N`, so concatenating consecutive chunks
matches a single longer run.

## Backend references

The protocol-oriented backend reference is the `backend:` field in the
orchestration config. A backend identifier may resolve as:

1. A built-in alias registered by `gwmock`.
2. An entry point in a `gwmock.*` group.
3. A `module:Class` reference imported directly by `gwmock`.
4. A legacy `module.Class` reference, which still works but emits a
   `DeprecationWarning`.

For third-party integrations, the important part is that resolution ends in a
class or instance that satisfies the relevant upstream protocol.

## Worked example: third-party `GWPopSimulator`

Any class with the required public members satisfies the population protocol; it
does not need to inherit from a gwmock-specific base class.

```python
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp


class GalacticBinaryPopulation:
    @property
    def parameter_names(self) -> Sequence[str]:
        return (
            "coa_time",
            "chirp_mass",
            "luminosity_distance",
            "right_ascension",
            "declination",
            "polarization_angle",
        )

    @property
    def source_type(self) -> str:
        return "bbh"

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, jnp.ndarray]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        return {
            "coa_time": jnp.linspace(1_577_491_218.0, 1_577_491_218.0 + n_samples, n_samples),
            "chirp_mass": jnp.full((n_samples,), 28.0),
            "luminosity_distance": jnp.full((n_samples,), 400.0),
            "right_ascension": jnp.zeros((n_samples,)),
            "declination": jnp.zeros((n_samples,)),
            "polarization_angle": jnp.zeros((n_samples,)),
        }
```

Once that class is importable from `my_pkg.populations`, expose it through an
entry point:

```toml
[project.entry-points."gwmock.population"]
galactic_binaries = "my_pkg.populations:GalacticBinaryPopulation"
```

Then reference the entry-point alias in YAML:

```yaml
orchestration:
    population:
        backend: galactic_binaries
        n-samples: 128
        arguments:
            catalog_path: /data/catalogs/galactic-binaries.h5
    signal:
        backend: my_pkg.signals:GalacticBinarySimulator
        waveform-model: IMRPhenomXPHM
        detectors:
            - E1_triangle_emr
            - E2_triangle_emr
            - E3_triangle_emr
    noise:
        backend: my_pkg.noise:LaserShotNoise
        arguments:
            seed: 42
            psd_file: ET_10_full_cryo_psd.txt
```

The same pattern applies to signal and noise backends: implement the upstream
protocol, make the class importable, and reference it through the corresponding
`backend:` field when that orchestration section is resolved through the public
backend registry.
