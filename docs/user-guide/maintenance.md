# Maintenance & Quality

This page documents how `gwmock` is kept correct and healthy over time: the test
suite, the automated dependency-management policies, and the release process —
the information a user needs to judge whether they can depend on the package.

## Testing and continuous integration

Every pull request and every push to `main` runs the full
[CI workflow](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/ci.yml):

- **Test matrix** — the pytest suite (under `tests/`, mirroring the source
  layout) runs on Ubuntu and macOS across Python 3.13 and 3.14.
- **Lowest-resolution job** — one job installs every direct dependency at its
  _minimum_ declared version (`uv sync --resolution lowest-direct`) on the
  oldest supported Python. This proves the version floors in `pyproject.toml`
  are real, not aspirational — a floor that breaks the test suite cannot ship.
- **Coverage** — results are uploaded to
  [Codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock) on every run,
  so coverage changes are visible per pull request.

Static checks run alongside the tests:

- **pre-commit** — formatting and lint hooks (including
  [Ruff](https://github.com/astral-sh/ruff)) run on every commit locally and are
  enforced on every pull request by
  [pre-commit.ci](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock/main).
- **CodeQL** — GitHub's static security analysis runs on the repository.

To run the suite locally:

```bash
uv sync --group dev
uv run pytest
```

## Dependency management

Dependency maintenance is deliberately split into two automated systems with
disjoint responsibilities, so each version constraint has exactly one owner:

### Version floors: the SPEC 0 policy

`gwmock` follows
[Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/):
support for a Python version is dropped 3 years (36 months) after its initial
release, and support for a dependency's minor series 2 years (24 months) after
its first stable release. SPEC 0 defines this window for the core scientific
ecosystem (NumPy, SciPy, Astropy, …); `gwmock` applies the same 24-month window
to **all** of its runtime dependencies, so its supported range stays compatible
with the packages it builds on.

The policy is enforced mechanically, not by hand:

- The policy is declared in `pyproject.toml` under
  `[tool.dependency-support-policy]` (`policy = "spec0"`).
- A scheduled workflow
  ([`support_floor_update.yml`](https://github.com/Leuven-Gravity-Institute/gwmock/blob/main/.github/workflows/support_floor_update.yml))
  runs monthly, recomputes the SPEC 0 floors, and opens an auto-merging pull
  request raising the lower bounds in `pyproject.toml`.
- The CI lowest-resolution job (above) then verifies the package actually works
  at those floors before the pull request can merge.

A floor may sit _above_ the SPEC 0 minimum when a feature requires it; such
exceptions are annotated inline in `pyproject.toml` with the reason and the SPEC
0 floor they supersede.

The ecosystem sibling packages (`gwmock-signal`, `gwmock-noise`, `gwmock-pop`)
are excluded from the SPEC 0 mechanism and are instead kept current by Renovate,
since they evolve in lock-step with `gwmock` itself.

### Everything else: Renovate

[Renovate](https://docs.renovatebot.com/) (configured in `renovate.json`,
extending `config:recommended` and `config:best-practices`) automates all
remaining dependency chores:

| What                              | Policy                                                              |
| --------------------------------- | ------------------------------------------------------------------- |
| Runtime patch & minor updates     | Grouped, auto-merged after a 3-day minimum release age, gated on CI |
| Runtime major updates             | **Never automated** — always require human review                   |
| Lock file (`uv.lock`) maintenance | Refreshed weekly (Monday before 04:00 UTC), auto-merged             |
| GitHub Actions                    | Grouped weekly, pinned to commit digests, auto-merged after 3 days  |
| pre-commit hooks                  | Auto-merged                                                         |

Every automated merge still has to pass the full CI matrix, including the
lowest-resolution job. A
[Dependency Dashboard](https://github.com/Leuven-Gravity-Institute/gwmock/issues)
issue tracks all pending and blocked updates.

## Releases

Releases are scheduled and automated rather than ad hoc:

- **Cadence** — a release is cut **every Tuesday at 00:00 UTC** whenever
  something that reaches the built package has changed since the previous
  release (`scheduled_release.yml`): the sources, the packaging configuration,
  the readme, the license, or the example configurations and catalogues that
  ship in the wheel. CI, tests, docs and dev-dependency bumps alone do not cut a
  release. Emergency bugfix releases can be triggered manually.
- **Versioning** — versions are derived from git tags via
  `uv-dynamic-versioning`; there is no hand-edited version string. Pull request
  titles follow Conventional Commits (enforced by a PR-title check), which
  drives the version bump and the changelog. While the package is pre-1.0, a
  breaking change bumps the **minor** rather than the major, because `1.x`
  declares a stable API: `scripts/release_version.py` applies that rule to the
  proposal `git-cliff` makes. Dispatching `scheduled_release.yml` with
  `allow_major_bump` releases `1.0.0` when it is meant.
- **Changelog** — generated with [git-cliff](https://git-cliff.org/) from the
  commit history. A rolling **draft release** on the
  [GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock/releases)
  always previews what the next release will contain.
- **Publishing** — each release is published to
  [PyPI](https://pypi.org/project/gwmock/) (with TestPyPI used for release
  rehearsal) and archived on Zenodo with a citable DOI
  ([10.5281/zenodo.17925458](https://doi.org/10.5281/zenodo.17925458)).

## Where to look

| Concern              | File / location                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| Test suite           | `tests/`                                                                                          |
| CI matrix            | `.github/workflows/ci.yml`                                                                        |
| SPEC 0 policy        | `pyproject.toml` `[tool.dependency-support-policy]`, `.github/workflows/support_floor_update.yml` |
| Renovate policy      | `renovate.json`                                                                                   |
| Release automation   | `.github/workflows/scheduled_release.yml`, `draft_release.yml`, `publish.yml`                     |
| Lint & format config | `.pre-commit-config.yaml`, `pyproject.toml` (Ruff)                                                |
| Contribution process | [Contributing](../contributing.md)                                                                |
