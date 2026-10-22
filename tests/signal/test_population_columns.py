"""The known population-only columns cover what gwmock-pop and gwmock's own examples write.

A population column the waveform does not read stops a run unless it is known to be
population-only or declared ignored. If gwmock-pop starts writing a column outside the known set
plus what the waveform reads, every run on its output would stop -- so this fails first, naming it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gwmock.signal.device_chunks import BATCHED_PARAMETERS, canonicalise_parameters
from gwmock.signal.population_columns import (
    KNOWN_COLUMNS,
    ZERO_ONLY_COLUMNS,
    require_redundant_columns_agree,
    unread_columns_to_refuse,
)

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _refused(columns) -> list[str]:
    canonical = set(canonicalise_parameters(dict.fromkeys(columns, 0.0)))
    return unread_columns_to_refuse(canonical - BATCHED_PARAMETERS, ignored=())


def test_the_canonical_parameter_names_gwmock_pop_exports_are_covered():
    from gwmock_pop.constants import CBC_PARAMETER_NAMES

    assert _refused(CBC_PARAMETER_NAMES) == []


_PRESETS = ["bbh_flat", "bbh_gwtc4", "bns_flat", "nsbh_flat"]


@pytest.mark.parametrize("preset", _PRESETS)
def test_a_real_gwmock_pop_draw_is_covered_and_its_zero_only_columns_are_zero(preset):
    from gwmock_pop.simulators.graph import GraphSimulator

    draw = GraphSimulator.from_preset(preset, seed=1).simulate(8)

    assert _refused(draw.keys()) == []
    for name in ZERO_ONLY_COLUMNS & set(draw.keys()):
        assert not np.any(np.asarray(draw[name], dtype=float)), f"{preset} draws a nonzero {name}"


def test_the_mdc_list_and_catalogue_writers_are_covered():
    from gwmock_pop.catalogue import mdc_list, population

    assert _refused(mdc_list._POPULATION_COLUMNS) == []
    assert _refused(mdc_list._INTRINSIC_COLUMNS) == []
    assert _refused(population.PARAMETER_NAMES) == []


def test_the_shipped_example_catalogues_are_covered():
    catalogues = sorted(_EXAMPLES.rglob("*_population.csv"))
    assert catalogues
    for catalogue in catalogues:
        if catalogue.name == "cw_population.csv":
            continue  # continuous waves are not generated through a CBC waveform backend
        with catalogue.open(encoding="utf-8") as stream:
            assert _refused(next(csv.reader(stream))) == [], catalogue


def test_a_misspelt_spin_is_not_known():
    assert _refused(["detector_frame_mass_1", "spin1_z"]) == ["spin1_z"]


def test_no_known_column_is_one_the_waveform_reads():
    assert not KNOWN_COLUMNS & BATCHED_PARAMETERS


def _events(draw) -> list[dict]:
    names = list(draw.keys())
    columns = {name: np.asarray(draw[name], dtype=float) for name in names}
    return [{name: float(columns[name][i]) for name in names} for i in range(len(columns[names[0]]))]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "gwmock-pop before 0.18.0: its presets draw theta_jn independently of inclination although "
        "they draw no in-plane spin, so J cannot lie along L as it must. From gwmock-pop 0.18.0 the "
        "presets write no theta_jn and no f_ref; remove this mark when the dependency floor is "
        "raised to it"
    ),
)
@pytest.mark.parametrize("preset", _PRESETS)
def test_a_real_gwmock_pop_draw_is_left_out_consistently(preset):
    """Every conditional column of a real draw holds its condition, for a run referencing 20 Hz.

    Before gwmock-pop 0.18.0 the presets draw ``f_ref = 20``, so a run on their output sets
    ``f_ref: 20`` under ``signal.waveform-backend-arguments``; 20 Hz is what is passed here. From
    0.18.0 they write neither ``theta_jn`` nor ``f_ref``, and every column left is one the waveform
    reads or one left out unconditionally.
    """
    from gwmock_pop.simulators.graph import GraphSimulator

    draw = GraphSimulator.from_preset(preset, seed=1).simulate(8)
    canonical = set(canonicalise_parameters(dict.fromkeys(draw.keys(), 0.0)))

    require_redundant_columns_agree(
        canonical - BATCHED_PARAMETERS, ignored=(), reader="test", events=_events(draw), reference_frequency=20.0
    )
