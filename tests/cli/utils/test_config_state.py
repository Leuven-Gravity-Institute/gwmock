"""Unit tests for ConfigState."""

from __future__ import annotations

import pytest

from gwmock.cli.utils.config_state import (
    ConfigState,
    _clean_empty,
    _clean_none,
    _delete,
    _get,
    _set,
)


class TestConfigStateSet:
    def test_set_noise_psd(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        assert state.get("noise", "psd") == "ET_10_full_cryo_psd"

    def test_set_noise_seed_coerces_to_int(self):
        state = ConfigState()
        state.set("noise", "seed", "42")
        assert state.get("noise", "seed") == 42
        assert isinstance(state.get("noise", "seed"), int)

    def test_set_noise_detectors_splits_to_list(self):
        state = ConfigState()
        state.set("noise", "detectors", "E0 E1 E2")
        assert state.get("noise", "detectors") == ["E0", "E1", "E2"]

    def test_set_noise_minimum_frequency_coerces_to_float(self):
        state = ConfigState()
        state.set("noise", "minimum-frequency", "5.0")
        assert state.get("noise", "minimum-frequency") == 5.0

    def test_set_noise_minimum_frequency_writes_the_key_the_noise_backend_reads(self):
        state = ConfigState()
        state.set("noise", "minimum-frequency", "5.0")
        assert _get(state._data, ["orchestration", "noise", "arguments"]) == {"low_frequency_cutoff": 5.0}

    def test_set_signal_waveform_model(self):
        state = ConfigState()
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        assert state.get("signal", "waveform-model") == "IMRPhenomXPHM"

    def test_set_signal_earth_rotation_bool(self):
        state = ConfigState()
        state.set("signal", "earth-rotation", "true")
        assert state.get("signal", "earth-rotation") is True
        state.set("signal", "earth-rotation", "false")
        assert state.get("signal", "earth-rotation") is False

    def test_set_globals_sampling_frequency(self):
        state = ConfigState()
        state.set("globals", "sampling-frequency", "4096")
        assert state.get("globals", "sampling-frequency") == 4096

    def test_set_globals_total_duration_keeps_string(self):
        state = ConfigState()
        state.set("globals", "total-duration", "1 day")
        assert state.get("globals", "total-duration") == "1 day"

    def test_set_population_n_samples(self):
        state = ConfigState()
        state.set("population", "n-samples", "10")
        assert state.get("population", "n-samples") == 10

    def test_set_unknown_section_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown section"):
            state.set("unknown", "key", "value")

    def test_set_unknown_key_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown key"):
            state.set("noise", "nonexistent", "value")

    def test_set_invalid_bool_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="boolean"):
            state.set("signal", "earth-rotation", "maybe")

    def test_set_invalid_int_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="invalid literal"):
            state.set("noise", "seed", "not_a_number")


class TestConfigStateGet:
    def test_get_unset_returns_none(self):
        state = ConfigState()
        assert state.get("noise", "psd") is None

    def test_get_unknown_key_returns_none(self):
        state = ConfigState()
        assert state.get("noise", "nonexistent") is None

    def test_get_section_returns_none_when_empty(self):
        state = ConfigState()
        assert state.get_section("noise") is None

    def test_get_section_returns_dict_when_set(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        section = state.get_section("noise")
        assert section is not None
        assert "arguments" in section


class TestConfigStateToDict:
    def test_empty_state_produces_empty_dict(self):
        state = ConfigState()
        assert state.to_dict() == {}

    def test_single_setting(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        d = state.to_dict()
        assert d["orchestration"]["noise"]["arguments"]["psd_file"] == "ET_10_full_cryo_psd"

    def test_multiple_sections(self):
        state = ConfigState()
        state.set("noise", "psd", "ET_10_full_cryo_psd")
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        state.set("globals", "sampling-frequency", "4096")
        d = state.to_dict()
        assert "orchestration" in d
        assert "globals" in d
        assert d["orchestration"]["signal"]["waveform-model"] == "IMRPhenomXPHM"
        assert d["globals"]["simulator-arguments"]["sampling-frequency"] == 4096

    def test_empty_sections_are_excluded(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.reset("noise")
        d = state.to_dict()
        assert "orchestration" not in d or not d.get("orchestration")


class TestConfigStateReset:
    def test_reset_section(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.set("noise", "seed", "42")
        state.reset("noise")
        assert state.get_section("noise") is None
        # Other sections unaffected
        state.set("signal", "waveform-model", "IMRPhenomXPHM")
        state.reset("noise")
        assert state.get("signal", "waveform-model") == "IMRPhenomXPHM"

    def test_reset_all(self):
        state = ConfigState()
        state.set("noise", "psd", "test")
        state.set("signal", "waveform-model", "test")
        state.reset()
        assert state.to_dict() == {}
        assert state.is_empty

    def test_reset_unknown_section_raises(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown section"):
            state.reset("nonexistent")


class TestConfigStateGlitches:
    def test_add_glitch(self):
        state = ConfigState()
        idx = state.add_glitch("gengli_blip")
        assert idx == 0
        d = state.to_dict()
        glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
        assert len(glitches) == 1
        assert glitches[0]["kind"] == "gengli_blip"

    def test_add_multiple_glitches(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.add_glitch("blip")
        d = state.to_dict()
        assert len(d["orchestration"]["noise"]["arguments"]["glitches"]) == 2

    def test_remove_glitch(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.add_glitch("blip")
        removed = state.remove_glitch(0)
        assert removed["kind"] == "gengli_blip"
        d = state.to_dict()
        glitches = d["orchestration"]["noise"]["arguments"]["glitches"]
        assert len(glitches) == 1
        assert glitches[0]["kind"] == "blip"

    def test_remove_last_glitch_cleans_key(self):
        state = ConfigState()
        state.add_glitch("gengli_blip")
        state.remove_glitch(0)
        d = state.to_dict()
        noise_args = d.get("orchestration", {}).get("noise", {}).get("arguments", {})
        assert "glitches" not in noise_args

    def test_remove_glitch_out_of_range_raises(self):
        state = ConfigState()
        with pytest.raises(IndexError):
            state.remove_glitch(0)


class TestConfigStateBatch:
    def test_set_batch_resource(self):
        state = ConfigState()
        state.set_batch_resource("nodes", "4")
        d = state.to_dict()
        assert d["batch"]["resources"]["nodes"] == "4"

    def test_set_batch_submit(self):
        state = ConfigState()
        state.set_batch_submit("account", "myaccount")
        d = state.to_dict()
        assert d["batch"]["submit"]["account"] == "myaccount"


class TestConfigStateLoad:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "orchestration:\n  noise:\n    arguments:\n      psd_file: ET_10_full_cryo_psd\n      seed: 42\n"
        )
        state = ConfigState()
        state.load(config_file)
        assert state.get("noise", "psd") == "ET_10_full_cryo_psd"
        assert state.get("noise", "seed") == 42

    def test_load_invalid_yaml_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- just\n- a\n- list\n")
        state = ConfigState()
        with pytest.raises(ValueError, match="YAML mapping"):
            state.load(config_file)


class TestConfigStateValidate:
    def test_empty_config_is_invalid(self):
        state = ConfigState()
        valid, error = state.validate()
        assert not valid
        assert error

    def test_noise_only_is_valid(self):
        state = ConfigState()
        state.set("noise", "psd", "test_psd")
        state.set("noise", "seed", "42")
        valid, _ = state.validate()
        assert valid


class TestTheNestedDictHelpers:
    """The four helpers every setter and getter goes through.

    They survived the first mutation run because the tests above exercise them only through
    ``set``/``get`` on paths that exist, where a wrong slice or a swapped boolean operator gives
    the same answer as the right one.
    """

    def test_a_missing_intermediate_level_gives_the_default(self):
        assert _get({"a": {}}, ["a", "b", "c"], "fallback") == "fallback"

    def test_a_non_mapping_on_the_way_down_gives_the_default(self):
        """``globals`` set to a scalar by hand must not raise on the way to a nested key."""
        assert _get({"a": 5}, ["a", "b"], "fallback") == "fallback"

    def test_the_default_is_returned_for_a_missing_leaf(self):
        assert _get({"a": {"b": {}}}, ["a", "b", "c"], 7) == 7

    def test_the_default_is_carried_down_rather_than_replaced_by_none(self):
        """The default is passed to every level's ``get``, so a partially-present path still ends
        at the caller's default instead of at ``None``."""
        assert _get({}, ["a", "b"], []) == []

    def test_a_present_value_is_returned_rather_than_the_default(self):
        assert _get({"a": {"b": 1}}, ["a", "b"], 99) == 1

    def test_setting_creates_the_intermediate_levels(self):
        data: dict = {}
        _set(data, ["a", "b", "c"], 1)
        assert data == {"a": {"b": {"c": 1}}}

    def test_setting_keeps_the_siblings_it_did_not_touch(self):
        data = {"a": {"keep": 1}}
        _set(data, ["a", "b"], 2)
        assert data == {"a": {"keep": 1, "b": 2}}

    def test_setting_a_single_key_needs_no_intermediate_level(self):
        data: dict = {}
        _set(data, ["only"], 1)
        assert data == {"only": 1}

    def test_deleting_removes_only_the_leaf(self):
        data = {"a": {"b": 1, "c": 2}}
        _delete(data, ["a", "b"])
        assert data == {"a": {"c": 2}}

    def test_deleting_a_top_level_key(self):
        data = {"a": 1, "b": 2}
        _delete(data, ["a"])
        assert data == {"b": 2}

    def test_deleting_an_absent_leaf_changes_nothing(self):
        data = {"a": {"b": 1}}
        _delete(data, ["a", "missing"])
        assert data == {"a": {"b": 1}}

    def test_deleting_through_an_absent_level_changes_nothing(self):
        """Both halves of the guard matter: a missing key and a non-mapping have to stop the walk,
        and stopping means returning rather than indexing into whatever is there."""
        data = {"a": {"b": 1}}
        _delete(data, ["missing", "b"])
        assert data == {"a": {"b": 1}}

    def test_deleting_through_a_scalar_changes_nothing(self):
        data = {"a": 5}
        _delete(data, ["a", "b"])
        assert data == {"a": 5}

    def test_cleaning_drops_an_empty_mapping_but_keeps_the_rest(self):
        assert _clean_empty({"a": {}, "b": 1}) == {"b": 1}

    def test_cleaning_drops_an_empty_list(self):
        assert _clean_empty({"a": [], "b": 1}) == {"b": 1}

    def test_cleaning_drops_a_branch_that_is_empty_all_the_way_down(self):
        assert _clean_empty({"a": {"b": {}}}) == {}

    def test_cleaning_keeps_falsy_scalars(self):
        """0, False and "" are settings a user chose; only *empty containers* are noise."""
        assert _clean_empty({"a": 0, "b": False, "c": ""}) == {"a": 0, "b": False, "c": ""}

    def test_cleaning_keeps_every_key_after_an_empty_one(self):
        """The loop has to skip an empty value and carry on: stopping there would silently drop the
        rest of the section."""
        assert _clean_empty({"first": {}, "second": 1, "third": 2}) == {"second": 1, "third": 2}

    def test_cleaning_keeps_every_key_after_an_empty_list(self):
        assert _clean_empty({"first": [], "second": 1, "third": 2}) == {"second": 1, "third": 2}

    def test_cleaning_recurses_into_lists(self):
        assert _clean_empty({"a": [{"b": {}, "c": 1}]}) == {"a": [{"c": 1}]}

    def test_none_valued_keys_are_dropped_on_load(self):
        assert _clean_none({"a": None, "b": 1}) == {"b": 1}

    def test_none_valued_keys_are_dropped_at_every_level(self):
        assert _clean_none({"a": {"b": None, "c": 2}}) == {"a": {"c": 2}}

    def test_list_entries_are_cleaned_rather_than_replaced(self):
        """Each entry is cleaned in place; cleaning something else and keeping the length would
        leave a list of identical values."""
        assert _clean_none([{"a": None, "keep": 1}, {"b": 2}]) == [{"keep": 1}, {"b": 2}]

    def test_a_none_inside_a_list_is_kept(self):
        """Only mapping *values* are dropped: a list is positional, so dropping an entry would
        renumber the rest."""
        assert _clean_none({"a": [None, 1]}) == {"a": [None, 1]}


class TestBooleanCoercion:
    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "yes", "YES", "1"])
    def test_the_spellings_that_mean_true(self, raw):
        state = ConfigState()
        state.set("signal", "earth-rotation", raw)
        assert state.get("signal", "earth-rotation") is True

    @pytest.mark.parametrize("raw", ["false", "FALSE", "False", "no", "NO", "0"])
    def test_the_spellings_that_mean_false(self, raw):
        state = ConfigState()
        state.set("signal", "earth-rotation", raw)
        assert state.get("signal", "earth-rotation") is False

    @pytest.mark.parametrize("raw", ["y", "n", "on", "off", "2", ""])
    def test_anything_else_is_refused_rather_than_guessed(self, raw):
        """A silent wrong answer here turns Earth rotation on or off for a whole run."""
        state = ConfigState()
        with pytest.raises(ValueError, match="boolean"):
            state.set("signal", "earth-rotation", raw)

    def test_a_list_is_split_on_whitespace(self):
        state = ConfigState()
        state.set("noise", "detectors", "  H1   L1  V1 ")
        assert state.get("noise", "detectors") == ["H1", "L1", "V1"]

    def test_an_empty_list_value_gives_an_empty_list(self):
        state = ConfigState()
        state.set("noise", "detectors", "   ")
        assert state.get("noise", "detectors") == []


class TestRangeConstraints:
    @pytest.mark.parametrize(
        ("section", "key", "raw", "message"),
        [
            ("population", "n-samples", "0", "n-samples must be at least 1"),
            ("batch", "chunks-n-chunks", "0", "chunks-n-chunks must be at least 1"),
            ("globals", "sampling-frequency", "0", "sampling-frequency must be at least 1"),
            ("globals", "duration", "0", "duration must be at least 1"),
            ("globals", "seed", "-1", "seed must be non-negative"),
            ("noise", "minimum-frequency", "-0.5", "minimum-frequency must be non-negative"),
        ],
    )
    def test_a_value_below_the_floor_is_refused_with_its_own_message(self, section, key, raw, message):
        state = ConfigState()
        with pytest.raises(ValueError, match=message):
            state.set(section, key, raw)

    @pytest.mark.parametrize(
        ("section", "key", "raw", "expected"),
        [
            ("population", "n-samples", "1", 1),
            ("batch", "chunks-n-chunks", "1", 1),
            ("globals", "sampling-frequency", "1", 1),
            ("globals", "duration", "1", 1),
            ("globals", "seed", "0", 0),
            ("noise", "minimum-frequency", "0", 0.0),
        ],
    )
    def test_the_floor_itself_is_allowed(self, section, key, raw, expected):
        """The boundary is inclusive: ``seed 0`` and ``minimum-frequency 0`` are legal settings, and
        one chunk or one sample is the smallest sensible run rather than an error."""
        state = ConfigState()
        state.set(section, key, raw)
        assert state.get(section, key) == expected

    def test_a_key_without_constraints_is_left_alone(self):
        state = ConfigState()
        state.set("globals", "start-time", "-1")
        assert state.get("globals", "start-time") == -1

    def test_the_message_names_the_key_that_was_wrong(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="at least 1") as error:
            state.set("population", "n-samples", "-5")
        assert "n-samples" in str(error.value)


class TestWhatTheEditorIsToldWhenItIsWrong:
    def test_an_unknown_key_lists_the_ones_that_would_work(self):
        """The message is the only discovery mechanism in the editor, so the valid keys have to be
        in it rather than the word "None"."""
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown key") as error:
            state.set("noise", "psd-file", "x")
        message = str(error.value)
        assert "psd" in message
        assert "detectors" in message

    def test_an_unknown_section_names_the_section(self):
        state = ConfigState()
        with pytest.raises(ValueError, match="Unknown section: sgwb"):
            state.set("sgwb", "psd", "x")

    def test_a_fresh_state_has_no_file_behind_it(self):
        """``None`` rather than an empty string: the editor prints "unsaved" for one and treats the
        other as a path it can save to."""
        assert ConfigState().config_file is None

    def test_a_state_built_from_a_dict_shares_it(self):
        data: dict = {"globals": {"simulator-arguments": {"seed": 1}}}
        state = ConfigState(data)
        state.set("globals", "seed", "2")
        assert data["globals"]["simulator-arguments"]["seed"] == 2, "the state must edit the dict it was given"


class TestLoadingFromDisk:
    def test_a_mapping_is_loaded(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("globals:\n  simulator-arguments:\n    seed: 3\n")
        state = ConfigState()
        state.load(path)
        assert state.get("globals", "seed") == 3

    def test_null_values_are_dropped_on_load(self, tmp_path):
        """A config written with explicit nulls must not come back as keys the model rejects."""
        path = tmp_path / "config.yaml"
        path.write_text("globals:\n  seed: null\n  duration: 4\n")
        state = ConfigState()
        state.load(path)
        assert state.to_dict() == {"globals": {"duration": 4}}

    def test_a_yaml_list_is_refused(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- one\n- two\n")
        state = ConfigState()
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            state.load(path)

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("")
        state = ConfigState()
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            state.load(path)


class TestEmptiness:
    def test_a_fresh_state_is_empty(self):
        assert ConfigState().is_empty is True

    def test_a_state_holding_only_empty_containers_is_empty(self):
        assert ConfigState({"orchestration": {"noise": {}}}).is_empty is True

    def test_a_state_holding_a_setting_is_not_empty(self):
        state = ConfigState()
        state.set("globals", "seed", "1")
        assert state.is_empty is False

    def test_a_state_holding_only_a_falsy_setting_is_not_empty(self):
        state = ConfigState()
        state.set("globals", "start-time", "0")
        assert state.is_empty is False
