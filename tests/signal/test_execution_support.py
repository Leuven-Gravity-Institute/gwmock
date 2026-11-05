"""Refusing settings the chosen execution path would ignore.

Three bugs in the batched path shared one shape: a setting was read from the configuration and never
forwarded to the generator, so the run produced plausible output while quietly disregarding what it
had been told.

The fix is not a fourth special case but an inverted default: the batched path declares what it
honours, and anything else the user set is refused. A configuration key added later therefore fails
loudly there until someone wires it.
"""

from __future__ import annotations

import logging

import pytest

from gwmock.cli.utils.config import SignalConfig
from gwmock.signal.execution_support import require_execution_supports_configuration

_BASE = {"detectors": ["H1"], "source-type": "bbh", "waveform-model": "IMRPhenomD"}


def _config(**extra) -> SignalConfig:
    return SignalConfig.model_validate({**_BASE, **extra})


class TestBatchedRefusesWhatItIgnores:
    """The strict rule, applied where the path is new and has no users to break."""

    def test_a_plain_configuration_is_accepted(self):
        """The check must not refuse configurations the path can honour."""
        require_execution_supports_configuration(_config(), "batched")

    def test_every_honoured_setting_is_accepted_together(self):
        """Guards the opposite failure: a rule so strict nothing passes."""
        require_execution_supports_configuration(
            _config(
                **{
                    "waveform-backend": "ripple",
                    "waveform-backend-arguments": {"taper_fraction": 0.05},
                    "waveform-arguments": {"f_ref": 20.0},
                    "minimum-frequency": 30,
                    "effects": {"earth-rotation": False, "finite-arm-length": False},
                    "execution": "batched",
                }
            ),
            "batched",
        )

    @pytest.mark.parametrize(
        ("setting", "value"),
        [
            # No equivalent parameter on the batched entry point.
            ("waveform-options", {"ModeArray": [[2, 2]]}),
            # These configure the simulator's constructor, and the batched path bypasses the
            # simulator entirely.
            ("arguments", {"segment_duration": 8}),
            # Read only by the stochastic path.
            ("parameters", {"omega_ref": 1e-9}),
        ],
    )
    def test_a_setting_the_path_cannot_apply_is_refused(self, setting: str, value):
        with pytest.raises(ValueError, match="would ignore settings") as raised:
            require_execution_supports_configuration(_config(**{setting: value}), "batched")

        assert setting in str(raised.value), "the message must name the setting at fault"

    def test_the_message_says_why_each_setting_cannot_be_applied(self):
        """A list of rejected names does not tell the reader what to do about them."""
        with pytest.raises(ValueError, match="would ignore settings") as raised:
            require_execution_supports_configuration(_config(**{"arguments": {"a": 1}}), "batched")

        assert "bypasses" in str(raised.value) or "directly rather than" in str(raised.value)

    def test_an_unrecognised_key_is_refused(self):
        """An unknown setting -- a typo, or one from a newer version -- is absent rather than wrong.

        The config model allows extra keys, so nothing else reports them: the run simply behaves as
        though the line were not there. The key here is deliberately not a plausible misspelling,
        because the repository's spell checker rejects those in source.
        """
        with pytest.raises(ValueError, match="not-a-real-setting"):
            require_execution_supports_configuration(_config(**{"not-a-real-setting": "IMRPhenomD"}), "batched")

    def test_an_unrecognised_key_inside_a_nested_block_is_refused(self):
        """Checking only the top level would leave the rule with an escape hatch.

        ``output`` is a model that allows extras too, so ``output.<typo>`` validates and is never
        read -- and because ``output`` as a whole is honoured, a top-level-only check would wave it
        through. That is the same silent drop, one level down.
        """
        configuration = _config(**{"output": {"file_name": "x.gwf", "not-a-real-setting": 1}, "execution": "batched"})

        with pytest.raises(ValueError, match=r"output\.not-a-real-setting"):
            require_execution_supports_configuration(configuration, "batched")

    def test_known_nested_settings_are_still_accepted(self):
        """The nested check must reject unknown keys, not every nested block."""
        require_execution_supports_configuration(
            _config(**{"output": {"file_name": "x.gwf", "arguments": {"channel": "H1:STRAIN"}}}),
            "batched",
        )

    def test_defaults_are_not_treated_as_configured(self):
        """Only what the user wrote counts; otherwise every configuration would be refused.

        ``model_fields_set`` is what makes the inverted default workable -- without it the check
        cannot distinguish a field left alone from one deliberately set.
        """
        configuration = _config()

        assert "minimum_frequency" not in configuration.model_fields_set
        require_execution_supports_configuration(configuration, "batched")


class TestPerEventWarnsRatherThanRefuses:
    """The default path has users, and one of its settings has been ignored since before this."""

    def test_an_unrecognised_key_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(_config(**{"not-a-real-setting": "x"}), "per-event")

        assert "not a recognised setting" in caplog.text

    def test_parameters_warns_for_cbc(self, caplog):
        """Pre-existing: only the stochastic path reads it, so CBC runs discard it silently."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(_config(**{"parameters": {"a": 1}}), "per-event", "bbh")

        assert "stochastic-background path" in caplog.text

    def test_parameters_does_not_warn_for_sgwb(self, caplog):
        """The stochastic path genuinely reads them, so warning there would be simply wrong.

        ``_simulate_stationary_signal_segment`` merges ``signal_parameters`` into what it passes to
        the simulator, which is the one case where this setting has an effect.
        """
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(
                _config(**{"source-type": "sgwb", "parameters": {"omega_ref": 1e-9}}), "per-event", "sgwb"
            )

        assert caplog.text == ""

    def test_an_unrecognised_key_inside_a_nested_block_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(
                _config(**{"output": {"file_name": "x.gwf", "not-a-real-setting": 1}}), "per-event"
            )

        assert "output.not-a-real-setting" in caplog.text

    def test_a_plain_configuration_warns_about_nothing(self, caplog):
        """Otherwise every ordinary run would emit noise, and the warnings would stop being read."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(_config(), "per-event")

        assert caplog.text == ""

    def test_settings_the_batched_path_refuses_are_still_allowed(self, caplog):
        """Refusing these here would break working configurations for a problem they did not cause."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            require_execution_supports_configuration(
                _config(**{"waveform-options": {"ModeArray": [[2, 2]]}}), "per-event"
            )

        assert "waveform-options" not in caplog.text
