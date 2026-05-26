import json
from unittest.mock import patch

import pytest

from usansred.reduce import Experiment, Sample


class TestLogBinning:
    MINIMAL_CONFIG = {
        "samples": [{"name": "s", "start_scan_num": 1, "num_of_scans": 1, "thickness": 0.1}],
    }

    @pytest.mark.parametrize(
        ("config_extra", "expected_log_binning", "expected_steps_per_decade"),
        [
            ({"binning": {"log_binning": True, "steps_per_decade": 44}}, True, 44),
            ({"binning": {"log_binning": 1}}, True, 33),
            ({"binning": {"log_binning": False}}, False, 33),
            ({"binning": {"log_binning": 0}}, False, 33),
            ({"binning": {"log_binning": ""}}, False, 33),
            ({}, False, 33),  # absent → defaults to False
        ],
    )
    def test_log_binning_from_json_config(
        self, tmp_path, config_extra, expected_log_binning, expected_steps_per_decade
    ):
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps({**self.MINIMAL_CONFIG, **config_extra}), encoding="utf-8")

        with patch.object(Sample, "model_post_init", return_value=None):
            experiment = Experiment(config_file=str(config_file))

        assert experiment.log_binning is expected_log_binning
        assert experiment.config["binning"]["steps_per_decade"] == expected_steps_per_decade

    def test_cli_logbin_overrides_json_config(self, tmp_path):
        """CLI --logbin=True takes precedence over binning.log_binning: false in the JSON."""
        config_file = tmp_path / "setup.json"
        config_file.write_text(
            json.dumps({**self.MINIMAL_CONFIG, "binning": {"log_binning": False, "steps_per_decade": 44}}),
            encoding="utf-8",
        )

        with patch.object(Sample, "model_post_init", return_value=None):
            experiment = Experiment(config_file=str(config_file))
        experiment.amend_log_binning(True)

        assert experiment.log_binning is True
        assert experiment.config["binning"]["steps_per_decade"] == 44

    def test_json_log_binning_governs_when_cli_not_set(self, tmp_path):
        """JSON binning.log_binning: true takes effect when CLI --logbin is not passed."""
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps({**self.MINIMAL_CONFIG, "binning": {"log_binning": True}}), encoding="utf-8")

        with patch.object(Sample, "model_post_init", return_value=None):
            experiment = Experiment(config_file=str(config_file), log_binning=False)

        assert experiment.log_binning is True
        assert experiment.config["binning"]["steps_per_decade"] == 33
