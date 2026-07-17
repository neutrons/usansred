import json
import logging
from unittest.mock import patch

import pytest

import usansred.reduce
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
        assert experiment.config.binning.steps_per_decade == expected_steps_per_decade

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
        assert experiment.config.binning.steps_per_decade == 44

    def test_json_log_binning_governs_when_cli_not_set(self, tmp_path):
        """JSON binning.log_binning: true takes effect when CLI --logbin is not passed."""
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps({**self.MINIMAL_CONFIG, "binning": {"log_binning": True}}), encoding="utf-8")

        with patch.object(Sample, "model_post_init", return_value=None):
            experiment = Experiment(config_file=str(config_file), log_binning=False)

        assert experiment.log_binning is True
        assert experiment.config.binning.steps_per_decade == 33


class TestReduceOrderingAndDump:
    """Tests for the empty-cell/background reduction order and output-file dumping in Experiment.reduce."""

    SAMPLES = [
        {"name": "s1", "start_scan_num": 1, "num_of_scans": 1, "thickness": 0.1},
        {"name": "s2", "start_scan_num": 2, "num_of_scans": 1, "thickness": 0.1},
    ]
    BACKGROUND = {"name": "bg", "start_scan_num": 3, "num_of_scans": 1, "thickness": 0.1}
    EMPTY_CELL = {"name": "ec", "start_scan_num": 4, "num_of_scans": 1}

    @staticmethod
    def _make_experiment(tmp_path, config: dict) -> Experiment:
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")
        with patch.object(Sample, "model_post_init", return_value=None):
            experiment = Experiment(config_file=str(config_file), output_dir=str(tmp_path))
        return experiment

    @staticmethod
    def _run_reduce(experiment: Experiment) -> list[Sample]:
        """Run Experiment.reduce with Sample.reduce mocked; return the measurements reduced, in order."""
        reduced = []
        with (
            patch.object(Sample, "reduce", autospec=True, side_effect=lambda self: reduced.append(self)),
            patch.object(Experiment, "dump_reduced_data", autospec=True, return_value=None),
        ):
            experiment.reduce()
        return reduced

    def test_empty_cell_reduced_first_in_absence_of_background(self, tmp_path):
        experiment = self._make_experiment(tmp_path, {"samples": self.SAMPLES, "empty_cell": self.EMPTY_CELL})

        reduced = self._run_reduce(experiment)

        assert reduced == [experiment.empty_cell, *experiment.samples]

    def test_empty_cell_not_reduced_when_background_present(self, tmp_path, caplog, monkeypatch):
        experiment = self._make_experiment(
            tmp_path, {"samples": self.SAMPLES, "background": self.BACKGROUND, "empty_cell": self.EMPTY_CELL}
        )
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)

        with caplog.at_level(logging.INFO):
            reduced = self._run_reduce(experiment)

        assert reduced == [experiment.background, *experiment.samples]
        assert experiment.empty_cell not in reduced
        assert any("Skipping reduction of empty cell ec" in message for message in caplog.messages)

    def test_background_only(self, tmp_path):
        experiment = self._make_experiment(tmp_path, {"samples": self.SAMPLES, "background": self.BACKGROUND})

        reduced = self._run_reduce(experiment)

        assert reduced == [experiment.background, *experiment.samples]

    def test_samples_only(self, tmp_path):
        experiment = self._make_experiment(tmp_path, {"samples": self.SAMPLES})

        reduced = self._run_reduce(experiment)

        assert reduced == experiment.samples

    def test_dump_writes_samples_and_background_but_not_empty_cell(self, tmp_path):
        experiment = self._make_experiment(
            tmp_path, {"samples": self.SAMPLES, "background": self.BACKGROUND, "empty_cell": self.EMPTY_CELL}
        )
        dumped = []
        with patch.object(Sample, "dump_reduced_data_to_csv", autospec=True, side_effect=dumped.append):
            experiment.dump_reduced_data()

        assert dumped == [*experiment.samples, experiment.background]
        assert experiment.empty_cell not in dumped
