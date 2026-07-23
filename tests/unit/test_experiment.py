import json
import logging
from unittest.mock import patch

import pytest

import usansred.reduce
from usansred.reduce import Experiment, Sample


class TestLogBinningDeprecation:
    """Log binning has been removed. A deprecated ``binning``/``log_binning`` entry in an
    old JSON setup file must still load, but emit a deprecation warning and be ignored."""

    MINIMAL_CONFIG = {
        "samples": [{"name": "s", "start_scan_num": 1, "num_of_scans": 1, "thickness": 0.1}],
    }

    @pytest.mark.parametrize(
        "binning",
        [
            {"log_binning": True, "steps_per_decade": 44},
            {"log_binning": 1},
            {"log_binning": False},
            {},  # a bare binning block with no log_binning key
        ],
    )
    def test_deprecated_binning_block_warns_and_is_ignored(self, tmp_path, caplog, binning):
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps({**self.MINIMAL_CONFIG, "binning": binning}), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            with patch.object(Sample, "model_post_init", return_value=None):
                experiment = Experiment(config_file=str(config_file))

        # The deprecated entry is stripped: the validated config carries no binning attribute.
        assert not hasattr(experiment.config, "binning")
        # A deprecation warning was emitted for the user's benefit.
        assert any("deprecated" in message and "log binning" in message.lower() for message in caplog.messages)

    def test_config_without_binning_does_not_warn(self, tmp_path, caplog):
        config_file = tmp_path / "setup.json"
        config_file.write_text(json.dumps(self.MINIMAL_CONFIG), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            with patch.object(Sample, "model_post_init", return_value=None):
                Experiment(config_file=str(config_file))

        assert not any("log binning" in message.lower() for message in caplog.messages)


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

    def test_reduce_aborts_when_empty_cell_reduction_fails_without_background(self, tmp_path):
        experiment = self._make_experiment(tmp_path, {"samples": self.SAMPLES, "empty_cell": self.EMPTY_CELL})

        reduced = []

        def fake_reduce(sample):
            if sample is experiment.empty_cell:
                raise ValueError("boom")
            reduced.append(sample)

        with (
            patch.object(Sample, "reduce", autospec=True, side_effect=fake_reduce),
            patch.object(Experiment, "dump_reduced_data", autospec=True, return_value=None),
            pytest.raises(RuntimeError, match="empty cell"),
        ):
            experiment.reduce()

        assert reduced == []

    def test_reduce_aborts_when_background_reduction_fails(self, tmp_path):
        experiment = self._make_experiment(tmp_path, {"samples": self.SAMPLES, "background": self.BACKGROUND})

        reduced = []

        def fake_reduce(sample):
            if sample is experiment.background:
                raise ValueError("boom")
            reduced.append(sample)

        with (
            patch.object(Sample, "reduce", autospec=True, side_effect=fake_reduce),
            patch.object(Experiment, "dump_reduced_data", autospec=True, return_value=None),
            pytest.raises(RuntimeError, match="background"),
        ):
            experiment.reduce()

        assert reduced == []

    def test_dump_writes_samples_and_background_but_not_empty_cell(self, tmp_path):
        experiment = self._make_experiment(
            tmp_path, {"samples": self.SAMPLES, "background": self.BACKGROUND, "empty_cell": self.EMPTY_CELL}
        )
        dumped = []
        with patch.object(Sample, "dump_reduced_data_to_csv", autospec=True, side_effect=dumped.append):
            experiment.dump_reduced_data()

        assert dumped == [*experiment.samples, experiment.background]
        assert experiment.empty_cell not in dumped
