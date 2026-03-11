import math
import os
from unittest.mock import patch

import numpy as np
import pytest
from usansred.model import IQData, MonitorData, XYData
from usansred.reduce import CombinedSample, Experiment, Sample, Scan


@pytest.mark.datarepo()
def test_data_server(data_server):
    r"""find one file within the data-directory"""
    expected = os.path.join(data_server.directory, "testdataserver", "empty.txt")
    assert data_server.path_to("empty.txt") == expected


@pytest.fixture()
def mock_experiment():
    """Create a minimal Experiment with model_post_init bypassed."""
    with patch.object(Experiment, "model_post_init", return_value=None):
        exp = Experiment(config="dummy.json")
    exp.folder = ""
    exp.output_dir = ""
    exp.num_of_banks = 1
    exp.prim_wave = 3.6
    exp.darwin_width = 5.1
    exp.logbin = False
    exp.v_angle = 0.042
    exp.min_q = 1e-6
    exp.step_per_dec = 33
    exp.background = None
    exp.samples = []
    return exp


@pytest.fixture()
def mock_experiment_2banks():
    """Create a minimal Experiment with 2 detector banks."""
    with patch.object(Experiment, "model_post_init", return_value=None):
        exp = Experiment(config="dummy.json")
    exp.folder = ""
    exp.output_dir = ""
    exp.num_of_banks = 2
    exp.prim_wave = 3.6
    exp.darwin_width = 5.1
    exp.logbin = False
    exp.v_angle = 0.042
    exp.min_q = 1e-6
    exp.step_per_dec = 33
    exp.background = None
    exp.samples = []
    return exp


def _make_scan(experiment: Experiment, xy_monitor: XYData, xy_detector: XYData) -> Scan:
    """Create a Scan with pre-populated monitor and detector data (no file I/O)."""
    scan = Scan(number=0, experiment=experiment, load_data=False)
    scan.monitor_data = MonitorData(xy_data=xy_monitor, iq_data=IQData())
    scan.detector_data = [MonitorData(xy_data=xy_detector, iq_data=IQData())]
    return scan


def _make_scan_multi_bank(experiment: Experiment, xy_monitor: XYData, xy_detectors: list[XYData]) -> Scan:
    """Create a Scan with multiple detector banks."""
    scan = Scan(number=0, experiment=experiment, load_data=False)
    scan.monitor_data = MonitorData(xy_data=xy_monitor, iq_data=IQData())
    scan.detector_data = [MonitorData(xy_data=xy_det, iq_data=IQData()) for xy_det in xy_detectors]
    return scan


def _make_sample(experiment: Experiment, name: str, scans: list[Scan]) -> Sample:
    """Create a Sample with model_post_init bypassed and scans injected."""
    with patch.object(Sample, "model_post_init", return_value=None):
        sample = Sample(name=name, experiment=experiment, start_scan_num=0, num_of_scans=0)
    sample.scans = scans
    sample.detector_data = []
    sample.data_scaled = []
    sample.data_log_binned = IQData()
    sample.data_bg_subtracted = IQData()
    return sample


if __name__ == "__main__":
    pytest.main([__file__])
