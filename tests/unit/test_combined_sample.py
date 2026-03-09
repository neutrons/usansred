"""Unit tests for the CombinedSample class in usansred.reduce."""

import math
from unittest.mock import patch

import numpy as np
import pytest

from usansred.model import IQData, MonitorData, XYData
from usansred.reduce import CombinedSample, Experiment, Sample, Scan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    return exp


def _make_scan(experiment: Experiment, xy_monitor: XYData, xy_detector: XYData) -> Scan:
    """Create a Scan with pre-populated monitor and detector data (no file I/O)."""
    scan = Scan(number=0, experiment=experiment, load_data=False)
    scan.monitor_data = MonitorData(xy_data=xy_monitor, iq_data=IQData())
    scan.detector_data = [MonitorData(xy_data=xy_detector, iq_data=IQData())]
    return scan


def _make_sample(experiment: Experiment, name: str, scans: list[Scan]) -> Sample:
    """Create a Sample with model_post_init bypassed and scans injected."""
    with patch.object(Sample, "model_post_init", return_value=None):
        sample = Sample(name=name, experiment=experiment, start_scan_num=0, num_of_scans=0)
    sample.scans = scans
    return sample


# ---------------------------------------------------------------------------
# Tests for _combine_xy_data_pair (static helper)
# ---------------------------------------------------------------------------


class TestCombineXYDataPair:
    """Tests for CombinedSample._combine_xy_data_pair."""

    def test_basic_summation(self):
        """Two identical XYData objects should produce doubled Y and √2 × E."""
        xy1 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0, 200.0])
        xy2 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0, 200.0])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        assert len(result.x) == 2
        np.testing.assert_allclose(result.y, [20.0, 40.0])
        np.testing.assert_allclose(result.e, [math.sqrt(2.0), math.sqrt(8.0)])
        np.testing.assert_allclose(result.t, [100.0, 200.0])

    def test_non_overlapping_x_values(self):
        """Non-overlapping X values should all appear in the output."""
        xy1 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[])
        xy2 = XYData(x=[3.0, 4.0], y=[30.0, 40.0], e=[3.0, 4.0], t=[])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        assert len(result.x) == 4
        np.testing.assert_allclose(sorted(result.y), [10.0, 20.0, 30.0, 40.0])

    def test_empty_inputs(self):
        """Combining two empty XYData objects should produce empty output."""
        xy_empty = XYData()
        result = CombinedSample._combine_xy_data_pair(xy_empty, xy_empty)

        assert result.x == []
        assert result.y == []
        assert result.e == []
        assert result.t == []

    def test_one_empty_input(self):
        """Combining with an empty XYData should return the non-empty data unchanged."""
        xy1 = XYData(x=[1.0], y=[5.0], e=[0.5], t=[10.0])
        xy_empty = XYData()

        result = CombinedSample._combine_xy_data_pair(xy1, xy_empty)

        assert len(result.x) == 1
        np.testing.assert_allclose(result.y, [5.0])
        np.testing.assert_allclose(result.e, [0.5])


# ---------------------------------------------------------------------------
# Tests for CombinedSample.combine
# ---------------------------------------------------------------------------


class TestCombinedSampleCombine:
    """Tests for the CombinedSample.combine method."""

    def test_combine_no_samples_raises(self, mock_experiment):
        """combine() should raise if combined_samples is empty."""
        cs = CombinedSample(name="empty", experiment=mock_experiment)
        with pytest.raises(AssertionError, match="No samples to combine"):
            cs.combine()

    def test_combine_single_sample(self, mock_experiment):
        """Combining a single sample should produce identical scan data."""
        xy_mon = XYData(x=[1.0, 2.0], y=[100.0, 200.0], e=[10.0, 14.0], t=[])
        xy_det = XYData(x=[1.0, 2.0], y=[50.0, 80.0], e=[7.0, 9.0], t=[])
        scan = _make_scan(mock_experiment, xy_mon, xy_det)
        sample = _make_sample(mock_experiment, "s1", [scan])

        cs = CombinedSample(name="combined_s1", experiment=mock_experiment, combined_samples=[sample])
        cs.combine()

        assert len(cs.combined_scans) == 1
        # Monitor Y should be unchanged (single sample)
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [100.0, 200.0])
        # Detector Y should be unchanged
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.y, [50.0, 80.0])
        # IQ data should have been generated
        assert len(cs.combined_scans[0].monitor_data.iq_data.q) == 2

    def test_combine_two_samples(self, mock_experiment):
        """Combining two samples should sum the Y values and propagate errors."""
        xy_mon1 = XYData(x=[1.0, 2.0], y=[100.0, 200.0], e=[10.0, 14.0], t=[])
        xy_det1 = XYData(x=[1.0, 2.0], y=[50.0, 80.0], e=[7.0, 9.0], t=[])
        scan1 = _make_scan(mock_experiment, xy_mon1, xy_det1)
        sample1 = _make_sample(mock_experiment, "s1", [scan1])

        xy_mon2 = XYData(x=[1.0, 2.0], y=[110.0, 210.0], e=[11.0, 15.0], t=[])
        xy_det2 = XYData(x=[1.0, 2.0], y=[55.0, 85.0], e=[8.0, 10.0], t=[])
        scan2 = _make_scan(mock_experiment, xy_mon2, xy_det2)
        sample2 = _make_sample(mock_experiment, "s2", [scan2])

        cs = CombinedSample(
            name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2]
        )
        cs.combine()

        assert len(cs.combined_scans) == 1
        # Monitor Y should be summed
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [210.0, 410.0])
        # Detector Y should be summed
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.y, [105.0, 165.0])
        # Errors should be propagated in quadrature
        expected_mon_e = [math.sqrt(10.0**2 + 11.0**2), math.sqrt(14.0**2 + 15.0**2)]
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.e, expected_mon_e)

    def test_combine_mismatched_scan_counts(self, mock_experiment):
        """Samples with different scan counts should still combine, skipping missing scans."""
        xy1 = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan1a = _make_scan(mock_experiment, xy1, xy1)
        scan1b = _make_scan(mock_experiment, xy1, xy1)
        sample1 = _make_sample(mock_experiment, "s1", [scan1a, scan1b])

        scan2a = _make_scan(mock_experiment, xy1, xy1)
        sample2 = _make_sample(mock_experiment, "s2", [scan2a])

        cs = CombinedSample(
            name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2]
        )
        cs.combine()

        # Should have 2 scans (max of the two samples)
        assert len(cs.combined_scans) == 2
        # First scan: both samples contributed → Y doubled
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [20.0])
        # Second scan: only sample1 contributed → Y unchanged
        np.testing.assert_allclose(cs.combined_scans[1].monitor_data.xy_data.y, [10.0])


# ---------------------------------------------------------------------------
# Tests for Scan.load_data flag
# ---------------------------------------------------------------------------


class TestScanLoadData:
    """Tests for the Scan.load_data field."""

    def test_scan_load_data_false_skips_io(self, mock_experiment):
        """Creating a Scan with load_data=False should not attempt to read files."""
        # This should NOT raise a FileNotFoundError
        scan = Scan(number=99999, experiment=mock_experiment, load_data=False)
        assert scan.number == 99999
        # Monitor data should be default (empty)
        assert scan.monitor_data.xy_data.x == []