"""Unit tests for the CombinedSample, Scan, Sample, and model classes in usansred."""

import logging
import math
from unittest.mock import patch

import numpy as np
import pytest
from usansred.model import IQData, MonitorData, XYData
from usansred.reduce import CombinedSample, Experiment, Sample, Scan

from tests.test_fixtures import _make_sample, _make_scan, _make_scan_multi_bank

# ===========================================================================
# Tests for _combine_xy_data_pair (static helper)
# ===========================================================================


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

    def test_custom_tolerance(self):
        """Using a custom tolerance should bin X values accordingly."""
        # With tolerance=0.1, x=1.0 and x=1.05 should fall in the same bin (both round to 10)
        xy1 = XYData(x=[1.0], y=[10.0], e=[1.0], t=[100.0])
        xy2 = XYData(x=[1.05], y=[20.0], e=[2.0], t=[200.0])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2, tolerance=0.1)

        # Both should be binned together since int(round(1.0/0.1)) == int(round(1.05/0.1)) == 10
        assert len(result.x) == 1
        np.testing.assert_allclose(result.y, [30.0])
        np.testing.assert_allclose(result.e, [math.sqrt(1.0 + 4.0)])
        np.testing.assert_allclose(result.t, [150.0])  # average of 100 and 200

    def test_mixed_t_values(self):
        """One input has t values, the other has empty t."""
        xy1 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0, 200.0])
        xy2 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        assert len(result.x) == 2
        # t for xy2 defaults to [0.0, 0.0], so average of [100, 0] = 50 and [200, 0] = 100
        np.testing.assert_allclose(result.t, [50.0, 100.0])

    def test_partially_overlapping_x_values(self):
        """Some X values overlap, some don't."""
        xy1 = XYData(x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], e=[1.0, 2.0, 3.0], t=[])
        xy2 = XYData(x=[2.0, 3.0, 4.0], y=[25.0, 35.0, 45.0], e=[2.5, 3.5, 4.5], t=[])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        # x=1.0 only from xy1, x=2.0 and x=3.0 overlap, x=4.0 only from xy2
        assert len(result.x) == 4
        # Sorted output
        sorted_idx = np.argsort(result.x)
        x_sorted = np.array(result.x)[sorted_idx]
        y_sorted = np.array(result.y)[sorted_idx]

        np.testing.assert_allclose(x_sorted, [1.0, 2.0, 3.0, 4.0], atol=1e-7)
        np.testing.assert_allclose(y_sorted, [10.0, 45.0, 65.0, 45.0])

    def test_t_length_mismatch_falls_back_to_zeros(self):
        """When t has values but not same length as x, it should fall back to zeros."""
        xy1 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0])  # len(t) != len(x)
        xy2 = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[50.0, 60.0])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        assert len(result.x) == 2
        # xy1 t falls back to [0.0, 0.0], xy2 t is [50.0, 60.0]
        # average: [25.0, 30.0]
        np.testing.assert_allclose(result.t, [25.0, 30.0])

    def test_output_sorted_by_x_key(self):
        """Output should be sorted by discretized x key, regardless of input order."""
        xy1 = XYData(x=[3.0, 1.0], y=[30.0, 10.0], e=[3.0, 1.0], t=[])
        xy2 = XYData(x=[2.0, 4.0], y=[20.0, 40.0], e=[2.0, 4.0], t=[])

        result = CombinedSample._combine_xy_data_pair(xy1, xy2)

        assert len(result.x) == 4
        # Should be sorted by x
        np.testing.assert_allclose(result.x, sorted(result.x))


# ===========================================================================
# Tests for CombinedSample.combine
# ===========================================================================


class TestCombinedSampleCombine:
    """Tests for the CombinedSample.combine method."""

    def test_combine_no_samples_raises(self, mock_experiment):
        """combine() should raise if combined_samples is empty."""
        cs = CombinedSample(name="empty", experiment=mock_experiment)
        with pytest.raises(AssertionError, match="No samples to combine"):
            cs.combine()

    def test_combine_no_scans_raises(self, mock_experiment):
        """combine() should raise if all samples have empty scan lists."""
        sample = _make_sample(mock_experiment, "empty_scans", [])
        cs = CombinedSample(name="no_scans", experiment=mock_experiment, combined_samples=[sample])
        with pytest.raises(AssertionError, match="No scans in any sample to combine"):
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

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2])
        cs.combine()

        assert len(cs.combined_scans) == 1
        # Monitor Y should be summed
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [210.0, 410.0])
        # Detector Y should be summed
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.y, [105.0, 165.0])
        # Errors should be propagated in quadrature
        expected_mon_e = [math.sqrt(10.0**2 + 11.0**2), math.sqrt(14.0**2 + 15.0**2)]
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.e, expected_mon_e)
        # Detector errors should be propagated in quadrature
        expected_det_e = [math.sqrt(7.0**2 + 8.0**2), math.sqrt(9.0**2 + 10.0**2)]
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.e, expected_det_e)

    def test_combine_two_samples_iq_generated(self, mock_experiment):
        """After combining two samples, IQ data should be generated for both monitor and detector."""
        xy = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[])
        scan1 = _make_scan(mock_experiment, xy, xy)
        scan2 = _make_scan(mock_experiment, xy, xy)
        sample1 = _make_sample(mock_experiment, "s1", [scan1])
        sample2 = _make_sample(mock_experiment, "s2", [scan2])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2])
        cs.combine()

        # IQ data should be generated for monitor
        assert len(cs.combined_scans[0].monitor_data.iq_data.q) == 2
        assert len(cs.combined_scans[0].monitor_data.iq_data.i) == 2
        assert len(cs.combined_scans[0].monitor_data.iq_data.e) == 2
        # IQ data should be generated for detector bank 0
        assert len(cs.combined_scans[0].detector_data[0].iq_data.q) == 2
        assert len(cs.combined_scans[0].detector_data[0].iq_data.i) == 2
        assert len(cs.combined_scans[0].detector_data[0].iq_data.e) == 2

    def test_combine_three_samples(self, mock_experiment):
        """Combining three samples should sum all Y values correctly."""
        xy = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan1 = _make_scan(mock_experiment, xy, xy)
        scan2 = _make_scan(mock_experiment, xy, xy)
        scan3 = _make_scan(mock_experiment, xy, xy)
        sample1 = _make_sample(mock_experiment, "s1", [scan1])
        sample2 = _make_sample(mock_experiment, "s2", [scan2])
        sample3 = _make_sample(mock_experiment, "s3", [scan3])

        cs = CombinedSample(
            name="combined_3",
            experiment=mock_experiment,
            combined_samples=[sample1, sample2, sample3],
        )
        cs.combine()

        assert len(cs.combined_scans) == 1
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [30.0])
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.e, [math.sqrt(3.0)])

    def test_combine_mismatched_scan_counts(self, mock_experiment):
        """Samples with different scan counts should still combine, skipping missing scans."""
        xy1 = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan1a = _make_scan(mock_experiment, xy1, xy1)
        scan1b = _make_scan(mock_experiment, xy1, xy1)
        sample1 = _make_sample(mock_experiment, "s1", [scan1a, scan1b])

        scan2a = _make_scan(mock_experiment, xy1, xy1)
        sample2 = _make_sample(mock_experiment, "s2", [scan2a])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2])
        cs.combine()

        # Should have 2 scans (max of the two samples)
        assert len(cs.combined_scans) == 2
        # First scan: both samples contributed → Y doubled
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [20.0])
        # Second scan: only sample1 contributed → Y unchanged
        np.testing.assert_allclose(cs.combined_scans[1].monitor_data.xy_data.y, [10.0])

    def test_combine_mismatched_scan_counts_logs_warning(self, mock_experiment, caplog):
        """Mismatched scan counts should produce a warning log."""
        xy1 = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan1a = _make_scan(mock_experiment, xy1, xy1)
        scan1b = _make_scan(mock_experiment, xy1, xy1)
        sample1 = _make_sample(mock_experiment, "s1", [scan1a, scan1b])

        scan2a = _make_scan(mock_experiment, xy1, xy1)
        sample2 = _make_sample(mock_experiment, "s2", [scan2a])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2])

        with caplog.at_level(logging.WARNING):
            cs.combine()

        assert any("contains fewer scans" in msg for msg in caplog.messages)

    def test_combine_called_twice_resets(self, mock_experiment):
        """Calling combine() twice should reset combined_scans properly."""
        xy = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan = _make_scan(mock_experiment, xy, xy)
        sample = _make_sample(mock_experiment, "s1", [scan])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample])
        cs.combine()
        assert len(cs.combined_scans) == 1
        first_y = cs.combined_scans[0].monitor_data.xy_data.y[0]

        # Call combine again
        cs.combine()
        assert len(cs.combined_scans) == 1  # Should still be 1, not 2
        # Values should be the same (not doubled from re-accumulation)
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [first_y])

    def test_combine_multiple_banks(self, mock_experiment_2banks):
        """Combining with multiple detector banks should process all banks."""
        xy_mon = XYData(x=[1.0], y=[100.0], e=[10.0], t=[])
        xy_det1 = XYData(x=[1.0], y=[50.0], e=[5.0], t=[])
        xy_det2 = XYData(x=[1.0], y=[60.0], e=[6.0], t=[])

        scan1 = _make_scan_multi_bank(mock_experiment_2banks, xy_mon, [xy_det1, xy_det2])
        sample1 = _make_sample(mock_experiment_2banks, "s1", [scan1])

        xy_det1b = XYData(x=[1.0], y=[55.0], e=[5.5], t=[])
        xy_det2b = XYData(x=[1.0], y=[65.0], e=[6.5], t=[])
        scan2 = _make_scan_multi_bank(mock_experiment_2banks, xy_mon, [xy_det1b, xy_det2b])
        sample2 = _make_sample(mock_experiment_2banks, "s2", [scan2])

        cs = CombinedSample(
            name="combined_2bank",
            experiment=mock_experiment_2banks,
            combined_samples=[sample1, sample2],
        )
        cs.combine()

        assert len(cs.combined_scans) == 1
        assert len(cs.combined_scans[0].detector_data) == 2
        # Bank 1: 50 + 55 = 105
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.y, [105.0])
        # Bank 2: 60 + 65 = 125
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[1].xy_data.y, [125.0])
        # IQ data should have been generated for both banks
        assert len(cs.combined_scans[0].detector_data[0].iq_data.q) == 1
        assert len(cs.combined_scans[0].detector_data[1].iq_data.q) == 1

    def test_combine_multiple_scans_multiple_samples(self, mock_experiment):
        """Combining two samples each with 2 scans should produce 2 combined scans."""
        xy_a = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        xy_b = XYData(x=[1.0], y=[20.0], e=[2.0], t=[])
        xy_c = XYData(x=[1.0], y=[30.0], e=[3.0], t=[])
        xy_d = XYData(x=[1.0], y=[40.0], e=[4.0], t=[])

        scan1a = _make_scan(mock_experiment, xy_a, xy_a)
        scan1b = _make_scan(mock_experiment, xy_b, xy_b)
        sample1 = _make_sample(mock_experiment, "s1", [scan1a, scan1b])

        scan2a = _make_scan(mock_experiment, xy_c, xy_c)
        scan2b = _make_scan(mock_experiment, xy_d, xy_d)
        sample2 = _make_sample(mock_experiment, "s2", [scan2a, scan2b])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample1, sample2])
        cs.combine()

        assert len(cs.combined_scans) == 2
        # Scan 0: 10 + 30 = 40
        np.testing.assert_allclose(cs.combined_scans[0].monitor_data.xy_data.y, [40.0])
        # Scan 1: 20 + 40 = 60
        np.testing.assert_allclose(cs.combined_scans[1].monitor_data.xy_data.y, [60.0])
        # IQ data should be generated for both scans
        assert len(cs.combined_scans[0].monitor_data.iq_data.q) == 1
        assert len(cs.combined_scans[1].monitor_data.iq_data.q) == 1

    def test_combine_logs_info_message(self, mock_experiment, caplog):
        """combine() should log an info message upon completion."""
        xy = XYData(x=[1.0], y=[10.0], e=[1.0], t=[])
        scan = _make_scan(mock_experiment, xy, xy)
        sample = _make_sample(mock_experiment, "s1", [scan])

        cs = CombinedSample(name="my_combined", experiment=mock_experiment, combined_samples=[sample])

        with caplog.at_level(logging.INFO):
            cs.combine()

        assert any("Combined 1 samples into 'my_combined'" in msg for msg in caplog.messages)

    def test_combine_does_not_mutate_source_scans(self, mock_experiment):
        """combine() should deep-copy source data so originals are unmodified."""
        xy_mon = XYData(x=[1.0], y=[100.0], e=[10.0], t=[])
        xy_det = XYData(x=[1.0], y=[50.0], e=[5.0], t=[])
        scan = _make_scan(mock_experiment, xy_mon, xy_det)
        sample = _make_sample(mock_experiment, "s1", [scan])

        original_mon_y = scan.monitor_data.xy_data.y[0]
        original_det_y = scan.detector_data[0].xy_data.y[0]

        # Combine with a second identical sample
        xy_mon2 = XYData(x=[1.0], y=[100.0], e=[10.0], t=[])
        xy_det2 = XYData(x=[1.0], y=[50.0], e=[5.0], t=[])
        scan2 = _make_scan(mock_experiment, xy_mon2, xy_det2)
        sample2 = _make_sample(mock_experiment, "s2", [scan2])

        cs = CombinedSample(name="combined", experiment=mock_experiment, combined_samples=[sample, sample2])
        cs.combine()

        # Source scans should be unchanged
        assert scan.monitor_data.xy_data.y[0] == original_mon_y
        assert scan.detector_data[0].xy_data.y[0] == original_det_y

    def test_combine_multiple_banks_mismatched_scans(self, mock_experiment_2banks):
        """Multi-bank combine with mismatched scan counts should handle all banks correctly."""
        xy_mon = XYData(x=[1.0], y=[100.0], e=[10.0], t=[])
        xy_det1 = XYData(x=[1.0], y=[50.0], e=[5.0], t=[])
        xy_det2 = XYData(x=[1.0], y=[60.0], e=[6.0], t=[])

        scan1a = _make_scan_multi_bank(mock_experiment_2banks, xy_mon, [xy_det1, xy_det2])
        scan1b = _make_scan_multi_bank(mock_experiment_2banks, xy_mon, [xy_det1, xy_det2])
        sample1 = _make_sample(mock_experiment_2banks, "s1", [scan1a, scan1b])

        scan2a = _make_scan_multi_bank(mock_experiment_2banks, xy_mon, [xy_det1, xy_det2])
        sample2 = _make_sample(mock_experiment_2banks, "s2", [scan2a])

        cs = CombinedSample(
            name="combined_2bank_mismatch",
            experiment=mock_experiment_2banks,
            combined_samples=[sample1, sample2],
        )
        cs.combine()

        assert len(cs.combined_scans) == 2
        # Both banks of scan 0 should be summed
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[0].xy_data.y, [100.0])
        np.testing.assert_allclose(cs.combined_scans[0].detector_data[1].xy_data.y, [120.0])
        # Scan 1 only from sample1
        np.testing.assert_allclose(cs.combined_scans[1].detector_data[0].xy_data.y, [50.0])
        np.testing.assert_allclose(cs.combined_scans[1].detector_data[1].xy_data.y, [60.0])
        # IQ data generated for all
        for scan_idx in range(2):
            for bank_idx in range(2):
                assert len(cs.combined_scans[scan_idx].detector_data[bank_idx].iq_data.q) == 1
