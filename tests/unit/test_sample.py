# ===========================================================================
# Tests for Sample class properties
# ===========================================================================

import csv
import os
import tempfile
from unittest.mock import patch

import numpy as np

from tests.test_fixtures import _make_sample
from usansred.model import IQData, XYData
from usansred.reduce import Experiment, Sample, Scan


class TestSampleProperties:
    """Tests for Sample properties."""

    def test_data_property_with_detector_data(self, mock_experiment):
        """data should return detector_data[0]."""
        sample = _make_sample(mock_experiment, "test", [])
        iq = IQData(q=[1.0, 2.0], i=[10.0, 20.0], e=[1.0, 2.0])
        sample.detector_data = [iq]
        assert sample.data is iq

    def test_data_property_empty(self, mock_experiment):
        """data should return None when detector_data is empty."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.data is None

    def test_size_property_with_data(self, mock_experiment):
        """size should return number of q-points."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.detector_data = [IQData(q=[1.0, 2.0, 3.0], i=[10.0, 20.0, 30.0], e=[1.0, 2.0, 3.0])]
        assert sample.size == 3

    def test_size_property_no_data(self, mock_experiment):
        """size should be 0 when no detector_data."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.size == 0

    def test_data_reduced_property(self, mock_experiment):
        """data_reduced should return data_bg_subtracted."""
        sample = _make_sample(mock_experiment, "test", [])
        bg = IQData(q=[1.0], i=[5.0], e=[0.5])
        sample.data_bg_subtracted = bg
        assert sample.data_reduced is bg

    def test_is_log_binned_false(self, mock_experiment):
        """is_log_binned should be False when data_log_binned.q is empty."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.is_log_binned is False

    def test_is_log_binned_true(self, mock_experiment):
        """is_log_binned should be True when data_log_binned.q has values."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_log_binned = IQData(q=[1.0], i=[10.0], e=[1.0])
        assert sample.is_log_binned is True

    def test_is_reduced_false(self, mock_experiment):
        """is_reduced should be False when data_bg_subtracted.q is empty."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.is_reduced is False

    def test_is_reduced_true(self, mock_experiment):
        """is_reduced should be True when data_bg_subtracted.q has values."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = IQData(q=[1.0], i=[10.0], e=[1.0])
        assert sample.is_reduced is True

    def test_size_reduced(self, mock_experiment):
        """size_reduced should return length of bg_subtracted q."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = IQData(q=[1.0, 2.0], i=[10.0, 20.0], e=[1.0, 2.0])
        assert sample.size_reduced == 2

    def test_size_reduced_empty(self, mock_experiment):
        """size_reduced should be 0 when no bg_subtracted data."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.size_reduced == 0

    def test_num_log_bins(self, mock_experiment):
        """num_log_bins should return length of data_log_binned.q."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_log_binned = IQData(q=[1.0, 2.0, 3.0], i=[10.0, 20.0, 30.0], e=[1.0, 2.0, 3.0])
        assert sample.num_log_bins == 3

    def test_num_of_banks_property(self, mock_experiment):
        """num_of_banks should delegate to experiment."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.num_of_banks == mock_experiment.num_of_banks


class TestSampleEquality:
    """Tests for Sample.__eq__."""

    def test_equal_samples(self, mock_experiment):
        """Two samples with same name and start_scan_num should be equal."""
        s1 = _make_sample(mock_experiment, "sample_a", [])
        s1.start_scan_num = 100
        s2 = _make_sample(mock_experiment, "sample_a", [])
        s2.start_scan_num = 100
        assert s1 == s2

    def test_different_name(self, mock_experiment):
        """Samples with different names should not be equal."""
        s1 = _make_sample(mock_experiment, "sample_a", [])
        s1.start_scan_num = 100
        s2 = _make_sample(mock_experiment, "sample_b", [])
        s2.start_scan_num = 100
        assert s1 != s2

    def test_different_start_scan_num(self, mock_experiment):
        """Samples with different start_scan_num should not be equal."""
        s1 = _make_sample(mock_experiment, "sample_a", [])
        s1.start_scan_num = 100
        s2 = _make_sample(mock_experiment, "sample_a", [])
        s2.start_scan_num = 200
        assert s1 != s2

    def test_not_equal_to_non_sample(self, mock_experiment):
        """Comparing with a non-Sample should return NotImplemented."""
        s1 = _make_sample(mock_experiment, "sample_a", [])
        result = s1.__eq__("not a sample")
        assert result is NotImplemented


class TestSampleDumpDataToCsv:
    """Tests for Sample.dump_data_to_csv."""

    def test_dump_iq_data(self, mock_experiment):
        """dump_data_to_csv should write IQData to CSV correctly."""
        sample = _make_sample(mock_experiment, "test", [])
        iq = IQData(q=[0.1, 0.2], i=[100.0, 200.0], e=[10.0, 14.0], t=[1.0, 2.0])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            sample.dump_data_to_csv(filepath, iq)

            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[0][0]) == 0.1
            assert float(rows[0][1]) == 100.0
        finally:
            os.unlink(filepath)

    def test_dump_xy_data(self, mock_experiment):
        """dump_data_to_csv should write XYData to CSV correctly."""
        sample = _make_sample(mock_experiment, "test", [])
        xy = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0, 200.0])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            sample.dump_data_to_csv(filepath, xy)

            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[0][0]) == 1.0
            assert float(rows[0][1]) == 10.0
        finally:
            os.unlink(filepath)

    def test_dump_with_title(self, mock_experiment):
        """dump_data_to_csv should prepend title row when provided."""
        sample = _make_sample(mock_experiment, "test", [])
        iq = IQData(q=[0.1], i=[100.0], e=[10.0], t=[])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            sample.dump_data_to_csv(filepath, iq, title="My Title")

            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["My Title"]
            assert len(rows) == 2  # title + 1 data row
        finally:
            os.unlink(filepath)

    def test_dump_uneven_lists(self, mock_experiment):
        """dump_data_to_csv should handle lists of different lengths with empty strings."""
        sample = _make_sample(mock_experiment, "test", [])
        # q has 2 values, t has 0 — the T column should fill with ""
        iq = IQData(q=[0.1, 0.2], i=[100.0, 200.0], e=[10.0, 14.0], t=[])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            sample.dump_data_to_csv(filepath, iq)

            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2
            # The T column (4th) should be empty string
            assert rows[0][3] == ""
        finally:
            os.unlink(filepath)


# ===========================================================================
# Tests for Sample._match_or_interpolate
# ===========================================================================


class TestMatchOrInterpolate:
    """Tests for Sample._match_or_interpolate."""

    def test_exact_match(self, mock_experiment):
        """When q values match exactly, take the values directly."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([1.0, 2.0, 3.0])
        q_bg = np.array([1.0, 2.0, 3.0])
        i_bg = np.array([10.0, 20.0, 30.0])
        e_bg = np.array([1.0, 2.0, 3.0])

        i_matched, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        np.testing.assert_allclose(i_matched, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(e_matched, [1.0, 2.0, 3.0])

    def test_interpolation(self, mock_experiment):
        """When q values are far apart, interpolation should be used."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([1.5])  # midpoint between 1.0 and 2.0
        q_bg = np.array([1.0, 2.0])
        i_bg = np.array([10.0, 20.0])
        e_bg = np.array([1.0, 2.0])

        i_matched, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        # Linear interpolation at midpoint: (10 + 20) / 2 = 15
        np.testing.assert_allclose(i_matched, [15.0])
        np.testing.assert_allclose(e_matched, [1.5])

    def test_close_match_within_tolerance(self, mock_experiment):
        """Values within tolerance should be matched directly, not interpolated."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([1.0])
        q_bg = np.array([1.000005])  # within default tolerance of 1e-5 * 1.0 = 1e-5
        i_bg = np.array([10.0])
        e_bg = np.array([1.0])

        i_matched, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        np.testing.assert_allclose(i_matched, [10.0])
        np.testing.assert_allclose(e_matched, [1.0])
