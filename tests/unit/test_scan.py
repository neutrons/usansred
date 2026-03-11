import csv
import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
from usansred.model import IQData, MonitorData, XYData
from usansred.reduce import Scan

# ===========================================================================
# Tests for Scan class
# ===========================================================================


class TestScanLoadData:
    """Tests for the Scan.load_data field."""

    def test_scan_load_data_false_skips_io(self, mock_experiment):
        """Creating a Scan with load_data=False should not attempt to read files."""
        scan = Scan(number=99999, experiment=mock_experiment, load_data=False)
        assert scan.number == 99999
        # Monitor data should be default (empty)
        assert scan.monitor_data.xy_data.x == []

    def test_scan_load_data_true_calls_load(self, mock_experiment):
        """Creating a Scan with load_data=True should call load()."""
        with patch.object(Scan, "load") as mock_load:
            scan = Scan(number=1, experiment=mock_experiment, load_data=True)
            mock_load.assert_called_once()
            assert scan.number == 1


class TestScanProperties:
    """Tests for Scan properties."""

    def test_size_property(self, mock_experiment):
        """size should return the number of IQ data points in monitor."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        scan.monitor_data = MonitorData(iq_data=IQData(q=[1.0, 2.0, 3.0], i=[10.0, 20.0, 30.0], e=[1.0, 2.0, 3.0]))
        assert scan.size == 3

    def test_size_property_empty(self, mock_experiment):
        """size should be 0 for a freshly created Scan."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        assert scan.size == 0

    def test_num_of_banks_property(self, mock_experiment):
        """num_of_banks should delegate to experiment."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        assert scan.num_of_banks == mock_experiment.num_of_banks

    def test_num_of_banks_property_multi(self, mock_experiment_2banks):
        """num_of_banks should reflect the experiment's value."""
        scan = Scan(number=0, experiment=mock_experiment_2banks, load_data=False)
        assert scan.num_of_banks == 2


class TestScanConvertXYToIQ:
    """Tests for Scan.convert_xy_to_iq."""

    def test_basic_conversion(self, mock_experiment):
        """convert_xy_to_iq should copy x→q, y→i, t→t and calculate Poisson errors."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        xy = XYData(x=[1.0, 2.0], y=[10.0, 20.0], e=[1.0, 2.0], t=[100.0, 200.0])

        iq = scan.convert_xy_to_iq(xy)

        assert iq.q == [1.0, 2.0]
        assert iq.i == [10.0, 20.0]
        assert iq.t == [100.0, 200.0]
        # Error: sqrt(|y - 0.5| + 0.5)
        expected_e0 = math.sqrt(math.fabs(10.0 - 0.5) + 0.5)
        expected_e1 = math.sqrt(math.fabs(20.0 - 0.5) + 0.5)
        np.testing.assert_allclose(iq.e, [expected_e0, expected_e1])

    def test_conversion_empty_data(self, mock_experiment):
        """convert_xy_to_iq with empty data should produce empty IQData."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        xy = XYData()

        iq = scan.convert_xy_to_iq(xy)

        assert iq.q == []
        assert iq.i == []
        assert iq.e == []
        assert iq.t == []

    def test_conversion_near_half(self, mock_experiment):
        """Test error calculation for y values near 0.5."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        xy = XYData(x=[1.0], y=[0.5], e=[0.1], t=[])

        iq = scan.convert_xy_to_iq(xy)

        # sqrt(|0.5 - 0.5| + 0.5) = sqrt(0.5)
        np.testing.assert_allclose(iq.e, [math.sqrt(0.5)])

    def test_conversion_negative_y(self, mock_experiment):
        """Test error calculation with negative y values."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        xy = XYData(x=[1.0], y=[-5.0], e=[1.0], t=[])

        iq = scan.convert_xy_to_iq(xy)

        # sqrt(|-5.0 - 0.5| + 0.5) = sqrt(5.5 + 0.5) = sqrt(6.0)
        np.testing.assert_allclose(iq.e, [math.sqrt(6.0)])


class TestScanReadXYFile:
    """Tests for Scan.read_xy_file."""

    def test_read_3_column_csv(self, mock_experiment):
        """read_xy_file should parse a 3-column CSV correctly."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1.0, 10.0, 0.5])
            writer.writerow([2.0, 20.0, 1.0])
            writer.writerow([3.0, 30.0, 1.5])
            f.flush()
            filepath = f.name

        try:
            xy = scan.read_xy_file(filepath)
            assert xy.x == [1.0, 2.0, 3.0]
            assert xy.y == [10.0, 20.0, 30.0]
            assert xy.e == [0.5, 1.0, 1.5]
            assert xy.t == []
        finally:
            os.unlink(filepath)

    def test_read_4_column_csv(self, mock_experiment):
        """read_xy_file should parse a 4-column CSV and include t values."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1.0, 10.0, 0.5, 100.0])
            writer.writerow([2.0, 20.0, 1.0, 200.0])
            f.flush()
            filepath = f.name

        try:
            xy = scan.read_xy_file(filepath)
            assert xy.x == [1.0, 2.0]
            assert xy.y == [10.0, 20.0]
            assert xy.e == [0.5, 1.0]
            assert xy.t == [100.0, 200.0]
        finally:
            os.unlink(filepath)

    def test_read_with_comment_lines(self, mock_experiment):
        """read_xy_file should skip lines starting with #."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, newline="") as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            writer = csv.writer(f)
            writer.writerow([1.0, 10.0, 0.5])
            writer.writerow([2.0, 20.0, 1.0])
            f.flush()
            filepath = f.name

        try:
            xy = scan.read_xy_file(filepath)
            assert len(xy.x) == 2
            assert xy.x == [1.0, 2.0]
        finally:
            os.unlink(filepath)

    def test_read_with_short_rows(self, mock_experiment):
        """read_xy_file should skip rows with fewer than 3 columns."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1.0, 10.0])  # Too few columns — should be skipped
            writer.writerow([2.0, 20.0, 1.0])
            f.flush()
            filepath = f.name

        try:
            xy = scan.read_xy_file(filepath)
            assert len(xy.x) == 1
            assert xy.x == [2.0]
        finally:
            os.unlink(filepath)

class TestScanLoadDataBranch:
    """Tests for the Scan.load_data conditional in model_post_init."""

    def test_load_data_false_does_not_call_load(self, mock_experiment):
        """Creating a Scan with load_data=False should not call load()."""
        with patch.object(Scan, "load") as mock_load:
            scan = Scan(number=99999, experiment=mock_experiment, load_data=False)
            mock_load.assert_not_called()
        assert scan.number == 99999

    def test_load_data_true_calls_load(self, mock_experiment):
        """Creating a Scan with load_data=True should call load()."""
        with patch.object(Scan, "load") as mock_load:
            scan = Scan(number=1, experiment=mock_experiment, load_data=True)
            mock_load.assert_called_once()
        assert scan.number == 1

    def test_load_data_default_is_true(self, mock_experiment):
        """The default value of load_data should be True (so load() is called)."""
        with patch.object(Scan, "load") as mock_load:
            Scan(number=1, experiment=mock_experiment)
            mock_load.assert_called_once()

class TestScanLoadDataField:
    """Cover the Scan.load_data field and model_post_init branching."""

    def test_load_data_field_default_value(self):
        """Scan.load_data should default to True."""
        # Check field default without actually creating (which would trigger load)
        field_info = Scan.model_fields["load_data"]
        assert field_info.default is True

    def test_scan_with_load_data_false_has_empty_data(self, mock_experiment):
        """A Scan created with load_data=False should have empty monitor/detector data."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        assert scan.monitor_data.xy_data.x == []
        assert scan.monitor_data.iq_data.q == []
        assert scan.detector_data == []

    def test_scan_number_preserved(self, mock_experiment):
        """Scan number should be stored correctly."""
        scan = Scan(number=42, experiment=mock_experiment, load_data=False)
        assert scan.number == 42

    def test_scan_experiment_reference(self, mock_experiment):
        """Scan should hold a reference to its experiment."""
        scan = Scan(number=0, experiment=mock_experiment, load_data=False)
        assert scan.experiment is mock_experiment

