# ===========================================================================
# Tests for Sample class properties
# ===========================================================================

import csv
import os
import tempfile
from unittest.mock import patch

import numpy as np

from tests.test_fixtures import _make_sample
from usansred.model import IQData, MonitorData, XYData
from usansred.reduce import ARCSEC_TO_RADIANS, Experiment, Sample, Scan, horizontal_rocking_width


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


class TestSampleNormalizeByMonitor:
    """Tests for Sample.normalize_by_monitor."""

    def test_normalizes_each_scan(self, mock_experiment):
        """Sample normalization should delegate to every scan."""
        scan_1 = Scan(number=1, experiment=mock_experiment, load_data=False)
        scan_2 = Scan(number=2, experiment=mock_experiment, load_data=False)
        sample = _make_sample(mock_experiment, "test", [scan_1, scan_2])
        normalized_scan_numbers = []

        def record_normalized_scan(scan):
            normalized_scan_numbers.append(scan.number)

        with patch.object(Scan, "normalize_by_monitor", autospec=True, side_effect=record_normalized_scan):
            sample.normalize_by_monitor()

        assert normalized_scan_numbers == [1, 2]


class TestSampleStitchScans:
    """Tests for Sample.stitch_scans."""

    @staticmethod
    def _make_scan_with_detector_iq(experiment: Experiment, q: list[float], i: list[float], e: list[float]) -> Scan:
        scan = Scan(number=123, experiment=experiment, load_data=False)
        scan.detector_data = [MonitorData(iq_data=IQData(q=q, i=i, e=e))]
        return scan

    def test_combines_duplicate_q_points_with_inverse_variance_weights(self, mock_experiment):
        """Duplicate Q points should weight lower-variance intensities more strongly."""
        scan_1 = self._make_scan_with_detector_iq(
            mock_experiment,
            q=[1.0, 2.0],
            i=[100.0, 20.0],
            e=[1.0, 2.0],
        )
        scan_2 = self._make_scan_with_detector_iq(
            mock_experiment,
            q=[1.0, 3.0],
            i=[200.0, 30.0],
            e=[3.0, 3.0],
        )
        sample = _make_sample(mock_experiment, "test", [scan_1, scan_2])

        sample.stitch_scans()

        expected_weight_sum = 1.0 / 1.0**2 + 1.0 / 3.0**2
        expected_intensity = (100.0 / 1.0**2 + 200.0 / 3.0**2) / expected_weight_sum
        expected_error = np.sqrt(1.0 / expected_weight_sum)
        np.testing.assert_allclose(sample.detector_data[0].q, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(sample.detector_data[0].i, [expected_intensity, 20.0, 30.0])
        np.testing.assert_allclose(sample.detector_data[0].e, [expected_error, 2.0, 3.0])


class TestCombineDuplicateQPoints:
    """Tests for Sample._combine_duplicate_q_points."""

    def test_sorts_q_and_averages_duplicate_points(self):
        """Duplicate Q points should be averaged and returned in ascending Q order."""
        q, i, e = Sample._combine_duplicate_q_points(
            q_scaled=[2.0, 1.0, 0.0, 1.0, 2.0],
            i_scaled=[20.0, 10.0, 5.0, 14.0, 30.0],
            e_scaled=[2.0, 1.0, 0.5, 3.0, 4.0],
        )

        np.testing.assert_allclose(q, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(i, [5.0, 12.0, 25.0])
        np.testing.assert_allclose(e, [0.5, np.sqrt(10.0) / 2.0, np.sqrt(20.0) / 2.0])


class TestSampleRescaleData:
    """Tests for Sample.rescale_data."""

    @staticmethod
    def _make_rescale_sample(experiment: Experiment, detector_data: list[IQData], thickness: float = 0.2) -> Sample:
        scan = Scan(number=123, experiment=experiment, load_data=False)
        sample = _make_sample(experiment, "test", [scan])
        sample.thickness = thickness
        sample.detector_data = detector_data
        return sample

    @staticmethod
    def _expected_rescaled_data(
        experiment: Experiment, harmonic: int, thickness: float, detector_data: IQData
    ) -> tuple[list[float], list[float], list[float]]:
        theta_to_q = ARCSEC_TO_RADIANS * (2 * np.pi / (experiment.prim_wave / harmonic))
        analyzer_solid_angle = experiment.v_angle * (horizontal_rocking_width(harmonic) * ARCSEC_TO_RADIANS)
        scaling_factor = 1.0 / (analyzer_solid_angle * thickness)

        q_scaled = [abs(theta) * theta_to_q for theta in detector_data.q]
        i_scaled = [i * scaling_factor for i in detector_data.i]
        e_scaled = [e * scaling_factor for e in detector_data.e]
        return Sample._combine_duplicate_q_points(q_scaled, i_scaled, e_scaled)

    def test_scales_single_bank_by_q_conversion_solid_angle_and_thickness(self, mock_experiment):
        detector_data = IQData(q=[0.0, 1.0, 3.0], i=[2.0, 4.0, 6.0], e=[0.2, 0.4, 0.6])
        sample = self._make_rescale_sample(mock_experiment, [detector_data], thickness=0.4)

        sample.rescale_data()

        expected_q, expected_i, expected_e = self._expected_rescaled_data(
            mock_experiment, harmonic=1, thickness=sample.thickness, detector_data=detector_data
        )
        assert len(sample.data_scaled) == 1
        np.testing.assert_allclose(sample.data_scaled[0].q, expected_q)
        np.testing.assert_allclose(sample.data_scaled[0].i, expected_i)
        np.testing.assert_allclose(sample.data_scaled[0].e, expected_e)

    def test_combines_positive_and_negative_angles_into_sorted_q(self, mock_experiment):
        detector_data = IQData(
            q=[-2.0, -1.0, 0.0, 1.0, 2.0],
            i=[20.0, 10.0, 5.0, 14.0, 30.0],
            e=[2.0, 1.0, 0.5, 3.0, 4.0],
        )
        sample = self._make_rescale_sample(mock_experiment, [detector_data], thickness=0.2)

        sample.rescale_data()

        expected_q, expected_i, expected_e = self._expected_rescaled_data(
            mock_experiment, harmonic=1, thickness=sample.thickness, detector_data=detector_data
        )
        np.testing.assert_allclose(sample.data_scaled[0].q, expected_q)
        np.testing.assert_allclose(sample.data_scaled[0].i, expected_i)
        np.testing.assert_allclose(sample.data_scaled[0].e, expected_e)
        assert sample.data_scaled[0].q == sorted(sample.data_scaled[0].q)

    def test_scales_each_harmonic_from_its_detector_bank(self, mock_experiment_2banks):
        bank_1_data = IQData(q=[0.0, 1.0, 2.0], i=[10.0, 20.0, 30.0], e=[1.0, 2.0, 3.0])
        bank_2_data = IQData(q=[0.0, 2.0, 4.0], i=[100.0, 200.0, 300.0], e=[10.0, 20.0, 30.0])
        sample = self._make_rescale_sample(mock_experiment_2banks, [bank_1_data, bank_2_data], thickness=0.3)

        sample.rescale_data()

        expected_bank_1 = self._expected_rescaled_data(
            mock_experiment_2banks, harmonic=1, thickness=sample.thickness, detector_data=bank_1_data
        )
        expected_bank_2 = self._expected_rescaled_data(
            mock_experiment_2banks, harmonic=2, thickness=sample.thickness, detector_data=bank_2_data
        )
        assert len(sample.data_scaled) == 2
        np.testing.assert_allclose(sample.data_scaled[0].q, expected_bank_1[0])
        np.testing.assert_allclose(sample.data_scaled[0].i, expected_bank_1[1])
        np.testing.assert_allclose(sample.data_scaled[0].e, expected_bank_1[2])
        np.testing.assert_allclose(sample.data_scaled[1].q, expected_bank_2[0])
        np.testing.assert_allclose(sample.data_scaled[1].i, expected_bank_2[1])
        np.testing.assert_allclose(sample.data_scaled[1].e, expected_bank_2[2])


class TestRockingCurveCentering:
    """Tests for Sample.rocking_curve_centering."""

    def test_centers_all_harmonics_using_symmetric_first_harmonic_range(self, mock_experiment_2banks):
        """Only the first harmonic's mostly symmetric range should define the center."""
        sample = _make_sample(mock_experiment_2banks, "test", [])
        q_first = np.linspace(-2.0, 4.0, 31)
        q_second = np.linspace(-4.0, 8.0, 31)
        expected_center = 0.35
        width = 0.6
        baseline = 0.2
        amplitude = 12.0
        intensity = baseline + amplitude * np.exp(-0.5 * ((q_first - expected_center) / width) ** 2)
        intensity[q_first > 2.0] = 50.0
        sample.detector_data = [
            IQData(q=q_first.tolist(), i=intensity.tolist(), e=[0.01] * len(q_first)),
            IQData(q=q_second.tolist(), i=[1.0] * len(q_second), e=[0.01] * len(q_second)),
        ]

        center = sample.rocking_curve_centering()

        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(sample.detector_data[0].q, q_first - expected_center)
        np.testing.assert_allclose(sample.detector_data[1].q, q_second - expected_center)


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
