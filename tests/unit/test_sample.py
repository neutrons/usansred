# ===========================================================================
# Tests for Sample class properties
# ===========================================================================

import csv
import logging
import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

import usansred.reduce
from tests.test_fixtures import _make_sample
from usansred.enums import MeasurementType
from usansred.models import EventCounts, IQData, MonitorData, XYData
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
        experiment: Experiment, harmonic: int, thickness: float, detector_data: IQData, transmission: float = 1.0
    ) -> tuple[list[float], list[float], list[float]]:
        theta_to_q = ARCSEC_TO_RADIANS * (2 * np.pi / (experiment.prim_wave / harmonic))
        analyzer_solid_angle = experiment.v_angle * (horizontal_rocking_width(harmonic) * ARCSEC_TO_RADIANS)
        scaling_factor = 1.0 / (analyzer_solid_angle * thickness * transmission)

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

    def test_scales_by_transmission_coefficient(self, mock_experiment):
        """Intensities and errors should be divided by the transmission coefficient (Q unchanged)."""
        detector_data = IQData(q=[0.0, 1.0, 3.0], i=[2.0, 4.0, 6.0], e=[0.2, 0.4, 0.6])
        sample = self._make_rescale_sample(mock_experiment, [detector_data], thickness=0.4)
        sample.transmission = 0.5

        sample.rescale_data()

        # A transmission of 0.5 should double intensities and errors w.r.t. the transmission-1 baseline
        baseline_q, baseline_i, baseline_e = self._expected_rescaled_data(
            mock_experiment, harmonic=1, thickness=sample.thickness, detector_data=detector_data, transmission=1.0
        )
        np.testing.assert_allclose(sample.data_scaled[0].q, baseline_q)
        np.testing.assert_allclose(sample.data_scaled[0].i, [2.0 * i for i in baseline_i])
        np.testing.assert_allclose(sample.data_scaled[0].e, [2.0 * e for e in baseline_e])

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


class TestSampleTransmissionValidation:
    """Tests for the transmission coefficient validation in Sample.model_post_init."""

    @staticmethod
    def _make_empty_cell(experiment: Experiment, transmitted: float) -> Sample:
        empty_cell = _make_sample(experiment, "ec", [])
        empty_cell.transmitted = transmitted
        return empty_cell

    def test_raises_when_sample_transmitted_counts_are_zero(self, mock_experiment):
        """Zero transmitted counts for the sample yield transmission == 0, which should raise."""
        mock_experiment.empty_cell = self._make_empty_cell(mock_experiment, transmitted=2.0)

        with pytest.raises(ValueError, match="Invalid transmission coefficient"):
            Sample(
                name="test",
                experiment=mock_experiment,
                start_scan_num=0,
                num_of_scans=0,
                counts=EventCounts(monitor=100, detector=0, transmission=0),
            )

    def test_raises_when_transmission_is_non_finite(self, mock_experiment):
        """A non-finite empty-cell transmitted value yields a non-finite transmission, which should raise."""
        mock_experiment.empty_cell = self._make_empty_cell(mock_experiment, transmitted=math.nan)

        with pytest.raises(ValueError, match="Invalid transmission coefficient"):
            Sample(
                name="test",
                experiment=mock_experiment,
                start_scan_num=0,
                num_of_scans=0,
                counts=EventCounts(monitor=100, detector=10, transmission=5),
            )

    def test_raises_when_transmission_is_negative(self, mock_experiment):
        """A negative empty-cell transmitted value yields a negative transmission, which should raise."""
        mock_experiment.empty_cell = self._make_empty_cell(mock_experiment, transmitted=-1.0)

        with pytest.raises(ValueError, match="Invalid transmission coefficient"):
            Sample(
                name="test",
                experiment=mock_experiment,
                start_scan_num=0,
                num_of_scans=0,
                counts=EventCounts(monitor=100, detector=10, transmission=5),
            )

    def test_falls_back_to_unity_when_empty_cell_transmitted_is_zero(self, mock_experiment):
        """A ZeroDivisionError (empty cell transmitted == 0) should still fall back to 1.0, not raise."""
        mock_experiment.empty_cell = self._make_empty_cell(mock_experiment, transmitted=0.0)

        sample = Sample(
            name="test",
            experiment=mock_experiment,
            start_scan_num=0,
            num_of_scans=0,
            counts=EventCounts(monitor=100, detector=10, transmission=5),
        )

        assert sample.transmission == 1.0


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


class TestDumpAllHarmonics:
    """Tests for the ``save_all_harmonics`` output files of Sample.dump_reduced_data_to_csv."""

    @staticmethod
    def _make_two_bank_sample(experiment: Experiment, save_all_harmonics: bool, tmp_path) -> Sample:
        experiment.output_dir = str(tmp_path)
        experiment._config.save_all_harmonics = save_all_harmonics
        sample = _make_sample(experiment, "test", [])
        # One (Q,I,E) curve per detector bank, both for the unscaled and the rescaled data
        sample.detector_data = [
            IQData(q=[0.1, 0.2], i=[100.0, 200.0], e=[10.0, 14.0]),
            IQData(q=[0.3, 0.4], i=[300.0, 400.0], e=[17.0, 20.0]),
        ]
        sample.data_scaled = [
            IQData(q=[0.1, 0.2], i=[1.0, 2.0], e=[0.1, 0.2]),
            IQData(q=[0.3, 0.4], i=[3.0, 4.0], e=[0.3, 0.4]),
        ]
        return sample

    def test_writes_one_file_per_harmonic(self, mock_experiment_2banks, tmp_path):
        """With save_all_harmonics, each bank is written flat as ``_det_<n>``."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)

        sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        for harmonic in (1, 2):
            assert (tmp_path / f"UN_test_det_{harmonic}_unscaled.txt").is_file()
            assert (tmp_path / f"UN_test_det_{harmonic}.txt").is_file()
        assert not list(tmp_path.glob("bank_*")), "no per-bank subdirectories should be created"

    def test_higher_harmonics_hold_their_own_bank_data(self, mock_experiment_2banks, tmp_path):
        """The ``_det_2`` files hold bank 2's curve, not a copy of bank 1's."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)

        sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        rows = list(csv.reader((tmp_path / "UN_test_det_2_unscaled.txt").read_text().splitlines()))
        assert [float(row[0]) for row in rows] == [0.3, 0.4]
        assert [float(row[1]) for row in rows] == [300.0, 400.0]

    def test_only_first_harmonic_without_the_flag(self, mock_experiment_2banks, tmp_path):
        """Without save_all_harmonics, only the first harmonic is written."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, False, tmp_path)

        sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        assert (tmp_path / "UN_test_det_1_unscaled.txt").is_file()
        assert (tmp_path / "UN_test_det_1.txt").is_file()
        assert not (tmp_path / "UN_test_det_2_unscaled.txt").exists()
        assert not (tmp_path / "UN_test_det_2.txt").exists()

    def test_missing_first_detector_harmonic_aborts_output(self, mock_experiment_2banks, tmp_path):
        """Missing first-harmonic detector data must abort output generation."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)
        sample.detector_data = []

        with pytest.raises(RuntimeError, match="first harmonic data is missing"):
            sample.dump_reduced_data_to_csv(scaled_data=False, background_subtracted_data=False)

    def test_missing_first_scaled_harmonic_aborts_output(self, mock_experiment_2banks, tmp_path):
        """Missing first-harmonic scaled data must abort output generation."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)
        sample.data_scaled = []

        with pytest.raises(RuntimeError, match="first harmonic data is missing"):
            sample.dump_reduced_data_to_csv(detector_data=False, background_subtracted_data=False)

    def test_missing_higher_harmonics_warns_and_skips_files(self, mock_experiment_2banks, tmp_path, caplog):
        """Missing higher harmonics should be warned about and skipped."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)
        sample.detector_data = sample.detector_data[:1]
        sample.data_scaled = sample.data_scaled[:1]

        with caplog.at_level(logging.WARNING):
            sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        assert (tmp_path / "UN_test_det_1_unscaled.txt").is_file()
        assert (tmp_path / "UN_test_det_1.txt").is_file()
        assert not (tmp_path / "UN_test_det_2_unscaled.txt").exists()
        assert not (tmp_path / "UN_test_det_2.txt").exists()
        assert "No detector data is available" in caplog.text
        assert "No scaled data is available" in caplog.text

    def test_empty_first_scaled_harmonic_aborts_output(self, mock_experiment_2banks, tmp_path):
        """An empty first-harmonic scaled curve must abort output generation."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)
        sample.data_scaled[0] = IQData()

        with pytest.raises(RuntimeError, match="first harmonic data is missing"):
            sample.dump_reduced_data_to_csv(detector_data=False, background_subtracted_data=False)


class TestDumpBackgroundSubtracted:
    """Tests for dumping the background-subtracted data file."""

    def test_written_when_subtraction_occurred(self, mock_experiment, tmp_path):
        mock_experiment.output_dir = str(tmp_path)
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = IQData(q=[0.1, 0.2], i=[10.0, 20.0], e=[1.0, 2.0])

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        subtracted_file = tmp_path / "UN_test_det_1_background_subtracted.txt"
        assert subtracted_file.is_file()
        assert subtracted_file.stat().st_size > 0

    def test_not_written_when_nothing_subtracted(self, mock_experiment, tmp_path):
        mock_experiment.output_dir = str(tmp_path)
        sample = _make_sample(mock_experiment, "test", [])

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        assert list(tmp_path.iterdir()) == []


class TestSampleReduceBranching:
    """Tests for the background/empty-cell subtraction branching in Sample.reduce."""

    @staticmethod
    def _run_reduce(sample: Sample) -> list[Sample]:
        """Run Sample.reduce with the pipeline steps mocked; return the subtracted measurements."""
        subtracted = []

        def fake_rescale(self):
            self.data_scaled = [IQData(q=[1.0], i=[1.0], e=[0.1])]

        with (
            patch.object(Sample, "normalize_by_monitor", autospec=True, return_value=None),
            patch.object(Sample, "stitch_scans", autospec=True, return_value=None),
            patch.object(Sample, "rocking_curve_centering", autospec=True, return_value=None),
            patch.object(Sample, "rescale_data", autospec=True, side_effect=fake_rescale),
            patch.object(
                Sample,
                "subtract_background",
                autospec=True,
                side_effect=lambda _self, background: subtracted.append(background),
            ),
        ):
            sample.reduce()
        return subtracted

    @staticmethod
    def _add_background_and_empty_cell(
        experiment: Experiment, background: bool, empty_cell: bool
    ) -> tuple[Sample | None, Sample | None]:
        if background:
            experiment.background = _make_sample(experiment, "bg", [])
            experiment.background.measurement_type = MeasurementType.BACKGROUND
        if empty_cell:
            experiment.empty_cell = _make_sample(experiment, "ec", [])
            experiment.empty_cell.measurement_type = MeasurementType.EMPTY_CELL
        return experiment.background, experiment.empty_cell

    def test_background_takes_precedence_over_empty_cell(self, mock_experiment):
        background, _ = self._add_background_and_empty_cell(mock_experiment, background=True, empty_cell=True)
        sample = _make_sample(mock_experiment, "test", [])

        subtracted = self._run_reduce(sample)

        assert subtracted == [background]

    def test_empty_cell_subtracted_in_absence_of_background(self, mock_experiment):
        _, empty_cell = self._add_background_and_empty_cell(mock_experiment, background=False, empty_cell=True)
        sample = _make_sample(mock_experiment, "test", [])

        subtracted = self._run_reduce(sample)

        assert subtracted == [empty_cell]

    def test_background_subtracted_when_no_empty_cell(self, mock_experiment):
        background, _ = self._add_background_and_empty_cell(mock_experiment, background=True, empty_cell=False)
        sample = _make_sample(mock_experiment, "test", [])

        subtracted = self._run_reduce(sample)

        assert subtracted == [background]

    def test_no_subtraction_without_background_or_empty_cell(self, mock_experiment):
        sample = _make_sample(mock_experiment, "test", [])

        subtracted = self._run_reduce(sample)

        assert subtracted == []

    @pytest.mark.parametrize("measurement_type", [MeasurementType.BACKGROUND, MeasurementType.EMPTY_CELL])
    def test_no_subtraction_for_non_sample_measurements(self, mock_experiment, measurement_type):
        self._add_background_and_empty_cell(mock_experiment, background=True, empty_cell=True)
        measurement = _make_sample(mock_experiment, "test", [])
        measurement.measurement_type = measurement_type

        subtracted = self._run_reduce(measurement)

        assert subtracted == []


class TestSampleLogLabels:
    """Tests for the measurement-type-aware label used in log messages."""

    @pytest.mark.parametrize(
        ("measurement_type", "expected_label"),
        [
            (MeasurementType.SAMPLE, "sample test"),
            (MeasurementType.BACKGROUND, "background test"),
            (MeasurementType.EMPTY_CELL, "empty cell test"),
        ],
    )
    def test_label_spells_out_measurement_type(self, mock_experiment, measurement_type, expected_label):
        sample = _make_sample(mock_experiment, "test", [])
        sample.measurement_type = measurement_type

        assert sample.label == expected_label

    @pytest.mark.parametrize(
        ("measurement_type", "expected_label"),
        [
            (MeasurementType.SAMPLE, "sample test"),
            (MeasurementType.BACKGROUND, "background test"),
            (MeasurementType.EMPTY_CELL, "empty cell test"),
        ],
    )
    def test_reduce_logs_measurement_type(self, mock_experiment, measurement_type, expected_label, caplog, monkeypatch):
        sample = _make_sample(mock_experiment, "test", [])
        sample.measurement_type = measurement_type
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)

        with caplog.at_level(logging.INFO):
            TestSampleReduceBranching._run_reduce(sample)

        assert f"Starting reduction for {expected_label} with 0 scans." in caplog.messages
        assert f"Data reduction finished for {expected_label}." in caplog.messages


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
