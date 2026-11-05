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
from usansred.reduce import (
    ARCSEC_TO_RADIANS,
    Experiment,
    Sample,
    Scan,
    horizontal_rocking_width,
    unscattered_beam_q_cutoff,
)


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
        """data_reduced should return the first harmonic of data_bg_subtracted."""
        sample = _make_sample(mock_experiment, "test", [])
        bg = IQData(q=[1.0], i=[5.0], e=[0.5])
        sample.data_bg_subtracted = [bg, IQData(q=[2.0], i=[6.0], e=[0.6])]
        assert sample.data_reduced is bg

    def test_data_reduced_property_no_data(self, mock_experiment):
        """data_reduced should be None when no subtraction has happened."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.data_reduced is None

    def test_is_reduced_false(self, mock_experiment):
        """is_reduced should be False when data_bg_subtracted is empty."""
        sample = _make_sample(mock_experiment, "test", [])
        assert sample.is_reduced is False

    def test_is_reduced_false_empty_first_harmonic(self, mock_experiment):
        """is_reduced should be False when the first harmonic carries no points."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = [IQData()]
        assert sample.is_reduced is False

    def test_is_reduced_true(self, mock_experiment):
        """is_reduced should be True when the first harmonic has values."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = [IQData(q=[1.0], i=[10.0], e=[1.0])]
        assert sample.is_reduced is True

    def test_size_reduced(self, mock_experiment):
        """size_reduced should return length of the first harmonic's q."""
        sample = _make_sample(mock_experiment, "test", [])
        sample.data_bg_subtracted = [IQData(q=[1.0, 2.0], i=[10.0, 20.0], e=[1.0, 2.0])]
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

    def test_falls_back_to_unity_when_empty_cell_transmitted_is_zero(self, mock_experiment, caplog):
        """A ZeroDivisionError (empty cell transmitted == 0) should still fall back to 1.0, not raise."""
        mock_experiment.empty_cell = self._make_empty_cell(mock_experiment, transmitted=0.0)

        with caplog.at_level(logging.WARNING, logger="usansred.reduce"):
            sample = Sample(
                name="test",
                experiment=mock_experiment,
                start_scan_num=0,
                num_of_scans=0,
                counts=EventCounts(monitor=100, detector=10, transmission=5),
            )

        assert sample.transmission == 1.0
        warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "zero transmitted counts" in warnings[0]
        assert "NoneType" not in warnings[0]

    def test_falls_back_to_unity_when_no_empty_cell_is_configured(self, mock_experiment, caplog):
        """Without an empty cell the correction is skipped silently; the Experiment logs it once."""
        assert mock_experiment.empty_cell is None

        with caplog.at_level(logging.WARNING, logger="usansred.reduce"):
            sample = Sample(
                name="test",
                experiment=mock_experiment,
                start_scan_num=0,
                num_of_scans=0,
                counts=EventCounts(monitor=100, detector=10, transmission=5),
            )

        assert sample.transmission == 1.0
        assert [record.message for record in caplog.records if record.levelno == logging.WARNING] == []


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
            assert (tmp_path / f"test_det_{harmonic}_unscaled.txt").is_file()
            assert (tmp_path / f"test_det_{harmonic}.txt").is_file()
        assert not list(tmp_path.glob("bank_*")), "no per-bank subdirectories should be created"

    def test_higher_harmonics_hold_their_own_bank_data(self, mock_experiment_2banks, tmp_path):
        """The ``_det_2`` files hold bank 2's curve, not a copy of bank 1's."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, True, tmp_path)

        sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        rows = list(csv.reader((tmp_path / "test_det_2_unscaled.txt").read_text().splitlines()))
        assert [float(row[0]) for row in rows] == [0.3, 0.4]
        assert [float(row[1]) for row in rows] == [300.0, 400.0]

    def test_only_first_harmonic_without_the_flag(self, mock_experiment_2banks, tmp_path):
        """Without save_all_harmonics, only the first harmonic is written."""
        sample = self._make_two_bank_sample(mock_experiment_2banks, False, tmp_path)

        sample.dump_reduced_data_to_csv(background_subtracted_data=False)

        assert (tmp_path / "test_det_1_unscaled.txt").is_file()
        assert (tmp_path / "test_det_1.txt").is_file()
        assert not (tmp_path / "test_det_2_unscaled.txt").exists()
        assert not (tmp_path / "test_det_2.txt").exists()

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

        assert (tmp_path / "test_det_1_unscaled.txt").is_file()
        assert (tmp_path / "test_det_1.txt").is_file()
        assert not (tmp_path / "test_det_2_unscaled.txt").exists()
        assert not (tmp_path / "test_det_2.txt").exists()
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
        sample.data_bg_subtracted = [IQData(q=[0.1, 0.2], i=[10.0, 20.0], e=[1.0, 2.0])]

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        subtracted_file = tmp_path / "test_det_1_background_subtracted.txt"
        assert subtracted_file.is_file()
        assert subtracted_file.stat().st_size > 0

    def test_not_written_when_nothing_subtracted(self, mock_experiment, tmp_path):
        mock_experiment.output_dir = str(tmp_path)
        sample = _make_sample(mock_experiment, "test", [])

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        assert list(tmp_path.iterdir()) == []

    @staticmethod
    def _two_harmonic_sample(experiment: Experiment, save_all_harmonics: bool, tmp_path) -> Sample:
        experiment.output_dir = str(tmp_path)
        experiment._config.save_all_harmonics = save_all_harmonics
        sample = _make_sample(experiment, "test", [])
        sample.data_bg_subtracted = [
            IQData(q=[0.1, 0.2], i=[1.0, 2.0], e=[0.1, 0.2]),
            IQData(q=[0.3, 0.4], i=[3.0, 4.0], e=[0.3, 0.4]),
        ]
        return sample

    def test_higher_harmonics_written_with_save_all_harmonics(self, mock_experiment_2banks, tmp_path):
        """With save_all_harmonics, every harmonic gets its own subtracted file."""
        sample = self._two_harmonic_sample(mock_experiment_2banks, True, tmp_path)

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        for harmonic in (1, 2):
            assert (tmp_path / f"test_det_{harmonic}_background_subtracted.txt").is_file()

    def test_higher_harmonics_hold_their_own_data(self, mock_experiment_2banks, tmp_path):
        """The ``_det_2`` subtracted file holds harmonic 2's curve, not a copy of harmonic 1's."""
        sample = self._two_harmonic_sample(mock_experiment_2banks, True, tmp_path)

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        contents = (tmp_path / "test_det_2_background_subtracted.txt").read_text()
        assert "0.3" in contents
        assert "0.4" in contents
        assert "0.1" not in contents

    def test_higher_harmonics_skipped_without_save_all_harmonics(self, mock_experiment_2banks, tmp_path):
        """Without save_all_harmonics, only the first harmonic is written."""
        sample = self._two_harmonic_sample(mock_experiment_2banks, False, tmp_path)

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        assert (tmp_path / "test_det_1_background_subtracted.txt").is_file()
        assert not (tmp_path / "test_det_2_background_subtracted.txt").exists()

    def test_empty_higher_harmonic_is_skipped(self, mock_experiment_2banks, tmp_path):
        """A placeholder harmonic produces no file, and does not block the first harmonic."""
        sample = self._two_harmonic_sample(mock_experiment_2banks, True, tmp_path)
        sample.data_bg_subtracted[1] = IQData()

        sample.dump_reduced_data_to_csv(detector_data=False, scaled_data=False)

        assert (tmp_path / "test_det_1_background_subtracted.txt").is_file()
        assert not (tmp_path / "test_det_2_background_subtracted.txt").exists()


class TestSubtractBackground:
    """Tests for the per-harmonic arithmetic of Sample.subtract_background."""

    @staticmethod
    def _pair(experiment: Experiment) -> tuple[Sample, Sample]:
        """A sample and a background sharing a Q grid, with distinct intensities per harmonic."""
        sample = _make_sample(experiment, "test", [])
        background = _make_sample(experiment, "bg", [])
        # Distinct intensities per harmonic, so subtracting the wrong harmonic would be detected
        sample.data_scaled = [
            IQData(q=[0.1, 0.2], i=[10.0, 20.0], e=[1.0, 2.0]),
            IQData(q=[0.1, 0.2], i=[30.0, 40.0], e=[3.0, 4.0]),
        ]
        background.data_scaled = [
            IQData(q=[0.1, 0.2], i=[1.0, 2.0], e=[0.5, 0.5]),
            IQData(q=[0.1, 0.2], i=[3.0, 4.0], e=[0.5, 0.5]),
        ]
        return sample, background

    def test_all_harmonics_subtracted(self, mock_experiment_2banks):
        """Every harmonic is subtracted, harmonic n from harmonic n."""
        sample, background = self._pair(mock_experiment_2banks)

        sample.subtract_background(background)

        assert len(sample.data_bg_subtracted) == 2
        assert sample.data_bg_subtracted[0].i == pytest.approx([9.0, 18.0])
        assert sample.data_bg_subtracted[1].i == pytest.approx([27.0, 36.0])

    def test_errors_propagated_in_quadrature(self, mock_experiment_2banks):
        """Uncertainties add in quadrature for each harmonic."""
        sample, background = self._pair(mock_experiment_2banks)

        sample.subtract_background(background)

        assert sample.data_bg_subtracted[0].e == pytest.approx([math.hypot(1.0, 0.5), math.hypot(2.0, 0.5)])
        assert sample.data_bg_subtracted[1].e == pytest.approx([math.hypot(3.0, 0.5), math.hypot(4.0, 0.5)])

    def test_q_grid_preserved(self, mock_experiment_2banks):
        """The subtracted curve keeps the sample's momentum-transfer grid, in 1/angstrom."""
        sample, background = self._pair(mock_experiment_2banks)

        sample.subtract_background(background)

        for harmonic in (0, 1):
            assert sample.data_bg_subtracted[harmonic].q == pytest.approx([0.1, 0.2])

    def test_missing_higher_harmonic_yields_placeholder(self, mock_experiment_2banks):
        """A missing background harmonic leaves an empty placeholder without aborting the rest."""
        sample, background = self._pair(mock_experiment_2banks)
        background.data_scaled[1] = IQData()

        sample.subtract_background(background)

        assert len(sample.data_bg_subtracted) == 2
        assert sample.data_bg_subtracted[0].i == pytest.approx([9.0, 18.0])
        assert sample.data_bg_subtracted[1].q == []
        assert sample.is_reduced is True

    def test_repeated_calls_do_not_accumulate(self, mock_experiment_2banks):
        """Calling twice replaces the results instead of appending to them."""
        sample, background = self._pair(mock_experiment_2banks)

        sample.subtract_background(background)
        sample.subtract_background(background)

        assert len(sample.data_bg_subtracted) == 2

    def test_missing_first_harmonic_raises(self, mock_experiment_2banks):
        """The first harmonic is required for both the sample and the background."""
        sample, background = self._pair(mock_experiment_2banks)
        background.data_scaled[0] = IQData()

        with pytest.raises(
            RuntimeError,
            match=(
                "Cannot subtract sample bg from sample test: the first-harmonic scaled data of sample bg is missing"
            ),
        ):
            sample.subtract_background(background)

    def test_missing_sample_first_harmonic_raises(self, mock_experiment_2banks):
        """A missing sample first harmonic identifies the sample in the error."""
        sample, background = self._pair(mock_experiment_2banks)
        sample.data_scaled[0] = IQData()

        with pytest.raises(
            RuntimeError,
            match=(
                "Cannot subtract sample bg from sample test: the first-harmonic scaled data of sample test is missing"
            ),
        ):
            sample.subtract_background(background)

    def test_interpolates_background_onto_sample_grid(self, mock_experiment_2banks):
        """A background measured on a different grid is interpolated before subtraction."""
        sample, background = self._pair(mock_experiment_2banks)
        # Background sampled at the midpoint only; interp clamps to 2.0 at both sample Q values
        background.data_scaled[0] = IQData(q=[0.15], i=[2.0], e=[0.5])

        sample.subtract_background(background)

        assert sample.data_bg_subtracted[0].i == pytest.approx([8.0, 18.0])


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
        # Uncertainties combine in quadrature with squared weights:
        # sqrt((0.5 * 1.0)**2 + (0.5 * 2.0)**2) = 1.118034
        np.testing.assert_allclose(e_matched, [np.sqrt(0.5**2 + 1.0**2)])

    def test_interpolated_error_is_propagated_in_quadrature(self, mock_experiment):
        """Interpolated uncertainties combine in quadrature, not linearly (issue #87)."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([1.5])  # midpoint between 1.0 and 2.0
        q_bg = np.array([1.0, 2.0])
        i_bg = np.array([10.0, 20.0])
        e_bg = np.array([1.0, 3.0])

        _, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        # sqrt((0.5 * 1.0)**2 + (0.5 * 3.0)**2) = 1.5811388
        np.testing.assert_allclose(e_matched, [np.sqrt(0.5**2 + 1.5**2)])
        # Not the linear interpolation of sigma, which overestimates the uncertainty
        assert not np.isclose(e_matched[0], 2.0)
        # Nor the variance interpolated with unsquared weights
        assert not np.isclose(e_matched[0], np.sqrt(0.5 * 1.0**2 + 0.5 * 3.0**2))

    def test_interpolated_error_with_unequal_weights(self, mock_experiment):
        """Away from the midpoint the two endpoint uncertainties carry different weights."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([1.25])  # weight 0.25 towards the upper background point
        q_bg = np.array([1.0, 2.0])
        i_bg = np.array([10.0, 20.0])
        e_bg = np.array([1.0, 3.0])

        i_matched, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        np.testing.assert_allclose(i_matched, [12.5])
        # sqrt((0.75 * 1.0)**2 + (0.25 * 3.0)**2) = 1.0606602
        np.testing.assert_allclose(e_matched, [np.sqrt(0.75**2 + 0.75**2)])

    def test_interpolated_error_outside_background_range(self, mock_experiment):
        """Outside the background range, clamp to the nearest endpoint like ``np.interp`` does."""
        sample = _make_sample(mock_experiment, "test", [])

        q_data = np.array([0.5, 2.5])  # below q_bg[0] and above q_bg[-1]
        q_bg = np.array([1.0, 2.0])
        i_bg = np.array([10.0, 20.0])
        e_bg = np.array([1.0, 3.0])

        i_matched, e_matched = sample._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        np.testing.assert_allclose(i_matched, [10.0, 20.0])
        np.testing.assert_allclose(e_matched, [1.0, 3.0])

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


# ===========================================================================
# Tests for Sample.stitch_orders and the stitched output files
# ===========================================================================


class TestStitchOrders:
    """Tests for stitching the harmonics of a Sample and writing the stitched profiles."""

    # Harmonic n spans a Q range n times wider than harmonic 1, on its own Q grid (1/angstrom)
    Q1 = np.linspace(1.0e-5, 1.0e-3, 20)
    Q2 = np.linspace(2.0e-5, 2.0e-3, 17)
    SCALE = 2.5  # factor that scales harmonic 2 onto harmonic 1
    BACKGROUND = 0.4  # constant background of harmonic 2, i.e. SCALE * BACKGROUND = 1.0 after stitching

    @staticmethod
    def _profile(q: np.ndarray, scale: float = 1.0, offset: float = 0.0) -> IQData:
        """Synthetic profile ``(10 - 2000 q) / scale - offset``, linear in Q so interpolation is exact.

        Parameters
        ----------
        q : np.ndarray
            Momentum transfer in ``1/angstrom``.
        scale : float
            Factor that stitching must apply to the profile to match harmonic 1.
        offset : float
            Constant subtracted from the intensity, standing in for a subtracted background.

        Returns
        -------
        IQData
            The synthetic profile, with 5 % relative uncertainties before the offset.
        """
        i = (10.0 - 2000.0 * q) / scale
        return IQData(q=q.tolist(), i=(i - offset).tolist(), e=(0.05 * i).tolist(), t=[])

    @classmethod
    def _sample(cls, experiment: Experiment, tmp_path, save_all_harmonics: bool = True, subtracted: bool = True):
        """Two-harmonic sample with scaled data and, optionally, background-subtracted data.

        Parameters
        ----------
        experiment : Experiment
            The ``mock_experiment_2banks`` fixture.
        tmp_path : pathlib.Path
            Output directory.
        save_all_harmonics : bool
            Value of the ``save_all_harmonics`` setup flag.
        subtracted : bool
            Whether a background subtraction took place.

        Returns
        -------
        Sample
            The sample, ready for ``stitch_orders`` and ``dump_reduced_data_to_csv``.
        """
        experiment.output_dir = str(tmp_path)
        experiment._config.save_all_harmonics = save_all_harmonics
        sample = _make_sample(experiment, "test", [])
        sample.detector_data = [cls._profile(cls.Q1), cls._profile(cls.Q2)]
        sample.data_scaled = [cls._profile(cls.Q1), cls._profile(cls.Q2, scale=cls.SCALE)]
        if subtracted:
            sample.data_bg_subtracted = [
                cls._profile(cls.Q1, offset=1.0),
                cls._profile(cls.Q2, scale=cls.SCALE, offset=cls.BACKGROUND),
            ]
        return sample

    @staticmethod
    def _read(path) -> list[list[float]]:
        """Read the Q, I, and E columns of an output file.

        Parameters
        ----------
        path : pathlib.Path
            Output file written by ``Sample.dump_data_to_csv``.

        Returns
        -------
        list[list[float]]
            One ``[Q, I, E]`` row per line.
        """
        return [[float(value) for value in row[:3]] for row in csv.reader(path.read_text().splitlines())]

    def test_stitches_scaled_data(self, mock_experiment_2banks, tmp_path):
        """The scaled harmonics are stitched, with harmonic 2 scaled by the known factor."""
        sample = self._sample(mock_experiment_2banks, tmp_path)

        sample.stitch_orders()

        assert sample.stitched_orders == [1, 2]
        assert sample.stitch_scale_factors == pytest.approx([1.0, self.SCALE], rel=1e-9)
        q = np.array(sample.data_stitched.q)
        assert np.all(np.diff(q) >= 0)
        np.testing.assert_allclose(sample.data_stitched.i, 10.0 - 2000.0 * q, rtol=1e-9)

    def test_background_subtracted_stitch_records_its_orders(self, mock_experiment_2banks, tmp_path):
        """With background-subtracted data for every harmonic, both stitched profiles hold the same orders."""
        sample = self._sample(mock_experiment_2banks, tmp_path)

        sample.stitch_orders()

        assert sample.stitched_orders == [1, 2]
        assert sample.bg_subtracted_stitched_orders == [1, 2]

    def test_background_subtracted_stitch_stops_at_missing_background_harmonic(
        self, mock_experiment_2banks, tmp_path, caplog, monkeypatch
    ):
        """A harmonic missing from the background stops the background-subtracted stitch only, with a warning.

        ``Sample.subtract_background`` leaves an empty placeholder for a harmonic that the
        background lacks. The stitch of the scaled data still holds harmonics 1 to 3, while the
        background-subtracted one holds harmonics 1 and 2, and the log says which stitch stopped.
        """
        q3 = np.linspace(3.0e-5, 3.0e-3, 15)
        scale3 = 4.0
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled.append(self._profile(q3, scale=scale3))
        sample.data_bg_subtracted.append(IQData())  # placeholder for the missing background harmonic
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.stitch_orders()

        assert sample.stitched_orders == [1, 2, 3]
        assert sample.stitch_scale_factors == pytest.approx([1.0, self.SCALE, scale3], rel=1e-9)
        assert sample.bg_subtracted_stitched_orders == [1, 2]
        assert sample.data_bg_subtracted_stitched.q  # the profile of orders 1 and 2 is kept
        [record] = [r for r in caplog.records if "Cannot join" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert record.getMessage().startswith(
            "Cannot join order 3 onto the stitched profile of background-subtracted sample test: order 3 has no data."
        )
        assert "Stitched background-subtracted orders [1, 2] of sample test." in caplog.messages

    def test_unscattered_beam_q_cutoff(self):
        """The cutoff is 3 sigma of the harmonic-1 resolution, converted to Q, i.e. 5.65e-5 1/angstrom at 3.6 angstrom.

        It scales with the number of standard deviations and inversely with the wavelength.
        """
        cutoff = unscattered_beam_q_cutoff(3.6)

        assert cutoff == pytest.approx(5.654e-5, rel=1e-3)
        assert unscattered_beam_q_cutoff(3.6, num_sigmas=1.0) == pytest.approx(cutoff / 3.0)
        assert unscattered_beam_q_cutoff(7.2) == pytest.approx(cutoff / 2.0)

    def test_stitching_uses_unscattered_beam_cutoff(self, mock_experiment_2banks, tmp_path, monkeypatch):
        """stitch_orders fits the scale factors above the unscattered-beam cutoff of the primary wavelength."""
        sample = self._sample(mock_experiment_2banks, tmp_path)
        calls = []
        original = usansred.reduce.stitch_profiles

        def spy(*args, **kwargs):
            calls.append(kwargs["q_fit_min"])
            return original(*args, **kwargs)

        monkeypatch.setattr(usansred.reduce, "stitch_profiles", spy)

        sample.stitch_orders()

        assert calls == [pytest.approx(unscattered_beam_q_cutoff(mock_experiment_2banks.prim_wave))]
        # Harmonic-2 points below the cutoff are left out of the stitched profile
        added = np.count_nonzero(self.Q2 >= calls[0])
        assert len(sample.data_stitched.q) == len(self.Q1) + added

    def test_background_subtracted_data_reuses_scale_factors(self, mock_experiment_2banks, tmp_path):
        """The background-subtracted harmonics are stitched with the factors of the scaled data.

        The background of harmonic 2 becomes SCALE * BACKGROUND = 1.0 after scaling, the same as
        that of harmonic 1, so the stitched background-subtracted profile is the stitched scaled
        profile minus 1 on the same Q points.
        """
        sample = self._sample(mock_experiment_2banks, tmp_path)

        sample.stitch_orders()

        assert sample.data_bg_subtracted_stitched.q == sample.data_stitched.q
        difference = np.array(sample.data_stitched.i) - np.array(sample.data_bg_subtracted_stitched.i)
        np.testing.assert_allclose(difference, 1.0, rtol=1e-9)

    def test_nothing_subtracted_leaves_background_subtracted_profile_empty(self, mock_experiment_2banks, tmp_path):
        """Without a background subtraction, only the scaled data are stitched."""
        sample = self._sample(mock_experiment_2banks, tmp_path, subtracted=False)

        sample.stitch_orders()

        assert sample.data_stitched.q
        assert sample.data_bg_subtracted_stitched.q == []

    def test_failed_join_of_harmonic_2_produces_no_profile(self, mock_experiment_2banks, tmp_path, caplog):
        """When harmonics 1 and 2 cannot be stitched, both stitched profiles stay empty.

        Harmonic 2 is shifted to Q values beyond harmonic 1, so the two do not overlap.
        """
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled[1] = self._profile(self.Q2 + 1.0e-2)

        with caplog.at_level(logging.WARNING):
            sample.stitch_orders()

        assert sample.data_stitched.q == []
        assert sample.data_bg_subtracted_stitched.q == []
        assert sample.stitched_orders == []
        assert "do not overlap" in caplog.text
        assert "Orders 1 and 2 of sample test could not be stitched" in caplog.text

    def test_mismatched_background_subtracted_harmonic_2_produces_no_profile(
        self, mock_experiment_2banks, tmp_path, caplog
    ):
        """A background-subtracted harmonic 2 that does not match its scaled data is not stitched.

        The scaled profile is still stitched; only the background-subtracted one stays empty.
        """
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.data_bg_subtracted[1] = IQData()

        with caplog.at_level(logging.WARNING):
            sample.stitch_orders()

        assert sample.stitched_orders == [1, 2]
        assert sample.data_bg_subtracted_stitched.q == []
        assert "Background-subtracted orders 1 and 2 of sample test could not be stitched" in caplog.text

    def test_unexpected_exception_is_logged_and_does_not_propagate(
        self, mock_experiment_2banks, tmp_path, caplog, monkeypatch
    ):
        """An unexpected exception during stitching is logged, and the stitched profiles stay empty."""
        sample = self._sample(mock_experiment_2banks, tmp_path)
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)

        def broken_stitch(*_args, **_kwargs):
            raise ZeroDivisionError("synthetic failure")

        monkeypatch.setattr(usansred.reduce, "stitch_profiles", broken_stitch)

        with caplog.at_level(logging.ERROR):
            sample.stitch_orders()

        assert sample.data_stitched.q == []
        assert sample.stitch_scale_factors == []
        assert "Cannot stitch the orders of sample test: synthetic failure" in caplog.text

    def test_reduce_stitches_after_background_subtraction(self, mock_experiment_2banks):
        """Sample.reduce calls stitch_orders after subtract_background when save_all_harmonics is set."""
        mock_experiment_2banks._config.save_all_harmonics = True
        mock_experiment_2banks.background = _make_sample(mock_experiment_2banks, "bg", [])
        sample = _make_sample(mock_experiment_2banks, "test", [])
        calls = []

        with (
            patch.object(Sample, "normalize_by_monitor", autospec=True, return_value=None),
            patch.object(Sample, "stitch_scans", autospec=True, return_value=None),
            patch.object(Sample, "rocking_curve_centering", autospec=True, return_value=None),
            patch.object(Sample, "rescale_data", autospec=True, return_value=None),
            patch.object(Sample, "subtract_background", autospec=True, side_effect=lambda *_: calls.append("subtract")),
            patch.object(Sample, "stitch_orders", autospec=True, side_effect=lambda *_: calls.append("stitch")),
        ):
            sample.reduce()

        assert calls == ["subtract", "stitch"]

    def test_reduce_does_not_stitch_without_save_all_harmonics(self, mock_experiment_2banks):
        """Without save_all_harmonics, Sample.reduce does not stitch the harmonics."""
        sample = _make_sample(mock_experiment_2banks, "test", [])

        with (
            patch.object(Sample, "normalize_by_monitor", autospec=True, return_value=None),
            patch.object(Sample, "stitch_scans", autospec=True, return_value=None),
            patch.object(Sample, "rocking_curve_centering", autospec=True, return_value=None),
            patch.object(Sample, "rescale_data", autospec=True, return_value=None),
            patch.object(Sample, "stitch_orders", autospec=True) as stitch_orders,
        ):
            sample.reduce()

        stitch_orders.assert_not_called()

    @pytest.mark.parametrize(
        ("measurement_type", "stitched"),
        [(MeasurementType.SAMPLE, True), (MeasurementType.BACKGROUND, True), (MeasurementType.EMPTY_CELL, False)],
    )
    def test_reduce_does_not_stitch_empty_cell(self, mock_experiment_2banks, measurement_type, stitched):
        """With save_all_harmonics, Sample.reduce stitches samples and the background, not the empty cell.

        No output file is written for the empty cell, so its stitched profile would be discarded.
        """
        mock_experiment_2banks._config.save_all_harmonics = True
        measurement = _make_sample(mock_experiment_2banks, "test", [])
        measurement.measurement_type = measurement_type

        with (
            patch.object(Sample, "normalize_by_monitor", autospec=True, return_value=None),
            patch.object(Sample, "stitch_scans", autospec=True, return_value=None),
            patch.object(Sample, "rocking_curve_centering", autospec=True, return_value=None),
            patch.object(Sample, "rescale_data", autospec=True, return_value=None),
            patch.object(Sample, "subtract_background", autospec=True, return_value=None),
            patch.object(Sample, "stitch_orders", autospec=True) as stitch_orders,
        ):
            measurement.reduce()

        assert stitch_orders.called is stitched

    def test_writes_stitched_files(self, mock_experiment_2banks, tmp_path):
        """With save_all_harmonics, both stitched files are written and hold the stitched profiles."""
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        stitched = self._read(tmp_path / "test_det_stitched.txt")
        assert [row[0] for row in stitched] == pytest.approx(sample.data_stitched.q)
        assert [row[1] for row in stitched] == pytest.approx(sample.data_stitched.i)
        assert [row[2] for row in stitched] == pytest.approx(sample.data_stitched.e)
        subtracted = self._read(tmp_path / "test_det_background_subtracted_stitched.txt")
        assert [row[1] for row in subtracted] == pytest.approx(sample.data_bg_subtracted_stitched.i)

    def test_per_harmonic_files_unchanged_by_stitching(self, mock_experiment_2banks, tmp_path):
        """The ``_det_<n>`` files are identical whether or not the harmonics were stitched."""
        unstitched_dir = tmp_path / "unstitched"
        stitched_dir = tmp_path / "stitched"
        self._sample(mock_experiment_2banks, unstitched_dir).dump_reduced_data_to_csv()
        sample = self._sample(mock_experiment_2banks, stitched_dir)
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        per_harmonic = sorted(path.name for path in unstitched_dir.iterdir())
        assert len(per_harmonic) == 6  # unscaled, scaled, and subtracted for harmonics 1 and 2
        for name in per_harmonic:
            assert (stitched_dir / name).read_bytes() == (unstitched_dir / name).read_bytes()
        assert sorted(path.name for path in stitched_dir.iterdir()) == sorted(
            [*per_harmonic, "test_det_stitched.txt", "test_det_background_subtracted_stitched.txt"]
        )

    def test_no_stitched_files_without_save_all_harmonics(self, mock_experiment_2banks, tmp_path):
        """Without save_all_harmonics, no stitched file is written, even if profiles were stitched."""
        sample = self._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        assert not list(tmp_path.glob("*stitched*"))

    def test_no_stitched_files_when_harmonic_2_fails(self, mock_experiment_2banks, tmp_path):
        """When harmonics 1 and 2 cannot be stitched, no stitched file is written."""
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled[1] = self._profile(self.Q2 + 1.0e-2)
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        assert not list(tmp_path.glob("*stitched*"))
        assert (tmp_path / "test_det_2.txt").is_file()

    def test_stale_stitched_files_removed_when_harmonic_2_fails(
        self, mock_experiment_2banks, tmp_path, caplog, monkeypatch
    ):
        """Stitched files of an earlier reduction into the same directory are removed when stitching fails.

        Otherwise the summary report would mix them with the data of the new reduction.
        """
        stale = [tmp_path / "test_det_stitched.txt", tmp_path / "test_det_background_subtracted_stitched.txt"]
        for path in stale:
            path.write_text("stale\n")
        sample = self._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled[1] = self._profile(self.Q2 + 1.0e-2)
        sample.stitch_orders()
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert f"Removed {stale[0]}, left by an earlier reduction." in caplog.text

    def test_no_background_subtracted_stitched_file_when_nothing_subtracted(self, mock_experiment_2banks, tmp_path):
        """Without a background subtraction, only ``<name>_det_stitched.txt`` is written.

        This is the case of the background measurement itself.
        """
        sample = self._sample(mock_experiment_2banks, tmp_path, subtracted=False)
        sample.measurement_type = MeasurementType.BACKGROUND
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        assert (tmp_path / "test_det_stitched.txt").is_file()
        assert not (tmp_path / "test_det_background_subtracted_stitched.txt").exists()


class TestRemoveEarlierOutputFiles:
    """Tests for removing the output files of an earlier reduction into the same directory.

    Otherwise those files would stay next to the files of the new reduction, and the summary
    report would include them.
    """

    @staticmethod
    def _write_stale(tmp_path, filenames: list[str]) -> list:
        """Write placeholder files standing in for the output of an earlier reduction."""
        paths = [tmp_path / filename for filename in filenames]
        for path in paths:
            path.write_text("stale\n")
        return paths

    def test_higher_harmonics_and_stitched_files_removed_without_save_all_harmonics(
        self, mock_experiment_2banks, tmp_path, caplog, monkeypatch
    ):
        """Files written by an earlier reduction with save_all_harmonics are removed without it."""
        stale = self._write_stale(
            tmp_path,
            [
                "test_det_2.txt",
                "test_det_2_unscaled.txt",
                "test_det_2_background_subtracted.txt",
                "test_det_stitched.txt",
                "test_det_background_subtracted_stitched.txt",
            ],
        )
        overwritten = self._write_stale(tmp_path, ["test_det_1.txt"])[0]
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert overwritten.read_text() != "stale\n"
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "test_det_1.txt",
            "test_det_1_background_subtracted.txt",
            "test_det_1_unscaled.txt",
        ]
        assert f"Removed {stale[0]}, left by an earlier reduction." in caplog.text

    def test_background_subtracted_files_removed_when_nothing_subtracted(self, mock_experiment_2banks, tmp_path):
        """Background-subtracted files of an earlier reduction are removed when nothing is subtracted."""
        stale = self._write_stale(
            tmp_path,
            [
                "test_det_1_background_subtracted.txt",
                "test_det_2_background_subtracted.txt",
                "test_det_background_subtracted_stitched.txt",
            ],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, subtracted=False)
        sample.stitch_orders()

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert (tmp_path / "test_det_stitched.txt").is_file()

    def test_files_of_other_measurements_kept(self, mock_experiment_2banks, tmp_path):
        """Only the files of the measurement being written are removed."""
        kept = self._write_stale(
            tmp_path,
            [
                "test2_det_2.txt",
                "tes_det_2.txt",
                "test_det_5.txt",
                "test_det_2_notes.txt",
                "summary.xlsx",
            ],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv()

        assert all(path.read_text() == "stale\n" for path in kept)

    def test_only_enabled_categories_removed(self, mock_experiment_2banks, tmp_path):
        """Files of a category that is not written are neither written nor removed."""
        stale = self._write_stale(tmp_path, ["test_det_2.txt", "test_det_stitched.txt"])
        kept = self._write_stale(
            tmp_path,
            [
                "test_det_2_unscaled.txt",
                "test_det_2_background_subtracted.txt",
                "test_det_background_subtracted_stitched.txt",
            ],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv(detector_data=False, background_subtracted_data=False)

        assert not any(path.exists() for path in stale)
        assert all(path.read_text() == "stale\n" for path in kept)

    def test_files_removed_when_measurement_cannot_be_written(self, mock_experiment_2banks, tmp_path):
        """A measurement whose first harmonic is missing does not keep the files of an earlier reduction."""
        stale = self._write_stale(tmp_path, ["test_det_1.txt", "test_det_1_unscaled.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled = []

        with pytest.raises(RuntimeError, match="Cannot write scaled data"):
            sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)

    # Files with the names written by usansred up to version 1.9.0, which start with ``UN_``

    def test_legacy_files_removed(self, mock_experiment_2banks, tmp_path, caplog, monkeypatch):
        """Files of an earlier reduction by a version that wrote ``UN_`` names are removed."""
        stale = self._write_stale(
            tmp_path,
            [
                f"UN_test_det_{harmonic}{suffix}.txt"
                for harmonic in (1, 2)
                for suffix in ("", "_unscaled", "_background_subtracted")
            ]
            + ["UN_test_det_stitched.txt", "UN_test_det_background_subtracted_stitched.txt"],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        sample.stitch_orders()
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert not any(path.name.startswith("UN_") for path in tmp_path.iterdir())
        assert (tmp_path / "test_det_1.txt").is_file()
        assert f"Removed {stale[0]}, left by an earlier reduction." in caplog.text

    def test_legacy_higher_harmonics_removed_without_save_all_harmonics(self, mock_experiment_2banks, tmp_path):
        """Legacy files of every harmonic are removed, regardless of ``save_all_harmonics``."""
        stale = self._write_stale(
            tmp_path,
            [
                "UN_test_det_2.txt",
                "UN_test_det_2_unscaled.txt",
                "UN_test_det_2_background_subtracted.txt",
                "UN_test_det_stitched.txt",
                "UN_test_det_background_subtracted_stitched.txt",
            ],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "test_det_1.txt",
            "test_det_1_background_subtracted.txt",
            "test_det_1_unscaled.txt",
        ]

    def test_legacy_files_of_other_measurements_kept(self, mock_experiment_2banks, tmp_path):
        """Only the legacy files of the measurement being written are removed."""
        kept = self._write_stale(
            tmp_path,
            ["UN_test2_det_1.txt", "UN_tes_det_1.txt", "UN_test_det_5.txt", "UN_test_det_1_notes.txt"],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv()

        assert all(path.read_text() == "stale\n" for path in kept)

    def test_only_enabled_legacy_categories_removed(self, mock_experiment_2banks, tmp_path):
        """Legacy files of a category that is not written are kept."""
        stale = self._write_stale(tmp_path, ["UN_test_det_2.txt", "UN_test_det_stitched.txt"])
        kept = self._write_stale(
            tmp_path,
            [
                "UN_test_det_2_unscaled.txt",
                "UN_test_det_2_background_subtracted.txt",
                "UN_test_det_background_subtracted_stitched.txt",
            ],
        )
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv(detector_data=False, background_subtracted_data=False)

        assert not any(path.exists() for path in stale)
        assert all(path.read_text() == "stale\n" for path in kept)

    def test_legacy_files_removed_when_measurement_cannot_be_written(self, mock_experiment_2banks, tmp_path):
        """A measurement whose first harmonic is missing does not keep the legacy files either."""
        stale = self._write_stale(tmp_path, ["UN_test_det_1.txt", "UN_test_det_1_unscaled.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        sample.data_scaled = []

        with pytest.raises(RuntimeError, match="Cannot write scaled data"):
            sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)

    def test_legacy_names_kept_when_output_of_another_sample(
        self, mock_experiment_2banks, tmp_path, caplog, monkeypatch
    ):
        """Files named UN_test_* are kept when a sample named UN_test is in the experiment, since they are its output.

        ``Experiment.dump_reduced_data`` writes the samples in setup order, so ``UN_test`` is written before ``test``.
        """
        other_output = self._write_stale(tmp_path, ["UN_test_det_1.txt", "UN_test_det_1_unscaled.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        mock_experiment_2banks.samples = [_make_sample(mock_experiment_2banks, "UN_test", []), sample]
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.dump_reduced_data_to_csv()

        assert all(path.read_text() == "stale\n" for path in other_output)
        assert "Files named UN_test_det_* are not removed for sample test" in caplog.text

    def test_legacy_names_kept_when_output_of_the_background(self, mock_experiment_2banks, tmp_path):
        """Files named UN_test_* are kept when the background is named UN_test."""
        other_output = self._write_stale(tmp_path, ["UN_test_det_1.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        mock_experiment_2banks.samples = [sample]
        mock_experiment_2banks.background = _make_sample(mock_experiment_2banks, "UN_test", [])

        sample.dump_reduced_data_to_csv()

        assert other_output[0].read_text() == "stale\n"

    def test_legacy_names_removed_when_empty_cell_has_the_prefixed_name(self, mock_experiment_2banks, tmp_path):
        """An empty cell named UN_test does not keep the legacy files of test, since no files are written for it."""
        stale = self._write_stale(tmp_path, ["UN_test_det_1.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path)
        mock_experiment_2banks.samples = [sample]
        mock_experiment_2banks.empty_cell = _make_sample(mock_experiment_2banks, "UN_test", [])

        sample.dump_reduced_data_to_csv()

        assert not stale[0].exists()

    # Files written only by usansred 1.0.0 to 1.8.0: log-binned files and bank_<n>/ subdirectories

    PRE_1_9_0_FILES = [
        "UN_test_det_1_lb.txt",
        "UN_test_det_1_lbs.txt",
        "bank_2/UN_test.txt",
        "bank_2/UN_test_unscaled.txt",
    ]

    @classmethod
    def _write_stale_pre_1_9_0(cls, tmp_path) -> list:
        (tmp_path / "bank_2").mkdir()
        return cls._write_stale(tmp_path, cls.PRE_1_9_0_FILES)

    def test_files_of_versions_before_1_9_0_removed(self, mock_experiment_2banks, tmp_path, caplog, monkeypatch):
        """The log-binned files and the bank_<n>/ files of versions 1.0.0 to 1.8.0 are removed, and so is
        the bank_<n>/ subdirectory once empty."""
        stale = self._write_stale_pre_1_9_0(tmp_path)
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)
        monkeypatch.setattr(usansred.reduce.logger, "propagate", True)
        caplog.set_level(logging.INFO)

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "test_det_1.txt",
            "test_det_1_background_subtracted.txt",
            "test_det_1_unscaled.txt",
        ]
        assert f"Removed the empty directory {tmp_path / 'bank_2'}, left by an earlier reduction." in caplog.text

    def test_bank_directory_with_other_files_kept(self, mock_experiment_2banks, tmp_path):
        """A bank_<n>/ subdirectory that holds other files is kept, with those files."""
        self._write_stale_pre_1_9_0(tmp_path)
        kept = self._write_stale(tmp_path, ["bank_2/UN_test2.txt", "bank_2/notes.txt"])
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv()

        assert sorted(path.name for path in (tmp_path / "bank_2").iterdir()) == ["UN_test2.txt", "notes.txt"]
        assert all(path.read_text() == "stale\n" for path in kept)

    def test_only_enabled_categories_of_versions_before_1_9_0_removed(self, mock_experiment_2banks, tmp_path):
        """Without detector data, the monitor-normalized bank_<n>/ file is kept."""
        self._write_stale_pre_1_9_0(tmp_path)
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)

        sample.dump_reduced_data_to_csv(detector_data=False, background_subtracted_data=False)

        assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("UN_*")) == [
            "bank_2/UN_test_unscaled.txt"
        ]

    def test_files_of_versions_before_1_9_0_removed_when_output_of_another_sample(
        self, mock_experiment_2banks, tmp_path
    ):
        """A sample named UN_test keeps the 1.9.0 names of test, but not the names written only before 1.9.0,
        which are never its output."""
        stale = self._write_stale_pre_1_9_0(tmp_path)
        sample = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)
        mock_experiment_2banks.samples = [_make_sample(mock_experiment_2banks, "UN_test", []), sample]

        sample.dump_reduced_data_to_csv()

        assert not any(path.exists() for path in stale)

    @pytest.mark.parametrize("order", [("test", "UN_test"), ("UN_test", "test")])
    def test_experiment_with_measurements_x_and_UN_x(self, mock_experiment_2banks, tmp_path, order):
        """With samples test and UN_test, each writes its own files, in either setup order.

        The files named UN_test_det_* are the output of UN_test, so they are kept; the files written only by
        versions before 1.9.0 for test, and the legacy files of UN_test, are removed.
        """
        stale = self._write_stale_pre_1_9_0(tmp_path)
        stale += self._write_stale(tmp_path, ["UN_UN_test_det_1.txt", "UN_UN_test_det_2.txt"])
        replaced = self._write_stale(tmp_path, ["UN_test_det_1.txt", "test_det_1.txt"])
        samples = {}
        for name in order:
            samples[name] = TestStitchOrders._sample(mock_experiment_2banks, tmp_path, save_all_harmonics=False)
            samples[name].name = name
        mock_experiment_2banks.samples = list(samples.values())

        mock_experiment_2banks.dump_reduced_data()

        assert not any(path.exists() for path in stale)
        assert all(path.read_text() != "stale\n" for path in replaced)
        assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == sorted(
            f"{name}_det_1{suffix}.txt"
            for name in ("test", "UN_test")
            for suffix in ("", "_background_subtracted", "_unscaled")
        )
