# PYTHON_ARGCOMPLETE_OK
import argparse
import copy
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import argcomplete
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from scipy.optimize import curve_fit

from usansred.enums import MeasurementType
from usansred.io.read import read_config
from usansred.models import EventCounts, IQData, MonitorData, ReductionConfig, XYData
from usansred.summary import generate_report
from usansred.utils.logging import get_logger, log_to_file, set_log_config

ARCSEC_TO_RADIANS = math.pi / (3600.0 * 180.0)

# Setup root logging config
set_log_config()

logger = get_logger(__name__)


def _gaussian(x: np.ndarray, background: float, amplitude: float, sigma: float, center: float) -> np.ndarray:
    """Gaussian peak with arbitrary amplitude on a constant background."""
    return background + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2.0)


def horizontal_rocking_width(order: int) -> float:
    """
    FWHM (arcs) of the resolution function at the detector for a given reflection order

    References
    ----------
    M. Agamalian et al., "Progress on The Time-of-Flight Ultra Small Angle Neutron Scattering Instrument at SNS",
    J. Phys.: Conf. Ser. 1021 (2018) 012033.

    Parameters
    ----------
    order : int
        Positive reflection order (a.k.a. harmonic or bank)

    Returns
    -------
    float
        Computed horizontal angular resolution.
    """
    assert order > 0, "Order must be positive"
    return 5.34 * math.exp(-0.01793 * order**2) / order**2


class Scan(BaseModel):
    """A single scan (run) of a sample in an experiment

    Attributes
    ----------
    number : int
        Scan (run) number
    counts : EventCounts
        Event counts object for this scan
    experiment : Experiment
        Experiment this scan belongs to
    monitor_data : MonitorData
        Monitor data associated with this scan
    detector_data : list[MonitorData]
        Detector data associated with this scan
    load_data : bool
        Whether to load data files during initialization. Set to False when
        creating placeholder scans (e.g. for CombinedSample).
    """

    number: int = Field(..., description="Scan (run) number")
    counts: EventCounts = Field(default_factory=EventCounts, description="Event counts object for this scan")
    experiment: "Experiment" = Field(..., description="Experiment this scan belongs to")
    monitor_data: MonitorData = Field(default_factory=MonitorData, description="Monitor data associated with this scan")
    detector_data: list[MonitorData] = Field(
        default_factory=list, description="Detector data associated with this scan"
    )
    load_data: bool = Field(True, description="Whether to load data files during initialization")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""
        if self.load_data:
            self.load()
            try:
                self._get_event_counts()
            except FileNotFoundError as e:
                logger.error(f"Data files for scan {self.number} not found: {e}. Leaving event counts as zero.")

    def _get_event_counts(self):
        """Update event counts based on loaded data"""

        def _count_valid_rows(filepath: str) -> int:
            with open(filepath, "r") as file:
                reader = csv.reader(file, delimiter=",")
                count = sum(int(row[1]) for row in reader if len(row) >= 3 and not row[0].startswith("#"))
                return count

        for fn, attr in [
            (f"USANS_{self.number}_monitor.txt", "monitor"),
            (f"USANS_{self.number}_detector.txt", "detector"),
            (f"USANS_{self.number}_trans.txt", "transmission"),
        ]:
            fp = os.path.join(self.experiment.folder, fn)
            if not os.path.isfile(fp):
                logger.warning(f"Event count file {fn} not found for scan {self.number}. Setting {attr} count to 0.")
                setattr(self.counts, attr, 0)
            else:
                setattr(self.counts, attr, _count_valid_rows(fp))

    @property
    def size(self) -> int:
        """Number of data points"""
        return len(self.monitor_data.iq_data.q)

    @property
    def num_of_banks(self) -> int:
        """Number of detector banks in the Experiment"""
        return self.experiment.num_of_banks

    def load(self):
        """Load data files for this scan."""
        self._load_monitor_data()
        self._load_detector_data()

    def _load_monitor_data(self):
        filename = f"USANS_{self.number}_monitor_scan_ARN.txt"
        filepath = os.path.join(self.experiment.folder, filename)
        xy_data = self.read_xy_file(filepath)
        iq_data = self.convert_xy_to_iq(xy_data)
        self.monitor_data = MonitorData(xy_data=xy_data, iq_data=iq_data, filepath=filepath)

    def _load_detector_data(self):
        for bank in range(1, self.num_of_banks + 1):
            filename = f"USANS_{self.number}_detector_scan_ARN_peak_{bank}.txt"
            filepath = os.path.join(self.experiment.folder, filename)
            xy_data = self.read_xy_file(filepath)
            iq_data = self.convert_xy_to_iq(xy_data)
            monitor_data = MonitorData(xy_data=xy_data, iq_data=iq_data, filepath=filepath)
            self.detector_data.append(monitor_data)

    def read_xy_file(self, filepath: str) -> XYData:
        """Read XY data from a file"""
        x, y, e, t = [], [], [], []
        with open(filepath, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                if len(row) < 3 or row[0].startswith("#"):
                    continue
                x.append(float(row[0]))
                y.append(float(row[1]))
                e.append(float(row[2]))
                if len(row) == 4:
                    t.append(float(row[3]))

        return XYData(x=x, y=y, e=e, t=t)

    def convert_xy_to_iq(self, xy_data: XYData) -> IQData:
        """Convert XY data to IQ data

        Directly copies x to q, y to i, and t.
        For error, calculates based on a Poisson-like statistical model:
        ``err = sqrt(|y - 0.5| + 0.5)``
        which ensures a minimum value to avoid zero error for low counts.
        """
        iq_data = IQData(
            q=xy_data.x.copy(),
            i=xy_data.y.copy(),
            e=[math.sqrt(math.fabs(y - 0.5) + 0.5) for y in xy_data.y],
            t=xy_data.t.copy(),
        )
        return iq_data

    def normalize_by_monitor(self) -> None:
        """Normalize detector intensities by monitor counts.

        Each harmonic in the scan is normalized independently, and within each harmonic,
        the counts collected at the detector during the time the analyzer-motor remained at a particular angle
        are divided by the counts collected at the monitor during such time.
        """
        for harmonic in self.detector_data:
            intensity_normalized = []
            error_normalized = []

            for monitor_i, monitor_e, detector_i, detector_e in zip(
                self.monitor_data.iq_data.i,
                self.monitor_data.iq_data.e,
                harmonic.iq_data.i,
                harmonic.iq_data.e,
            ):
                intensity = detector_i / monitor_i
                error = np.sqrt(detector_e**2 + (intensity * monitor_e) ** 2) / monitor_i
                intensity_normalized.append(intensity)
                error_normalized.append(error)

            harmonic.iq_data.i = intensity_normalized
            harmonic.iq_data.e = error_normalized


class Sample(BaseModel):
    """Container for sample information, related scans, and data reduction methods"""

    name: str = Field(..., description="Sample name")
    experiment: "Experiment" = Field(..., description="Experiment this sample belongs to")
    start_scan_num: int = Field(..., description="Starting number for this sample")
    num_of_scans: int = Field(..., description="Number of scans for this sample")
    scans: list[Scan] = Field(default_factory=list, description="List of scans for this sample")
    thickness: float = Field(1.0, description="Sample thickness in cm")
    counts: EventCounts = Field(default_factory=EventCounts, description="Event counts object for this sample")
    measurement_type: MeasurementType = Field(
        MeasurementType.SAMPLE, description="Type of measurement (sample, background, or empty cell)"
    )
    exclude: list[int] = Field(default_factory=list, description="List of scan numbers to exclude")
    # Fields that are initialized in model_post_init and not expected from user input
    detector_data: list[IQData] = Field(default_factory=list, init=False, description="Original detector data")
    data_scaled: list[IQData] = Field(default_factory=list, init=False, description="Data scaled to thickness")
    data_bg_subtracted: list[IQData] = Field(
        default_factory=list, init=False, description="Background subtracted data, one entry per harmonic"
    )
    transmitted: float = Field(0, description="Ratio of transmitted neutrons (for transmission correction)")
    transmission: float = Field(1.0, description="Transmission coefficient (for transmission correction)")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""
        for i in range(self.num_of_scans):
            if i + self.start_scan_num in self.exclude:
                continue
            scan = Scan(
                number=i + self.start_scan_num,
                experiment=self.experiment,
            )
            self.counts.monitor += scan.counts.monitor
            self.counts.detector += scan.counts.detector
            self.counts.transmission += scan.counts.transmission
            self.scans.append(scan)
        self.num_of_scans = len(self.scans)

        self.transmitted = (
            (self.counts.detector + self.counts.transmission) / self.counts.monitor if self.counts.monitor > 0 else 0
        )
        # Calculate transmission coefficient using the empty cell's transmitted value if available
        if self.measurement_type in [MeasurementType.SAMPLE, MeasurementType.BACKGROUND]:
            empty_cell = self.experiment.empty_cell
            if empty_cell is None:
                # No empty cell configured, so transmission correction is skipped. The Experiment
                # logs this once, hence no message here that would repeat for every sample.
                self.transmission = 1.0
            else:
                try:
                    self.transmission = self.transmitted / empty_cell.transmitted
                except ZeroDivisionError:
                    logger.warning(
                        f"The {empty_cell.label} has zero transmitted counts, so the transmission "
                        f"coefficient for {self.label} cannot be computed. Setting transmission to 1.0."
                    )
                    self.transmission = 1.0
                if self.transmission <= 0 or not math.isfinite(self.transmission):
                    raise ValueError(
                        f"Invalid transmission coefficient ({self.transmission}) for {self.label}. "
                        "Check the transmitted counts for this sample and for the empty cell."
                    )

        # NOTE:
        #  - detector_data: original data after being stitched with another monitor-normalized scan
        #  - data_scaled: data after being scaled to thickness
        #  - data_bg_subtracted: data_scaled after background subtraction, one entry per harmonic,
        #    positionally aligned with data_scaled (harmonic n is entry n-1). The first harmonic
        #    is aliased as self.data_reduced.
        self.detector_data = []
        self.data_scaled = []
        self.data_bg_subtracted = []

    @property
    def data(self):
        """Main detector data, currently an alias for detector_data[0]"""
        return self.detector_data[0] if self.detector_data else None

    @property
    def size(self) -> int:
        """Number of detector data points"""
        return len(self.data.q) if self.data else 0

    @property
    def data_reduced(self) -> IQData | None:
        """Reduced data, currently an alias for the first harmonic of the bg_subtracted data

        Returns
        -------
        IQData | None
            The background-subtracted first harmonic, or ``None`` when no subtraction has
            been performed for this measurement.
        """
        return self.data_bg_subtracted[0] if self.data_bg_subtracted else None

    @property
    def is_reduced(self) -> bool:
        """Flag to indicate if the sample has been reduced"""
        return bool(self.data_reduced and self.data_reduced.q)

    @property
    def size_reduced(self) -> int:
        """Number of reduced data points in the first harmonic"""
        return len(self.data_reduced.q) if self.data_reduced else 0

    @property
    def config(self):
        return self.experiment.config

    @property
    def label(self) -> str:
        """Display label for log messages, e.g. 'sample S115_dry' or 'empty cell EmptyPCell'."""
        return f"{self.measurement_type.replace('_', ' ')} {self.name}"

    @property
    def num_of_banks(self) -> int:
        """Number of detector banks in the Experiment"""
        return self.experiment.num_of_banks

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on sample name and start number."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return other.name == self.name and other.start_scan_num == self.start_scan_num

    def dump_data_to_csv(self, filepath: str, data: IQData | XYData, title: str | None = None):
        """Dump IQ or XY data to a CSV file."""
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Output directory {output_dir} does not exist; creating it.")
            os.makedirs(output_dir)

        data_dict = data.as_dict()
        keys = list(data_dict.keys())

        # Longest list determines number of rows
        num_rows = max([len(data_dict[key]) for key in keys])

        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            if title:
                writer.writerow([title])
            for i in range(num_rows):
                row = []
                for k in keys:
                    try:
                        row.append(data_dict[k][i])
                    except IndexError:
                        row.append("")
                writer.writerow(row)
        return

    def dump_reduced_data_to_csv(
        self,
        detector_data: bool = True,
        scaled_data: bool = True,
        background_subtracted_data: bool = True,
    ) -> None:
        """Write this measurement's reduced data to CSV text files in the experiment's output directory.

        Each flag enables one category of output file (all default to True; the reduction
        workflow in ``Experiment.dump_reduced_data`` always uses the defaults). Higher
        harmonics with no data are skipped with a warning, but the first harmonic is
        required for the ``detector_data`` and ``scaled_data`` categories.

        Parameters
        ----------
        detector_data : bool
            Write the stitched, monitor-normalized data, ``UN_<name>_det_1_unscaled.txt``.
            With ``save_all_harmonics``, higher harmonics go to ``UN_<name>_det_<n>_unscaled.txt``.
        scaled_data : bool
            Write the data rescaled by analyzer solid angle, sample thickness, and transmission,
            ``UN_<name>_det_1.txt``. With ``save_all_harmonics``, higher harmonics go to
            ``UN_<name>_det_<n>.txt``.
        background_subtracted_data : bool
            Write the background- (or empty-cell-) subtracted data,
            ``UN_<name>_det_1_background_subtracted.txt``. With ``save_all_harmonics``, higher
            harmonics go to ``UN_<name>_det_<n>_background_subtracted.txt``. Only written when a
            subtraction actually occurred (``is_reduced`` is True).

        Raises
        ------
        RuntimeError
            If ``detector_data`` or ``scaled_data`` is requested but the first harmonic of
            the corresponding data is missing or empty.
        """
        # Harmonic (detector bank) n is written to ``_det_<n>``; only the first harmonic is
        # written unless ``save_all_harmonics`` is set.
        num_of_harmonics = self.num_of_banks if self.config.save_all_harmonics else 1

        def has_data(data: IQData) -> bool:
            return any((data.q, data.i, data.e, data.t))

        if detector_data and (not self.detector_data or not has_data(self.detector_data[0])):
            raise RuntimeError(f"Cannot write detector data for {self.label}: first harmonic data is missing.")

        if scaled_data and (not self.data_scaled or not has_data(self.data_scaled[0])):
            raise RuntimeError(f"Cannot write scaled data for {self.label}: first harmonic data is missing.")

        if detector_data:
            missing_harmonics = []
            for harmonic in range(1, min(num_of_harmonics, len(self.detector_data)) + 1):
                if not has_data(self.detector_data[harmonic - 1]):
                    missing_harmonics.append(harmonic)
                    continue
                filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_{harmonic}_unscaled.txt")
                self.dump_data_to_csv(filepath, self.detector_data[harmonic - 1])
            missing_harmonics.extend(range(len(self.detector_data) + 1, num_of_harmonics + 1))
            if missing_harmonics:
                logger.warning(
                    f"No detector data is available for {self.label} for harmonics {missing_harmonics}; "
                    "skipping those data dumps."
                )

        if scaled_data:
            missing_harmonics = []
            for harmonic in range(1, min(num_of_harmonics, len(self.data_scaled)) + 1):
                if not has_data(self.data_scaled[harmonic - 1]):
                    missing_harmonics.append(harmonic)
                    continue
                filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_{harmonic}.txt")
                self.dump_data_to_csv(filepath, self.data_scaled[harmonic - 1])
            missing_harmonics.extend(range(len(self.data_scaled) + 1, num_of_harmonics + 1))
            if missing_harmonics:
                logger.warning(
                    f"No scaled data is available for {self.label} for harmonics {missing_harmonics}; "
                    "skipping those data dumps."
                )

        if background_subtracted_data:
            # Only written when a background or empty cell was actually subtracted
            if self.is_reduced:
                missing_harmonics = []
                for harmonic in range(1, min(num_of_harmonics, len(self.data_bg_subtracted)) + 1):
                    if not has_data(self.data_bg_subtracted[harmonic - 1]):
                        missing_harmonics.append(harmonic)
                        continue
                    filepath = os.path.join(
                        self.experiment.output_dir, f"UN_{self.name}_det_{harmonic}_background_subtracted.txt"
                    )
                    self.dump_data_to_csv(filepath, self.data_bg_subtracted[harmonic - 1])
                missing_harmonics.extend(range(len(self.data_bg_subtracted) + 1, num_of_harmonics + 1))
                if missing_harmonics:
                    logger.warning(
                        f"No background-subtracted data is available for {self.label} for harmonics "
                        f"{missing_harmonics}; skipping those data dumps."
                    )
            else:
                logger.info(
                    f"No background or empty cell was subtracted from the {self.label}; skipping that data dump."
                )

        return

    def normalize_by_monitor(self) -> None:
        """Normalize detector intensities by monitor counts for all scans."""
        for scan in self.scans:
            scan.normalize_by_monitor()

    def reduce(self) -> None:
        """Reduce this measurement's scans.

        Normalize each scan by its monitor counts, stitch the scans into one rocking curve per
        harmonic, center the curves on the analyzer motor angle of the unscattered beam, and
        rescale to momentum transfer in ``1/angstrom``. For samples, the background (or, in its
        absence, the empty cell) is then subtracted from every harmonic.
        """
        logger.info(f"Starting reduction for {self.label} with {len(self.scans)} scans.")
        logger.info(f"Transmission coefficient for {self.label}: {self.transmission:.4f}")

        self.normalize_by_monitor()
        self.stitch_scans()
        self.rocking_curve_centering()
        self.rescale_data()

        # The background takes precedence over the empty cell: subtracting the empty cell from
        # both sample and background would cancel out, since
        # (sample - empty_cell) - (background - empty_cell) == sample - background
        if self.measurement_type == MeasurementType.SAMPLE:
            if self.experiment.background:
                self.subtract_background(self.experiment.background)
            elif self.experiment.empty_cell:
                self.subtract_background(self.experiment.empty_cell)

        logger.info(f"Data reduction finished for {self.label}.")
        return

    @staticmethod
    def _combine_duplicate_q_points(
        q_scaled: list[float], i_scaled: list[float], e_scaled: list[float]
    ) -> tuple[list[float], list[float], list[float]]:
        """Sort by Q and average duplicate momentum transfer points.

        Duplicate Q values happen when negative and positive analyzer-motor angles
        have the same magnitude after conversion to momentum transfer in 1/angstrom.
        Intensities are averaged, and uncertainties are propagated for the averaged
        values.
        """
        # Dictionary to store sums for averaging I and propagating errors for E.
        # One dictionary entry per unique Q value, which is itself a dictionary.
        sum_dict = defaultdict(lambda: {"I_sum": 0, "I_count": 0, "E_sum_squares": 0})

        for q, i, e in zip(q_scaled, i_scaled, e_scaled):
            sum_dict[q]["I_sum"] += i
            sum_dict[q]["I_count"] += 1
            sum_dict[q]["E_sum_squares"] += e**2

        q_cleaned = []
        i_cleaned = []
        e_cleaned = []

        for q, values in sorted(sum_dict.items()):
            q_cleaned.append(q)
            i_cleaned.append(values["I_sum"] / values["I_count"])
            e_cleaned.append(math.sqrt(values["E_sum_squares"]) / values["I_count"])

        return q_cleaned, i_cleaned, e_cleaned

    def rescale_data(self) -> None:
        """Rescale reflected data by the analyzer's solid angle acceptance, by sample thickness,
        and by the transmission coefficient."""

        assert self.size > 0, "No data points to rescale. Please check if the scans have been stitched correctly."

        self.data_scaled = []

        for harmonic in range(1, 1 + self.num_of_banks):
            # angle-to-Q conversion factor: radians_per_arcsecond * (2π / λ_n)
            theta_to_q = ARCSEC_TO_RADIANS * (2 * math.pi / (self.experiment.prim_wave / harmonic))

            # analyzer solid angle acceptance ΔΩ = vertical angular width * horizontal angular width
            analyzer_solid_angle = self.experiment.v_angle * (horizontal_rocking_width(harmonic) * ARCSEC_TO_RADIANS)

            # negative theta angles do correspond to positive values of the momentum transfer, hence abs()
            iq_data = self.detector_data[harmonic - 1]
            q_scaled = [abs(theta) * theta_to_q for theta in iq_data.q]
            scaling_factor = 1.0 / (analyzer_solid_angle * self.thickness * self.transmission)
            i_scaled = [i * scaling_factor for i in iq_data.i]
            e_scaled = [e * scaling_factor for e in iq_data.e]

            q_cleaned, i_cleaned, e_cleaned = self._combine_duplicate_q_points(q_scaled, i_scaled, e_scaled)
            iq_scaled = IQData(q=q_cleaned, i=i_cleaned, e=e_cleaned, t=[])
            self.data_scaled.append(iq_scaled)

        q_range = f"{min(self.data_scaled[0].q)} - {max(self.data_scaled[0].q)}"
        logger.info(f"Rescaled data for {self.label}, Q-range: {q_range} 1/angstrom")
        return

    def stitch_scans(self):
        """Stitch scan data from each detector bank into per-bank intensity curves.

        For each detector bank (harmonic), combine all scans in ``self.scans`` onto a single
        Intensity-versus-Q profile. Detector intensities and errors are expected to
        already be normalized by ``Scan.normalize_by_monitor`` before being added
        to the stitched output.

        Notice that at this stage, the "Q" values are actually analyzer-motor angles,
        that is, detector bank ``iq_data.q`` values are analyzer-motor angles (in arcsec units).

        If two or more scans contain the same "Q" value, their intensities are combined
        into one point using inverse-variance weighting, and the combined uncertainty
        is stored as the square root of the inverse summed weights.

        After all scans for a bank are processed, the stitched points are sorted by Q

        The method also generates log messages for the raw scan theta ranges
        and converted Q ranges in ``1/angstrom`` for the first detector bank.
        Results are stored on ``self.detector_data``; no value is returned.
        """
        # Build one stitched Q/I/E curve for each detector bank(harmonic).
        for bank in range(self.num_of_banks):
            momentum_transfer = []
            intensity = []
            error = []
            # transmission = []  # omitted for now

            for scan in self.scans:
                scan_data = zip(
                    scan.detector_data[bank].iq_data.q,
                    scan.detector_data[bank].iq_data.i,
                    scan.detector_data[bank].iq_data.e,
                )

                for scan_q, detector_intensity, detector_error in scan_data:
                    if scan_q in momentum_transfer:
                        # Merge repeated Q values from multiple scans using inverse-variance weights.
                        matched_q_index = momentum_transfer.index(scan_q)
                        matched_weight = 1.0 / error[matched_q_index] ** 2
                        detector_weight = 1.0 / detector_error**2
                        combined_weight = matched_weight + detector_weight
                        intensity[matched_q_index] = (
                            intensity[matched_q_index] * matched_weight + detector_intensity * detector_weight
                        ) / combined_weight
                        error[matched_q_index] = (1.0 / combined_weight) ** 0.5
                    else:
                        # Add Q values that have not appeared in earlier scans.
                        momentum_transfer.append(scan_q)
                        intensity.append(detector_intensity)
                        error.append(detector_error)

            # Sort by momentum transfer (outside the scan loop, inside the bank loop)
            sorted_indices = np.argsort(momentum_transfer)
            momentum_transfer = np.array(momentum_transfer)[sorted_indices]
            intensity = np.array(intensity)[sorted_indices]
            error = np.array(error)[sorted_indices]

            # Store the stitched curve for this detector bank.
            self.detector_data.append(
                IQData(
                    q=momentum_transfer.tolist(),
                    i=intensity.tolist(),
                    e=error.tolist(),
                )
            )
        logger.info(f"Scans stitched together for {self.label}.")

        theta_to_q = 2 * (math.pi**2.0) * 1.0 / (self.experiment.prim_wave * 3600.0 * 180.0)
        theta_range_msg = ""
        q_range_msg = ""

        # Log raw theta ranges and converted Q ranges for each scan. Remember at this stage in the reduction,
        # scan.detector_data[0].iq_data.q stores analyzer-motor angles, not yet converted to Q values
        for scan in self.scans:
            theta_range = f"{min(scan.detector_data[0].iq_data.q)} - {max(scan.detector_data[0].iq_data.q)}"
            theta_range_msg += f"Theta range for scan {scan.number}: {theta_range}\n"
            temp_q = [math.fabs(theta * theta_to_q) for theta in scan.detector_data[0].iq_data.q]
            q_range_msg += f"Q range for scan {scan.number}: {min(temp_q)} - {max(temp_q)} 1/angstrom\n"

        logger.info(theta_range_msg)
        logger.info(q_range_msg)
        return

    def rocking_curve_centering(self) -> float:
        """Center the stitched rocking curves by fitting a Gaussian peak to the rocking curve of the first harmonic.

        Notice that the ``q`` values of the stitched rocking curves are analyzer motor angles at this
        stage of reduction, not yet converted to Q values.
        The first harmonic is fit to a Gaussian over the  mostly symmetric angle range ``[q_min, -q_min]``,
        where ``q_min`` is the minimum analyzier motor angle. It will be a negative value.
        The center of the fitted Gaussian represents the value of the analyzer motor angle at which
        the analyzer reflects neutrons that have not been scattered by the sample. It should be very close to zero.

        Returns
        -------
        float
            Fitted first-harmonic motor-angle center.
        """
        assert self.detector_data, "Detector data must be stitched before centering."

        first_harmonic_rocking_curve = self.detector_data[0]
        if not first_harmonic_rocking_curve.q:
            raise ValueError("Cannot center rocking curve because first-harmonic curve is empty.")

        q = np.array(first_harmonic_rocking_curve.q)
        intensity = np.array(first_harmonic_rocking_curve.i)
        error = np.array(first_harmonic_rocking_curve.e)

        q_min = float(np.min(q))
        if q_min >= 0:
            raise ValueError("Can't center rocking curve because angles don't include negative values.")

        fit_mask = (q >= q_min) & (q <= -q_min)
        q_fit = q[fit_mask]
        intensity_fit = intensity[fit_mask]
        error_fit = error[fit_mask] if len(error) == len(q) else None

        if len(q_fit) < 3:
            raise ValueError("Can't center rocking curve because fewer than three points are in the symmetric range.")

        # Initial guess for Gaussian parameters: background, amplitude, sigma, center
        initial_guess = {
            "background": float(np.min(intensity_fit)),
            "amplitude": float(np.max(intensity_fit) - np.min(intensity_fit)),
            "sigma": float(max(np.std(q_fit), np.finfo(float).eps)),
            "center": float(q_fit[np.argmax(intensity_fit)]),
        }

        best_vals, _sigma = curve_fit(
            _gaussian,
            q_fit,
            intensity_fit,
            p0=list(initial_guess.values()),
            sigma=error_fit,
            maxfev=100000,
        )
        q_offset = float(best_vals[3])  # the center of the fitted Gaussian

        for rocking_curve in self.detector_data:
            rocking_curve.q = [float(harmonic_q - q_offset) for harmonic_q in rocking_curve.q]

        logger.info(f"Centered rocking curves for {self.label} using offset {q_offset}.")
        return q_offset

    def _match_or_interpolate(
        self,
        q_data: np.ndarray,
        q_bg: np.ndarray,
        i_bg: np.ndarray,
        e_bg: np.ndarray,
        tolerance: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match q_bg values to q_data directly if close enough, otherwise interpolate.

        Used for background subtraction"""
        i_bg_matched = np.zeros_like(q_data)
        e_bg_matched = np.zeros_like(q_data)

        for i, q in enumerate(q_data):
            # Find index in q_bg that is closest to q
            idx = np.abs(q_bg - q).argmin()
            if np.abs(q_bg[idx] - q) <= max(tolerance * q, 1e-6):
                # If within tolerance, take the value directly
                i_bg_matched[i] = i_bg[idx]
                e_bg_matched[i] = e_bg[idx]
            else:
                # Otherwise, interpolate
                i_bg_matched[i] = np.interp(q, q_bg, i_bg)
                e_bg_matched[i] = np.interp(q, q_bg, e_bg)

        return i_bg_matched, e_bg_matched

    def _subtract_harmonic(self, data: IQData, bg_data: IQData) -> IQData:
        """Subtract one background harmonic from the matching sample harmonic.

        The background intensities are matched to the sample's momentum-transfer grid by
        interpolation (see ``_match_or_interpolate``) before subtraction. Uncertainties are
        propagated in quadrature, treating the sample and background measurements as
        independent.

        Parameters
        ----------
        data : IQData
            Scaled sample data for one harmonic.
        bg_data : IQData
            Scaled background (or empty-cell) data for the same harmonic.

        Returns
        -------
        IQData
            The subtracted curve on the sample's momentum-transfer grid, in ``1/angstrom``.
        """
        # Convert to numpy arrays for easier manipulation
        q_data = np.array(data.q)
        i_data = np.array(data.i)
        e_data = np.array(data.e)

        q_bg = np.array(bg_data.q)
        i_bg = np.array(bg_data.i)
        e_bg = np.array(bg_data.e)

        # Match/interpolate background data to sample q values
        i_bg_matched, e_bg_matched = self._match_or_interpolate(q_data, q_bg, i_bg, e_bg)

        # Subtract background
        i_subtracted = i_data - i_bg_matched
        e_subtracted = np.sqrt(e_data**2 + e_bg_matched**2)

        return IQData(q=q_data.tolist(), i=i_subtracted.tolist(), e=e_subtracted.tolist(), t=[])

    def subtract_background(self, background: "Sample") -> None:
        """Subtract background (or empty-cell) data from this sample's data, harmonic by harmonic.

        Harmonic ``n`` of the background is subtracted from harmonic ``n`` of the sample, and
        never from a different harmonic: ``rescale_data`` applies the order-dependent
        angle-to-Q factor ``2 * pi * n / wavelength`` to sample and background alike, and
        ``rocking_curve_centering`` applies a single fitted motor-angle offset to every
        harmonic, so only same-order curves share a comparable momentum-transfer axis.

        Results are stored in ``self.data_bg_subtracted``, positionally aligned with
        ``self.data_scaled``: harmonic ``n`` is entry ``n - 1``. A harmonic that cannot be
        subtracted gets an empty ``IQData`` placeholder so the indices never shift.

        Parameters
        ----------
        background : Sample
            The background or empty-cell sample to subtract.
            Must be processed (stitched and scaled).

        Raises
        ------
        RuntimeError
            If the first harmonic is missing or empty for either this sample or the background.
        """

        def has_data(harmonic: int, data_scaled: list[IQData]) -> bool:
            return harmonic <= len(data_scaled) and bool(data_scaled[harmonic - 1].q)

        if not has_data(1, self.data_scaled):
            raise RuntimeError(
                f"Cannot subtract {background.label} from {self.label}: "
                f"the first-harmonic scaled data of {self.label} is missing."
            )
        if not has_data(1, background.data_scaled):
            raise RuntimeError(
                f"Cannot subtract {background.label} from {self.label}: "
                f"the first-harmonic scaled data of {background.label} is missing."
            )

        self.data_bg_subtracted = []
        subtracted_harmonics, skipped_harmonics = [], []

        for harmonic in range(1, self.num_of_banks + 1):
            if not (has_data(harmonic, self.data_scaled) and has_data(harmonic, background.data_scaled)):
                # Placeholder keeps harmonic n at entry n - 1
                self.data_bg_subtracted.append(IQData())
                skipped_harmonics.append(harmonic)
                continue
            self.data_bg_subtracted.append(
                self._subtract_harmonic(self.data_scaled[harmonic - 1], background.data_scaled[harmonic - 1])
            )
            subtracted_harmonics.append(harmonic)

        if skipped_harmonics:
            logger.warning(
                f"No scaled data for {self.label} or {background.label} for harmonics {skipped_harmonics}; "
                "skipping those subtractions."
            )
        logger.info(f"Subtracted {background.label} from {self.label} for harmonics {subtracted_harmonics}")
        return


class CombinedSample(BaseModel):
    """Combine multiple Sample measurements at the raw (X,Y,E) scan level before converting to (Q,I,E).

    Attributes
    ----------
    name : str
        Combined sample name.
    experiment : Experiment
        Experiment this combined sample belongs to.
    thickness : float
        Sample thickness in cm.
    measurement_type : MeasurementType
        Type of measurement (sample, background, or empty cell).
    combined_samples : list[Sample]
        Individual Sample objects whose scans will be combined.
    combined_scans : list[Scan]
        Scans produced by the combination.
    """

    name: str = Field(..., description="Combined sample name")
    experiment: "Experiment" = Field(..., description="Experiment associated with this combined sample")
    thickness: float = Field(0.1, description="Sample thickness in cm")
    measurement_type: MeasurementType = Field(
        MeasurementType.SAMPLE, description="Type of measurement (sample, background, or empty cell)"
    )
    combined_samples: list[Sample] = Field(default_factory=list, description="Individual samples to combine")
    combined_scans: list[Scan] = Field(default_factory=list, description="Combined scans (populated by combine)")

    def combine(self) -> None:
        """Sum raw XY data from all combined samples scan-by-scan, then generate IQ data.

        For each scan index, the monitor and detector XY data from every sample are
        accumulated.  If a sample has fewer scans than others a warning is logged and
        it is skipped for that index.  After accumulation the XY to IQ conversion is
        performed on the combined data and ready for reduction.

        Raises
        ------
        AssertionError
            If ``combined_samples`` is empty or none of them contain scans.
        """
        assert len(self.combined_samples) > 0, "No samples to combine."

        # Reset combined scans in case this method is called multiple times
        self.combined_scans: list[Scan] = []

        max_scans = max((len(sample.scans) for sample in self.combined_samples), default=0)
        assert max_scans > 0, "No scans in any sample to combine."

        for scan_idx in range(max_scans):
            for sample in self.combined_samples:
                if scan_idx >= len(sample.scans):
                    logger.warning(
                        f"Sample '{sample.name}' contains fewer scans than others "
                        f"(has {len(sample.scans)}, expected at least {scan_idx + 1}). Skipping."
                    )
                    continue

                source_scan = sample.scans[scan_idx]

                if scan_idx >= len(self.combined_scans):
                    # First contribution for this scan index – create a new placeholder scan
                    new_scan = Scan(number=0, experiment=self.experiment, load_data=False)

                    # Seed monitor data from first contributor
                    new_scan.monitor_data = MonitorData(
                        xy_data=copy.deepcopy(source_scan.monitor_data.xy_data),
                        iq_data=IQData(),
                    )

                    # Seed detector data from first contributor
                    for bank_id in range(self.experiment.num_of_banks):
                        new_scan.detector_data.append(
                            MonitorData(
                                xy_data=copy.deepcopy(source_scan.detector_data[bank_id].xy_data),
                                iq_data=IQData(),
                            )
                        )
                    self.combined_scans.append(new_scan)
                else:
                    # Subsequent contributions – accumulate into existing scan
                    self.combined_scans[scan_idx].monitor_data.xy_data = self._combine_xy_data_pair(
                        self.combined_scans[scan_idx].monitor_data.xy_data,
                        source_scan.monitor_data.xy_data,
                    )
                    for bank_id in range(self.experiment.num_of_banks):
                        self.combined_scans[scan_idx].detector_data[bank_id].xy_data = self._combine_xy_data_pair(
                            self.combined_scans[scan_idx].detector_data[bank_id].xy_data,
                            source_scan.detector_data[bank_id].xy_data,
                        )

            # After all samples have contributed, convert XY → IQ for this scan
            scan = self.combined_scans[scan_idx]
            scan.monitor_data.iq_data = scan.convert_xy_to_iq(scan.monitor_data.xy_data)
            for bank_id in range(self.experiment.num_of_banks):
                scan.detector_data[bank_id].iq_data = scan.convert_xy_to_iq(
                    scan.detector_data[bank_id].xy_data,
                )

        logger.info(
            f"Combined {len(self.combined_samples)} samples into '{self.name}' ({len(self.combined_scans)} scans)."
        )

    # ------------------------------------------------------------------
    # Static helper – combines two XYData objects by binning close X values
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_xy_data_pair(base: XYData, other: XYData, tolerance: float = 1e-8) -> XYData:
        """Combine two :class:`XYData` objects by summing Y values at matching X bins.

        X values are discretised to integer bins of width ``tolerance`` so that
        floating-point rounding does not prevent matching.  Y values are summed,
        errors are propagated in quadrature, and T values are averaged.

        Parameters
        ----------
        base : XYData
            Accumulated data so far.
        other : XYData
            New data to add.
        tolerance : float
            Bin width used to discretise X values (default ``1e-8``).

        Returns
        -------
        XYData
            Merged result.
        """
        combined: dict[int, dict] = defaultdict(lambda: {"y_sum": 0.0, "e_sq_sum": 0.0, "t_list": [], "count": 0})

        for xy_data in (base, other):
            x_arr = np.array(xy_data.x)
            y_arr = np.array(xy_data.y)
            e_arr = np.array(xy_data.e)
            t_vals = xy_data.t if xy_data.t and len(xy_data.t) == len(xy_data.x) else [0.0] * len(xy_data.x)

            for x, y, e, t in zip(x_arr, y_arr, e_arr, t_vals):
                x_key = int(np.round(x / tolerance))
                combined[x_key]["y_sum"] += y
                combined[x_key]["e_sq_sum"] += e**2
                combined[x_key]["t_list"].append(t)
                combined[x_key]["count"] += 1

        out_x: list[float] = []
        out_y: list[float] = []
        out_e: list[float] = []
        out_t: list[float] = []

        for x_key in sorted(combined.keys()):
            entry = combined[x_key]
            out_x.append(x_key * tolerance)
            out_y.append(entry["y_sum"])
            out_e.append(np.sqrt(entry["e_sq_sum"]))
            out_t.append(float(np.mean(entry["t_list"])) if entry["t_list"] else 0.0)

        return XYData(x=out_x, y=out_y, e=out_e, t=out_t)


class Experiment(BaseModel):
    """Experiment configuration for USANS data reduction

    Attributes
    ----------
    config_file : str
        Path to the configuration file
    _config : ReductionConfig
        Validated reduction configuration, always set after construction (private attribute)
    output_dir : str | None
        Output folder for reduced data, default is current folder
    prim_wave : float
        Primary wavelength in Angstroms, default is 3.6
    v_angle : float
        Vertical angle, default is 0.042
    num_of_banks : int
        Number of detector banks, default is 4 (not expected to change)
    folder : str
        Working folder for this experiment, derived from config file path (private attribute)
    samples : list[Sample]
        List of samples, populated from config file (private attribute)
    background : Sample | None
        Background sample, populated from config file if specified (private attribute)
    empty_cell : Sample | None
        Empty cell sample, populated from config file if specified (private attribute)
    """

    config_file: str = Field(..., description="Path to the configuration file")
    _config: ReductionConfig | None = PrivateAttr(default=None)
    output_dir: str = Field("", description="Output folder for reduced data")
    prim_wave: float = Field(3.6, description="Primary wavelength in Angstroms")
    v_angle: float = Field(0.042, description="Vertical angle")
    num_of_banks: int = Field(default=4, init=False, description="Number of detector banks")
    folder: str = Field(default="", init=False, description="Working folder for this experiment")
    samples: list["Sample"] = Field(default_factory=list, init=False, description="List of samples")
    background: "Sample | None" = Field(default=None, init=False, description="Background sample")
    empty_cell: "Sample | None" = Field(default=None, init=False, description="Empty cell sample")

    @property
    def config(self) -> "ReductionConfig":
        """Validated reduction configuration, always set after construction."""
        if self._config is None:
            raise RuntimeError("Experiment.config accessed before model_post_init completed")
        return self._config

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""

        # The working folder for this experiment, default is current folder
        _setupfile = os.path.abspath(self.config_file)
        self.folder = os.path.dirname(_setupfile)

        if bool(self.output_dir) is False:  # in case `output_dir` is an empty string
            self.output_dir = os.path.join(self.folder, "reduced")

        self.num_of_banks: int = 4

        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"The file path: {self.config_file} does not exist")

        self.folder = os.path.dirname(self.config_file)
        self._config = read_config(self.config_file)

        ec = self.config.empty_cell
        if ec is not None:
            self.empty_cell = Sample(
                **ec.model_dump(),
                thickness=1.0,  # ignore any user-provided thickness, used for transmission correction only
                experiment=self,
                measurement_type=MeasurementType.EMPTY_CELL,
            )
        else:
            logger.info(
                "No empty cell in the setup file, so transmission correction is skipped "
                "(transmission = 1.0 for all samples and for the background)."
            )

        background = self.config.background
        if background is not None:
            self.background = Sample(
                **background.model_dump(), experiment=self, measurement_type=MeasurementType.BACKGROUND
            )

        self.samples = [Sample(**s.model_dump(), experiment=self) for s in self.config.samples]

    def reduce(self, output_dir: str | None = None):
        """Reduce the USANS data

        Parameters
        ----------
        output_dir: str | None
            The result will be dumped to the output folder. If none will just use current folder
        """
        if output_dir is not None:
            self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # The empty cell is reduced first: its reduced curve is subtracted from each sample
        # when no background is present. When a background is present the empty-cell reduction
        # is skipped because (sample - empty_cell) - (background - empty_cell) == sample - background.
        # The empty cell is still used for the transmission coefficients, which are computed
        # from raw event counts at construction time, independent of reduction.
        if self.empty_cell:
            if self.background:
                logger.info(
                    f"Skipping reduction of {self.empty_cell.label}: a background is present. "
                    "The empty cell is still used to compute transmission coefficients."
                )
            else:  # reduce the empty cell if no background is present
                log_fn = Path(self.output_dir) / f"reduction_{self.empty_cell.name}.log"
                with log_to_file(logger, log_fn):
                    try:
                        self.empty_cell.reduce()
                    except Exception as e:  # noqa BLE001
                        logger.exception(f"Cannot reduce empty cell {self.empty_cell.name}: {e}")
                        raise RuntimeError(
                            f"Aborting reduction: empty cell {self.empty_cell.name} failed to reduce "
                            "and no background is available for subtraction."
                        ) from e

        if self.background:
            log_fn = Path(self.output_dir) / f"reduction_{self.background.name}.log"
            with log_to_file(logger, log_fn):
                try:
                    self.background.reduce()
                except Exception as e:  # noqa BLE001
                    logger.exception(f"Cannot reduce background {self.background.name}: {e}")
                    raise RuntimeError(
                        f"Aborting reduction: background {self.background.name} failed to reduce."
                    ) from e

        for sample in self.samples:
            log_fn = Path(self.output_dir) / f"reduction_{sample.name}.log"
            with log_to_file(logger, log_fn):
                try:
                    sample.reduce()
                except Exception as e:  # noqa BLE001
                    logger.exception(f"Cannot reduce sample {sample.name}: {e}")

        self.dump_reduced_data()

        return

    def dump_reduced_data(self):
        """Dump reduced data to txt files.

        Output files are written for samples and the background, never for the empty cell.
        """
        for sample in self.samples:
            sample.dump_reduced_data_to_csv()

        if self.background is not None:
            self.background.dump_reduced_data_to_csv()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for USANS data reduction."""
    parser = argparse.ArgumentParser(description="USANS Data Reduction")
    parser.add_argument("path", help="Path to the configuration file")
    # Deprecated: log binning has been removed from the reduction workflow. The flag is kept
    # (hidden from --help) so old invocations do not error; passing it only logs a warning.
    parser.add_argument(
        "-l",
        "--logbin",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("-o", "--output", default="", help="Output folder for reduced data (default: current folder)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for USANS data reduction.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments to parse, excluding the program name. When
        ``None``, arguments are read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the setup-file path, output directory, and deprecated
        ``logbin`` compatibility flag.
    """
    parser = _build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    return args


def main():
    """Main function to run USANS data reduction"""
    args = parse_args()
    if args.logbin:
        logger.warning(
            "The --logbin option is deprecated and ignored: log binning has been removed "
            "from the reduction workflow. Plot I(Q) with a logarithmic X axis instead."
        )
    experiment = Experiment(config_file=args.path, output_dir=args.output)
    experiment.reduce()
    generate_report(config_file_path=args.path, output_dir=experiment.output_dir)

    logger.info("USANS data reduction completed.")


if __name__ == "__main__":
    main()
