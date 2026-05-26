import copy
import csv
import logging
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit

from usansred.io.read import is_csv, read_config
from usansred.model import IQData, MonitorData, XYData
from usansred.summary import generate_report

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


ARCSEC_TO_RADIANS = math.pi / (3600.0 * 180.0)


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
    experiment: "Experiment" = Field(..., description="Experiment this scan belongs to")
    monitor_data: MonitorData = Field(default_factory=MonitorData, description="Monitor data associated with this scan")
    detector_data: list[MonitorData] = Field(
        default_factory=list, description="Detector data associated with this scan"
    )
    load_data: bool = Field(True, description="Whether to load data files during initialization")
    # NOTE: Not currently used, but may be useful in the future
    # sample: "Sample" = Field(..., description="The sample associated with this scan")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""
        if self.load_data:
            self.load()

    @property
    def size(self) -> int:
        """Number of data points"""
        return len(self.monitor_data.iq_data.q)

    @property
    def num_of_banks(self) -> int:
        """Number of detector banks in the Experiment"""
        return self.experiment.num_of_banks

    def load(self):
        """Load experiment data files"""
        self.load_monitor_data()
        self.load_detector_data()

    def load_monitor_data(self):
        filename = f"USANS_{self.number}_monitor_scan_ARN.txt"
        filepath = os.path.join(self.experiment.folder, filename)
        xy_data = self.read_xy_file(filepath)
        iq_data = self.convert_xy_to_iq(xy_data)
        self.monitor_data = MonitorData(xy_data=xy_data, iq_data=iq_data, filepath=filepath)

    def load_detector_data(self):
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
            err = sqrt(|y - 0.5| + 0.5)
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
    thickness: float = Field(0.1, description="Sample thickness in cm")
    is_background: bool = Field(False, description="Flag to indicate if this is a background sample")
    exclude: list[int] = Field(default_factory=list, description="List of scan numbers to exclude")
    # Fields that are initialized in model_post_init and not expected from user input
    detector_data: list[IQData] = Field(default_factory=list, init=False, description="Original detector data")
    data_scaled: list[IQData] = Field(default_factory=list, init=False, description="Data scaled to thickness")
    data_log_binned: IQData = Field(default_factory=IQData, init=False, description="Log-binned data")
    data_bg_subtracted: IQData = Field(default_factory=IQData, init=False, description="Background subtracted data")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""
        for i in range(self.num_of_scans):
            if i + self.start_scan_num in self.exclude:
                continue
            scan = Scan(
                number=i + self.start_scan_num,
                experiment=self.experiment,
            )
            self.scans.append(scan)
        self.num_of_scans = len(self.scans)

        # NOTE:
        #  - detector_data: original data after being stitched with another monitor-normalized scan
        #  - data_scaled: data after being scaled to thickness
        #  - data_log_binned: data_scaled after being log-binned
        #  - data_bg_subtracted: data_log_binned after background subtraction (aliased as self.data_reduced)
        self.detector_data = []
        self.data_scaled = []
        self.data_log_binned = IQData()
        self.data_bg_subtracted = IQData()

    @property
    def data(self):
        """Main detector data, currently an alias for detector_data[0]"""
        return self.detector_data[0] if self.detector_data else None

    @property
    def size(self) -> int:
        """Number of detector data points"""
        return len(self.data.q) if self.data else 0

    @property
    def data_reduced(self):
        """Reduced data, currently an alias for bg_subtracted data"""
        return self.data_bg_subtracted

    @property
    def is_log_binned(self) -> bool:
        """Flag to indicate if the sample has been log-binned"""
        return bool(self.data_log_binned.q)

    @property
    def is_reduced(self) -> bool:
        """Flag to indicate if the sample has been reduced"""
        return bool(self.data_bg_subtracted.q)

    @property
    def size_reduced(self) -> int:
        """Number of reduced data points"""
        return len(self.data_reduced.q)

    @property
    def config(self):
        return self.experiment.config

    @property
    def num_of_banks(self) -> int:
        """Number of detector banks in the Experiment"""
        return self.experiment.num_of_banks

    @property
    def num_log_bins(self) -> int:
        """Size of the log-binned data"""
        return len(self.data_log_binned.q)

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on sample name and start number."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return other.name == self.name and other.start_scan_num == self.start_scan_num

    def dump_data_to_csv(self, filepath: str, data: IQData | XYData, title: str | None = None):
        """Dump IQ or XY data to a CSV file."""
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            logging.info(f"Output directory {output_dir} does not exist; creating it.")
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
        bg_subtracted_data: bool = True,
        log_binned_data: bool = True,
    ):
        """Dump reduced data to CSV files based on specified flags."""
        if detector_data and self.data:
            filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_1_unscaled.txt")
            self.dump_data_to_csv(filepath, self.data)
            if self.config.get("save_all_harmonics", False):
                for i in range(1, self.num_of_banks):
                    bank = i + 1  # start with the second order
                    filepath = os.path.join(
                        self.experiment.output_dir,
                        f"bank_{bank}",
                        f"UN_{self.name}_unscaled.txt",
                    )
                    self.dump_data_to_csv(filepath, self.detector_data[i])

        if scaled_data:
            filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_1.txt")
            self.dump_data_to_csv(filepath, self.data_scaled[0])
            if self.config.get("save_all_harmonics", False):
                for i in range(1, self.num_of_banks):
                    bank = i + 1  # start with the second order
                    filepath = os.path.join(
                        self.experiment.output_dir,
                        f"bank_{bank}",
                        f"UN_{self.name}.txt",
                    )
                    self.dump_data_to_csv(filepath, self.data_scaled[i])

        if bg_subtracted_data:
            filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_1_lbs.txt")
            self.dump_data_to_csv(filepath, self.data_bg_subtracted)

        if log_binned_data:
            if not self.is_log_binned:
                logging.info(f"Sample {self.name} has not been log-binned; skipping log-binned data dump.")
                return
            filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_1_lb.txt")
            self.dump_data_to_csv(filepath, self.data_log_binned)

        return

    def normalize_by_monitor(self) -> None:
        """Normalize detector intensities by monitor counts for all scans."""
        for scan in self.scans:
            scan.normalize_by_monitor()

    def reduce(self):
        """Reduce this sample's scans"""
        self.normalize_by_monitor()
        self.stitch_scans()
        self.rocking_curve_centering()
        self.rescale_data()

        # Only process first detector bank
        data_scaled = self.data_scaled[0]
        logging.info(f"Only the first bank data is used for sample {self.name}.")

        # Log-binning is optional
        if self.experiment.log_binning:
            self.data_log_binned = self.log_bin_data(data_scaled)

        if self.experiment.background and not self.is_background:
            self.subtract_background(self.experiment.background)

        logging.info(f"Data reduction finished for sample {self.name}.")
        return

    # TODO: This function should be re-written from scratch
    def log_bin_data(self, data: IQData) -> IQData:
        """Log-bin the I(Q) data."""
        assert len(data.q) == len(data.i) == len(data.e)

        # Sort by momentum transfer
        sorted_indices = np.argsort(data.q)
        q = np.array(data.q)[sorted_indices]
        i = np.array(data.i)[sorted_indices]
        e = np.array(data.e)[sorted_indices]

        iq_dict = {"I": list(i), "Q": list(q), "E": list(e)}

        # The resolution in Q, ΔQ=2πΔθ/λ_1, where Δθ is the FWHM of the resolution function at the detector
        fundamentalStep = 2 * math.pi * horizontal_rocking_width(1) * ARCSEC_TO_RADIANS / self.experiment.prim_wave

        # Step multiplier
        steps_per_decade = self.experiment.config["binning"]["steps_per_decade"]
        alpha = math.exp(math.log(10) / steps_per_decade)
        # step relative width
        kappa = 2.0 * (alpha - 1) / (alpha + 1)

        # floor ((ln((MyQ[InLength-1])/Qmin))/(ln(alpha)))
        q_min = self.experiment.config["binning"]["q_min"]
        numOfBins = math.floor(math.log(max(iq_dict["Q"]) / q_min) / math.log(alpha))

        logQ = [q_min * (alpha**n) for n in range(numOfBins)]
        logI = [None] * numOfBins
        logE = [None] * numOfBins
        logW = [1] * numOfBins

        origIdx = 0

        k2 = None
        k3 = None
        stepmin = None
        stepmax = None

        testVal = None
        for lIdx, lq in enumerate(logQ):
            testVal = kappa * lq

            if testVal <= fundamentalStep:
                while logI[lIdx] is None:
                    if origIdx < (len(iq_dict["Q"]) - 1) and iq_dict["Q"][origIdx + 1] > lq:
                        k2 = iq_dict["Q"][origIdx + 1] - iq_dict["Q"][origIdx]
                        k3 = lq - iq_dict["Q"][origIdx + 1]
                        # rtemp[outindex]=((k3/k2)+1)*MyR[inindex+1]-(k3/k2)*MyR[inindex]
                        logI[lIdx] = ((k3 / k2) + 1) * iq_dict["I"][origIdx + 1] - (k3 / k2) * iq_dict["I"][origIdx]
                        logE[lIdx] = (((k3 / k2) + 1) ** 2.0) * (iq_dict["E"][origIdx + 1] ** 2.0) + (
                            (k3 / k2) ** 2.0
                        ) * (iq_dict["E"][origIdx] ** 2.0)
                        logW[lIdx] = 1
                    else:
                        origIdx += 1
            else:
                stepmin = lq - testVal / 2.0
                stepmax = lq + testVal / 2.0
                origIdx = 1
                while origIdx < len(iq_dict["Q"]):
                    if (iq_dict["Q"][origIdx] + fundamentalStep / 2.0) >= stepmin:
                        break
                    origIdx += 1

                while origIdx < len(iq_dict["Q"]):
                    if (iq_dict["Q"][origIdx] - fundamentalStep / 2.0) <= stepmin:
                        if logI[lIdx] is None:
                            # rtemp[outindex]=MyR[Inindex]*((MyQ[inindex]+FunStep/2)-stepmin)/funstep
                            # wtemp[outindex]=(MyQ[inindex]+FunStep/2-stepmin)/funstep
                            # stemp[outindex]=(MyS[InIndex]^2)*((MyQ[inindex]+FunStep/2-stepmin)/funstep)^2
                            logI[lIdx] = (
                                iq_dict["I"][origIdx]
                                * ((iq_dict["Q"][origIdx] + fundamentalStep / 2.0) - stepmin)
                                / fundamentalStep
                            )
                            logW[lIdx] = (iq_dict["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            logE[lIdx] = (iq_dict["E"][origIdx] ** 2.0) * (
                                (iq_dict["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            ) ** 2.0
                        else:
                            # rtemp[outindex]+=MyR[Inindex]*((MyQ[inindex]+FunStep/2)-stepmin)/funstep
                            # wtemp[outindex]+=(MyQ[inindex]+FunStep/2-stepmin)/funstep
                            # stemp[outindex]+=(MyS[InIndex]^2)*((MyQ[inindex]+FunStep/2-stepmin)/funstep)^2
                            logI[lIdx] += (
                                iq_dict["I"][origIdx]
                                * ((iq_dict["Q"][origIdx] + fundamentalStep / 2.0) - stepmin)
                                / fundamentalStep
                            )
                            logW[lIdx] += (iq_dict["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            logE[lIdx] += (iq_dict["E"][origIdx] ** 2.0) * (
                                (iq_dict["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            ) ** 2.0
                    elif (iq_dict["Q"][origIdx] + fundamentalStep / 2.0) > stepmax:
                        if logI[lIdx] is None:
                            # rtemp[outindex]=MyR[Inindex]*(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # wtemp[outindex]=(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # stemp[outindex]=(MyS[InIndex]^2)*((stepmax-(MyQ[inindex]-FunStep/2))/funstep)^2
                            logI[lIdx] = (
                                iq_dict["I"][origIdx]
                                * (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
                            )
                            logW[lIdx] = (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            logE[lIdx] = (iq_dict["E"][origIdx] ** 2.0) * (
                                (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            ) ** 2.0
                        else:
                            logI[lIdx] += (
                                iq_dict["I"][origIdx]
                                * (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
                            )
                            logW[lIdx] += (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            logE[lIdx] += (iq_dict["E"][origIdx] ** 2.0) * (
                                (stepmax - (iq_dict["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            ) ** 2.0
                    else:
                        if logI[lIdx] is None:
                            logI[lIdx] = iq_dict["I"][origIdx]
                            logW[lIdx] = 1.0
                            logE[lIdx] = iq_dict["E"][origIdx] ** 2.0
                        else:
                            logI[lIdx] += iq_dict["I"][origIdx]
                            logW[lIdx] += 1.0
                            logE[lIdx] += iq_dict["E"][origIdx] ** 2.0

                    origIdx += 1
                    if origIdx < len(iq_dict["Q"]) and (iq_dict["Q"][origIdx] - fundamentalStep / 2.0) >= stepmax:
                        break

        logI = [logI[ii] / logW[ii] for ii in range(numOfBins)]
        logE = [logE[ii] / (logW[ii] ** 2.0) for ii in range(numOfBins)]
        logE = [le**0.5 for le in logE]

        return IQData(q=logQ, i=logI, e=logE)

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
        """Rescale reflected data by the analyzer's solid angle acceptance and by sample thickness."""

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
            scaling_factor = 1.0 / (analyzer_solid_angle * self.thickness)
            i_scaled = [i * scaling_factor for i in iq_data.i]
            e_scaled = [e * scaling_factor for e in iq_data.e]

            q_cleaned, i_cleaned, e_cleaned = self._combine_duplicate_q_points(q_scaled, i_scaled, e_scaled)
            iq_scaled = IQData(q=q_cleaned, i=i_cleaned, e=e_cleaned, t=[])
            self.data_scaled.append(iq_scaled)

        q_range = f"{min(self.data_scaled[0].q)} - {max(self.data_scaled[0].q)}"
        logging.info(f"Rescaled data for sample {self.name}, Q-range: {q_range} 1/angstrom\n")
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
        logging.info(f"Scans stitched together for sample {self.name}.\n")

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

        logging.info(theta_range_msg)
        logging.info(q_range_msg)
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

        logging.info(f"Centered rocking curves for sample {self.name} using offset {q_offset}.")
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

    def subtract_background(self, background: "Sample", v_scale: float = 1.0) -> None:
        """Subtract background data from this sample's data.

        Parameters
        ----------
        background : Sample
            The background sample to subtract. Must be processed (stitched, scaled, and binned).
        v_scale : float
            Scaling factor for background intensity before subtraction.
        """

        if self.experiment.log_binning:
            assert self.is_log_binned, f"Sample {self.name} must be log-binned before background subtraction."
            assert background.is_log_binned, (
                f"Background {background.name} must be log-binned before background subtraction."
            )
            logging.info(
                f"Logbinned data are used for background subtraction. Sample {self.name}, background {background.name}"
            )
            data = self.data_log_binned
            bg_data = background.data_log_binned

            scale_f = v_scale * self.thickness / background.thickness

            sample_num_of_bins = self.num_log_bins
            bg_num_of_bins = self.num_log_bins
            # TODO: This should instead be background.num_log_bins, but we need to fix log binning first
            # bg_num_of_bins = background.num_log_bins

            if sample_num_of_bins < bg_num_of_bins:
                num_of_bins = sample_num_of_bins
                momentum_transfer = data.q.copy()
            else:
                num_of_bins = bg_num_of_bins
                momentum_transfer = bg_data.q.copy()

            intensity = [data.i[i] - (scale_f * bg_data.i[i]) for i in range(num_of_bins)]
            error = [math.sqrt(data.e[i] ** 2.0 + (scale_f * bg_data.e[i]) ** 2.0) for i in range(num_of_bins)]

            self.data_bg_subtracted.q = momentum_transfer
            self.data_bg_subtracted.i = intensity
            self.data_bg_subtracted.e = error

        # Use interpolation if log-binning is not applied
        else:
            # Only process the first detector bank for now
            data = self.data_scaled[0]
            bg_data = background.data_scaled[0]

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

            self.data_bg_subtracted.q = q_data.tolist()
            self.data_bg_subtracted.i = i_subtracted.tolist()
            self.data_bg_subtracted.e = e_subtracted.tolist()

        logging.info(f"Subtracted background {background.name} from sample {self.name}")
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
    is_background : bool
        Whether this combined sample represents a background measurement.
    combined_samples : list[Sample]
        Individual Sample objects whose scans will be combined.
    combined_scans : list[Scan]
        Scans produced by the combination.
    """

    name: str = Field(..., description="Combined sample name")
    experiment: "Experiment" = Field(..., description="Experiment associated with this combined sample")
    thickness: float = Field(0.1, description="Sample thickness in cm")
    is_background: bool = Field(False, description="Whether this is a background sample")
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
                    logging.warning(
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

        logging.info(
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
    config : dict
        Configuration file loaded into a Python dictionary. Populated for both JSON and CSV config files
    output_dir : str | None
        Output folder for reduced data, default is current folder
    prim_wave : float
        Primary wavelength in Angstroms, default is 3.6
    v_angle : float
        Vertical angle, default is 0.042
    log_binning : bool
        Flag for log-binning, default is False
    """

    config_file: str = Field(..., description="Path to the configuration file")
    config: dict = Field(default_factory=dict, description="configuration file loaded into a Python dictionary")
    output_dir: str = Field("", description="Output folder for reduced data")
    prim_wave: float = Field(3.6, description="Primary wavelength in Angstroms")
    v_angle: float = Field(0.042, description="Vertical angle")
    log_binning: bool = Field(False, description="Flag for log-binning")
    num_of_banks: int = Field(default=4, init=False, description="Number of detector banks")
    folder: str = Field(default="", init=False, description="Working folder for this experiment")
    background: "Sample | None" = Field(default=None, init=False, description="Background sample")
    samples: list["Sample"] = Field(default_factory=list, init=False, description="List of samples")

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
        self.config = read_config(self.config_file)

        self.log_binning = bool(self.config["binning"]["log_binning"])

        background = self.config["background"]
        if background is None:
            logging.info("No background sample defined in the configuration file.")
            self.background = None
        else:
            self.background = Sample(**background, experiment=self)

        self.samples = [Sample(**s, experiment=self) for s in self.config["samples"]]

    def amend_log_binning(self, logbin: bool) -> None:
        """Override the log-binning setting with the command-line --logbin flag.

        For backwards compatibility when user enters a CSV file

        Parameters
        ----------
        logbin : bool
            When True, enables log-binning regardless of what the config file says.
            When False, the config-file value set during initialisation is preserved.
        """
        if logbin:
            self.log_binning = True
            self.config["binning"]["log_binning"] = True

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

        if self.background:
            try:
                self.background.reduce()
            except Exception as e:  # noqa BLE001
                logging.exception(f"Cannot reduce background {self.background.name}: {e}")

        for sample in self.samples:
            try:
                sample.reduce()
            except Exception as e:  # noqa BLE001
                logging.exception(f"Cannot reduce sample {sample.name}: {e}")

        self.dump_reduced_data()

        return

    def dump_reduced_data(self):
        """Dump reduced data to txt files"""
        for sample in self.samples:
            sample.dump_reduced_data_to_csv()

        if self.background is not None:
            self.background.dump_reduced_data_to_csv()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="USANS Data Reduction")
    parser.add_argument("path", help="Path to the configuration file")
    parser.add_argument(
        "-l",
        "--logbin",
        action="store_true",
        help="Enable log-binning of data during reduction. Option only valid for CSV files",
    )
    parser.add_argument("-o", "--output", default="", help="Output folder for reduced data (default: current folder)")
    args = parser.parse_args()
    return args


def main():
    """Main function to run USANS data reduction"""
    args = parse_args()
    experiment = Experiment(config_file=args.path, output_dir=args.output)
    if is_csv(args.path):  # backwards compatibility for CSV files, which don't have log-binning settings
        experiment.amend_log_binning(args.logbin)
    experiment.reduce()
    generate_report(config_file_path=args.path, output_dir=experiment.output_dir)

    logging.info("USANS data reduction completed.")


if __name__ == "__main__":
    main()
