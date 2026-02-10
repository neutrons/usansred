import csv
import logging
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit, differential_evolution

from usansred.io.read import read_config
from usansred.model import IQData, MonitorData, XYData
from usansred.summary import generate_report

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

DARWIN_WIDTH = 5.1  # Darwin width, shouldn't really ever change unless instrument is disassembled


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
    """

    number: int = Field(..., description="Scan (run) number")
    experiment: "Experiment" = Field(..., description="Experiment this scan belongs to")
    monitor_data: MonitorData = Field(default_factory=MonitorData, description="Monitor data associated with this scan")
    detector_data: list[MonitorData] = Field(
        default_factory=list, description="Detector data associated with this scan"
    )
    # NOTE: Not currently used, but may be useful in the future
    # sample: "Sample" = Field(..., description="The sample associated with this scan")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""
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
        """Convert XY data to IQ data"""
        iq_data = IQData(
            q=xy_data.x.copy(),
            i=xy_data.y.copy(),
            e=[math.sqrt(math.fabs(y - 0.5) + 0.5) for y in xy_data.y],
            t=xy_data.t.copy(),
        )
        return iq_data


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
        for i in range(self.num_of_scans + len(self.exclude)):
            if i + self.start_scan_num in self.exclude:
                continue
            scan = Scan(
                number=i + self.start_scan_num,
                experiment=self.experiment,
            )
            self.scans.append(scan)
        self.num_of_scans = len(self.scans)

        # NOTE:
        #  - detector_data: original data after being stitched with another scan
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
    def num_of_banks(self) -> int:
        """Number of detector banks in the Experiment"""
        return self.experiment.num_of_banks

    @property
    def num_log_bins(self) -> int:
        """Size of the log-binned data"""
        return len(self.data_log_binned.q)

    def __eq__(self, other: "Sample") -> bool:
        """Equality comparison based on sample name and start number."""
        if other.name == self.name and other.start_scan_num == self.start_scan_num:
            return True
        return False

    def dump_data_to_csv(self, filepath: str, data: IQData | XYData, title: str | None = None):
        """Dump IQ or XY data to a CSV file."""
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

        if scaled_data:
            filepath = os.path.join(self.experiment.output_dir, f"UN_{self.name}_det_1.txt")
            self.dump_data_to_csv(filepath, self.data_scaled[0])

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

    def reduce(self):
        """Reduce this sample's scans"""
        self.stitch_data()
        self.rescale_data()

        # Only process first detector bank
        data_scaled = self.data_scaled[0]
        logging.info(f"Only the first bank data is used for sample {self.name}.")

        # Log-binning is optional
        if self.experiment.logbin:
            self.data_log_binned.q, self.data_log_binned.i, self.data_log_binned.e = self.log_bin_data(
                data_scaled.q, data_scaled.i, data_scaled.e
            )

        if not self.is_background:
            self.subtract_background(self.experiment.background)

        logging.info(f"Data reduction finished for sample {self.name}.")
        return

    # TODO: This function should be re-written from scratch
    def log_bin_data(
        self, momentum_transfer: list[float], intensity: list[float], energy: list[float]
    ) -> tuple[list[float], list[float], list[float]]:
        """Log-bin the I(Q) data."""
        assert len(momentum_transfer) == len(intensity) == len(energy)

        # Sort by momentum transfer
        sorted_indices = np.argsort(momentum_transfer)
        momentum_transfer = np.array(momentum_transfer)[sorted_indices]
        intensity = np.array(intensity)[sorted_indices]
        energy = np.array(energy)[sorted_indices]

        data = {"I": list(intensity), "Q": list(momentum_transfer), "E": list(energy)}

        # The fundamental Q width of the measurement
        harNo = 1.0  # Only the first harmonic peak is used
        fundamentalStep = (
            2 * math.pi**2 * self.experiment.darwin_width * harNo / (self.experiment.prim_wave * 3600.0 * 180.0)
        )

        # Step multiplier
        alpha = math.exp(math.log(10) / self.experiment.step_per_dec)
        # step relative width
        kappa = 2.0 * (alpha - 1) / (alpha + 1)

        # floor ((ln((MyQ[InLength-1])/Qmin))/(ln(alpha)))
        numOfBins = math.floor(math.log(max(data["Q"]) / self.experiment.min_q) / math.log(alpha))

        logQ = [self.experiment.min_q * (alpha**i) for i in range(numOfBins)]
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
                    if origIdx < (len(data["Q"]) - 1) and data["Q"][origIdx + 1] > lq:
                        k2 = data["Q"][origIdx + 1] - data["Q"][origIdx]
                        k3 = lq - data["Q"][origIdx + 1]
                        # rtemp[outindex]=((k3/k2)+1)*MyR[inindex+1]-(k3/k2)*MyR[inindex]
                        logI[lIdx] = ((k3 / k2) + 1) * data["I"][origIdx + 1] - (k3 / k2) * data["I"][origIdx]
                        logE[lIdx] = (((k3 / k2) + 1) ** 2.0) * (data["E"][origIdx + 1] ** 2.0) + ((k3 / k2) ** 2.0) * (
                            data["E"][origIdx] ** 2.0
                        )
                        logW[lIdx] = 1
                    else:
                        origIdx += 1
            else:
                stepmin = lq - testVal / 2.0
                stepmax = lq + testVal / 2.0
                origIdx = 0
                while origIdx < len(data["Q"]):
                    origIdx += 1
                    # emulate the do-while loop
                    if not ((data["Q"][origIdx] + fundamentalStep / 2.0) < stepmin):
                        break

                while origIdx < len(data["Q"]):
                    if (data["Q"][origIdx] - fundamentalStep / 2.0) <= stepmin:
                        if logI[lIdx] is None:
                            # rtemp[outindex]=MyR[Inindex]*((MyQ[inindex]+FunStep/2)-stepmin)/funstep
                            # wtemp[outindex]=(MyQ[inindex]+FunStep/2-stepmin)/funstep
                            # stemp[outindex]=(MyS[InIndex]^2)*((MyQ[inindex]+FunStep/2-stepmin)/funstep)^2
                            logI[lIdx] = (
                                data["I"][origIdx]
                                * ((data["Q"][origIdx] + fundamentalStep / 2.0) - stepmin)
                                / fundamentalStep
                            )
                            logW[lIdx] = (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            logE[lIdx] = (data["E"][origIdx] ** 2.0) * (
                                (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            ) ** 2.0
                        else:
                            # rtemp[outindex]+=MyR[Inindex]*((MyQ[inindex]+FunStep/2)-stepmin)/funstep
                            # wtemp[outindex]+=(MyQ[inindex]+FunStep/2-stepmin)/funstep
                            # stemp[outindex]+=(MyS[InIndex]^2)*((MyQ[inindex]+FunStep/2-stepmin)/funstep)^2
                            logI[lIdx] += (
                                data["I"][origIdx]
                                * ((data["Q"][origIdx] + fundamentalStep / 2.0) - stepmin)
                                / fundamentalStep
                            )
                            logW[lIdx] += (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            logE[lIdx] += (data["E"][origIdx] ** 2.0) * (
                                (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin) / fundamentalStep
                            ) ** 2.0
                    elif (self.data["Q"][origIdx] + fundamentalStep / 2.0) > stepmax:
                        if logI[lIdx] is None:
                            # rtemp[outindex]=MyR[Inindex]*(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # wtemp[outindex]=(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # stemp[outindex]=(MyS[InIndex]^2)*((stepmax-(MyQ[inindex]-FunStep/2))/funstep)^2
                            logI[lIdx] = (
                                data["I"][origIdx]
                                * (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
                            )
                            logW[lIdx] = (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            logE[lIdx] = (data["E"][origIdx] ** 2.0) * (
                                (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            ) ** 2.0
                        else:
                            logI[lIdx] += (
                                data["I"][origIdx]
                                * (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
                            )
                            logW[lIdx] += (
                                stepmax - (self.data["Q"][origIdx] - fundamentalStep / 2.0)
                            ) / fundamentalStep
                            logE[lIdx] += (data["E"][origIdx] ** 2.0) * (
                                (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0)) / fundamentalStep
                            ) ** 2.0
                    else:
                        if logI[lIdx] is None:
                            logI[lIdx] = data["I"][origIdx]
                            logW[lIdx] = 1.0
                            logE[lIdx] = data["E"][origIdx] ** 2.0
                        else:
                            logI[lIdx] += data["I"][origIdx]
                            logW[lIdx] += 1.0
                            logE[lIdx] += data["E"][origIdx] ** 2.0

                    origIdx += 1
                    # emulate the do-while loop
                    if not ((data["Q"][origIdx] - fundamentalStep / 2.0) < stepmax):
                        break

        logI = [logI[ii] / logW[ii] for ii in range(numOfBins)]
        logE = [logE[ii] / (logW[ii] ** 2.0) for ii in range(numOfBins)]
        logE = [le**0.5 for le in logE]

        return (logQ, logI, logE)

    def rescale_data(self, guess_init: bool = False) -> None:
        """Rescale data with thickness.
        Fit the I(Q) data to gaussian and calculate peak area.

        Parameters
        ----------
        guess_init : bool
            Whether to guess initial parameters for fitting with differential_evolution.
        """

        def _gaussian(x: np.ndarray, k0: float, k1: float, k2: float) -> np.ndarray:
            """The Gaussian equation according to Igor definition of gaussian curvefit.

            https://www.wavemetrics.net/doc/igorman/V-01%20Reference.pdf
            """
            return k0 + 1.0 / (k1 * np.sqrt(2 * math.pi)) * np.exp(-1.0 / 2.0 * ((x - k2) / k1) ** 2.0)

        def _sum_of_squared_error(params: tuple[float, float, float]) -> float:
            """Function for genetic algorithm to minimize."""
            k0, k1, k2 = params
            val = _gaussian(np.array(self.data.q), k0, k1, k2)
            result = np.sum((np.array(self.data.i) - val) ** 2)
            return result

        def _generate_initial_parameters(test_x: np.ndarray, test_y: np.ndarray):
            max_x = max(test_x)
            max_y = max(test_y)
            max_xy = max(max_x, max_y)

            parameter_bounds = [
                [-max_xy, max_xy],  # k0
                [-max_xy, max_xy],  # k1
                [-max_xy, max_xy],  # k2
            ]

            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(_sum_of_squared_error, parameter_bounds, seed=3)

            return result.x

        def _clean_iq(q_scaled: list[float], i_scaled: list[float], e_scaled: list[float]):
            """Remove duplicate values in q by:
            - Taking the average of all i-values with the same q
            - Taking the standard deviation of all e-values with the same q
            """
            # Dictionary to store sums for averaging I and propagating errors for E
            sum_dict = defaultdict(lambda: {"I_sum": 0, "I_count": 0, "E_sum_squares": 0})

            for q, i, e in zip(q_scaled, i_scaled, e_scaled):
                sum_dict[q]["I_sum"] += i
                sum_dict[q]["I_count"] += 1
                sum_dict[q]["E_sum_squares"] += e**2

            q_cleaned = []
            i_cleaned = []
            e_cleaned = []

            for q, values in sum_dict.items():
                q_cleaned.append(q)
                i_cleaned.append(values["I_sum"] / values["I_count"])
                e_cleaned.append(math.sqrt(values["E_sum_squares"]))

            return q_cleaned, i_cleaned, e_cleaned

        assert self.size > 0

        initial_vals = [3.8e-6, 0.1, 0.8]
        peak_area = None
        q_offset = None

        for b in range(self.num_of_banks):
            bank = b + 1

            # 2*(Pi^2)*HarNo/(PrimWavel*3600*180)
            h_scale = 2 * (math.pi**2.0) * bank / (self.experiment.prim_wave * 3600.0 * 180.0)
            # VertScale=VertAngle*(DarWidth/HarNo)*Pi/(3600*180)*Sthick//10
            v_scale = (
                self.experiment.v_angle
                * (self.experiment.darwin_width / bank)
                * math.pi
                / (3600.0 * 180.0)
                * self.thickness
            )

            if guess_init:
                initial_vals = _generate_initial_parameters(np.array(self.data.q), np.array(self.data.i))

            best_vals, sigma = curve_fit(
                _gaussian,
                np.array(self.data.q),
                np.array(self.data.i),
                p0=initial_vals,
                sigma=self.data.e,
                maxfev=100000,
            )

            initial_vals = list(best_vals)
            peak_area = (
                1.0
                / (initial_vals[1] * math.sqrt(2 * math.pi))
                * initial_vals[1]
                * math.sqrt(2.0)
                / math.sqrt(2 * math.pi)
            )
            dar_test = initial_vals[1] * math.sqrt(2.0) * (math.pi) ** 0.5 * bank / self.experiment.darwin_width

            if dar_test <= 0.13:
                q_offset = initial_vals[2]
            else:
                q_offset = 0.0

            q_scaled = [math.fabs((q - q_offset) * h_scale) for q in self.data.q]
            i_scaled = [i / (v_scale * peak_area) for i in self.data.i]
            e_scaled = [e / (v_scale * peak_area) for e in self.data.e]

            q_cleaned, i_cleaned, e_cleaned = _clean_iq(q_scaled, i_scaled, e_scaled)
            iq_scaled = IQData(q=q_cleaned, i=i_cleaned, e=e_cleaned, t=[])

            self.data_scaled.append(iq_scaled)

        q_range = f"{min(self.data_scaled[0].q)} - {max(self.data_scaled[0].q)}"
        logging.info(f"Rescaled data for sample {self.name}, Q-range: {q_range} 1/angstrom\n")
        return

    def stitch_data(self):
        """Stitch scans together and store in property `self.data`"""

        for bank in range(self.num_of_banks):
            momentum_transfer = []
            intensity = []
            energy = []
            # transmission = []  # omitted for now

            for scan in self.scans:
                it = list(
                    zip(
                        scan.monitor_data.iq_data.q,
                        scan.monitor_data.iq_data.i,
                        scan.monitor_data.iq_data.e,
                        scan.detector_data[bank].iq_data.i,
                        scan.detector_data[bank].iq_data.e,
                    )
                )

                for mq, mi, me, di, de in it:
                    if mq in momentum_transfer:
                        idx = momentum_transfer.index(mq)
                        var = energy[idx] ** 2 + (de / mi) ** 2
                        intensity[idx] = (intensity[idx] * (energy[idx] ** 2) + (di / mi) * (de / mi) ** 2) / var
                        energy[idx] = var**0.5
                    else:
                        momentum_transfer.append(mq)
                        intensity.append(di / mi)
                        energy.append(de / mi)

            # Sort by momentum transfer (outside the scan loop, inside the bank loop)
            sorted_indices = np.argsort(momentum_transfer)
            momentum_transfer = np.array(momentum_transfer)[sorted_indices]
            intensity = np.array(intensity)[sorted_indices]
            energy = np.array(energy)[sorted_indices]

            self.detector_data.append(
                IQData(
                    q=momentum_transfer.tolist(),
                    i=intensity.tolist(),
                    e=energy.tolist(),
                )
            )
        logging.info(f"Scans stitched together for sample {self.name}.\n")

        h_scale = 2 * (math.pi**2.0) * 1.0 / (self.experiment.prim_wave * 3600.0 * 180.0)
        theta_range_msg = ""
        q_range_msg = ""

        for scan in self.scans:
            q_range = f"{min(scan.detector_data[0].iq_data.q)} - {max(scan.detector_data[0].iq_data.q)}"
            theta_range_msg += f"Theta range for scan {scan.number}: {q_range}\n"

            temp_q = [math.fabs(q * h_scale) for q in scan.detector_data[0].iq_data.q]
            q_range_msg += f"Q range for scan {scan.number}: {min(temp_q)} - {max(temp_q)} 1/angstrom\n"

        logging.info(theta_range_msg)
        logging.info(q_range_msg)
        return

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

        if self.experiment.logbin:
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
            energy = [math.sqrt(data.e[i] ** 2.0 + (scale_f * bg_data.e[i]) ** 2.0) for i in range(num_of_bins)]

            self.data_bg_subtracted.q = momentum_transfer
            self.data_bg_subtracted.i = intensity
            self.data_bg_subtracted.e = energy

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


class Experiment(BaseModel):
    """Experiment configuration for USANS data reduction

    Attributes
    ----------
    config : str
        Path to the configuration file
    output_dir : str | None
        Output folder for reduced data, default is current folder
    prim_wave : float
        Primary wavelength in Angstroms, default is 3.6
    darwin_width : float
        Darwin width, default is 5.1
    v_angle : float
        Vertical angle, default is 0.042
    min_q : float
        Minimum Q value for binning output, default is 1e-6
    step_per_dec : int
        Steps per decade in binning, default is 33
    logbin : bool
        Flag for log-binning, default is False
    """

    config: str = Field(..., description="Path to the configuration file")
    output_dir: str = Field("", description="Output folder for reduced data")
    prim_wave: float = Field(3.6, description="Primary wavelength in Angstroms")
    darwin_width: float = Field(5.1, description="Darwin width")
    v_angle: float = Field(0.042, description="Vertical angle")
    min_q: float = Field(1e-6, description="Minimum Q value for binning output")
    step_per_dec: int = Field(33, description="Steps per decade in binning")
    logbin: bool = Field(False, description="Flag for logbinning")
    # Fields that are initialized in model_post_init and not expected from user input
    folder: str = Field(default="", init=False, description="Working folder for this experiment")
    num_of_banks: int = Field(default=4, init=False, description="Number of detector banks")
    background: "Sample | None" = Field(default=None, init=False, description="Background sample")
    samples: list["Sample"] = Field(default_factory=list, init=False, description="List of samples")

    def model_post_init(self, _context: Any) -> None:  # noqa ANN401
        """Post-validation initializer"""

        # The working folder for this experiment, default is current folder
        _setupfile = os.path.abspath(self.config)
        self.folder = os.path.dirname(_setupfile)

        if bool(self.output_dir) is False:  # in case `output_dir` is an empty string
            self.output_dir = os.path.join(self.folder, "reduced")

        self.num_of_banks: int = 4

        if not os.path.exists(self.config):
            raise FileNotFoundError(f"The file path: {self.config} does not exist")

        self.folder = os.path.dirname(self.config)
        background, samples = read_config(self.config)
        if background is None:
            logging.warning("No background sample defined in the configuration file.")
            self.background = None
        else:
            self.background = Sample(**background, experiment=self)
        self.samples = [Sample(**s, experiment=self) for s in samples]

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
    parser.add_argument("-l", "--logbin", action="store_true", help="Enable log-binning of data during reduction")
    parser.add_argument("-o", "--output", default="", help="Output folder for reduced data (default: current folder)")
    args = parser.parse_args()
    return args


def main():
    """Main function to run USANS data reduction"""
    args = parse_args()
    experiment = Experiment(config=args.path, logbin=args.logbin, output_dir=args.output)
    experiment.reduce()
    generate_report(config_file_path=args.path, output_dir=experiment.output_dir)

    logging.info("USANS data reduction completed.")


if __name__ == "__main__":
    main()
