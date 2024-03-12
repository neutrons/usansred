#!/usr/bin/env python
import os
import csv
import math
import numpy
import copy
from scipy.optimize import curve_fit
import logging


"""summary.py: summary of the reduced data."""
# from mpmath import fp
# from openpyxl import chartsheet
# from numpy.distutils.exec_command import filepath_from_subprocess_output

__author__ = "Yingrui Shang"
__copyright__ = "Copyright 2021, NSD, ORNL"

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


class Scan:
    """
    Contains sample information
    """

    isBackground = False

    def __init__(self, experiment, sample, number, isBackground=False):
        """
        Initialization of sample
        experiment - the experiment this sample belongs to
        name - sample name
        thickness - the thickness of the sample in cm
        number - sample number
        count - file count for this sample
        range - the q range of this scan
        isBackground - if this sample is empty sample for background measurement

        return
        """
        self.experiment = experiment
        self.sample = sample
        self.number = number
        self.isBackground = isBackground

        self.monitorData = {
            "XYData": None,
            "IQData": None,
            "FilePath": None,
        }

        self.detectorData = []
        """
        list of
        {
            'XYData':None,
            'IQData':None,
            'FilePath':[],
            }
        """
        self.load()

        return

    @property
    def range(self):
        """
        property of q range
        """

    def load(self):
        """
        Load experiment data files
        """
        self.loadMonitorFile()
        self.loadDetectorFile()
        return

    def loadMonitorFile(self):
        """
        Load monitor file
        """
        self.monitorData["FilePath"] = os.path.join(
            self.experiment.folder, self._getMonitorFileName()
        )
        self.monitorData["XYData"] = self.readXYFile(self.monitorData["FilePath"])
        self.monitorData["IQData"] = self.convertXYToIQData(self.monitorData["XYData"])
        return

    def loadDetectorFile(self):
        """
        Load detector file
        """
        XYData = []
        IQData = []

        for bank in range(self.numOfBanks):
            filePath = os.path.join(
                self.experiment.folder, self._getDetectorFileName(bank)
            )

            # self.detectorData['FilePath'].append( filePath )

            XYData = self.readXYFile(filePath)
            IQData = self.convertXYToIQData(XYData)

            self.detectorData.append(
                {
                    "XYData": copy.deepcopy(XYData),
                    "IQData": copy.deepcopy(IQData),
                    "FilePath": filePath,
                }
            )
        return

    def readXYFile(self, XYFilePath):
        """
        Load XY File to data structure
        XYFilePath - the xy file path
        return - a dictionary of lists
        """
        XYData = {"X": [], "Y": [], "E": [], "T": []}

        with open(XYFilePath, "r") as xyf:
            csvReader = csv.reader(xyf, delimiter=",")
            for row in csvReader:
                if len(row) < 3 or row[0].startswith("#"):
                    continue

                XYData["X"].append(float(row[0]))
                XYData["Y"].append(float(row[1]))
                XYData["E"].append(float(row[2]))
                if len(row) > 3:
                    XYData["T"].append(float(row[3]))

        return XYData

    def convertXYToIQData(self, XYData):
        """
        Convert XY data to I(Q)
        XYData - a dictionary of lists
                XYData = {
                    "X": [],
                    "Y": [],
                    "E": [],
                    "T": []
                    }
        return - a dictionary of lists
                IQData = {
                    "Q": [],
                    "I": [],
                    "E": [],
                    "T": []
                    }
        """

        IQData = {"Q": [], "I": [], "E": [], "T": []}

        IQData["Q"] = XYData["X"].copy()
        IQData["I"] = XYData["Y"].copy()
        IQData["E"] = [math.sqrt(math.fabs(e - 0.5) + 0.5) for e in XYData["Y"]]
        IQData["T"] = XYData["T"].copy()

        return IQData

    def _getDetectorFileName(self, bank):
        """
        Get the detector file name from sample number
        harNum - the serial number

        return - a string of file name
        """

        return (
            "USANS_" + self.number + "_detector_scan_ARN_peak_" + str(bank + 1) + ".txt"
        )

    def _getMonitorFileName(self):
        """
        Get the monitor file name
        return - string of the file name
        """

        return "USANS_" + self.number + "_monitor_scan_ARN" + ".txt"

    @property
    def numOfBanks(self):
        """
        number of banks are defined in Experiment class
        """
        return self.experiment.numOfBanks

    @property
    def size(self):
        """
        Number of data points in this scan
        """

        return len(self.monitorData["IQData"]["Q"])


class Sample:
    """
    A scan consists of multiple experiments
    """

    isBackground = False

    def __init__(
        self,
        experiment,
        name,
        startNumber,
        numOfScans,
        thickness=0.1,
        isBackground=False,
        exclude=None,
    ):
        """
        name - the name of the scan
        startNumber - the starting number of the sample
        numOfScans - number of consist samples that included in this scan

        return
        """
        self.experiment = experiment
        self.name = name
        self.startSampleNum = startNumber
        self.scans = []
        self.thickness = thickness
        self.isBackground = isBackground

        """
        Exclude the bad runs then keep counting
        """

        if exclude is not None:
            exRunNums = [int(ex) for ex in exclude]
        else:
            exRunNums = []

        for n in range(numOfScans + len(exRunNums)):
            if n + int(self.startSampleNum) in exRunNums:
                continue

            s = Scan(
                experiment,
                self,
                str(n + int(self.startSampleNum)),
                isBackground=self.isBackground,
            )
            self.scans.append(s)

        """
        self.data is the original data after stitched from different scan
        self.dataScaled is the data after self.data is scaled with thickness
        self.dataLogBinned is self.dataScaled log binned
        self.dataBgSubtracted is self.dataScaled with background subtracted
        self.dataReduced is the finally reduced data, currently equivalent to self.dataBgSubtracted
        """
        """
        set as property fundction
        self.data = {
            "Q": [],
            "I": [],
            "E": [],
            "T": []
            }
        """
        self.detectorData = []

        self.dataScaled = []
        self.dataLogBinned = {"Q": [], "I": [], "E": [], "T": []}
        self.dataBgSubtracted = {"Q": [], "I": [], "E": [], "T": []}

        return

    @property
    def dataReduced(self):
        """
        Return the reduced data, this is a wrapper when one decide the stage of reduction

        """
        return self.dataBgSubtracted

    @property
    def isReduced(self):
        """
        Check if the data is reduced
        """
        return self.sizeReduced > 0

    @property
    def isLogBinned(self):
        """
        Check if he data is log binned
        """
        return len(self.dataLogBinned["Q"]) > 0

    @property
    def sizeReduced(self):
        """
        The data size of reduced data
        """
        return len(self.dataReduced["Q"])

    def reduce(self):
        """
        Reduce this scan
        return

        """
        self.stitchData()

        self.rescaleData()

        # only the first bank is processed
        dataScaled = self.dataScaled[0]
        msg = f"Only the first band data are used for sample {self.name}"
        logging.info(msg)

        # logbinning is optional
        (
            self.dataLogBinned["Q"],
            self.dataLogBinned["I"],
            self.dataLogBinned["E"],
        ) = self.logBin(dataScaled["Q"], dataScaled["I"], dataScaled["E"])

        if not self.isBackground:
            self.subtractBg(self.experiment.background)

        msg = f"data reduction finished for sample {self.name}"
        logging.info(msg)

        return

    def stitchData(self):
        """
        Stitch scans together
        The stitched data will be stored in property self.data

        return
        """
        intensity = []
        energy = []
        momentum_transfer = []
        transmission = []  # omitted for now

        # it = None

        for bank in range(self.experiment.numOfBanks):
            for scan in self.scans:
                it = zip(
                    scan.monitorData["IQData"]["Q"],
                    scan.monitorData["IQData"]["I"],
                    scan.monitorData["IQData"]["E"],
                    scan.detectorData[bank]["IQData"]["I"],
                    scan.detectorData[bank]["IQData"]["E"],
                )

                for mq, mi, me, di, de in it:
                    if mq in momentum_transfer:
                        idx = momentum_transfer.index(mq)
                        # Rdat[BeanPoint]=((Rdat[BeanPoint]*(Sdat[BeanPoint])^2+(rdu[MyCount]/rmu[MyCount])*(sdu[MyCount]/rmu[MyCount])^2))/((Sdat[BeanPoint])^2+(sdu[MyCount]/rmu[MyCount])^2)
                        var = energy[idx] ** 2 + (de / mi) ** 2

                        intensity[idx] = (
                            intensity[idx] * (energy[idx] ** 2)
                            + (di / mi) * (de / mi) ** 2
                        ) / var
                        energy[idx] = var**0.5
                    else:
                        momentum_transfer.append(mq)
                        intensity.append(di / mi)
                        energy.append(de / mi)

            zipped = list(zip(momentum_transfer, intensity, energy))  # suppose no T

            zipped = sorted(zipped, key=lambda z: z[0], reverse=False)
            # unzip
            momentum_transfer, intensity, energy = zip(*zipped)

            self.detectorData.append(
                {
                    "Q": list(momentum_transfer).copy(),
                    "I": list(intensity).copy(),
                    "E": list(energy).copy(),
                    "T": list(transmission).copy(),
                }
            )

            intensity = []
            momentum_transfer = []
            energy = []
            transmission = []

        msg = f"Scans stitched together for sample {self.name}\n"
        logging.info(msg)
        for scan in self.scans:
            msg += f"theta range ({scan.number}): {min(scan.detectorData[0]['IQData']['Q'])} - {max(scan.detectorData[0]['IQData']['Q'])}  \n"
        logging.info(msg)

        hScale = 2 * (math.pi**2.0) * 1.0 / (Experiment.primWave * 3600.0 * 180.0)
        for scan in self.scans:
            tempq = [qq * hScale for qq in scan.detectorData[0]["IQData"]["Q"]]
            tempq = [math.fabs(qq) for qq in tempq]
            msg += (
                f"Q range ({scan.number}): {min(tempq)} - {max(tempq)} Angtrom^(-1) \n"
            )
        logging.info(msg)

        return

    def rescaleData(self, guess_init=False):
        """
        Rescale data with thickness.
        Fit the I(Q) data with gaussian and calculate the peak area.

        guess_init: if guess the initial value with differential_evolution
        return
        """

        def _gaussian(x, k0, k1, k2):
            """
            The Gaussian equation defined abide to Igor definition of gaussian curvefit
            https://www.wavemetrics.net/doc/igorman/V-01%20Reference.pdf
            """
            # return k0 + k1 * numpy.exp( -1 * ( (x - k2) / k3 ) ** 2. )
            return k0 + 1.0 / (k1 * numpy.sqrt(2 * math.pi)) * numpy.exp(
                -1.0 / 2.0 * ((x - k2) / k1) ** 2.0
            )

        assert self.size > 0

        initVals = [3.8e-6, 0.1, 0.8]
        peakArea = None
        qOffset = None

        from scipy.optimize import differential_evolution
        import warnings

        # function for genetic algorithm to minimize (sum of squared error)
        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings(
                "ignore"
            )  # do not print warnings by genetic algorithm
            val = _gaussian(numpy.array(self.data["Q"]), *parameterTuple)
            return numpy.sum((numpy.array(self.data["I"]) - val) ** 2.0)

        def generate_Initial_Parameters(test_x, test_y):
            # min and max used for bounds
            max_x = max(test_x)
            # minX = min(test_X)
            max_y = max(test_y)
            # minY = min(test_Y)
            max_xy = max(max_x, max_y)
            # minXY = min(minX, minY)

            parameterBounds = []
            parameterBounds.append([-max_xy, max_xy])  # seach bounds for k0
            parameterBounds.append([-max_xy, max_xy])  # seach bounds for k1
            parameterBounds.append([-max_xy, max_xy])  # seach bounds for k2
            # parameterBounds.append([-maxXY, maxXY]) # seach bounds for k3

            # "seed" the numpy random number generator for repeatable results
            result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
            return result.x

        for bb in range(self.numOfBanks):
            bank = bb + 1

            # 2*(Pi^2)*HarNo/(PrimWavel*3600*180)
            hScale = 2 * (math.pi**2.0) * bank / (Experiment.primWave * 3600.0 * 180.0)
            # VerScale=VertAngle*(DarWidth/HarNo)*Pi/(3600*180)*Sthick//10
            vScale = (
                Experiment.vAngle
                * (Experiment.DarwinWidth / bank)
                * math.pi
                / (3600.0 * 180.0)
                * self.thickness
            )

            print(self.name)

            if guess_init:
                initVals = generate_Initial_Parameters(
                    numpy.array(self.data["Q"]), numpy.array(self.data["I"])
                )

            bestVals, sigma = curve_fit(
                _gaussian,
                numpy.array(self.data["Q"]),
                numpy.array(self.data["I"]),
                p0=initVals,
                sigma=self.data["E"],
                maxfev=100000,
            )

            initVals = list(bestVals)

            # peakArea = bestVals[1] * bestVals[3] / ( ( 2 * math.pi ) ** .5 )
            peakArea = (
                1.0
                / (bestVals[1] * math.sqrt(2 * math.pi))
                * bestVals[1]
                * math.sqrt(2.0)
                / ((2 * math.pi) ** 0.5)
            )

            # DarTest = bestVals[3] * (math.pi) ** .5 * bank / Experiment.DarwinWidth
            DarTest = (
                bestVals[1]
                * math.sqrt(2.0)
                * (math.pi) ** 0.5
                * bank
                / Experiment.DarwinWidth
            )

            if DarTest <= 0.13:
                qOffset = bestVals[2]
            else:
                qOffset = 0.0

            qScaled = [(qq - qOffset) * hScale for qq in self.data["Q"]]
            qScaled = [math.fabs(qq) for qq in qScaled]

            iScaled = [ii / vScale / peakArea for ii in self.data["I"]]
            eScaled = [ee / vScale / peakArea for ee in self.data["E"]]

            dataScaled = {
                "Q": qScaled.copy(),
                "I": iScaled.copy(),
                "E": eScaled.copy(),
                "T": [],
            }

            self.dataScaled.append(dataScaled)

        msg = f"Recale finished for {self.name}, Q range: {min(self.dataScaled[0]['Q'])} - {max(self.dataScaled[0]['Q'])} 1/angtrom \n"
        logging.info(msg)

        return

    def logBin(self, momentum_transfer, intensity, energy):
        """
        log bin the data
        return
        """

        assert len(momentum_transfer) == len(intensity) == len(energy)

        zipped = list(zip(momentum_transfer, intensity, energy))  # suppose no T

        zipped = sorted(zipped, key=lambda z: z[0], reverse=False)
        # unzip
        momentum_transfer, intensity, energy = zip(*zipped)

        data = {"I": list(intensity), "Q": list(momentum_transfer), "E": list(energy)}

        # The fundamental Q width of the measurement
        harNo = 1.0  # Only the first harmonic peak is used
        fundamentalStep = (
            2
            * math.pi**2
            * Experiment.DarwinWidth
            * harNo
            / (Experiment.primWave * 3600.0 * 180.0)
        )

        # Step multiplier
        alpha = math.exp(math.log(10) / Experiment.stepPerDec)
        # step relative width
        kappa = 2.0 * (alpha - 1) / (alpha + 1)

        # floor ((ln((MyQ[InLength-1])/Qmin))/(ln(alpha)))
        numOfBins = math.floor(
            math.log(max(data["Q"]) / Experiment.minQ) / math.log(alpha)
        )

        logQ = [Experiment.minQ * (alpha**ii) for ii in range(numOfBins)]
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
                        logI[lIdx] = ((k3 / k2) + 1) * data["I"][origIdx + 1] - (
                            k3 / k2
                        ) * data["I"][origIdx]
                        # stemp[outindex]=((((k3/k2)+1)^2)*(MyS[inindex+1]^2)+((k3/k2)^2)*(MyS[inindex]^2))//*(k2/Testval)^2
                        logE[lIdx] = (((k3 / k2) + 1) ** 2.0) * (
                            data["E"][origIdx + 1] ** 2.0
                        ) + ((k3 / k2) ** 2.0) * (data["E"][origIdx] ** 2.0)
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
                                * (
                                    (data["Q"][origIdx] + fundamentalStep / 2.0)
                                    - stepmin
                                )
                                / fundamentalStep
                            )
                            logW[lIdx] = (
                                data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin
                            ) / fundamentalStep
                            logE[lIdx] = (data["E"][origIdx] ** 2.0) * (
                                (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin)
                                / fundamentalStep
                            ) ** 2.0
                        else:
                            # rtemp[outindex]+=MyR[Inindex]*((MyQ[inindex]+FunStep/2)-stepmin)/funstep
                            # wtemp[outindex]+=(MyQ[inindex]+FunStep/2-stepmin)/funstep
                            # stemp[outindex]+=(MyS[InIndex]^2)*((MyQ[inindex]+FunStep/2-stepmin)/funstep)^2
                            logI[lIdx] += (
                                data["I"][origIdx]
                                * (
                                    (data["Q"][origIdx] + fundamentalStep / 2.0)
                                    - stepmin
                                )
                                / fundamentalStep
                            )
                            logW[lIdx] += (
                                data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin
                            ) / fundamentalStep
                            logE[lIdx] += (data["E"][origIdx] ** 2.0) * (
                                (data["Q"][origIdx] + fundamentalStep / 2.0 - stepmin)
                                / fundamentalStep
                            ) ** 2.0
                    elif (self.data["Q"][origIdx] + fundamentalStep / 2.0) > stepmax:
                        if logI[lIdx] is None:
                            # rtemp[outindex]=MyR[Inindex]*(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # wtemp[outindex]=(stepmax-(MyQ[inindex]-FunStep/2))/funstep
                            # stemp[outindex]=(MyS[InIndex]^2)*((stepmax-(MyQ[inindex]-FunStep/2))/funstep)^2
                            logI[lIdx] = (
                                data["I"][origIdx]
                                * (
                                    stepmax
                                    - (data["Q"][origIdx] - fundamentalStep / 2.0)
                                )
                                / fundamentalStep
                            )
                            logW[lIdx] = (
                                stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0)
                            ) / fundamentalStep
                            logE[lIdx] = (data["E"][origIdx] ** 2.0) * (
                                (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
                            ) ** 2.0
                        else:
                            logI[lIdx] += (
                                data["I"][origIdx]
                                * (
                                    stepmax
                                    - (data["Q"][origIdx] - fundamentalStep / 2.0)
                                )
                                / fundamentalStep
                            )
                            logW[lIdx] += (
                                stepmax
                                - (self.data["Q"][origIdx] - fundamentalStep / 2.0)
                            ) / fundamentalStep
                            logE[lIdx] += (data["E"][origIdx] ** 2.0) * (
                                (stepmax - (data["Q"][origIdx] - fundamentalStep / 2.0))
                                / fundamentalStep
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

        # End If
        logI = [logI[ii] / logW[ii] for ii in range(numOfBins)]
        logE = [logE[ii] / (logW[ii] ** 2.0) for ii in range(numOfBins)]
        logE = [le**0.5 for le in logE]

        return (logQ, logI, logE)

    def subtractBg(self, background, vScale=1.0):
        """
        Subtract the background
        background - the background sample, should be processed (stitched, scaled, and binned)

        return
        """
        if self.experiment.logbin:
            assert self.isLogBinned
            msg = f"Logbinned data are used for background subtraction. Sample {self.name}, background {background.name}"
            logging.info(msg)
            # The logbinned data a subtracted from logbinned
            data = self.dataLogBinned
            bgData = background.dataLogBinned

            scale_f = vScale * self.thickness / background.thickness

            sampleNumOfBins = self.numOfLogBins
            bgNumOfBins = self.numOfLogBins

            if sampleNumOfBins < bgNumOfBins:
                num_of_bins = sampleNumOfBins
                momentum_transfer = data["Q"].copy()
            else:
                num_of_bins = bgNumOfBins
                momentum_transfer = bgData["Q"].copy()

            intensity = [
                data["I"][ii] - scale_f * bgData["I"][ii] for ii in range(num_of_bins)
            ]
            energy = [
                (data["E"][ii] ** 2.0 + (scale_f * bgData["E"][ii]) ** 2.0) ** 0.5
                for ii in range(num_of_bins)
            ]

            self.dataBgSubtracted["Q"] = momentum_transfer.copy()
            self.dataBgSubtracted["I"] = intensity.copy()
            self.dataBgSubtracted["E"] = energy.copy()

        else:  # if logbinned data subtraction is not called for, then use interpolation
            # only the first bank is processed
            dataScaled = self.dataScaled[0]
            bgScaled = self.experiment.background.dataScaled[0]
            interBg = numpy.interp(dataScaled["Q"], bgScaled["Q"], bgScaled["I"])
            interBgE = numpy.interp(dataScaled["Q"], bgScaled["Q"], bgScaled["E"])
            self.dataBgSubtracted["Q"] = dataScaled["Q"].copy()
            self.dataBgSubtracted["I"] = (dataScaled["I"] - interBg).copy()
            # FIXME The error bar got intepolated as well
            self.dataBgSubtracted["E"] = (
                (numpy.array(dataScaled["E"]) ** 2 + interBgE**2) ** 0.5
            ).copy()

        msg = f"background subtracted from sample {self.name}, (background sample {background.name})"
        logging.info(msg)

        return

    def dumpReducedDataToCSV(
        self,
        detectorData=True,
        scaledData=True,
        logBinnedData=True,
        bgSubtractedData=True,
    ):
        """
        dump the data to output folder defined in experiment
        detectorData, scaleData, logBinnedData, bgSubtractedData - flags to indicate which set of data to dump

        filenames follow the legacy Igor output file names
                "UN_" + sampleName + "_det_1.txt", \
                "UN_" + sampleName + "_det_1_lb.txt", \
                "UN_" + sampleName + "_det_1_lbs.txt"
        return
        """
        if detectorData is True and self.data:
            file_name = "UN_" + self.name + "_det_1_unscaled.txt"
            file_path = os.path.join(self.experiment.outputFolder, file_name)
            self.dumpDataToCSV(file_path, self.data)

        if self.isLogBinned:
            if scaledData is True:
                file_name = "UN_" + self.name + "_det_1.txt"
                file_path = os.path.join(self.experiment.outputFolder, file_name)
                self.dumpDataToCSV(file_path, self.dataScaled[0])

            if logBinnedData is True:
                file_name = "UN_" + self.name + "_det_1_lb.txt"
                file_path = os.path.join(self.experiment.outputFolder, file_name)
                self.dumpDataToCSV(file_path, self.dataLogBinned)

            if bgSubtractedData is True and (not self.isBackground):
                file_name = "UN_" + self.name + "_det_1_lbs.txt"
                file_path = os.path.join(self.experiment.outputFolder, file_name)
                self.dumpDataToCSV(file_path, self.dataBgSubtracted)
        return

    def dumpDataToCSV(self, filePath, data, title=None):
        """
        Dump data to CSV file
        filePath - the file path to dump
        data - the data to dump, each key is a column.
        title - the colume title in list or tuple, if not specified use the key

        return
        """
        keys = list(data.keys())
        # has to make sure it runs to the longest column
        nRows = max([len(data[key]) for key in keys])

        with open(filePath, "w") as fp:
            writer = csv.writer(fp, delimiter=",")
            if title is not None:
                writer.writerow(title)
            for ii in range(nRows):
                rr = []
                for kk in keys:
                    if ii < len(data[kk]):
                        rr.append(data[kk][ii])
                    else:
                        rr.append("")
                writer.writerow(rr)

        return

    @property
    def size(self):
        """
        return the length of detector data points
        """
        return len(self.data["Q"])

    @property
    def data(self):
        """
        The main data from detector, stitched and normalized with monitor data, currently the first bank detector data, scaled with monitor
        """
        if self.detectorData:
            return self.detectorData[0]
        else:
            return None

    @property
    def numOfScans(self):
        return len(self.scans)

    @property
    def numOfBanks(self):
        """
        number of banks are defined in experiment class

        """

        return self.experiment.numOfBanks

    @property
    def numOfLogBins(self):
        """
        Return the size of log binnd data
        """
        return len(self.dataLogBinned["Q"])

    def __eq__(self, sc2):
        """
        Overloading the equal operator ==
        """

        if sc2.name == self.name and sc2.startNumber == self.startNumber:
            return True
        else:
            return False

    def getFileNamesFromSample(self, sampleName):
        """
        Get the reduced file names from sample name
        sampleName - sample name
        return - a list of reduced file names associated with this sample

        """
        if sampleName:
            return (
                "UN_" + sampleName + "_det_1.txt",
                "UN_" + sampleName + "_det_1_lb.txt",
                "UN_" + sampleName + "_det_1_lbs.txt",
            )
        else:
            logging.info("Sample name is empty or not valid")
            raise

        return


class Experiment:
    primWave = 3.6
    DarwinWidth = 5.1
    vAngle = 0.042
    minQ = 1e-6  # minimum Q value for binning output
    stepPerDec = 33  # steps per decade in binning
    logbin = False  # flag for logbinning

    def __init__(self, csvFilePath, logbin=False, outputFolder=None):
        """
        Constructer of USANS class

        csvFilePath - the setup file for reduction

        return
        """
        self.background = None
        self._setupfile = os.path.abspath(csvFilePath)
        self.folder = os.path.dirname(
            self._setupfile
        )  # The working folder for this experiment, default is current folder
        self.samples = []
        self.logbin = logbin

        if outputFolder is None:
            self.outputFolder = os.path.join(self.folder, "reduced")
        self.numOfBanks = 4

        if not os.path.exists(csvFilePath):
            logging.info(f"The file path: {csvFilePath} does not exist")
            raise

        self.folder = os.path.dirname(csvFilePath)

        with open(csvFilePath, newline="") as csvFile:
            csvReader = csv.reader(csvFile, delimiter=",")

            for row in csvReader:
                # skip comment rows
                if row[0].startswith("#"):
                    continue
                try:
                    exRunNo = None
                    if len(row) == 6:
                        exRunNo = row[5].split(";")
                    sample = Sample(
                        self,
                        row[1],
                        row[2],
                        numOfScans=int(row[3]),
                        thickness=float(row[4]),
                        exclude=exRunNo,
                    )
                except:  # noqa E722
                    sample = None
                    logging.info("Cannot initiate sample instance!")
                    # traceback.print_exc()

                if sample is not None:
                    if row[0] == "b":
                        # Check if this sample is the empty sample
                        sample.isBackground = True
                        self.background = sample
                    else:
                        self.samples.append(sample)
        return

    def reduce(self, outputFolder=None):
        """
        Reduce the USANS data
        outputFolder - the result will be dumped to the output folder. If none will just use current folder
        return
        """
        if outputFolder is not None:
            self.outputFolder = outputFolder

        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        try:
            self.background.reduce()

        except:  # noqa E722
            msg = "Cannot reduce background" + self.background.name
            logging.info(msg)

            # traceback.print_exc()

        for sample in self.samples:
            try:
                sample.reduce()
            except Exception as ee:
                msg = "Cannot reduce sample" + sample.name
                msg += str(ee)
                logging.info(msg)

                # traceback.print_exc()

        self.dumpReducedData()

        return

    def dumpReducedData(self):
        """
        dump reduced data to txt files
        return
        """
        for sample in self.samples:
            sample.dumpReducedDataToCSV()

        self.background.dumpReducedDataToCSV()


if __name__ == "__main__":
    from usansred.summary import reportFromCSV
    import argparse

    parser = argparse.ArgumentParser(description="USANS data reduction at ORNL.")
    # parser.add_argument('csv file', metavar='csv_file', type=string, nargs='+',
    #                    help='path to the csv data file')
    parser.add_argument(
        "-l", "--logbin", help="flag of logbinning", action="store_true"
    )

    parser.add_argument("path")

    args = parser.parse_args()

    # target_csvfile = sys.path(args.path)

    if not os.path.exists(args.path):
        print("The csv file doesn't exist")
        raise SystemExit(1)

    # csvFile = sys.argv[1]

    exp = Experiment(args.path, logbin=args.logbin)
    exp.reduce()

    reportFromCSV(args.path, exp.outputFolder)
