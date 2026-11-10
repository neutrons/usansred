import json
import logging
import math
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np
from mantid.simpleapi import (
    ConvertTableToMatrixWorkspace,
    CropWorkspace,
    LoadEventNexus,
    Rebin,
    StepScan,
    mtd,
)
from matplotlib import use
from plot_publisher import plot1d

from usansred import reduce
from usansred.io.save import save_ascii, save_summed_spectra
from usansred.models.reduction_config import ReductionConfig, SampleConfig
from usansred.summary import generate_report
from usansred.utils.logging import get_logger

logger = get_logger(__name__)

use("agg")
np.seterr(all="ignore")
warnings.filterwarnings("ignore", module="numpy")
peaks = []


def update_sequence_info(out_file, info):
    scan_dict_global = {}

    if os.path.isfile(out_file):
        with open(out_file, "r") as fd:
            try:
                scan_dict_global = json.loads(fd.read())
            except:  # noqa E722
                logger.error("Could not read json file: creating a new one")

    scan_dict_global.update(info)

    with open(out_file, "w") as fd:
        fd.write(json.dumps(scan_dict_global))


def get_sequence_info(seq_file):
    """Get the sequence information from local autoreduce folder json files for autoreduction"""
    if os.path.exists(seq_file):
        with open(seq_file) as json_file:
            seq_dict = json.load(json_file)
    else:
        return None

    return seq_dict


# Name of the setup file written by step 2, in the output directory
SETUP_FILENAME = "auto.json"

# Name of the directory, in the output directory, where step 2 writes the reduced data
REDUCTION_DIRNAME = "auto"

# Files in which step 1 records the runs of a sequence, named after the first run of the sequence
SCAN_FILE_PATTERN = re.compile(r"^scan_(?P<first_run>.+)\.json$")

# Log holding the run number of the first run of the sequence that a run belongs to
FIRST_RUN_LOG = "BL1A:CS:Scan:USANS:FirstRun"

# Value of the log FIRST_RUN_LOG for a run that is not part of any sequence
NO_SEQUENCE = -1

# Log holding the thickness of the sample, in cm
THICKNESS_LOG = "BL1A:CS:Scan:USANS:SampleThickness"

# Thickness of the sample, in cm, when the log THICKNESS_LOG gives none
DEFAULT_THICKNESS = 0.1


def read_first_run(filename: str) -> int | None:
    """Read the first run of the sequence that a run belongs to, from its event file.

    The log ``BL1A:CS:Scan:USANS:FirstRun`` is read with h5py, without loading the events, so that
    the runs outside a sequence can be skipped quickly.

    Parameters
    ----------
    filename : str
        Path of the event file of the run.

    Returns
    -------
    int | None
        The last value of the log: the run number of the first run of the sequence, or -1 for a run
        that is not part of any sequence. ``None`` if the file has no such log, or the log no value.

    Raises
    ------
    OSError
        If the file cannot be opened as an HDF5 file.
    """
    with h5py.File(filename, "r") as nexus:
        path = f"/entry/DASlogs/{FIRST_RUN_LOG}/value"
        if path not in nexus:
            return None
        values = np.atleast_1d(nexus[path][()]).ravel()
    if values.size == 0:
        return None
    return int(values[-1])


def sample_thickness(run) -> float:
    """Read the thickness of the sample, in cm, from the log ``BL1A:CS:Scan:USANS:SampleThickness``.

    Parameters
    ----------
    run : mantid.api.Run
        Run of the workspace of the events.

    Returns
    -------
    float
        The last value of the log, in cm. 0.1 cm if the log is missing, is not a number, or is not
        above 0, which is logged.
    """
    if not run.hasProperty(THICKNESS_LOG):
        logger.info(f"No log {THICKNESS_LOG}: the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    value = run.getProperty(THICKNESS_LOG).value
    # A time-series log holds a sequence of values, of which the last one is taken
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        value = value[-1] if len(value) > 0 else None
    try:
        thickness = float(value)
    except (TypeError, ValueError):
        logger.info(f"The log {THICKNESS_LOG} is not a number ({value!r}): the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    if not (math.isfinite(thickness) and thickness > 0):
        logger.info(f"The log {THICKNESS_LOG} is not above 0 ({thickness}): the thickness is {DEFAULT_THICKNESS} cm")
        return DEFAULT_THICKNESS
    return thickness


def record_run(outdir: str, first_run: int, run_number: str, iq_file: str, run_title: str, thickness: float) -> None:
    """Record a run in the files of its sequence, ``scan_<N>.json`` and ``sample_<N>.json``.

    ``scan_<N>.json`` gets an entry for the run, with its I(Q) file and its title. ``sample_<N>.json``
    gets the title, the thickness and the background flag 0, replacing the values written before.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.
    first_run : int
        Run number of the first run of the sequence, the ``N`` of the file names.
    run_number : str
        Run number of the run.
    iq_file : str
        Path of the file of the scan of the main harmonic.
    run_title : str
        Title of the run.
    thickness : float
        Thickness of the sample, in cm.
    """
    update_sequence_info(
        os.path.join(outdir, f"scan_{first_run}.json"),
        {str(run_number): {"iq": iq_file, "title": run_title}},
    )
    update_sequence_info(
        os.path.join(outdir, f"sample_{first_run}.json"),
        {"title": run_title, "background": 0, "thickness": thickness},
    )


@dataclass
class RunSequence:
    """The information recorded by step 1 for one sequence of runs.

    Parameters
    ----------
    first_run : int
        Run number of the first run of the sequence, the ``N`` of ``scan_<N>.json``.
    runs : dict[str, dict]
        Content of ``scan_<N>.json``: one entry per run, keyed by the run number.
    sample : dict
        Content of ``sample_<N>.json``, or an empty dictionary if the file is missing.
    """

    first_run: int
    runs: dict[str, dict] = field(default_factory=dict)
    sample: dict = field(default_factory=dict)


def collect_sequences(outdir: str) -> list[RunSequence]:
    """Collect the sequences of runs recorded by step 1 in an output directory.

    There is one sequence for each file ``scan_<N>.json``, where ``N`` is an integer, ordered by ``N``.
    Its sample information is read from ``sample_<N>.json``, with the same ``N``; if that file is missing,
    the sample information is empty, and a warning is logged.
    ``scan_-1.json`` is ignored: it holds the runs whose log ``BL1A:CS:Scan:USANS:FirstRun`` is -1,
    which are not part of any sequence. A file whose ``N`` is not an integer is ignored, with a warning.

    Parameters
    ----------
    outdir : str
        Output directory of the preprocessing script.

    Returns
    -------
    list[RunSequence]
        The sequences, ordered by the run number of their first run.
    """
    scan_files = []
    for filename in os.listdir(outdir):
        match = SCAN_FILE_PATTERN.match(filename)
        if match is None:
            continue
        try:
            first_run = int(match.group("first_run"))
        except ValueError:
            logger.warning(f"Ignoring {filename}: {match.group('first_run')} is not a run number")
            continue
        scan_files.append((first_run, filename))
    # Order by run number, not by file name, which would put scan_10001.json before scan_9990.json
    scan_files.sort()

    sequences = []
    for first_run, filename in scan_files:
        scan_file = os.path.join(outdir, filename)
        runs = get_sequence_info(scan_file)
        if first_run == NO_SEQUENCE:
            logger.info(f"Ignoring {filename}: its {len(runs)} runs are not part of any sequence (FirstRun = -1)")
            continue
        sample_filename = f"sample_{first_run}.json"
        sample = get_sequence_info(os.path.join(outdir, sample_filename))
        if sample is None:
            logger.warning(f"{sample_filename} not found: the sample of {filename} takes the default values")
            sample = {}
        sequences.append(RunSequence(first_run=first_run, runs=runs, sample=sample))
    return sequences


def build_setup(sequences: list[RunSequence]) -> ReductionConfig:
    """Build the setup of the reduction of the sequences.

    Every sequence is a sample: the setup has no background and no empty cell. The runs of a sample
    span the run numbers of its sequence, from the lowest to the highest; the run numbers of that span
    missing from the sequence are excluded. ``Sample`` reads ``num_of_scans`` consecutive runs from
    ``start_scan_num`` and skips the excluded ones, so ``num_of_scans`` is the length of the whole span.

    Parameters
    ----------
    sequences : list[RunSequence]
        The sequences, as returned by ``collect_sequences``.

    Returns
    -------
    ReductionConfig
        The setup, with one sample per sequence.
    """
    samples = []
    for index, sequence in enumerate(sequences):
        sample_info = sequence.sample
        # Thickness in cm
        thickness = float(sample_info["thickness"]) if "thickness" in sample_info else DEFAULT_THICKNESS
        run_numbers = sorted(int(run) for run in sequence.runs)
        first, last = run_numbers[0], run_numbers[-1]
        samples.append(
            SampleConfig(
                # The names are taken from the run titles in a later change (#107)
                name=f"Sample{index}",
                start_scan_num=first,
                num_of_scans=last - first + 1,
                thickness=thickness,
                exclude=sorted(set(range(first, last + 1)) - set(run_numbers)),
            )
        )
    return ReductionConfig(samples=samples)


def write_setup_file(config: ReductionConfig, outdir: str) -> str:
    """Write the setup of the reduction as ``auto.json`` in the output directory.

    Parameters
    ----------
    config : ReductionConfig
        The setup, as returned by ``build_setup``.
    outdir : str
        Output directory of the preprocessing script.

    Returns
    -------
    str
        Path of the setup file.
    """
    setup_file = os.path.join(outdir, SETUP_FILENAME)
    with open(setup_file, "w") as fd:
        fd.write(config.model_dump_json(indent=2, exclude_none=True))
        fd.write("\n")
    return setup_file


def reduce_sequences(setup_file: str, outdir: str) -> None:
    """Reduce the setup file into ``<outdir>/auto``, and write the summary report.

    Parameters
    ----------
    setup_file : str
        Path of the setup file, as returned by ``write_setup_file``.
    outdir : str
        Output directory of the preprocessing script.
    """
    autodir = os.path.join(outdir, REDUCTION_DIRNAME)
    os.makedirs(autodir, exist_ok=True)

    exp = reduce.Experiment(config_file=setup_file)
    exp.reduce(output_dir=autodir)

    generate_report(config_file_path=setup_file, output_dir=exp.output_dir)


def main():
    # check number of arguments
    if len(sys.argv) != 3:
        print("Autoreduction code requires a filename and an output directory")
        sys.exit(1)

    if not (os.path.isfile(sys.argv[1])):
        print(f"Data file {sys.argv[1]} not found")
        sys.exit()
    else:
        filename = sys.argv[1]
        outdir = sys.argv[2]

    # The runs outside a sequence are neither preprocessed nor reduced (#107)
    first_run = read_first_run(filename)
    if first_run is None or first_run == NO_SEQUENCE:
        reason = f"it has no log {FIRST_RUN_LOG}" if first_run is None else f"its log {FIRST_RUN_LOG} is -1"
        logger.info(f"{filename} is not processed: {reason}, so the run is not part of any sequence")
        return

    # Don't load monitors unless we really need them
    try:
        LoadEventNexus(filename, LoadMonitors=True, OutputWorkspace="USANS")
        load_monitors = True
    except:  # noqa E722
        LoadEventNexus(filename, LoadMonitors=False, OutputWorkspace="USANS")
        load_monitors = False

    # LoadEventNexus(filename, LoadMonitors=False, OutputWorkspace="USANS")
    # load_monitors = False

    file_prefix = os.path.split(filename)[1].split(".")[0]

    # if mtd['USANS'].getRun().hasProperty("BL1A:CS:Scan:USANS:Wavelength"):
    #    main_wl = mtd['USANS'].getRun().getProperty("BL1A:CS:Scan:USANS:Wavelength").value[0]
    # else:
    #    main_wl = "main_peak"

    # Get ROI from logs
    wavelength = [3.6, 1.8, 1.2, 0.9, 0.72, 0.6]
    roi_min = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Min").value[-1]
    # roi_step = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Size").value[-1]

    # Reference to the item in the wavelength array
    main_index = 1
    for i in range(1, 8):
        lower_bound = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Min" % i).value[-1]
        tof_step = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Size" % i).value[-1]
        if i > 1 and lower_bound == roi_min:
            main_index = i - 2
        peaks.append([lower_bound * 1000.0, (lower_bound + tof_step) * 1000.0])
    main_wl = wavelength[main_index]

    # Produce ASCII data
    Rebin(InputWorkspace="USANS", Params="0,10,17000", OutputWorkspace="USANS")
    file_path = os.path.join(outdir, "%s_detector_trans.txt" % file_prefix)
    save_summed_spectra("USANS", file_path, header="TOF, COUNTS, ERROR")

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=0,
        EndWorkspaceIndex=1023,
        OutputWorkspace="USANS_detector",
    )
    file_path = os.path.join(outdir, "%s_detector.txt" % file_prefix)
    save_summed_spectra("USANS_detector", file_path, header="TOF, COUNTS, ERROR")

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=1024,
        EndWorkspaceIndex=2047,
        OutputWorkspace="USANS_trans",
    )
    file_path = os.path.join(outdir, "%s_trans.txt" % file_prefix)
    save_summed_spectra("USANS_trans", file_path, header="TOF, COUNTS, ERROR")

    if load_monitors:
        Rebin(
            InputWorkspace="USANS_monitors",
            Params="0,10,17000",
            OutputWorkspace="USANS_monitors",
        )
        file_path = os.path.join(outdir, "%s_monitor.txt" % file_prefix)
        save_ascii("USANS_monitors", file_path, header="TOF, COUNTS, ERROR")

    # Find whether we have a motor turning
    short_name = ""
    for item in mtd["USANS"].getRun().getProperties():
        if item.name.startswith("BL1A:Mot:") and not item.name.endswith(".RBV"):
            stats = item.getStatistics()
            if (
                abs(stats.mean) > 0
                and abs(stats.standard_deviation / item.getStatistics().mean) > 0.01
                or abs(stats.mean) == 0
                and abs(stats.standard_deviation) > 0.0
            ):
                scan_var = item.name
                short_name = item.name.replace("BL1A:Mot:", "")

                y_monitor = None
                if load_monitors:
                    StepScan(
                        InputWorkspace="USANS_monitors",
                        OutputWorkspace="mon_scan_table",
                    )
                    ConvertTableToMatrixWorkspace(
                        InputWorkspace="mon_scan_table",
                        ColumnX=scan_var,
                        ColumnY="Counts",
                        ColumnE="Error",
                        OutputWorkspace="USANS_scan_monitor",
                    )
                    file_path = os.path.join(outdir, "%s_monitor_scan_%s.txt" % (file_prefix, short_name))
                    save_ascii("USANS_scan_monitor", file_path, header=f"{scan_var}, COUNTS, ERROR")
                    y_monitor = mtd["USANS_scan_monitor"].readY(0)

                iq_file_path_simple = os.path.join(outdir, "%s_iq_%s_%s.txt" % (file_prefix, short_name, main_wl))
                iq_fd_simple = open(iq_file_path_simple, "w")

                iq_file_path = os.path.join(outdir, "%s_iq_%s.txt" % (file_prefix, short_name))
                iq_fd = open(iq_file_path, "w")

                start_time = mtd["USANS"].getRun().getProperty("start_time").value
                experiment = mtd["USANS"].getRun().getProperty("experiment_identifier").value
                run_number = mtd["USANS"].getRun().getProperty("run_number").value
                run_title = mtd["USANS"].getRun().getProperty("run_title").value
                sequence_first_run = mtd["USANS"].getRun().getProperty("BL1A:CS:Scan:USANS:FirstRun").value[-1]
                sequence_index = mtd["USANS"].getRun().getProperty("BL1A:CS:Scan:USANS:Index").value[-1]
                meta_wavelength = mtd["USANS"].getRun().getProperty("BL1A:CS:Scan:USANS:Wavelength").value[-1]
                print(f"Wavelength: {wavelength[main_index]} [{meta_wavelength}]")

                iq_fd.write("# Experiment %s Run %s\n" % (experiment, run_number))
                iq_fd.write("# Run start time: %s\n" % start_time)
                iq_fd.write("# Title: %s\n" % run_title)
                iq_fd.write("# Sequence ID: %s\n" % sequence_first_run)
                iq_fd.write("# Sequence index: %s\n" % sequence_index)
                iq_fd.write("# Selected wavelength: %s\n" % wavelength[main_index])
                iq_fd.write(
                    "# %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-5s\n"
                    % ("Q", "I(Q)", "dI(Q)", "dQ", "N(Q)", "dN(Q)", "Mon(Q)", "Lambda")
                )
                iq_fd_simple.write("# Experiment %s Run %s\n" % (experiment, run_number))
                iq_fd_simple.write("# Run start time: %s\n" % start_time)
                iq_fd_simple.write("# Title: %s\n" % run_title)
                iq_fd_simple.write("# Sequence ID: %s\n" % sequence_first_run)
                iq_fd_simple.write("# Sequence index: %s\n" % sequence_index)
                iq_fd_simple.write("# Selected wavelength: %s\n" % wavelength[main_index])
                iq_fd_simple.write("# %-8s %-10s %-10s\n" % ("Q", "I(Q)", "dI(Q)"))

                for i in range(len(peaks)):
                    peak = peaks[i]
                    CropWorkspace(
                        InputWorkspace="USANS_detector",
                        OutputWorkspace="peak_detector",
                        XMin=peak[0],
                        XMax=peak[1],
                    )
                    StepScan(InputWorkspace="peak_detector", OutputWorkspace="scan_table")
                    ConvertTableToMatrixWorkspace(
                        InputWorkspace="scan_table",
                        ColumnX=scan_var,
                        ColumnY="Counts",
                        ColumnE="Error",
                        OutputWorkspace="USANS_scan_detector",
                    )
                    mtd["USANS_scan_detector"].getAxis(1).getUnit().setLabel("Counts", "Counts")
                    x_data = mtd["USANS_scan_detector"].readX(0)  # rotation angle (arcsec) of the analyzer
                    y_data = mtd["USANS_scan_detector"].readY(0)  # total counts at detector for each analyzer angle
                    e_data = mtd["USANS_scan_detector"].readE(0)

                    if i == 0:
                        file_path = os.path.join(outdir, "%s_detector_%s.txt" % (file_prefix, main_wl))
                        save_ascii("USANS_scan_detector", file_path, header=f"{scan_var}, COUNTS, ERROR")
                        # json_file_path = os.path.join(outdir, "%s_plot_data.json" % file_prefix)
                        # SavePlot1DAsJson(
                        #     InputWorkspace="USANS_scan_detector", JsonFilename=json_file_path, PlotName="main_output"
                        # )

                        try:
                            plot1d(
                                run_number,
                                [[x_data, y_data, e_data]],
                                instrument="USANS",
                                x_title=scan_var,
                                y_title="Counts",
                                y_log=True,
                            )
                        except:  # noqa E722
                            logger.error("Error calling plot1d, no image plotted")
                            traceback.print_exc()

                        # Save scan info to use for stitching later
                        record_run(
                            outdir,
                            sequence_first_run,
                            run_number,
                            file_path,
                            run_title,
                            sample_thickness(mtd["USANS"].getRun()),
                        )

                        for i_theta in range(len(x_data)):
                            q = (
                                2.0
                                * math.pi
                                * math.sin(x_data[i_theta] * math.pi / 180.0 / 3600.0)
                                / wavelength[main_index]
                            )
                            # if q<=0:
                            #    continue

                            # Write I(q) file
                            i_q = y_data[i_theta] / y_monitor[i_theta]
                            di_q = math.sqrt(
                                (e_data[i_theta] / y_monitor[i_theta]) ** 2
                                + y_data[i_theta] ** 2 / y_monitor[i_theta] ** 3
                            )
                            iq_fd_simple.write("%-10.6g %-10.6g %-10.6g\n" % (q, i_q, di_q))

                    else:
                        file_path = os.path.join(
                            outdir,
                            "%s_detector_scan_%s_peak_%s.txt" % (file_prefix, short_name, i),
                        )
                        save_ascii("USANS_scan_detector", file_path, header=f"{scan_var}, COUNTS, ERROR")
                        for i_theta in range(len(x_data)):
                            q = 2.0 * math.pi * math.sin(x_data[i_theta] * math.pi / 180.0 / 3600.0) / wavelength[i - 1]
                            # if q<=0:
                            #    continue

                            # Write complete I(q) file
                            i_q = y_data[i_theta] / y_monitor[i_theta]
                            di_q = math.sqrt(
                                (e_data[i_theta] / y_monitor[i_theta]) ** 2
                                + y_data[i_theta] ** 2 / y_monitor[i_theta] ** 3
                            )
                            if i_q > 0:
                                iq_fd.write(
                                    "%-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-10.6g %-5.4g\n"
                                    % (
                                        q,
                                        i_q,
                                        di_q,
                                        0,
                                        y_data[i_theta],
                                        e_data[i_theta],
                                        y_monitor[i_theta],
                                        wavelength[i - 1],
                                    )
                                )

                    CropWorkspace(
                        InputWorkspace="USANS_trans",
                        OutputWorkspace="peak_trans",
                        XMin=peak[0],
                        XMax=peak[1],
                    )
                    StepScan(InputWorkspace="peak_trans", OutputWorkspace="scan_table")
                    ConvertTableToMatrixWorkspace(
                        InputWorkspace="scan_table",
                        ColumnX=scan_var,
                        ColumnY="Counts",
                        OutputWorkspace="USANS_scan_trans",
                    )

                    if i == 0:
                        file_path = os.path.join(outdir, "%s_trans_%s.txt" % (file_prefix, main_wl))
                    else:
                        file_path = os.path.join(
                            outdir,
                            "%s_trans_scan_%s_peak_%s.txt" % (file_prefix, short_name, i),
                        )
                    save_ascii("USANS_scan_trans", file_path, header=f"{scan_var}, COUNTS, ERROR")

                iq_fd.close()
                iq_fd_simple.close()

    # Reduce all the sequences of the output directory
    sequences = collect_sequences(outdir)
    if not sequences:
        logger.warning(f"No sequence of runs in {outdir}: nothing to reduce")
        return
    setup_file = write_setup_file(build_setup(sequences), outdir)
    reduce_sequences(setup_file, outdir)


if __name__ == "__main__":
    main()
