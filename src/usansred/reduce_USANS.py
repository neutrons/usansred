# standard imports
import csv
import json
import logging
import math
import os
import sys
import traceback
import warnings

# third-party imports
# from debugpy.common.log import newline
from mantid.simpleapi import (
    mtd,
    ConvertTableToMatrixWorkspace,
    CropWorkspace,
    LoadEventNexus,
    SaveAscii,
    StepScan,
    SumSpectra,
    Rebin,
)
from matplotlib import use
import numpy as np

# usansred imports
from usansred import reduce
from usansred.summary import reportFromCSV

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
                logging.error("Could not read json file: creating a new one")

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


def main():
    # check number of arguments
    if len(sys.argv) != 3:
        print("autoreduction code requires a filename and an output directory")
        sys.exit(1)

    if not (os.path.isfile(sys.argv[1])):
        print("data file ", sys.argv[1], " not found")
        sys.exit()
    else:
        filename = sys.argv[1]
        outdir = sys.argv[2]

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
    roi_min = (
        mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Min").value[-1]
    )
    # roi_step = mtd["USANS"].getRun().getProperty("BL1A:Det:N1:Det1:TOF:ROI:1:Size").value[-1]

    # Reference to the item in the wavelength array
    main_index = 1
    for i in range(1, 8):
        lower_bound = (
            mtd["USANS"]
            .getRun()
            .getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Min" % i)
            .value[-1]
        )
        tof_step = (
            mtd["USANS"]
            .getRun()
            .getProperty("BL1A:Det:N1:Det1:TOF:ROI:%s:Size" % i)
            .value[-1]
        )
        if i > 1 and lower_bound == roi_min:
            main_index = i - 2
        peaks.append([lower_bound * 1000.0, (lower_bound + tof_step) * 1000.0])
    main_wl = wavelength[main_index]

    # Produce ASCII data
    Rebin(InputWorkspace="USANS", Params="0,10,17000", OutputWorkspace="USANS")
    SumSpectra(InputWorkspace="USANS", OutputWorkspace="summed")
    file_path = os.path.join(outdir, "%s_detector_trans.txt" % file_prefix)
    SaveAscii(InputWorkspace="summed", Filename=file_path, WriteSpectrumID=False)

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=0,
        EndWorkspaceIndex=1023,
        OutputWorkspace="USANS_detector",
    )
    SumSpectra(InputWorkspace="USANS_detector", OutputWorkspace="summed")
    file_path = os.path.join(outdir, "%s_detector.txt" % file_prefix)
    SaveAscii(InputWorkspace="summed", Filename=file_path, WriteSpectrumID=False)

    CropWorkspace(
        InputWorkspace="USANS",
        StartWorkspaceIndex=1024,
        EndWorkspaceIndex=2047,
        OutputWorkspace="USANS_trans",
    )
    SumSpectra(InputWorkspace="USANS_trans", OutputWorkspace="summed")
    file_path = os.path.join(outdir, "%s_trans.txt" % file_prefix)
    SaveAscii(InputWorkspace="summed", Filename=file_path, WriteSpectrumID=False)

    if load_monitors:
        Rebin(
            InputWorkspace="USANS_monitors",
            Params="0,10,17000",
            OutputWorkspace="USANS_monitors",
        )
        file_path = os.path.join(outdir, "%s_monitor.txt" % file_prefix)
        SaveAscii(
            InputWorkspace="USANS_monitors", Filename=file_path, WriteSpectrumID=False
        )

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
                    file_path = os.path.join(
                        outdir, "%s_monitor_scan_%s.txt" % (file_prefix, short_name)
                    )
                    SaveAscii(
                        InputWorkspace="USANS_scan_monitor",
                        Filename=file_path,
                        WriteSpectrumID=False,
                    )
                    y_monitor = mtd["USANS_scan_monitor"].readY(0)

                iq_file_path_simple = os.path.join(
                    outdir, "%s_iq_%s_%s.txt" % (file_prefix, short_name, main_wl)
                )
                iq_fd_simple = open(iq_file_path_simple, "w")

                iq_file_path = os.path.join(
                    outdir, "%s_iq_%s.txt" % (file_prefix, short_name)
                )
                iq_fd = open(iq_file_path, "w")

                start_time = mtd["USANS"].getRun().getProperty("start_time").value
                experiment = (
                    mtd["USANS"].getRun().getProperty("experiment_identifier").value
                )
                run_number = mtd["USANS"].getRun().getProperty("run_number").value
                run_title = mtd["USANS"].getRun().getProperty("run_title").value
                sequence_first_run = (
                    mtd["USANS"]
                    .getRun()
                    .getProperty("BL1A:CS:Scan:USANS:FirstRun")
                    .value[-1]
                )
                sequence_index = (
                    mtd["USANS"]
                    .getRun()
                    .getProperty("BL1A:CS:Scan:USANS:Index")
                    .value[-1]
                )
                meta_wavelength = (
                    mtd["USANS"]
                    .getRun()
                    .getProperty("BL1A:CS:Scan:USANS:Wavelength")
                    .value[-1]
                )
                print("Wavelength: %s [%s]" % (wavelength[main_index], meta_wavelength))

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
                iq_fd_simple.write(
                    "# Experiment %s Run %s\n" % (experiment, run_number)
                )
                iq_fd_simple.write("# Run start time: %s\n" % start_time)
                iq_fd_simple.write("# Title: %s\n" % run_title)
                iq_fd_simple.write("# Sequence ID: %s\n" % sequence_first_run)
                iq_fd_simple.write("# Sequence index: %s\n" % sequence_index)
                iq_fd_simple.write(
                    "# Selected wavelength: %s\n" % wavelength[main_index]
                )
                iq_fd_simple.write("# %-8s %-10s %-10s\n" % ("Q", "I(Q)", "dI(Q)"))

                for i in range(len(peaks)):
                    peak = peaks[i]
                    CropWorkspace(
                        InputWorkspace="USANS_detector",
                        OutputWorkspace="peak_detector",
                        XMin=peak[0],
                        XMax=peak[1],
                    )
                    StepScan(
                        InputWorkspace="peak_detector", OutputWorkspace="scan_table"
                    )
                    ConvertTableToMatrixWorkspace(
                        InputWorkspace="scan_table",
                        ColumnX=scan_var,
                        ColumnY="Counts",
                        ColumnE="Error",
                        OutputWorkspace="USANS_scan_detector",
                    )
                    mtd["USANS_scan_detector"].getAxis(1).getUnit().setLabel(
                        "Counts", "Counts"
                    )
                    x_data = mtd["USANS_scan_detector"].readX(0)
                    y_data = mtd["USANS_scan_detector"].readY(0)
                    e_data = mtd["USANS_scan_detector"].readE(0)

                    if i == 0:
                        file_path = os.path.join(
                            outdir, "%s_detector_%s.txt" % (file_prefix, main_wl)
                        )
                        SaveAscii(
                            InputWorkspace="USANS_scan_detector",
                            Filename=file_path,
                            WriteSpectrumID=False,
                        )
                        # json_file_path = os.path.join(outdir, "%s_plot_data.json" % file_prefix)
                        # SavePlot1DAsJson(InputWorkspace="USANS_scan_detector", JsonFilename=json_file_path, PlotName="main_output")

                        try:
                            from postprocessing.publish_plot import plot1d
                        except ImportError:
                            try:
                                from finddata.publish_plot import plot1d
                            except:  # noqa E722
                                logging.error(
                                    "Cannot import postprocessing or finddata."
                                )

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
                            logging.error("Error calling plot1d, no image plotted")
                            traceback.print_exc()

                        # Save scan info to use for stitching later
                        update_sequence_info(
                            os.path.join(outdir, "scan_%s.json" % sequence_first_run),
                            {run_number: {"iq": file_path}},
                        )
                        update_sequence_info(
                            os.path.join(outdir, "sample_%s.json" % sequence_first_run),
                            {"title": run_title, "background": 0, "thickness": 0.1},
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
                            iq_fd_simple.write(
                                "%-10.6g %-10.6g %-10.6g\n" % (q, i_q, di_q)
                            )

                    else:
                        file_path = os.path.join(
                            outdir,
                            "%s_detector_scan_%s_peak_%s.txt"
                            % (file_prefix, short_name, i),
                        )
                        SaveAscii(
                            InputWorkspace="USANS_scan_detector",
                            Filename=file_path,
                            WriteSpectrumID=False,
                        )
                        for i_theta in range(len(x_data)):
                            q = (
                                2.0
                                * math.pi
                                * math.sin(x_data[i_theta] * math.pi / 180.0 / 3600.0)
                                / wavelength[i - 1]
                            )
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
                        file_path = os.path.join(
                            outdir, "%s_trans_%s.txt" % (file_prefix, main_wl)
                        )
                        SaveAscii(
                            InputWorkspace="USANS_scan_trans",
                            Filename=file_path,
                            WriteSpectrumID=False,
                        )
                    else:
                        file_path = os.path.join(
                            outdir,
                            "%s_trans_scan_%s_peak_%s.txt"
                            % (file_prefix, short_name, i),
                        )
                        SaveAscii(
                            InputWorkspace="USANS_scan_trans",
                            Filename=file_path,
                            WriteSpectrumID=False,
                        )

                iq_fd.close()
                iq_fd_simple.close()

    # list all the sequence files
    json_files = [
        pos_json
        for pos_json in os.listdir(outdir)
        if pos_json.endswith(".json") and pos_json.startswith("scan")
    ]
    json_files.sort()
    json_files = [os.path.join(outdir, ff) for ff in json_files]

    sample_json_files = [
        pos_json
        for pos_json in os.listdir(outdir)
        if pos_json.endswith(".json") and pos_json.startswith("sample")
    ]
    sample_json_files.sort()
    sample_json_files = [os.path.join(outdir, ff) for ff in json_files]

    # construct the csv file
    seq_len = []
    seq_thickness = []
    seq_titles = []
    seq_start_runnums = []
    seq_background_flag = []

    for idx, ff in enumerate(json_files):
        seq_dict = get_sequence_info(ff)
        try:
            sample_dict = get_sequence_info(sample_json_files[idx])
        except:  # noqa E722
            sample_dict = {}

        seq_len.append(len(seq_dict) - 1)

        if "title" in sample_dict:
            seq_titles.append(sample_dict["title"])
        else:
            seq_titles.append("Sample" + str(idx))
        if "thickness" in sample_dict:
            seq_thickness.append(int(sample_dict["thickness"]))
        else:
            seq_thickness.append(0.1)
        if "background_flag" in sample_dict:
            seq_background_flag.append(int(sample_dict["background"]))
        else:
            seq_background_flag.append(0)

        seq_start_runnums.append(list(seq_dict)[0])
    # in case there is no background designated, assign the first sample
    if sum(seq_background_flag) == 0.0:
        seq_background_flag[0] = 1
    else:
        pass

    autocsvfile = os.path.join(outdir, "auto.csv")

    with open(autocsvfile, "w", newline="") as autocsv:
        datawriter = csv.writer(autocsv, delimiter=",")

        for flag, title, first_run, ll, thickness in zip(
            seq_background_flag, seq_start_runnums, seq_titles, seq_len, seq_thickness
        ):
            flag = "b" if flag == 1 else "s"
            rr = [str(col) for col in [flag, first_run, title, ll, thickness]]
            datawriter.writerow(rr)

    sys.path.insert(2, "/SNS/USANS/shared/autoreduce/usans-reduction")

    autodir = os.path.join(outdir, "auto")
    if not os.path.exists(autodir):
        os.makedirs(autodir)

    exp = reduce.Experiment(autocsvfile)
    exp.reduce(outputFolder=autodir)

    reportFromCSV(autocsvfile, exp.outputFolder)

    # seq_dict = get_sequence_info(os.path.join(outdir, "scan_%s.json" % sequence_first_run))


if __name__ == "__main__":
    main()
