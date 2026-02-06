"""summary.py: summary of the reduced data."""

import csv
import logging
import os
import sys

import pandas

__author__ = "Yingrui Shang"
__copyright__ = "Copyright 2021, NSD, ORNL"
__all__ = ["report_from_csv"]

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

suffix = 0


def format_sheet_name(fn):
    """
    Reformat the file name to a valid sheetname in excel worksheet
    """
    global suffix
    invalid = ["[", "]", ":", "*", "?", "/", "\\"]
    wn = fn
    for cc in invalid:
        wn = wn.replace(cc, "_")
    if len(wn) > 20:
        suffix += 1
        wn = wn[:20] + str(suffix)
    return wn


def get_filenames_from_samples(sample_name):
    """
    Get the reduced file names from sample name
    sample_name - sample name
    return - a list of reduced file names associated with this sample

    """
    if sample_name:
        return (
            "UN_" + sample_name + "_det_1.txt",
            "UN_" + sample_name + "_det_1_lb.txt",
            "UN_" + sample_name + "_det_1_lbs.txt",
            "UN_" + sample_name + "_det_1_unscaled.txt",
        )
    else:
        logging.info("Sample name is empty or not valid")
        raise

    return


def report_from_csv(csv_file_path, output_folder=None):
    """generate report from csv file with a list
    csvFile - file location of csv file
    """
    if not os.path.exists(csv_file_path):
        logging.info(f"The file path: {csv_file_path} does not exist")
        raise

    data_folder_name = os.path.dirname(csv_file_path)

    if output_folder is None:
        output_folder = os.path.join(data_folder_name, "reduced")

    xlsx_writer = pandas.ExcelWriter(os.path.join(output_folder, "summary.xlsx"), engine="xlsxwriter")

    workbook = xlsx_writer.book
    # nbFormat = workbook.add_format({"bold": False})

    # Create a chart sheet for unscaled data
    chartsheet_unscaled = workbook.add_chartsheet("Unscaled")
    main_chart_unscaled = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

    # Create a chart sheet for original data
    chartsheet_orig = workbook.add_chartsheet("Original")
    main_chart_orig = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

    # log binned data
    chartsheet_log_binned = workbook.add_chartsheet("Log Binned")
    main_chart_log_binned = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

    # log binned data with background removed
    chartsheet_subtracted = workbook.add_chartsheet("BG Subtracted")
    main_chart_subtracted = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

    main_chart_unscaled.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_unscaled.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    main_chart_unscaled.set_title({"name": "Unscaled Data"})

    main_chart_orig.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_orig.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    main_chart_orig.set_title({"name": "Original Data"})

    main_chart_log_binned.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_log_binned.set_y_axis({"name": "I (1/cn)", "log_base": 10})

    main_chart_log_binned.set_title({"name": "Log Binned"})

    main_chart_subtracted.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_subtracted.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    main_chart_subtracted.set_title({"name": "Background Subtracted"})

    with open(csv_file_path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            for fn in get_filenames_from_samples(row[1]):
                ffp = os.path.join(output_folder, fn)
                if os.path.exists(ffp):
                    # detect the separator
                    # dialect = csv.Sniffer().sniff(ffp)

                    if os.stat(ffp).st_size == 0:
                        logging.warn(f"{fn} is empty and will be skipped. ")
                    else:
                        logging.info(f"Reading {fn} to summary.xlsx")
                        df = pandas.read_csv(
                            ffp,
                            # delim_whitespace=True,
                            sep=",",
                            # index_col = 0
                            names=["Q(1/A)", "I(1/cm)", "dI(1/cm)"],
                            index_col=False,
                        )

                        # drop all nonpositive values for log-log ploting
                        # df = df.assign(F = (df["Q(1/A)"] > 0) & (df["I(1/cm)"] > 0) )

                        # Append new columns with non zero values
                        df = pandas.concat(
                            [
                                df,
                                df[(df["Q(1/A)"] > 0) & (df["I(1/cm)"] > 0) & (df["dI(1/cm)"] > 0)],
                            ],
                            ignore_index=False,
                            axis=1,
                        )

                        # df.reset_index(drop=True, inplace=True)
                        cnames = [
                            "Q(1/A)",
                            "I(1/cm)",
                            "dI(1/cm)",
                            "Q(1/A)_positive",
                            "I(1/cm)_positive",
                            "dI(1/cm)_positive",
                        ]
                        df.columns = cnames

                        wn = format_sheet_name(fn)
                        df.to_excel(xlsx_writer, sheet_name=wn, index=False)

                        worksheet = xlsx_writer.sheets[wn]
                        # worksheet.set_column('A:A', 12, nbFormat)

                        chart = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

                        chart.add_series(
                            {
                                "name": f"{wn}",
                                "categories": f"={wn}!$D$2:$D$100",
                                "values": f"={wn}!$E$2:$E$100",
                            }
                        )

                        # Add data series to the main chartsheets
                        if fn.endswith("lbs.txt"):
                            main_chart_subtracted.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        elif fn.endswith("lb.txt"):
                            main_chart_log_binned.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        elif fn.endswith("unscaled.txt"):
                            main_chart_unscaled.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        else:
                            main_chart_orig.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )

                        chart.set_x_axis({"name": f"={wn}!$A$1", "log_base": 10})

                        chart.set_y_axis({"name": f"={wn}!$B$1", "log_base": 10})

                        # print(f'={fn}!$A:$B')
                        worksheet.insert_chart("F1", chart)

                else:
                    logging.info(f"{fn} file path does not exist!")
    if main_chart_unscaled.series:
        chartsheet_unscaled.set_chart(main_chart_unscaled)

    if main_chart_orig.series:
        chartsheet_orig.set_chart(main_chart_orig)

    if main_chart_log_binned.series:
        chartsheet_log_binned.set_chart(main_chart_log_binned)

    if main_chart_subtracted.series:
        chartsheet_subtracted.set_chart(main_chart_subtracted)

        chartsheet_subtracted.activate()
    # workbook.close()
    xlsx_writer.close()

    logging.info(f"complete processing {csv_file_path}")
    return


if __name__ == "__main__":
    report_from_csv(sys.argv[1])
