"""summary.py: summary of the reduced data."""

# standard imports
import csv
import logging
import os
import pandas
import sys

__author__ = "Yingrui Shang"
__copyright__ = "Copyright 2021, NSD, ORNL"

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)
__all__ = ["reportFromCSV"]
suffix = 0


def formatSheetName(fn):
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


def reportFromCSV(csvFilePath, outputFolder=None):
    """generate report from csv file with a list
    csvFile - file location of csv file
    """
    if not os.path.exists(csvFilePath):
        logging.info(f"The file path: {csvFilePath} does not exist")
        raise

    dataFolderName = os.path.dirname(csvFilePath)

    if outputFolder is None:
        outputFolder = os.path.join(dataFolderName, "reduced")

    xlsxWriter = pandas.ExcelWriter(
        os.path.join(outputFolder, "summary.xlsx"), engine="xlsxwriter"
    )

    workbook = xlsxWriter.book
    # nbFormat = workbook.add_format({"bold": False})

    # Create a chart sheet for unscaled data
    chartsheetUnscaled = workbook.add_chartsheet("Unscaled")
    mainChartUnscaled = workbook.add_chart(
        {"type": "scatter", "subtype": "smooth_with_markers"}
    )

    # Create a chart sheet for original data
    chartsheetOrig = workbook.add_chartsheet("Original")
    mainChartOrig = workbook.add_chart(
        {"type": "scatter", "subtype": "smooth_with_markers"}
    )

    # log binned data
    chartsheetLogBinned = workbook.add_chartsheet("Log Binned")
    mainChartLogBinned = workbook.add_chart(
        {"type": "scatter", "subtype": "smooth_with_markers"}
    )

    # log binned data with background removed
    chartsheetSubtracted = workbook.add_chartsheet("BG Subtracted")
    mainChartSubtracted = workbook.add_chart(
        {"type": "scatter", "subtype": "smooth_with_markers"}
    )

    mainChartUnscaled.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    mainChartUnscaled.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    mainChartUnscaled.set_title({"name": "Unscaled Data"})

    mainChartOrig.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    mainChartOrig.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    mainChartOrig.set_title({"name": "Original Data"})

    mainChartLogBinned.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    mainChartLogBinned.set_y_axis({"name": "I (1/cn)", "log_base": 10})

    mainChartLogBinned.set_title({"name": "Log Binned"})

    mainChartSubtracted.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    mainChartSubtracted.set_y_axis({"name": "I (1/cn)", "log_base": 10})
    mainChartSubtracted.set_title({"name": "Background Subtracted"})

    with open(csvFilePath, newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")

        for row in csvReader:
            for fn in getFileNamesFromSamples(row[1]):
                ffp = os.path.join(outputFolder, fn)
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
                                df[
                                    (df["Q(1/A)"] > 0)
                                    & (df["I(1/cm)"] > 0)
                                    & (df["dI(1/cm)"] > 0)
                                ],
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

                        wn = formatSheetName(fn)
                        df.to_excel(xlsxWriter, sheet_name=wn, index=False)

                        worksheet = xlsxWriter.sheets[wn]
                        # worksheet.set_column('A:A', 12, nbFormat)

                        chart = workbook.add_chart(
                            {"type": "scatter", "subtype": "smooth_with_markers"}
                        )

                        chart.add_series(
                            {
                                "name": f"{wn}",
                                "categories": f"={wn}!$D$2:$D$100",
                                "values": f"={wn}!$E$2:$E$100",
                            }
                        )

                        # Add data series to the main chartsheets
                        if fn.endswith("lbs.txt"):
                            mainChartSubtracted.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        elif fn.endswith("lb.txt"):
                            mainChartLogBinned.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        elif fn.endswith("unscaled.txt"):
                            mainChartUnscaled.add_series(
                                {
                                    "name": f"{wn}",
                                    "categories": f"={wn}!$D$2:$D$100",
                                    "values": f"={wn}!$E$2:$E$100",
                                }
                            )
                        else:
                            mainChartOrig.add_series(
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
    if mainChartUnscaled.series:
        chartsheetUnscaled.set_chart(mainChartUnscaled)

    if mainChartOrig.series:
        chartsheetOrig.set_chart(mainChartOrig)

    if mainChartLogBinned.series:
        chartsheetLogBinned.set_chart(mainChartLogBinned)

    if mainChartSubtracted.series:
        chartsheetSubtracted.set_chart(mainChartSubtracted)

        chartsheetSubtracted.activate()
    # workbook.close()
    xlsxWriter.close()

    logging.info(f"complete processing {csvFilePath}")
    return


def getFileNamesFromSamples(sampleName):
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
            "UN_" + sampleName + "_det_1_unscaled.txt",
        )
    else:
        logging.info("Sample name is empty or not valid")
        raise

    return


if __name__ == "__main__":
    reportFromCSV(sys.argv[1])
