"""summary.py: summary of the reduced data."""

import csv
import json
import logging
import os

import pandas

__author__ = "Yingrui Shang"
__copyright__ = "Copyright 2021, NSD, ORNL"
__all__ = ["generate_report"]

# separate logging in file and console
logging.basicConfig(filename="file.log", filemode="w", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

suffix = 0


def format_sheet_name(filename: str) -> str:
    """Reformat the file name to a valid sheetname in excel worksheet."""
    global suffix
    invalid = ["[", "]", ":", "*", "?", "/", "\\"]
    new_filename = filename
    for cc in invalid:
        new_filename = new_filename.replace(cc, "_")
    if len(new_filename) > 20:
        suffix += 1
        new_filename = new_filename[:20] + str(suffix)
    return new_filename


def get_filenames_from_samples(sample_name: str) -> list[str]:
    """Get a list of reduced file names based on a sample name."""
    if sample_name:
        return [
            "UN_" + sample_name + "_det_1.txt",
            "UN_" + sample_name + "_det_1_lb.txt",
            "UN_" + sample_name + "_det_1_lbs.txt",
            "UN_" + sample_name + "_det_1_unscaled.txt",
        ]
    else:
        raise ValueError(f"Sample name is empty or not valid: {sample_name}")


def generate_report(config_file_path: str, data_dir: str | None = None, output_dir: str | None = None):
    """Generate report from a reduction config file.

    Parameters
    ----------
    config_file_path : str
        Path to the configuration file (CSV or JSON).
    data_dir : str | None
        Directory where the reduced data are stored. If None, use the `reduced` dir in the config file directory.
    output_dir : str | None
        Where to save the report. If None, use `data_folder/reduced`.
    """
    # Validate inputs
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"The file path: {config_file_path} does not exist")

    _, ext = os.path.splitext(config_file_path)
    if ext.lower() not in [".csv", ".json"]:
        raise ValueError(f"Unsupported configuration file format: {ext}")

    # Set up directories
    if not data_dir:
        data_dir = os.path.dirname(config_file_path)

    if not output_dir:
        output_dir = os.path.join(data_dir, "reduced")

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    xlsx_writer = pandas.ExcelWriter(os.path.join(output_dir, "summary.xlsx"), engine="xlsxwriter")

    # Create a workbook and add chartsheets for different data types

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

    # Collect sample files from the config file
    if ext.lower() == ".json":
        with open(config_file_path) as json_file:
            data = json.load(json_file)
            sample_files = []

            background = data.get("background", {})
            background_name = background.get("name")
            if background_name:
                sample_files.extend(get_filenames_from_samples(background_name))

            for sample in data.get("samples", []):
                sample_name = sample.get("name", "")
                sample_files.extend(get_filenames_from_samples(sample_name))
    else:
        sample_files = []
        with open(config_file_path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in filter(lambda r: len(r) > 1 and not r[0].startswith("#"), csv_reader):
                sample_files.extend(get_filenames_from_samples(row[1]))

    # Process each sample file and add data to the corresponding charts
    for file in sample_files:
        fp = os.path.join(data_dir, file)

        if not os.path.exists(fp):
            logging.info(f"Sample {file} file path does not exist!")
            continue

        if os.stat(fp).st_size == 0:
            logging.warning(f"Sample file {file} is empty and will be skipped. ")
            continue

        logging.info(f"Reading sample file {file} to summary.xlsx")
        df = pandas.read_csv(
            fp,
            sep=",",
            # delim_whitespace=True,
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

        wn = format_sheet_name(file)
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
        if file.endswith("lbs.txt"):
            main_chart_subtracted.add_series(
                {
                    "name": f"{wn}",
                    "categories": f"={wn}!$D$2:$D$100",
                    "values": f"={wn}!$E$2:$E$100",
                }
            )
        elif file.endswith("lb.txt"):
            main_chart_log_binned.add_series(
                {
                    "name": f"{wn}",
                    "categories": f"={wn}!$D$2:$D$100",
                    "values": f"={wn}!$E$2:$E$100",
                }
            )
        elif file.endswith("unscaled.txt"):
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

        # print(f'={file}!$A:$B')
        worksheet.insert_chart("F1", chart)

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

    logging.info(f"complete processing {config_file_path}")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a report from a config file.")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("-d", "--data-folder", help="Folder where the reduced data are stored.", default=None)
    parser.add_argument(
        "-o",
        "--output",
        help="Where to save the report. If not provided, a 'reduced' folder will be created in the config file folder.",
        default=None,
    )
    args = parser.parse_args()

    generate_report(config_file_path=args.config_file, data_dir=args.data_folder, output_dir=args.output)
