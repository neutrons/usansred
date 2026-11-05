"""summary.py: summary of the reduced data."""

import csv
import json
import logging
import os

import pandas

from usansred.io.filenames import reduced_data_filenames
from usansred.utils.logging import get_logger

__author__ = "Yingrui Shang"
__copyright__ = "Copyright 2021, NSD, ORNL"
__all__ = ["generate_report"]

logger = get_logger(__name__)

# Excel limits worksheet names to 31 characters
SHEET_NAME_MAX_LENGTH = 31

# Characters that Excel does not allow in worksheet names
INVALID_SHEET_NAME_CHARACTERS = ["[", "]", ":", "*", "?", "/", "\\"]

# Substitutions applied in order, one at a time, to the part of a sheet name that follows the
# measurement name, until the sheet name fits in SHEET_NAME_MAX_LENGTH characters
SHEET_NAME_SHORTENING_RULES = [
    ("background", "bkgr"),
    ("stitched", "stitch"),
    ("_det", ""),
    ("subtracted", "sub"),
]


def format_sheet_name(filename: str, used_names: set[str] | None = None) -> str:
    """Derive a valid Excel worksheet name from the name of a reduced data file.

    1. The ``.txt`` extension is removed.
    2. Characters that Excel does not allow in worksheet names are replaced by ``_``, as is an
       apostrophe at the start or end of the name.
    3. While the name is longer than 31 characters, the substitutions of
       ``SHEET_NAME_SHORTENING_RULES`` are applied in order, one at a time: ``background`` to
       ``bkgr``, ``stitched`` to ``stitch``, ``_det`` dropped, and ``subtracted`` to ``sub``.
       They apply only to the part of the name from the last ``_det_`` onward, so that the
       measurement name is never altered.
    4. If the name is still too long, or already in ``used_names``, it is cut and a counter is
       appended, so that it ends at no more than 31 characters and is unique. The counter is
       the smallest integer, starting at 1, that gives a name not in ``used_names``, so the
       result depends only on ``filename`` and ``used_names``.

    For example, ``S115_pc3_det_background_subtracted_stitched.txt`` becomes
    ``S115_pc3_bkgr_subtracted_stitch``.

    Parameters
    ----------
    filename : str
        Name of a reduced data file, e.g. ``S115_pc3_det_1.txt``.
    used_names : set[str] | None
        Worksheet names already in the workbook, in lower case since Excel compares worksheet
        names without regard to case. The returned name is added to it. If None, uniqueness is
        not checked.

    Returns
    -------
    str
        The worksheet name.
    """
    name = filename.removesuffix(".txt")
    for character in INVALID_SHEET_NAME_CHARACTERS:
        name = name.replace(character, "_")
    # Excel does not allow a worksheet name to start or end with an apostrophe
    if name.startswith("'"):
        name = "_" + name[1:]
    if name.endswith("'"):
        name = name[:-1] + "_"

    start = name.rfind("_det_")
    if start >= 0:
        measurement, tail = name[:start], name[start:]
        for old, new in SHEET_NAME_SHORTENING_RULES:
            if len(measurement) + len(tail) <= SHEET_NAME_MAX_LENGTH:
                break
            tail = tail.replace(old, new)
        name = measurement + tail

    def is_taken(candidate: str) -> bool:
        return used_names is not None and candidate.lower() in used_names

    # The counter starts at 1 for every name, so that the same files always get the same names
    if len(name) > SHEET_NAME_MAX_LENGTH or is_taken(name):
        base = name
        counter = 1
        while True:
            name = base[: SHEET_NAME_MAX_LENGTH - len(str(counter))] + str(counter)
            if not is_taken(name):
                break
            counter += 1

    if used_names is not None:
        used_names.add(name.lower())
    return name


def get_filenames_from_samples(sample_name: str) -> list[str]:
    """Names of the reduced data files of a measurement that go into the summary report.

    These are the first-order files and the files with all orders stitched together, as listed
    by ``reduced_data_filenames`` for a single harmonic. The stitched files exist only when
    ``save_all_harmonics`` is enabled and at least orders 1 and 2 could be stitched, and the
    background-subtracted files only when a subtraction occurred; ``generate_report`` skips the
    files that do not exist. Files of an earlier reduction into the same directory are removed
    by ``Sample.dump_reduced_data_to_csv``, so the files that exist are those of the latest
    reduction.

    Parameters
    ----------
    sample_name : str
        Name of the measurement.

    Returns
    -------
    list[str]
        The file names.

    Raises
    ------
    ValueError
        If ``sample_name`` is empty or None.
    """
    if sample_name:
        return reduced_data_filenames(sample_name, num_of_harmonics=1)
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

    # data with background removed
    chartsheet_subtracted = workbook.add_chartsheet("BG Subtracted")
    main_chart_subtracted = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

    main_chart_unscaled.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_unscaled.set_y_axis({"name": "I (1/cm)", "log_base": 10})
    main_chart_unscaled.set_title({"name": "Unscaled Data"})

    main_chart_orig.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_orig.set_y_axis({"name": "I (1/cm)", "log_base": 10})
    main_chart_orig.set_title({"name": "Original Data"})

    main_chart_subtracted.set_x_axis({"name": "Q (1/A)", "log_base": 10})

    main_chart_subtracted.set_y_axis({"name": "I (1/cm)", "log_base": 10})
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
    used_sheet_names = {"unscaled", "original", "bg subtracted"}  # the chart sheets
    for file in sample_files:
        fp = os.path.join(output_dir, file)

        if not os.path.exists(fp):
            logger.info(f"Sample {file} file path does not exist!")
            continue

        if os.stat(fp).st_size == 0:
            logger.warning(f"Sample file {file} is empty and will be skipped. ")
            continue

        logger.info(f"Reading sample file {file} to summary.xlsx")
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

        wn = format_sheet_name(file, used_sheet_names)
        df.to_excel(xlsx_writer, sheet_name=wn, index=False)

        worksheet = xlsx_writer.sheets[wn]
        # worksheet.set_column('A:A', 12, nbFormat)

        chart = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})

        # Plot the positive Q (column D) and I (column E) values. Row 0 holds the column names,
        # and rows 1 to len(df) the data. The list form of the ranges quotes the sheet name.
        series = {
            "name": file.removesuffix(".txt"),
            "categories": [wn, 1, 3, len(df), 3],
            "values": [wn, 1, 4, len(df), 4],
        }
        if file.endswith("stitched.txt"):
            # Points of different orders are interleaved in the stitched profile, so a line
            # through them would zigzag wherever the orders scatter
            series["line"] = {"none": True}
        chart.add_series(series)

        # Add data series to the main chartsheets
        # Match the file-name suffixes, not a substring, since the measurement name itself
        # may contain "background_subtracted"
        if file.endswith(("_background_subtracted.txt", "_background_subtracted_stitched.txt")):
            main_chart_subtracted.add_series(dict(series))
        elif file.endswith("unscaled.txt"):
            main_chart_unscaled.add_series(dict(series))
        else:
            main_chart_orig.add_series(dict(series))

        chart.set_x_axis({"name": [wn, 0, 0], "log_base": 10})
        chart.set_y_axis({"name": [wn, 0, 1], "log_base": 10})
        worksheet.insert_chart("F1", chart)

    if main_chart_unscaled.series:
        chartsheet_unscaled.set_chart(main_chart_unscaled)

    if main_chart_orig.series:
        chartsheet_orig.set_chart(main_chart_orig)

    if main_chart_subtracted.series:
        chartsheet_subtracted.set_chart(main_chart_subtracted)

        chartsheet_subtracted.activate()

    # workbook.close()
    xlsx_writer.close()

    logger.info(f"complete processing {config_file_path}")
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
