"""summary.py: summary of the reduced data."""

import logging
import os

import pandas

from usansred.io.filenames import reduced_data_filenames
from usansred.io.read import read_config
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
        If ``sample_name`` is empty or None, or contains a path separator, ``/`` or ``\\``.
    """
    if sample_name:
        return reduced_data_filenames(sample_name, num_of_harmonics=1)
    else:
        raise ValueError(f"Sample name is empty or not valid: {sample_name}")


# Chart sheets of the summary report, in the order of the workbook, with the title of their chart
CHART_SHEETS = {
    "Unscaled": "Unscaled Data",
    "Original": "Original Data",
    "BG Subtracted": "Background Subtracted",
}


def chart_sheet_of(filename: str) -> str:
    """Name of the chart sheet that shows the curve of a reduced data file.

    Parameters
    ----------
    filename : str
        Name of the reduced data file.

    Returns
    -------
    str
        One of the keys of ``CHART_SHEETS``.
    """
    # Match the file-name suffixes, not a substring, since the measurement name itself
    # may contain "background_subtracted"
    if filename.endswith(("_background_subtracted.txt", "_background_subtracted_stitched.txt")):
        return "BG Subtracted"
    if filename.endswith("unscaled.txt"):
        return "Unscaled"
    return "Original"


def generate_report(
    config_file_path: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    include_unscaled: bool = True,
):
    """Generate report from a reduction config file.

    The report, ``summary.xlsx``, has a worksheet for each reduced data file found, and a chart sheet
    for each kind of file (see ``CHART_SHEETS``) that holds at least one curve. It is written to a
    temporary file in ``output_dir``, then renamed, so that a reader never sees it partly written.

    Parameters
    ----------
    config_file_path : str
        Path to the configuration file (CSV or JSON).
    data_dir : str | None
        Directory where the reduced data are stored. If None, use the `reduced` dir in the config file directory.
    output_dir : str | None
        Where to save the report. If None, use `data_folder/reduced`.
    include_unscaled : bool
        Whether the report includes the ``_unscaled`` files and the ``Unscaled`` chart sheet.
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

    # Collect the files of the background and the samples, before creating summary.xlsx. The setup
    # file is read as for the reduction, so the same rows are skipped and the same names rejected
    config = read_config(config_file_path)
    names = ([config.background.name] if config.background is not None else []) + [s.name for s in config.samples]
    sample_files = [filename for name in names for filename in get_filenames_from_samples(name)]
    if not include_unscaled:
        sample_files = [filename for filename in sample_files if chart_sheet_of(filename) != "Unscaled"]

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # The files found, and the chart sheets that hold their curves, are known before the workbook is
    # created, so that only the chart sheets with a curve are created, in the order of CHART_SHEETS
    found_files = []
    for file in sample_files:
        fp = os.path.join(output_dir, file)

        if not os.path.exists(fp):
            logger.info(f"Sample {file} file path does not exist!")
            continue

        if os.stat(fp).st_size == 0:
            logger.warning(f"Sample file {file} is empty and will be skipped. ")
            continue

        found_files.append(file)

    report_file = os.path.join(output_dir, "summary.xlsx")
    # The process ID keeps the temporary files of two overlapping calls apart; the extension must be
    # that of the engine
    temporary_file = os.path.join(output_dir, f"summary.{os.getpid()}.tmp.xlsx")
    try:
        _write_workbook(temporary_file, found_files, output_dir)
        os.replace(temporary_file, report_file)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)

    logger.info(f"complete processing {config_file_path}")
    return


def _write_workbook(path: str, files: list[str], data_dir: str) -> None:
    """Write the workbook of the summary report.

    Parameters
    ----------
    path : str
        Path of the workbook.
    files : list[str]
        Names of the reduced data files, which exist and are not empty.
    data_dir : str
        Directory of the reduced data files.
    """
    xlsx_writer = pandas.ExcelWriter(path, engine="xlsxwriter")
    workbook = xlsx_writer.book

    # Create the chart sheets that hold at least one curve
    needed = {chart_sheet_of(file) for file in files}
    chartsheets = {}
    main_charts = {}
    for name, title in CHART_SHEETS.items():
        if name not in needed:
            continue
        chartsheets[name] = workbook.add_chartsheet(name)
        main_chart = workbook.add_chart({"type": "scatter", "subtype": "smooth_with_markers"})
        main_chart.set_x_axis({"name": "Q (1/A)", "log_base": 10})
        main_chart.set_y_axis({"name": "I (1/cm)", "log_base": 10})
        main_chart.set_title({"name": title})
        main_charts[name] = main_chart

    # Process each sample file and add data to the corresponding charts
    used_sheet_names = {name.lower() for name in CHART_SHEETS}  # the chart sheets
    for file in files:
        fp = os.path.join(data_dir, file)

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
        main_charts[chart_sheet_of(file)].add_series(dict(series))

        chart.set_x_axis({"name": [wn, 0, 0], "log_base": 10})
        chart.set_y_axis({"name": [wn, 0, 1], "log_base": 10})
        worksheet.insert_chart("F1", chart)

    for name, chartsheet in chartsheets.items():
        chartsheet.set_chart(main_charts[name])
    if "BG Subtracted" in chartsheets:
        chartsheets["BG Subtracted"].activate()

    xlsx_writer.close()


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
