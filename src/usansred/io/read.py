import json
import logging
from pathlib import Path

from pydantic import ValidationError

from usansred.model import ReductionConfig


def _format_validation_error(e: ValidationError) -> str:
    """Format a Pydantic validation error for setup-file users."""
    first = e.errors()[0]
    loc = ".".join(str(p) for p in first["loc"])
    prefix = f"{loc}: " if loc else ""
    return f"{prefix}{first['msg']}"


def config_from_csv(csv_path: str) -> ReductionConfig:
    """Reads the reduction configuration from a CSV file.

    Returns
    -------
    ReductionConfig
        Validated reduction configuration.
    """
    import csv

    background = None
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=",")

        for row in reader:
            # skip comments and empty rows
            if not row or row[0].startswith("#"):
                continue

            exclude_runs = []
            if len(row) == 6:
                exclude_runs = [int(x) for x in row[5].split(";")]

            try:
                sample = {
                    "name": row[1],
                    "start_scan_num": int(row[2]),
                    "num_of_scans": int(row[3]),
                    "thickness": float(row[4]),
                    "exclude": exclude_runs,
                }
            except Exception as e:  # noqa E722
                sample = None
                logging.info(f"Error parsing sample {row}: {e}")
                # traceback.print_exc()

            if sample is not None:
                if row[0] == "b":
                    background = sample
                else:
                    samples.append(sample)

    try:
        return ReductionConfig.model_validate({"background": background, "samples": samples})
    except ValidationError as e:
        raise ValueError(_format_validation_error(e)) from e


def config_from_json(json_path: str) -> ReductionConfig:
    """Reads the reduction configuration from a JSON file.

    Returns
    -------
    ReductionConfig
        Validated reduction configuration.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    try:
        return ReductionConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(_format_validation_error(e)) from e


def read_config(file_path: str) -> ReductionConfig:
    """Wrapper function to read configuration from different file formats.

    Parameters
    ----------
    file_path: str
        Path to the reduction configuration file.

    Returns
    -------
    ReductionConfig
        Validated reduction configuration.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return config_from_csv(file_path)
    elif ext == ".json":
        return config_from_json(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}. Valid formats are .csv and .json.")
