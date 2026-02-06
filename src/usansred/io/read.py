# TODO: Modify to allow multiple backgrounds

import logging
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usansred.reduce import Experiment


def config_from_csv(csv_path: str) -> tuple[dict | None, list[dict]]:
    import csv

    from usansred.reduce import Sample

    background = None
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        csvReader = csv.reader(f, delimiter=",")

        for row in csvReader:
            # skip comment rows
            if row[0].startswith("#"):
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
                    # Check if this sample is the empty sample
                    sample["is_background"] = True
                    background = sample
                else:
                    samples.append(sample)

    return background, samples


def config_from_json(json_path: str) -> tuple[dict | None, list[dict]]:
    import json

    from usansred.reduce import Sample

    with open(json_path, "r") as f:
        data = json.load(f)

    bg = data.get("background")
    if bg:
        try:
            background = {
                "name": bg["name"],
                "start_scan_num": int(bg["start_scan_num"]),
                "num_of_scans": int(bg["num_of_scans"]),
                "thickness": float(bg["thickness"]),
                "is_background": True,
            }
        except Exception as e:  # noqa E722
            logging.info(f"Error parsing background {bg}: {e}")
            background = None
            traceback.print_exc()

    _samples: list[dict] = data.get("samples")
    if not _samples:
        logging.info("No samples found in the configuration.")
        return background, []

    samples = []
    for s in _samples:
        try:
            sample = {
                "name": s["name"],
                "start_scan_num": s["start_scan_num"],
                "num_of_scans": int(s["num_of_scans"]),
                "thickness": float(s["thickness"]),
                "exclude": s.get("exclude", []),
            }
            samples.append(sample)
        except Exception as e:  # noqa E722
            logging.info(f"Error parsing sample {s}: {e}")
            # traceback.print_exc()

    return background, samples


def read_config(file_path: str) -> tuple[dict | None, list[dict]]:
    """Wrapper function to read configuration from different file formats.

    Parameters
    ----------
    file_path: str
        Path to the reduction configuration file.

    Returns
    -------
    tuple[dict | None, list[dict]]
        A tuple containing the background configuration (or None) and a list of sample configurations.
    """
    import os

    _, ext = os.path.splitext(file_path)
    if ext.lower() == ".csv":
        return config_from_csv(file_path)
    elif ext.lower() == ".json":
        return config_from_json(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")
