import json
import logging
import os
import random
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import numpy as np
import pytest

from usansred.reduce import Experiment
from usansred.reduce import main as reduce

### Helper functions for tests ###


def read_numbers_from_file(filename):
    """
    Read numbers from a file and return a list of lists, where each inner list contains the numbers from a line.
    """
    numbers_list = []
    with open(filename, "r") as file:
        for line in file:
            numbers = line.strip().split(",")
            numbers_list.append([float(num) for num in numbers if num])
    return numbers_list


def compare_lines(file1, file2, threshold=0.01):
    """Compare corresponding numbers in two files line by line."""
    numbers_list1 = read_numbers_from_file(file1)
    numbers_list2 = read_numbers_from_file(file2)

    for i, (line1, line2) in enumerate(zip(numbers_list1, numbers_list2), start=1):
        for num1, num2 in zip(line1, line2):
            try:
                relative_diff = abs(num1 - num2) / max(abs(num1), abs(num2))
            except ZeroDivisionError:
                pass
            if relative_diff > threshold:
                raise ValueError(f"Line {i}, Number {num1:.6f} differs significantly from {num2:.6f}")


def assert_reduction_log_files(output_dir: str | Path, measurements: list[tuple[str, str]]):
    """Assert per-measurement reduction log files exist and include completion messages.

    Parameters
    ----------
    output_dir : str | Path
        Directory containing the ``reduction_<name>.log`` files.
    measurements : list[tuple[str, str]]
        Pairs of ``(name, label_prefix)``, e.g. ``("EmptyPCell", "background")``.
    """
    output_path = Path(output_dir)
    for name, label_prefix in measurements:
        logfile = output_path / f"reduction_{name}.log"
        assert logfile.is_file()
        content = logfile.read_text(encoding="utf-8")
        assert f"Data reduction finished for {label_prefix} {name}." in content


### Tests ###


@pytest.mark.datarepo
@mock_patch("usansred.reduce.parse_args")
def test_main(mock_parse_args, data_server, tmp_path):
    # Setup mock objects
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = data_server.path_to("setup.json")
    mock_args.output = str(tmp_path)
    mock_parse_args.return_value = mock_args
    reduce()
    # compare the content of output files with files containing expected results
    goldendir = os.path.join(os.path.dirname(mock_args.path), "reduced")  # where the expected content resides
    for name in ["S115_pc3", "S115_dry", "EmptyPCell"]:
        file_suffixes = {
            "unscaled data": "_unscaled",
            "scaled data": "",
            "background subtracted": "_background_subtracted",
        }
        for suffix in file_suffixes.values():
            filename = f"UN_{name}_det_1{suffix}.txt"
            output, expected = os.path.join(tmp_path, filename), os.path.join(goldendir, filename)
            if os.path.exists(expected):
                assert os.path.exists(output), f"Missing expected output file: {output}"
                compare_lines(output, expected)

    # The background-subtracted file is written for the samples ...
    for name in ["S115_pc3", "S115_dry"]:
        assert os.path.exists(os.path.join(tmp_path, f"UN_{name}_det_1_background_subtracted.txt"))
    # ... but not for the background itself (nothing is subtracted from it)
    assert not os.path.exists(os.path.join(tmp_path, "UN_EmptyPCell_det_1_background_subtracted.txt"))

    assert_reduction_log_files(tmp_path, [("S115_pc3", "sample"), ("S115_dry", "sample"), ("EmptyPCell", "background")])


@mock_patch("usansred.reduce.parse_args")
def test_main_invalid_file(mock_parse_args):
    # Setup mock objects
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = "invalid_path.csv"
    mock_args.output = ""
    mock_parse_args.return_value = mock_args
    with pytest.raises(FileNotFoundError) as error:
        reduce()
    assert str(error.value) == f"The file path: {mock_args.path} does not exist"


@mock_patch("usansred.reduce.parse_args")
def test_main_logbin_deprecation_warning(mock_parse_args, caplog):
    """Passing the removed --logbin flag logs a deprecation warning and is otherwise ignored."""
    mock_args = MagicMock()
    mock_args.logbin = True
    mock_args.path = "invalid_path.csv"
    mock_args.output = ""
    mock_parse_args.return_value = mock_args

    with caplog.at_level(logging.WARNING):
        # main() emits the deprecation warning before it fails on the missing setup file
        with pytest.raises(FileNotFoundError):
            reduce()

    assert any("--logbin" in message and "deprecated" in message for message in caplog.messages)


@pytest.mark.datarepo
@mock_patch("usansred.reduce.parse_args")
def test_main_save_all_harmonics(mock_parse_args, data_server, tmp_path):
    source_config = Path(data_server.path_to("setup.json"))
    source_data_dir = source_config.parent
    reduction_data_dir = tmp_path / "data"
    reduction_data_dir.mkdir()

    for source_file in source_data_dir.glob("USANS_*.txt"):
        (reduction_data_dir / source_file.name).symlink_to(source_file)

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["save_all_harmonics"] = True
    config_file = reduction_data_dir / "setup.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")

    output_dir = tmp_path / "reduced"
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = str(config_file)
    mock_args.output = str(output_dir)
    mock_parse_args.return_value = mock_args

    reduce()

    # Higher harmonics are written flat in the output directory as ``_det_<n>``, alongside
    # the first-harmonic ``_det_1`` files; no per-bank subdirectories are created.
    for harmonic in range(2, 5):
        for name in ["EmptyPCell", "S115_dry", "S115_pc3"]:
            harmonic_file = output_dir / f"UN_{name}_det_{harmonic}_unscaled.txt"
            assert harmonic_file.is_file()
            assert harmonic_file.stat().st_size > 0
            assert len(harmonic_file.read_text(encoding="utf-8").splitlines()) == 65

            scaled_file = output_dir / f"UN_{name}_det_{harmonic}.txt"
            assert scaled_file.is_file()
            assert scaled_file.stat().st_size > 0

    assert not list(output_dir.glob("bank_*")), "per-bank subdirectories should no longer be created"

    assert_reduction_log_files(
        output_dir, [("S115_pc3", "sample"), ("S115_dry", "sample"), ("EmptyPCell", "background")]
    )


@pytest.mark.datarepo
def test_reduce_empty_cell(data_server, tmp_path):
    """Empty-cell reduction and subtraction in the absence of a background.

    This exercises the interpolation branch of ``Sample.subtract_background``.
    """
    config_file = data_server.path_to("setup-empty-cell.json")
    experiment = Experiment(config_file=config_file, output_dir=str(tmp_path))

    experiment.reduce()

    # The empty cell was reduced (first, in the absence of a background)
    assert_reduction_log_files(tmp_path, [("EmptyPCell", "empty cell"), ("S115_pc3", "sample"), ("S115_dry", "sample")])

    # No output files are written for the empty cell
    assert list(tmp_path.glob("UN_EmptyPCell*")) == []

    # The empty cell was subtracted from each sample
    for name in ["S115_pc3", "S115_dry"]:
        subtracted_file = tmp_path / f"UN_{name}_det_1_background_subtracted.txt"
        assert subtracted_file.is_file()
        assert subtracted_file.stat().st_size > 0
        logfile = (tmp_path / f"reduction_{name}.log").read_text(encoding="utf-8")
        assert f"Subtracted empty cell EmptyPCell from sample {name}" in logfile

    # The scaled data equals the golden result (produced with transmission == 1.0) with
    # intensities and errors divided by the transmission coefficient. The golden file was
    # generated from setup.json, whose sample definitions are identical; scaled data is
    # independent of binning and subtraction.
    goldendir = os.path.join(os.path.dirname(config_file), "reduced")
    for sample in experiment.samples:
        transmission = sample.transmission
        assert 0.0 < transmission < 1.0
        golden = np.array(read_numbers_from_file(os.path.join(goldendir, f"UN_{sample.name}_det_1.txt")))
        output = np.array(read_numbers_from_file(str(tmp_path / f"UN_{sample.name}_det_1.txt")))
        np.testing.assert_allclose(output[:, 0], golden[:, 0], rtol=1e-6)  # Q (1/angstrom) unchanged
        np.testing.assert_allclose(output[:, 1], golden[:, 1] / transmission, rtol=1e-6)
        np.testing.assert_allclose(output[:, 2], golden[:, 2] / transmission, rtol=1e-6)


@pytest.mark.datarepo
def test_sample_match_or_interpolate(data_server, tmp_path):
    # Get the testing data and temp output directory
    # Create new Experiment instance
    csvpath = data_server.path_to("setup.csv")
    tmpoutput = str(tmp_path)
    exp = Experiment(config_file=csvpath, output_dir=tmpoutput)

    # Genearte testing data
    qq = np.array([dd * 1e-5 for dd in range(1, 100)])
    ii = -np.log(qq) * 1e3
    bb = ii * 0.01

    # Generate a list of 100 random numbers
    ee = [random.random() for _ in range(1, 100)]

    sample_test = exp.samples[0]

    iibgmatched, eebgmatched = sample_test._match_or_interpolate(qq, qq, bb, ee)

    check = iibgmatched - bb == 0.0
    assert np.all(check), "Background interpolation calculation is not right in Sample._match_or_interpolate"


if __name__ == "__main__":
    pytest.main([__file__])
