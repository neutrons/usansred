import json
import logging
import os
import random
import re
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import numpy as np
import pytest

from usansred.io.filenames import LEGACY_PREFIX
from usansred.reduce import Experiment, unscattered_beam_q_cutoff
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


def assert_no_legacy_names(output_dir: str | Path):
    """Assert that no output file carries the ``UN_`` prefix of usansred up to version 1.1.0."""
    legacy = sorted(path.name for path in Path(output_dir).glob(f"{LEGACY_PREFIX}*"))
    assert legacy == [], f"output files with the legacy prefix: {legacy}"


def stitched_scale_factors(log: str, label: str, name: str) -> tuple[list[int], list[float]]:
    """Read the stitched orders and their scale factors from the summary line of a reduction log.

    Parameters
    ----------
    log : str
        Content of ``reduction_<name>.log``.
    label : str
        Measurement type in the log, e.g. ``"sample"`` or ``"background"``.
    name : str
        Name of the measurement.

    Returns
    -------
    tuple[list[int], list[float]]
        The stitched orders and the scale factor of each.
    """
    match = re.search(rf"Stitched orders \[([\d, ]+)\] of {label} {name} with scale factors \[([^\]]+)\]", log)
    assert match, f"no stitching summary for {label} {name}"
    orders = [int(order) for order in match.group(1).split(",")]
    factors = [float(factor) for factor in match.group(2).split(",")]
    return orders, factors


def assert_stitched_profile(output_dir: Path, name: str, q_cut: float) -> np.ndarray:
    """Check the stitched scaled profile of a measurement against the stitching rules.

    The profile is sorted by Q and holds no zero-count points. It keeps every valid point of
    harmonic 1 unchanged, and any other point lies at or above the unscattered-beam cutoff,
    since points of higher harmonics below it are never added.

    Parameters
    ----------
    output_dir : Path
        Directory holding the reduced files.
    name : str
        Name of the measurement.
    q_cut : float
        Unscattered-beam cutoff, in ``1/angstrom``.

    Returns
    -------
    np.ndarray
        The stitched profile, one ``[Q, I, E]`` row per point.
    """
    stitched = np.array(read_numbers_from_file(output_dir / f"{name}_det_stitched.txt"))
    harmonic_1 = np.array(read_numbers_from_file(output_dir / f"{name}_det_1.txt"))
    harmonic_1 = harmonic_1[harmonic_1[:, 1] != 0]

    assert np.all(np.diff(stitched[:, 0]) >= 0), f"stitched Q of {name} is not sorted"
    assert np.all(stitched[:, 1] != 0), f"stitched profile of {name} holds zero-count points"
    # Every valid harmonic-1 point is kept, with its intensity and uncertainty unchanged
    is_harmonic_1 = np.zeros(len(stitched), dtype=bool)
    for q, i, e in harmonic_1:
        match = np.isclose(stitched[:, 0], q, rtol=1e-12) & np.isclose(stitched[:, 1], i, rtol=1e-12)
        assert match.any(), f"harmonic-1 point at Q = {q} of {name} missing from the stitched profile"
        assert np.isclose(stitched[match, 2], e, rtol=1e-12).any()
        is_harmonic_1 |= match
    # Points of higher harmonics lie above the unscattered-beam peak
    assert np.all(stitched[~is_harmonic_1, 0] >= q_cut), f"higher-harmonic points of {name} below the cutoff"
    assert np.count_nonzero(~is_harmonic_1) > 0, f"no higher-harmonic points stitched for {name}"
    return stitched


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
            filename = f"{name}_det_1{suffix}.txt"
            output, expected = os.path.join(tmp_path, filename), os.path.join(goldendir, filename)
            if os.path.exists(expected):
                assert os.path.exists(output), f"Missing expected output file: {output}"
                compare_lines(output, expected)

    # The background-subtracted file is written for the samples ...
    for name in ["S115_pc3", "S115_dry"]:
        assert os.path.exists(os.path.join(tmp_path, f"{name}_det_1_background_subtracted.txt"))
    # ... but not for the background itself (nothing is subtracted from it)
    assert not os.path.exists(os.path.join(tmp_path, "EmptyPCell_det_1_background_subtracted.txt"))

    assert_reduction_log_files(tmp_path, [("S115_pc3", "sample"), ("S115_dry", "sample"), ("EmptyPCell", "background")])
    assert_no_legacy_names(tmp_path)


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
            harmonic_file = output_dir / f"{name}_det_{harmonic}_unscaled.txt"
            assert harmonic_file.is_file()
            assert harmonic_file.stat().st_size > 0
            assert len(harmonic_file.read_text(encoding="utf-8").splitlines()) == 65

            scaled_file = output_dir / f"{name}_det_{harmonic}.txt"
            assert scaled_file.is_file()
            assert scaled_file.stat().st_size > 0

    # Background subtraction is performed for every harmonic, for the samples only
    for harmonic in range(1, 5):
        for name in ["S115_dry", "S115_pc3"]:
            subtracted_file = output_dir / f"{name}_det_{harmonic}_background_subtracted.txt"
            assert subtracted_file.is_file()
            assert subtracted_file.stat().st_size > 0
        # Nothing is subtracted from the background itself
        assert not (output_dir / f"EmptyPCell_det_{harmonic}_background_subtracted.txt").exists()

    assert not list(output_dir.glob("bank_*")), "per-bank subdirectories should no longer be created"

    assert_reduction_log_files(
        output_dir, [("S115_pc3", "sample"), ("S115_dry", "sample"), ("EmptyPCell", "background")]
    )

    # Stitching writes one stitched profile per measurement, and reports its outcome in the
    # measurement's reduction log. The scale factors are regression values for the reference
    # dataset: a change to the stitching method changes them.
    expected_scale_factors = {
        "S115_pc3": ("sample", [1.0, 16.6985]),
        "S115_dry": ("sample", [1.0, 11.8484, 29.9429, 82.3565]),
        "EmptyPCell": ("background", [1.0, 3.7645]),
    }
    q_cut = unscattered_beam_q_cutoff(3.6)  # default primary wavelength of Experiment, in angstrom
    for name, (label, expected) in expected_scale_factors.items():
        stitched = assert_stitched_profile(output_dir, name, q_cut)
        log = (output_dir / f"reduction_{name}.log").read_text(encoding="utf-8")
        orders, factors = stitched_scale_factors(log, label, name)
        assert orders == list(range(1, len(expected) + 1))
        assert all(np.isfinite(factor) and factor > 0 for factor in factors)
        assert factors == pytest.approx(expected, rel=1e-3)
        if label == "sample":
            # The background-subtracted harmonics are stitched on the same Q points
            subtracted = np.array(read_numbers_from_file(output_dir / f"{name}_det_background_subtracted_stitched.txt"))
            np.testing.assert_allclose(subtracted[:, 0], stitched[:, 0], rtol=1e-12)
    assert not (output_dir / "EmptyPCell_det_background_subtracted_stitched.txt").exists()
    # Harmonic 3 of S115_pc3 does not extend the stitched profile to higher Q
    pc3_log = (output_dir / "reduction_S115_pc3.log").read_text(encoding="utf-8")
    assert "Stitched orders [1, 2] of sample S115_pc3" in pc3_log
    assert "order 3 does not extend the stitched profile to higher Q" in pc3_log
    assert "Stitched orders [1, 2, 3, 4] of sample S115_dry" in (output_dir / "reduction_S115_dry.log").read_text(
        encoding="utf-8"
    )
    assert_no_legacy_names(output_dir)


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
    assert list(tmp_path.glob("EmptyPCell_det_*")) == []
    assert_no_legacy_names(tmp_path)

    # The empty cell was subtracted from each sample, for every harmonic
    for name in ["S115_pc3", "S115_dry"]:
        for harmonic in range(1, 5):
            subtracted_file = tmp_path / f"{name}_det_{harmonic}_background_subtracted.txt"
            assert subtracted_file.is_file()
            assert subtracted_file.stat().st_size > 0
        logfile = (tmp_path / f"reduction_{name}.log").read_text(encoding="utf-8")
        assert f"Subtracted empty cell EmptyPCell from sample {name} for harmonics [1, 2, 3, 4]" in logfile

    # The scaled data equals the golden result (produced with transmission == 1.0) with
    # intensities and errors divided by the transmission coefficient. The golden file was
    # generated from setup.json, whose sample definitions are identical; scaled data is
    # independent of binning and subtraction.
    goldendir = os.path.join(os.path.dirname(config_file), "reduced")
    for sample in experiment.samples:
        transmission = sample.transmission
        assert 0.0 < transmission < 1.0
        golden = np.array(read_numbers_from_file(os.path.join(goldendir, f"{sample.name}_det_1.txt")))
        output = np.array(read_numbers_from_file(str(tmp_path / f"{sample.name}_det_1.txt")))
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
