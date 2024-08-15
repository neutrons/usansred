# import standard
import os
import numpy as np
import random
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

# third party packages
import pytest

# usansred imports
from usansred.reduce import main as reduceUSANS
from usansred.reduce import Sample,Experiment


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
    """
    Compare corresponding numbers in two files line by line.
    If the relative difference exceeds the threshold, print a warning.
    """
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


@mock_patch("usansred.reduce.parse_arguments")
def test_main_nonvalid_file(mock_parse_arguments):
    # Setup mock objects
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = "invalid_path.csv"
    mock_parse_arguments.return_value = mock_args
    with pytest.raises(FileNotFoundError) as error:
        reduceUSANS()
    assert str(error.value) == "The csv file invalid_path.csv doesn't exist"


@pytest.mark.datarepo()
@mock_patch("usansred.reduce.parse_arguments")
def test_main(mock_parse_arguments, data_server, tmp_path):
    # Setup mock objects
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = data_server.path_to("setup.csv")
    mock_args.output = str(tmp_path)
    mock_parse_arguments.return_value = mock_args
    reduceUSANS()
    # compare the content of output files with files containing expected results
    goldendir = os.path.join(os.path.dirname(mock_args.path), "reduced")  # where the expected content resides
    for name in ["EmptyPCell", "S115_dry", "S115_pc3"]:
        for suffix in ["", "_lbs", "_lb", "_unscaled"]:
            filename = f"UN_{name}_det_1{suffix}.txt"
            output, expected = os.path.join(tmp_path, filename), os.path.join(goldendir, filename)
            if os.path.exists(expected):  # file "UN_EmptyPCell_det_1_lbs.txt" does not exist
                compare_lines(output, expected)

@pytest.mark.datarepo()
def test_sample_match_or_interpolate(data_server, tmp_path):
    # Get the testing data and temp output directory
    # Create new Experiment instance
    csvpath = data_server.path_to("setup.csv")
    tmpoutput = str(tmp_path)
    exp = Experiment(csvpath, logbin=False, outputFolder=tmpoutput)

    # Genearte testing data
    qq = np.array([dd * 1e-5 for dd in range(1, 100)])
    ii = -np.log(qq) * 1e3
    bb = ii * 0.01
    
    # Generate a list of 100 random numbers
    ee = [random.random() for _ in range(1, 100)]

    sample_test = exp.samples[0]

    iibgmatched, eebgmatched = sample_test._match_or_interpolate(qq, qq, bb, ee)

    check = (iibgmatched - bb == 0.)
    assert np.all(check), "Background interpolation calculation is not right in Sample._match_or_interpolate"

if __name__ == "__main__":
    pytest.main([__file__])
