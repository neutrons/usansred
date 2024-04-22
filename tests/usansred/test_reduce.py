# import standard
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

# third party packages
import pytest

# usansred imports
from usansred.reduce import main as reduceUSANS


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


@pytest.mark.datarepo
@mock_patch("usansred.reduce.parse_arguments")
def test_main(mock_parse_arguments, data_server, tmp_path):
    # Setup mock objects
    mock_args = MagicMock()
    mock_args.logbin = False
    mock_args.path = data_server.path_to("setup.csv")
    mock_args.output = str(tmp_path)
    mock_parse_arguments.return_value = mock_args
    reduceUSANS()


if __name__ == "__main__":
    pytest.main([__file__])
