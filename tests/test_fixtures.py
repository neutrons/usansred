# standard packages
import os

# third party packages
import pytest


@pytest.mark.datarepo()
def test_data_server(data_server):
    r"""find one file within the data-directory"""
    expected = os.path.join(data_server.directory, "testdataserver", "empty.txt")
    assert data_server.path_to("empty.txt") == expected


if __name__ == "__main__":
    pytest.main([__file__])
