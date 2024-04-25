# standard imports
import os
import sys

# third party imports
from mantid.simpleapi import config
import pytest

# usansred imports


this_module_path = sys.modules[__name__].__file__


@pytest.fixture(scope="session")
def data_server():
    r"""Object containing info and functionality for data files
    Also, it adds the path of the data-repo to the list of Mantid data directories
    """

    _options = ["datasearch.directories", "default.facility", "default.instrument"]
    _backup = {key: config[key] for key in _options}

    class _DataServe(object):
        def __init__(self):
            self._directory = os.path.join(
                os.path.dirname(this_module_path), "usansred-data"
            )
            config.appendDataSearchDir(self._directory)
            config["default.facility"] = "SNS"
            config["default.instrument"] = "USANS"

        @property
        def directory(self) -> str:
            r"""Absolute path to the data-repo directory"""
            return self._directory

        def path_to(self, basename: str) -> str:
            r"""
            Absolute path to a file in the data directory or its subdirectories.

            Parameters
            ----------
            basename
                file name (with extension) to look for

            Returns
            -------
                First match of the file in the data directory or its subdirectories
            """
            for dirpath, dirnames, filenames in os.walk(self._directory):
                if basename in filenames:
                    return os.path.join(dirpath, basename)
            raise IOError(
                f"File {basename} not found in data directory {self._directory}"
            )

    yield _DataServe()
    for key, val in _backup.items():
        config[key] = val
