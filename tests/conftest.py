import os
import sys
from unittest.mock import patch

import pytest
from mantid.simpleapi import config
from usansred.reduce import Experiment

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
            self._directory = os.path.join(os.path.dirname(this_module_path), "usansred-data")
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
            raise IOError(f"File {basename} not found in data directory {self._directory}")

    yield _DataServe()
    for key, val in _backup.items():
        config[key] = val


@pytest.fixture()
def mock_experiment():
    """Create a minimal Experiment with model_post_init bypassed."""
    with patch.object(Experiment, "model_post_init", return_value=None):
        exp = Experiment(config="dummy.json")
    exp.folder = ""
    exp.output_dir = ""
    exp.num_of_banks = 1
    exp.prim_wave = 3.6
    exp.darwin_width = 5.1
    exp.logbin = False
    exp.v_angle = 0.042
    exp.min_q = 1e-6
    exp.step_per_dec = 33
    exp.background = None
    exp.samples = []
    return exp


@pytest.fixture()
def mock_experiment_2banks():
    """Create a minimal Experiment with 2 detector banks."""
    with patch.object(Experiment, "model_post_init", return_value=None):
        exp = Experiment(config="dummy.json")
    exp.folder = ""
    exp.output_dir = ""
    exp.num_of_banks = 2
    exp.prim_wave = 3.6
    exp.darwin_width = 5.1
    exp.logbin = False
    exp.v_angle = 0.042
    exp.min_q = 1e-6
    exp.step_per_dec = 33
    exp.background = None
    exp.samples = []
    return exp
