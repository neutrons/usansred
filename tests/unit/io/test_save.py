import numpy as np
import pytest
from mantid.simpleapi import CreateWorkspace, LoadAscii
from numpy.testing import assert_allclose

from usansred.io.save import save_ascii, save_summed_spectra


@pytest.fixture
def sample_workspace():
    x = np.tile(np.arange(10), 3)
    y = np.repeat([10, 100, 1000], 10)
    e = 0.1 * y
    workspace = CreateWorkspace(DataX=x, DataY=y, DataE=e, NSpec=3, UnitX="TOF")
    assert workspace.getNumberHistograms() == 3
    assert workspace.getAxis(0).getUnit().unitID() == "TOF"
    return workspace


def test_save_ascii(tmp_path, sample_workspace):
    file_path = str(tmp_path / "ascii_output.txt")
    save_ascii(sample_workspace, file_path, header="TOF, COUNTS, ERROR")
    data = LoadAscii(Filename=file_path)  # shape: (30, 3)
    assert data.getNumberHistograms() == 3
    for i in range(3):
        x_vals = data.dataX(i)
        y_vals = data.dataY(i)
        e_vals = data.dataE(i)

        assert_allclose(x_vals, np.arange(10))
        assert_allclose(y_vals, 10 * 10**i * np.ones(10))
        assert_allclose(e_vals, 0.1 * y_vals)
    assert open(file_path, "r").readline() == "# TOF, COUNTS, ERROR\n"


def test_saved_summed_spectra(tmp_path, sample_workspace):
    file_path = str(tmp_path / "summed_spectra.txt")
    save_summed_spectra(sample_workspace, file_path, header="TOF, COUNTS, ERROR")

    data = LoadAscii(Filename=file_path)  # shape: (10, 3)
    assert data.getNumberHistograms() == 1
    x_vals = data.dataX(0)
    y_vals = data.dataY(0)
    e_vals = data.dataE(0)

    assert_allclose(x_vals, np.arange(10))
    assert_allclose(y_vals, 1110.0 * np.ones(10))
    assert_allclose(e_vals, 100.504 * np.ones(10))

    assert open(file_path, "r").readline() == "# TOF, COUNTS, ERROR\n"
