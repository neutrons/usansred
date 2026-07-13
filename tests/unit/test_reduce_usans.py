"""Unit tests for :mod:`usansred.reduce_USANS`.

The autoreduction entry point ``main`` is a single monolithic function that is
tightly coupled to Mantid. These tests replace the Mantid algorithms, the
``mtd`` workspace store, and the downstream reduction/report calls with light
fakes so that the pure-Python control flow can be exercised without loading a
real NeXus event file.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from usansred import reduce_USANS


class _FakeProperty:
    """Minimal stand-in for a Mantid run log property."""

    def __init__(self, name, value, statistics=None):
        self.name = name
        self.value = value
        self._statistics = statistics

    def getStatistics(self):  # noqa: N802 (match Mantid API name)
        return self._statistics


class _FakeStatistics:
    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation


class _FakeRun:
    """Fake ``mtd['USANS'].getRun()`` returning canned log properties."""

    # Properties read via ``.value`` directly (must be hashable/printable).
    _SCALARS = {
        "start_time": "2020-01-01T00:00:00",
        "experiment_identifier": "IPTS-1",
        "run_number": "12345",
        "run_title": "TestSample",
    }
    # Properties read via ``.value[-1]``.
    _SEQUENCES = {
        "BL1A:CS:Scan:USANS:FirstRun": [1],
        "BL1A:CS:Scan:USANS:Index": [0],
        "BL1A:CS:Scan:USANS:Wavelength": [3.6],
    }

    def getProperty(self, name):  # noqa: N802 (match Mantid API name)
        # ROI Min/Size logs are read as ``.value[-1]``. Keeping every ROI:Min
        # equal to ``roi_min`` steers ``main_index`` to a valid wavelength slot.
        if name.endswith(":Min") or name.endswith(":Size"):
            return _FakeProperty(name, [1.0])
        if name in self._SCALARS:
            return _FakeProperty(name, self._SCALARS[name])
        if name in self._SEQUENCES:
            return _FakeProperty(name, self._SEQUENCES[name])
        return _FakeProperty(name, [1])

    def getProperties(self):  # noqa: N802 (match Mantid API name)
        # A single moving motor so the scan branch (which contains line 232)
        # is entered: std/mean = 0.1 > 0.01.
        motor = _FakeProperty(
            "BL1A:Mot:AnalyzerRot",
            [1, 2, 3],
            statistics=_FakeStatistics(mean=10.0, standard_deviation=1.0),
        )
        return [motor]


class _FakeMtd:
    """Dict-like fake for Mantid's ``mtd`` workspace store."""

    def __init__(self):
        run_container = MagicMock()
        run_container.getRun.return_value = _FakeRun()

        # Equal-length arrays so the per-angle I(Q) loop divides cleanly.
        scan_detector = MagicMock()
        scan_detector.readX.return_value = np.array([1.0, 2.0, 3.0])
        scan_detector.readY.return_value = np.array([10.0, 20.0, 30.0])
        scan_detector.readE.return_value = np.array([1.0, 2.0, 3.0])

        scan_monitor = MagicMock()
        scan_monitor.readY.return_value = np.array([100.0, 100.0, 100.0])

        self._workspaces = {
            "USANS": run_container,
            "USANS_scan_detector": scan_detector,
            "USANS_scan_monitor": scan_monitor,
        }

    def __getitem__(self, key):
        return self._workspaces.get(key, MagicMock())


@pytest.fixture
def reset_peaks():
    """The module keeps ``peaks`` as global state; isolate it per test."""
    saved = list(reduce_USANS.peaks)
    reduce_USANS.peaks.clear()
    yield
    reduce_USANS.peaks[:] = saved


@pytest.mark.usefixtures("reset_peaks")
def test_main_runs_end_to_end(tmp_path, monkeypatch):
    """``main`` runs to completion with Mantid mocked out.

    Exercising the full control flow reaches the scan branch and its
    ``from plot_publisher import plot1d`` import (line 232 of
    :mod:`usansred.reduce_USANS`). The test only asserts that ``main`` runs
    without error; it does not inspect the reduction results.
    """
    data_file = tmp_path / "USANS_12345.nxs.h5"
    data_file.write_text("")  # only needs to exist; LoadEventNexus is mocked
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(reduce_USANS, "mtd", _FakeMtd())
    for algo in (
        "LoadEventNexus",
        "Rebin",
        "CropWorkspace",
        "StepScan",
        "ConvertTableToMatrixWorkspace",
        "save_ascii",
        "save_summed_spectra",
        "generate_report",
    ):
        monkeypatch.setattr(reduce_USANS, algo, MagicMock())

    with (
        patch("plot_publisher.plot1d", MagicMock()),
        patch.object(reduce_USANS.reduce, "Experiment", MagicMock()),
        patch.object(reduce_USANS.sys, "argv", ["reduceUSANS", str(data_file), str(out_dir)]),
    ):
        reduce_USANS.main()
