"""The preprocessing script run on the event file of run 41713 of IPTS-30410, from the data repository."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from usansred import reduce_USANS
from usansred.reduce_USANS import PREPROCESSING_FAILED, main

RUN = "41713"


def numbers(path: Path) -> np.ndarray:
    """The numbers of the lines of a per-run file made only of numbers, the header lines aside."""
    values = []
    for line in path.read_text().splitlines():
        try:
            values.extend(float(value) for value in re.split(r"[,\s]+", line.strip()) if value)
        except ValueError:
            continue
    return np.array(values)


@pytest.fixture
def event_file(data_server) -> str:
    return data_server.path_to(f"USANS_{RUN}.nxs.h5")


@pytest.fixture
def reference_dir(data_server) -> Path:
    return Path(data_server.directory) / "IPTS-30410" / "shared"


def run_script(monkeypatch, event_file: str, outdir: Path, *options: str) -> None:
    monkeypatch.setattr(sys, "argv", ["reduce_USANS.py", event_file, str(outdir), *options])
    main()


@pytest.mark.datarepo
def test_preprocesses_and_reduces_run(event_file, reference_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(reduce_USANS, "publish_plot", lambda *_args, **_kwargs: pytest.fail("figure uploaded"))

    run_script(monkeypatch, event_file, tmp_path, "--no_publish")

    # The per-run files are those of the autoreduction, in the data repository
    written = sorted(path.name for path in tmp_path.glob(f"USANS_{RUN}_*.txt"))
    expected = sorted(path.name for path in reference_dir.glob(f"USANS_{RUN}_*.txt"))
    assert len(written) == 21
    assert written == expected
    for name in written:
        np.testing.assert_allclose(numbers(tmp_path / name), numbers(reference_dir / name), rtol=1e-9, err_msg=name)
    # The run is recorded in its sequence, under the first run
    assert list(json.loads((tmp_path / f"scan_{RUN}.json").read_text())) == [RUN]
    assert json.loads((tmp_path / f"sample_{RUN}.json").read_text())["thickness"] == 0.1
    # Step 2 reduced the sequence, of one run so far
    (sample,) = json.loads((tmp_path / f"autoconfig_{RUN}.json").read_text())["samples"]
    assert (sample["name"], sample["start_scan_num"], sample["num_of_scans"]) == ("S115_dry", int(RUN), 1)
    assert (tmp_path / "auto" / "S115_dry_det_1.txt").exists()
    # The figure holds the raw scan and the reduced sequence
    page = (tmp_path / f"USANS_{RUN}.html").read_text()
    assert page.count('class="plotly-graph-div"') == 2
    assert "S115_dry_det_1" in page


@pytest.mark.datarepo
def test_run_not_recorded_if_preprocessing_fails(event_file, tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setattr(reduce_USANS, "publish_plot", lambda *args, **kwargs: uploads.append((args, kwargs)))
    save_ascii = reduce_USANS.save_ascii

    def failing_save_ascii(workspace, file_path, *args, **kwargs):
        # Fails after the files of the main peak and of peaks 1 and 2 are written
        if file_path.endswith("_detector_scan_ARN_peak_3.txt"):
            raise OSError(f"simulated failure while writing {file_path}")
        return save_ascii(workspace, file_path, *args, **kwargs)

    monkeypatch.setattr(reduce_USANS, "save_ascii", failing_save_ascii)

    with pytest.raises(OSError, match="simulated failure"):
        run_script(monkeypatch, event_file, tmp_path)

    # The run is not recorded, so that step 2 excludes it, and step 2 did not run
    assert not list(tmp_path.glob("*.json"))
    assert not (tmp_path / "auto").exists()
    # The figure holds the raw scan and the error, and is uploaded
    page = (tmp_path / f"USANS_{RUN}.html").read_text()
    assert page.count('class="plotly-graph-div"') == 1
    assert f"<h2>{PREPROCESSING_FAILED}</h2>" in page
    assert "OSError: simulated failure while writing" in page
    ((args, kwargs),) = uploads
    assert args == ("USANS", RUN)
    assert kwargs["files"]["file"] in page
