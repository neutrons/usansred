"""Tests of the preprocessing script (#107).

Step 1 preprocesses one run and records it in the files of its sequence. Step 2 goes from the files of
the sequences to the reduction.
"""

import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import requests

from usansred import reduce_USANS
from usansred.io.read import read_config
from usansred.reduce_USANS import (
    FIRST_RUN_LOG,
    PREPROCESSING_FAILED,
    REDUCTION_FAILED,
    THICKNESS_LOG,
    PreprocessingError,
    as_run_number,
    build_setup,
    collect_sequences,
    failure_block,
    main,
    parse_arguments,
    raw_scan_plot,
    read_first_run,
    record_run,
    reduce_and_show,
    reduce_run_sequence,
    reduce_sequence,
    reduced_sequence_plot,
    sample_thickness,
    save_figure,
    scanned_motors,
    update_sequence_info,
    upload_figure,
    write_config,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "preprocessing"

# First run and number of recorded runs of each sequence of the fixtures, from IPTS-30410
SEQUENCES = {41713: 6, 41741: 6, 41832: 6}

# Titles of the sequences of the fixtures, as recorded in sample_<N>.json
TITLES = {41713: "Align:S115_dry", 41741: "Align:EmptyPCell", 41832: "Align:S115_pc3"}

# Runs of the fixture scan_-1.json, recorded just before each sequence
NO_SEQUENCE_RUNS = {"41712", "41740", "41831"}


@pytest.fixture
def outdir(tmp_path: Path) -> Path:
    """An output directory of the preprocessing script, holding the sequence files of the fixtures."""
    for path in DATA_DIR.glob("*.json"):
        shutil.copy(path, tmp_path)
    return tmp_path


class LogCapture:
    """The messages logged by the preprocessing script."""

    def __init__(self, caplog: pytest.LogCaptureFixture):
        self.caplog = caplog

    @property
    def messages(self) -> list[str]:
        """Messages at level INFO and above."""
        return self.caplog.messages

    def records_at_warning(self) -> list[str]:
        """Messages at level WARNING and above."""
        return [record.getMessage() for record in self.caplog.records if record.levelno >= logging.WARNING]


@pytest.fixture
def log(monkeypatch, caplog) -> LogCapture:
    """Captures the messages logged by the preprocessing script, at level INFO and above."""
    # The logger of the module does not propagate to the root logger, where caplog listens
    monkeypatch.setattr(reduce_USANS.logger, "propagate", True)
    caplog.set_level(logging.INFO)
    return LogCapture(caplog)


def write_scan_file(
    outdir: Path,
    first_run: int,
    num_of_runs: int = 1,
    runs: list[int] | None = None,
    titles: str | list[str | None] | None = None,
) -> None:
    """Write a scan_<N>.json file for a sequence starting at ``first_run``.

    The sequence holds ``runs``, in the order given, or else ``num_of_runs`` consecutive runs. ``titles``
    is the title of every run, or a list with the title of each run (``None`` for a run without a title),
    or ``None`` for runs without a title, as written before the titles were recorded.
    """
    if runs is None:
        runs = list(range(first_run, first_run + num_of_runs))
    if not isinstance(titles, list):
        titles = [titles] * len(runs)
    entries = {}
    for run, title in zip(runs, titles, strict=True):
        entries[str(run)] = {"iq": f"USANS_{run}_detector_3.6.txt"}
        if title is not None:
            entries[str(run)]["title"] = title
    (outdir / f"scan_{first_run}.json").write_text(json.dumps(entries))


def sample_names(outdir: Path) -> list[str]:
    """Names of the samples of the setup built from the sequence files of ``outdir``."""
    return [sample.name for sample in build_setup(collect_sequences(str(outdir))).samples]


def write_sample_file(outdir: Path, first_run: int, **sample_info) -> None:
    """Change the values of the sample_<N>.json file of a fixture sequence."""
    sample_file = outdir / f"sample_{first_run}.json"
    sample_file.write_text(json.dumps(json.loads(sample_file.read_text()) | sample_info))


def write_event_file(path: Path, first_run: list | None, dtype=np.int32) -> Path:
    """Write an HDF5 file holding only the log FirstRun of an event file, with the values given.

    The values are stored with ``dtype``, as integers by default; ``h5py.string_dtype()`` stores text.
    Without values, the file has no such log.
    """
    with h5py.File(path, "w") as nexus:
        logs = nexus.create_group("entry/DASlogs")
        if first_run is not None:
            logs.create_dataset(f"{FIRST_RUN_LOG}/value", data=np.array(first_run, dtype=dtype))
    return path


class TestReadFirstRun:
    def test_no_sequence(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41712.nxs.h5", [-1])

        assert read_first_run(str(event_file)) == -1

    def test_sequence(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41714.nxs.h5", [41713])

        assert read_first_run(str(event_file)) == 41713

    def test_last_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", [-1, 41713])

        assert read_first_run(str(event_file)) == 41713

    def test_no_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", [])

        assert read_first_run(str(event_file)) is None

    def test_no_log(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", None)

        assert read_first_run(str(event_file)) is None

    def test_not_an_hdf5_file(self, tmp_path):
        event_file = tmp_path / "USANS_41713.nxs.h5"
        event_file.write_text("not an HDF5 file")

        with pytest.raises(OSError, match="file signature not found"):
            read_first_run(str(event_file))

    def test_text_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", ["Not Set"], dtype=h5py.string_dtype())

        assert read_first_run(str(event_file)) == "Not Set"

    def test_float_value(self, tmp_path):
        event_file = write_event_file(tmp_path / "USANS_41713.nxs.h5", [41713.5], dtype=np.float64)

        value = read_first_run(str(event_file))

        assert value == 41713.5
        assert type(value) is float


class TestAsRunNumber:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (41713, 41713),
            (41713.0, 41713),
            ("41713", 41713),
            (-1, -1),
            (0, 0),
            (41713.5, None),
            (float("nan"), None),
            (float("inf"), None),
            ("Not Set", None),
            ("", None),
            (True, None),
            (None, None),
        ],
        ids=[
            "integer",
            "whole_float",
            "numeric_text",
            "minus_one",
            "zero",
            "fractional",
            "nan",
            "infinity",
            "text",
            "empty_text",
            "boolean",
            "none",
        ],
    )
    def test_conversion(self, value, expected):
        assert as_run_number(value) == expected


class TestMainSkipsRuns:
    @pytest.mark.parametrize(
        ("first_run", "dtype", "reason"),
        [
            ([-1], np.int32, f"its log {FIRST_RUN_LOG} is -1, which is not a run number"),
            ([41845, 0], np.int32, f"its log {FIRST_RUN_LOG} is 0, which is not a run number"),
            ([-5], np.int32, f"its log {FIRST_RUN_LOG} is -5, which is not a run number"),
            (None, np.int32, f"it has no log {FIRST_RUN_LOG}"),
            (["Not Set"], h5py.string_dtype(), f"its log {FIRST_RUN_LOG} is 'Not Set', which is not a run number"),
            ([float("nan")], np.float64, f"its log {FIRST_RUN_LOG} is nan, which is not a run number"),
            ([41713.5], np.float64, f"its log {FIRST_RUN_LOG} is 41713.5, which is not a run number"),
        ],
        ids=["minus_one", "zero", "negative", "no_log", "text", "nan", "fractional"],
    )
    def test_run_outside_a_sequence(self, tmp_path, monkeypatch, log, first_run, dtype, reason):
        event_file = write_event_file(tmp_path / "USANS_41712.nxs.h5", first_run, dtype)
        outdir = tmp_path / "shared"
        outdir.mkdir()

        def load_event_nexus(*_args, **_kwargs):
            pytest.fail("the events of a run outside a sequence are loaded")

        def publish_plot(*_args, **_kwargs):
            pytest.fail("a figure is uploaded for a run outside a sequence")

        monkeypatch.setattr(reduce_USANS, "LoadEventNexus", load_event_nexus)
        monkeypatch.setattr(reduce_USANS, "publish_plot", publish_plot)
        monkeypatch.setattr(sys, "argv", ["reduce_USANS.py", str(event_file), str(outdir)])

        main()

        assert list(outdir.iterdir()) == []
        assert any("is not processed" in message and reason in message for message in log.messages)


class StandInLog:
    """Stands in for a Mantid time-series log, with the statistics of its values."""

    def __init__(self, name: str, values: list[float]):
        self.name = name
        self.values = np.asarray(values, dtype=float)

    def getStatistics(self):
        return SimpleNamespace(mean=self.values.mean(), standard_deviation=self.values.std())


class StandInScanRun:
    """Stands in for the run information of a Mantid workspace, holding the logs of motors."""

    def __init__(self, logs: list[StandInLog], run_number: str = "41714"):
        self.logs = logs
        self.run_number = run_number

    def getProperties(self):
        return self.logs

    def getProperty(self, name: str):
        assert name == "run_number"
        return SimpleNamespace(value=self.run_number)


class TestScannedMotors:
    def test_varying_motor(self):
        run = StandInScanRun([StandInLog("BL1A:Mot:ARN", [200, 220, 240]), StandInLog("BL1A:Mot:Sample", [5, 5, 5])])

        assert scanned_motors(run) == ["BL1A:Mot:ARN"]

    def test_motor_varying_around_zero(self):
        run = StandInScanRun([StandInLog("BL1A:Mot:ARN", [-20, 0, 20])])

        assert scanned_motors(run) == ["BL1A:Mot:ARN"]

    def test_small_variation_is_not_a_scan(self):
        # Standard deviation of about 0.4 % of the mean
        run = StandInScanRun([StandInLog("BL1A:Mot:ARN", [200, 201, 202])])

        assert scanned_motors(run) == []

    def test_readback_and_other_logs_ignored(self):
        run = StandInScanRun([StandInLog("BL1A:Mot:ARN.RBV", [200, 300]), StandInLog("BL1A:Det:Rate", [1, 100])])

        assert scanned_motors(run) == []

    def test_several_motors(self):
        run = StandInScanRun([StandInLog("BL1A:Mot:ARN", [200, 300]), StandInLog("BL1A:Mot:ASN", [10, 30])])

        assert scanned_motors(run) == ["BL1A:Mot:ARN", "BL1A:Mot:ASN"]


class TestMainProcessesRuns:
    """main() for a run of a sequence, with stand-ins for the loading of the events and for both steps."""

    @pytest.fixture
    def run(self, monkeypatch):
        """Stand-ins for Mantid and for step 2; returns the calls of preprocess_run and reduce_and_show."""
        calls = SimpleNamespace(preprocess=[], reduce_and_show=[])
        monkeypatch.setattr(reduce_USANS, "LoadEventNexus", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(reduce_USANS, "mtd", {"USANS": SimpleNamespace(getRun=lambda: StandInScanRun([]))})
        monkeypatch.setattr(reduce_USANS, "reduce_and_show", lambda *args: calls.reduce_and_show.append(args))
        return calls

    def run_main(self, tmp_path, monkeypatch, first_run, *options) -> Path:
        event_file = write_event_file(tmp_path / "USANS_41714.nxs.h5", [first_run], dtype=type(first_run))
        outdir = tmp_path / "shared"
        outdir.mkdir()
        monkeypatch.setattr(sys, "argv", ["reduce_USANS.py", str(event_file), str(outdir), *options])
        main()
        return outdir

    def test_steps_receive_the_validated_first_run(self, tmp_path, monkeypatch, run):
        def preprocess_run(_filename, _outdir, first_run, run_number, load_monitors, figure):
            run.preprocess.append((first_run, run_number, load_monitors))
            figure.raw_scan = "<div>raw</div>"

        monkeypatch.setattr(reduce_USANS, "preprocess_run", preprocess_run)

        outdir = self.run_main(tmp_path, monkeypatch, 41713.0, "--no_publish")

        assert run.preprocess == [(41713, "41714", True)]
        assert type(run.preprocess[0][0]) is int
        assert run.reduce_and_show == [(str(outdir), 41713, "41714", "<div>raw</div>", True)]

    def test_preprocessing_failure_shown(self, tmp_path, monkeypatch, run):
        upload = StandInUpload(monkeypatch)

        def preprocess_run(*args):
            figure = args[-1]
            figure.raw_scan = "<div>raw</div>"
            raise RuntimeError("SaveAscii failed: <disk full>")

        monkeypatch.setattr(reduce_USANS, "preprocess_run", preprocess_run)

        with pytest.raises(RuntimeError, match="SaveAscii failed"):
            self.run_main(tmp_path, monkeypatch, 41713)

        ((instrument, run_number, files),) = upload.uploads
        figure = files["file"]
        assert (instrument, run_number) == ("USANS", "41714")
        assert figure.startswith("<div>raw</div>\n")
        assert f"<h2>{PREPROCESSING_FAILED}</h2>" in figure
        assert "RuntimeError: SaveAscii failed: &lt;disk full&gt;" in figure
        outdir = tmp_path / "shared"
        assert figure in (outdir / "USANS_41714.html").read_text()
        assert run.reduce_and_show == []
        assert not list(outdir.glob("*.json"))

    def test_run_without_scanned_motor(self, tmp_path, monkeypatch, run):
        upload = StandInUpload(monkeypatch)

        def preprocess_run(*_args):
            raise PreprocessingError("no log BL1A:Mot:* varies during the run, so the run is not a scan")

        monkeypatch.setattr(reduce_USANS, "preprocess_run", preprocess_run)

        with pytest.raises(PreprocessingError, match="not a scan"):
            self.run_main(tmp_path, monkeypatch, 41713)

        ((_, _, files),) = upload.uploads
        assert files["file"].startswith(f"<h2>{PREPROCESSING_FAILED}</h2>")
        assert "PreprocessingError: no log BL1A:Mot:* varies during the run" in files["file"]
        assert run.reduce_and_show == []

    @pytest.mark.usefixtures("run")
    def test_preprocessing_failure_not_uploaded_with_no_publish(self, tmp_path, monkeypatch):
        upload = StandInUpload(monkeypatch)

        def preprocess_run(*_args):
            raise RuntimeError("SaveAscii failed")

        monkeypatch.setattr(reduce_USANS, "preprocess_run", preprocess_run)

        with pytest.raises(RuntimeError, match="SaveAscii failed"):
            self.run_main(tmp_path, monkeypatch, 41713, "--no_publish")

        assert upload.uploads == []
        assert PREPROCESSING_FAILED in (tmp_path / "shared" / "USANS_41714.html").read_text()


class TestCommandLine:
    def test_arguments(self):
        args = parse_arguments(["USANS_41713.nxs.h5", "shared"])

        assert (args.event_file, args.outdir, args.no_publish) == ("USANS_41713.nxs.h5", "shared", False)

    def test_no_publish(self):
        assert parse_arguments(["USANS_41713.nxs.h5", "shared", "--no_publish"]).no_publish is True

    @pytest.mark.parametrize("argv", [[], ["USANS_41713.nxs.h5"], ["a", "b", "c"]], ids=["none", "one", "three"])
    def test_wrong_number_of_arguments(self, argv):
        with pytest.raises(SystemExit) as error:
            parse_arguments(argv)

        assert error.value.code == 2

    def test_missing_event_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["reduce_USANS.py", str(tmp_path / "USANS_1.nxs.h5"), str(tmp_path)])

        with pytest.raises(SystemExit) as error:
            main()

        assert error.value.code is None
        assert "not found" in capsys.readouterr().out


class StandInRun:
    """Stands in for mantid.api.Run, holding the logs given as keyword arguments."""

    class Property:
        def __init__(self, value):
            self.value = value

    def __init__(self, **logs):
        self.logs = logs

    def hasProperty(self, name: str) -> bool:  # noqa: N802
        return name in self.logs

    def getProperty(self, name: str) -> "StandInRun.Property":  # noqa: N802
        return self.Property(self.logs[name])


class TestSampleThickness:
    def test_number(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: [0.3]})) == 0.3

    def test_last_value(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: [0.2, 0.3]})) == 0.3

    def test_numeric_string(self):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: ["0.3"]})) == 0.3

    @pytest.mark.parametrize(
        "value", [["Not Set"], [], [0.0], [-0.3], [float("nan")]], ids=["not_set", "empty", "zero", "negative", "nan"]
    )
    def test_not_a_thickness(self, log, value):
        assert sample_thickness(StandInRun(**{THICKNESS_LOG: value})) == 0.1
        assert any(THICKNESS_LOG in message for message in log.messages)

    def test_no_log(self, log):
        assert sample_thickness(StandInRun()) == 0.1
        assert any(THICKNESS_LOG in message for message in log.messages)


class TestRecordRun:
    def test_title_in_scan_file(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.3)

        scan = json.loads((tmp_path / "scan_41713.json").read_text())
        assert scan == {"41713": {"iq": "USANS_41713_detector_3.6.txt", "title": "Align:S115_dry"}}

    def test_thickness_in_sample_file(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.3)

        sample = json.loads((tmp_path / "sample_41713.json").read_text())
        assert sample == {"title": "Align:S115_dry", "background": 0, "thickness": 0.3}

    def test_second_run(self, tmp_path):
        record_run(str(tmp_path), 41713, "41713", "USANS_41713_detector_3.6.txt", "Align:S115_dry", 0.1)
        record_run(str(tmp_path), 41713, "41714", "USANS_41714_detector_3.6.txt", "Align:S115_dry", 0.1)

        scan = json.loads((tmp_path / "scan_41713.json").read_text())
        assert list(scan) == ["41713", "41714"]

    def test_replaces_thickness_edited_by_hand(self, tmp_path):
        sample_file = tmp_path / "sample_41713.json"
        sample_file.write_text(json.dumps({"title": "Align:S115_dry", "background": 0, "thickness": 0.5}))

        record_run(str(tmp_path), 41713, "41714", "USANS_41714_detector_3.6.txt", "Align:S115_dry", 0.1)

        assert json.loads(sample_file.read_text())["thickness"] == 0.1

    def test_unreadable_scan_file_is_left_unchanged(self, tmp_path):
        scan_file = tmp_path / "scan_41713.json"
        # A file left partly written by a job that was stopped
        scan_file.write_text('{"41713": {"iq": "USANS_41713_detector_3.6.txt", "ti')

        with pytest.raises(ValueError, match="scan_41713.json cannot be read, and is left unchanged"):
            record_run(str(tmp_path), 41713, "41714", "USANS_41714_detector_3.6.txt", "Align:S115_dry", 0.1)

        assert scan_file.read_text() == '{"41713": {"iq": "USANS_41713_detector_3.6.txt", "ti'

    def test_step_2_reads_recorded_runs(self, tmp_path):
        for run in range(41713, 41719):
            record_run(str(tmp_path), 41713, str(run), f"USANS_{run}_detector_3.6.txt", "Align:S115_dry", 0.3)

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.thickness) == (41713, 6, 0.3)


class TestUpdateSequenceInfo:
    def test_creates_file(self, tmp_path):
        out_file = tmp_path / "scan_41713.json"

        update_sequence_info(str(out_file), {"41713": {"title": "S115"}})

        assert json.loads(out_file.read_text()) == {"41713": {"title": "S115"}}

    def test_adds_and_replaces_entries(self, tmp_path):
        out_file = tmp_path / "scan_41713.json"
        out_file.write_text(json.dumps({"41713": {"title": "S115"}, "41714": {"title": "old"}}))

        update_sequence_info(str(out_file), {"41714": {"title": "S115"}, "41715": {"title": "S115"}})

        assert json.loads(out_file.read_text()) == {
            "41713": {"title": "S115"},
            "41714": {"title": "S115"},
            "41715": {"title": "S115"},
        }

    def test_leaves_no_temporary_file(self, tmp_path):
        out_file = tmp_path / "scan_41713.json"
        out_file.write_text(json.dumps({"41713": {"title": "S115"}}))

        update_sequence_info(str(out_file), {"41714": {"title": "S115"}})

        assert list(tmp_path.iterdir()) == [out_file]

    @pytest.mark.parametrize("content", ["", '{"41713": {', "not JSON"], ids=["empty", "partly_written", "text"])
    def test_unreadable_file_raises(self, tmp_path, content):
        out_file = tmp_path / "scan_41713.json"
        out_file.write_text(content)

        with pytest.raises(ValueError, match="cannot be read, and is left unchanged"):
            update_sequence_info(str(out_file), {"41714": {"title": "S115"}})

        assert out_file.read_text() == content

    def test_file_without_object_raises(self, tmp_path):
        out_file = tmp_path / "scan_41713.json"
        out_file.write_text("[41713]")

        with pytest.raises(ValueError, match="does not hold a JSON object"):
            update_sequence_info(str(out_file), {"41714": {"title": "S115"}})

        assert out_file.read_text() == "[41713]"


class TestCollectSequences:
    def test_one_sequence_per_scan_file(self, outdir):
        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert [len(sequence.runs) for sequence in sequences] == list(SEQUENCES.values())

    def test_ignores_runs_outside_a_sequence(self, outdir, log):
        sequences = collect_sequences(str(outdir))

        assert NO_SEQUENCE_RUNS.isdisjoint(run for sequence in sequences for run in sequence.runs)
        assert any("scan_-1.json" in message and "3 runs" in message for message in log.messages)

    def test_ignores_first_run_zero(self, outdir, log):
        # In IPTS-30410, runs 41845-41855 have FirstRun = 0
        write_scan_file(outdir, 0, runs=[41845, 41846, 41847])

        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert any(
            "scan_0.json" in message and "3 runs" in message and "FirstRun = 0" in message for message in log.messages
        )

    def test_orders_by_run_number(self, tmp_path):
        write_scan_file(tmp_path, 10001)
        write_scan_file(tmp_path, 9990)

        sequences = collect_sequences(str(tmp_path))

        assert [sequence.first_run for sequence in sequences] == [9990, 10001]

    def test_ignores_file_without_run_number(self, outdir, log):
        (outdir / "scan_old.json").write_text(json.dumps({"41713": {"iq": "USANS_41713_detector_1.8.txt"}}))

        sequences = collect_sequences(str(outdir))

        assert [sequence.first_run for sequence in sequences] == list(SEQUENCES)
        assert any("scan_old.json" in message for message in log.messages)

    def test_no_scan_file(self, tmp_path):
        assert collect_sequences(str(tmp_path)) == []

    def test_reads_sample_files(self, outdir):
        sequences = collect_sequences(str(outdir))

        assert [sequence.sample["title"] for sequence in sequences] == list(TITLES.values())

    def test_missing_sample_file(self, outdir, log):
        (outdir / "sample_41741.json").unlink()

        sequences = collect_sequences(str(outdir))

        assert sequences[1].sample == {}
        assert any("sample_41741.json" in message for message in log.messages)


class TestBuildSetup:
    def test_every_sequence_is_a_sample(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.name for sample in config.samples] == ["S115_dry", "EmptyPCell", "S115_pc3"]
        assert [sample.start_scan_num for sample in config.samples] == list(SEQUENCES)
        assert [sample.thickness for sample in config.samples] == [0.1, 0.1, 0.1]
        assert config.background is None
        assert config.empty_cell is None

    def test_sequence_flagged_as_background_is_a_sample(self, outdir):
        write_sample_file(outdir, 41741, background=1)

        config = build_setup(collect_sequences(str(outdir)))

        assert len(config.samples) == len(SEQUENCES)
        assert config.background is None

    def test_single_sequence(self, tmp_path):
        write_scan_file(tmp_path, 41713, num_of_runs=6)

        config = build_setup(collect_sequences(str(tmp_path)))

        assert len(config.samples) == 1
        assert config.background is None

    def test_keeps_every_run(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.num_of_scans for sample in config.samples] == list(SEQUENCES.values())
        assert [sample.exclude for sample in config.samples] == [[], [], []]

    def test_excludes_missing_runs(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[100, 101, 103])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert sample.start_scan_num == 100
        assert sample.num_of_scans == 4
        assert sample.exclude == [102]

    def test_span_starts_at_first_run(self, tmp_path):
        """Run 100, the first run of the sequence, was not recorded."""
        write_scan_file(tmp_path, 100, runs=[101, 102])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.exclude) == (100, 3, [100])

    def test_leaves_out_runs_before_first_run(self, tmp_path, log):
        write_scan_file(tmp_path, 100, runs=[99, 100, 101])

        (sample,) = build_setup(collect_sequences(str(tmp_path))).samples

        assert (sample.start_scan_num, sample.num_of_scans, sample.exclude) == (100, 2, [])
        assert any("[99]" in message and "scan_100.json" in message for message in log.records_at_warning())

    def test_order_of_runs_in_scan_file(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[103, 100, 101])
        write_scan_file(tmp_path, 200, runs=[200, 201, 203])

        first, second = build_setup(collect_sequences(str(tmp_path))).samples

        assert (first.start_scan_num, first.num_of_scans, first.exclude) == (100, 4, [102])
        assert (second.start_scan_num, second.num_of_scans, second.exclude) == (200, 4, [202])

    def test_thickness_from_sample_file(self, outdir):
        write_sample_file(outdir, 41741, thickness=0.3)

        config = build_setup(collect_sequences(str(outdir)))

        assert [sample.thickness for sample in config.samples] == [0.1, 0.3, 0.1]

    def test_thickness_as_string(self, outdir):
        write_sample_file(outdir, 41741, thickness="0.3")

        config = build_setup(collect_sequences(str(outdir)))

        assert config.samples[1].thickness == 0.3

    def test_default_thickness_without_sample_file(self, outdir):
        (outdir / "sample_41741.json").unlink()

        config = build_setup(collect_sequences(str(outdir)))

        assert config.samples[1].thickness == 0.1

    def test_saves_only_harmonic_1(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))

        assert config.save_all_harmonics is False


class TestSampleNames:
    def test_names_from_sample_files(self, outdir, log):
        """The fixtures were written before the run titles were recorded in scan_<N>.json."""
        names = sample_names(outdir)

        assert names == [title.removeprefix("Align:") for title in TITLES.values()]
        assert any("sample_41713.json" in message and "Align:S115_dry" in message for message in log.messages)

    def test_name_from_first_run(self, outdir):
        write_scan_file(outdir, 41741, num_of_runs=6, titles="EmptyCell")

        assert sample_names(outdir)[1] == "EmptyCell"

    def test_first_run_is_lowest_run_number(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[102, 100, 101], titles=["Third", "First", "Second"])

        assert sample_names(tmp_path) == ["First"]

    def test_removes_align_prefix(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="Align:S115_dry")

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("Align:" in message and "removed" in message for message in log.messages)

    def test_removes_white_space(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="  Align: S115 dry \t")

        assert sample_names(tmp_path) == ["S115 dry"]

    def test_replaces_separators(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="a/b\\c:d")

        assert sample_names(tmp_path) == ["a_b_c_d"]
        assert any("replaced" in message and "a_b_c_d" in message for message in log.messages)

    @pytest.mark.parametrize("title", [None, "", "   ", "Align:"], ids=["none", "empty", "white_space", "align"])
    def test_no_title(self, tmp_path, log, title):
        write_scan_file(tmp_path, 100, titles="First")
        write_scan_file(tmp_path, 200, titles=title)

        assert sample_names(tmp_path) == ["First", "Sample_1"]
        assert any("200" in message and "Sample_1" in message for message in log.records_at_warning())

    def test_repeated_names(self, tmp_path, log):
        for first_run in (100, 200, 300):
            write_scan_file(tmp_path, first_run, titles="EmptyCell")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_3"]
        warnings = log.records_at_warning()
        assert any("200" in message and "EmptyCell_2" in message for message in warnings)
        assert any("300" in message and "EmptyCell_3" in message for message in warnings)

    def test_repeated_names_after_cleaning(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="a/b")
        write_scan_file(tmp_path, 200, titles="Align:a:b")

        assert sample_names(tmp_path) == ["a_b", "a_b_2"]

    def test_counter_skips_name_of_a_title(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="EmptyCell")
        write_scan_file(tmp_path, 200, titles="EmptyCell_2")
        write_scan_file(tmp_path, 300, titles="EmptyCell")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_3"]

    def test_title_repeating_a_counter(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="EmptyCell")
        write_scan_file(tmp_path, 200, titles="EmptyCell")
        write_scan_file(tmp_path, 300, titles="EmptyCell_2")

        assert sample_names(tmp_path) == ["EmptyCell", "EmptyCell_2", "EmptyCell_2_2"]

    def test_fallback_repeating_a_title(self, tmp_path):
        write_scan_file(tmp_path, 100, titles="Sample_1")
        write_scan_file(tmp_path, 200)

        assert sample_names(tmp_path) == ["Sample_1", "Sample_1_2"]

    def test_differing_titles(self, tmp_path, log):
        write_scan_file(tmp_path, 100, num_of_runs=4, titles=["S115_dry", "S115_dry", "S115_wet", "S115_pc3"])

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("[102, 103]" in message and "'S115_dry'" in message for message in log.records_at_warning())

    def test_runs_without_title_not_compared(self, tmp_path, log):
        write_scan_file(tmp_path, 100, num_of_runs=3, titles=["S115_dry", None, "S115_dry"])

        assert sample_names(tmp_path) == ["S115_dry"]
        assert any("[101]" in message and "no title" in message for message in log.messages)
        assert not any("different" in message for message in log.records_at_warning())


class TestWriteConfig:
    def test_writes_setup_file(self, outdir):
        config = build_setup(collect_sequences(str(outdir)))
        setup_file = str(outdir / "autoconfig.json")

        write_config(config, setup_file)

        assert read_config(setup_file) == config
        assert read_config(setup_file).save_all_harmonics is False

    def test_writes_excluded_runs(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[100, 101, 103])
        config = build_setup(collect_sequences(str(tmp_path)))
        setup_file = tmp_path / "autoconfig.json"

        write_config(config, str(setup_file))

        assert json.loads(setup_file.read_text())["samples"][0]["exclude"] == [102]
        assert read_config(str(setup_file)) == config

    def test_replaces_file_and_leaves_no_temporary_file(self, outdir):
        setup_file = outdir / "autoconfig.json"
        setup_file.write_text("earlier content")
        files_before = set(outdir.iterdir())

        write_config(build_setup(collect_sequences(str(outdir))), str(setup_file))

        assert set(outdir.iterdir()) == files_before
        assert len(read_config(str(setup_file)).samples) == len(SEQUENCES)


class TestReduceSequence:
    def test_reduces_sequence_and_reports_every_sequence(self, tmp_path, monkeypatch):
        reduction = StandInReduction(monkeypatch)
        sequence_file, setup_file = str(tmp_path / "autoconfig_100.json"), str(tmp_path / "autoconfig.json")

        reduce_sequence(sequence_file, setup_file, str(tmp_path))

        autodir = str(tmp_path / "auto")
        assert reduction.reduced == [(sequence_file, autodir)]
        assert reduction.reports == [{"config_file_path": setup_file, "output_dir": autodir, "include_unscaled": False}]


class TestReductionLogs:
    """reduce_sequence adds the logs of a failed reduction to its error."""

    def stand_in_failing_experiment(self, monkeypatch, logs: dict[str, str]):
        """Replace Experiment with one that writes ``logs`` in its output directory, then raises."""

        class FailingExperiment:
            def __init__(self, **_kwargs):
                self.output_dir = None

            def reduce(self, output_dir: str | None = None) -> None:
                for name, text in logs.items():
                    (Path(output_dir) / f"reduction_{name}.log").write_text(text)
                raise RuntimeError("Cannot write detector data for sample S115_dry: first harmonic data is missing.")

        monkeypatch.setattr(reduce_USANS.reduce, "Experiment", FailingExperiment)
        monkeypatch.setattr(reduce_USANS, "generate_report", lambda **_kwargs: pytest.fail("report written"))

    def write_sequence_setup(self, tmp_path, background: str | None = None) -> str:
        setup = {"samples": [entry("S115_dry", 41713, 6)]}
        if background is not None:
            setup["background"] = {**entry(background, 41741, 6), "thickness": 0.1}
        write_setup(tmp_path, setup, 41713)
        return str(tmp_path / "autoconfig_41713.json")

    def test_log_of_failed_sample_added(self, tmp_path, monkeypatch):
        self.stand_in_failing_experiment(monkeypatch, {"S115_dry": "FileNotFoundError: USANS_41714_peak_1.txt\n"})
        sequence_file = self.write_sequence_setup(tmp_path)

        with pytest.raises(RuntimeError, match="first harmonic data is missing") as raised:
            reduce_sequence(sequence_file, str(tmp_path / "autoconfig.json"), str(tmp_path))

        (note,) = raised.value.__notes__
        assert "auto/reduction_S115_dry.log" in note
        assert "FileNotFoundError: USANS_41714_peak_1.txt" in note

    def test_logs_of_background_and_sample_added(self, tmp_path, monkeypatch):
        self.stand_in_failing_experiment(monkeypatch, {"Empty": "background log\n", "S115_dry": "sample log\n"})
        sequence_file = self.write_sequence_setup(tmp_path, background="Empty")

        with pytest.raises(RuntimeError) as raised:
            reduce_sequence(sequence_file, str(tmp_path / "autoconfig.json"), str(tmp_path))

        notes = raised.value.__notes__
        assert ["sample log" in notes[0], "background log" in notes[1]] == [True, True]

    def test_earlier_log_left_out(self, tmp_path, monkeypatch):
        self.stand_in_failing_experiment(monkeypatch, {})
        sequence_file = self.write_sequence_setup(tmp_path)
        (tmp_path / "auto").mkdir()
        earlier_log = tmp_path / "auto" / "reduction_S115_dry.log"
        earlier_log.write_text("log of an earlier reduction\n")
        an_hour_ago = time.time() - 3600
        os.utime(earlier_log, (an_hour_ago, an_hour_ago))

        with pytest.raises(RuntimeError) as raised:
            reduce_sequence(sequence_file, str(tmp_path / "autoconfig.json"), str(tmp_path))

        assert not hasattr(raised.value, "__notes__")

    def test_logs_shown_in_figure(self, tmp_path, monkeypatch):
        upload = StandInUpload(monkeypatch)
        self.stand_in_failing_experiment(monkeypatch, {"S115_dry": "FileNotFoundError: USANS_41714_peak_1.txt\n"})
        sequence_file = self.write_sequence_setup(tmp_path)
        monkeypatch.setattr(
            reduce_USANS,
            "reduce_run_sequence",
            lambda outdir, _first_run: reduce_sequence(sequence_file, str(tmp_path / "autoconfig.json"), outdir),
        )

        with pytest.raises(RuntimeError, match="first harmonic data is missing"):
            reduce_and_show(str(tmp_path), 41713, "41714", "<div>raw</div>", no_publish=False)

        ((_, _, files),) = upload.uploads
        assert f"<h2>{REDUCTION_FAILED}</h2>" in files["file"]
        assert "FileNotFoundError: USANS_41714_peak_1.txt" in files["file"]


class StandInReduction:
    """Stands in for usansred.reduce.Experiment and generate_report, recording their calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.reduced = []
        self.reports = []
        reduction = self

        class StandInExperiment:
            def __init__(self, config_file: str):
                self.config_file = config_file
                self.output_dir = None

            def reduce(self, output_dir: str | None = None) -> None:
                self.output_dir = output_dir
                reduction.reduced.append((self.config_file, output_dir))

        monkeypatch.setattr(reduce_USANS.reduce, "Experiment", StandInExperiment)
        monkeypatch.setattr(reduce_USANS, "generate_report", lambda **kwargs: reduction.reports.append(kwargs))


def read_setup(outdir: Path, first_run: int | None = None) -> dict:
    """Content of autoconfig.json, or of autoconfig_<N>.json for the sequence starting at ``first_run``."""
    filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
    return json.loads((outdir / filename).read_text())


def write_setup(outdir: Path, setup: dict, first_run: int | None = None) -> None:
    """Write autoconfig.json, or autoconfig_<N>.json, by hand."""
    filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
    (outdir / filename).write_text(json.dumps(setup))


def entry(name: str, start_scan_num: int, num_of_scans: int, exclude: list[int] | None = None) -> dict:
    """An entry of a setup file, as step 2 writes it, with the default thickness."""
    sample = {"name": name, "start_scan_num": start_scan_num, "num_of_scans": num_of_scans, "thickness": 0.1}
    return sample | {"exclude": exclude or []}


@pytest.mark.usefixtures("reduction")
class TestReduceRunSequence:
    @pytest.fixture
    def reduction(self, monkeypatch) -> StandInReduction:
        return StandInReduction(monkeypatch)

    def test_builds_missing_setup_from_every_sequence(self, outdir):
        reduce_run_sequence(str(outdir), 41741)

        assert [sample["start_scan_num"] for sample in read_setup(outdir)["samples"]] == list(SEQUENCES)
        assert read_setup(outdir, 41741) == {"samples": [entry("EmptyPCell", 41741, 6)], "save_all_harmonics": False}
        assert not (outdir / "autoconfig_41713.json").exists()
        assert not (outdir / "autoconfig_41832.json").exists()

    def test_reduces_only_the_sequence(self, outdir, reduction):
        reduce_run_sequence(str(outdir), 41741)

        autodir = str(outdir / "auto")
        assert reduction.reduced == [(str(outdir / "autoconfig_41741.json"), autodir)]
        assert reduction.reports == [
            {"config_file_path": str(outdir / "autoconfig.json"), "output_dir": autodir, "include_unscaled": False}
        ]

    def test_inserts_in_order_of_first_run(self, tmp_path):
        for first_run, title in ((100, "First"), (300, "Third"), (200, "Second")):
            write_scan_file(tmp_path, first_run, num_of_runs=2, titles=title)
            reduce_run_sequence(str(tmp_path), first_run)

        assert read_setup(tmp_path)["samples"] == [
            entry("First", 100, 2),
            entry("Second", 200, 2),
            entry("Third", 300, 2),
        ]

    def test_rewrites_only_the_entry_of_the_sequence(self, outdir):
        reduce_run_sequence(str(outdir), 41713)
        setup = read_setup(outdir)
        setup["samples"][1]["thickness"] = 0.5
        setup["samples"][2]["name"] = "Renamed"
        write_setup(outdir, setup)
        write_scan_file(outdir, 41713, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41713)

        samples = read_setup(outdir)["samples"]
        assert samples[0] == entry("S115_dry", 41713, 7)
        assert samples[1:] == setup["samples"][1:]
        assert read_setup(outdir, 41713)["samples"] == [entry("S115_dry", 41713, 7)]

    def test_run_of_earlier_sequence_updates_its_entry(self, outdir):
        reduce_run_sequence(str(outdir), 41832)
        write_scan_file(outdir, 41713, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41713)

        assert [sample["num_of_scans"] for sample in read_setup(outdir)["samples"]] == [7, 6, 6]
        assert read_setup(outdir, 41713)["samples"] == [entry("S115_dry", 41713, 7)]
        assert read_setup(outdir, 41832)["samples"] == [entry("S115_pc3", 41832, 6)]

    def test_name_fixed_once_inserted(self, tmp_path):
        write_scan_file(tmp_path, 100, num_of_runs=2, titles="First")
        reduce_run_sequence(str(tmp_path), 100)
        write_scan_file(tmp_path, 100, num_of_runs=3, titles="Changed")

        reduce_run_sequence(str(tmp_path), 100)

        assert read_setup(tmp_path)["samples"] == [entry("First", 100, 3)]
        assert read_setup(tmp_path, 100)["samples"] == [entry("First", 100, 3)]

    def test_names_at_insertion(self, tmp_path, log):
        write_scan_file(tmp_path, 100, titles="S115")
        reduce_run_sequence(str(tmp_path), 100)
        write_scan_file(tmp_path, 200, titles="S115")
        reduce_run_sequence(str(tmp_path), 200)
        write_scan_file(tmp_path, 300)

        reduce_run_sequence(str(tmp_path), 300)

        assert [sample["name"] for sample in read_setup(tmp_path)["samples"]] == ["S115", "S115_2", "Sample_2"]
        assert any("S115_2" in message for message in log.records_at_warning())

    def test_name_not_taken_by_background(self, tmp_path):
        write_setup(
            tmp_path,
            {"samples": [entry("S115", 100, 1)], "background": entry("Empty", 50, 1) | {"thickness": 0.1}},
        )
        write_scan_file(tmp_path, 200, titles="Empty")

        reduce_run_sequence(str(tmp_path), 200)

        assert read_setup(tmp_path)["samples"][1]["name"] == "Empty_2"

    def test_same_run_twice_gives_same_files(self, outdir):
        reduce_run_sequence(str(outdir), 41741)
        setup, sequence_setup = (outdir / "autoconfig.json").read_text(), (outdir / "autoconfig_41741.json").read_text()

        reduce_run_sequence(str(outdir), 41741)

        assert (outdir / "autoconfig.json").read_text() == setup
        assert (outdir / "autoconfig_41741.json").read_text() == sequence_setup

    def test_first_run_recorded_later(self, tmp_path):
        write_scan_file(tmp_path, 100, runs=[101, 102], titles="S115")
        reduce_run_sequence(str(tmp_path), 100)
        assert read_setup(tmp_path)["samples"] == [entry("S115", 100, 3, exclude=[100])]
        write_scan_file(tmp_path, 100, runs=[101, 102, 100], titles="S115")

        reduce_run_sequence(str(tmp_path), 100)

        assert read_setup(tmp_path)["samples"] == [entry("S115", 100, 3)]
        assert read_setup(tmp_path, 100)["samples"] == [entry("S115", 100, 3)]

    def test_new_sequence_setup_copies_options(self, outdir):
        background = entry("Empty", 41741, 6)
        empty_cell = {"name": "Cell", "start_scan_num": 41741, "num_of_scans": 6}
        write_setup(
            outdir,
            {
                "samples": [entry("S115_dry", 41713, 6)],
                "background": background,
                "empty_cell": empty_cell,
                "save_all_harmonics": True,
            },
        )

        reduce_run_sequence(str(outdir), 41832)

        sequence_setup = read_setup(outdir, 41832)
        assert sequence_setup["samples"] == [entry("S115_pc3", 41832, 6)]
        assert sequence_setup["background"] == background
        assert sequence_setup["empty_cell"] == empty_cell | {"exclude": []}
        assert sequence_setup["save_all_harmonics"] is True

    def test_existing_sequence_setup_keeps_all_but_samples(self, outdir):
        reduce_run_sequence(str(outdir), 41832)
        sequence_setup = read_setup(outdir, 41832)
        sequence_setup["background"] = entry("Empty", 41741, 6)
        sequence_setup["save_all_harmonics"] = True
        write_setup(outdir, sequence_setup, 41832)
        write_scan_file(outdir, 41832, num_of_runs=7)

        reduce_run_sequence(str(outdir), 41832)

        assert read_setup(outdir, 41832) == sequence_setup | {"samples": [entry("S115_pc3", 41832, 7)]}

    @pytest.mark.parametrize("first_run", [None, 41713], ids=["autoconfig", "autoconfig_N"])
    @pytest.mark.parametrize("content", ["{not json", '{"samples": []}'], ids=["invalid_json", "rejected"])
    def test_unreadable_setup_file(self, outdir, reduction, log, first_run, content):
        filename = "autoconfig.json" if first_run is None else f"autoconfig_{first_run}.json"
        (outdir / filename).write_text(content)
        files_before = set(outdir.iterdir())

        with pytest.raises(ValueError):  # noqa: PT011 - the messages of json and pydantic differ
            reduce_run_sequence(str(outdir), 41713)

        assert (outdir / filename).read_text() == content
        assert set(outdir.iterdir()) == files_before
        assert reduction.reduced == []
        assert any(filename in message and "cannot be read" in message for message in log.records_at_warning())

    def test_no_temporary_file_left(self, outdir):
        reduce_run_sequence(str(outdir), 41741)
        reduce_run_sequence(str(outdir), 41741)

        assert not [path for path in outdir.iterdir() if path.suffix == ".tmp"]

    def test_sequence_not_recorded(self, outdir, reduction, log):
        files_before = set(outdir.iterdir())

        reduce_run_sequence(str(outdir), 41900)

        assert set(outdir.iterdir()) == files_before
        assert reduction.reduced == []
        assert any("41900" in message and "nothing to reduce" in message for message in log.records_at_warning())


class TestSummary:
    def test_summary_covers_every_sequence(self, outdir, monkeypatch, log):
        """The reduced files of 41713 and 41741 are in auto/, those of 41832 are missing."""
        autodir = outdir / "auto"
        autodir.mkdir()
        for name in ("S115_dry", "EmptyPCell"):
            for suffix in ("det_1", "det_1_unscaled"):
                rows = [f"{1e-5 * (k + 1)},{100.0 / (k + 1)},{1.0 / (k + 1)}" for k in range(10)]
                (autodir / f"{name}_{suffix}.txt").write_text("\n".join(rows) + "\n")

        class StandInExperiment:
            """Stands in for usansred.reduce.Experiment, leaving the files of auto/ as they are."""

            def __init__(self, **_kwargs):
                self.output_dir = None

            def reduce(self, output_dir: str | None = None) -> None:
                self.output_dir = output_dir

        monkeypatch.setattr(reduce_USANS.reduce, "Experiment", StandInExperiment)

        reduce_run_sequence(str(outdir), 41741)

        with zipfile.ZipFile(autodir / "summary.xlsx") as archive:
            workbook = archive.read("xl/workbook.xml").decode()
        assert re.findall(r'<sheet name="([^"]*)"', workbook) == ["Original", "S115_dry_det_1", "EmptyPCell_det_1"]
        assert any("S115_pc3_det_1.txt" in message and "does not exist" in message for message in log.messages)
        assert not [path for path in autodir.iterdir() if path.name != "summary.xlsx" and path.suffix == ".xlsx"]


def plot_text(block: str) -> str:
    """The HTML block of a panel, with the slashes that plotly escapes restored."""
    return block.replace("\\u002f", "/")


def write_det_1_file(path: Path, num_points: int = 5) -> Path:
    """Write a reduced data file in the format of the reduction: Q, I, dI, and a trailing comma."""
    rows = [f"{1e-5 * (k + 1)},{100.0 / (k + 1)},{1.0 / (k + 1)}," for k in range(num_points)]
    path.write_text("\n".join(rows) + "\n")
    return path


class TestPanels:
    def test_raw_scan(self):
        block = plot_text(
            raw_scan_plot("41715", np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.array([1.0, 2.0]), "BL1A:Mot:m2rot")
        )

        assert '"x":[1.0,2.0],"y":[10.0,20.0]' in block
        assert '"title":{"text":"BL1A:Mot:m2rot"}' in block
        assert '"title":{"text":"Counts"}' in block
        assert block.count('"type":"log"') == 1

    def test_raw_scan_failure(self, monkeypatch, log):
        def plot1d(*_args, **_kwargs):
            raise RuntimeError("malformed data")

        monkeypatch.setattr(reduce_USANS, "plot1d", plot1d)

        assert raw_scan_plot("41715", [1.0], [10.0], [1.0], "BL1A:Mot:m2rot") is None
        assert any("raw scan of run 41715" in message for message in log.records_at_warning())

    def test_reduced_sequence(self, tmp_path):
        det_1_file = write_det_1_file(tmp_path / "S115_dry_det_1.txt", num_points=2)

        block = plot_text(reduced_sequence_plot("41715", str(det_1_file)))

        assert '"x":[1e-05,2e-05],"y":[100.0,50.0]' in block
        assert '"title":{"text":"Q (1/A)"}' in block
        assert '"title":{"text":"I (1/cm)"}' in block
        assert '"title":{"text":"S115_dry_det_1"}' in block
        assert block.count('"type":"log"') == 2

    def test_reduced_sequence_with_one_point(self, tmp_path):
        det_1_file = write_det_1_file(tmp_path / "S115_dry_det_1.txt", num_points=1)

        assert '"x":[1e-05]' in reduced_sequence_plot("41715", str(det_1_file))

    def test_reduced_sequence_missing_file(self, tmp_path, log):
        assert reduced_sequence_plot("41715", str(tmp_path / "S115_dry_det_1.txt")) is None
        assert any("S115_dry_det_1.txt" in message for message in log.records_at_warning())


class StandInUpload:
    """Stands in for plot_publisher.publish_plot, recording the uploads, or raising the error given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, error: Exception | None = None):
        self.uploads = []
        self.error = error
        monkeypatch.setattr(reduce_USANS, "publish_plot", self)

    def __call__(self, instrument: str, run_number: str, files: dict) -> None:
        if self.error is not None:
            raise self.error
        self.uploads.append((instrument, run_number, files))


class TestSaveFigure:
    def test_saves_page_with_panels(self, tmp_path):
        figure = save_figure("41715", ["<div>raw</div>", "<div>reduced</div>"], str(tmp_path))

        page = (tmp_path / "USANS_41715.html").read_text()
        assert figure == "<div>raw</div>\n<div>reduced</div>"
        assert figure in page
        assert f'<script src="https://cdn.plot.ly/plotly-{reduce_USANS.get_plotlyjs_version()}.min.js">' in page

    def test_leaves_out_missing_panel(self, tmp_path):
        assert save_figure("41715", ["<div>raw</div>", None], str(tmp_path)) == "<div>raw</div>"

    def test_no_panel(self, tmp_path, log):
        assert save_figure("41715", [None, None], str(tmp_path)) is None
        assert list(tmp_path.iterdir()) == []
        assert any("No panel" in message for message in log.records_at_warning())

    def test_page_cannot_be_written(self, tmp_path, log):
        assert save_figure("41715", ["<div>raw</div>"], str(tmp_path / "missing")) == "<div>raw</div>"
        assert any("USANS_41715.html" in message for message in log.records_at_warning())


class TestUploadFigure:
    def test_uploads_under_the_run(self, monkeypatch):
        upload = StandInUpload(monkeypatch)

        upload_figure("41715", "<div>raw</div>")

        assert upload.uploads == [("USANS", "41715", {"file": "<div>raw</div>"})]

    @pytest.mark.parametrize(
        "error",
        [requests.HTTPError("401 Unauthorized"), requests.ConnectionError("no route to host")],
        ids=["http_error", "connection_error"],
    )
    def test_request_failure_is_logged(self, monkeypatch, log, error):
        StandInUpload(monkeypatch, error)

        upload_figure("41715", "<div>raw</div>")

        assert any("run 41715" in message and "uploaded" in message for message in log.records_at_warning())

    def test_missing_configuration_raises(self, monkeypatch):
        StandInUpload(monkeypatch, RuntimeError("Failed to find Configuration file"))

        with pytest.raises(RuntimeError, match="Configuration"):
            upload_figure("41715", "<div>raw</div>")


class TestFailureBlock:
    @pytest.mark.parametrize("heading", [PREPROCESSING_FAILED, REDUCTION_FAILED])
    def test_heading_and_traceback(self, heading):
        try:
            raise RuntimeError("Experiment failed: x < 0")
        except RuntimeError as error:
            block = failure_block(heading, error)

        assert block.startswith(f"<h2>{heading}</h2>\n<pre>")
        assert block.endswith("</pre>")
        assert "Traceback (most recent call last)" in block
        assert "test_heading_and_traceback" in block
        assert "RuntimeError: Experiment failed: x &lt; 0" in block

    def test_notes_after_traceback(self):
        error = RuntimeError("first harmonic data is missing")
        error.add_note("Log of the reduction, auto/reduction_S115_dry.log:\nFileNotFoundError: <peak_1>")

        block = failure_block(REDUCTION_FAILED, error)

        assert block.index("first harmonic data is missing") < block.index("auto/reduction_S115_dry.log")
        assert "FileNotFoundError: &lt;peak_1&gt;" in block


class TestReduceAndShow:
    @pytest.fixture
    def upload(self, monkeypatch) -> StandInUpload:
        return StandInUpload(monkeypatch)

    def stand_in_step_2(self, monkeypatch, result):
        """Replace step 2 with a function returning ``result``, or raising it if it is an exception."""

        def reduce_run_sequence(*_args):
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(reduce_USANS, "reduce_run_sequence", reduce_run_sequence)

    def test_shows_both_panels(self, tmp_path, monkeypatch, upload):
        det_1_file = write_det_1_file(tmp_path / "S115_dry_det_1.txt")
        self.stand_in_step_2(monkeypatch, str(det_1_file))

        reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=False)

        ((instrument, run_number, files),) = upload.uploads
        assert (instrument, run_number) == ("USANS", "41715")
        assert files["file"].startswith("<div>raw</div>\n")
        assert "S115_dry_det_1" in files["file"]
        assert files["file"] in (tmp_path / "USANS_41715.html").read_text()

    def test_no_publish(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, str(write_det_1_file(tmp_path / "S115_dry_det_1.txt")))

        reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=True)

        assert upload.uploads == []
        assert (tmp_path / "USANS_41715.html").exists()

    def test_failure_shown_if_step_2_fails(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, ValueError("autoconfig.json: <invalid> JSON"))

        with pytest.raises(ValueError, match="autoconfig.json"):
            reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=False)

        ((instrument, run_number, files),) = upload.uploads
        figure = files["file"]
        assert (instrument, run_number) == ("USANS", "41715")
        assert figure.startswith("<div>raw</div>\n")
        assert f"<h2>{REDUCTION_FAILED}</h2>" in figure
        assert "Traceback (most recent call last)" in figure
        assert "ValueError: autoconfig.json: &lt;invalid&gt; JSON" in figure
        assert figure in (tmp_path / "USANS_41715.html").read_text()

    def test_failure_alone_without_raw_scan(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, ValueError("autoconfig.json: invalid JSON"))

        with pytest.raises(ValueError, match="autoconfig.json"):
            reduce_and_show(str(tmp_path), 41713, "41715", None, no_publish=False)

        ((_, _, files),) = upload.uploads
        assert files["file"].startswith(f"<h2>{REDUCTION_FAILED}</h2>")

    def test_failure_not_uploaded_with_no_publish(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, ValueError("autoconfig.json: invalid JSON"))

        with pytest.raises(ValueError, match="autoconfig.json"):
            reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=True)

        assert upload.uploads == []
        assert REDUCTION_FAILED in (tmp_path / "USANS_41715.html").read_text()

    def test_failure_raised_after_upload_failure(self, tmp_path, monkeypatch, log):
        StandInUpload(monkeypatch, requests.ConnectionError("no connection"))
        self.stand_in_step_2(monkeypatch, ValueError("autoconfig.json: invalid JSON"))

        with pytest.raises(ValueError, match="autoconfig.json"):
            reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=False)

        assert any("cannot be uploaded" in message for message in log.messages)
        assert REDUCTION_FAILED in (tmp_path / "USANS_41715.html").read_text()

    def test_raw_scan_alone_if_nothing_reduced(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, None)

        reduce_and_show(str(tmp_path), 41713, "41715", "<div>raw</div>", no_publish=False)

        assert upload.uploads == [("USANS", "41715", {"file": "<div>raw</div>"})]

    def test_nothing_to_show(self, tmp_path, monkeypatch, upload):
        self.stand_in_step_2(monkeypatch, None)

        reduce_and_show(str(tmp_path), 41713, "41715", None, no_publish=False)

        assert upload.uploads == []
        assert list(tmp_path.iterdir()) == []

    def test_step_2_returns_the_det_1_file(self, outdir, monkeypatch):
        StandInReduction(monkeypatch)

        assert reduce_run_sequence(str(outdir), 41741) == str(outdir / "auto" / "EmptyPCell_det_1.txt")
